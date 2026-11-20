"""E8.3：文本扰动审计 —— 模型是否真的「读文本」。

两种角色（2026-11-20 增补）：
  - --donor same   主协议（判读用）。门槛（C-28）：文本侧翻转率 < 0.20。
  - --donor cross  阳性对照（描述性，**不套门槛**）。用于证明扰动装置本身有检验力：
                   若连「文本内容整体换成另一类」都翻不动，则同类协议的 2.42%
                   不具备任何证据力。该 run 的 gate 标记为 applicable=False。

协议：
  1. 扰动对象 = 文本分支出现的 20 个字段（TEXT_FEATURES），取值来自 donor 样本
     （same=同类别真实样本 / cross=跨类别样本）；句式模板不动，由
     generate_text_description 重新生成，因此只变数字、不变结构。
  2. 对照 = 只改这 20 个字段对应的**数值特征**（文本保持原样），扰动强度完全对齐。
  3. 指标 = 相对原始预测的翻转率。
     判读：同类文本侧低 + 跨类文本侧显著高于特征侧 → 文本分支确实在读内容，
     但同类内扰动不足以改变判定；**仅凭同类低翻转率不得推出「模型不读文本」**。

用法：
  python scripts/e83_perturbation.py \
      --split_dir split_data/dataset_200/split_openset_0 \
      --llm_path models/qwen2.5-1.5b --model_dir saved_models/model_2013 \
      --out results/2026-11-20/e83_perturbation
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.flow_features import TEXT_FEATURES, generate_text_description, select_numeric_columns  # noqa: E402

NUM_KNOWN = 10
GATE_FLIP = 0.20


def build_cases(df, X, rng, donor_mode="same"):
    """为每条已知样本配一个 donor，生成文本扰动与特征扰动。

    donor_mode=same：同类别真实样本（主协议）；cross：跨类别（阳性对照，仅补充用）。
    """
    y = df["Label"].to_numpy(dtype=np.int64)
    known = y < NUM_KNOWN
    cols = select_numeric_columns(df)
    pos = {c: i for i, c in enumerate(cols)}
    idx_text = [pos[c] for c in TEXT_FEATURES]

    rows, xs, texts, xs_p, texts_p = [], [], [], [], []
    for i in np.where(known)[0]:
        pool = np.where(y == y[i])[0] if donor_mode == "same" else np.where(y != y[i])[0]
        if donor_mode == "same":
            pool = pool[pool != i]
        j = int(rng.choice(pool))

        row = df.iloc[i].copy()
        for c in TEXT_FEATURES:
            row[c] = df.iloc[j][c]
        rows.append(i)
        xs.append(X[i])
        texts.append(str(df.iloc[i]["text_description"]))
        xp = X[i].copy()
        xp[idx_text] = X[j][idx_text]
        xs_p.append(xp)
        texts_p.append(generate_text_description(row))

    return np.array(rows), y[known], np.stack(xs), texts, np.stack(xs_p), texts_p


def predict(model, tokenizer, X, texts, device, batch_size, max_length):
    preds = []
    for i in range(0, len(texts), batch_size):
        bt = tokenizer(texts[i:i + batch_size], return_tensors="pt", padding=True,
                       truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            out = model(stat_tensor=torch.tensor(X[i:i + batch_size], dtype=torch.float32),
                        bert_tensor=None, input_ids=bt["input_ids"],
                        attention_mask=bt["attention_mask"])
        preds.append(out["logits"].argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--llm_path", default="models/qwen2.5-1.5b")
    ap.add_argument("--model_dir", default="saved_models/model_2013")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--donor", choices=["same", "cross"], default="same",
                    help="same=同类真实样本（主协议）；cross=跨类（阳性对照）")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from src.model_architectures.multi_modal_model import MultiModalFusionModel

    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(os.path.join(args.split_dir, "test_data.csv"))
    d = np.load(os.path.join(args.split_dir, "test.npz"))
    X = d["scaled_features"].astype(np.float32)
    assert len(df) == len(X), (len(df), len(X))

    rows, y_known, Xo, To, Xp, Tp = build_cases(df, X, rng, args.donor)
    print(f"[cases] 已知样本 {len(rows)}，文本扰动 {len(Tp)} 条")

    model = MultiModalFusionModel.from_pretrained(
        llm_model_path_or_save_dir=args.llm_path, save_dir=args.model_dir)
    model.eval()
    device = model.device
    print(f"[model] device={device} use_numeric={model.use_numeric} "
          f"use_bert={model.use_bert} use_llm={model.use_llm}")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, local_files_only=True)

    p_o = predict(model, tokenizer, Xo, To, device, args.batch_size, args.max_length)
    p_t = predict(model, tokenizer, Xo, Tp, device, args.batch_size, args.max_length)
    p_f = predict(model, tokenizer, Xp, To, device, args.batch_size, args.max_length)

    acc = float((p_o == y_known).mean())
    flip_t = (p_t != p_o)
    flip_f = (p_f != p_o)

    boot_t, boot_f, boot_d = [], [], []
    n = len(rows)
    for _ in range(args.boot):
        b = rng.integers(0, n, n)
        ft, ff = flip_t[b].mean(), flip_f[b].mean()
        boot_t.append(ft); boot_f.append(ff); boot_d.append(ff - ft)

    out = {
        "experiment": "E8.3 文本扰动审计",
        "split_dir": args.split_dir,
        "model_dir": args.model_dir,
        "protocol": {
            "donor": args.donor,
            "target": "TEXT_FEATURES 20 字段（真实样本替换）",
            "text_perturb": f"重生成文本模板，数值来自 donor（{args.donor}）",
            "feature_perturb": "仅替换同 20 字段对应的数值特征，文本不动",
            "n_known": int(n),
        },
        "original_acc": acc,
        "text_flip_rate": float(flip_t.mean()),
        "text_flip_ci95": [float(np.quantile(boot_t, 0.025)), float(np.quantile(boot_t, 0.975))],
        "feature_flip_rate": float(flip_f.mean()),
        "feature_flip_ci95": [float(np.quantile(boot_f, 0.025)), float(np.quantile(boot_f, 0.975))],
        "feature_minus_text_ci95": [float(np.quantile(boot_d, 0.025)), float(np.quantile(boot_d, 0.975))],
        "flip_both": float((flip_t & flip_f).mean()),
        "flip_neither": float((~flip_t & ~flip_f).mean()),
        "role": "main_protocol" if args.donor == "same" else "positive_control",
        "gate": (
            {"applicable": True, "text_flip_lt": GATE_FLIP,
             "pass": bool(flip_t.mean() < GATE_FLIP)}
            if args.donor == "same"
            else {"applicable": False,
                  "reason": "阳性对照（跨类 donor）为描述性证据，不套判读门槛；"
                            "pass 字段不适用于该 run"}
        ),
        "seed": args.seed, "batch_size": args.batch_size,
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    np.savez(os.path.join(args.out, "predictions.npz"),
             rows=rows, y=y_known, p_orig=p_o, p_text=p_t, p_feat=p_f)
    print(json.dumps({k: out[k] for k in (
        "original_acc", "text_flip_rate", "text_flip_ci95",
        "feature_flip_rate", "feature_flip_ci95", "gate")}, ensure_ascii=False, indent=2))
    print(f"[OK] -> {args.out}")


if __name__ == "__main__":
    main()
