"""E7：原型构造消融（描述来源 / 原型条数 / 距离度量）。

只做零样本闭集 acc，用于回答「是不是描述写得不好」：把 3 条描述逐条加进来、
把原型条数从 1 增到 3、把距离从 cosine 换到 euclidean，看结果是否变化。
α 的影响已在 E4/E5 的 curve 里，不重复。

用法：
  python scripts/ablate_descriptions.py --backend bert --out results/2026-11-20/e7_ablation/bert.json
  python scripts/ablate_descriptions.py --backend qwen --out results/2026-11-20/e7_ablation/qwen.json
"""
import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.build_semantic_prototypes import build_descriptions, CLASS_NAMES  # noqa: E402

NUM_KNOWN = 10


def l2norm(a):
    return a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12, None)


def encode(backend, texts, bert_path, qwen_path):
    if backend == "bert":
        from src.model_architectures.bert_encoder import BertEncoder
        enc = BertEncoder(local_model_path=bert_path, batch_size=8, force_cpu=True)
        return enc(texts).numpy().astype(np.float64)
    from scripts.cache_text_embeddings import load_model, encode_texts
    tok, model = load_model(qwen_path)
    return encode_texts(texts, tok, model, 8, 256).astype(np.float64)


def acc_zero_shot(Xte, yte, P, metric):
    if metric == "cosine":
        d = 1.0 - l2norm(Xte) @ l2norm(P).T
    else:
        d = np.linalg.norm(Xte[:, None, :] - P[None, :, :], axis=2)
    pred = np.argmin(d, axis=1)
    return float((pred == yte).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["bert", "qwen"], required=True)
    ap.add_argument("--split_dir", default="split_data/dataset_200/split_openset_0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bert_path", default="models/bert")
    ap.add_argument("--qwen_path", default="models/qwen2.5-1.5b")
    args = ap.parse_args()

    info = build_descriptions()
    # 固定顺序：10 类 × 3 条
    all_desc = [info[n]["descriptions"] for n in CLASS_NAMES]

    subsets = [("d1", 1), ("d1+d2", 2), ("all", 3)]
    n_protos = [1, 2, 3]

    # 一次性编码全部 30 条（子集只是切片，省重复编码）
    flat = [t for ds in all_desc for t in ds]
    E = encode(args.backend, flat, args.bert_path, args.qwen_path)
    E = E.reshape(NUM_KNOWN, 3, -1)

    d = np.load(os.path.join(args.split_dir, "test.npz"))
    X = d["text_embeddings"].astype(np.float64) if args.backend == "bert" \
        else np.load("emb/qwen_texts.npz", allow_pickle=False)
    if args.backend == "bert":
        Xte, yte = X[d["labels"] < NUM_KNOWN], d["labels"][d["labels"] < NUM_KNOWN]
    else:
        zt = X
        ids = zt["ids"].astype(str).tolist()
        ix = [i for i, s in enumerate(ids) if s.startswith("test_")]
        rows = [int(ids[i].split("_")[1]) for i in ix]
        Xall = zt["emb"].astype(np.float64)[ix]
        yall = d["labels"][rows]
        Xte, yte = Xall[yall < NUM_KNOWN], yall[yall < NUM_KNOWN]

    table = []
    for name, n_desc in subsets:
        for n in n_protos:
            if n > n_desc:
                continue
            P = E[:, :n, :].mean(axis=1)
            for metric in ("cosine", "euclidean"):
                table.append({
                    "subset": name, "n_desc": n_desc, "n_proto": n,
                    "metric": metric,
                    "test_acc": acc_zero_shot(Xte, yte, P, metric),
                })

    out = {
        "experiment": "E7 原型构造消融（零样本）",
        "backend": args.backend,
        "n_test_known": int(len(yte)),
        "class_names": CLASS_NAMES,
        "table": table,
        "note": "α 的影响见 E4/E5 的 curve.csv；本表不含 α",
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    for r in table:
        print(f"{r['subset']:6s} n_proto={r['n_proto']} {r['metric']:9s} acc={r['test_acc']:.4f}")
    print(f"[OK] -> {args.out}")


if __name__ == "__main__":
    main()
