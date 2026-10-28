"""P0 诊断：解释链路文本截断（128 vs 256）对 backbone 预测的影响。

背景：generate_explanation.py 的 predict_with_backbone 曾把文本截断到 128 token，
而训练/测试主链路统一用 utils.flow_features.TEXT_MAX_LENGTH=256。
本脚本只读模型与 split，不训练、不写台账，用于判断历史解释产物是否需要重跑。

用法：
    /root/miniconda3/bin/python run.py /root/miniconda3/bin/python scripts/diag_text_truncation.py \
        --backbone_model_id 2000 --result explanation_results/explanations_model2000_20261026_100154.json
"""

import argparse
import json
import os

import sys

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.explain.generate_explanation import load_backbone_model, load_test_samples
from utils.flow_features import TEXT_MAX_LENGTH


@torch.no_grad()
def predict_at(model, tokenizer, samples, max_length, device, batch_size=32):
    """固定 max_length 跑一遍预测，返回每个样本的 argmax 标签。"""
    preds = []
    model.eval()
    use_llm = model.use_llm and tokenizer is not None
    for i in tqdm(range(0, len(samples), batch_size), desc=f"predict@{max_length}"):
        batch = samples[i : i + batch_size]
        stat = torch.tensor(
            np.stack([s["stat"] for s in batch]), dtype=torch.float32
        ).to(device)
        bert = torch.tensor(
            np.stack([s["bert"] for s in batch]), dtype=torch.float32
        ).to(device)
        kwargs = {}
        if use_llm:
            enc = tokenizer(
                [s["text"] for s in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            kwargs = {
                "input_ids": enc["input_ids"].to(device),
                "attention_mask": enc["attention_mask"].to(device),
            }
        logits = model(stat, bert, **kwargs)["logits"]
        preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
    return preds


def main():
    ap = argparse.ArgumentParser(description="P0 文本截断诊断：128 vs 256 预测差异")
    ap.add_argument("--backbone_model_id", type=int, default=2000)
    ap.add_argument("--dataset_id", type=int, default=200)
    ap.add_argument("--split_id", type=int, default=0)
    ap.add_argument("--short_length", type=int, default=128)
    ap.add_argument("--result", default=None, help="历史解释产物 JSON，用于定位同一批抽样样本")
    ap.add_argument("--out", default="results/2026-10-28/p0_truncation_diag.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_backbone_model(args.backbone_model_id, device=device)
    num_known = int(getattr(model, "num_classes", 10))
    samples = load_test_samples(
        dataset_id=args.dataset_id, split_id=args.split_id, openset=True, n_samples=None
    )
    labels = np.array([s["label"] for s in samples], dtype=np.int64)
    known_mask = labels < num_known

    pred_short = predict_at(model, tokenizer, samples, args.short_length, device)
    pred_long = predict_at(model, tokenizer, samples, TEXT_MAX_LENGTH, device)

    pred_short = np.array(pred_short)
    pred_long = np.array(pred_long)
    changed = pred_short != pred_long

    def known_acc(pred):
        return float((pred[known_mask] == labels[known_mask]).mean())

    summary = {
        "backbone_model_id": args.backbone_model_id,
        "dataset_id": args.dataset_id,
        "split_id": args.split_id,
        "short_length": args.short_length,
        "long_length": int(TEXT_MAX_LENGTH),
        "n_total": int(len(samples)),
        "n_known": int(known_mask.sum()),
        "changed_total": int(changed.sum()),
        "changed_rate": float(changed.mean()),
        "known_accuracy_short": known_acc(pred_short),
        "known_accuracy_long": known_acc(pred_long),
    }

    if args.result and os.path.exists(args.result):
        with open(args.result, "r", encoding="utf-8") as f:
            artifact = json.load(f)
        idx = [
            int(r["sample_index"])
            for r in artifact.get("results", [])
            if "sample_index" in r
        ]
        idx = [i for i in idx if 0 <= i < len(samples)]
        sub_changed = [i for i in idx if changed[i]]
        sub_known = [i for i in idx if labels[i] < num_known]
        summary["artifact_subset"] = {
            "path": args.result,
            "n_samples": len(idx),
            "n_known": len(sub_known),
            "n_changed": len(sub_changed),
            "changed_indices": sorted(sub_changed),
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 60)
    print(f"结果已写入 {args.out}")


if __name__ == "__main__":
    main()
