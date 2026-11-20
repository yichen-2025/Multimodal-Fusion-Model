"""闭集评估：加载指定 checkpoint，在 test 的已知类子集上算 acc / macro-F1。

用于 E8.4（A0*_no_text 重训版）与 A0* 的同口径对照。

用法
  python scripts/eval_closed_set.py --model_dir saved_models/model_5023 \
      --split_dir split_data/dataset_200/split_openset_0 --out results/2026-11-20/e84_retrain
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer  # noqa: E402

from src.model_architectures.multi_modal_model import MultiModalFusionModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--llm_path", default="models/qwen2.5-1.5b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_classes", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    d = np.load(os.path.join(args.split_dir, "test.npz"))
    df = pd.read_csv(os.path.join(args.split_dir, "test_data.csv"))
    X = d["scaled_features"].astype(np.float32)
    y = d["labels"].astype(np.int64)
    texts = df["text_description"].astype(str).tolist()
    keep = y < args.num_classes
    X, y = X[keep], y[keep]
    texts = [t for t, k in zip(texts, keep) if k]

    model = MultiModalFusionModel.from_pretrained(
        llm_model_path_or_save_dir=args.llm_path, save_dir=args.model_dir)
    model.eval()
    dev = model.device
    tok = AutoTokenizer.from_pretrained(args.llm_path, local_files_only=True)

    preds = []
    for i in range(0, len(X), args.batch_size):
        xb = torch.tensor(X[i:i + args.batch_size]).to(dev)
        tb = tok(texts[i:i + args.batch_size], return_tensors="pt", padding=True,
                 truncation=True, max_length=args.max_length).to(dev)
        with torch.no_grad():
            out = model(stat_tensor=xb, bert_tensor=None,
                        input_ids=tb["input_ids"], attention_mask=tb["attention_mask"])
        preds.append(out["logits"][:, :args.num_classes].argmax(-1).cpu().numpy())
    pred = np.concatenate(preds)

    res = {"model_dir": args.model_dir, "n": int(len(y)),
           "accuracy": float(accuracy_score(y, pred)),
           "macro_f1": float(f1_score(y, pred, average="macro"))}
    name = os.path.basename(os.path.normpath(args.model_dir))
    with open(os.path.join(args.out, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
