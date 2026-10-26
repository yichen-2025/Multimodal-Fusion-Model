"""CTI-RAG 离线检索评估：top-k 命中 true_mitre_ids。

只读划分与文本特征，不参与分类；BENIGN（无技术）不计入分母。

用法：
    python scripts/eval_retrieval.py --backbone_model_id 2000 --dataset_id 200 --split_id 0
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.explain.attack_mapping import get_true_mitre_ids
from src.explain.retriever import BM25Retriever
from utils.caliber import read_model_num_classes


def main():
    ap = argparse.ArgumentParser(description="CTI 检索离线评估")
    ap.add_argument("--backbone_model_id", type=int, default=2000, help="用于读取 num_known_classes")
    ap.add_argument("--dataset_id", type=int, default=200)
    ap.add_argument("--split_id", type=int, default=0)
    ap.add_argument("--openset", dest="openset", action="store_true", default=True)
    ap.add_argument("--no-openset", dest="openset", action="store_false")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="只评估前 N 条（调试用）")
    args = ap.parse_args()

    split_name = f"split_openset_{args.split_id}" if args.openset else f"split_{args.split_id}"
    split_dir = os.path.join(PROJECT_ROOT, "split_data", f"dataset_{args.dataset_id}", split_name)
    npz = np.load(os.path.join(split_dir, "test.npz"), allow_pickle=True)
    labels = npz["labels"]
    df = pd.read_csv(os.path.join(split_dir, "test_data.csv"))
    texts = df["text_description"].tolist() if "text_description" in df.columns else [""] * len(labels)

    K = read_model_num_classes(os.path.join(PROJECT_ROOT, "saved_models", f"model_{args.backbone_model_id}"))

    retriever = BM25Retriever()
    idxs = [i for i in range(len(labels))
            if int(labels[i]) < K and get_true_mitre_ids(int(labels[i]))]
    if args.limit:
        idxs = idxs[: args.limit]

    hits = Counter()
    per_class = defaultdict(lambda: Counter())     # label -> hit@1 计数
    per_class_n = Counter()
    confusion = defaultdict(Counter)               # label -> top1 technique
    misses = []

    for i in idxs:
        truth = get_true_mitre_ids(int(labels[i]))
        ranked = retriever.retrieve(texts[i] if i < len(texts) else "", args.topk)
        top_ids = [mid for mid, _ in ranked]
        lab = int(labels[i])
        per_class_n[lab] += 1
        confusion[lab][top_ids[0] if top_ids else None] += 1
        if top_ids:
            per_class[lab]["hit1"] += 1 if top_ids[0] in truth else 0
        for k in (1, 3, 5):
            if k > args.topk:
                continue
            if set(top_ids[:k]) & truth:
                hits[k] += 1
        if top_ids and top_ids[0] not in truth and len(misses) < 12:
            misses.append({
                "sample_index": i, "true_label": lab, "true_mitre": sorted(truth),
                "top": ranked[:3], "text_head": (texts[i] if i < len(texts) else "")[:160],
            })

    n = len(idxs)
    metrics = {
        "n_attack_samples": n,
        "topk": args.topk,
        "hit@1": round(hits[1] / n, 4) if n else None,
        "hit@3": round(hits[3] / n, 4) if n else None,
        "hit@5": round(hits[5] / n, 4) if n else None,
        "per_class_top1": {int(k): f"{per_class[k]['hit1']}/{per_class_n[k]}" for k in sorted(per_class_n)},
        "confusion_top1": {int(k): dict(v.most_common()) for k, v in sorted(confusion.items())},
        "gate_top1_met": (hits[1] / n) >= 0.50 if n else False,
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(PROJECT_ROOT, "explanation_results", f"eval_retrieval_{ts}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "metrics": metrics, "misses": misses}, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("\n离线性检索评估已保存:", out)
    for m in misses[:4]:
        print("\n[MISS] label=%s true=%s top=%s" % (m["true_label"], m["true_mitre"], m["top"]))
        print("   text:", m["text_head"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
