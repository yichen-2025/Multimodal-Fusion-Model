"""CTI-RAG 离线检索评估：top-k 命中 true_mitre_ids。

只读划分与文本特征，不参与分类；BENIGN（无技术）不计入分母。

用法：
    python scripts/eval_retrieval.py --backbone_model_id 2000 --dataset_id 200 --split_id 0
    python scripts/eval_retrieval.py --kb b --tag b    # 扩库实验：B 臂，产物进 results/2026-10-28/

扩库实验约定（--kb）：
    a  原 7 条（逐字复制），用于复现基线；
    b  7 条 + 3 条父技术；
    c  10 条 + 12 条混淆技术。
缺省不带 --kb 时仍用 MITRE_KB，行为与改动前完全一致。
"""

import argparse
import csv
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


BEHAVIOR_PREFIX = ("port_", "pkt_", "dur_", "pps_", "ack_", "syn_")


def family_of(mitre_id):
    """技术编号 -> 父技术编号（T1498.001 -> T1498）。"""
    return mitre_id.split(".")[0]


def build_retriever(args):
    """按 --kb / --strip_keywords 构造检索器，返回 (retriever, kb_desc, kb_size)。"""
    if not args.kb and not args.strip_keywords:
        # 保持历史路径：默认构造，索引与 MITRE_KB 完全一致
        return BM25Retriever(), "src/explain/knowledge_base.py:MITRE_KB", len(BM25Retriever().ids)

    if args.kb:
        from src.explain import kb_candidates
        entries = kb_candidates.get_arm(args.kb)
        kb_desc = "src/explain/kb_candidates.py:" + kb_candidates.ARM_LABELS[args.kb]
    else:
        from src.explain.knowledge_base import get_kb_entries
        entries = [dict(e) for e in get_kb_entries()]
        kb_desc = "src/explain/knowledge_base.py:MITRE_KB"

    if args.strip_keywords:
        entries = [dict(e, keywords=[]) for e in entries]
        kb_desc += "+strip_keywords"
    if args.keep_behavior_keywords:
        entries = [dict(e, keywords=[k for k in e.get("keywords", [])
                                    if str(k).startswith(BEHAVIOR_PREFIX)]) for e in entries]
        kb_desc += "+behavior_keywords_only"
    return BM25Retriever(entries=entries), kb_desc, len(entries)


def main():
    ap = argparse.ArgumentParser(description="CTI 检索离线评估")
    ap.add_argument("--backbone_model_id", type=int, default=2000, help="用于读取 num_known_classes")
    ap.add_argument("--dataset_id", type=int, default=200)
    ap.add_argument("--split_id", type=int, default=0)
    ap.add_argument("--openset", dest="openset", action="store_true", default=True)
    ap.add_argument("--no-openset", dest="openset", action="store_false")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="只评估前 N 条（调试用）")
    ap.add_argument("--kb", choices=["a", "b", "c"], default=None,
                    help="扩库实验臂；缺省用 MITRE_KB 7 条")
    ap.add_argument("--strip_keywords", action="store_true",
                    help="清空条目 keywords 后建索引（敏感性对照）")
    ap.add_argument("--keep_behavior_keywords", action="store_true",
                    help="只保留 port_/pkt_/dur_/pps_/ack_/syn_ 行为词元（敏感性对照）")
    ap.add_argument("--tag", default=None, help="产物标签；给出时写入 out_dir")
    ap.add_argument("--out_dir", default=os.path.join("results", "2026-10-28"),
                    help="--tag 时的产物目录（相对项目根）")
    args = ap.parse_args()

    split_name = f"split_openset_{args.split_id}" if args.openset else f"split_{args.split_id}"
    split_dir = os.path.join(PROJECT_ROOT, "split_data", f"dataset_{args.dataset_id}", split_name)
    npz = np.load(os.path.join(split_dir, "test.npz"), allow_pickle=True)
    labels = npz["labels"]
    df = pd.read_csv(os.path.join(split_dir, "test_data.csv"))
    texts = df["text_description"].tolist() if "text_description" in df.columns else [""] * len(labels)

    K = read_model_num_classes(os.path.join(PROJECT_ROOT, "saved_models", f"model_{args.backbone_model_id}"))

    retriever, kb_desc, kb_size = build_retriever(args)
    idxs = [i for i in range(len(labels))
            if int(labels[i]) < K and get_true_mitre_ids(int(labels[i]))]
    if args.limit:
        idxs = idxs[: args.limit]

    hits = Counter()
    fam_hits = Counter()
    per_class = defaultdict(lambda: Counter())     # label -> hit@1 计数
    per_class_n = Counter()
    fam_per_class = defaultdict(lambda: Counter())  # label -> 家族级 hit@1 计数
    confusion = defaultdict(Counter)               # label -> top1 technique
    fam_confusion = defaultdict(Counter)           # label -> top1 family
    misses = []
    points = []

    for i in idxs:
        truth = get_true_mitre_ids(int(labels[i]))
        truth_fam = {family_of(m) for m in truth}
        ranked = retriever.retrieve(texts[i] if i < len(texts) else "", args.topk)
        top_ids = [mid for mid, _ in ranked]
        top1 = top_ids[0] if top_ids else None
        lab = int(labels[i])
        per_class_n[lab] += 1
        confusion[lab][top1] += 1
        fam_confusion[lab][family_of(top1) if top1 else None] += 1

        strict1 = bool(top1 in truth) if top1 else False
        fam1 = bool(top1 and family_of(top1) in truth_fam)
        if top1:
            per_class[lab]["hit1"] += 1 if strict1 else 0
            fam_per_class[lab]["hit1"] += 1 if fam1 else 0

        for k in (1, 3, 5):
            if k > args.topk:
                continue
            if set(top_ids[:k]) & truth:
                hits[k] += 1
            if {family_of(m) for m in top_ids[:k]} & truth_fam:
                fam_hits[k] += 1

        if top1 and not strict1 and len(misses) < 12:
            misses.append({
                "sample_index": i, "true_label": lab, "true_mitre": sorted(truth),
                "top": ranked[:3], "text_head": (texts[i] if i < len(texts) else "")[:160],
            })

        if args.tag:
            points.append({
                "sample_index": i, "label": lab,
                "true_mitre": "|".join(sorted(truth)),
                "top1": top1 or "", "top2": top_ids[1] if len(top_ids) > 1 else "",
                "top3": top_ids[2] if len(top_ids) > 2 else "",
                "score1": round(ranked[0][1], 6) if ranked else 0.0,
                "strict_hit1": int(strict1), "family_hit1": int(fam1),
            })

    n = len(idxs)
    metrics = {
        "n_attack_samples": n,
        "topk": args.topk,
        "kb": args.kb or "default",
        "kb_desc": kb_desc,
        "kb_size": kb_size,
        "strip_keywords": bool(args.strip_keywords),
        "hit@1": round(hits[1] / n, 4) if n else None,
        "hit@3": round(hits[3] / n, 4) if n else None,
        "hit@5": round(hits[5] / n, 4) if n else None,
        "family_hit@1": round(fam_hits[1] / n, 4) if n else None,
        "family_hit@3": round(fam_hits[3] / n, 4) if n else None,
        "per_class_top1": {int(k): f"{per_class[k]['hit1']}/{per_class_n[k]}" for k in sorted(per_class_n)},
        "family_per_class_top1": {int(k): f"{fam_per_class[k]['hit1']}/{per_class_n[k]}" for k in sorted(per_class_n)},
        "confusion_top1": {int(k): dict(v.most_common()) for k, v in sorted(confusion.items())},
        "family_confusion_top1": {int(k): dict(v.most_common()) for k, v in sorted(fam_confusion.items())},
        "gate_top1_met": (hits[1] / n) >= 0.50 if n else False,
    }

    if args.tag:
        out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"cti_kb_scale_{args.tag}.json")
        points_path = os.path.join(out_dir, f"cti_kb_scale_{args.tag}_points.csv")
        with open(points_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(points[0].keys()) if points else ["sample_index"])
            writer.writeheader()
            writer.writerows(points)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = os.path.join(PROJECT_ROOT, "explanation_results", f"eval_retrieval_{ts}.json")
        points_path = None

    with open(out, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "metrics": metrics, "misses": misses}, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("\n离线性检索评估已保存:", out)
    if points_path:
        print("逐条命中已保存:", points_path)
    for m in misses[:4]:
        print("\n[MISS] label=%s true=%s top=%s" % (m["true_label"], m["true_mitre"], m["top"]))
        print("   text:", m["text_head"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
