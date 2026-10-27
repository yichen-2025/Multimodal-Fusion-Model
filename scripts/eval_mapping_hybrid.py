"""同一批攻击样本上对比三种 MITRE 映射路径：LLM 自选 / 检索 top-1 / 两者并集。

只读取已有解释结果 JSON，不重新生成、不参与判类；分母只取 true_label 属已知类且有真值技术的样本
（与 eval_explanations.py 的 attack 口径一致）。

用法：
    python scripts/eval_mapping_hybrid.py \
        --result explanation_results/explanations_model2000_20261026_100154.json \
        --dataset_id 200 --split_id 0
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

from src.explain.attack_mapping import get_true_mitre_ids
from src.explain.retriever import get_retriever

LABEL_NAMES = {
    0: "BENIGN", 1: "DoS Hulk", 2: "DoS GoldenEye", 3: "DoS slowloris", 4: "DoS Slowhttptest",
    5: "DDoS", 6: "PortScan", 7: "FTP-Patator", 8: "SSH-Patator", 9: "Bot",
}


def main():
    ap = argparse.ArgumentParser(description="LLM / 检索 / 并集 MITRE 映射对比")
    ap.add_argument("--result", type=str, required=True, help="解释结果 JSON（explanations_*.json）")
    ap.add_argument("--dataset_id", type=int, default=200)
    ap.add_argument("--split_id", type=int, default=0)
    ap.add_argument("--min_rag_hit", type=float, default=0.70, help="检索 top-1 命中通过线")
    ap.add_argument("--min_union_hit", type=float, default=0.75, help="并集命中通过线")
    args = ap.parse_args()

    result_path = args.result if os.path.isabs(args.result) else os.path.join(PROJECT_ROOT, args.result)
    payload = json.load(open(result_path, encoding="utf-8"))
    results = payload["results"]
    K = payload["config"]["num_known_classes"]

    attack = [
        r for r in results
        if r["true_label"] < K and get_true_mitre_ids(r["true_label"])
    ]

    retriever = get_retriever()
    hits = Counter()
    per_class = defaultdict(Counter)
    per_class_n = Counter()
    details = []

    for r in attack:
        truth = set(get_true_mitre_ids(r["true_label"]))
        llm_pred = r.get("predicted_mitre_id")
        rag_pred = retriever.top1(r.get("text_description", "") or "")

        llm_hit = llm_pred in truth
        rag_hit = rag_pred in truth
        union_hit = llm_hit or rag_hit
        lab = r["true_label"]
        per_class_n[lab] += 1
        per_class[lab]["llm"] += int(llm_hit)
        per_class[lab]["rag"] += int(rag_hit)
        per_class[lab]["union"] += int(union_hit)

        hits["llm"] += int(llm_hit)
        hits["rag"] += int(rag_hit)
        hits["union"] += int(union_hit)
        hits["both"] += int(llm_hit and rag_hit)
        hits["rag_rescues"] += int((not llm_hit) and rag_hit)
        hits["rag_hurts"] += int(llm_hit and (not rag_hit))
        hits["neither"] += int(not union_hit)

        details.append({
            "sample_index": r["sample_index"],
            "true_label": lab,
            "true_label_name": LABEL_NAMES.get(lab, str(lab)),
            "true_mitre_ids": sorted(truth),
            "llm_pred": llm_pred,
            "rag_pred": rag_pred,
            "llm_hit": llm_hit,
            "rag_hit": rag_hit,
            "union_hit": union_hit,
        })

    n = len(attack)
    metrics = {
        "result_file": os.path.basename(result_path),
        "n_attack_samples": n,
        "llm_hit": f"{hits['llm']}/{n}",
        "llm_hit_rate": round(hits["llm"] / n, 4) if n else None,
        "rag_hit": f"{hits['rag']}/{n}",
        "rag_hit_rate": round(hits["rag"] / n, 4) if n else None,
        "union_hit": f"{hits['union']}/{n}",
        "union_hit_rate": round(hits["union"] / n, 4) if n else None,
        "both_hit": f"{hits['both']}/{n}",
        "agree_rate": round(sum(1 for d in details if d["llm_pred"] == d["rag_pred"]) / n, 4) if n else None,
        "rag_rescues": hits["rag_rescues"],
        "rag_hurts": hits["rag_hurts"],
        "neither_hit": hits["neither"],
        "per_class": {
            int(k): f"n={per_class_n[k]} llm={per_class[k]['llm']} rag={per_class[k]['rag']} union={per_class[k]['union']}"
            for k in sorted(per_class_n)
        },
        "gates": {
            "rag_hit_met": (hits["rag"] / n if n else 0) >= args.min_rag_hit,
            "union_hit_met": (hits["union"] / n if n else 0) >= args.min_union_hit,
            "rag_net_positive": hits["rag_rescues"] > hits["rag_hurts"],
        },
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(PROJECT_ROOT, "explanation_results", f"eval_mapping_hybrid_{ts}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "metrics": metrics, "details": details}, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("\n映射对比结果已保存:", out)
    print("[逐条差异] LLM 错而检索对：", [d["sample_index"] for d in details if (not d["llm_hit"] and d["rag_hit"])])
    print("[逐条差异] LLM 对而检索错：", [d["sample_index"] for d in details if (d["llm_hit"] and not d["rag_hit"])])
    print("[逐条差异] 两者都错：", [d["sample_index"] for d in details if not d["union_hit"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())

