"""评估解释生成结果：分类准确率、MITRE 命中率、特征引用率、类别一致性。

只读取解释结果 JSON 与划分文件，不参与生成过程；unknown_origin 仅用于评估分组。

用法：
    python scripts/eval_explanations.py --result explanation_results/explanations_model2000_20261026_090753.json
"""

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from src.explain.attack_mapping import get_true_mitre_ids

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")

# 规则版"解释与预测类别一致"：命中预测类别的关键词即可（粗糙代理指标，非语义判分）
CLASS_KEYWORDS = {
    0: ["benign", "正常流量", "正常", "无攻击"],
    1: ["hulk"],
    2: ["goldeneye"],
    3: ["slowloris", "慢速"],
    4: ["slowhttptest"],
    5: ["ddos", "分布式拒绝"],
    6: ["portscan", "端口扫描", "扫描"],
    7: ["ftp", "patator"],
    8: ["ssh", "patator"],
    9: ["bot", "僵尸"],
}

ORIGIN_NAMES = {0: "known", 1: "genuine", 2: "simulated"}


def parse_numbers(text):
    out = []
    for tok in NUM_RE.findall(text or ""):
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def number_in(values, target):
    for v in values:
        if math.isclose(v, target, rel_tol=1e-6, abs_tol=1e-9):
            return True
    return False


def load_split(dataset_id, split_id, openset):
    split_name = f"split_openset_{split_id}" if openset else f"split_{split_id}"
    npz = os.path.join(PROJECT_ROOT, "split_data", f"dataset_{dataset_id}", split_name, "test.npz")
    d = np.load(npz, allow_pickle=True)
    labels = d["labels"]
    origins = d["unknown_origin"] if "unknown_origin" in d.files else None
    return labels, origins


def main():
    ap = argparse.ArgumentParser(description="解释结果评估")
    ap.add_argument("--result", type=str, default=None, help="解释结果 JSON（默认取最新的 explanations_*.json）")
    ap.add_argument("--output", type=str, default=None, help="评估结果 JSON 输出路径（默认自动命名）")
    ap.add_argument("--min_ref_rate", type=float, default=0.90, help="特征引用率通过线")
    ap.add_argument("--min_mitre_hit", type=float, default=0.60, help="MITRE 攻击类命中率目标线")
    args = ap.parse_args()

    result_path = args.result or sorted(glob.glob(os.path.join(PROJECT_ROOT, "explanation_results", "explanations_*.json")))[-1]
    payload = json.load(open(result_path))
    cfg = payload["config"]
    results = payload["results"]
    K = cfg["num_known_classes"]

    labels, origins = load_split(cfg["dataset_id"], cfg["split_id"], cfg.get("openset", True))
    full_counts = Counter(int(x) for x in labels)
    total = len(labels)

    # ---------- 1) 分类准确率 ----------
    n = len(results)
    correct = sum(1 for r in results if r["correct"])
    known = [r for r in results if r["true_label"] < K]
    unknown = [r for r in results if r["true_label"] >= K]
    known_correct = sum(1 for r in known if r["correct"])

    per_class = defaultdict(lambda: [0, 0])  # label -> [correct, total]（样本内）
    for r in results:
        per_class[r["true_label"]][1] += 1
        per_class[r["true_label"]][0] += 1 if r["correct"] else 0
    weighted_acc = sum(full_counts[k] / total * (v[0] / v[1]) for k, v in per_class.items() if v[1])

    unknown_by_origin = Counter()
    for r in unknown:
        idx = r["sample_index"]
        if origins is not None and idx < len(origins):
            unknown_by_origin[ORIGIN_NAMES.get(int(origins[idx]), str(int(origins[idx])))] += 1
        else:
            unknown_by_origin["未记录"] += 1

    # ---------- 2) MITRE ----------
    attack = [r for r in known if get_true_mitre_ids(r["true_label"])]
    benign = [r for r in known if not get_true_mitre_ids(r["true_label"])]
    mitre_hits = [r for r in attack if r["predicted_mitre_id"] in get_true_mitre_ids(r["true_label"])]
    benign_ok = [r for r in benign if r["predicted_mitre_id"] is None]

    per_class_mitre = defaultdict(lambda: [0, 0])
    for r in attack:
        k = r["true_label"]
        per_class_mitre[k][1] += 1
        per_class_mitre[k][0] += 1 if r["predicted_mitre_id"] in get_true_mitre_ids(k) else 0

    pred_dist = Counter(str(r["predicted_mitre_id"]) for r in known)

    # ---------- 3) 特征引用率 ----------
    ref_ok = 0
    halluc = 0
    ref_details = []
    for r in results:
        text_nums = parse_numbers(r.get("text_description", ""))
        exp_nums = parse_numbers(r.get("explanation", ""))
        cited = [x for x in exp_nums if number_in(text_nums, x)]
        bad = [x for x in exp_nums if not number_in(text_nums, x)]
        if cited:
            ref_ok += 1
        if bad:
            halluc += 1
        ref_details.append({"sample_index": r["sample_index"], "n_numbers": len(exp_nums),
                           "n_verified": len(cited), "n_unverified": len(bad)})

    # ---------- 4) 类别一致性（规则版） ----------
    consistent = 0
    incons_samples = []
    for r in results:
        kws = CLASS_KEYWORDS.get(r["pred_label"], [])
        text = (r.get("explanation") or "").lower()
        if any(k.lower() in text for k in kws):
            consistent += 1
        else:
            incons_samples.append(r["sample_index"])

    metrics = {
        "result_file": os.path.basename(result_path),
        "n_samples": n,
        "sampling": cfg.get("sampling", "random(未记录)"),
        "backbone": {
            "accuracy_overall": round(correct / n, 4),
            "accuracy_known_only": round(known_correct / len(known), 4) if known else None,
            "accuracy_class_weighted": round(weighted_acc, 4),
            "n_known": len(known),
            "n_unknown": len(unknown),
            "unknown_by_origin": dict(unknown_by_origin),
            "per_class_sample_acc": {int(k): f"{v[0]}/{v[1]}" for k, v in sorted(per_class.items())},
        },
        "mitre": {
            "attack_only_hit_rate": round(len(mitre_hits) / len(attack), 4) if attack else None,
            "attack_n": len(attack),
            "benign_null_correct_rate": round(len(benign_ok) / len(benign), 4) if benign else None,
            "benign_n": len(benign),
            "known_task_acc": round((len(mitre_hits) + len(benign_ok)) / len(known), 4) if known else None,
            "per_class_hit": {int(k): f"{v[0]}/{v[1]}" for k, v in sorted(per_class_mitre.items())},
            "predicted_distribution": dict(pred_dist.most_common()),
        },
        "feature_reference": {
            "reference_rate": round(ref_ok / n, 4),
            "unverified_number_rate": round(halluc / n, 4),
            "gate_pass": (ref_ok / n) >= args.min_ref_rate,
        },
        "class_consistency_rule": {
            "rate": round(consistent / n, 4),
            "inconsistent_samples": incons_samples[:10],
        },
        "gates": {
            "mitre_hit_target_met": (len(mitre_hits) / len(attack) if attack else 0) >= args.min_mitre_hit,
            "feature_ref_met": (ref_ok / n) >= args.min_ref_rate,
        },
    }

    out_path = args.output or os.path.join(
        PROJECT_ROOT, "explanation_results",
        "eval_" + os.path.basename(result_path).replace("explanations_", "").replace(".json", "") + ".json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "reference_details": ref_details}, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("\n评估结果已保存:", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
