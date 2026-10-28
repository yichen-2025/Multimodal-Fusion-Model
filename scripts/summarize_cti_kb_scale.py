"""知识库扩容实验（A/B/C 三臂 + 敏感性对照）汇总。

只读 ``results/2026-10-28/cti_kb_scale_<tag>.json``，不重跑检索。
输出汇总 JSON 与 CSV（严格/家族级命中、分类别命中、top-1 混淆分布）。

用法：
    python scripts/summarize_cti_kb_scale.py --in_dir results/2026-10-28 \
        --out results/2026-10-28/cti_kb_scale_summary_20261028.json
"""

import argparse
import csv
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUNS = [
    ("a", "cti_kb_scale_a.json", "A：原 7 条"),
    ("b", "cti_kb_scale_b.json", "B：7 条 + 3 父技术"),
    ("c", "cti_kb_scale_c.json", "C：10 条 + 12 混淆项"),
    ("c_behavior", "cti_kb_scale_c_behavior.json", "C：仅保留行为词元"),
    ("c_strip", "cti_kb_scale_c_strip.json", "C：清空全部 keywords"),
]

NOTES = [
    "A 臂逐位复现基线：hit@1 0.8494 / hit@3 0.9498 / hit@5 1.0。",
    "B 臂 strict hit@1 大幅下降来自父技术抢占（T1498/T1110），家族级几乎不变。",
    "C 臂 strict 与家族级双双下降，主因 T1105 与 T1499.002 抢占。",
    "C 条目文本由本实验撰写，属假设前提；c_behavior / c_strip 为敏感性对照。",
]


def main():
    ap = argparse.ArgumentParser(description="知识库扩容三臂汇总")
    ap.add_argument("--in_dir", default=os.path.join("results", "2026-10-28"))
    ap.add_argument("--out", default=os.path.join("results", "2026-10-28",
                                                  "cti_kb_scale_summary_20261028.json"))
    args = ap.parse_args()

    in_dir = args.in_dir if os.path.isabs(args.in_dir) else os.path.join(PROJECT_ROOT, args.in_dir)
    out = args.out if os.path.isabs(args.out) else os.path.join(PROJECT_ROOT, args.out)

    data = {}
    for tag, fn, _ in RUNS:
        with open(os.path.join(in_dir, fn), encoding="utf-8") as f:
            data[tag] = json.load(f)

    base = data["a"]["metrics"]
    rows = []
    for tag, _, label in RUNS:
        m = data[tag]["metrics"]
        n = m["n_attack_samples"]
        rows.append({
            "arm": tag,
            "label": label,
            "kb_size": m["kb_size"],
            "n_attack_samples": n,
            "strict_hit@1": m["hit@1"],
            "strict_hit@3": m["hit@3"],
            "strict_hit@5": m["hit@5"],
            "family_hit@1": m["family_hit@1"],
            "family_hit@3": m["family_hit@3"],
            "strict_hit1_count": int(round(m["hit@1"] * n)),
            "delta_vs_a": int(round((m["hit@1"] - base["hit@1"]) * n)),
            "family_delta_vs_a": int(round((m["family_hit@1"] - base["family_hit@1"]) * n)),
            "per_class_top1": m["per_class_top1"],
            "family_per_class_top1": m["family_per_class_top1"],
            "confusion_top1": m["confusion_top1"],
            "kb_desc": m["kb_desc"],
        })

    summary = {
        "n_attack_samples": base["n_attack_samples"],
        "baseline_per_class_top1": base["per_class_top1"],
        "arms": rows,
        "notes": NOTES,
    }

    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    csv_path = out[:-5] + ".csv" if out.endswith(".json") else out + ".csv"
    fields = ["arm", "label", "kb_size", "strict_hit@1", "strict_hit@3", "strict_hit@5",
              "family_hit@1", "family_hit@3", "strict_hit1_count", "delta_vs_a", "family_delta_vs_a"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    print("| 臂 | 库大小 | strict hit@1 | hit@3 | hit@5 | 家族 hit@1 | 家族 hit@3 | 相对A条数 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| %s | %d | %.4f | %.4f | %.4f | %.4f | %.4f | %+d |" % (
            r["label"], r["kb_size"], r["strict_hit@1"], r["strict_hit@3"], r["strict_hit@5"],
            r["family_hit@1"], r["family_hit@3"], r["delta_vs_a"]))
    print("\nsaved:", out)
    print("saved:", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
