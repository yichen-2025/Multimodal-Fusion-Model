"""汇总 OOD 多 seed 指标，区分两级随机性并输出 mean±std（ddof=1）。

- backbone 级：每个变体 3 个 backbone seed（42/123/456），取 head seed 42，
  与历史口径一致，n=3；
- head 级：同一 backbone 上 head seed 42/43/44 的极差与标准差，用于判断
  head 初始化是否构成方差来源。

只读 ood_reports，不重跑训练或路由。每个 (变体, backbone, head seed) 对应一个
ood_id，取该 ood_id 下最新一份报告。
"""

import argparse
import glob
import json
import os
import statistics
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OOD_DIR = os.path.join(PROJECT_ROOT, "ood_reports")

# (variant, backbone_model_id, backbone_seed, head_seed, ood_id)
POINTS = [
    ("A0*_no_num", 2000, 42, 42, 10),
    ("A0*_no_num", 2001, 123, 42, 15),
    ("A0*_no_num", 2002, 456, 42, 16),
    ("A0*_no_num", 2001, 123, 43, 23),
    ("A0*_no_num", 2001, 123, 44, 24),
    ("A0*_no_num", 2002, 456, 43, 25),
    ("A0*_no_num", 2002, 456, 44, 26),
    ("A0*", 2013, 42, 42, 14),
    ("A0*", 2004, 123, 42, 17),
    ("A0*", 2005, 456, 42, 18),
    ("A0*", 2004, 123, 43, 27),
    ("A0*", 2004, 123, 44, 28),
    ("A0*", 2005, 456, 43, 29),
    ("A0*", 2005, 456, 44, 30),
    ("A0*_add", 2006, 42, 42, 12),
    ("A0*_add", 2007, 123, 42, 19),
    ("A0*_add", 2012, 456, 42, 20),
    ("A0*_add", 2007, 123, 43, 31),
    ("A0*_add", 2007, 123, 44, 32),
    ("A0*_add", 2012, 456, 43, 33),
    ("A0*_add", 2012, 456, 44, 34),
    ("A0*_attn", 2009, 42, 42, 13),
    ("A0*_attn", 2010, 123, 42, 21),
    ("A0*_attn", 2011, 456, 42, 22),
    ("A0*_attn", 2010, 123, 43, 35),
    ("A0*_attn", 2010, 123, 44, 36),
    ("A0*_attn", 2011, 456, 43, 37),
    ("A0*_attn", 2011, 456, 44, 38),
    ("A3", 3001, 42, 42, 39),
    ("A3", 3001, 42, 43, 40),
    ("A3", 3001, 42, 44, 41),
    ("A3", 3002, 123, 42, 42),
    ("A3", 3002, 123, 43, 43),
    ("A3", 3002, 123, 44, 44),
    ("A3", 3003, 456, 42, 45),
    ("A3", 3003, 456, 43, 46),
    ("A3", 3003, 456, 44, 47),
    # ── 公平编码器对照（plan/2026-11-03）：A0*_last / A0_frozen / A3_adapter ──
    ("A0*_last", 4001, 42, 42, 50),
    ("A0*_last", 4001, 42, 43, 51),
    ("A0*_last", 4001, 42, 44, 52),
    ("A0*_last", 4002, 123, 42, 53),
    ("A0*_last", 4002, 123, 43, 54),
    ("A0*_last", 4002, 123, 44, 55),
    ("A0*_last", 4003, 456, 42, 56),
    ("A0*_last", 4003, 456, 43, 57),
    ("A0*_last", 4003, 456, 44, 58),
    ("A0_frozen", 4011, 42, 42, 59),
    ("A0_frozen", 4011, 42, 43, 60),
    ("A0_frozen", 4011, 42, 44, 61),
    ("A0_frozen", 4012, 123, 42, 62),
    ("A0_frozen", 4012, 123, 43, 63),
    ("A0_frozen", 4012, 123, 44, 64),
    ("A0_frozen", 4013, 456, 42, 65),
    ("A0_frozen", 4013, 456, 43, 66),
    ("A0_frozen", 4013, 456, 44, 67),
    ("A3_adapter", 4021, 42, 42, 68),
    ("A3_adapter", 4021, 42, 43, 69),
    ("A3_adapter", 4021, 42, 44, 70),
    ("A3_adapter", 4022, 123, 42, 71),
    ("A3_adapter", 4022, 123, 43, 72),
    ("A3_adapter", 4022, 123, 44, 73),
    ("A3_adapter", 4023, 456, 42, 74),
    ("A3_adapter", 4023, 456, 43, 75),
    ("A3_adapter", 4023, 456, 44, 76),
    # ── 补 seed（2026-11-05）：A3/A3_adapter/A0* 各加 2 个 backbone，head seed 42/43/44 ──
    ("A3", 5001, 789, 42, 80),
    ("A3", 5001, 789, 43, 81),
    ("A3", 5001, 789, 44, 82),
    ("A3", 5002, 1011, 42, 83),
    ("A3", 5002, 1011, 43, 84),
    ("A3", 5002, 1011, 44, 85),
    ("A3_adapter", 5011, 789, 42, 86),
    ("A3_adapter", 5011, 789, 43, 87),
    ("A3_adapter", 5011, 789, 44, 88),
    ("A3_adapter", 5012, 1011, 42, 89),
    ("A3_adapter", 5012, 1011, 43, 90),
    ("A3_adapter", 5012, 1011, 44, 91),
    ("A0*", 5021, 789, 42, 92),
    ("A0*", 5021, 789, 43, 93),
    ("A0*", 5021, 789, 44, 94),
    ("A0*", 5022, 1011, 42, 95),
    ("A0*", 5022, 1011, 43, 96),
    ("A0*", 5022, 1011, 44, 97),
]

VARIANTS = ["A3", "A0*_no_num", "A0*", "A0*_add", "A0*_attn",
            "A0*_last", "A0_frozen", "A3_adapter"]
METRICS = ["auroc_genuine", "tpr_at_fpr_1", "unknown_recall_simulated",
           "known_accuracy", "unknown_recall_genuine"]


def newest_report(ood_id, model_id):
    """取 ood_id 与 backbone_model_id 都匹配的最新报告。"""
    best = None
    for path in glob.glob(os.path.join(OOD_DIR, "ood_routing_*.json")):
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        if d.get("ood_id") != ood_id or d.get("backbone_model_id") != model_id:
            continue
        ts = os.path.basename(path).split("_", 2)[-1].replace(".json", "")
        if best is None or ts > best[0]:
            best = (ts, os.path.basename(path), d)
    return best


def mean_sd(values):
    if not values:
        return None, None
    m = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) >= 2 else None
    return m, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None, help="汇总 JSON 输出路径")
    args = ap.parse_args()

    rows, missing = [], []
    for variant, model, bseed, hseed, ood_id in POINTS:
        hit = newest_report(ood_id, model)
        if hit is None:
            missing.append((variant, model, ood_id))
            continue
        _, name, d = hit
        m = d["routed"]
        rows.append({
            "variant": variant,
            "backbone_model_id": model,
            "backbone_seed": bseed,
            "head_seed": hseed,
            "ood_id": ood_id,
            "report": name,
            **{k: m[k] for k in METRICS},
            "num_unknown_routed": m["num_unknown_routed"],
        })
    if missing:
        print("[警告] 缺报告组合: %s" % (missing,), file=sys.stderr)

    out_variants = []
    for v in VARIANTS:
        vs = [r for r in rows if r["variant"] == v]
        if not vs:
            # 新变体尚未训练：不参与汇总，也不出现在 compare_* 里（避免 NaN）
            continue
        base = [r for r in vs if r["head_seed"] == 42]
        stats = {}
        for k in METRICS:
            m_b, sd_b = mean_sd([r[k] for r in base])
            # head 级：同一 backbone 内跨 head seed 的离散度
            head_spreads, head_sds, head_maxmin = [], [], []
            for model in sorted({r["backbone_model_id"] for r in vs}):
                vals = [r[k] for r in vs if r["backbone_model_id"] == model]
                if len(vals) >= 2:
                    head_maxmin.append(max(vals) - min(vals))
                    head_sds.append(statistics.stdev(vals))
                head_spreads.append(max(vals) - min(vals))
            stats[k] = {
                "backbone_level_mean": m_b,
                "backbone_level_sd": sd_b,
                "n_backbone": len(base),
                "head_max_range": max(head_maxmin) if head_maxmin else None,
                "head_max_sd": max(head_sds) if head_sds else None,
            }
        out_variants.append({"variant": v, "n_points": len(vs), "n_backbone": len(base),
                             "metrics": stats})

    by_name = {x["variant"]: x for x in out_variants}

    def compare(a_name, b_name):
        a, b = by_name.get(a_name), by_name.get(b_name)
        if not a or not b:
            return None
        out = {}
        for k in ("auroc_genuine", "tpr_at_fpr_1", "unknown_recall_simulated"):
            ma = a["metrics"][k]["backbone_level_mean"]
            mb = b["metrics"][k]["backbone_level_mean"]
            sda = a["metrics"][k]["backbone_level_sd"]
            sdb = b["metrics"][k]["backbone_level_sd"]
            diff = ma - mb
            maxsd = max(sda, sdb)
            pooled = ((sda ** 2 + sdb ** 2) / 2) ** 0.5
            out[k] = {
                "diff_pp": diff * 100,
                "sd_A0*": sda, "sd_other": sdb,
                "ratio_max_sd": abs(diff) / maxsd if maxsd else None,
                "ratio_pooled_sd": abs(diff) / pooled if pooled else None,
            }
        return out

    summary = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "ddof": 1,
        "note": ("backbone_level_* 用 head seed 42，n=3 与历史口径一致；"
                 "head_max_range/head_max_sd 为同一 backbone 上 head seed 42/43/44 的离散度"),
        "metrics": METRICS,
        "points": rows,
        "variants": out_variants,
        "compare_A0*_vs": {v: compare("A0*", v) for v in ("A3", "A0*_no_num", "A0*_add", "A0*_attn",
                                                          "A0*_last", "A0_frozen")},
        "compare_A3_vs": {v: compare("A3", v) for v in ("A0*", "A0*_no_num", "A3_adapter")},
        "compare_A0*_last_vs": {v: compare("A0*_last", v) for v in ("A3_adapter",)},
    }

    print("=" * 104)
    print("变体级统计（backbone_level：head seed 42，n=3，ddof=1；head 级：同 backbone 跨 head seed 42/43/44）")
    print("=" * 104)
    print("%-12s %-26s %-18s %-22s" % ("variant", "metric", "mean±sd (n=3)", "head_max_range / head_max_sd"))
    for v in out_variants:
        for k, s in v["metrics"].items():
            print("%-12s %-26s %.4f±%.4f      %.5f / %s" % (
                v["variant"], k, s["backbone_level_mean"], s["backbone_level_sd"],
                s["head_max_range"] if s["head_max_range"] is not None else float("nan"),
                ("%.5f" % s["head_max_sd"]) if s["head_max_sd"] is not None else "-"))
        print("-" * 104)

    print("逐点明细：")
    print("%-12s %-6s %-6s %-6s %-4s %-11s %-9s %-9s %-9s %-9s %-6s" % (
        "variant", "model", "bseed", "hseed", "ood", "auroc", "tpr@fpr1", "rec_sim", "rec_gen", "known_acc", "unk"))
    for r in rows:
        print("%-12s %-6d %-6d %-6d %-4d %-11.7f %-9.5f %-9.5f %-9.5f %-9.5f %-6d" % (
            r["variant"], r["backbone_model_id"], r["backbone_seed"], r["head_seed"], r["ood_id"],
            r["auroc_genuine"], r["tpr_at_fpr_1"], r["unknown_recall_simulated"],
            r["unknown_recall_genuine"], r["known_accuracy"], r["num_unknown_routed"]))

    print("=" * 104)
    print("关键对比（A0*/concat 减其它变体；倍数以 backbone 级 sd 为尺）")
    for prefix, comps in (("A0* - ", summary["compare_A0*_vs"]),
                          ("A3 - ", summary["compare_A3_vs"])):
        for name, cmp in comps.items():
            if not cmp:
                continue
            for k, c in cmp.items():
                print("%-14s %-26s diff=%+6.2fpp  sd_A0*=%.5f sd_other=%.5f  max倍数=%.1f pooled倍数=%.1f" % (
                    prefix + name, k, c["diff_pp"], c["sd_A0*"], c["sd_other"],
                    c["ratio_max_sd"], c["ratio_pooled_sd"]))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("\n[写出] %s" % args.out)


if __name__ == "__main__":
    main()
