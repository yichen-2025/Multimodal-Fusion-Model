#!/usr/bin/env python3
"""C1 配对差复算：从 5-seed 汇总 JSON 复算 A0* 与对照变体的配对差与 t-CI。

用法:
    python scripts/verify_c1_pair.py \
        --points results/2026-11-03/ood_multiseed_fair_encoder_n5.json \
        --a "A0*" --b A3 --metric tpr_at_fpr_1
"""
import argparse
import json
import math
import statistics

# 双侧 95% t 临界值（df=1..20），避免额外依赖 scipy
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
        19: 2.093, 20: 2.086}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--a", default="A0*")
    ap.add_argument("--b", default="A3")
    ap.add_argument("--metric", default="tpr_at_fpr_1")
    ap.add_argument("--head_seed", type=int, default=42,
                    help="只取该 head seed 的点位（n5 汇总口径固定为 42）")
    args = ap.parse_args()

    d = json.load(open(args.points))
    by = {}
    for p in d["points"]:
        if p.get("head_seed", args.head_seed) != args.head_seed:
            continue
        by.setdefault(p["variant"], {})[p["backbone_seed"]] = p[args.metric]

    seeds = sorted(set(by[args.a]) & set(by[args.b]))
    if len(seeds) < 2:
        raise SystemExit("可配对 seed 不足: %s" % seeds)
    diffs = [(by[args.a][s] - by[args.b][s]) * 100 for s in seeds]  # pp
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    se = sd / math.sqrt(len(diffs))
    tc = T975.get(len(diffs) - 1, 2.086)
    lo, hi = mean - tc * se, mean + tc * se
    print("配对 seed:", seeds)
    print("%s %s = %.4f ± %.4f (n=%d)" % (args.a, args.metric,
          statistics.mean([by[args.a][s] for s in seeds]) * 100,
          statistics.stdev([by[args.a][s] for s in seeds]) * 100, len(seeds)))
    print("%s %s = %.4f ± %.4f (n=%d)" % (args.b, args.metric,
          statistics.mean([by[args.b][s] for s in seeds]) * 100,
          statistics.stdev([by[args.b][s] for s in seeds]) * 100, len(seeds)))
    print("diff = %+.4fpp, t-CI95 [%+.2f, %+.2f], 跨零=%s"
          % (mean, lo, hi, lo <= 0 <= hi))


if __name__ == "__main__":
    main()
