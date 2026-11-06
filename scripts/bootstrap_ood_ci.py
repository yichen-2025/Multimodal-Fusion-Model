"""对现有 OOD 路由运行做 bootstrap 置信区间（不重跑训练/路由，只读报告 + test.npz）。

回答两个问题：
1. 单次运行的指标有多不稳定（样本级 bootstrap）；
2. concat 与纯文本的差值是否真的可分（配对 bootstrap：同一重采样样本上比较，
   并叠加 run 级重采样以纳入 backbone seed 波动）。

指标口径与报告一致：
- `auroc_genuine`  = origin 0(known) vs 1(真实未知) 的 AUROC；
- `tpr_at_fpr_1`   = 未知(label==10) vs 已知，FPR<=1% 下可达最大 TPR；
- `unknown_recall_simulated` = origin 2 中被阈值判为未知的比例（阈值取该 head 的 config）。
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OOD_DIR = os.path.join(PROJECT_ROOT, "ood_reports")
HEAD_DIR = os.path.join(PROJECT_ROOT, "saved_ood_heads")
NPZ = os.path.join(PROJECT_ROOT, "split_data", "dataset_200", "split_openset_0", "test.npz")

VARIANTS = {
    "A0*_no_num": [10, 15, 16],
    "A0*": [14, 17, 18, 92, 95],
    "A0*_add": [12, 19, 20],
    "A0*_attn": [13, 21, 22],
    "A3": [39, 42, 45, 80, 83],
    # 公平编码器对照（plan/2026-11-03）：每变体取 head seed 42、3 个 backbone
    # 2026-11-05 补 seed 后 A3/A3_adapter/A0* 扩到 5 个 backbone
    "A0*_last": [50, 53, 56],
    "A0_frozen": [59, 62, 65],
    "A3_adapter": [68, 71, 74, 86, 89],
}

# 改动前就存在的变体：缺报告属硬错误；新变体未训练时可跳过（否则整个脚本无法运行）
REQUIRED_VARIANTS = {"A0*_no_num", "A0*", "A0*_add", "A0*_attn", "A3"}


def newest_report(ood_id):
    best = None
    for path in glob.glob(os.path.join(OOD_DIR, "ood_routing_*.json")):
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        if d.get("ood_id") != ood_id:
            continue
        ts = os.path.basename(path).split("_", 2)[-1]
        if best is None or ts > best[0]:
            best = (ts, os.path.basename(path), d)
    return best


def head_threshold(ood_id):
    cfg = os.path.join(HEAD_DIR, "ood_head_%d" % ood_id, "config.json")
    return float(json.load(open(cfg, encoding="utf-8"))["ood_threshold"])


def tpr_at_fpr(y_true, score, target_fpr):
    """FPR <= target_fpr 下可达最大 TPR（分数越大越可能是未知），与报告实现同口径。"""
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(-score, kind="stable")
    s = score[order]
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    # 只在分数严格变化的边界取值（>= 阈值 等价于包含所有同分样本）
    last = np.r_[s[1:] != s[:-1], True]
    tpr = tp[last] / n_pos
    fpr = fp[last] / n_neg
    ok = fpr <= target_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def compute(idx, score, masks):
    """在给定样本索引上计算三个指标（先按 idx 取样，再套用掩码）。"""
    s = score[idx]
    gm = masks["genuine"][idx]
    yg = masks["y_genuine"][idx]
    out = {}
    if gm.any() and 0 < int(yg[gm].sum()) < int(gm.sum()):
        out["auroc_genuine"] = float(roc_auc_score(yg[gm], s[gm]))
    else:
        out["auroc_genuine"] = None
    out["tpr_at_fpr_1"] = tpr_at_fpr(masks["y_ood"][idx], s, 0.01)
    sm = masks["simulated"][idx]
    out["unknown_recall_simulated"] = float(np.mean(s[sm] >= masks["thr"])) if sm.any() else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    z = np.load(NPZ)
    labels = np.asarray(z["labels"], dtype=int)
    origin = np.asarray(z["unknown_origin"], dtype=int)
    n = len(labels)

    y_ood = (labels == 10).astype(int)
    genuine_mask = (origin == 0) | (origin == 1)
    y_genuine = (origin == 1).astype(int)

    runs = {}
    missing = []
    for v, oids in VARIANTS.items():
        runs[v] = []
        for oid in oids:
            hit = newest_report(oid)
            if hit is None:
                missing.append((v, oid))
                continue
            _, name, d = hit
            score = np.asarray(d["all_ood_scores"], dtype=float)
            if not np.array_equal(np.asarray(d["all_labels"]), labels):
                raise SystemExit("样本顺序不一致: %s" % name)
            runs[v].append({"ood_id": oid, "report": name, "score": score,
                            "thr": head_threshold(oid)})
    if missing:
        hard = [m for m in missing if m[0] in REQUIRED_VARIANTS]
        if hard:
            raise SystemExit("缺报告: %s" % (hard,))
        print("[警告] 以下新变体尚未训练，本轮跳过: %s" % (missing,), file=sys.stderr)
    runs = {v: rs for v, rs in runs.items() if rs}

    # 1) 复算校验：全样本上必须与报告数值一致
    print("=" * 100)
    print("复算校验（全样本）")
    print("=" * 100)
    bad = []
    for v, rs in runs.items():
        for r in rs:
            masks = {"genuine": genuine_mask, "y_genuine": y_genuine, "y_ood": y_ood,
                     "simulated": origin == 2, "thr": r["thr"]}
            m = compute(np.arange(n), r["score"], masks)
            d = json.load(open(os.path.join(OOD_DIR, r["report"]), encoding="utf-8"))["routed"]
            diffs = {k: abs(m[k] - d[k]) for k in m}
            flag = max(diffs.values()) > 1e-6
            if flag:
                bad.append((r["ood_id"], diffs))
            print("ood %-3d %-40s auroc d=%9.2e  tpr d=%9.2e  recsim d=%9.2e %s" % (
                r["ood_id"], r["report"], diffs["auroc_genuine"], diffs["tpr_at_fpr_1"],
                diffs["unknown_recall_simulated"], "  <== 不一致" if flag else ""))
    if bad:
        print("[警告] 复算与报告不一致: %s" % (bad,))
        print("[警告] 原实现按 unique 阈值全量扫描，本脚本用去重累积 ROC，若仅 tpr 末位不同可接受")

    rng = np.random.default_rng(args.seed)
    metrics = ["auroc_genuine", "tpr_at_fpr_1", "unknown_recall_simulated"]

    def masks_for(rs):
        return [{"genuine": genuine_mask, "y_genuine": y_genuine, "y_ood": y_ood,
                 "simulated": origin == 2, "thr": r["thr"]} for r in rs]

    ms = {v: masks_for(rs) for v, rs in runs.items()}

    # 2) 样本级 bootstrap（展示单次运行的不稳定度）
    print("\n" + "=" * 100)
    print("单次运行指标 95%% CI（样本级 bootstrap，B=%d）" % args.n_boot)
    print("=" * 100)
    sample_ci = {}
    for v, rs in runs.items():
        for i, r in enumerate(rs):
            vals = {k: [] for k in metrics}
            for _ in range(args.n_boot):
                idx = rng.integers(0, n, n)
                m = compute(idx, r["score"], ms[v][i])
                for k in metrics:
                    if m[k] is not None:
                        vals[k].append(m[k])
            sample_ci[(v, r["ood_id"])] = {k: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
                                           for k, x in vals.items()}
            pt = compute(np.arange(n), r["score"], ms[v][i])
            print("%-11s ood %-3d  auroc %.4f [%.4f, %.4f]   tpr1 %.4f [%.4f, %.4f]   recsim %.4f [%.4f, %.4f]" % (
                v, r["ood_id"],
                pt["auroc_genuine"], *sample_ci[(v, r["ood_id"])]["auroc_genuine"],
                pt["tpr_at_fpr_1"], *sample_ci[(v, r["ood_id"])]["tpr_at_fpr_1"],
                pt["unknown_recall_simulated"], *sample_ci[(v, r["ood_id"])]["unknown_recall_simulated"]))

    # 3) 配对 bootstrap：A0* vs 其它变体；A3 vs A0* / A0*_no_num
    print("\n" + "=" * 100)
    print("差值 95% CI（配对 bootstrap：同一重采样样本，两个变体同口径比较）")
    print("=" * 100)

    def diff_between(ref_name, other_name):
        dst = {k: [] for k in metrics}
        dst_h = {k: [] for k in metrics}
        for _ in range(args.n_boot):
            idx = rng.integers(0, n, n)
            a = [compute(idx, r["score"], m) for r, m in zip(runs[ref_name], ms[ref_name])]
            b = [compute(idx, r["score"], m) for r, m in zip(runs[other_name], ms[other_name])]
            for k in metrics:
                av = [x[k] for x in a if x[k] is not None]
                bv = [x[k] for x in b if x[k] is not None]
                if av and bv:
                    dst[k].append(float(np.mean(av) - np.mean(bv)))
            # 叠加 run 级重采样（纳入 backbone seed 波动）
            ra = rng.integers(0, len(runs[ref_name]), len(runs[ref_name]))
            rb = rng.integers(0, len(runs[other_name]), len(runs[other_name]))
            a2 = [compute(idx, runs[ref_name][j]["score"], ms[ref_name][j]) for j in ra]
            b2 = [compute(idx, runs[other_name][j]["score"], ms[other_name][j]) for j in rb]
            for k in metrics:
                av = [x[k] for x in a2 if x[k] is not None]
                bv = [x[k] for x in b2 if x[k] is not None]
                if av and bv:
                    dst_h[k].append(float(np.mean(av) - np.mean(bv)))

        point = {}
        for k in metrics:
            pa = np.mean([compute(np.arange(n), r["score"], m)[k] for r, m in zip(runs[ref_name], ms[ref_name])])
            pb = np.mean([compute(np.arange(n), r["score"], m)[k] for r, m in zip(runs[other_name], ms[other_name])])
            point[k] = float(pa - pb)
        res = {
            k: {
                "point": point[k],
                "ci_sample": [float(np.percentile(dst[k], 2.5)), float(np.percentile(dst[k], 97.5))],
                "p_sample_two_sided": float(2 * min(np.mean(np.array(dst[k]) <= 0), np.mean(np.array(dst[k]) >= 0))),
                "ci_sample_plus_seed": [float(np.percentile(dst_h[k], 2.5)), float(np.percentile(dst_h[k], 97.5))],
                "p_sample_plus_seed_two_sided": float(2 * min(np.mean(np.array(dst_h[k]) <= 0), np.mean(np.array(dst_h[k]) >= 0))),
            } for k in metrics
        }
        for k in metrics:
            r0 = res[k]
            print("%-22s %-25s diff=%+7.2fpp  CI[%.2f, %.2f] p=%.4f   (含 seed) CI[%.2f, %.2f] p=%.4f" % (
                ref_name + " - " + other_name, k, r0["point"] * 100, r0["ci_sample"][0] * 100, r0["ci_sample"][1] * 100,
                r0["p_sample_two_sided"],
                r0["ci_sample_plus_seed"][0] * 100, r0["ci_sample_plus_seed"][1] * 100,
                r0["p_sample_plus_seed_two_sided"]))
        return res

    pair_vs_concat = [v for v in ("A0*_no_num", "A0*_add", "A0*_attn", "A0*_last", "A0_frozen")
                      if v in runs and "A0*" in runs]
    pair_vs_a3 = [v for v in ("A0*", "A0*_no_num", "A3_adapter") if v in runs and "A3" in runs]
    diff_res = {other: diff_between("A0*", other) for other in pair_vs_concat}
    diff_a3 = {other: diff_between("A3", other) for other in pair_vs_a3}
    diff_last_vs_adapter = ({"A3_adapter": diff_between("A0*_last", "A3_adapter")}
                            if "A0*_last" in runs and "A3_adapter" in runs else {})

    summary = {
        "n_boot": args.n_boot,
        "seed": args.seed,
        "test_samples": n,
        "runs": {v: [{"ood_id": r["ood_id"], "report": r["report"]} for r in rs] for v, rs in runs.items()},
        "single_run_ci": {"%s|%d" % k: v for k, v in sample_ci.items()},
        "diff_vs_concat": diff_res,
        "diff_A3": diff_a3,
        "diff_A0*_last_vs": diff_last_vs_adapter,
        "recompute_mismatch": bad,
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(summary, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("\n[写出] %s" % args.out)


if __name__ == "__main__":
    main()
