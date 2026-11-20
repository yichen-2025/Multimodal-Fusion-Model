"""E6：原型距离型开集检测的三组对照（CPU 分钟级）。

定义严格来自总纲 §4.1 的开集规则：`min_c dist(e_x, p_c) > τ → unknown`。
三组原型来源：
  zero_shot   P_llm（类描述嵌入）
  warm_start  P_mix(α, k-shot 均值)（语义 + 少量标注）
  random      P_rand（随机初始化，下限对照）

指标：AUROC（known vs 未知）、`tpr@fpr=1%`（阈值取 known 分数的 99 分位）。
门槛（C-28 §5.1）：warm_start 相对 zero_shot 的 `tpr@fpr=1%` +2pp，且配对 bootstrap 95% CI 不跨零。
未知分两类报告：真实未知（origin=1）与模拟未知（origin=2），不合并。

用法：
  python scripts/eval_open_set_prototypes.py \
      --split_dir split_data/dataset_200/split_openset_0 \
      --protos emb/qwen_protos.npy --text_npz emb/qwen_texts.npz \
      --out results/2026-11-20/e6_qwen
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

NUM_KNOWN = 10
SEEDS = [42, 123, 2024, 7, 13]
GATE_PP = 0.02


def l2norm(a):
    return a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12, None)


def dists(X, P, metric):
    if metric == "cosine":
        return 1.0 - l2norm(X) @ l2norm(P).T
    return np.linalg.norm(X[:, None, :] - P[None, :, :], axis=2)


def scores(X, P, metric):
    """min 距离越大越像未知。"""
    return dists(X, P, metric).min(axis=1)


def pick_metric(Xva, yva, P_llm):
    best = (None, -1.0)
    for m in ("cosine", "euclidean"):
        P = l2norm(P_llm) if m == "cosine" else P_llm
        d = scores(Xva, P, m)
        known = yva < NUM_KNOWN
        # 零样本闭集 acc 作选择依据
        pred = np.argmin(dists(Xva, P, m), axis=1)
        acc = float((pred[known] == yva[known]).mean())
        if acc > best[1]:
            best = (m, acc)
    return best[0], best[1]


def shot_protos(X, y, k, rng):
    P = np.zeros((NUM_KNOWN, X.shape[1]))
    for c in range(NUM_KNOWN):
        idx = np.where(y == c)[0]
        P[c] = X[rng.choice(idx, k, replace=False)].mean(axis=0)
    return P


def mix(P_shot, P_llm, alpha, metric):
    if metric == "cosine":
        return l2norm((1 - alpha) * l2norm(P_shot) + alpha * l2norm(P_llm))
    return (1 - alpha) * P_shot + alpha * P_llm


def tpr_at_fpr1(sc_known, sc_unk):
    thr = np.quantile(sc_known, 0.99)
    return float((sc_unk > thr).mean()), float(thr)


def arm_metrics(sc_all, yte, ote, kte):
    known = sc_all[kte]
    real = sc_all[(yte == NUM_KNOWN) & (ote == 1)]
    sim = sc_all[(yte == NUM_KNOWN) & (ote == 2)]
    out = {}
    for tag, unk in (("genuine", real), ("simulated", sim)):
        if len(unk) == 0:
            out[f"auroc_{tag}"] = None
            out[f"tpr_at_fpr1_{tag}"] = None
            continue
        y = np.r_[np.ones(len(known)), np.zeros(len(unk))]
        s = np.r_[known, unk]
        out[f"auroc_{tag}"] = float(roc_auc_score(y, s))
        out[f"tpr_at_fpr1_{tag}"], out[f"threshold_{tag}"] = tpr_at_fpr1(known, unk)
    out["n_known"] = int(kte.sum())
    out["n_genuine"] = int(len(real))
    out["n_simulated"] = int(len(sim))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--protos", required=True)
    ap.add_argument("--text_npz", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.text_npz:
        zt = np.load(args.text_npz, allow_pickle=False)
        ids = zt["ids"].astype(str).tolist()
        E = zt["emb"].astype(np.float64)
        raw = {}
        for split in ("train", "val", "test"):
            d = np.load(os.path.join(args.split_dir, f"{split}.npz"))
            raw[split] = (d["labels"].astype(np.int64), d["unknown_origin"].astype(np.int64))
        def gather(split):
            ix = [i for i, s in enumerate(ids) if s.startswith(split + "_")]
            rows = [int(ids[i].split("_")[1]) for i in ix]
            y_all, o_all = raw[split]
            return E[ix], y_all[rows], o_all[rows]
        Xtr, ytr, _ = gather("train")
        Xva, yva, _ = gather("val")
        Xte, yte, ote = gather("test")
    else:
        def ld(split):
            d = np.load(os.path.join(args.split_dir, f"{split}.npz"))
            return (d["text_embeddings"].astype(np.float64),
                    d["labels"].astype(np.int64), d["unknown_origin"].astype(np.int64))
        Xtr, ytr, _ = ld("train")
        Xva, yva, _ = ld("val")
        Xte, yte, ote = ld("test")

    P_llm = np.load(args.protos).astype(np.float64) if args.protos.endswith(".npy") \
        else np.load(args.protos)["proto"].astype(np.float64)

    metric, val_acc = pick_metric(Xva, yva, P_llm)
    ktr, kte = ytr < NUM_KNOWN, yte < NUM_KNOWN

    # 三组分数
    sc_zero = scores(Xte, P_llm, metric)
    sc_rand, sc_warm = [], []
    for s in SEEDS:
        Pm = mix(shot_protos(Xtr[ktr], ytr[ktr], args.k, np.random.default_rng(1000 + s)),
                 P_llm, args.alpha, metric)
        sc_warm.append(scores(Xte, Pm, metric))
        pr = l2norm(np.random.default_rng(2000 + s).standard_normal(P_llm.shape)) \
            if metric == "cosine" else np.random.default_rng(2000 + s).standard_normal(P_llm.shape)
        sc_rand.append(scores(Xte, pr, metric))
    sc_warm = np.mean(sc_warm, axis=0)
    sc_rand = np.mean(sc_rand, axis=0)

    arms = {
        "zero_shot": arm_metrics(sc_zero, yte, ote, kte),
        "warm_start": arm_metrics(sc_warm, yte, ote, kte),
        "random": arm_metrics(sc_rand, yte, ote, kte),
    }

    # 配对 bootstrap：tpr@fpr=1%，known/unknown 分别重采样
    idx_k = np.where(kte)[0]
    idx_u = np.where((yte == NUM_KNOWN) & (ote == 1))[0]
    diffs, diffs_a = [], []
    for _ in range(args.boot):
        bk = idx_k[rng.integers(0, len(idx_k), len(idx_k))]
        bu = idx_u[rng.integers(0, len(idx_u), len(idx_u))]
        thr_z = np.quantile(sc_zero[bk], 0.99)
        thr_w = np.quantile(sc_warm[bk], 0.99)
        diffs.append(float((sc_warm[bu] > thr_w).mean() - (sc_zero[bu] > thr_z).mean()))
        yk = np.ones(len(bk)); yu = np.zeros(len(bu))
        diffs_a.append(float(roc_auc_score(np.r_[yk, yu], np.r_[sc_warm[bk], sc_warm[bu]])
                             - roc_auc_score(np.r_[yk, yu], np.r_[sc_zero[bk], sc_zero[bu]])))
    ci_tpr = [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]
    ci_auc = [float(np.quantile(diffs_a, 0.025)), float(np.quantile(diffs_a, 0.975))]

    out = {
        "experiment": "E6 原型距离开集三组对照",
        "split_dir": args.split_dir,
        "protos": args.protos,
        "text_source": args.text_npz or "split npz text_embeddings (BERT, 768)",
        "metric_selected_on_val": metric,
        "val_zero_shot_acc": val_acc,
        "k": args.k, "alpha": args.alpha, "seeds": SEEDS,
        "arms": arms,
        "delta_warm_minus_zero": {
            "tpr_at_fpr1_genuine": float(arms["warm_start"]["tpr_at_fpr1_genuine"]
                                         - arms["zero_shot"]["tpr_at_fpr1_genuine"]),
            "tpr_ci95": ci_tpr,
            "auroc_genuine": float(arms["warm_start"]["auroc_genuine"]
                                   - arms["zero_shot"]["auroc_genuine"]),
            "auroc_ci95": ci_auc,
        },
        "gate_pp": GATE_PP,
        "boot": args.boot,
    }
    d = out["delta_warm_minus_zero"]
    out["gate_pass"] = bool(d["tpr_at_fpr1_genuine"] >= GATE_PP and d["tpr_ci95"][0] > 0)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({
        "metric": metric,
        "zero_tpr1": arms["zero_shot"]["tpr_at_fpr1_genuine"],
        "warm_tpr1": arms["warm_start"]["tpr_at_fpr1_genuine"],
        "rand_tpr1": arms["random"]["tpr_at_fpr1_genuine"],
        "delta_pp": d["tpr_at_fpr1_genuine"], "ci": ci_tpr,
        "zero_auroc": arms["zero_shot"]["auroc_genuine"],
        "warm_auroc": arms["warm_start"]["auroc_genuine"],
        "gate_pass": out["gate_pass"],
    }, ensure_ascii=False, indent=2))
    print(f"[OK] -> {args.out}/metrics.json")


if __name__ == "__main__":
    main()
