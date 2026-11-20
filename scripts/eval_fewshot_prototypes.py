"""E3/E4：BERT 空间的零样本语义原型与少样本原型（CPU 分钟级）。

对应执行总纲 §4.3 / §5.1 与 docs/claims.md C-28：
- E3 零样本：P_llm（类描述嵌入）直接做最近原型分类，门槛 acc >= 0.50；
- E4 少样本：4 个来源 × k∈{1,5,10,20} × 5 个采样种子
    P_rand / P_shot / P_llm / P_mix(α∈{0,.25,.5,.75,1})
  门槛：P_mix − P_shot >= 3pp @ k<=5，且配对 bootstrap 95% CI 不跨零。
- 距离度量只在 val 上选（cosine / euclidean），test 只评估一次。

不加载 torch、不修改现有训练/路由代码。

用法：
  python scripts/eval_fewshot_prototypes.py \
      --split_dir split_data/dataset_200/split_openset_0 \
      --protos emb/bert_protos.npz \
      --out results/2026-11-20/e3_e4_bert
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

NUM_KNOWN = 10
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
KS = [1, 5, 10, 20]
SEEDS = [42, 123, 2024, 7, 13]
METRICS = ["cosine", "euclidean"]

E3_ACC_GATE = 0.50
E4_DELTA_GATE = 0.03


def load_npz(split_dir, name):
    d = np.load(os.path.join(split_dir, f"{name}.npz"))
    return (d["text_embeddings"].astype(np.float64),
            d["labels"].astype(np.int64),
            d["unknown_origin"].astype(np.int64))


def l2norm(a):
    return a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12, None)


def distances(X, P, metric):
    """返回 (n, k) 距离矩阵。cosine 用归一化后内积，euclidean 用原始向量。"""
    if metric == "cosine":
        return 1.0 - l2norm(X) @ l2norm(P).T
    return np.linalg.norm(X[:, None, :] - P[None, :, :], axis=2)


def predict(X, P, metric):
    return np.argmin(distances(X, P, metric), axis=1)


def min_dist(X, P, metric):
    return distances(X, P, metric).min(axis=1)


def macro_f1(y, p):
    return float(f1_score(y, p, labels=list(range(NUM_KNOWN)), average="macro", zero_division=0))


def tpr_at_fpr1(scores_known, scores_unknown):
    """分数越高越像未知；阈值取 known 的 99 分位。"""
    thr = float(np.quantile(scores_known, 0.99))
    return float(np.mean(scores_unknown > thr))


def shot_prototypes(X, y, k, rng):
    P = np.zeros((NUM_KNOWN, X.shape[1]), dtype=np.float64)
    for c in range(NUM_KNOWN):
        idx = np.where(y == c)[0]
        take = rng.choice(idx, size=k, replace=False)
        P[c] = X[take].mean(axis=0)
    return P


def mix_proto(P_shot, P_llm, alpha, metric):
    if metric == "cosine":
        return l2norm((1 - alpha) * l2norm(P_shot) + alpha * l2norm(P_llm))
    return (1 - alpha) * P_shot + alpha * P_llm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--protos", required=True)
    ap.add_argument("--text_npz", default="",
                    help="可选：Qwen 空间的 ids/emb npz（emb/qwen_texts.npz）；缺省用 split 的 BERT text_embeddings")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.text_npz:
        # Qwen 空间：npz 里是 ids + emb，train 只含抽样池（train_<原始行号>）
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
        Xtr, ytr, _ = load_npz(args.split_dir, "train")
        Xva, yva, _ = load_npz(args.split_dir, "val")
        Xte, yte, ote = load_npz(args.split_dir, "test")

    ktr, kva, kte = ytr < NUM_KNOWN, yva < NUM_KNOWN, yte < NUM_KNOWN
    Xtr_k, ytr_k = Xtr[ktr], ytr[ktr]
    Xva_k, yva_k = Xva[kva], yva[kva]
    Xte_k, yte_k = Xte[kte], yte[kte]

    if args.protos.endswith(".npy"):
        P_llm = np.load(args.protos).astype(np.float64)
    else:
        z = np.load(args.protos)
        P_llm = z["proto"].astype(np.float64)

    # ---------- 距离度量只在 val 上选（用 P_llm 零样本） ----------
    metric_val = {}
    for m in METRICS:
        acc = float(accuracy_score(yva_k, predict(Xva_k, P_llm, m)))
        metric_val[m] = acc
    metric = max(metric_val, key=metric_val.get)

    # ---------- E3 零样本 ----------
    p_te = predict(Xte_k, P_llm, metric)
    e3 = {
        "metric_selected_on_val": metric,
        "val_acc_by_metric": metric_val,
        "val_acc": metric_val[metric],
        "test_acc": float(accuracy_score(yte_k, p_te)),
        "test_macro_f1": macro_f1(yte_k, p_te),
        "gate_acc_ge_0.50": None,
        "n_test_known": int(len(yte_k)),
    }
    e3["gate_acc_ge_0.50"] = bool(e3["test_acc"] >= E3_ACC_GATE)

    # 零样本 OOD（描述性）：到最近原型的距离越大越像未知
    d_all = min_dist(Xte, P_llm, metric)
    known_sc = d_all[kte]
    genuine = (yte == NUM_KNOWN) & (ote == 1)
    simulated = (yte == NUM_KNOWN) & (ote == 2)
    e3["ood_descriptive"] = {
        "auroc_genuine": float(roc_auc_score(
            np.r_[np.ones(len(known_sc)), np.zeros(int(genuine.sum()))],
            np.r_[known_sc, d_all[genuine]])) if genuine.any() else None,
        "tpr_at_fpr1_genuine": tpr_at_fpr1(known_sc, d_all[genuine]) if genuine.any() else None,
        "auroc_simulated": float(roc_auc_score(
            np.r_[np.ones(len(known_sc)), np.zeros(int(simulated.sum()))],
            np.r_[known_sc, d_all[simulated]])) if simulated.any() else None,
        "tpr_at_fpr1_simulated": tpr_at_fpr1(known_sc, d_all[simulated]) if simulated.any() else None,
        "note": "零样本 OOD 只作描述，不加阈值、不调 τ",
    }

    # ---------- E4 少样本 ----------
    table = []
    # k=0：只有 P_llm / P_rand 有意义
    for name, P in [("P_llm", P_llm),
                    ("P_rand", l2norm(rng.standard_normal(P_llm.shape)))]:
        pred = predict(Xte_k, P, metric)
        table.append({"k": 0, "source": name, "alpha": None, "seed": None,
                      "acc": float(accuracy_score(yte_k, pred)),
                      "macro_f1": macro_f1(yte_k, pred)})

    # 少样本：先按 val 选 α（跨 5 seed 平均），再在 test 上评估
    per_seed_correct = {}
    for k in KS:
        val_curve = {a: [] for a in ALPHAS}
        test_curve = {a: [] for a in ALPHAS}
        shot_acc, rand_acc = [], []
        for s in SEEDS:
            rng_s = np.random.default_rng(1000 * k + s)
            P_shot = shot_prototypes(Xtr_k, ytr_k, k, rng_s)
            for a in ALPHAS:
                Pm = mix_proto(P_shot, P_llm, a, metric)
                val_curve[a].append(float(accuracy_score(yva_k, predict(Xva_k, Pm, metric))))
                pred = predict(Xte_k, Pm, metric)
                test_curve[a].append(float(accuracy_score(yte_k, pred)))
            pred_shot = predict(Xte_k, P_shot, metric)
            shot_acc.append(float(accuracy_score(yte_k, pred_shot)))
            P_rand = l2norm(np.random.default_rng(2000 * k + s).standard_normal(P_llm.shape))
            pred_rand = predict(Xte_k, P_rand, metric)
            rand_acc.append(float(accuracy_score(yte_k, pred_rand)))
            per_seed_correct[(k, s)] = {
                "shot": (pred_shot == yte_k).astype(np.float64),
                "mix": {a: (predict(Xte_k, mix_proto(P_shot, P_llm, a, metric), metric) == yte_k)
                        .astype(np.float64) for a in ALPHAS},
            }

        val_mean = {a: float(np.mean(v)) for a, v in val_curve.items()}
        alpha_star = max(val_mean, key=val_mean.get)

        for a in ALPHAS:
            table.append({"k": k, "source": "P_mix", "alpha": a, "seed": None,
                          "acc": float(np.mean(test_curve[a])),
                          "macro_f1": None})
        table.append({"k": k, "source": "P_shot", "alpha": None, "seed": None,
                      "acc": float(np.mean(shot_acc)), "macro_f1": None})
        table.append({"k": k, "source": "P_rand", "alpha": None, "seed": None,
                      "acc": float(np.mean(rand_acc)), "macro_f1": None})
        table.append({"k": k, "source": "P_mix", "alpha": alpha_star, "seed": None,
                      "acc": float(np.mean(test_curve[alpha_star])), "macro_f1": None,
                      "selected_on_val": True})
        table[-1]["val_acc_mean"] = val_mean[alpha_star]
        table[-1]["val_alpha_curve"] = val_mean

        # 配对 bootstrap：Δ = acc(P_mix, α*) − acc(P_shot)，seed 平均后按 test 样本重采样
        diffs = np.mean([
            per_seed_correct[(k, s)]["mix"][alpha_star] - per_seed_correct[(k, s)]["shot"]
            for s in SEEDS
        ], axis=0)
        bs = np.array([float(diffs[rng.integers(0, len(diffs), len(diffs))].mean())
                       for _ in range(args.boot)])
        ci = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
        table[-1]["delta_vs_shot"] = float(diffs.mean())
        table[-1]["delta_bootstrap_ci95"] = ci

    # 门槛判读（k<=5 只看最小 k）
    eligible = [r for r in table if r["source"] == "P_mix" and r.get("selected_on_val")
                and r["k"] <= 5]
    e4 = {
        "metric_selected_on_val": metric,
        "seeds": SEEDS,
        "alphas": ALPHAS,
        "curve": table,
        "gate_delta_ge_0.03_at_k_le_5": None,
        "gate_note": "达标需 Δ>=3pp 且 bootstrap 95% CI 下界 > 0",
    }
    if eligible:
        r = min(eligible, key=lambda x: x["k"])
        e4["gate_check_k"] = r["k"]
        e4["gate_delta"] = r["delta_vs_shot"]
        e4["gate_ci95"] = r["delta_bootstrap_ci95"]
        e4["gate_delta_ge_0.03_at_k_le_5"] = bool(
            r["delta_vs_shot"] >= E4_DELTA_GATE and r["delta_bootstrap_ci95"][0] > 0)

    out = {
        "experiment": "E5 Qwen 空间：零样本 / 少样本语义原型" if args.text_npz
                      else "E3/E4 BERT 语义原型（零样本 / 少样本）",
        "split_dir": args.split_dir,
        "protos": args.protos,
        "text_source": args.text_npz or "split npz text_embeddings (BERT, 768)",
        "n_train_known": int(len(ytr_k)),
        "n_val_known": int(len(yva_k)),
        "n_test_known": int(len(yte_k)),
        "e3_zero_shot": e3,
        "e4_few_shot": e4,
        "thresholds": {"E3_acc_ge": E3_ACC_GATE, "E4_delta_ge": E4_DELTA_GATE},
    }

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "curve.csv"), "w", encoding="utf-8") as f:
        f.write("k,source,alpha,acc,macro_f1,delta_vs_shot,ci_low,ci_high\n")
        for r in table:
            f.write(f"{r['k']},{r['source']},{r['alpha']},{r['acc']:.6f},"
                    f"{'' if r['macro_f1'] is None else round(r['macro_f1'], 6)},"
                    f"{'' if 'delta_vs_shot' not in r else round(r['delta_vs_shot'], 6)},"
                    f"{'' if 'delta_bootstrap_ci95' not in r else round(r['delta_bootstrap_ci95'][0], 6)},"
                    f"{'' if 'delta_bootstrap_ci95' not in r else round(r['delta_bootstrap_ci95'][1], 6)}\n")

    print(json.dumps({
        "metric": metric,
        "E3_test_acc": e3["test_acc"],
        "E3_gate": e3["gate_acc_ge_0.50"],
        "E4_gate": e4["gate_delta_ge_0.03_at_k_le_5"],
        "E4_gate_k": e4.get("gate_check_k"),
        "E4_delta": e4.get("gate_delta"),
        "E4_ci": e4.get("gate_ci95"),
    }, ensure_ascii=False, indent=2))
    print(f"[OK] 结果 -> {args.out}/metrics.json")


if __name__ == "__main__":
    main()
