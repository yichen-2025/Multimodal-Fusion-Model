"""E8 模态信息审计：文本嵌入是否只是数值特征的重编码（CPU，分钟级）。

对应执行总纲 §5.1：
- E8.1 文本↔特征相互可预测性：79 -> 768 与 768 -> 79 的 Ridge R²；
- E8.2 条件互信息：I(text;y)、I(features;y)、I(text;y|features)。
  用多项逻辑回归的测试集交叉熵差做 plug-in 估计：
      I(text;y)          ≈ H_test(y) − CE(features 之外的 text 单独模型)
      I(text;y|features) ≈ CE(features) − CE(features + text)
  bootstrap 只对 test 样本重采样（不重训练），CI 偏窄，属保守估计。
- E8.3/E8.4 需扰动/模型产物，本脚本暂不实现（E8.4 直接读已有报告）。

用法：
  python scripts/audit_modality_info.py --split_dir split_data/dataset_200/split_openset_0 \
      --out results/2026-11-19/e8_audit --parts e81,e82
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

NUM_KNOWN = 10
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]


def load_split(split_dir, name):
    d = np.load(os.path.join(split_dir, f"{name}.npz"))
    return (
        d["scaled_features"].astype(np.float64),
        d["text_embeddings"].astype(np.float64),
        d["labels"].astype(np.int64),
    )


def r2_summary(Y, Yhat):
    per = r2_score(Y, Yhat, multioutput="raw_values")
    return {
        "r2_mean_per_dim": float(np.mean(per)),
        "r2_median_per_dim": float(np.median(per)),
        "r2_variance_weighted": float(r2_score(Y, Yhat, multioutput="variance_weighted")),
        "r2_min_dim": float(np.min(per)),
        "r2_max_dim": float(np.max(per)),
        "n_dims_above_0.8": int(np.sum(per > 0.8)),
        "n_dims": int(len(per)),
    }


def ridge_fit_eval(Xtr, Ytr, Xva, Yva, Xte, Yte):
    best = None
    for a in ALPHAS:
        m = Ridge(alpha=a).fit(Xtr, Ytr)
        v = r2_score(Yva, m.predict(Xva), multioutput="variance_weighted")
        if best is None or v > best[1]:
            best = (a, v, m)
    alpha, val_r2, model = best
    return {
        "alpha": alpha,
        "val_r2_variance_weighted": float(val_r2),
        "test": r2_summary(Yte, model.predict(Xte)),
        "train_r2_variance_weighted": float(
            r2_score(Ytr, model.predict(Xtr), multioutput="variance_weighted")),
    }


def ce_and_logp(proba, y):
    eps = 1e-12
    logp = np.log(np.clip(proba, eps, 1.0))
    per = -logp[np.arange(len(y)), y]
    return float(per.mean()), per


def entropy(labels, n_classes=NUM_KNOWN):
    p = np.bincount(labels, minlength=n_classes).astype(np.float64)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum())


def make_clf(kind, seed):
    """条件互信息的估计器：同一函数类必须在 features / features+text 两侧保持一致。"""
    if kind == "logreg":
        return LogisticRegression(max_iter=3000, C=1.0)
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=10, random_state=seed,
        )
    raise ValueError(f"未知估计器 {kind}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="e81,e82")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--estimator", default="logreg", choices=["logreg", "hgb"],
                    help="E8.2 的条件互信息估计器（两侧同一函数类）")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    x_tr, t_tr, y_tr = load_split(args.split_dir, "train")
    x_va, t_va, y_va = load_split(args.split_dir, "val")
    x_te, t_te, y_te = load_split(args.split_dir, "test")

    k_tr, k_va, k_te = y_tr < NUM_KNOWN, y_va < NUM_KNOWN, y_te < NUM_KNOWN
    x_tr, t_tr, y_tr = x_tr[k_tr], t_tr[k_tr], y_tr[k_tr]
    x_va, t_va, y_va = x_va[k_va], t_va[k_va], y_va[k_va]
    x_te, t_te, y_te = x_te[k_te], t_te[k_te], y_te[k_te]

    out = {
        "experiment": "E8 模态信息审计",
        "split_dir": args.split_dir,
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
        "n_test": int(len(y_te)),
        "dims": {"features": int(x_tr.shape[1]), "text": int(t_tr.shape[1])},
    }

    if "e81" in parts:
        print("[E8.1] Ridge 79 -> 768 ...", flush=True)
        f2t = ridge_fit_eval(x_tr, t_tr, x_va, t_va, x_te, t_te)
        print("[E8.1] Ridge 768 -> 79 ...", flush=True)
        t2f = ridge_fit_eval(t_tr, x_tr, t_va, x_va, t_te, x_te)
        out["e81_predictability"] = {
            "features_to_text": f2t,
            "text_to_features": t2f,
            "conclusion_rule": "79->768 的 test R² 越高 → 文本越冗余（H1 成立）",
        }

    if "e82" in parts:
        print("[E8.2] 拟合三项逻辑回归 ...", flush=True)
        sc_x = StandardScaler().fit(x_tr)
        sc_t = StandardScaler().fit(t_tr)
        Xtr, Xva, Xte = sc_x.transform(x_tr), sc_x.transform(x_va), sc_x.transform(x_te)
        Ttr, Tva, Tte = sc_t.transform(t_tr), sc_t.transform(t_va), sc_t.transform(t_te)

        # features-only
        m_x = make_clf(args.estimator, args.seed).fit(Xtr, y_tr)
        ce_x, per_x = ce_and_logp(m_x.predict_proba(Xte), y_te)
        acc_x = float(m_x.score(Xte, y_te))
        # text-only
        m_t = make_clf(args.estimator, args.seed).fit(Ttr, y_tr)
        ce_t, per_t = ce_and_logp(m_t.predict_proba(Tte), y_te)
        acc_t = float(m_t.score(Tte, y_te))
        # features + text
        XT_tr = np.hstack([Xtr, Ttr])
        XT_te = np.hstack([Xte, Tte])
        m_xt = make_clf(args.estimator, args.seed).fit(XT_tr, y_tr)
        ce_xt, per_xt = ce_and_logp(m_xt.predict_proba(XT_te), y_te)
        acc_xt = float(m_xt.score(XT_te, y_te))

        H = entropy(y_te)
        i_text = H - ce_t
        i_feat = H - ce_x
        i_cond = ce_x - ce_xt

        diff = per_x - per_xt  # 逐样本 CE 下降量
        boots = np.array([
            float(diff[rng.integers(0, len(diff), len(diff))].mean())
            for _ in range(args.boot)
        ])
        ci = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]

        out["e82_conditional_mi"] = {
            "entropy_y_test_nats": H,
            "ce_features": ce_x,
            "ce_text": ce_t,
            "ce_features_plus_text": ce_xt,
            "acc_features": acc_x,
            "acc_text": acc_t,
            "acc_features_plus_text": acc_xt,
            "I_text_y_nats": i_text,
            "I_features_y_nats": i_feat,
            "I_text_y_given_features_nats": i_cond,
            "I_text_y_given_features_bootstrap_ci95": ci,
            "boot": args.boot,
            "estimator_kind": args.estimator,
            "estimator": f"CE difference of {args.estimator} classifier (plug-in), "
                         "bootstrap resamples test rows only (no refit)",
            "conclusion_rule": "I(text;y|features) ≈ 0（CI 覆盖 0 且上界很小）→ 文本无独立信息",
        }
        print(f"[E8.2] I(text;y|features) = {i_cond:.4f} nats, CI95 {ci}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[OK] 结果已写入 {path}")


if __name__ == "__main__":
    main()
