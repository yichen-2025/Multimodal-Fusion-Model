"""E8 模态信息审计：文本嵌入是否只是数值特征的重编码（CPU，分钟级）。

对应执行总纲 §5.1：
- E8.1 文本↔特征相互可预测性：79 -> 768 与 768 -> 79 的 Ridge R²；
- E8.2 条件互信息：I(text;y)、I(features;y)、I(text;y|features)。
  用多项逻辑回归的测试集交叉熵差做 plug-in 估计：
      I(text;y)          ≈ H_test(y) − CE(features 之外的 text 单独模型)
      I(text;y|features) ≈ CE(features) − CE(features + text)
  bootstrap 只对 test 样本重采样（不重训练），CI 偏窄，属保守估计。
- E8.4 去文本消融：读 `ood_reports/ood_routing_*.json`，按「同 seed」配对
  `A0*`（数值+LLM 文本，full）与 `A0*_no_text`（纯数值），比较已知类闭集准确率。
  E8.3 需扰动，本脚本不实现。

用法：
  python scripts/audit_modality_info.py --split_dir split_data/dataset_200/split_openset_0 \
      --out results/2026-11-19/e8_audit --parts e81,e82
  python scripts/audit_modality_info.py --split_dir split_data/dataset_200/split_openset_0 \
      --out results/2026-11-20/e84_text_ablation --parts e84
"""
import argparse
import glob
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


# ---------------------------------------------------------------- E8.4 helpers

def parse_model_configs(root):
    """扫描 saved_models/ 与 _archive/saved_models/，返回 model_id -> {variant, seed}。"""
    out = {}
    for base in ("saved_models", "_archive/saved_models"):
        for p in glob.glob(os.path.join(root, base, "model_*", "config.txt")):
            d = {}
            try:
                with open(p, encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if ":" in line:
                            k, v = line.split(":", 1)
                            d[k.strip()] = v.strip()
            except OSError:
                continue
            if "model_id" in d and "variant" in d:
                key = int(d["model_id"])
                out.setdefault(key, {"variant": d["variant"], "seed": d.get("seed"),
                                     "config": p})
    return out


def latest_routing_reports(root):
    """每个 backbone_model_id 取 timestamp 最新的一份路由报告。"""
    best = {}
    for p in glob.glob(os.path.join(root, "ood_reports", "ood_routing_*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        mid = d.get("backbone_model_id")
        if mid is None:
            continue
        ts = str(d.get("timestamp", ""))
        if mid not in best or ts > best[mid]["timestamp"]:
            best[mid] = {"timestamp": ts, "path": os.path.relpath(p, root), "report": d}
    return best


def known_closed_set_metrics(report):
    """闭集已知类指标：全部已知样本参与，路由到 unknown（pred=10）计为错。"""
    lab = np.asarray(report["all_labels"], dtype=np.int64)
    pred = np.asarray(report["all_routed_preds"], dtype=np.int64)
    known = lab != NUM_KNOWN
    correct = (pred[known] == lab[known])
    # 已知子集上的 macro-F1（pred=10 计为漏检，不产生已知类 FP）
    f1s = []
    for c in range(NUM_KNOWN):
        tp = int(np.sum((lab[known] == c) & (pred[known] == c)))
        fn = int(np.sum((lab[known] == c) & (pred[known] != c)))
        fp = int(np.sum((lab[known] != c) & (pred[known] == c)))
        if tp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    routed = report.get("routed") or {}
    return {
        "n_known": int(known.sum()),
        "acc_all_known": float(correct.mean()),
        "macro_f1_known": float(np.mean(f1s)) if f1s else None,
        "conditional_known_accuracy": routed.get("known_accuracy"),
        "unknown_recall": routed.get("unknown_recall"),
        "correct_mask": correct,
        "known_mask": known,
        "labels": lab,
    }


def e84_text_ablation(args, rng):
    """A0*（数值+LLM 文本）vs A0*_no_text（纯数值），同 seed 配对。"""
    cfgs = parse_model_configs(PROJECT_ROOT)
    reports = latest_routing_reports(PROJECT_ROOT)

    full_by_seed, notext_by_seed = {}, {}
    for mid, info in cfgs.items():
        if mid not in reports:
            continue
        v, seed = info["variant"], info["seed"]
        if v == "A0*":
            full_by_seed.setdefault(seed, []).append(mid)
        elif v == "A0*_no_text":
            notext_by_seed.setdefault(seed, []).append(mid)

    pairs = []
    for seed in sorted(set(full_by_seed) & set(notext_by_seed), key=lambda s: str(s)):
        # 每个 seed 只取两侧 timestamp 最新的一份报告，避免同一报告被重复配对
        fmid = max(full_by_seed[seed], key=lambda m: reports[m]["timestamp"])
        nmid = max(notext_by_seed[seed], key=lambda m: reports[m]["timestamp"])
        fr, nr = reports[fmid]["report"], reports[nmid]["report"]
        if len(fr["all_labels"]) != len(nr["all_labels"]):
            continue
        if not np.array_equal(np.asarray(fr["all_labels"]), np.asarray(nr["all_labels"])):
            continue
        mf = known_closed_set_metrics(fr)
        mn = known_closed_set_metrics(nr)
        pairs.append({
                    "seed": seed,
                    "full_model_id": fmid,
                    "notext_model_id": nmid,
                    "full_report": reports[fmid]["path"],
                    "notext_report": reports[nmid]["path"],
                    "n_known": mf["n_known"],
                    "acc_full": mf["acc_all_known"],
                    "acc_no_text": mn["acc_all_known"],
                    "delta_full_minus_notext": mf["acc_all_known"] - mn["acc_all_known"],
                    "macro_f1_full": mf["macro_f1_known"],
                    "macro_f1_no_text": mn["macro_f1_known"],
                    "conditional_known_accuracy_full": mf["conditional_known_accuracy"],
                    "conditional_known_accuracy_no_text": mn["conditional_known_accuracy"],
                    "_correct_full": mf["correct_mask"],
                    "_correct_notext": mn["correct_mask"],
                    "_known_mask": mf["known_mask"],
                })

    if not pairs:
        return {"status": "unavailable",
                "reason": "未找到同 seed 且同 test 行序的 A0* / A0*_no_text 路由报告对"}

    deltas = np.array([p["delta_full_minus_notext"] for p in pairs], dtype=np.float64)
    boot = []
    for _ in range(args.boot):
        per_seed = []
        for p in pairs:
            idx = np.where(p["_known_mask"])[0]
            take = rng.integers(0, len(idx), len(idx))
            cf = p["_correct_full"][idx][take]
            cn = p["_correct_notext"][idx][take]
            per_seed.append(cf.mean() - cn.mean())
        boot.append(float(np.mean(per_seed)))
    ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]

    table = []
    for p in pairs:
        row = {k: v for k, v in p.items() if not k.startswith("_")}
        table.append(row)

    return {
        "status": "ok",
        "comparison": "A0*（数值+LLM文本，full） vs A0*_no_text（纯数值）",
        "metric": "test 已知类（1655 条）闭集准确率；路由到 unknown 计为错",
        "pairs": table,
        "n_pairs": len(pairs),
        "delta_mean": float(deltas.mean()),
        "delta_sd_across_pairs": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
        "delta_bootstrap_ci95": ci,
        "delta_positive_means": "full（含文本）更高",
        "boot": args.boot,
        "conclusion_rule": "Δ≈0 且 CI 覆盖 0 → 去文本不掉性能（文本可省）；Δ 显著为正 → 文本仍有贡献",
        "caveat": ("A0*/A0*_no_text 检查点来自 2026-09/10 旧批次；纯数值臂的闭集准确率"
                   "显著低于同一 split 上的 E0 轻量 MLP（0.9855），疑为旧检查点/欠训练。"
                   "因此 Δ 只作描述，不作为「文本可省」的判定依据；判定以 E8.2 中"
                   "同估计器的 acc_features vs acc_features_plus_text 为准。"),
    }


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

    out = {"experiment": "E8 模态信息审计", "split_dir": args.split_dir}

    if "e81" in parts or "e82" in parts:
        x_tr, t_tr, y_tr = load_split(args.split_dir, "train")
        x_va, t_va, y_va = load_split(args.split_dir, "val")
        x_te, t_te, y_te = load_split(args.split_dir, "test")

        k_tr, k_va, k_te = y_tr < NUM_KNOWN, y_va < NUM_KNOWN, y_te < NUM_KNOWN
        x_tr, t_tr, y_tr = x_tr[k_tr], t_tr[k_tr], y_tr[k_tr]
        x_va, t_va, y_va = x_va[k_va], t_va[k_va], y_va[k_va]
        x_te, t_te, y_te = x_te[k_te], t_te[k_te], y_te[k_te]

        out.update({
            "n_train": int(len(y_tr)),
            "n_val": int(len(y_va)),
            "n_test": int(len(y_te)),
            "dims": {"features": int(x_tr.shape[1]), "text": int(t_tr.shape[1])},
        })

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

            m_x = make_clf(args.estimator, args.seed).fit(Xtr, y_tr)
            ce_x, per_x = ce_and_logp(m_x.predict_proba(Xte), y_te)
            acc_x = float(m_x.score(Xte, y_te))
            m_t = make_clf(args.estimator, args.seed).fit(Ttr, y_tr)
            ce_t, per_t = ce_and_logp(m_t.predict_proba(Tte), y_te)
            acc_t = float(m_t.score(Tte, y_te))
            XT_tr = np.hstack([Xtr, Ttr])
            XT_te = np.hstack([Xte, Tte])
            m_xt = make_clf(args.estimator, args.seed).fit(XT_tr, y_tr)
            ce_xt, per_xt = ce_and_logp(m_xt.predict_proba(XT_te), y_te)
            acc_xt = float(m_xt.score(XT_te, y_te))

            H = entropy(y_te)
            i_text = H - ce_t
            i_feat = H - ce_x
            i_cond = ce_x - ce_xt

            diff = per_x - per_xt
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

    if "e84" in parts:
        print("[E8.4] 读取 ood_reports 并做同 seed 配对 ...", flush=True)
        out["e84_text_ablation"] = e84_text_ablation(args, rng)
        if out["e84_text_ablation"].get("status") == "ok":
            print(f"[E8.4] Δacc = {out['e84_text_ablation']['delta_mean']:+.4f}, "
                  f"CI95 {out['e84_text_ablation']['delta_bootstrap_ci95']}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[OK] 结果已写入 {path}")


if __name__ == "__main__":
    main()
