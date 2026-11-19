"""E0 轻量基线：79 维数值特征 -> MLP / XGBoost（CPU，分钟级）。

目的：为「LLM 值不值 60x 成本」提供锚点。只做闭集训练 + max-softmax 开集分数，
不引入 LLM、不加载 torch、不改动任何现有训练/路由代码。

口径（见 docs/claims.md §5.1 / C-28）：
- 训练集：train.npz 全部已知类样本（10 类均衡）；
- 闭集评估：test 中 label < 10 的样本，报 accuracy 与 macro-F1；
- 开集分数：1 - max softmax，报 auroc_genuine（known vs 真实未知 origin=1）与
  tpr@fpr=1%（阈值取 known 分数 99 分位）；另报模拟未知（origin=2）的召回。

用法：
  python scripts/eval_light_baseline.py \
      --split_dir split_data/dataset_200/split_openset_0 \
      --model mlp --out results/2026-11-19/e0_baseline
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

NUM_KNOWN = 10


def load_split(split_dir, name):
    d = np.load(os.path.join(split_dir, f"{name}.npz"))
    return (
        d["scaled_features"].astype(np.float32),
        d["labels"].astype(np.int64),
        d["unknown_origin"].astype(np.int64),
    )


def build_model(name, seed):
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=400,
            random_state=seed,
            verbose=False,
        )
    if name == "xgb":
        from xgboost import XGBClassifier  # 可选依赖，按需安装

        return XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            n_jobs=16,
            random_state=seed,
            eval_metric="mlogloss",
        )
    raise ValueError(f"未知模型 {name}")


def tpr_at_fpr(known_scores, unknown_scores, fpr=0.01):
    """known_scores / unknown_scores 均为「越大越像未知」的分数。"""
    thr = float(np.quantile(known_scores, 1.0 - fpr))
    return float(np.mean(unknown_scores > thr)), thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--model", default="mlp", choices=["mlp", "xgb"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    x_tr, y_tr, _ = load_split(args.split_dir, "train")
    x_va, y_va, _ = load_split(args.split_dir, "val")
    x_te, y_te, o_te = load_split(args.split_dir, "test")

    train_known = y_tr < NUM_KNOWN
    val_known = y_va < NUM_KNOWN
    test_known = y_te < NUM_KNOWN

    clf = build_model(args.model, args.seed)
    t0 = time.time()
    clf.fit(x_tr[train_known], y_tr[train_known])
    train_sec = time.time() - t0

    # ── 闭集 ──
    t1 = time.time()
    proba_te = clf.predict_proba(x_te)
    infer_sec = time.time() - t1
    pred_te = proba_te.argmax(axis=1)

    closed = {
        "n_test_known": int(test_known.sum()),
        "accuracy": float(accuracy_score(y_te[test_known], pred_te[test_known])),
        "macro_f1": float(f1_score(y_te[test_known], pred_te[test_known],
                                   average="macro", zero_division=0,
                                   labels=list(range(NUM_KNOWN)))),
        "val_accuracy": float(accuracy_score(y_va[val_known],
                                             clf.predict(x_va[val_known]))),
    }

    # ── 开集：1 - max softmax，越大越像未知 ──
    ood_score = 1.0 - proba_te.max(axis=1)
    known_mask = test_known
    gen_mask = (y_te >= NUM_KNOWN) & (o_te == 1)
    sim_mask = (y_te >= NUM_KNOWN) & (o_te == 2)

    ood = {"n_genuine": int(gen_mask.sum()), "n_simulated": int(sim_mask.sum())}
    if gen_mask.sum() > 0:
        ood["auroc_genuine"] = float(
            roc_auc_score(np.r_[np.zeros(known_mask.sum()), np.ones(gen_mask.sum())],
                          np.r_[ood_score[known_mask], ood_score[gen_mask]]))
        tpr, thr = tpr_at_fpr(ood_score[known_mask], ood_score[gen_mask])
        ood["tpr_at_fpr1_genuine"] = tpr
        ood["threshold_fpr1"] = thr
    if sim_mask.sum() > 0:
        tpr_s, _ = tpr_at_fpr(ood_score[known_mask], ood_score[sim_mask])
        ood["tpr_at_fpr1_simulated"] = tpr_s
        ood["unknown_recall_simulated"] = tpr_s

    out = {
        "experiment": "E0 轻量基线",
        "model": args.model,
        "seed": args.seed,
        "split_dir": args.split_dir,
        "n_train": int(train_known.sum()),
        "n_features": int(x_tr.shape[1]),
        "closed_set": closed,
        "ood_max_softmax": ood,
        "timing": {
            "train_seconds": round(train_sec, 2),
            "infer_ms_per_1k": round(infer_sec / max(len(y_te), 1) * 1000 * 1000, 2),
        },
    }

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[OK] 结果已写入 {os.path.join(args.out, 'metrics.json')}")


if __name__ == "__main__":
    main()
