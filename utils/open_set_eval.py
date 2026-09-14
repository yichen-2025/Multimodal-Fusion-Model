"""
开集评估公共模块（B8 修复）

把 run_ood_routing.py 和 test_model.py 中重复的 evaluate_open_set 逻辑
统一到这里，两个脚本都 import 本函数，确保指标口径只有一份真相源。

口径：全样本二分类（unknown 当正类），同时报告 known 类被误报为 unknown 的 FP 率。
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# 标签名 fallback：优先从 config 导入，避免硬编码两份
try:
    from config.label_config import MERGED_LABEL_NAMES as _DEFAULT_LABEL_NAMES
except ImportError:
    _DEFAULT_LABEL_NAMES = {
        0: "BENIGN", 1: "DoS Hulk", 2: "DoS GoldenEye", 3: "DoS slowloris",
        4: "DoS Slowhttptest", 5: "DDoS", 6: "PortScan", 7: "FTP-Patator",
        8: "SSH-Patator", 9: "Bot", 10: "Web Attack", 11: "Other Attack",
    }


def evaluate_open_set(true_labels, pred_labels, num_known_classes,
                      label_names=None, include_routing_stats=True):
    """
    开集评估（三分类已知类 + unknown 统一评估）

    Args:
        true_labels: 真实标签数组 (int，known=0..K-1，unknown>=K)
        pred_labels: 预测标签数组 (int)
        num_known_classes: 已知类数量 K
        label_names: 可选，标签名映射 dict {int: str}，None 则用 MERGED_LABEL_NAMES
        include_routing_stats: 是否输出路由统计字段（num_known_routed 等）

    Returns:
        dict: 评估指标
    """
    if label_names is None:
        label_names = _DEFAULT_LABEL_NAMES

    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)

    results = {}

    # ── 三分类整体指标 ──
    results['accuracy'] = accuracy_score(true_labels, pred_labels)
    results['macro_f1'] = f1_score(true_labels, pred_labels, average='macro', zero_division=0)

    known_mask = true_labels < num_known_classes
    unknown_mask = true_labels >= num_known_classes

    # ── 已知类细分指标 ──
    if known_mask.sum() > 0:
        known_true = true_labels[known_mask]
        known_pred = pred_labels[known_mask]

        # known_accuracy: known 样本被判为 known（不要求类正确）
        known_correct = known_pred < num_known_classes
        results['known_accuracy'] = float(np.mean(known_correct))

        # known_inner_accuracy: 排除被判 unknown 后，类内分类正确率
        if known_correct.sum() > 0:
            results['known_inner_accuracy'] = float(
                accuracy_score(known_true[known_correct], known_pred[known_correct])
            )
        else:
            results['known_inner_accuracy'] = 0.0

        # 每类 known 的 recall 和 unknown_rate
        for c in range(num_known_classes):
            class_mask = known_true == c
            if class_mask.sum() > 0:
                results[f'class_{c}_recall'] = float(
                    np.mean(known_pred[class_mask] == c)
                )
                results[f'class_{c}_unknown_rate'] = float(
                    np.mean(known_pred[class_mask] >= num_known_classes)
                )
            else:
                results[f'class_{c}_recall'] = 0.0
                results[f'class_{c}_unknown_rate'] = 0.0
    else:
        results['known_accuracy'] = 0.0
        results['known_inner_accuracy'] = 0.0

    # ── unknown 全样本二分类指标（含 FP，B4 修复口径）──
    all_binary_true = (true_labels >= num_known_classes).astype(int)
    all_binary_pred = (pred_labels >= num_known_classes).astype(int)
    results['unknown_precision'] = precision_score(all_binary_true, all_binary_pred, zero_division=0)
    results['unknown_recall'] = recall_score(all_binary_true, all_binary_pred, zero_division=0)
    results['unknown_f1'] = f1_score(all_binary_true, all_binary_pred, zero_division=0)

    # known_leak_to_unknown: known 被判为 unknown 的 FP 率
    if known_mask.sum() > 0:
        results['known_leak_to_unknown'] = float(
            np.mean(pred_labels[known_mask] >= num_known_classes)
        )
    else:
        results['known_leak_to_unknown'] = 0.0

    # ── unknown 子集细分指标 ──
    if unknown_mask.sum() > 0:
        unknown_pred = pred_labels[unknown_mask]
        unknown_detected = unknown_pred >= num_known_classes
        results['unknown_leak_rate'] = float(np.mean(~unknown_detected))

        # 未知样本被错误分类到各已知类的比例
        for c in range(num_known_classes):
            results[f'unknown_leak_to_{c}'] = float(
                np.mean(unknown_pred == c)
            )
    else:
        results['unknown_leak_rate'] = 0.0

    # ── 混淆矩阵 + 分类报告 ──
    all_labels = sorted(set(true_labels.tolist() + pred_labels.tolist()))
    cm = confusion_matrix(true_labels, pred_labels, labels=all_labels)
    results['confusion_matrix'] = cm.tolist()
    results['confusion_labels'] = [label_names.get(l, str(l)) for l in all_labels]

    target_names = [label_names.get(l, str(l))
                    for l in sorted(set(true_labels.tolist() + pred_labels.tolist()))]
    results['classification_report'] = classification_report(
        true_labels, pred_labels, target_names=target_names,
        zero_division=0, output_dict=True
    )

    # ── 路由统计（run_ood_routing 用，test_model 不用时传 False）──
    if include_routing_stats:
        results['total'] = int(len(true_labels))
        results['num_known_routed'] = int(np.sum(pred_labels < num_known_classes))
        results['num_unknown_routed'] = int(np.sum(pred_labels >= num_known_classes))

    return results
