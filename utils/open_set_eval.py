"""
开集评估公共模块（口径唯一真相源）

唯一的 open-set 指标实现，run_ood_routing.py 与 test_model.py 都必须调用本函数。

口径定义（与 config/label_config.py 与报告 schema 一致，见 utils/caliber.py）：
- known_accuracy   = 在 label < K 样本上 argmax 预测 == 真实标签的比例（分类正确率，唯一定义）
- known_retention  = 在 label < K 样本上预测仍落在 0..K-1 的比例（= 1 - 误拒率）
- unknown_recall   = 在 label >= K 样本上预测 == K 的比例（含所有 unknown 来源的混合值）
- macro_f1_known   = 只在 label < K 样本、只在 0..K-1 标签集合上计算
- macro_f1_open    = 在 0..K（含 unknown）完整标签集合上计算，需显式标注
- unknown_origin   = {0: known, 1: genuine_ood, 2: simulated_in_distribution}；
                     旧划分在 data_loader 中统一填 -1（来源未知），此时
                     unknown_recall_genuine/simulated 返回 None，不得当作 0 召回
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from utils.caliber import CaliberError

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
                      label_names=None, include_routing_stats=True, origin=None):
    """
    开集评估（已知类 + unknown 统一评估）。

    Args:
        true_labels: 真实标签数组 (int，known=0..K-1，unknown>=K)
        pred_labels: 预测标签数组 (int)
        num_known_classes: 已知类数量 K
        label_names: 可选，标签名映射 dict {int: str}
        include_routing_stats: 是否输出路由统计字段（num_known_routed 等）
        origin: 可选，与 true_labels 等长的 unknown_origin 数组
                {0: known, 1: genuine_ood, 2: simulated_in_distribution}；
                -1 或 None 表示旧划分（来源未知）

    Returns:
        dict: 评估指标
    """
    if label_names is None:
        label_names = _DEFAULT_LABEL_NAMES

    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)

    # ── origin 校验与归一化：-1 = 来源未知，绝不能当作"真实 OOD 召回 = 0" ──
    origin_arr = None
    if origin is not None:
        origin_arr = np.asarray(origin, dtype=int)
        if origin_arr.size != true_labels.size:
            raise CaliberError(
                f"unknown_origin 长度 {origin_arr.size} != 标签长度 {true_labels.size}"
            )
        bad_codes = sorted(set(np.unique(origin_arr).tolist()) - {-1, 0, 1, 2})
        if bad_codes:
            raise CaliberError(f"unknown_origin 出现非法取值 {bad_codes}")
        mislabeled = (true_labels < num_known_classes) & np.isin(origin_arr, (1, 2))
        if mislabeled.any():
            raise CaliberError(
                f"{int(mislabeled.sum())} 条 known 样本被标记为 genuine/simulated unknown"
            )

    results = {}

    # ── 全样本混合指标（仅作次要指标，须标注 accuracy_open / macro_f1_open）──
    results['accuracy'] = accuracy_score(true_labels, pred_labels)  # 兼容旧字段
    results['accuracy_open'] = results['accuracy']
    # 兼容旧报告字段：全样本 macro 精确率/召回率/F1（含 unknown，语义同 macro_f1_open）
    results['precision'] = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
    results['recall'] = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
    results['macro_f1'] = f1_score(true_labels, pred_labels, average='macro', zero_division=0)  # 兼容旧字段
    results['f1'] = results['macro_f1']
    results['macro_f1_open'] = results['macro_f1']

    known_mask = true_labels < num_known_classes
    unknown_mask = true_labels >= num_known_classes

    known_classes_set = list(range(num_known_classes))

    # ── 已知类指标（拆 3 义）──
    if known_mask.sum() > 0:
        known_true = true_labels[known_mask]
        known_pred = pred_labels[known_mask]

        # known_accuracy: 分类正确率（唯一定义）
        results['known_accuracy'] = float(accuracy_score(known_true, known_pred))
        # known_retention: 被判为 known 的比例（= 1 - 误拒率），替代旧 known_accuracy
        results['known_retention'] = float(np.mean(known_pred < num_known_classes))

        # known_inner_accuracy: 排除被判 unknown 后，类内分类正确率（兼容保留）
        kept = known_pred < num_known_classes
        if kept.sum() > 0:
            results['known_inner_accuracy'] = float(
                accuracy_score(known_true[kept], known_pred[kept])
            )
        else:
            results['known_inner_accuracy'] = 0.0

        # macro_f1_known: 只在 0..K-1 标签集合上计算
        results['macro_f1_known'] = float(
            f1_score(known_true, known_pred, average='macro',
                     zero_division=0, labels=known_classes_set)
        )

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
        results['known_retention'] = 0.0
        results['known_inner_accuracy'] = 0.0
        results['macro_f1_known'] = 0.0
        for c in range(num_known_classes):
            results[f'class_{c}_recall'] = 0.0
            results[f'class_{c}_unknown_rate'] = 0.0

    # ── unknown 整体二分类指标 ──
    all_binary_true = (true_labels >= num_known_classes).astype(int)
    all_binary_pred = (pred_labels >= num_known_classes).astype(int)
    results['unknown_precision'] = precision_score(all_binary_true, all_binary_pred, zero_division=0)
    results['unknown_recall'] = recall_score(all_binary_true, all_binary_pred, zero_division=0)
    results['unknown_recall_mixed'] = results['unknown_recall']
    results['unknown_f1'] = f1_score(all_binary_true, all_binary_pred, zero_division=0)

    # known_leak_to_unknown: known 被判为 unknown 的 FP 率
    if known_mask.sum() > 0:
        results['known_leak_to_unknown'] = float(
            np.mean(pred_labels[known_mask] >= num_known_classes)
        )
    else:
        results['known_leak_to_unknown'] = 0.0

    # ── unknown 子集细分（来源分组）──
    unknown_origin = origin_arr[unknown_mask] if origin_arr is not None else None
    origin_usable = bool(
        unknown_origin is not None and np.isin(unknown_origin, (1, 2)).any()
    )

    if unknown_mask.sum() > 0:
        unknown_pred = pred_labels[unknown_mask]
        unknown_detected = unknown_pred >= num_known_classes
        results['unknown_leak_rate'] = float(np.mean(~unknown_detected))

        for c in range(num_known_classes):
            results[f'unknown_leak_to_{c}'] = float(
                np.mean(unknown_pred == c)
            )

        # 按 origin 分组报告 unknown recall（-1 表示来源未知，返回 None 而非 0.0）
        if origin_usable:
            for name, code in (("genuine", 1), ("simulated", 2)):
                sub = unknown_origin == code
                results[f'unknown_recall_{name}'] = (
                    float(np.mean(unknown_detected[sub])) if sub.any() else None
                )
        else:
            results['unknown_recall_genuine'] = None
            results['unknown_recall_simulated'] = None
    else:
        results['unknown_leak_rate'] = 0.0
        results['unknown_recall_genuine'] = None
        results['unknown_recall_simulated'] = None

    results['unknown_origin_available'] = origin_usable

    # ── 混淆矩阵 + 分类报告 ──
    all_labels = sorted(set(true_labels.tolist() + pred_labels.tolist()))
    cm = confusion_matrix(true_labels, pred_labels, labels=all_labels)
    results['confusion_matrix'] = cm.tolist()
    results['confusion_labels'] = [label_names.get(l, str(l)) for l in all_labels]

    target_names = [label_names.get(l, str(l)) for l in all_labels]
    results['classification_report'] = classification_report(
        true_labels, pred_labels, target_names=target_names,
        zero_division=0, output_dict=True
    )

    # ── 路由统计 ──
    if include_routing_stats:
        results['total'] = int(len(true_labels))
        results['num_known_routed'] = int(np.sum(pred_labels < num_known_classes))
        results['num_unknown_routed'] = int(np.sum(pred_labels >= num_known_classes))

    return results
