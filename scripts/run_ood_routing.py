import torch
import torch.nn.functional as F
import sys
import os
import time
import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.model_architectures.multi_modal_model import MultiModalFusionModel
from src.model_architectures.ood_head import OODHead
from src.data.data_loader import load_split_data, collate_fn
from utils.log_utils import save_log


LABEL_NAMES = {
    0: "BENIGN",
    1: "DoS Hulk",
    2: "DoS GoldenEye",
    3: "DoS slowloris",
    4: "DoS Slowhttptest",
    5: "DDoS",
    6: "PortScan",
    7: "FTP-Patator",
    8: "SSH-Patator",
    9: "Bot",
    10: "Web Attack - Brute Force",
    11: "Web Attack - XSS",
    12: "Web Attack - Sql Injection",
    13: "Infiltration",
    14: "Heartbleed",
    15: "unknown",
}


def load_ood_head(ood_id):
    """加载已训练的OOD头"""
    ood_path = os.path.join(PROJECT_ROOT, "saved_ood_heads", f"ood_head_{ood_id}")
    config_path = os.path.join(ood_path, "config.json")
    ood_file = os.path.join(ood_path, "ood_head.pt")

    if not os.path.exists(ood_file):
        raise FileNotFoundError(f"OOD头文件不存在: {ood_file}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    ood_head = OODHead(
        feature_dim=config['feature_dim'],
        num_known_classes=config['num_known_classes'],
        distance_type=config['distance_type'],
        temperature=config.get('temperature', 1.0)
    )
    ood_head.load(ood_file)
    return ood_head, config


def evaluate_open_set(true_labels, pred_labels, num_known_classes=2):
    """
    开集三分类评估

    Args:
        true_labels: 真实标签 (0=benign, 1=known DDoS, 2=unknown DDoS)
        pred_labels: 预测标签 (0, 1, 2)
        num_known_classes: 已知类数量

    Returns:
        dict: 评估指标
    """
    results = {}

    # 整体准确率
    results['accuracy'] = accuracy_score(true_labels, pred_labels)

    # 宏平均F1（三分类）
    results['macro_f1'] = f1_score(true_labels, pred_labels, average='macro', zero_division=0)

    # 已知类指标
    known_mask = true_labels < num_known_classes
    unknown_mask = true_labels >= num_known_classes

    # 已知类：二分类（benign vs known_DDoS）
    if known_mask.sum() > 0:
        known_true = true_labels[known_mask]
        known_pred = pred_labels[known_mask]

        # 已知类内部：将误判为unknown的视为错误
        known_correct = known_pred < num_known_classes
        results['known_accuracy'] = np.mean(known_correct)

        # 已知类内分类正确（不只是被判为known）
        results['known_inner_accuracy'] = accuracy_score(known_true[known_correct], known_pred[known_correct]) if known_correct.sum() > 0 else 0.0

        # 每类已知的召回率
        for c in range(num_known_classes):
            class_mask = known_true == c
            if class_mask.sum() > 0:
                class_pred_correct = known_pred[class_mask] == c
                class_pred_as_unknown = known_pred[class_mask] >= num_known_classes
                results[f'class_{c}_recall'] = np.mean(class_pred_correct)
                results[f'class_{c}_unknown_rate'] = np.mean(class_pred_as_unknown)
            else:
                results[f'class_{c}_recall'] = 0.0
                results[f'class_{c}_unknown_rate'] = 0.0

    # 未知类指标
    if unknown_mask.sum() > 0:
        unknown_true = true_labels[unknown_mask]
        unknown_pred = pred_labels[unknown_mask]

        # 未知类检测率（被判为unknown的比例）
        unknown_detected = unknown_pred >= num_known_classes
        results['unknown_recall'] = np.mean(unknown_detected)

        # 未知类被判为known的比例（= 泄漏率）
        results['unknown_leak_rate'] = np.mean(~unknown_detected)

        # 未知类F1（将unknown视为一个正类）
        binary_true = (unknown_true >= num_known_classes).astype(int)
        binary_pred = (unknown_pred >= num_known_classes).astype(int)
        results['unknown_f1'] = f1_score(binary_true, binary_pred, zero_division=0)

        # 未知样本被错误分类为各已知类的比例
        for c in range(num_known_classes):
            leak_to_c = unknown_pred == c
            results[f'unknown_leak_to_{c}'] = np.mean(leak_to_c)
    else:
        results['unknown_recall'] = 0.0
        results['unknown_leak_rate'] = 0.0
        results['unknown_f1'] = 0.0

    # 三分类混淆矩阵
    all_labels = sorted(set(true_labels.tolist() + pred_labels.tolist()))
    cm = confusion_matrix(true_labels, pred_labels, labels=all_labels)
    results['confusion_matrix'] = cm.tolist()
    results['confusion_labels'] = [LABEL_NAMES.get(l, str(l)) for l in all_labels]

    # 分类报告
    target_names = [LABEL_NAMES.get(l, str(l)) for l in sorted(set(true_labels.tolist() + pred_labels.tolist()))]
    results['classification_report'] = classification_report(
        true_labels, pred_labels, target_names=target_names,
        zero_division=0, output_dict=True
    )

    # 路由统计
    results['total'] = len(true_labels)
    results['num_known_routed'] = int(np.sum(pred_labels < num_known_classes))
    results['num_unknown_routed'] = int(np.sum(pred_labels >= num_known_classes))

    return results


def run_ood_routing(
    backbone_model_id=None,
    ood_id=None,
    llm_model_id=None,
    dataset_id=1,
    split_id=0,
    model_path=None,
    backbone_variant="A3",
    llm_variant="A0",
    batch_size=32,
    verbose=True,
    compare_baseline=True,
    fewshot_k=None
):
    """
    运行OOD路由评估

    流程：
    1. 加载A3 backbone模型（无LLM，快速）
    2. 加载A0 LLM模型（用于未知样本语义推理）
    3. 加载已训练的OOD头
    4. 对测试集执行：
       a. 提取融合特征 → OOD判定
       b. 已知样本 → A3分类器
       c. 未知样本 → A0 (LLM) 语义推理
    5. 对比：A3-only (闭集基线) vs OOD-routed (开集路由)
    6. 报告开集指标

    Args:
        backbone_model_id (int): A3 backbone模型ID
        ood_id (int): 已训练OOD头ID
        llm_model_id (int): A0 LLM模型ID（可为None，用backbone做路由）
        dataset_id (int): 数据集ID
        split_id (int): 划分ID
        model_path (str): LLM模型路径
        backbone_variant (str): backbone变体（默认A3）
        llm_variant (str): LLM变体（默认A0）
        batch_size (int): batch大小
        verbose (bool): 打印详细信息
        compare_baseline (bool): 是否对比A3-only基线

    Returns:
        dict: 完整评估结果
    """
    start_time = time.time()

    if model_path is None:
        model_path = os.path.join(PROJECT_ROOT, "models", "qwen2.5-1.5b")

    # ========== 1. 加载A3 Backbone ==========
    if verbose:
        print("=" * 60)
        print("加载A3 Backbone模型...")
        print("=" * 60)

    if backbone_model_id is None:
        raise ValueError("必须指定backbone_model_id")

    backbone_path = os.path.join(PROJECT_ROOT, "saved_models", f"model_{backbone_model_id}")
    if not os.path.exists(backbone_path):
        raise FileNotFoundError(f"Backbone模型不存在: {backbone_path}")

    backbone = MultiModalFusionModel.from_pretrained(model_path, backbone_path)
    backbone.eval()
    backbone_device = backbone.device

    if verbose:
        print(f"  - Backbone: model_{backbone_model_id} ({backbone_variant})")
        print(f"  - 设备: {backbone_device}")
        print(f"  - 融合特征维度: {backbone.get_feature_dim()}")

    # 冻结backbone
    for param in backbone.parameters():
        param.requires_grad = False

    # ========== 2. 加载A0 LLM模型 ==========
    llm_model = None
    llm_tokenizer = None

    if llm_model_id is not None:
        if verbose:
            print("\n" + "-" * 40)
            print("加载A0 LLM模型...")

        llm_path = os.path.join(PROJECT_ROOT, "saved_models", f"model_{llm_model_id}")
        if not os.path.exists(llm_path):
            raise FileNotFoundError(f"LLM模型不存在: {llm_path}")

        llm_model = MultiModalFusionModel.from_pretrained(model_path, llm_path)
        llm_model.eval()
        llm_tokenizer = llm_model.get_tokenizer()

        if verbose:
            print(f"  - LLM: model_{llm_model_id} ({llm_variant})")
            print(f"  - 设备: {llm_model.device}")
    else:
        if verbose:
            print("\n" + "-" * 40)
            print("跳过LLM模型加载（llm_model_id=None）")
            print("未知样本将被标记为unknown但不进行LLM推理")

    # ========== 3. 加载OOD头 ==========
    if verbose:
        print("\n" + "-" * 40)
        print("加载OOD检测头...")

    ood_head, ood_config = load_ood_head(ood_id)
    ood_head.eval()
    ood_head.to(backbone_device)

    if verbose:
        print(f"  - OOD头ID: {ood_id}")
        print(f"  - 距离度量: {ood_config['distance_type']}")
        print(f"  - 阈值: {ood_config['ood_threshold']:.4f}")
        print(f"  - 已知类数: {ood_config['num_known_classes']}")

    num_known_classes = ood_config['num_known_classes']

    # ========== 4. 加载测试集 ==========
    if verbose:
        print("\n" + "-" * 40)
        if fewshot_k is not None:
            print(f"加载少样本测试数据 (k={fewshot_k})...")
        else:
            print("加载开集测试数据...")

    test_dataset = load_split_data(
        data_dir=os.path.join(PROJECT_ROOT, "split_data"),
        data_type="test", dataset_id=dataset_id, split_id=split_id,
        openset=(fewshot_k is None),
        fewshot_k=fewshot_k
    )
    if test_dataset is None:
        split_name = f"split_fewshot_{fewshot_k}_{split_id}" if fewshot_k is not None else f"split_openset_{split_id}"
        raise FileNotFoundError(
            f"未找到测试数据: split_data/dataset_{dataset_id}/{split_name}/test.npz"
        )

    all_labels = np.array([s['label'] for s in test_dataset])
    label_counts = {int(k): int(v) for k, v in zip(*np.unique(all_labels, return_counts=True))}

    if verbose:
        print(f"  - 测试样本数: {len(test_dataset)}")
        for lbl, cnt in label_counts.items():
            print(f"    {LABEL_NAMES.get(lbl, str(lbl))}({lbl}): {cnt}")

    # ========== 5. 构造DataLoader ==========
    backbone_configs = {
        "A0": {"use_numeric": True,  "use_bert": True,  "use_llm": True},
        "A1": {"use_numeric": True,  "use_bert": False, "use_llm": True},
        "A2": {"use_numeric": False, "use_bert": True,  "use_llm": True},
        "A3": {"use_numeric": True,  "use_bert": True,  "use_llm": False},
    }
    backbone_cfg = backbone_configs.get(backbone_variant, backbone_configs["A3"])

    dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer=None,
                                         use_numeric=backbone_cfg["use_numeric"],
                                         use_bert=backbone_cfg["use_bert"],
                                         use_llm=False)
    )

    # ========== 6. OOD路由推理 ==========
    if verbose:
        print("\n" + "-" * 40)
        print("OOD路由推理...")
        print("=" * 60)

    all_a3_preds = []       # A3-only闭集预测
    all_routed_preds = []  # OOD路由预测
    all_ood_scores = []
    all_unknown_flags = []
    all_labels_collected = []
    routing_log = []

    # LLM提示模板（多分类版本）
    llm_prompt_template = (
        "根据以下网络流量统计特征判断该流量类型。\n"
        "统计特征: 持续时间={duration}, 包头长度={header_len}, 包长度={pkt_len}, "
        "速率={rate}, 方向={direction}, 标志位={flags}\n"
        "文本描述: {text}\n"
        "请判断流量类型: BENIGN(正常)、DoS Hulk、DoS GoldenEye、DoS slowloris、"
        "DoS Slowhttptest、DDoS、PortScan、FTP-Patator、SSH-Patator、Bot、"
        "Web Attack(Brute Force/XSS/Sql Injection)、Infiltration、Heartbleed。"
    )

    for batch_idx, batch in enumerate(dataloader):
        stat = batch["stat_tensor"]
        bert = batch["bert_tensor"]
        labels = batch["labels"]

        # 提取融合特征
        with torch.no_grad():
            features = backbone.extract_fusion_features(stat, bert)

        # OOD判定
        with torch.no_grad():
            ood_results = ood_head(features)
            ood_scores = ood_results['scores']
            unknown_mask = ood_results['unknown_mask']

        # A3闭集预测（所有样本）
        with torch.no_grad():
            outputs = backbone(stat, bert)
            a3_logits = outputs["logits"]
            a3_pred = torch.argmax(a3_logits, dim=1)

        # 路由预测
        routed_pred = a3_pred.clone()

        # 处理未知样本 → 路由到LLM或标记为unknown
        unknown_indices = torch.where(unknown_mask)[0]
        if len(unknown_indices) > 0:
            if llm_model is not None:
                # 收集未知样本的文本描述
                unknown_texts = []
                for idx in unknown_indices:
                    sample = test_dataset[idx.item()]
                    text = sample.get('text', '')
                    if not text:
                        text = "网络流量特征异常"
                    unknown_texts.append(text)

                # 批量LLM推理
                for i, (idx, text) in enumerate(zip(unknown_indices, unknown_texts)):
                    sample_idx = idx.item()
                    text = test_dataset[sample_idx].get('text', '') or "网络流量特征异常"

                    # 构造prompt（多分类版本）
                    prompt = f"判断以下网络流量类型。\n流量描述：{text}\n请选择：BENIGN、DoS Hulk、DoS GoldenEye、DoS slowloris、DoS Slowhttptest、DDoS、PortScan、FTP-Patator、SSH-Patator、Bot、Web Attack(Brute Force/XSS/Sql Injection)、Infiltration、Heartbleed。直接输出类别名。"

                    # LLM推理
                    llm_result = llm_model.predict(
                        stat_vector=stat[sample_idx],
                        bert_embedding=bert[sample_idx],
                        tokenizer=llm_tokenizer,
                        text_prompt=prompt
                    )
                    # llm_result: 0..K-1=已知类, K=unknown
                    routed_pred[idx] = llm_result

                    routing_log.append({
                        'sample_idx': sample_idx,
                        'ood_score': ood_scores[idx].item(),
                        'a3_pred': int(a3_pred[idx].item()),
                        'routed_pred': int(llm_result),
                        'ood_flagged': True,
                        'true_label': int(labels[idx].item())
                    })
            else:
                # 无LLM：直接标记为未知类
                for idx in unknown_indices:
                    routed_pred[idx] = num_known_classes

        all_a3_preds.extend(a3_pred.cpu().numpy().tolist())
        all_routed_preds.extend(routed_pred.cpu().numpy().tolist())
        all_ood_scores.extend(ood_scores.cpu().numpy().tolist())
        all_unknown_flags.extend(unknown_mask.cpu().numpy().tolist())
        all_labels_collected.extend(labels.cpu().numpy().tolist())

        if (batch_idx + 1) % 5 == 0:
            total_processed = len(all_labels_collected)
            unknown_count = sum(all_unknown_flags)
            if verbose:
                print(f"  进度: {total_processed}/{len(test_dataset)} | 路由到LLM: {unknown_count}", end="\r")

    if verbose:
        print(f"\n  推理完成! 总样本: {len(all_labels_collected)}, 路由到LLM: {sum(all_unknown_flags)}")

    # ========== 7. 评估 ==========
    if verbose:
        print("\n" + "-" * 40)
        print("开集评估（OOD路由 vs A3基线）")
        print("=" * 60)

    true_labels_arr = np.array(all_labels_collected)
    a3_pred_arr = np.array(all_a3_preds)
    routed_pred_arr = np.array(all_routed_preds)

    # A3基线评估（闭集：只看已知类）
    if compare_baseline:
        known_mask = true_labels_arr < num_known_classes
        if known_mask.sum() > 0:
            a3_known_acc = accuracy_score(
                true_labels_arr[known_mask], a3_pred_arr[known_mask]
            )
            a3_known_f1 = f1_score(
                true_labels_arr[known_mask], a3_pred_arr[known_mask],
                average='macro', zero_division=0
            )
        else:
            a3_known_acc = 0.0
            a3_known_f1 = 0.0

        # A3在unknown上的表现（全部误分类）
        unknown_mask = true_labels_arr >= num_known_classes
        if unknown_mask.sum() > 0:
            a3_unknown_leak = np.mean(a3_pred_arr[unknown_mask] < num_known_classes)
        else:
            a3_unknown_leak = 0.0

    # OOD路由评估（开集）
    routed_results = evaluate_open_set(
        true_labels_arr, routed_pred_arr, num_known_classes
    )

    if verbose:
        print(f"\n--- A3闭集基线 ---")
        print(f"  已知类准确率: {a3_known_acc:.4f}")
        print(f"  已知类Macro-F1: {a3_known_f1:.4f}")
        print(f"  未知样本泄漏率: {a3_unknown_leak:.4f} (全部被判为known)")

        print(f"\n--- OOD路由 (开集) ---")
        print(f"  整体准确率: {routed_results['accuracy']:.4f}")
        print(f"  Macro-F1: {routed_results['macro_f1']:.4f}")
        print(f"  未知类召回率: {routed_results['unknown_recall']:.4f}")
        print(f"  未知类F1: {routed_results['unknown_f1']:.4f}")
        print(f"  未知类泄漏率: {routed_results['unknown_leak_rate']:.4f}")

        print(f"\n  --- 各类表现 ---")
        for c in range(num_known_classes):
            recall = routed_results.get(f'class_{c}_recall', 0)
            unk_rate = routed_results.get(f'class_{c}_unknown_rate', 0)
            print(f"    {LABEL_NAMES.get(c, str(c))}({c}): 召回={recall:.4f}, 误判为unknown={unk_rate:.4f}")
        if 'unknown_recall' in routed_results:
            print(f"    {LABEL_NAMES.get(2, 'unknown')}(2): 召回={routed_results['unknown_recall']:.4f}")

    duration_seconds = time.time() - start_time

    # ========== 8. 保存结果 ==========
    if verbose:
        print("\n" + "-" * 40)
        print("保存评估结果...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(PROJECT_ROOT, "ood_reports")
    os.makedirs(report_dir, exist_ok=True)

    report_data = {
        'timestamp': timestamp,
        'dataset_id': dataset_id,
        'split_id': split_id,
        'backbone_model_id': backbone_model_id,
        'ood_id': ood_id,
        'llm_model_id': llm_model_id,
        'backbone_variant': backbone_variant,
        'llm_variant': llm_variant,
        'total_test': len(all_labels_collected),
        'routed': {
            'accuracy': routed_results['accuracy'],
            'macro_f1': routed_results['macro_f1'],
            'unknown_recall': routed_results['unknown_recall'],
            'unknown_f1': routed_results['unknown_f1'],
            'unknown_leak_rate': routed_results['unknown_leak_rate'],
            'known_accuracy': routed_results.get('known_accuracy', 0),
            'num_known_routed': routed_results['num_known_routed'],
            'num_unknown_routed': routed_results['num_unknown_routed'],
            'per_class': {
                LABEL_NAMES.get(c, str(c)): {
                    'recall': routed_results.get(f'class_{c}_recall', 0),
                    'unknown_rate': routed_results.get(f'class_{c}_unknown_rate', 0)
                }
                for c in range(num_known_classes)
            },
            'confusion_matrix': routed_results['confusion_matrix'],
            'confusion_labels': routed_results['confusion_labels'],
            'classification_report': routed_results['classification_report']
        },
        'a3_baseline': {
            'known_accuracy': a3_known_acc if compare_baseline else None,
            'known_macro_f1': a3_known_f1 if compare_baseline else None,
            'unknown_leak_rate': a3_unknown_leak if compare_baseline else None,
        },
        'routing_log': routing_log,
        'all_ood_scores': all_ood_scores,
        'all_labels': all_labels_collected,
        'all_routed_preds': all_routed_preds,
        'duration_seconds': round(duration_seconds, 3)
    }

    report_path = os.path.join(report_dir, f"ood_routing_{timestamp}.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)

    if verbose:
        print(f"  - 报告已保存: {report_path}")

    # ========== 9. 日志 ==========
    if verbose:
        print("\n" + "-" * 40)
        print("记录日志...")

    log_data = {
        'dataset_id': dataset_id,
        'split_id': split_id,
        'backbone_model_id': backbone_model_id,
        'ood_id': ood_id,
        'llm_model_id': llm_model_id,
        'total_test': len(all_labels_collected),
        'macro_f1': routed_results['macro_f1'],
        'unknown_f1': routed_results['unknown_f1'],
        'unknown_recall': routed_results['unknown_recall'],
        'duration_seconds': round(duration_seconds, 3)
    }

    log_id = save_log('ood_routing', log_data)

    if verbose:
        print(f"  - 日志已保存: logs/ood_routing/log_{log_id}.json")

    if verbose:
        print("\n" + "=" * 60)
        print(f"OOD路由评估完成！")
        print(f"  - 报告: {report_path}")
        print(f"  - 耗时: {duration_seconds:.2f}秒")
        print("=" * 60)

    return {
        'routed_results': routed_results,
        'a3_baseline': {
            'known_accuracy': a3_known_acc if compare_baseline else None,
            'known_macro_f1': a3_known_f1 if compare_baseline else None,
        },
        'report_path': report_path,
        'duration_seconds': duration_seconds
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OOD路由评估：A3(known) + LLM(unknown)")
    parser.add_argument("--backbone_model_id", type=int, required=True,
                        help="A3 backbone模型ID")
    parser.add_argument("--ood_id", type=int, required=True,
                        help="已训练OOD头ID")
    parser.add_argument("--llm_model_id", type=int, default=None,
                        help="A0 LLM模型ID（可选）")
    parser.add_argument("--dataset_id", type=int, default=1,
                        help="数据集ID（默认1）")
    parser.add_argument("--split_id", type=int, default=0,
                        help="划分ID（对应split_openset_{id}）")
    parser.add_argument("--model_path", type=str, default=None,
                        help="LLM模型路径")
    parser.add_argument("--backbone_variant", type=str, default="A3",
                        help="Backbone变体（默认A3）")
    parser.add_argument("--llm_variant", type=str, default="A0",
                        help="LLM变体（默认A0）")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch大小")
    parser.add_argument("--no_compare_baseline", action="store_true",
                        help="不对比A3基线")
    parser.add_argument("--quiet", action="store_true",
                        help="减少输出")
    parser.add_argument("--fewshot_k", type=int, default=None,
                        help="少样本k值（指定后加载split_fewshot_{k}_{split_id}）")
    args = parser.parse_args()

    run_ood_routing(
        backbone_model_id=args.backbone_model_id,
        ood_id=args.ood_id,
        llm_model_id=args.llm_model_id,
        dataset_id=args.dataset_id,
        split_id=args.split_id,
        model_path=args.model_path,
        backbone_variant=args.backbone_variant,
        llm_variant=args.llm_variant,
        batch_size=args.batch_size,
        verbose=not args.quiet,
        compare_baseline=not args.no_compare_baseline,
        fewshot_k=args.fewshot_k
    )