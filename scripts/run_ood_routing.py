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
from utils.dirs import ensure_dirs

from src.model_architectures.multi_modal_model import MultiModalFusionModel
from src.model_architectures.ood_head import OODHead
from src.data.data_loader import load_split_data, collate_fn
from src.explain.retriever import build_cti_explanation
from utils.log_utils import save_log
from utils.caliber import CaliberError

# P0-1: 固定种子，确保 eval 可复现（ood_head 已去扰动）
torch.manual_seed(42)
np.random.seed(42)


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


from utils.open_set_eval import evaluate_open_set  # B8：统一评估口径

# run_ood_routing.py 本地旧 evaluate_open_set 函数已迁移到 utils/open_set_eval.py
# （B8 修复：消除双份实现）


def _compute_auroc(y_true, y_score):
    """P0-3: 纯 NumPy 实现 AUROC（无 sklearn 依赖）

    秩次公式：对所有 pos-neg 样本对，统计 pos_score > neg_score + 0.5 * equal
    """
    y_true = np.array(y_true, dtype=int)
    y_score = np.array(y_score, dtype=float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    pair_gt = np.sum(pos[:, None] > neg[None, :])
    pair_eq = np.sum(pos[:, None] == neg[None, :])
    return float((pair_gt + 0.5 * pair_eq) / (n_pos * n_neg))


def _compute_fpr_at_95_recall(y_true, y_score):
    """计算在 95% True Positive Rate（unknown 召回）下的 False Positive Rate。

    positive = unknown (y_true==1)；分数越大越可能是 unknown。
    返回 None 当无 positive 或無 negative 样本。
    """
    y_true = np.array(y_true, dtype=int)
    y_score = np.array(y_score, dtype=float)
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return None

    # 遍历候选阈值（降序分数），找到 TPR 首次 >= 0.95 处对应的 FPR
    thresholds = np.sort(np.unique(y_score))[::-1]
    for th in thresholds:
        tp = np.sum(pos_scores >= th)
        fp = np.sum(neg_scores >= th)
        tpr = tp / n_pos
        if tpr >= 0.95:
            return float(fp / n_neg)
    # 最宽松阈值仍达不到 95% 召回时，返回最后一个阈值的 FPR
    return float(np.sum(neg_scores >= thresholds[-1]) / n_neg)


def _compute_auroc_by_origin(y_score, origin, code):
    """known(origin=0) vs 指定来源 unknown(origin=code) 的 AUROC。

    origin 不可用、或任一组为空时返回 None；禁止用 0 冒充"没有 OOD 能力"。
    """
    if origin is None:
        return None
    y_score = np.asarray(y_score, dtype=float)
    origin = np.asarray(origin, dtype=int)
    known = origin == 0
    target = origin == code
    if int(known.sum()) == 0 or int(target.sum()) == 0:
        return None
    mask = known | target
    return _compute_auroc(target[mask].astype(int), y_score[mask])


def _compute_tpr_at_fpr(y_true, y_score, target_fpr):
    """FPR <= target_fpr 约束下可达的最大 TPR（分数越大越可能是 unknown）。"""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return None

    best_tpr = 0.0
    for th in np.sort(np.unique(y_score))[::-1]:
        if np.sum(neg_scores >= th) / n_neg > target_fpr:
            continue
        tpr = float(np.sum(pos_scores >= th) / n_pos)
        if tpr > best_tpr:
            best_tpr = tpr
    return best_tpr


def run_ood_routing(
    backbone_model_id=None,
    ood_id=None,
    llm_model_id=None,
    dataset_id=0,
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

    # E1：数据集目录不存在不得靠猜，直接报错
    candidate_data_dir = os.path.join(PROJECT_ROOT, "split_data", f"dataset_{dataset_id}")
    if not os.path.exists(candidate_data_dir):
        raise CaliberError(
            f"dataset_{dataset_id} 不存在，禁止使用脚本默认值，请显式传 --dataset_id"
        )

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

    # unknown_origin 透传：-1 表示旧划分（来源未知），绝不能当成"真实 OOD 召回 = 0"
    test_origins = (
        np.array([int(s.get('unknown_origin', -1)) for s in test_dataset])
        if len(test_dataset) > 0 and 'unknown_origin' in test_dataset[0] else None
    )
    origin_available = bool(test_origins is not None and np.any(test_origins >= 1))

    # E2：阈值校准集(val) 与评估集(test) 的 unknown 构成必须同源，否则阈值不可迁移
    _split_dir_name = (f"split_fewshot_{fewshot_k}_{split_id}"
                       if fewshot_k is not None else f"split_openset_{split_id}")
    _val_npz = os.path.join(
        PROJECT_ROOT, "split_data", f"dataset_{dataset_id}", _split_dir_name, "val.npz"
    )
    if os.path.exists(_val_npz) and origin_available:
        with np.load(_val_npz) as _vd:
            if 'unknown_origin' in _vd:
                _val_origin = np.asarray(_vd['unknown_origin'])
                _val_sim = int(np.sum(_val_origin == 2))
                _test_sim = int(np.sum(test_origins == 2))
                _mixed_declared = bool(
                    'unknown_calibration_mixed' in _vd
                    and int(np.asarray(_vd['unknown_calibration_mixed']).ravel()[0]) == 1
                )
                if (_val_sim > 0) != (_test_sim > 0) and not _mixed_declared:
                    raise CaliberError(
                        f"校准集与评估集的 unknown 构成不同（val 模拟 unknown={_val_sim}, "
                        f"test 模拟 unknown={_test_sim}），阈值不可迁移；"
                        f"请重新生成划分使 val/test 同源，"
                        f"或用 --allow_mixed_calibration 声明混合校准"
                    )

    if verbose:
        print(f"  - 测试样本数: {len(test_dataset)}")
        print(f"  - unknown_origin 可用: {origin_available}")
        for lbl, cnt in label_counts.items():
            print(f"    {LABEL_NAMES.get(lbl, str(lbl))}({lbl}): {cnt}")

    # ========== 5. 构造DataLoader ==========
    # 直接从已加载的 backbone 模型读取配置（最可信），补齐 A0* 系新变体
    backbone_cfg = {
        "use_numeric": backbone.use_numeric,
        "use_bert": backbone.use_bert,
        "use_llm": backbone.use_llm,
    }

    # 获取 backbone 的 tokenizer（仅 use_llm=True 时有意义）
    backbone_tokenizer = backbone.get_tokenizer() if backbone.use_llm else None

    if verbose:
        print(f"\n  Backbone 实际配置: use_numeric={backbone_cfg['use_numeric']}, "
              f"use_bert={backbone_cfg['use_bert']}, use_llm={backbone_cfg['use_llm']}")
        if backbone.use_llm:
            print(f"  ✅ Tokenizer 已就绪，LLM 文本分支将被激活")

    dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer=backbone_tokenizer,
                                         use_numeric=backbone_cfg["use_numeric"],
                                         use_bert=backbone_cfg["use_bert"],
                                         use_llm=backbone_cfg["use_llm"])
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
    n_cti_explanations = 0   # 仅统计：unknown 路径生成的 CTI-RAG 解释条数（不参与判类）

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
        input_ids = batch.get("input_ids", None)
        attention_mask = batch.get("attention_mask", None)

        # 提取融合特征（✅ 关键修复：把 LLM 文本分支的 input_ids/attention_mask 传进去）
        with torch.no_grad():
            features = backbone.extract_fusion_features(
                stat, bert,
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # OOD判定
        with torch.no_grad():
            ood_results = ood_head(features)
            ood_scores = ood_results['scores']
            unknown_mask = ood_results['unknown_mask']

        # 闭集预测（所有样本）
        with torch.no_grad():
            outputs = backbone(stat, bert, input_ids=input_ids, attention_mask=attention_mask)
            a3_logits = outputs["logits"]
            a3_pred = torch.argmax(a3_logits, dim=1)

        # 路由预测
        routed_pred = a3_pred.clone()

        # ✅ B3 修复：unknown 路径由 OOD 头直判，不再走 LLM 闭集重判
        # 架构澄清（对应项目流程图）：
        #   OOD 头 —— 是/否 unknown（门控）
        #   MLP 分类头 —— 闭集已知类预测
        #   CTI-RAG（Phase 3）—— 仅在 unknown 时生成解释文本，不参与判类
        # 所以 OOD 头标记 unknown → 直接输出 num_known_classes 即可
        unknown_indices = torch.where(unknown_mask)[0]
        if len(unknown_indices) > 0:
            batch_start = len(all_labels_collected)   # shuffle=False，batch 内顺序即数据集顺序
            for idx in unknown_indices:
                routed_pred[idx] = num_known_classes

                # CTI-RAG：仅在 unknown 路径生成解释文本，不参与 known/unknown 判定
                sample_idx = int(idx.item())
                sample_text = test_dataset[batch_start + sample_idx].get("text", "")
                explanation = build_cti_explanation(sample_text)
                n_cti_explanations += 1

                # 路由日志
                routing_log.append({
                    'sample_idx': sample_idx,
                    'ood_score': ood_scores[idx].item(),
                    'a3_pred_known_class': int(a3_pred[idx].item()),
                    'routed_pred': num_known_classes,  # K = unknown label
                    'ood_flagged': True,
                    'true_label': int(labels[idx].item()),
                    'cti_rag_explanation': explanation,
                })

        # .float() 防 bf16：numpy 不支持 BFloat16 类型
        all_a3_preds.extend(a3_pred.cpu().numpy().tolist())
        all_routed_preds.extend(routed_pred.cpu().numpy().tolist())
        all_ood_scores.extend(ood_scores.cpu().float().numpy().tolist())
        all_unknown_flags.extend(unknown_mask.cpu().numpy().tolist())
        all_labels_collected.extend(labels.cpu().numpy().tolist())

        if (batch_idx + 1) % 5 == 0:
            total_processed = len(all_labels_collected)
            unknown_count = sum(all_unknown_flags)
            if verbose:
                print(f"  进度: {total_processed}/{len(test_dataset)} | 路由到LLM: {unknown_count}", end="\r")

    if verbose:
        print(f"\n  推理完成! 总样本: {len(all_labels_collected)}, 路由到LLM: {sum(all_unknown_flags)}")
        print(f"  CTI-RAG 解释已生成: {n_cti_explanations} 条（仅 unknown 路径，不参与判类）")

    # ========== 7. 评估 ==========
    if verbose:
        print("\n" + "-" * 40)
        print("开集评估（OOD路由 vs A3基线）")
        print("=" * 60)

    true_labels_arr = np.array(all_labels_collected)
    a3_pred_arr = np.array(all_a3_preds)
    routed_pred_arr = np.array(all_routed_preds)

    # P0-3: AUROC 计算（阈值无关主指标）
    # OOD head 分数：越大越可能是 unknown
    y_true_ood = (true_labels_arr >= num_known_classes).astype(int)  # 1=unknown, 0=known
    auroc_ood = _compute_auroc(y_true_ood, np.array(all_ood_scores))
    fpr_at_95 = _compute_fpr_at_95_recall(y_true_ood, np.array(all_ood_scores))

    # 阶段 1 新增：按 origin 拆分 AUROC + 低 FPR 操作点（均为阈值无关指标）
    auroc_genuine = _compute_auroc_by_origin(all_ood_scores, test_origins, 1)
    auroc_simulated = _compute_auroc_by_origin(all_ood_scores, test_origins, 2)
    tpr_at_fpr_1 = _compute_tpr_at_fpr(y_true_ood, np.array(all_ood_scores), 0.01)
    tpr_at_fpr_5 = _compute_tpr_at_fpr(y_true_ood, np.array(all_ood_scores), 0.05)

    # origin 可用时 genuine AUROC 必须可算；出现 None/NaN 说明数据异常，禁止静默写 0
    if origin_available and not (auroc_genuine is not None and np.isfinite(auroc_genuine)):
        raise CaliberError("unknown_origin 可用但 genuine AUROC 无法计算，禁止用 0/None 掩盖")
    if origin_available and int(np.sum(test_origins == 2)) > 0 and auroc_simulated is None:
        raise CaliberError("存在 simulated unknown 但 simulated AUROC 为 None")

    # E3：必须给出阈值无关指标，禁止只报固定阈值结果
    if not np.isfinite(auroc_ood):
        raise CaliberError("无法计算 AUROC（正/负样本不足），禁止只报固定阈值结果")

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

    # OOD路由评估（开集）：传 origin 用于按来源拆分 unknown_recall
    routed_results = evaluate_open_set(
        true_labels_arr, routed_pred_arr, num_known_classes,
        label_names=LABEL_NAMES,  # B8：传自定义标签名（含 "unknown" 条目）
        origin=test_origins,
    )

    # P0-3: 一致性断言（文档 §5 验收标准）
    n_unknown_routed = int(np.sum(routed_pred_arr == num_known_classes))
    n_routing_log = len(routing_log)
    if n_unknown_routed != n_routing_log:
        print(f"  ⚠️ 一致性警告: routed_pred 中 unknown={n_unknown_routed} 但 routing_log={n_routing_log}")
    else:
        if verbose:
            print(f"  ✅ 一致性通过: routing_log({n_routing_log}) == routed_pred unknown({n_unknown_routed})")

    if verbose:
        print(f"\n--- A3闭集基线 ---")
        print(f"  已知类准确率: {a3_known_acc:.4f}")
        print(f"  已知类Macro-F1: {a3_known_f1:.4f}")
        print(f"  未知样本泄漏率: {a3_unknown_leak:.4f} (全部被判为known)")

        print(f"\n--- OOD路由 (开集) ---")
        # P0-3: AUROC 作为首报主指标
        print(f"  ★ AUROC (主指标, 阈值无关): {auroc_ood:.4f}")
        if fpr_at_95 is not None:
            print(f"  ★ FPR@95TPR: {fpr_at_95:.4f}")
        if auroc_genuine is not None:
            print(f"  ★ AUROC (genuine: origin=1 vs known): {auroc_genuine:.4f}")
        else:
            print("  ★ AUROC (genuine): N/A（unknown_origin 不可用）")
        if auroc_simulated is not None:
            print(f"  ★ AUROC (simulated: origin=2 vs known): {auroc_simulated:.4f}")
        if tpr_at_fpr_1 is not None:
            print(f"  ★ TPR@FPR<=1%: {tpr_at_fpr_1:.4f}")
        if tpr_at_fpr_5 is not None:
            print(f"  ★ TPR@FPR<=5%: {tpr_at_fpr_5:.4f}")
        print(f"  未知类F1: {routed_results['unknown_f1']:.4f}")
        print(f"  未知类召回率(混合): {routed_results['unknown_recall']:.4f}")
        _g = routed_results.get('unknown_recall_genuine')
        _s = routed_results.get('unknown_recall_simulated')
        _g_txt = "N/A" if _g is None else f"{_g:.4f}"
        _s_txt = "N/A" if _s is None else f"{_s:.4f}"
        print(f"     - 真实OOD召回: {_g_txt}")
        print(f"     - 模拟unknown召回: {_s_txt}")
        print(f"  未知类泄漏率: {routed_results['unknown_leak_rate']:.4f}")
        print(f"  已知类分类准确率: {routed_results.get('known_accuracy', 0):.4f}  (known_retention 留存率: {routed_results.get('known_retention', 0):.4f})")
        print(f"  整体准确率: {routed_results['accuracy']:.4f}")
        print(f"  Macro-F1(known): {routed_results.get('macro_f1_known', 0):.4f}  Macro-F1(open): {routed_results['macro_f1']:.4f}")

        print(f"\n  --- 各类表现 ---")
        for c in range(num_known_classes):
            recall = routed_results.get(f'class_{c}_recall', 0)
            unk_rate = routed_results.get(f'class_{c}_unknown_rate', 0)
            print(f"    {LABEL_NAMES.get(c, str(c))}({c}): 召回={recall:.4f}, 误判为unknown={unk_rate:.4f}")
        if 'unknown_recall' in routed_results:
            print(f"    unknown: 召回={routed_results['unknown_recall']:.4f}")

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
            'auroc': auroc_ood,  # P0-3: 主指标
            'fpr_at_95tpr': fpr_at_95,
            'auroc_genuine': auroc_genuine,
            'auroc_simulated': auroc_simulated,
            'tpr_at_fpr_1': tpr_at_fpr_1,
            'tpr_at_fpr_5': tpr_at_fpr_5,
            'accuracy': routed_results['accuracy'],
            'accuracy_open': routed_results.get('accuracy_open', routed_results['accuracy']),
            'macro_f1': routed_results['macro_f1'],
            'macro_f1_known': routed_results.get('macro_f1_known', 0),
            'macro_f1_open': routed_results.get('macro_f1_open', routed_results['macro_f1']),
            'unknown_recall': routed_results['unknown_recall'],
            'unknown_recall_genuine': routed_results.get('unknown_recall_genuine'),
            'unknown_recall_simulated': routed_results.get('unknown_recall_simulated'),
            'unknown_f1': routed_results['unknown_f1'],
            'unknown_leak_rate': routed_results['unknown_leak_rate'],
            'known_accuracy': routed_results.get('known_accuracy', 0),
            'known_retention': routed_results.get('known_retention', 0),
            'unknown_origin_available': origin_available,
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
        'auroc': auroc_ood,  # P0-3: 主指标
        'fpr_at_95tpr': fpr_at_95,
        'auroc_genuine': auroc_genuine,
        'auroc_simulated': auroc_simulated,
        'tpr_at_fpr_1': tpr_at_fpr_1,
        'tpr_at_fpr_5': tpr_at_fpr_5,
        'unknown_f1': routed_results['unknown_f1'],
        'unknown_recall': routed_results['unknown_recall'],
        'unknown_recall_genuine': routed_results.get('unknown_recall_genuine'),
        'unknown_recall_simulated': routed_results.get('unknown_recall_simulated'),
        'unknown_leak_rate': routed_results['unknown_leak_rate'],
        'macro_f1': routed_results['macro_f1'],
        'known_accuracy': routed_results.get('known_accuracy', 0),
        'known_retention': routed_results.get('known_retention', 0),
        'unknown_origin_available': origin_available,
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
        'auroc': auroc_ood,  # P0-3: 主指标
        'auroc_genuine': auroc_genuine,
        'auroc_simulated': auroc_simulated,
        'tpr_at_fpr_1': tpr_at_fpr_1,
        'tpr_at_fpr_5': tpr_at_fpr_5,
        'routed_results': routed_results,
        'a3_baseline': {
            'known_accuracy': a3_known_acc if compare_baseline else None,
            'known_macro_f1': a3_known_f1 if compare_baseline else None,
        },
        'report_path': report_path,
        'duration_seconds': duration_seconds
    }


if __name__ == "__main__":
    ensure_dirs()
    parser = argparse.ArgumentParser(description="OOD路由评估：A3(known) + LLM(unknown)")
    parser.add_argument("--backbone_model_id", type=int, required=True,
                        help="A3 backbone模型ID")
    parser.add_argument("--ood_id", type=int, required=True,
                        help="已训练OOD头ID")
    # 允许用户传 "None" / "none" / 空字符串 来明确跳过
    def _optional_int(s):
        if s is None or s.strip().lower() in ('none', '', 'null'):
            return None
        return int(s)

    parser.add_argument("--llm_model_id", type=_optional_int, default=None,
                        help="A0 LLM模型ID（可选，不写或写 None 即跳过）")
    parser.add_argument("--dataset_id", type=int, default=0,
                        help="数据集ID（默认0）")
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