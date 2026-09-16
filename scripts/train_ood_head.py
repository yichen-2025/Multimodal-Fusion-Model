import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
import time
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.model_architectures.multi_modal_model import MultiModalFusionModel
from src.model_architectures.ood_head import OODHead, OODLoss
from src.data.data_loader import load_split_data, collate_fn
from utils.log_utils import save_log
from config.label_config import MERGED_LABEL_NAMES

# P0-1: 固定种子，确保 eval 可复现
torch.manual_seed(42)
np.random.seed(42)


def get_next_ood_id(base_dir=None):
    if base_dir is None:
        base_dir = os.path.join(PROJECT_ROOT, "saved_ood_heads")
    os.makedirs(base_dir, exist_ok=True)
    max_id = -1
    for f in os.listdir(base_dir):
        if f.startswith("ood_head_") and os.path.isdir(os.path.join(base_dir, f)):
            try:
                idx = int(f.replace("ood_head_", ""))
                if idx > max_id:
                    max_id = idx
            except ValueError:
                pass
    return max_id + 1


def train_ood_head(
    model_id=None,
    ood_id=None,
    dataset_id=1,
    split_id=0,
    model_path=None,
    variant="A3",
    distance_type="cosine",
    temperature=0.1,
    learning_rate=1e-3,
    num_epochs=10,
    batch_size=64,
    seed=42,
    fewshot_k=None
):
    """
    训练OOD检测头

    流程：
    1. 加载预训练的A3模型（冻结backbone）
    2. 提取训练集融合特征
    3. 初始化OOD头的类原型
    4. 在特征上微调OOD头的原型向量和缩放因子
    5. 在验证集上搜索最优OOD阈值
    6. 保存OOD头和配置

    Args:
        model_id (int): 已训练模型的ID（用于加载backbone）
        ood_id (int): OOD头ID（默认自动递增）
        dataset_id (int): 数据集ID
        split_id (int): 划分ID（支持openset划分，如split_openset_0）
        model_path (str): LLM模型路径
        variant (str): 变体名（默认A3，无LLM）
        distance_type (str): 距离度量 euclidean/cosine/mahalanobis
        temperature (float): 温度缩放系数
        learning_rate (float): 学习率
        num_epochs (int): 训练轮数
        batch_size (int): batch大小
        seed (int): 随机种子

    Returns:
        tuple: (ood_head, save_path)
    """
    start_time = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    if model_path is None:
        model_path = os.path.join(PROJECT_ROOT, "models", "qwen2.5-1.5b")

    # ========== 1. 加载预训练模型 ==========
    print("=" * 60)
    print("OOD检测头训练")
    print("=" * 60)

    variant_configs = {
        # 基础模态消融（路线 A）
        "A0": {"use_numeric": True,  "use_bert": True,  "use_llm": True},
        "A1": {"use_numeric": True,  "use_bert": False, "use_llm": True},
        "A2": {"use_numeric": False, "use_bert": True,  "use_llm": True},
        "A3": {"use_numeric": True,  "use_bert": True,  "use_llm": False},
        # 路线 B 新变体（LLM 做文本编码器）
        "A0*":         {"use_numeric": True,  "use_bert": False, "use_llm": True},
        "A0_frozen":   {"use_numeric": True,  "use_bert": False, "use_llm": True},
        "A0*_no_num":  {"use_numeric": False, "use_bert": False, "use_llm": True},
        "A0*_no_text": {"use_numeric": True,  "use_bert": False, "use_llm": False},
    }

    config = variant_configs.get(variant, variant_configs["A3"])
    use_numeric = config["use_numeric"]
    use_bert = config["use_bert"]
    use_llm = config["use_llm"]

    if model_id is not None:
        # 从保存的模型加载
        model_save_path = os.path.join(PROJECT_ROOT, "saved_models", f"model_{model_id}")
        if not os.path.exists(model_save_path):
            raise FileNotFoundError(f"模型目录不存在: {model_save_path}")
        print(f"\n加载预训练模型 (model_id={model_id}, variant={variant})...")
        model = MultiModalFusionModel.from_pretrained(model_path, model_save_path)
    else:
        # 从头初始化（不推荐，需要先训练）
        print(f"\n初始化新模型 (variant={variant})...")
        model = MultiModalFusionModel(
            llm_model_path=model_path,
            use_numeric=use_numeric,
            use_bert=use_bert,
            use_llm=use_llm
        )

    # 冻结backbone
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    feature_dim = model.get_feature_dim()
    print(f"  - 融合特征维度: {feature_dim}")

    # ========== 2. 加载数据 ==========
    print("\n" + "-" * 40)
    if fewshot_k is not None:
        print(f"加载少样本数据 (k={fewshot_k})...")
    else:
        print("加载开集数据...")

    train_dataset = load_split_data(
        data_dir=os.path.join(PROJECT_ROOT, "split_data"),
        data_type="train", dataset_id=dataset_id, split_id=split_id,
        openset=(fewshot_k is None),
        fewshot_k=fewshot_k
    )
    if train_dataset is None:
        split_name = f"split_fewshot_{fewshot_k}_{split_id}" if fewshot_k is not None else f"split_openset_{split_id}"
        raise FileNotFoundError(
            f"未找到训练数据: split_data/dataset_{dataset_id}/{split_name}/train.npz"
        )

    val_dataset = load_split_data(
        data_dir=os.path.join(PROJECT_ROOT, "split_data"),
        data_type="val", dataset_id=dataset_id, split_id=split_id,
        openset=(fewshot_k is None),
        fewshot_k=fewshot_k
    )

    # ========== 3. 提取融合特征 ==========
    print("\n" + "-" * 40)
    print("提取融合特征...")

    # 从模型获取 tokenizer（✅ B2 修复：use_llm=True 时必须有 tokenizer 来编码文本）
    model_tokenizer = model.get_tokenizer() if model.use_llm else None
    # 用模型的实际配置覆盖 variant_configs（最可信的真相源）
    actual_use_numeric = model.use_numeric
    actual_use_bert = model.use_bert
    actual_use_llm = model.use_llm
    print(f"  模型实际配置: use_numeric={actual_use_numeric}, "
          f"use_bert={actual_use_bert}, use_llm={actual_use_llm}")
    if actual_use_llm:
        print(f"  ✅ Tokenizer 已就绪，LLM 文本分支将被激活")

    def extract_features(dataset, desc=""):
        features_list = []
        labels_list = []

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            collate_fn=lambda b: collate_fn(b, tokenizer=model_tokenizer,
                                             use_numeric=actual_use_numeric,
                                             use_bert=actual_use_bert,
                                             use_llm=actual_use_llm)
        )

        for batch in dataloader:
            stat = batch["stat_tensor"]
            bert = batch["bert_tensor"]
            labels = batch["labels"]
            input_ids = batch.get("input_ids", None)
            attention_mask = batch.get("attention_mask", None)

            # ✅ B2 修复：传入 input_ids/attention_mask，让 LLM 编码器吃到真实文本
            with torch.no_grad():
                feats = model.extract_fusion_features(
                    stat, bert,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            # bf16 tensor 必须先 cast 成 float32 才能转 numpy（numpy 不支持 BFloat16）
            features_list.append(feats.cpu().float().numpy())
            labels_list.append(labels.numpy())

            progress = len(features_list)
            total = len(dataloader)
            if progress % 10 == 0 or progress == total:
                print(f"  [{desc}] {progress}/{total} batches", end="\r")

        print()
        return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0)

    train_features, train_labels = extract_features(train_dataset, "train")
    val_features, val_labels = extract_features(val_dataset, "val")

    print(f"  - 训练特征: {train_features.shape}, 标签: {np.unique(train_labels, return_counts=True)}")
    print(f"  - 验证特征: {val_features.shape}, 标签: {np.unique(val_labels, return_counts=True)}")

    # ========== 4. 初始化OOD头 ==========
    print("\n" + "-" * 40)
    print("初始化OOD检测头...")

    num_known_classes = len(np.unique(train_labels))
    ood_head = OODHead(
        feature_dim=feature_dim,
        num_known_classes=num_known_classes,
        temperature=temperature,
        distance_type=distance_type
    )
    ood_head.to(model.device)

    # 用训练集特征初始化原型
    train_features_tensor = torch.tensor(train_features, dtype=torch.float32, device=model.device)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long, device=model.device)
    ood_head.compute_prototypes(train_features_tensor, train_labels_tensor)

    print(f"  - 已知类数: {num_known_classes}")
    print(f"  - 原型初始化完成（基于训练集均值）")

    # 打印初始原型分布
    for c in range(num_known_classes):
        mask = train_labels == c
        if mask.sum() > 0:
            class_feat = train_features[mask]
            print(f"    类{c}: {mask.sum()}样本, 特征均值范数={np.linalg.norm(class_feat.mean(0)):.3f}")

    # ========== 5. 微调OOD头 ==========
    print("\n" + "-" * 40)
    print(f"微调OOD头 ({num_epochs} epochs, lr={learning_rate})...")

    ood_head.train()
    optimizer = torch.optim.Adam(ood_head.parameters(), lr=learning_rate)
    criterion = OODLoss(prototype_weight=0.1, margin_weight=0.05)

    dataset_size = len(train_features)
    steps_per_epoch = max(1, dataset_size // batch_size)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_proto = 0.0
        epoch_margin = 0.0
        num_batches = 0

        perm = torch.randperm(dataset_size, device=model.device)

        for step in range(steps_per_epoch):
            idx = perm[step * batch_size: (step + 1) * batch_size]

            batch_features = train_features_tensor[idx]
            batch_labels = train_labels_tensor[idx]

            distances = ood_head._compute_distances(batch_features)
            _, pred_labels = distances.min(dim=1)

            total_loss, loss_dict = criterion(
                distances, pred_labels, batch_labels, ood_head.prototypes
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_ce += loss_dict['ce_loss']
            epoch_proto += loss_dict['prototype_loss']
            epoch_margin += loss_dict['margin_loss']
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        avg_ce = epoch_ce / max(1, num_batches)
        avg_proto = epoch_proto / max(1, num_batches)
        avg_margin = epoch_margin / max(1, num_batches)

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f} "
                  f"(ce={avg_ce:.4f}, proto={avg_proto:.4f}, margin={avg_margin:.4f})")

    # ========== 6. 搜索最优阈值 ==========
    print("\n" + "-" * 40)
    print("在验证集上搜索最优OOD阈值...")

    val_features_tensor = torch.tensor(val_features, dtype=torch.float32, device=model.device)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long, device=model.device)

    threshold_info = ood_head.set_threshold(val_features_tensor, val_labels_tensor)

    print(f"  - 最优阈值: {threshold_info['threshold']:.4f}")
    print(f"  - 已知类分数均值: {threshold_info['known_score_mean']:.4f} ± {threshold_info['known_score_std']:.4f}")
    if threshold_info['unknown_score_mean'] is not None:
        print(f"  - 未知类分数均值: {threshold_info['unknown_score_mean']:.4f}")
    if threshold_info['f1_sum'] is not None:
        print(f"  - 已知F1+未知F1: {threshold_info['f1_sum']:.4f}")

    # ========== 7. 验证集性能评估 ==========
    print("\n" + "-" * 40)
    print("验证集性能评估...")

    ood_head.eval()
    with torch.no_grad():
        val_results = ood_head(val_features_tensor)

    val_pred = val_results['pred_labels'].cpu().numpy()
    val_scores = val_results['scores'].cpu().numpy()

    # 分离已知/未知
    known_mask = val_labels < num_known_classes
    unknown_mask = val_labels >= num_known_classes

    if known_mask.sum() > 0:
        known_pred = val_pred[known_mask]
        known_true = val_labels[known_mask]
        known_acc = np.mean(known_pred == known_true)
        print(f"  已知类准确率: {known_acc:.4f} ({known_mask.sum()}样本)")

        for c in range(num_known_classes):
            class_mask = known_true == c
            if class_mask.sum() > 0:
                class_acc = np.mean(known_pred[class_mask] == c)
                print(f"    类{c} 准确率: {class_acc:.4f} ({class_mask.sum()}样本)")

    if unknown_mask.sum() > 0:
        unknown_correct = val_results['unknown_mask'][unknown_mask].cpu().numpy()
        unknown_recall = np.mean(unknown_correct)
        print(f"  未知类召回率: {unknown_recall:.4f} ({unknown_mask.sum()}样本)")

    # ========== 7.5 分数多样性诊断（防 D1 退化复发）==========
    print("\n  [D1 防退化诊断]")
    scores_rounded = np.round(val_scores, decimals=12)
    unique_ratio = len(np.unique(scores_rounded)) / len(val_scores)
    # 占比 > 5% 的常数聚集
    _, counts = np.unique(scores_rounded, return_counts=True)
    degenerate_cnt = int(np.sum(counts / len(val_scores) >= 0.05))
    print(f"  唯一值占比: {unique_ratio:.2%} "
          f"{'✅ 正常' if unique_ratio > 0.9 else ('⚠️  偏低' if unique_ratio > 0.5 else '❌ 退化!')}")
    print(f"  聚集值(>5%): {degenerate_cnt} "
          f"{'✅' if degenerate_cnt == 0 else '❌ 存在常数聚集!'}")
    if unique_ratio <= 0.5:
        print(f"  🔴 警告: OOD 分数严重退化! 建议换 distance_type (→euclidean) 或调 temperature (→更小)")

    # ========== 8. 保存 ==========
    print("\n" + "-" * 40)
    print("保存OOD检测头...")

    if ood_id is None:
        ood_id = get_next_ood_id()

    save_path = os.path.join(PROJECT_ROOT, "saved_ood_heads", f"ood_head_{ood_id}")
    os.makedirs(save_path, exist_ok=True)

    ood_head.save(os.path.join(save_path, "ood_head.pt"))

    # 保存配置
    config = {
        'ood_id': ood_id,
        'model_id': model_id,
        'variant': variant,
        'dataset_id': dataset_id,
        'split_id': split_id,
        'feature_dim': feature_dim,
        'num_known_classes': num_known_classes,
        'distance_type': distance_type,
        'temperature': temperature,
        'ood_threshold': ood_head.ood_threshold,
        'label_mapping': {str(i): MERGED_LABEL_NAMES.get(i, f"class_{i}")
                          for i in range(num_known_classes)},
        'unknown_label': num_known_classes,
        'creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    import json
    with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # 保存训练特征原型（便于后续分析）
    np.savez(
        os.path.join(save_path, "training_prototypes.npz"),
        prototypes=ood_head.prototypes.data.cpu().numpy(),
        class_scales=ood_head.class_scales.data.cpu().numpy(),
        threshold=np.array([ood_head.ood_threshold])
    )

    duration_seconds = time.time() - start_time

    # ========== 9. 日志记录 ==========
    print("\n" + "-" * 40)
    print("记录日志...")

    train_known = int(np.sum(train_labels < num_known_classes))

    log_data = {
        'model_id': model_id,
        'ood_id': ood_id,
        'dataset_id': dataset_id,
        'split_id': split_id,
        'variant': variant,
        'feature_dim': feature_dim,
        'num_classes': num_known_classes,
        'distance_type': distance_type,
        'temperature': temperature,
        'threshold': ood_head.ood_threshold,
        'train_total': len(train_labels),
        'train_known': train_known,
        'val_total': len(val_labels),
        'val_known': int(known_mask.sum()),
        'val_unknown': int(unknown_mask.sum()),
        'learning_rate': learning_rate,
        'epochs': num_epochs,
        'batch_size': batch_size,
        'seed': seed,
        'save_path': os.path.abspath(save_path),
        'duration_seconds': round(duration_seconds, 3)
    }

    log_id = save_log('ood_training', log_data)
    print(f"  - 日志已保存: logs/ood_training/log_{log_id}.json")

    # ========== 完成 ==========
    print("\n" + "=" * 60)
    print(f"OOD检测头训练完成！")
    print(f"  - OOD头ID: {ood_id}")
    print(f"  - 保存路径: {os.path.abspath(save_path)}")
    print(f"  - 阈值: {ood_head.ood_threshold:.4f}")
    print(f"  - 耗时: {duration_seconds:.2f}秒")
    print("=" * 60)

    return ood_head, save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练OOD检测头（基于预训练模型）")
    parser.add_argument("--model_id", type=int, default=None,
                        help="已训练模型ID（saved_models/model_{id}）")
    parser.add_argument("--ood_id", type=int, default=None,
                        help="OOD头ID（默认自动递增）")
    parser.add_argument("--dataset_id", type=int, default=1,
                        help="数据集ID（默认1）")
    parser.add_argument("--split_id", type=int, default=0,
                        help="划分ID（对应split_openset_{id}）")
    parser.add_argument("--model_path", type=str, default=None,
                        help="LLM模型路径")
    parser.add_argument("--variant", type=str, default="A3",
                        help="消融变体（默认A3）")
    parser.add_argument("--distance_type", type=str, default="cosine",
                        choices=["euclidean", "cosine", "mahalanobis"],
                        help="距离度量类型（cosine 默认，所有距离都加 break-symmetry jitter 防退化）")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="温度缩放系数（默认0.1，<1 放大距离差异）")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="学习率")
    parser.add_argument("--epochs", type=int, default=10,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="batch大小")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--fewshot_k", type=int, default=None,
                        help="少样本k值（指定后加载split_fewshot_{k}_{split_id}）")
    args = parser.parse_args()

    train_ood_head(
        model_id=args.model_id,
        ood_id=args.ood_id,
        dataset_id=args.dataset_id,
        split_id=args.split_id,
        model_path=args.model_path,
        variant=args.variant,
        distance_type=args.distance_type,
        temperature=args.temperature,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        fewshot_k=args.fewshot_k
    )