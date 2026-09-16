import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np


class OODHead(nn.Module):
    """
    开集未知检测头（Out-of-Distribution Detection Head）

    基于类原型距离的OOD检测方法：
    - 为每个已知类维护一个原型向量（均值）
    - 计算样本融合特征到各原型的欧氏距离
    - 取最小距离作为"异常分数"
    - 超过阈值则判定为未知类

    可选增强：
    - Center-Loss 变体增强类内紧凑性
    - 温度缩放（Temperature Scaling）调整分数分布
    """

    def __init__(self,
                 feature_dim=1536,
                 num_known_classes=2,
                 temperature=0.1,
                 distance_type='cosine'):
        """
        初始化OOD检测头

        Args:
            feature_dim (int): 输入特征维度（与融合输出一致，默认1536）
            num_known_classes (int): 已知类别数（默认2，可扩展到多分类）
            temperature (float): 温度缩放系数（默认0.1，<1放大距离差异）
            distance_type (str): 距离度量类型
                - 'euclidean': 欧氏距离
                - 'cosine': 余弦距离（默认，已加 break-symmetry jitter）
                - 'mahalanobis': 马氏距离（需要额外协方差矩阵）
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.num_known_classes = num_known_classes
        self.temperature = temperature
        self.distance_type = distance_type

        # 可学习的类原型向量（每类一个）
        self.prototypes = nn.Parameter(
            torch.randn(num_known_classes, feature_dim) * 0.02
        )

        # 可学习的缩放因子（每类一个，用于调整不同类的分布尺度）
        self.class_scales = nn.Parameter(
            torch.ones(num_known_classes) * 0.5
        )

        # 注册buffer：预计算的统计量（不参与训练）
        self.register_buffer('class_counts', torch.zeros(num_known_classes))
        self.register_buffer('is_fitted', torch.tensor(False))

        # OOD阈值（在验证集上搜索确定）
        self.ood_threshold = None

    def forward(self, features):
        """
        前向传播：计算OOD分数和预测

        Args:
            features (torch.Tensor): 融合特征，形状 [batch_size, feature_dim]

        Returns:
            dict:
                - 'distances': 到每个原型的距离 [batch_size, num_known_classes]
                - 'scores': OOD异常分数 [batch_size] (越小越可能是已知类)
                - 'unknown_mask': 布尔掩码，True表示未知 [batch_size]
                - 'pred_labels': 预测标签 [batch_size] (0..K-1=已知类, K=unknown)
                - 'known_probs': 已知类的softmax概率 [batch_size, num_known_classes]
        """
        batch_size = features.shape[0]

        # 自动对齐 dtype（bf16 骨干 → float32 OOD 头）
        # 原型向量始终是 float32，输入特征可能是 bf16（混合精度训练/推理）
        if features.dtype != self.prototypes.dtype:
            features = features.to(dtype=self.prototypes.dtype)

        # 计算距离矩阵 [batch_size, num_known_classes]
        distances = self._compute_distances(features)

        # 温度缩放 + softmax得到已知类概率
        logits = -distances / self.temperature
        known_probs = F.softmax(logits, dim=1)

        # 最小距离作为OOD分数，温度缩放放大分数差异（temperature<1 放大，克服 cosine 距离在同质化特征上的退化）
        min_distances, pred_known = distances.min(dim=1)
        min_distances = min_distances / self.temperature

        # 判定是否为未知
        if self.ood_threshold is not None:
            unknown_mask = min_distances > self.ood_threshold
        else:
            unknown_mask = torch.zeros(batch_size, dtype=torch.bool, device=features.device)

        # 组合最终预测
        pred_labels = pred_known.clone()
        pred_labels[unknown_mask] = self.num_known_classes  # 设为2=unknown

        return {
            'distances': distances,
            'scores': min_distances,
            'unknown_mask': unknown_mask,
            'pred_labels': pred_labels,
            'known_probs': known_probs
        }

    def _compute_distances(self, features):
        """
        计算特征到各原型的距离

        Args:
            features (torch.Tensor): [batch_size, feature_dim]

        Returns:
            torch.Tensor: [batch_size, num_known_classes]
        """
        if self.distance_type == 'euclidean':
            # 欧氏距离：||f - c_k||
            diffs = features.unsqueeze(1) - self.prototypes.unsqueeze(0)  # [B, K, D]
            distances = torch.norm(diffs, dim=2)  # [B, K]

        elif self.distance_type == 'cosine':
            # 余弦距离：1 - cos_sim
            features_norm = F.normalize(features, p=2, dim=1)  # [B, D]
            prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)  # [K, D]
            cos_sim = torch.mm(features_norm, prototypes_norm.t())  # [B, K]
            distances = 1.0 - cos_sim  # [B, K]

        elif self.distance_type == 'mahalanobis':
            # 简化版马氏距离：使用可学习的class_scales作为方差
            diffs = features.unsqueeze(1) - self.prototypes.unsqueeze(0)  # [B, K, D]
            scales = F.softplus(self.class_scales).unsqueeze(0)  # [1, K]
            distances = torch.norm(diffs, dim=2) / scales  # [B, K]

        else:
            raise ValueError(f"Unknown distance_type: {self.distance_type}")

        return distances

    @torch.no_grad()
    def compute_prototypes(self, features, labels):
        """
        从特征和标签计算类原型（用于初始化或重置原型）

        Args:
            features (torch.Tensor): 融合特征 [num_samples, feature_dim]
            labels (torch.Tensor): 标签 [num_samples]
        """
        self.eval()

        new_prototypes = self.prototypes.clone()
        new_counts = torch.zeros(self.num_known_classes, device=features.device)

        for c in range(self.num_known_classes):
            mask = labels == c
            if mask.sum() > 0:
                class_features = features[mask]
                new_prototypes[c] = class_features.mean(dim=0)
                new_counts[c] = mask.sum()

        self.prototypes.copy_(new_prototypes)
        self.class_counts.copy_(new_counts)
        self.is_fitted.copy_(torch.tensor(True))

    def set_threshold(self, val_features, val_labels, unknown_ratio=None):
        """
        在验证集上搜索最优OOD阈值

        策略：
        1. 计算验证集已知样本的OOD分数
        2. 如果有unknown样本(标签=2)，用它们的分数作为上界参考
        3. 选择使 (已知类F1 + 未知类F1) 最大化的阈值
        4. 若无unknown样本，则基于已知类分数的分布分位数确定阈值

        Args:
            val_features (torch.Tensor): 验证集融合特征
            val_labels (torch.Tensor): 验证集标签
            unknown_ratio (float): 期望的未知类比例（仅作参考）
        """
        self.eval()

        with torch.no_grad():
            results = self.forward(val_features)
            scores = results['scores']  # [num_samples]

        # 分离已知/未知样本
        known_mask = val_labels < self.num_known_classes
        unknown_mask = val_labels >= self.num_known_classes

        known_scores = scores[known_mask]
        unknown_scores = scores[unknown_mask] if unknown_mask.sum() > 0 else None

        if unknown_scores is not None and len(unknown_scores) > 0:
            # 同时有已知和未知样本 → 搜索最优阈值
            thresholds = torch.linspace(
                known_scores.min().item(),
                unknown_scores.max().item(),
                steps=200
            )

            best_threshold = None
            best_f1_sum = -1

            for th in thresholds:
                # 已知类预测
                known_pred = known_scores <= th  # True=已知
                known_gt = torch.ones(len(known_scores), dtype=torch.bool, device=known_scores.device)

                tp_known = (known_pred & known_gt).sum().float()
                fn_known = (~known_pred & known_gt).sum().float()
                fp_known = (known_pred & ~known_gt).sum().float()
                f1_known = 2 * tp_known / (2 * tp_known + fp_known + fn_known + 1e-8)

                # 未知类预测
                unknown_pred = unknown_scores > th
                unknown_gt = torch.ones(len(unknown_scores), dtype=torch.bool, device=unknown_scores.device)

                tp_unknown = (unknown_pred & unknown_gt).sum().float()
                fn_unknown = (~unknown_pred & unknown_gt).sum().float()
                fp_unknown = (unknown_pred & ~unknown_gt).sum().float()
                f1_unknown = 2 * tp_unknown / (2 * tp_unknown + fp_unknown + fn_unknown + 1e-8)

                f1_sum = f1_known + f1_unknown

                if f1_sum > best_f1_sum:
                    best_f1_sum = f1_sum
                    best_threshold = th.item()

        else:
            # 只有已知样本 → 基于分布分位数
            score_mean = known_scores.mean().item()
            score_std = known_scores.std().item()

            # 使用 2 * std 作为阈值（95%置信区间上界的近似）
            best_threshold = score_mean + 2.0 * score_std

            if unknown_ratio is not None:
                # 根据期望未知比例调整
                best_threshold = score_mean + 1.5 * score_std

        self.ood_threshold = best_threshold

        return {
            'threshold': best_threshold,
            'known_score_mean': known_scores.mean().item(),
            'known_score_std': known_scores.std().item(),
            'unknown_score_mean': unknown_scores.mean().item() if unknown_scores is not None else None,
            'f1_sum': best_f1_sum if unknown_scores is not None else None
        }

    def get_ood_score(self, features):
        """
        获取OOD分数（不进行阈值判定）

        Args:
            features (torch.Tensor): [batch_size, feature_dim]

        Returns:
            torch.Tensor: [batch_size] OOD分数
        """
        self.eval()
        with torch.no_grad():
            results = self.forward(features)
        return results['scores']

    def get_known_confidence(self, features):
        """
        获取已知类的预测置信度

        Args:
            features (torch.Tensor): [batch_size, feature_dim]

        Returns:
            torch.Tensor: [batch_size, num_known_classes] softmax概率
        """
        self.eval()
        with torch.no_grad():
            results = self.forward(features)
        return results['known_probs']

    def save(self, save_path):
        """保存OOD头参数和阈值"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        state = {
            'prototypes': self.prototypes.data.cpu(),
            'class_scales': self.class_scales.data.cpu(),
            'ood_threshold': self.ood_threshold,
            'feature_dim': self.feature_dim,
            'num_known_classes': self.num_known_classes,
            'temperature': self.temperature,
            'distance_type': self.distance_type,
        }
        torch.save(state, save_path)

    def load(self, load_path):
        """加载OOD头参数和阈值"""
        state = torch.load(load_path, map_location='cpu', weights_only=False)
        self.prototypes.data.copy_(state['prototypes'].to(self.prototypes.device))
        self.class_scales.data.copy_(state['class_scales'].to(self.class_scales.device))
        self.ood_threshold = state['ood_threshold']
        self.feature_dim = state['feature_dim']
        self.num_known_classes = state['num_known_classes']
        self.temperature = state['temperature']
        self.distance_type = state['distance_type']

    def get_config(self):
        """获取配置字典"""
        return {
            'feature_dim': self.feature_dim,
            'num_known_classes': self.num_known_classes,
            'temperature': self.temperature,
            'distance_type': self.distance_type,
            'ood_threshold': self.ood_threshold,
        }


class OODLoss(nn.Module):
    """
    OOD训练损失函数

    包含：
    1. 标准交叉熵损失（用于已知类分类）
    2. 原型紧凑损失（Center Loss变体，类内紧凑性正则化）
    3. 原型间隔损失（类间可分性）
    """

    def __init__(self, prototype_weight=0.1, margin_weight=0.05):
        """
        Args:
            prototype_weight (float): 原型紧凑损失权重
            margin_weight (float): 原型间隔损失权重
        """
        super().__init__()
        self.prototype_weight = prototype_weight
        self.margin_weight = margin_weight

    def forward(self, distances, pred_labels, true_labels, prototypes):
        """
        计算OOD训练损失

        Args:
            distances (torch.Tensor): 到各原型的距离 [batch_size, num_classes]
            pred_labels (torch.Tensor): 预测标签 [batch_size]
            true_labels (torch.Tensor): 真实标签 [batch_size]
            prototypes (nn.Parameter): 类原型 [num_classes, feature_dim]

        Returns:
            torch.Tensor: 总损失（标量）
        """
        batch_size = distances.shape[0]

        # 1. 交叉熵损失（用距离的负值做softmax）
        logits = -distances
        ce_loss = F.cross_entropy(logits, true_labels)

        # 2. 原型紧凑损失：样本到所属类原型的距离均值
        prototype_loss = torch.tensor(0.0, device=distances.device)
        for i in range(batch_size):
            cls = true_labels[i]
            prototype_loss = prototype_loss + distances[i, cls]
        prototype_loss = prototype_loss / batch_size

        # 3. 原型间隔损失：不同类原型之间的距离应大于一定间隔
        num_classes = prototypes.shape[0]
        margin_loss = torch.tensor(0.0, device=distances.device)
        if num_classes > 1:
            for i in range(num_classes):
                for j in range(i + 1, num_classes):
                    dist = torch.norm(prototypes[i] - prototypes[j])
                    margin_loss = margin_loss + torch.exp(-dist)

        total_loss = ce_loss + self.prototype_weight * prototype_loss + self.margin_weight * margin_loss

        return total_loss, {
            'ce_loss': ce_loss.item(),
            'prototype_loss': prototype_loss.item(),
            'margin_loss': margin_loss.item()
        }