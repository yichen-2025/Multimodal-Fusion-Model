import torch
import torch.nn as nn


class FeatureFusionProjection(nn.Module):
    """
    特征融合与投影模块
    
    支持三种融合策略：
    - concat: 拼接后通过MLP（默认）
    - add: 投影到同一维度后相加，再通过MLP
    - attention: 用数值特征作为Query，对BERT特征做cross-attention，再通过MLP
    
    支持单模态退化：当某模态dim=0时，自动退化为使用另一模态特征
    """

    def __init__(self, numeric_dim=128, bert_dim=768, hidden_dim=2048, output_dim=3584, fusion_type="concat"):
        """
        初始化特征融合投影模块
        
        Args:
            numeric_dim (int): 数值特征维度
            bert_dim (int): BERT文本特征维度
            hidden_dim (int): 融合网络隐藏层维度
            output_dim (int): 输出维度
            fusion_type (str): 融合策略 concat/add/attention
        """
        super().__init__()
        self.fusion_type = fusion_type
        self.numeric_dim = numeric_dim
        self.bert_dim = bert_dim
        self.output_dim = output_dim

        # 根据融合策略构建不同的网络结构
        if fusion_type == "concat":
            # concat策略：拼接后MLP
            input_dim = numeric_dim + bert_dim
            self.fusion_projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim)
            )
            
        elif fusion_type == "add":
            # add策略：投影到同一维度后相加
            common_dim = 256  # 公共维度
            self.numeric_projection = nn.Linear(numeric_dim, common_dim) if numeric_dim > 0 else None
            self.bert_projection = nn.Linear(bert_dim, common_dim) if bert_dim > 0 else None
            self.fusion_projection = nn.Sequential(
                nn.Linear(common_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim)
            )
            
        elif fusion_type == "attention":
            # attention策略：cross-attention
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=numeric_dim if numeric_dim > 0 else bert_dim,
                num_heads=4,
                batch_first=True
            )
            self.fusion_projection = nn.Sequential(
                nn.Linear(numeric_dim if numeric_dim > 0 else bert_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim)
            )
            
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}. Use concat/add/attention.")

    def forward(self, numeric_features, bert_features):
        """
        前向传播：融合数值特征和文本特征并投影到目标维度
        
        Args:
            numeric_features (torch.Tensor): 数值编码特征
            bert_features (torch.Tensor): BERT文本特征
            
        Returns:
            torch.Tensor: 融合投影后的特征
        """
        if self.fusion_type == "concat":
            return self._forward_concat(numeric_features, bert_features)
        elif self.fusion_type == "add":
            return self._forward_add(numeric_features, bert_features)
        elif self.fusion_type == "attention":
            return self._forward_attention(numeric_features, bert_features)
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

    def _forward_concat(self, numeric_features, bert_features):
        """concat融合策略"""
        # 处理单模态情况
        if self.numeric_dim == 0 and self.bert_dim > 0:
            combined = bert_features
        elif self.bert_dim == 0 and self.numeric_dim > 0:
            combined = numeric_features
        elif self.numeric_dim > 0 and self.bert_dim > 0:
            combined = torch.cat([numeric_features, bert_features], dim=1)
        else:
            # 两个模态都为0（不应出现）
            combined = torch.zeros(numeric_features.shape[0], 1, device=numeric_features.device)
        
        projected = self.fusion_projection(combined)
        return projected

    def _forward_add(self, numeric_features, bert_features):
        """add融合策略"""
        batch_size = numeric_features.shape[0]
        
        # 投影到公共维度
        if self.numeric_dim > 0 and self.numeric_projection is not None:
            num_projected = self.numeric_projection(numeric_features)
        else:
            num_projected = torch.zeros(batch_size, self.bert_projection.out_features if hasattr(self, 'bert_projection') and self.bert_projection is not None else 256, 
                                       device=numeric_features.device, dtype=numeric_features.dtype)
        
        if self.bert_dim > 0 and self.bert_projection is not None:
            bert_projected = self.bert_projection(bert_features)
        else:
            bert_projected = torch.zeros(batch_size, self.numeric_projection.out_features if hasattr(self, 'numeric_projection') and self.numeric_projection is not None else 256,
                                        device=bert_features.device, dtype=bert_features.dtype)
        
        # 相加
        combined = num_projected + bert_projected
        projected = self.fusion_projection(combined)
        return projected

    def _forward_attention(self, numeric_features, bert_features):
        """attention融合策略"""
        # 单模态情况直接退化
        if self.numeric_dim == 0 and self.bert_dim > 0:
            return self.fusion_projection(bert_features)
        elif self.bert_dim == 0 and self.numeric_dim > 0:
            return self.fusion_projection(numeric_features)
        
        # cross-attention：数值特征作为Query，BERT特征作为Key/Value
        # 调整维度以满足attention要求
        if self.numeric_dim != self.bert_dim:
            # 如果维度不同，先投影到相同维度
            common_dim = self.cross_attention.embed_dim
            if numeric_features.shape[1] != common_dim:
                numeric_features = nn.functional.pad(numeric_features, (0, common_dim - numeric_features.shape[1]))
            if bert_features.shape[1] != common_dim:
                bert_features = nn.functional.pad(bert_features, (0, common_dim - bert_features.shape[1]))
        
        # 添加序列维度
        query = numeric_features.unsqueeze(1)  # [batch, 1, dim]
        key_value = bert_features.unsqueeze(1)  # [batch, 1, dim]
        
        # cross-attention
        attn_output, _ = self.cross_attention(query, key_value, key_value)
        attn_output = attn_output.squeeze(1)  # [batch, dim]
        
        projected = self.fusion_projection(attn_output)
        return projected

    def get_output_dim(self):
        """返回投影输出维度"""
        return self.fusion_projection[-1].out_features
