import torch
import torch.nn as nn


class FeatureFusionProjection(nn.Module):
    """
    特征融合与投影模块
    功能：将数值特征和文本特征进行融合，并投影到LLM的隐藏层维度，实现跨模态特征对齐
    
    工作流程：
    1. 将数值特征（128维）和BERT文本特征（768维）在维度1上拼接，得到896维的联合特征
    2. 通过两层全连接网络将联合特征映射到LLM的隐藏层维度（如3584维）
    3. 输出维度与LLM隐藏层一致的融合特征，可直接作为LLM的输入嵌入
    
    网络结构：
    - 第一层：Linear(128+768=896, 2048) + GELU
    - 第二层：Linear(2048, 3584)
    """

    def __init__(self, numeric_dim=128, bert_dim=768, hidden_dim=2048, output_dim=3584):
        """
        初始化特征融合投影模块
        
        Args:
            numeric_dim (int): 数值特征维度，默认为128
            bert_dim (int): BERT文本特征维度，默认为768
            hidden_dim (int): 融合网络隐藏层维度，默认为2048
            output_dim (int): 输出维度，需与LLM隐藏层维度一致，默认为3584（Qwen2.5-7B的隐藏层维度）
        """
        super().__init__()
        # 构建融合投影网络
        self.fusion_projection = nn.Sequential(
            # 第一层：将拼接后的特征映射到隐藏层
            nn.Linear(numeric_dim + bert_dim, hidden_dim),
            # GELU激活函数，平滑的非线性激活，适合Transformer架构
            nn.GELU(),
            # 第二层：将隐藏层特征映射到LLM隐藏层维度
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, numeric_features, bert_features):
        """
        前向传播：融合数值特征和文本特征并投影到目标维度
        
        Args:
            numeric_features (torch.Tensor): 数值编码特征，形状为 [batch_size, numeric_dim]
            bert_features (torch.Tensor): BERT文本特征，形状为 [batch_size, bert_dim]
            
        Returns:
            torch.Tensor: 融合投影后的特征，形状为 [batch_size, output_dim]
        """
        # 在特征维度（dim=1）上拼接数值特征和文本特征
        combined = torch.cat([numeric_features, bert_features], dim=1)
        # 输入融合投影网络进行特征变换
        projected = self.fusion_projection(combined)
        return projected

    def get_output_dim(self):
        """返回投影输出维度"""
        return self.fusion_projection[-1].out_features
