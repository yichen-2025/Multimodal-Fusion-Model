import torch
import torch.nn as nn


class NumericEncoder(nn.Module):
    """
    数值特征编码器模块
    功能：将网络流量的连续数值特征（如流量大小、数据包数量、持续时间等）编码为固定维度的特征向量
    
    工作流程：
    1. 接收9维的流量统计特征向量
    2. 通过两层全连接网络进行特征变换和非线性映射
    3. 输出128维的编码特征向量，用于后续的多模态融合
    
    网络结构：
    - 第一层：Linear(9, 128) + ReLU + BatchNorm1d
    - 第二层：Linear(128, 128) + ReLU + BatchNorm1d
    """

    def __init__(self, input_dim=9, hidden_dim=128, output_dim=128):
        """
        初始化数值特征编码器
        
        Args:
            input_dim (int): 输入特征维度，默认为9（对应9种流量统计特征）
            hidden_dim (int): 隐藏层维度，默认为128
            output_dim (int): 输出特征维度，默认为128
        """
        super().__init__()
        # 构建两层全连接网络，包含非线性激活和批量归一化
        self.encoder = nn.Sequential(
            # 第一层：输入维度映射到隐藏层维度
            nn.Linear(input_dim, hidden_dim),
            # ReLU激活函数，引入非线性
            nn.ReLU(),
            # 批量归一化，加速训练收敛，防止过拟合
            nn.BatchNorm1d(hidden_dim),
            # 第二层：隐藏层维度映射到输出维度
            nn.Linear(hidden_dim, output_dim),
            # ReLU激活函数
            nn.ReLU(),
            # 批量归一化
            nn.BatchNorm1d(output_dim)
        )

    def forward(self, stat_features):
        """
        前向传播：将数值特征编码为固定维度向量
        
        Args:
            stat_features (torch.Tensor or np.ndarray): 流量统计特征，形状为 [batch_size, input_dim]
            
        Returns:
            torch.Tensor: 编码后的特征向量，形状为 [batch_size, output_dim]
        """
        if not isinstance(stat_features, torch.Tensor):
            stat_features = torch.tensor(stat_features, dtype=torch.float32)
        
        param_dtype = next(self.parameters()).dtype
        if stat_features.dtype != param_dtype:
            stat_features = stat_features.to(dtype=param_dtype)
        
        return self.encoder(stat_features)

    def get_output_dim(self):
        """返回编码器输出维度"""
        return self.encoder[-3].out_features
