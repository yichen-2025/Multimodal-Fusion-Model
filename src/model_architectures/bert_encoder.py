import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class BertEncoder(nn.Module):
    """
    BERT文本编码器模块
    功能：将网络流量的文本描述（如协议类型、URL路径等）编码为语义特征向量
    
    工作流程：
    1. 使用BERT预训练模型对文本进行token化
    2. 将文本输入BERT模型，提取[CLS]标记的输出作为文本语义特征
    3. 冻结BERT参数，仅作为特征提取器使用，不参与训练更新
    """

    def __init__(self, bert_model_name="bert-base-chinese", local_model_path=None):
        """
        初始化BERT编码器
        
        Args:
            bert_model_name (str): BERT预训练模型名称，默认使用bert-base-chinese
            local_model_path (str): 本地模型目录路径，若提供则从本地加载模型（离线模式）
        """
        super().__init__()
        
        model_path = local_model_path if local_model_path is not None else bert_model_name
        
        self.bert = AutoModel.from_pretrained(model_path, local_files_only=(local_model_path is not None))
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=(local_model_path is not None))
        
        self.device = next(self.bert.parameters()).device
        self.hidden_size = self.bert.config.hidden_size

        for param in self.bert.parameters():
            param.requires_grad = False

    @torch.no_grad()  # 推理模式，禁用梯度计算以节省内存和计算量
    def forward(self, text_descriptions):
        """
        前向传播：将文本描述转换为BERT语义特征
        
        Args:
            text_descriptions (str or list): 单个文本字符串或文本列表
            
        Returns:
            torch.Tensor: [CLS]标记的嵌入向量，形状为 [batch_size, hidden_size]
        """
        # 处理单个文本输入的情况，转换为列表形式
        if isinstance(text_descriptions, str):
            text_descriptions = [text_descriptions]

        # 使用tokenizer对文本进行编码，生成input_ids和attention_mask
        inputs = self.tokenizer(
            text_descriptions,
            padding=True,       # 对短文本进行padding，使batch内文本长度一致
            truncation=True,    # 对超长文本进行截断
            max_length=128,     # 最大序列长度限制
            return_tensors="pt" # 返回PyTorch张量
        ).to(self.device)  # 将张量移动到模型所在设备

        # 输入BERT模型进行前向传播
        outputs = self.bert(**inputs)
        # 提取[CLS]标记的输出作为文本特征（[CLS]位于序列第一个位置）
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        return cls_embedding

    def get_tokenizer(self):
        """返回tokenizer实例"""
        return self.tokenizer

    def get_hidden_size(self):
        """返回BERT模型的隐藏层维度"""
        return self.hidden_size
