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
    
    优化：
    - GPU上使用半精度(FP16)推理以节省显存
    - 支持分批处理，避免一次性处理大量文本导致OOM
    """

    def __init__(self, bert_model_name="bert-base-chinese", local_model_path=None, batch_size=32, force_cpu=False):
        """
        初始化BERT编码器
        
        Args:
            bert_model_name (str): BERT预训练模型名称，默认使用bert-base-chinese
            local_model_path (str): 本地模型目录路径，若提供则从本地加载模型（离线模式）
            batch_size (int): 分批处理的批次大小，默认32
            force_cpu (bool): 强制使用CPU运行，即使CUDA可用
        """
        super().__init__()
        
        model_path = local_model_path if local_model_path is not None else bert_model_name
        
        self.bert = AutoModel.from_pretrained(model_path, local_files_only=(local_model_path is not None))
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=(local_model_path is not None))
        
        self.batch_size = batch_size
        
        if not force_cpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.bert.to(self.device)
            self.bert.half()
            print(f"BERT编码器已加载到GPU（半精度FP16），批次大小: {batch_size}")
        else:
            self.device = torch.device("cpu")
            print(f"BERT编码器已加载到CPU，批次大小: {batch_size}")
        
        self.hidden_size = self.bert.config.hidden_size

        for param in self.bert.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, text_descriptions):
        """
        前向传播：将文本描述转换为BERT语义特征（分批处理）
        
        Args:
            text_descriptions (str or list): 单个文本字符串或文本列表
            
        Returns:
            torch.Tensor: [CLS]标记的嵌入向量，形状为 [total_samples, hidden_size]
        """
        if isinstance(text_descriptions, str):
            text_descriptions = [text_descriptions]

        all_embeddings = []
        num_total = len(text_descriptions)

        for start_idx in range(0, num_total, self.batch_size):
            batch_texts = text_descriptions[start_idx:start_idx + self.batch_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.bert(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]

            all_embeddings.append(cls_embedding.cpu())

            del inputs, outputs, cls_embedding
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            current = min(start_idx + self.batch_size, num_total)
            if current % (self.batch_size * 10) == 0 or current == num_total:
                print(f"  BERT编码进度: {current}/{num_total}")

        return torch.cat(all_embeddings, dim=0)

    def get_tokenizer(self):
        """返回tokenizer实例"""
        return self.tokenizer

    def get_hidden_size(self):
        """返回BERT模型的隐藏层维度"""
        return self.hidden_size
