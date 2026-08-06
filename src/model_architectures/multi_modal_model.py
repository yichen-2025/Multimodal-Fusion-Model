import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from .numeric_encoder import NumericEncoder
from .bert_encoder import BertEncoder
from .fusion_projection import FeatureFusionProjection


class MultiModalFusionModel(nn.Module):
    """
    多模态融合模型（核心类）
    功能：整合数值特征编码器、BERT文本编码器、特征融合投影模块和LLM，实现网络流量分类
    
    整体架构（对应流程图）：
    ┌─────────────────┐    ┌─────────────────┐
    │  数值统计特征   │    │   文本描述特征   │
    │  (9维流量特征)  │    │ (协议/URL等文本) │
    └────────┬────────┘    └────────┬────────┘
             │                      │
             ▼                      ▼
    ┌─────────────────┐    ┌─────────────────┐
    │ NumericEncoder  │    │   BertEncoder   │
    │ (MLP编码网络)   │    │  (BERT预训练)   │
    └────────┬────────┘    └────────┬────────┘
             │                      │
             ▼                      ▼
         [128维向量]          [768维向量]
             │                      │
             └──────────┬───────────┘
                        ▼
            ┌──────────────────────┐
            │ FeatureFusionProjection │
            │   (特征融合+投影)     │
            └──────────┬───────────┘
                       │
                       ▼
              [3584维融合特征]
                       │
                       ▼
            ┌──────────────────────┐
            │   LLM (Qwen2.5)     │
            │   (冻结预训练参数)   │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │   Classifier         │
            │   (线性分类层)       │
            └──────────┬───────────┘
                       │
                       ▼
              [正常流量/恶意流量]
    """

    def __init__(self, 
                 llm_model_path="./models/qwen2.5-1.5b",
                 bert_model_path="./models/bert",
                 numeric_input_dim=9,
                 numeric_hidden_dim=128,
                 numeric_output_dim=128):
        """
        初始化多模态融合模型
        
        Args:
            llm_model_path (str): LLM模型路径或名称，默认使用Qwen2.5-7B-Instruct
            bert_model_path (str): BERT模型路径或名称，默认使用bert-base-chinese
            numeric_input_dim (int): 数值特征输入维度，默认为9
            numeric_hidden_dim (int): 数值编码器隐藏层维度，默认为128
            numeric_output_dim (int): 数值编码器输出维度，默认为128
        """
        super().__init__()

        # 加载LLM模型（因果语言模型，用于文本生成和特征提取）
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_path,
            torch_dtype=torch.bfloat16,   # 使用bfloat16精度，不需要梯度缩放
            device_map="auto",            # 自动分配模型到可用设备（CPU/GPU）
            local_files_only=True         # 仅从本地加载模型，不访问huggingface
        )

        # 获取LLM所在设备和隐藏层维度
        self.device = next(self.llm.parameters()).device
        self.hidden_size = self.llm.config.hidden_size

        # 冻结LLM所有参数，采用Adapter微调策略，仅训练新增的融合和分类层
        for param in self.llm.parameters():
            param.requires_grad = False

        # 初始化BERT文本编码器并移动到LLM所在设备
        self.bert_encoder = BertEncoder(local_model_path=bert_model_path)
        self.bert_encoder.to(self.device)

        # 初始化数值特征编码器并移动到LLM所在设备
        self.numeric_encoder = NumericEncoder(
            input_dim=numeric_input_dim,
            hidden_dim=numeric_hidden_dim,
            output_dim=numeric_output_dim
        )
        self.numeric_encoder.to(self.device)

        # 获取BERT输出维度，初始化特征融合投影模块
        bert_dim = self.bert_encoder.get_hidden_size()
        self.fusion_projection = FeatureFusionProjection(
            numeric_dim=numeric_output_dim,
            bert_dim=bert_dim,
            hidden_dim=2048,
            output_dim=self.hidden_size  # 输出维度与LLM隐藏层一致
        )
        self.fusion_projection.to(self.device)

        # 初始化分类器：将LLM输出映射到分类结果（正常/恶意二分类）
        self.classifier = nn.Linear(self.hidden_size, 2).to(self.device)

        # 统一所有子模块的数据类型，与LLM保持一致
        dtype = next(self.llm.parameters()).dtype
        self.numeric_encoder.to(dtype=dtype)
        self.fusion_projection.to(dtype=dtype)
        self.classifier.to(dtype=dtype)

    def forward(self, stat_tensor, bert_tensor, input_ids=None, attention_mask=None, labels=None):
        """
        前向传播：执行完整的多模态融合和分类流程
        
        Args:
            stat_tensor (torch.Tensor): 数值统计特征，形状为 [batch_size, 9]
            bert_tensor (torch.Tensor): BERT文本特征，形状为 [batch_size, 768]
            input_ids (torch.Tensor, optional): 文本prompt的token id，形状为 [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): 注意力掩码，形状为 [batch_size, seq_len]
            labels (torch.Tensor, optional): 分类标签，形状为 [batch_size]
            
        Returns:
            dict: 包含logits（分类预测）和loss（损失值，可选）的字典
        """
        batch_size = stat_tensor.shape[0]

        # 将输入张量转换为与模型一致的数据类型和设备
        target_dtype = next(self.fusion_projection.parameters()).dtype
        stat_tensor = stat_tensor.to(dtype=target_dtype).to(self.device)
        bert_tensor = bert_tensor.to(dtype=target_dtype).to(self.device)

        # 步骤1：数值特征编码
        numeric_features = self.numeric_encoder(stat_tensor)
        
        # 步骤2：多模态特征融合与投影
        projected_features = self.fusion_projection(numeric_features, bert_tensor)

        # 步骤3：获取文本嵌入（如果提供了input_ids）
        if input_ids is not None:
            input_ids = input_ids.long().to(self.device)
            # 通过LLM的嵌入层获取文本token的嵌入表示
            text_embeds = self.llm.get_input_embeddings()(input_ids)
        else:
            text_embeds = None

        # 步骤4：将融合特征转换为序列形式（增加序列维度）
        fusion_embeds = projected_features.unsqueeze(1)  # [batch, 1, hidden_size]

        # 步骤5：拼接融合特征和文本特征，作为LLM的输入
        if text_embeds is not None:
            # 在序列维度（dim=1）上拼接，融合特征作为序列的第一个token
            inputs_embeds = torch.cat([fusion_embeds, text_embeds], dim=1)
        else:
            # 仅使用融合特征作为输入
            inputs_embeds = fusion_embeds

        # 步骤6：处理注意力掩码
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            # 为融合特征位置创建全1掩码（表示需要关注）
            fusion_mask = torch.ones(batch_size, 1, dtype=attention_mask.dtype).to(self.device)
            # 在序列维度上拼接掩码
            attention_mask = torch.cat([fusion_mask, attention_mask], dim=1)
        else:
            # 创建全1掩码
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=inputs_embeds.dtype).to(self.device)

        # 步骤7：输入LLM进行前向传播，获取隐藏层输出
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True  # 返回所有隐藏层状态
        )

        # 步骤8：提取融合特征位置的输出（序列第一个位置）
        fusion_output = outputs.hidden_states[-1][:, 0, :]
        
        # 步骤9：输入分类器进行分类预测
        logits = self.classifier(fusion_output)

        # 计算损失（如果提供了标签）
        loss = None
        if labels is not None:
            labels = labels.to(self.device)
            loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失
            loss = loss_fn(logits, labels)

        return {"logits": logits, "loss": loss}

    @torch.no_grad()  # 推理模式，禁用梯度计算
    def predict(self, stat_vector, bert_embedding, tokenizer, text_prompt=None):
        """
        推理预测：对单个样本进行流量分类预测
        
        Args:
            stat_vector (torch.Tensor or np.ndarray): 数值统计特征向量，形状为 [9]
            bert_embedding (torch.Tensor or np.ndarray): BERT文本特征向量，形状为 [768]
            tokenizer: LLM的tokenizer
            text_prompt (str, optional): 推理时使用的文本提示
            
        Returns:
            int: 预测标签，0表示正常流量，1表示恶意流量
        """
        self.eval()  # 设置模型为评估模式

        # 将输入转换为PyTorch张量
        if not isinstance(stat_vector, torch.Tensor):
            stat_vector = torch.tensor(stat_vector, dtype=torch.float32)
        if not isinstance(bert_embedding, torch.Tensor):
            bert_embedding = torch.tensor(bert_embedding, dtype=torch.float32)

        # 增加batch维度并移动到模型所在设备
        stat_tensor = stat_vector.unsqueeze(0).to(self.device)
        bert_tensor = bert_embedding.unsqueeze(0).to(self.device)

        # 使用默认提示词（如果未提供）
        if text_prompt is None:
            text_prompt = "根据流量特征判断这个流量是正常流量还是恶意流量。只能输出“正常流量”或“恶意流量”。"

        # 对提示词进行编码
        inputs = tokenizer(text_prompt, return_tensors="pt").to(self.device)

        # 调用forward方法进行预测
        result = self(stat_tensor, bert_tensor, inputs.input_ids, inputs.attention_mask)
        logits = result["logits"]

        # 获取预测结果（取概率最大的类别）
        pred = torch.argmax(logits, dim=1).item()
        return pred

    def get_tokenizer(self):
        """返回LLM对应的tokenizer"""
        return AutoTokenizer.from_pretrained(self.llm.config.name_or_path)

    def save_pretrained(self, save_dir):
        """
        保存模型的可训练参数（不保存冻结的LLM和BERT权重）
        
        Args:
            save_dir (str): 保存目录路径
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        trainable_state = {
            'numeric_encoder': self.numeric_encoder.state_dict(),
            'fusion_projection': self.fusion_projection.state_dict(),
            'classifier': self.classifier.state_dict(),
            'config': {
                'numeric_input_dim': self.numeric_encoder.encoder[0].in_features,
                'numeric_hidden_dim': self.numeric_encoder.encoder[0].out_features,
                'numeric_output_dim': self.numeric_encoder.get_output_dim(),
                'bert_dim': self.bert_encoder.get_hidden_size(),
                'hidden_size': self.hidden_size,
                'llm_model_path': self.llm.config.name_or_path,
                'bert_model_path': self.bert_encoder.bert.config.name_or_path
            }
        }
        
        torch.save(trainable_state, os.path.join(save_dir, 'pytorch_model.bin'))
        print(f"模型可训练参数已保存到 {os.path.abspath(save_dir)}")

    @classmethod
    def from_pretrained(cls, llm_model_path, save_dir):
        """
        从保存的参数加载模型
        
        Args:
            llm_model_path (str): LLM模型路径
            save_dir (str): 保存的参数目录
            
        Returns:
            MultiModalFusionModel: 加载了训练参数的模型
        """
        import os
        
        config_path = os.path.join(save_dir, 'pytorch_model.bin')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"模型文件不存在: {config_path}")
        
        state_dict = torch.load(config_path, map_location='cpu')
        
        config = state_dict['config']
        
        model = cls(
            llm_model_path=llm_model_path,
            bert_model_path=config['bert_model_path'],
            numeric_input_dim=config['numeric_input_dim'],
            numeric_hidden_dim=config['numeric_hidden_dim'],
            numeric_output_dim=config['numeric_output_dim']
        )
        
        model.numeric_encoder.load_state_dict(state_dict['numeric_encoder'])
        model.fusion_projection.load_state_dict(state_dict['fusion_projection'])
        model.classifier.load_state_dict(state_dict['classifier'])
        
        print(f"模型参数已从 {os.path.abspath(save_dir)} 加载")
        return model
