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
    
    支持消融实验配置：
    - use_numeric: 是否使用数值模态
    - use_bert: 是否使用文本模态
    - use_llm: 是否使用LLM
    - fusion_type: 融合策略 (concat/add/attention)
    - bert_trainable: BERT是否可训练
    """

    def __init__(self, 
                 llm_model_path="./models/qwen2.5-1.5b",
                 bert_model_path="./models/bert",
                 numeric_input_dim=9,
                 numeric_hidden_dim=128,
                 numeric_output_dim=128,
                 use_numeric=True,
                 use_bert=True,
                 use_llm=True,
                 fusion_type="concat",
                 bert_trainable=False):
        """
        初始化多模态融合模型
        
        Args:
            llm_model_path (str): LLM模型路径
            bert_model_path (str): BERT模型路径
            numeric_input_dim (int): 数值特征输入维度
            numeric_hidden_dim (int): 数值编码器隐藏层维度
            numeric_output_dim (int): 数值编码器输出维度
            use_numeric (bool): 是否使用数值模态
            use_bert (bool): 是否使用文本模态
            use_llm (bool): 是否使用LLM
            fusion_type (str): 融合策略 concat/add/attention
            bert_trainable (bool): BERT是否可训练
        """
        super().__init__()

        self.use_numeric = use_numeric
        self.use_bert = use_bert
        self.use_llm = use_llm
        self.fusion_type = fusion_type
        self.bert_trainable = bert_trainable

        self._keys_to_ignore_on_save = set()

        # 设备选择：无LLM时不强制要求GPU
        if use_llm:
            if not torch.cuda.is_available():
                print("=" * 60)
                print("警告: 未检测到GPU (CUDA)！")
                print("当前将使用CPU运行，这可能导致训练/推理速度极慢。")
                print("请确认是否继续...")
                print("=" * 60)
                choice = input("输入 'y' 继续使用CPU，输入其他键退出: ")
                if choice.strip().lower() != 'y':
                    raise RuntimeError("用户选择终止：未检测到GPU。请检查CUDA环境或安装GPU版PyTorch。")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            # 无LLM时，自动选择设备
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化数值特征编码器（始终初始化，方便统一state_dict）
        self.numeric_encoder = NumericEncoder(
            input_dim=numeric_input_dim,
            hidden_dim=numeric_hidden_dim,
            output_dim=numeric_output_dim
        )
        self.numeric_encoder.to(self.device)

        # 初始化BERT文本编码器（条件加载）
        if use_bert:
            self.bert_encoder = BertEncoder(local_model_path=bert_model_path)
            self.bert_encoder.to(self.device)
            if bert_trainable:
                for param in self.bert_encoder.parameters():
                    param.requires_grad = True
        else:
            self.bert_encoder = None

        # 根据配置确定各模块的维度
        bert_dim = self.bert_encoder.get_hidden_size() if use_bert else 0
        numeric_dim = numeric_output_dim if use_numeric else 0

        # 加载LLM模型（条件加载）
        if use_llm:
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True
            )
            self.hidden_size = self.llm.config.hidden_size
            for param in self.llm.parameters():
                param.requires_grad = False
            self.llm_model_path = llm_model_path
        else:
            self.llm = None
            self.hidden_size = 1536  # 无LLM时使用默认融合输出维度

        # 初始化特征融合投影模块
        self.fusion_projection = FeatureFusionProjection(
            numeric_dim=numeric_dim,
            bert_dim=bert_dim,
            hidden_dim=2048,
            output_dim=self.hidden_size,
            fusion_type=fusion_type
        )
        self.fusion_projection.to(self.device)

        # 初始化分类器
        self.classifier = nn.Linear(self.hidden_size, 2).to(self.device)

        # 统一数据类型
        if use_llm:
            dtype = next(self.llm.parameters()).dtype
        else:
            dtype = torch.float32
        
        self.numeric_encoder.to(dtype=dtype)
        self.fusion_projection.to(dtype=dtype)
        self.classifier.to(dtype=dtype)
        if self.bert_encoder is not None:
            self.bert_encoder.to(dtype=dtype)

    def forward(self, stat_tensor, bert_tensor, input_ids=None, attention_mask=None, labels=None):
        """
        前向传播：执行完整的多模态融合和分类流程
        
        Args:
            stat_tensor (torch.Tensor): 数值统计特征，形状为 [batch_size, 9]
            bert_tensor (torch.Tensor): BERT文本特征，形状为 [batch_size, 768]
            input_ids (torch.Tensor, optional): 文本prompt的token id
            attention_mask (torch.Tensor, optional): 注意力掩码
            labels (torch.Tensor, optional): 分类标签
            
        Returns:
            dict: 包含logits和loss的字典
        """
        batch_size = stat_tensor.shape[0]
        target_dtype = next(self.fusion_projection.parameters()).dtype

        # 处理数值特征
        if self.use_numeric:
            stat_tensor = stat_tensor.to(dtype=target_dtype).to(self.device)
            numeric_features = self.numeric_encoder(stat_tensor)
        else:
            numeric_features = torch.zeros(batch_size, 128, dtype=target_dtype, device=self.device)

        # 处理BERT特征
        if self.use_bert and self.bert_encoder is not None:
            bert_tensor = bert_tensor.to(dtype=target_dtype).to(self.device)
        else:
            bert_tensor = torch.zeros(batch_size, 768, dtype=target_dtype, device=self.device)

        # 多模态特征融合
        projected_features = self.fusion_projection(numeric_features, bert_tensor)

        # 分支处理：有LLM或无LLM
        if self.use_llm and self.llm is not None:
            # 步骤1：获取文本嵌入
            if input_ids is not None:
                input_ids = input_ids.long().to(self.device)
                text_embeds = self.llm.get_input_embeddings()(input_ids)
            else:
                text_embeds = None

            # 步骤2：将融合特征转换为序列形式
            fusion_embeds = projected_features.unsqueeze(1)

            # 步骤3：拼接融合特征和文本特征
            if text_embeds is not None:
                inputs_embeds = torch.cat([fusion_embeds, text_embeds], dim=1)
            else:
                inputs_embeds = fusion_embeds

            # 步骤4：处理注意力掩码
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
                fusion_mask = torch.ones(batch_size, 1, dtype=attention_mask.dtype).to(self.device)
                attention_mask = torch.cat([fusion_mask, attention_mask], dim=1)
            else:
                attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=inputs_embeds.dtype).to(self.device)

            # 步骤5：输入LLM进行前向传播
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

            # 步骤6：提取融合特征位置的输出
            fusion_output = outputs.hidden_states[-1][:, 0, :]
        else:
            # 无LLM：融合特征直接用于分类
            fusion_output = projected_features

        # 分类预测
        logits = self.classifier(fusion_output)

        # 计算损失
        loss = None
        if labels is not None:
            labels = labels.to(self.device)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def predict(self, stat_vector, bert_embedding, tokenizer=None, text_prompt=None):
        """
        推理预测：对单个样本进行流量分类预测
        
        Args:
            stat_vector: 数值统计特征向量
            bert_embedding: BERT文本特征向量
            tokenizer: LLM的tokenizer（无LLM时可为None）
            text_prompt: 推理时使用的文本提示
            
        Returns:
            int: 预测标签
        """
        self.eval()

        if not isinstance(stat_vector, torch.Tensor):
            stat_vector = torch.tensor(stat_vector, dtype=torch.float32)
        if not isinstance(bert_embedding, torch.Tensor):
            bert_embedding = torch.tensor(bert_embedding, dtype=torch.float32)

        stat_tensor = stat_vector.unsqueeze(0).to(self.device)
        bert_tensor = bert_embedding.unsqueeze(0).to(self.device)

        # 无LLM时不需要tokenizer和text_prompt
        if self.use_llm and tokenizer is not None:
            if text_prompt is None:
                text_prompt = '根据流量特征判断这个流量是正常流量还是恶意流量。只能输出"正常流量"或"恶意流量"。'
            inputs = tokenizer(text_prompt, return_tensors="pt").to(self.device)
            result = self(stat_tensor, bert_tensor, inputs.input_ids, inputs.attention_mask)
        else:
            result = self(stat_tensor, bert_tensor)

        logits = result["logits"]
        pred = torch.argmax(logits, dim=1).item()
        return pred

    def get_tokenizer(self):
        """返回LLM对应的tokenizer"""
        if self.use_llm and self.llm is not None:
            return AutoTokenizer.from_pretrained(self.llm.config.name_or_path)
        return None

    def state_dict(self, *args, **kwargs):
        """
        覆写state_dict，只返回可训练参数
        """
        full_state = super().state_dict(*args, **kwargs)
        trainable_state = {
            k: v for k, v in full_state.items()
            if not k.startswith('llm.') and not k.startswith('bert_encoder.')
        }
        return trainable_state

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        覆写load_state_dict，只加载可训练参数
        """
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith('llm.') and not k.startswith('bert_encoder.')
        }
        return super().load_state_dict(filtered_state_dict, *args, **kwargs)

    def save_pretrained(self, save_dir):
        """
        保存模型的可训练参数
        
        Args:
            save_dir (str): 保存目录路径
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        config = {
            'numeric_input_dim': self.numeric_encoder.encoder[0].in_features,
            'numeric_hidden_dim': self.numeric_encoder.encoder[0].out_features,
            'numeric_output_dim': self.numeric_encoder.get_output_dim(),
            'hidden_size': self.hidden_size,
            'use_numeric': self.use_numeric,
            'use_bert': self.use_bert,
            'use_llm': self.use_llm,
            'fusion_type': self.fusion_type,
            'bert_trainable': self.bert_trainable,
        }
        
        if self.use_llm and self.llm is not None:
            config['llm_model_path'] = self.llm.config.name_or_path
        else:
            config['llm_model_path'] = None
            
        if self.bert_encoder is not None:
            config['bert_dim'] = self.bert_encoder.get_hidden_size()
            config['bert_model_path'] = self.bert_encoder.bert.config.name_or_path
        else:
            config['bert_dim'] = 0
            config['bert_model_path'] = None
        
        trainable_state = {
            'numeric_encoder': self.numeric_encoder.state_dict(),
            'fusion_projection': self.fusion_projection.state_dict(),
            'classifier': self.classifier.state_dict(),
            'config': config
        }
        
        torch.save(trainable_state, os.path.join(save_dir, 'pytorch_model.bin'))
        print(f"模型可训练参数已保存到 {os.path.abspath(save_dir)}")

    @classmethod
    def from_pretrained(cls, llm_model_path_or_save_dir, save_dir=None):
        """
        从保存的参数加载模型
        
        支持两种调用方式：
        1. from_pretrained(llm_model_path, save_dir) - 原始方式
        2. from_pretrained(save_dir) - HuggingFace Trainer兼容方式
        
        Args:
            llm_model_path_or_save_dir: LLM模型路径或保存目录
            save_dir: 保存的参数目录
            
        Returns:
            MultiModalFusionModel: 加载了训练参数的模型
        """
        import os
        
        if save_dir is None:
            save_dir = llm_model_path_or_save_dir
            config_path = os.path.join(save_dir, 'pytorch_model.bin')
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"模型文件不存在: {config_path}")
            state_dict = torch.load(config_path, map_location='cpu', weights_only=False)
            config = state_dict['config']
            llm_model_path = config.get('llm_model_path', None)
            if llm_model_path is None:
                llm_model_path = "./models/qwen2.5-1.5b"
        else:
            llm_model_path = llm_model_path_or_save_dir
            config_path = os.path.join(save_dir, 'pytorch_model.bin')
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"模型文件不存在: {config_path}")
            state_dict = torch.load(config_path, map_location='cpu', weights_only=False)
            config = state_dict['config']
        
        # 根据config中的配置实例化模型
        model = cls(
            llm_model_path=llm_model_path,
            bert_model_path=config.get('bert_model_path', './models/bert'),
            numeric_input_dim=config.get('numeric_input_dim', 9),
            numeric_hidden_dim=config.get('numeric_hidden_dim', 128),
            numeric_output_dim=config.get('numeric_output_dim', 128),
            use_numeric=config.get('use_numeric', True),
            use_bert=config.get('use_bert', True),
            use_llm=config.get('use_llm', True),
            fusion_type=config.get('fusion_type', 'concat'),
            bert_trainable=config.get('bert_trainable', False)
        )
        
        model.numeric_encoder.load_state_dict(state_dict['numeric_encoder'])
        model.fusion_projection.load_state_dict(state_dict['fusion_projection'])
        model.classifier.load_state_dict(state_dict['classifier'])
        
        print(f"模型参数已从 {os.path.abspath(save_dir)} 加载")
        return model
