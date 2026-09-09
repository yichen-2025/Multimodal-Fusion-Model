import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from .numeric_encoder import NumericEncoder
from .bert_encoder import BertEncoder
from .fusion_projection import FeatureFusionProjection


class FocalLoss(nn.Module):
    """
    Focal Loss 损失函数
    适用于类别不平衡场景，通过降低易分类样本的权重，聚焦于难分类样本
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha (torch.Tensor, optional): 类别权重张量，形状为 [num_classes]
        gamma (float): 聚焦参数，默认2.0。值越大，对难分类样本的关注越多
    """
    
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        """
        Args:
            logits: 模型输出的 logits，形状为 [batch_size, num_classes]
            targets: 真实标签，形状为 [batch_size]
        """
        # 计算 log_softmax
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        
        # 获取目标类别的 log 概率
        batch_size = logits.size(0)
        num_classes = logits.size(1)
        
        # 对目标类别进行 one-hot 编码
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        
        # 计算每个样本在目标类别上的概率
        pt = (probs * targets_one_hot).sum(dim=1)
        log_pt = (log_probs * targets_one_hot).sum(dim=1)
        
        # Focal Loss 的调制因子
        focal_weight = (1 - pt) ** self.gamma
        
        # 类别加权
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = -alpha_t * focal_weight * log_pt
        else:
            focal_loss = -focal_weight * log_pt
        
        return focal_loss.mean()


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
                 bert_trainable=False,
                 num_classes=2,
                 class_weights=None,
                 classifier_type="mlp",
                 use_temperature=True,
                 dropout_rate=0.1,
                 use_focal_loss=False,
                 focal_gamma=2.0,
                 use_prototype_learning=False,
                 prototype_temperature=0.07,
                 prototype_loss_weight=0.1,
                 llm_use_lora=True,
                 lora_r=8,
                 lora_alpha=16,
                 lora_dropout=0.05,
                 lora_target_modules=None):
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
            num_classes (int): 分类类别数
            class_weights (torch.Tensor, optional): 类别权重张量，用于处理类别不平衡
            classifier_type (str): 分类器类型 "linear" 或 "mlp"
            use_temperature (bool): 是否使用温度缩放
            dropout_rate (float): MLP分类器的dropout率
            use_focal_loss (bool): 是否使用Focal Loss
            focal_gamma (float): Focal Loss的聚焦参数
            use_prototype_learning (bool): 是否使用类别原型对比学习
            prototype_temperature (float): 原型对比学习的温度参数
            prototype_loss_weight (float): 原型对比学习损失的权重
            llm_use_lora (bool): 是否使用 LoRA 微调 LLM（True=LoRA adapter 训练，False=全冻结）
            lora_r (int): LoRA 秩
            lora_alpha (int): LoRA 缩放系数
            lora_dropout (float): LoRA dropout
            lora_target_modules (list): LoRA 目标模块，默认 ["q_proj", "v_proj"]
        """
        super().__init__()

        self.use_numeric = use_numeric
        self.use_bert = use_bert
        self.use_llm = use_llm
        self.fusion_type = fusion_type
        self.bert_trainable = bert_trainable
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.classifier_type = classifier_type
        self.use_temperature = use_temperature
        self.dropout_rate = dropout_rate
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.use_prototype_learning = use_prototype_learning
        self.prototype_temperature = prototype_temperature
        self.prototype_loss_weight = prototype_loss_weight

        # LoRA 配置
        self.llm_use_lora = llm_use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or ["q_proj", "v_proj"]

        # 标记：LLM 是否被 peft 包装过
        self._llm_is_peft = False

        self._keys_to_ignore_on_save = set()

        # 设备选择
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if use_llm and not torch.cuda.is_available():
            print("=" * 60)
            print("警告: 未检测到GPU (CUDA)！")
            print("当前将使用CPU运行，这可能导致训练/推理速度极慢。")
            print("由于是非交互式运行，自动继续CPU模式...")
            print("=" * 60)

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
        # 决定文本模态的来源和维度：LLM 优先（直接编码文本），否则用 BERT
        if use_llm:
            text_dim = 0  # 先占位，LLM 加载后 self.hidden_size 会被设置
        elif use_bert:
            text_dim = self.bert_encoder.get_hidden_size()
        else:
            text_dim = 0
        numeric_dim = numeric_output_dim if use_numeric else 0

        # 加载LLM模型（条件加载）
        if use_llm:
            # 根据设备选择合适的数据类型
            if self.device.type == "cuda":
                llm_dtype = torch.bfloat16
            else:
                llm_dtype = torch.float32
            # 注意：不使用 device_map="auto"，由 Trainer/外部统一管理设备分发，
            # 避免在 PyTorch 2.12+ 下产生 meta tensor 导致 .to(device) 报错
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model_path,
                torch_dtype=llm_dtype,
                local_files_only=True
            )
            self.hidden_size = self.llm.config.hidden_size
            self.llm_model_path = llm_model_path

            # 先冻结全部参数
            for param in self.llm.parameters():
                param.requires_grad = False

            # 加 LoRA adapter
            if llm_use_lora:
                try:
                    from peft import get_peft_model, LoraConfig, TaskType
                    lora_config = LoraConfig(
                        task_type=TaskType.FEATURE_EXTRACTION,
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        target_modules=self.lora_target_modules,
                        bias="none",
                    )
                    self.llm = get_peft_model(self.llm, lora_config)
                    self._llm_is_peft = True
                    print(f"LLM 已加载 LoRA adapter (r={lora_r}, target={self.lora_target_modules})")
                except ImportError:
                    print("警告: peft 未安装，跳过 LoRA。运行 pip install peft")
            else:
                print("LLM 全冻结模式（不使用 LoRA）")
            # 显式把 LLM 搬到 self.device，避免 CPU↔CUDA 设备不一致
            # （去掉 device_map="auto" 后 from_pretrained 默认留在 CPU）
            self.llm.to(self.device)
            print(f"LLM 已加载到 {self.device}")
        else:
            self.llm = None
            self.hidden_size = 1536  # 无LLM时使用默认融合输出维度

        # LLM 加载完后，用 LLM 的 hidden_size 更新 text_dim
        if use_llm and self.hidden_size is not None:
            text_dim = self.hidden_size

        # 保存文本维度供后续使用
        self.text_dim = text_dim

        # 初始化特征融合投影模块
        self.fusion_projection = FeatureFusionProjection(
            numeric_dim=numeric_dim,
            bert_dim=text_dim,  # 参数名仍是 bert_dim，语义改为"文本模态特征维度"
            hidden_dim=2048,
            output_dim=self.hidden_size,
            fusion_type=fusion_type
        )
        self.fusion_projection.to(self.device)

        # 初始化分类器
        if classifier_type == "mlp":
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.LayerNorm(self.hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.hidden_size // 2, num_classes)
            ).to(self.device)
        else:
            self.classifier = nn.Linear(self.hidden_size, num_classes).to(self.device)

        # 温度缩放参数
        if use_temperature:
            self.temperature = nn.Parameter(torch.ones(1) * 2.0).to(self.device)
        else:
            self.temperature = None

        # 类别原型（用于对比学习）
        if use_prototype_learning:
            self.class_prototypes = nn.Parameter(
                torch.randn(num_classes, self.hidden_size) * 0.02
            ).to(self.device)
            self.prototype_projection = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size)
            ).to(self.device)
        else:
            self.class_prototypes = None
            self.prototype_projection = None

        # Focal Loss 损失函数
        if use_focal_loss:
            self.focal_loss_fn = FocalLoss(
                alpha=class_weights,
                gamma=focal_gamma
            )
        else:
            self.focal_loss_fn = None

        # 统一数据类型
        if use_llm and self.llm is not None:
            dtype = next(self.llm.parameters()).dtype
        else:
            dtype = torch.float32
        
        self.numeric_encoder.to(dtype=dtype)
        self.fusion_projection.to(dtype=dtype)
        self.classifier.to(dtype=dtype)
        if self.temperature is not None:
            self.temperature.data = self.temperature.data.to(dtype=dtype)
        if self.bert_encoder is not None:
            self.bert_encoder.to(dtype=dtype)
        if self.class_prototypes is not None:
            self.class_prototypes.data = self.class_prototypes.data.to(dtype=dtype)
        if self.prototype_projection is not None:
            self.prototype_projection.to(dtype=dtype)

    def forward(self, stat_tensor, bert_tensor, input_ids=None, attention_mask=None, labels=None,
                return_features=False):
        """
        前向传播：执行完整的多模态融合和分类流程
        
        Args:
            stat_tensor (torch.Tensor): 数值统计特征，形状为 [batch_size, 9]
            bert_tensor (torch.Tensor): BERT文本特征，形状为 [batch_size, 768]（当 use_bert=True 时有效）
            input_ids (torch.Tensor, optional): 文本的token id（LLM 分支需要）
            attention_mask (torch.Tensor, optional): 注意力掩码
            labels (torch.Tensor, optional): 分类标签
            return_features (bool): 是否在返回中包含融合特征（用于OOD检测）
            
        Returns:
            dict: 包含logits和loss的字典，若return_features=True还包含projected_features
        """
        batch_size = stat_tensor.shape[0]
        target_dtype = next(self.fusion_projection.parameters()).dtype

        # ── Step 1: 处理数值特征 ──
        if self.use_numeric:
            stat_tensor = stat_tensor.to(dtype=target_dtype).to(self.device)
            numeric_features = self.numeric_encoder(stat_tensor)
        else:
            numeric_features = torch.zeros(batch_size, 128, dtype=target_dtype, device=self.device)

        # ── Step 2: 获取文本向量（两条路径二选一：LLM 或 BERT） ──
        text_tensor = None

        if self.use_llm and self.llm is not None:
            # 路径 A：LLM 读真实文本 → mean pooling
            if input_ids is not None:
                input_ids = input_ids.long().to(self.device)
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)
                attention_mask = attention_mask.to(self.device)

                outputs = self.llm(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

                # mean pooling：对有效 token 的最后一层 hidden states 做平均
                # CausalLM 没有 last_hidden_state，需要从 hidden_states tuple 取最后一层
                last_hidden = outputs.hidden_states[-1]  # [batch, seq_len, hidden]
                # 注意：用 last_hidden.dtype 替代 .float()，避免在 bf16/bf16 训练时 dtype 冲突
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).to(dtype=last_hidden.dtype)
                masked_embeddings = last_hidden * mask_expanded
                denom = mask_expanded.sum(dim=1)
                text_tensor = masked_embeddings.sum(dim=1) / (denom + torch.finfo(last_hidden.dtype).eps)
                # text_tensor: [batch, hidden_size]
            else:
                # 没有 input_ids，用零向量占位
                text_tensor = torch.zeros(batch_size, self.hidden_size, dtype=target_dtype, device=self.device)

        elif self.use_bert and self.bert_encoder is not None:
            # 路径 B：使用外部传入的 BERT 嵌入
            text_tensor = bert_tensor.to(dtype=target_dtype).to(self.device)

        else:
            # 无文本模态
            text_tensor = torch.zeros(batch_size, self.text_dim, dtype=target_dtype, device=self.device)

        # ── Step 3: 外层融合 ──
        # 融合位置从 LLM 内部移到 LLM 外部！
        projected_features = self.fusion_projection(numeric_features, text_tensor)

        # ── Step 4: 分类预测 ──
        fusion_output = projected_features
        logits = self.classifier(fusion_output)

        # 类别原型对比学习：将原型相似度添加到 logits
        if self.use_prototype_learning and self.class_prototypes is not None:
            # 投影融合特征到原型空间
            projected_fusion = self.prototype_projection(fusion_output)
            # 计算融合特征与各类别原型的余弦相似度
            normed_fusion = F.normalize(projected_fusion, p=2, dim=1)
            normed_prototypes = F.normalize(self.class_prototypes, p=2, dim=1)
            prototype_similarities = torch.matmul(normed_fusion, normed_prototypes.t())
            # 缩放相似度并添加到 logits
            logits = logits + prototype_similarities / self.prototype_temperature

        # 温度缩放
        if self.temperature is not None:
            logits = logits / self.temperature

        # 计算损失
        loss = None
        if labels is not None:
            labels = labels.to(self.device)
            
            # 主分类损失
            if self.use_focal_loss and self.focal_loss_fn is not None:
                classification_loss = self.focal_loss_fn(logits, labels)
            else:
                if self.class_weights is not None:
                    loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))
                else:
                    loss_fn = nn.CrossEntropyLoss()
                classification_loss = loss_fn(logits, labels)
            
            # 原型对比损失
            prototype_loss = torch.tensor(0.0, device=self.device, dtype=logits.dtype)
            if self.use_prototype_learning and self.class_prototypes is not None:
                # 计算样本特征与所属类别原型的对比损失
                normed_fusion = F.normalize(self.prototype_projection(fusion_output), p=2, dim=1)
                normed_prototypes = F.normalize(self.class_prototypes, p=2, dim=1)
                
                # 获取每个样本对应类别的原型
                batch_prototypes = normed_prototypes[labels]  # [batch_size, hidden_size]
                
                # 正样本对的相似度
                positive_similarity = (normed_fusion * batch_prototypes).sum(dim=1) / self.prototype_temperature
                
                # 所有原型的相似度
                all_similarities = torch.matmul(normed_fusion, normed_prototypes.t()) / self.prototype_temperature
                
                # InfoNCE loss
                prototype_loss = -positive_similarity + torch.logsumexp(all_similarities, dim=1)
                prototype_loss = prototype_loss.mean()
            
            # 总损失
            loss = classification_loss + self.prototype_loss_weight * prototype_loss

        return_dict = {"logits": logits, "loss": loss}

        if return_features:
            return_dict["projected_features"] = projected_features

        return return_dict

    @torch.no_grad()
    def predict(self, stat_vector, bert_embedding=None, tokenizer=None, text=None, text_prompt=None):
        """
        推理预测：对单个样本进行流量分类预测
        
        Args:
            stat_vector: 数值统计特征向量
            bert_embedding: BERT文本特征向量（当 use_bert=True 时有效）
            tokenizer: LLM的tokenizer（当 use_llm=True 时需要）
            text: 真实文本描述（当 use_llm=True 时需要，传给 LLM 做编码）
            text_prompt: 向后兼容参数（同 text，旧代码用 text_prompt）
            
        Returns:
            int: 预测标签
        """
        # 向后兼容：text_prompt 作为 text 的别名
        if text is None and text_prompt is not None:
            text = text_prompt

        self.eval()

        if not isinstance(stat_vector, torch.Tensor):
            stat_vector = torch.tensor(stat_vector, dtype=torch.float32)

        stat_tensor = stat_vector.unsqueeze(0).to(self.device)

        # 构造 bert_tensor（占位）
        if bert_embedding is not None and not isinstance(bert_embedding, torch.Tensor):
            bert_embedding = torch.tensor(bert_embedding, dtype=torch.float32)
        bert_tensor = bert_embedding.unsqueeze(0).to(self.device) if bert_embedding is not None else None

        # LLM 路径：需要真实文本，不再用固定 prompt
        input_ids = None
        attention_mask = None
        if self.use_llm and tokenizer is not None:
            if text is None:
                text = ''  # fallback
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

        # BERT 路径（无 LLM 时）需要 bert_tensor
        if bert_tensor is None and (not self.use_llm or self.llm is None):
            bert_tensor = torch.zeros(1, 768, dtype=torch.float32, device=self.device)

        result = self(stat_tensor, bert_tensor, input_ids=input_ids, attention_mask=attention_mask)

        logits = result["logits"]
        pred = torch.argmax(logits, dim=1).item()
        return pred

    @torch.no_grad()
    def extract_fusion_features(self, stat_tensor, bert_tensor=None, input_ids=None, attention_mask=None):
        """
        提取融合特征（到 fusion_projection 层）

        用于 OOD 检测：获取 hidden_size 维融合特征供 OOD 头使用

        Args:
            stat_tensor (torch.Tensor): 数值统计特征 [batch_size, 9]
            bert_tensor (torch.Tensor, optional): BERT 文本特征 [batch_size, 768]
            input_ids (torch.Tensor, optional): 文本 token ids（LLM 分支）
            attention_mask (torch.Tensor, optional): 注意力掩码（LLM 分支）

        Returns:
            torch.Tensor: 融合投影特征 [batch_size, hidden_size]
        """
        self.eval()
        batch_size = stat_tensor.shape[0]
        target_dtype = next(self.fusion_projection.parameters()).dtype

        # 处理数值特征
        if self.use_numeric:
            stat_tensor = stat_tensor.to(dtype=target_dtype).to(self.device)
            numeric_features = self.numeric_encoder(stat_tensor)
        else:
            numeric_features = torch.zeros(batch_size, 128, dtype=target_dtype, device=self.device)

        # 获取文本向量（与 forward 相同的逻辑）
        if self.use_llm and self.llm is not None and input_ids is not None:
            input_ids = input_ids.long().to(self.device)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            attention_mask = attention_mask.to(self.device)

            outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # CausalLM: 从 hidden_states tuple 取最后一层
            last_hidden = outputs.hidden_states[-1]
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).to(dtype=last_hidden.dtype)
            denom = mask_expanded.sum(dim=1)
            text_tensor = (last_hidden * mask_expanded).sum(dim=1) / (denom + torch.finfo(last_hidden.dtype).eps)
        elif self.use_bert and bert_tensor is not None:
            text_tensor = bert_tensor.to(dtype=target_dtype).to(self.device)
        else:
            text_tensor = torch.zeros(batch_size, self.text_dim, dtype=target_dtype, device=self.device)

        projected_features = self.fusion_projection(numeric_features, text_tensor)
        return projected_features

    def get_feature_dim(self):
        """返回融合特征的维度"""
        return self.fusion_projection.get_output_dim()

    def get_tokenizer(self):
        """返回LLM对应的tokenizer"""
        if self.use_llm and self.llm is not None:
            return AutoTokenizer.from_pretrained(self.llm.config.name_or_path)
        return None

    def state_dict(self, *args, **kwargs):
        """
        覆写state_dict，只返回可训练参数
        使用 requires_grad 判断（适配 peft 包装后 key 前缀变化）
        """
        full_state = super().state_dict(*args, **kwargs)
        trainable_state = {
            k: v for k, v in full_state.items()
            if v.requires_grad
        }
        # 如果全冻结（包括 LoRA 关闭），fallback 到旧的前缀过滤
        if len(trainable_state) == 0:
            trainable_state = {
                k: v for k, v in full_state.items()
                if not k.startswith('llm.') and not k.startswith('bert_encoder.')
                and 'base_model' not in k
            }
        return trainable_state

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        覆写load_state_dict，只加载可训练参数
        """
        filtered_state_dict = {}
        for k, v in state_dict.items():
            # 排除冻结的 LLM 底座参数和 BERT 参数
            if k.startswith('llm.') or k.startswith('bert_encoder.'):
                continue
            if 'base_model.model.llm' in k and 'lora_' not in k:
                continue
            filtered_state_dict[k] = v
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
            'num_classes': self.num_classes,
            'has_class_weights': self.class_weights is not None,
            'classifier_type': self.classifier_type,
            'use_temperature': self.use_temperature,
            'dropout_rate': self.dropout_rate,
            'use_focal_loss': self.use_focal_loss,
            'focal_gamma': self.focal_gamma,
            'use_prototype_learning': self.use_prototype_learning,
            'prototype_temperature': self.prototype_temperature,
            'prototype_loss_weight': self.prototype_loss_weight,
            # LoRA 配置
            'llm_use_lora': self.llm_use_lora,
            'lora_r': self.lora_r,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'lora_target_modules': self.lora_target_modules,
            # 文本维度信息
            'text_dim': self.text_dim,
            '_llm_is_peft': self._llm_is_peft,
        }
        
        if self.class_weights is not None:
            config['class_weights'] = self.class_weights.tolist()
        
        if self.temperature is not None:
            config['temperature'] = self.temperature.data.cpu().tolist()
        
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
        
        if self.temperature is not None:
            trainable_state['temperature'] = self.temperature.data.cpu()
        
        if self.class_prototypes is not None:
            trainable_state['class_prototypes'] = self.class_prototypes.data.cpu()
        
        if self.prototype_projection is not None:
            trainable_state['prototype_projection'] = self.prototype_projection.state_dict()
        
        torch.save(trainable_state, os.path.join(save_dir, 'pytorch_model.bin'))
        print(f"模型可训练参数已保存到 {os.path.abspath(save_dir)}")

        # 如果用了 LoRA，额外保存 adapter 权重
        if self._llm_is_peft and self.llm is not None:
            try:
                lora_dir = os.path.join(save_dir, 'lora_adapter')
                self.llm.save_pretrained(lora_dir)
                print(f"LoRA adapter 已保存到 {os.path.abspath(lora_dir)}")
            except Exception as e:
                print(f"保存 LoRA adapter 时出错: {e}")

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
        class_weights = None
        if config.get('has_class_weights', False) and 'class_weights' in config:
            class_weights = torch.tensor(config['class_weights'], dtype=torch.float32)
        
        classifier_type = config.get('classifier_type', 'linear')
        use_temperature = config.get('use_temperature', False)
        dropout_rate = config.get('dropout_rate', 0.1)
        use_focal_loss = config.get('use_focal_loss', False)
        focal_gamma = config.get('focal_gamma', 2.0)
        use_prototype_learning = config.get('use_prototype_learning', False)
        prototype_temperature = config.get('prototype_temperature', 0.07)
        prototype_loss_weight = config.get('prototype_loss_weight', 0.1)
        
        # LoRA 配置（新增）
        llm_use_lora = config.get('llm_use_lora', True)
        lora_r = config.get('lora_r', 8)
        lora_alpha = config.get('lora_alpha', 16)
        lora_dropout = config.get('lora_dropout', 0.05)
        lora_target_modules = config.get('lora_target_modules', None)
        
        # ⚠️ 关键：先以"不启用 LoRA"的方式创建模型
        # 这样 __init__ 只会加载原始 LLM（保持正确的 dtype: bf16 on GPU / float32 on CPU），
        # 不会提前套一层空的 LoRA adapter，避免后面 PeftModel.from_pretrained 再套一层导致
        # 重复 adapter 和 dtype 漂移（safetensors 加载默认 float32）。
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
            bert_trainable=config.get('bert_trainable', False),
            num_classes=config.get('num_classes', 2),
            class_weights=class_weights,
            classifier_type=classifier_type,
            use_temperature=use_temperature,
            dropout_rate=dropout_rate,
            use_focal_loss=use_focal_loss,
            focal_gamma=focal_gamma,
            use_prototype_learning=use_prototype_learning,
            prototype_temperature=prototype_temperature,
            prototype_loss_weight=prototype_loss_weight,
            llm_use_lora=False,   # ← 关闭，后面手动加载保存的 LoRA
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
        )

        # 加载保存的 LoRA adapter（如果存在）
        lora_adapter_dir = os.path.join(save_dir, 'lora_adapter')
        if os.path.isdir(lora_adapter_dir) and llm_use_lora:
            try:
                from peft import PeftModel
                # PeftModel.from_pretrained 会在不改变底座 dtype 的前提下加载 adapter
                # lora_adapter/ 目录里有 adapter_config.json + adapter_model.safetensors
                base_dtype = next(model.llm.parameters()).dtype
                model.llm = PeftModel.from_pretrained(model.llm, lora_adapter_dir, is_trainable=False)
                model._llm_is_peft = True
                model.llm_use_lora = True  # 修复：加载 adapter 后同步标记，避免 get_config() 误报 False
                print(f"LoRA adapter 已从 {lora_adapter_dir} 加载 (dtype={base_dtype})")
                # 兜底：确保整个 PeftModel 保持与底座一致的 dtype
                model.llm = model.llm.to(dtype=base_dtype)
            except Exception as e:
                print(f"加载 LoRA adapter 时出错: {e}")
                model._llm_is_peft = False

        # 把 numeric / fusion / classifier 等模块同步到 LLM 的 dtype
        # CPU 上 float32，GPU 上 bf16
        if model.llm is not None:
            llm_dtype = next(model.llm.parameters()).dtype
            model.numeric_encoder.to(dtype=llm_dtype)
            model.fusion_projection.to(dtype=llm_dtype)
            model.classifier.to(dtype=llm_dtype)
            if model.temperature is not None:
                model.temperature.data = model.temperature.data.to(dtype=llm_dtype)
            if model.bert_encoder is not None:
                model.bert_encoder.to(dtype=llm_dtype)
            if model.class_prototypes is not None:
                model.class_prototypes.data = model.class_prototypes.data.to(dtype=llm_dtype)
            if model.prototype_projection is not None:
                model.prototype_projection.to(dtype=llm_dtype)

        # 加载训练好的可训练参数权重
        model.numeric_encoder.load_state_dict(state_dict['numeric_encoder'])
        model.fusion_projection.load_state_dict(state_dict['fusion_projection'])
        model.classifier.load_state_dict(state_dict['classifier'])
        
        if 'temperature' in state_dict and model.temperature is not None:
            model.temperature.data.copy_(state_dict['temperature'])
        
        if 'class_prototypes' in state_dict and model.class_prototypes is not None:
            model.class_prototypes.data.copy_(state_dict['class_prototypes'])
        
        if 'prototype_projection' in state_dict and model.prototype_projection is not None:
            model.prototype_projection.load_state_dict(state_dict['prototype_projection'])

        # 最终保险：state_dict 通过 torch.load('cpu') 加载后可能丢失 bf16 精度，
        # 再次把所有模块强制同步到 LLM 的 dtype（bf16 on GPU / float32 on CPU）
        if model.llm is not None:
            llm_dtype = next(model.llm.parameters()).dtype
            model.numeric_encoder.to(dtype=llm_dtype)
            model.fusion_projection.to(dtype=llm_dtype)
            model.classifier.to(dtype=llm_dtype)
            if model.temperature is not None:
                model.temperature.data = model.temperature.data.to(dtype=llm_dtype)
            if model.bert_encoder is not None:
                model.bert_encoder.to(dtype=llm_dtype)
            if model.class_prototypes is not None:
                model.class_prototypes.data = model.class_prototypes.data.to(dtype=llm_dtype)
            if model.prototype_projection is not None:
                model.prototype_projection.to(dtype=llm_dtype)

        print(f"模型参数已从 {os.path.abspath(save_dir)} 加载")
        return model
