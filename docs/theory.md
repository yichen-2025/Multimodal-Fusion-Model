# 多模态融合网络流量分类模型原理

## 一、项目背景

网络流量分类是网络安全领域的重要任务，用于识别恶意流量（如DDoS攻击、入侵检测等）。传统方法主要依赖统计特征进行分类，但随着攻击手段的复杂化，单一模态的特征表达能力有限。

本项目采用多模态融合技术，将网络流量数据转换为两种模态：
1. **统计特征模态**：提取流量的数值统计特征
2. **文本描述模态**：将流量特征转换为自然语言描述，通过BERT提取语义特征

通过融合两种模态的信息，提升恶意流量检测的准确率。

## 二、数据处理流程

### 2.1 数据清洗

原始数据集（如CSE-CIC-IDS2018）包含大量网络流量记录，每条记录包含多个特征列。数据清洗步骤包括：

1. **缺失值处理**：删除包含缺失值的样本
2. **异常值检测**：使用IQR方法检测并处理异常值
3. **标签编码**：将`BENIGN`编码为0（正常流量），`DDoS`编码为1（恶意流量）
4. **特征选择**：选择9个关键特征（Destination Port, Bwd Packet Length Mean, Avg Bwd Segment Size, Bwd Packet Length Max, Bwd Packet Length Std, URG Flag Count, Packet Length Mean, Average Packet Size, Packet Length Std）

### 2.2 数据集子集提取

由于原始数据集样本量过大（20万+），为了加速实验，随机抽取部分样本作为子集：

1. **分层抽样**：确保正负样本比例大致相等
2. **标准化**：使用StandardScaler对数值特征进行标准化处理
3. **保存**：将子集保存为CSV文件，同时保存标准化后的特征和标签（npy格式）

### 2.3 模态分离

将数据集转换为两种模态：

#### 统计特征模态

```python
# 9维数值特征
features = [
    "Destination Port",
    "Bwd Packet Length Mean",
    "Avg Bwd Segment Size",
    "Bwd Packet Length Max",
    "Bwd Packet Length Std",
    "URG Flag Count",
    "Packet Length Mean",
    "Average Packet Size",
    "Packet Length Std"
]
```

这些特征反映了流量的统计规律，如包长度分布、端口号、标志位等。

#### 文本描述模态

将每个样本的特征转换为中文文本描述：

```python
# 生成文本描述示例
text = f"""网络流量特征描述：
目的端口：{row['Destination Port']}
反向包长度均值：{row['Bwd Packet Length Mean']}
反向段大小均值：{row['Avg Bwd Segment Size']}
反向包长度最大值：{row['Bwd Packet Length Max']}
反向包长度标准差：{row['Bwd Packet Length Std']}
URG标志位计数：{row['URG Flag Count']}
包长度均值：{row['Packet Length Mean']}
平均包大小：{row['Average Packet Size']}
包长度标准差：{row['Packet Length Std']}
"""
```

然后使用BERT中文模型提取768维语义嵌入向量。

### 2.4 数据集划分

使用分层划分（stratified split）将数据集划分为训练集和测试集：

- **训练集**：80%（用于模型训练）
- **测试集**：20%（用于模型评估）

分层划分确保训练集和测试集的类别分布一致。

## 三、模型架构

### 3.1 整体架构

```
┌─────────────────┐     ┌─────────────────┐
│ 统计特征模态    │     │ 文本描述模态    │
│ (9维)           │     │ (BERT嵌入768维) │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ NumericEncoder  │     │ BertEncoder     │
│ (MLP)           │     │ (冻结BERT)      │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └──────────┬────────────┘
                    ▼
         ┌─────────────────┐
         │ FusionProjection│
         │ (特征融合)      │
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │   Classifier    │
         │   (分类器)      │
         └────────┬────────┘
                  ▼
            ┌──────────┐
            │ 输出概率  │
            │ (正常/恶意)│
            └──────────┘
```

### 3.2 NumericEncoder（数值编码器）

将9维统计特征编码为高维向量：

```python
class NumericEncoder(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=128, output_dim=512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)
```

**设计思路**：
- 通过MLP将低维统计特征映射到高维空间
- 输出维度512，与BERT嵌入维度（768）相近，便于后续融合

### 3.3 BertEncoder（文本编码器）

使用预训练的BERT中文模型提取文本语义特征：

```python
class BertEncoder(nn.Module):
    def __init__(self, bert_model_path):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_path)
        # 冻结BERT参数，只使用其提取特征
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 使用cls token作为句子表示
        return outputs.last_hidden_state[:, 0, :]
```

**设计思路**：
- 使用预训练BERT提取文本语义特征
- 冻结BERT参数，避免破坏预训练知识
- 提取768维的[CLS] token表示作为文本特征

### 3.4 FusionProjection（特征融合投影层）

将两种模态的特征进行融合：

```python
class FusionProjection(nn.Module):
    def __init__(self, numeric_dim=512, bert_dim=768, fusion_dim=512):
        super().__init__()
        self.numeric_proj = nn.Linear(numeric_dim, fusion_dim)
        self.bert_proj = nn.Linear(bert_dim, fusion_dim)
        self.fusion_layer = nn.Linear(fusion_dim * 2, fusion_dim)
    
    def forward(self, numeric_features, bert_features):
        # 分别投影到相同维度
        numeric_proj = self.numeric_proj(numeric_features)
        bert_proj = self.bert_proj(bert_features)
        
        # 拼接后融合
        fused = torch.cat([numeric_proj, bert_proj], dim=1)
        fused = self.fusion_layer(fused)
        
        return fused
```

**设计思路**：
- 将两种模态的特征投影到相同维度（512）
- 通过拼接+线性层进行特征融合
- 融合后的特征包含两种模态的互补信息

### 3.5 Classifier（分类器）

将融合后的特征进行分类：

```python
class Classifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)
```

**设计思路**：
- 两层MLP分类器
- 使用Dropout防止过拟合
- 输出2类概率（正常/恶意）

### 3.6 MultiModalFusionModel（多模态融合模型）

整合所有组件：

```python
class MultiModalFusionModel(nn.Module):
    def __init__(self, llm_model_path):
        super().__init__()
        self.numeric_encoder = NumericEncoder()
        self.bert_encoder = BertEncoder(llm_model_path)
        self.fusion = FusionProjection()
        self.classifier = Classifier()
    
    def forward(self, stat_features, bert_features, input_ids=None, attention_mask=None):
        # 编码两种模态
        numeric_out = self.numeric_encoder(stat_features)
        
        # 如果提供了原始文本，重新提取BERT特征
        if input_ids is not None and attention_mask is not None:
            bert_out = self.bert_encoder(input_ids, attention_mask)
        else:
            bert_out = bert_features
        
        # 融合特征
        fused = self.fusion(numeric_out, bert_out)
        
        # 分类
        logits = self.classifier(fused)
        
        return logits
```

**设计思路**：
- 支持两种模式：使用预提取的BERT嵌入或实时提取
- 灵活的前向传播，适应不同场景

## 四、训练流程

### 4.1 数据加载

使用Hugging Face Dataset加载划分好的数据：

```python
def load_split_data(data_dir="split_data", data_type="train"):
    # 加载npz文件
    data = np.load(os.path.join(data_dir, f"{data_type}.npz"))
    
    # 构建Dataset
    dataset = Dataset.from_dict({
        "stat": data["scaled_features"],
        "bert": data["text_embeddings"],
        "label": data["labels"]
    })
    
    return dataset
```

### 4.2 训练配置

使用Hugging Face Trainer进行训练：

```python
training_args = TrainingArguments(
    output_dir="./saved_models",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # 等效batch_size=8
    learning_rate=1e-4,
    num_train_epochs=3,
    bf16=True,  # 混合精度训练
    logging_steps=10,
    save_strategy="no"
)
```

**关键参数**：
- **batch_size=2**：受显存限制，使用较小的batch_size
- **gradient_accumulation=4**：累积4个batch的梯度后更新参数，等效于batch_size=8
- **bf16=True**：使用bfloat16混合精度训练，节省显存
- **learning_rate=1e-4**：针对小样本微调的学习率

### 4.3 损失函数

使用交叉熵损失函数：

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
```

### 4.4 Loss记录

使用自定义回调记录训练过程中的loss值：

```python
class LossLoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and 'loss' in logs:
            self.loss_log.append({
                'step': state.global_step,
                'epoch': state.epoch,
                'loss': logs['loss']
            })
    
    def on_train_end(self, args, state, control, **kwargs):
        df = pd.DataFrame(self.loss_log)
        df.to_csv(os.path.join(self.log_dir, "loss_log.csv"), index=False)
```

训练完成后可以绘制loss曲线，分析训练过程。

## 五、测试流程

### 5.1 模型加载

```python
def load_model(model_path, llm_model_path):
    model = MultiModalFusionModel(llm_model_path)
    state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
    model.load_state_dict(state_dict)
    model.eval()
    return model
```

### 5.2 评估指标

计算以下评估指标：

| 指标 | 公式 | 说明 |
|------|------|------|
| 准确率 | (TP + TN) / (TP + TN + FP + FN) | 预测正确的比例 |
| 精确率 | TP / (TP + FP) | 预测为正的样本中真正为正的比例 |
| 召回率 | TP / (TP + FN) | 真正为正的样本中被预测为正的比例 |
| F1值 | 2 * 精确率 * 召回率 / (精确率 + 召回率) | 精确率和召回率的调和平均 |

### 5.3 测试报告

自动保存测试报告：

```json
{
  "report_id": 0,
  "timestamp": "2026-07-22T11:51:26",
  "model_id": 0,
  "dataset_id": 0,
  "split_id": 0,
  "test_samples": 1000,
  "test_positive": 500,
  "test_negative": 500,
  "accuracy": 0.952,
  "precision": 0.931,
  "recall": 0.976,
  "f1": 0.953,
  "tp": 488,
  "tn": 464,
  "fp": 36,
  "fn": 12,
  "duration_seconds": 37.07
}
```

## 六、多模态融合原理

### 6.1 为什么需要多模态融合

网络流量数据包含两种类型的信息：

1. **统计信息**：数值特征反映流量的量化属性（如包长度、端口号等）
2. **语义信息**：文本描述反映流量的语义属性（如流量行为模式）

单一模态的局限性：
- 统计特征：难以捕捉复杂的行为模式
- 文本描述：缺乏精确的量化信息

多模态融合可以整合两种信息，提升模型的表达能力。

### 6.2 融合策略

本项目采用**拼接融合**策略：

1. 将两种模态的特征投影到相同维度
2. 拼接成更高维的特征向量
3. 通过线性层进行融合

这种策略的优点：
- 简单有效，保留了两种模态的原始信息
- 训练稳定，易于实现

### 6.3 模态对齐

确保两种模态的特征在语义空间中对齐：

1. **数值特征标准化**：使用StandardScaler将数值特征归一化
2. **文本嵌入提取**：使用预训练BERT提取语义嵌入
3. **投影层对齐**：通过线性层将两种模态投影到相同维度

## 七、关键技术点

### 7.1 参数冻结策略

- **BERT冻结**：BERT参数数量大，训练数据少，冻结避免过拟合
- **Qwen冻结**：只使用Qwen的配置，不加载权重（节省显存）
- **可训练参数**：NumericEncoder、FusionProjection、Classifier

### 7.2 梯度累积

由于显存限制，使用梯度累积技术：
- 小batch_size（2）进行前向传播
- 累积多个batch的梯度
- 达到指定步数后更新参数

### 7.3 混合精度训练

使用bfloat16混合精度训练：
- 减少显存占用
- 加速训练
- 需要NVIDIA Ampere架构及以上GPU支持

## 八、未来改进方向

1. **更复杂的融合策略**：尝试注意力机制、门控机制等高级融合方法
2. **端到端训练**：将BERT也纳入训练，进行微调
3. **更多模态**：加入时序特征、图像特征等更多模态
4. **模型压缩**：使用知识蒸馏、量化等方法压缩模型
5. **实时检测**：优化推理速度，实现实时流量检测