# 多模态融合网络流量分类模型 — 当前架构原理图

> 本图依据 `src/model_architectures/multi_modal_model.py` 的实际代码绘制，反映当前项目的真实前向流程（与 README/docs/theory.md 中描述的“纯 MLP 分类”旧架构不同）。

```mermaid
flowchart TB
    subgraph Input ["输入：一条网络流量样本"]
        N["9 维连续数值统计特征"]
        T["中文文本描述<br/>如：目的端口、包长度均值等"]
    end

    subgraph NumericBranch ["统计特征模态（可训练）"]
        N --> NE["NumericEncoder<br/>Linear(9→128) + ReLU + BatchNorm<br/>Linear(128→128) + ReLU + BatchNorm"]
        NE --> NV["数值特征向量<br/>128 维"]
    end

    subgraph TextBranch ["文本描述模态（冻结）"]
        T --> Tok["Tokenizer<br/>bert-base-chinese"]
        Tok --> BE["BERT 编码器<br/>参数冻结"]
        BE --> BV["文本特征向量<br/>[CLS] 768 维"]
    end

    NV --> FP
    BV --> FP

    subgraph Fusion ["特征融合投影（可训练）"]
        FP["FeatureFusionProjection<br/>concat 策略: [128 ⊕ 768]<br/>Linear(896→2048) + GELU<br/>Linear(2048→1536)"]
        FP --> PF["融合 Embedding<br/>1536 维"]
    end

    PF -->|"unsqueeze(1)<br/>作为首 token"| FE["fusion_embeds<br/>[batch, 1, 1536]"]

    subgraph LLMBranch ["LLM 分支（冻结）"]
        Prompt["文本 Prompt<br/>'根据流量特征判断...'"] --> PTok["Tokenizer"]
        PTok --> PE["text_embeds<br/>[batch, L, 1536]"]
        FE --> Cat["拼接 inputs_embeds<br/>[fusion_embeds ; text_embeds]"]
        PE --> Cat
        Cat --> LLM["Qwen2.5-1.5B<br/>参数冻结"]
        LLM --> FO["取第 0 位输出<br/>hidden_states[-1][:, 0, :]<br/>[batch, 1536]"]
    end

    FO --> CLF["分类器 Classifier<br/>Linear(1536→2)<br/>可训练"]
    CLF --> Out["预测结果<br/>正常流量 / 恶意流量"]

    style LLM fill:#e1f5e1,stroke:#333
    style BE fill:#e1f5e1,stroke:#333
    style CLF fill:#fff2cc,stroke:#333
    style NE fill:#fff2cc,stroke:#333
    style FP fill:#fff2cc,stroke:#333
```

## 关键说明

1. **可训练参数（黄色）**：`NumericEncoder`、`FeatureFusionProjection`、`Classifier`。
2. **冻结参数（绿色）**：`BERT` 与 `Qwen2.5-1.5B` 均不参与反向传播。
3. **LLM 的实际用法**：融合后的 1536 维向量被插入为输入序列的**第一个 token**，与 Prompt 文本的 token embeddings 拼接后送入冻结 LLM；最终取 LLM 输出序列**第 0 位**的 hidden state 进行分类。
4. **消融变体**：
   - `A0`：完整流程（数值 + 文本 + LLM）
   - `A1`：去掉文本模态（数值 + LLM）
   - `A2`：去掉数值模态（文本 + LLM）
   - `A3`：去掉 LLM（数值 + 文本 + 纯 MLP 分类）
