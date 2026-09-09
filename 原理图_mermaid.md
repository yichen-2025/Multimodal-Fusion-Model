# 多模态融合网络流量分类模型 — 当前架构原理图

> 本图依据 `src/model_architectures/multi_modal_model.py` 的实际代码绘制，反映**当前**前向流程。
> 与旧 README/docs/theory.md 中描述的「融合向量作首 token 喂冻结 LLM 取第 0 位分类」用法**已不同**——
> 真实代码里 LLM 仅作为**文本编码器**在 LLM 外部完成融合，再送入分类头。

## 1. 整体前向流程（架构总览）

```mermaid
flowchart TB
    subgraph Input ["输入：一条网络流量样本"]
        N["9 维连续数值统计特征<br/>（CSV 流特征）"]
        T["中文文本描述<br/>目的端口 / 包长均值 / ... "]
    end

    subgraph NumericBranch ["数值模态（可训练）"]
        N --> NE["NumericEncoder<br/>Linear(9→128) + ReLU + LayerNorm<br/>Linear(128→128) + ReLU + LayerNorm"]
        NE --> NV["数值特征向量<br/>numeric_features<br/>[batch, 128]"]
    end

    subgraph TextBranch ["文本模态：两条路径二选一（互斥）"]
        T --> Tok["Tokenizer<br/>Qwen2.5-1.5B"]
        Tok --> IDS["input_ids + attention_mask<br/>[batch, L]"]

        IDS -- "路径 A · use_llm=True<br/>（LLM-as-Encoder，默认）" --> LLM
        LLM["Qwen2.5-1.5B<br/>底座全冻结<br/>+ 可选 LoRA adapter<br/>(r=8, target=q/v)<br/>output_hidden_states=True"] --> Last["last_hidden<br/>hidden_states[-1]<br/>[batch, L, 1536]"]
        Last --> Pool["mean pooling<br/>按 attention_mask 加权<br/>屏蔽 padding"]
        Pool --> TV["text_tensor<br/>[batch, 1536]"]

        IDS -. "路径 B · use_llm=False 且 use_bert=True" .-> BE["BertEncoder<br/>参数冻结<br/>读取预生成的 bert_tensor"]
        BE -.-> TV2["text_tensor<br/>[batch, 768]"]
    end

    NV -- "始终参与" --> FP
    TV -- "走路径 A 时" --> FP
    TV2 -. "走路径 B 时" .-> FP

    subgraph Fusion ["特征融合投影：FeatureFusionProjection<br/>（在 LLM 外部完成融合，可训练）"]
        FP["支持三种策略<br/>• concat：Linear(1664→2048) + GELU + Linear(2048→1536)<br/>• add：投影到公共维后相加再 MLP<br/>• attention：cross-attn(num→text) 再 MLP"]
        FP --> PF["projected_features<br/>融合 Embedding<br/>[batch, 1536]"]
    end

    PF --> CLF["分类头 Classifier（可训练）<br/>Linear(1536→768) + LayerNorm + GELU + Dropout<br/>Linear(768→num_classes)"]
    CLF --> Logit["logits<br/>[batch, num_classes]"]

    Logit -- "可选 · 温度缩放" --> T1["logits / temperature<br/>（temperature 设为可学习参数）"]
    Logit -- "可选 · 原型对比学习" --> T2["logits + cos(proto) / T_proto<br/>（prototype_similarity）"]
    T1 --> Final["最终 logits"]
    T2 --> Final

    Final --> Out["argmax → 类别标签<br/>（正常 / 恶意 / 12 / 15 类 等）"]

    %% 配色：冻结=绿，可训练=黄
    style LLM fill:#e1f5e1,stroke:#2e7d32
    style BE fill:#e1f5e1,stroke:#2e7d32
    style NE fill:#fff2cc,stroke:#b8860b
    style FP fill:#fff2cc,stroke:#b8860b
    style CLF fill:#fff2cc,stroke:#b8860b
```

## 2. 与旧架构的关键区别

| 维度 | 旧架构（README / 旧 mermaid 描述） | 当前真实架构（multi_modal_model.py） |
|---|---|---|
| 融合位置 | **LLM 内部**：把 1536 维融合向量 unsqueeze 成首 token，与 Prompt token 拼成序列喂入冻结 Qwen，取第 0 位分类 | **LLM 外部**：`FeatureFusionProjection` 先完成两模态融合，再送入纯 MLP 分类头 |
| LLM 角色 | 分类头 | **文本编码器**（mean pooling 取 1536 维向量） |
| 文本输入 | 用固定 Prompt（"根据流量特征判断..."） | 直接喂真实中文流量描述（input_ids + attention_mask） |
| BERT 分支 | 一直启用 → 768 维向量 | 当 `use_llm=True` 时**完全被绕过**（forward Step2 优先走 LLM pooling） |
| LLM 训练性 | 全冻结 | 底座全冻结 + **可选 LoRA adapter**（r=8, q/v） |
| 分类头 | LLM 第 0 位 hidden state → Linear | `Linear→LN→GELU→Dropout→Linear`（MLP），或可选纯 Linear |
| 额外增强 | 无 | 可选温度缩放、原型对比学习、Focal Loss |

## 3. 数据维度速查

| 模块 | 输入维度 | 输出维度 | 说明 |
|---|---|---|---|
| NumericEncoder | 9 | 128 | `use_numeric=False` 时输出全 0 向量占位 |
| Qwen2.5-1.5B（路径 A） | `[batch, L]` (input_ids) | `[batch, 1536]` | mean pooling 末层 hidden states |
| BertEncoder（路径 B） | `[batch, 768]` (bert_tensor) | `[batch, 768]` | 外部传入，向量维度固定 768 |
| FeatureFusionProjection（concat） | `[batch, 128] ⊕ [batch, 1536]` | `[batch, 1536]` | 内部 hidden=2048 |
| Classifier（mlp） | `[batch, 1536]` | `[batch, num_classes]` | num_classes 实际由训练数据决定（2 / 12 / 15） |

## 4. 消融变体（来自 `scripts/train.py` 的 VARIANT_CONFIGS）

| 变体 | use_numeric | use_bert | use_llm | LoRA | 用途 |
|---|---|---|---|---|---|
| A3 | ✓ | ✓ | ✗ | — | 基线：纯数值+BERT+MLP，**验证冻结 LLM 是否冗余** |
| A0（旧用法） | ✓ | ✓ | ✓ | ✗ | 旧"首 token 喂 LLM"用法的当代等效实现 |
| A0*（新用法） | ✓ | ✗ | ✓ | ✓ | LLM-as-Encoder + LoRA，**新正确用法** |
| A0_frozen | ✓ | ✗ | ✓ | ✗ | 对照 A0*：**验证 LoRA 领域适配的价值** |
| A0*_no_num | ✗ | ✗ | ✓ | ✓ | 纯文本+LLM+LoRA，验证**数值模态贡献** |
| A0*_no_text | ✓ | ✗ | ✗ | — | 纯数值+MLP，验证**文本模态贡献** |

可回答的 4 个问题：
1. **LLM 作文本编码器行不行？** → A0* vs A3
2. **旧用法错在 placement，不是 LLM 没用？** → A0* vs A0
3. **LoRA 领域适配是否有用？** → A0* vs A0_frozen
4. **各模态贡献？** → A0*_no_num / A0*_no_text / A0*

## 5. 关键代码位置

- 模型主类：`src/model_architectures/multi_modal_model.py::MultiModalFusionModel`
- forward 中关键三步：
  - Step 1：`numeric_encoder(stat_tensor)` → `[batch, 128]`
  - Step 2：若 `use_llm=True`，`outputs.hidden_states[-1]` + mean pooling → `[batch, 1536]`
  - Step 3：`fusion_projection(numeric_features, text_tensor)` → `[batch, 1536]`
  - Step 4：`classifier(fusion_output)` → logits
- 融合模块：`src/model_architectures/fusion_projection.py::FeatureFusionProjection`
- 数值编码器：`src/model_architectures/numeric_encoder.py::NumericEncoder`
- 变体配置：`scripts/train.py::VARIANT_CONFIGS`