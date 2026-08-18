# 多模态融合流量分类 — 项目流程说明（Phase 1）

> 依据：`todos/todo_1.md`（总实施计划）+ `todos/Phase 1实现.md`（Phase 1 落地总结）
> 分支：`exp/openworld-fewshot`

---

## 一、项目一句话定位

用「数值 + 文本(BERT) + 冻结 LLM(Qwen)」三模态融合做网络流量分类（BENIGN / DDoS）。
闭集二分类已到 0.98 的天花板、冻结 LLM 冗余，于是 **Phase 1 把任务推到「开集未知攻击 + 少样本」场景，制造 LLM 的用武之地**，目标是证明 **A0(有LLM) 在开集/少样本下显著优于 A3(无LLM)**。

---

## 二、整体动机与路线

```mermaid
flowchart TD
    A["闭集二分类<br/>BENIGN / DDoS"] --> B{"准确率已达 0.98？<br/>冻结 LLM 冗余"}
    B -- "是，处于舒适区" --> C["把任务推出舒适区"]
    C --> D["开集未知攻击<br/>留出零日 DDoS"]
    C --> E["少样本 k≤20"]
    C --> F["跨域 / 可解释"]
    D --> G["Phase 1：开集 + 少样本"]
    E --> G
    G --> H{"证明 A0(有LLM)<br/>显著优于 A3(无LLM)？"}
    H -- "是" --> I["LLM 在未知检测中确有价值<br/>推进 Phase 2/3/4"]
    H -- "否" --> J["优先修正 Phase 1 方法"]
```

**说明**：当前只做 Phase 1（开集+少样本）。Phase 2（可解释层 + CTI-RAG）、Phase 3（跨域/时间切分）、Phase 4（整合出论文图）是后续线，三条实验线都从 `ablation` 分支切出，共用 `split_data/dataset_1/split_0` 作闭集对照。

---

## 三、Phase 1 主实验流程（数据 → 模型 → 路由 → 少样本 → 报告）

```mermaid
flowchart TD
    DS["闭集数据<br/>split_data/dataset_1/split_0<br/>train/val/test"] --> P11
    P11["make_openset_split.py<br/>P1.1 开集划分<br/>训练集去掉未知DDoS<br/>测试集混入未知"] --> OS["split_openset_0<br/>train:已知 / test:已知+未知"]
    P14["make_fewshot.py<br/>P1.4 少样本<br/>每类抽 k∈{5,10,20}"] --> FS["split_fewshot_k_0<br/>train:仅 2k 条"]

    BB["训练 A3 Backbone<br/>数值+BERT+MLP(冻结)"] --> OOD
    OOD["train_ood_head.py<br/>P1.2 OOD检测头<br/>类原型距离 + Center-Loss"]
    LLM["训练 A0 LLM<br/>冻结 Qwen 语义推理"]

    OS --> BB
    OS --> OOD
    FS --> OOD
    OOD --> ROUTE
    LLM --> ROUTE

    ROUTE["run_ood_routing.py<br/>P1.3 开集路由"] --> KNOWN["known → A3 分类器<br/>benign / known_DDoS"]
    ROUTE --> UNK["unknown → A0 LLM<br/>语义判定攻击"]
    KNOWN --> EXP
    UNK --> EXP
    EXP["run_openworld_experiment.py<br/>P1.5 自动化实验"] --> OUT["openworld_fewshot.csv<br/>summary.md / metrics.json"]
```

**各阶段要点**：
- **P1.1 开集划分**：从闭集里把一部分 DDoS 标记为「未知」并从训练/验证集移除，只在测试集放回，制造开集。
- **P1.2 OOD 头**：A3 主干冻结，只训「类原型向量」，用欧氏/余弦距离 + 自适应阈值判断未知。
- **P1.3 路由**：已知走 A3 分类器（快），未知走 A0(Qwen) 语义推理（慢但能识别零日）。
- **P1.4 少样本**：每已知类只给 k 条训练样本（k=5/10/20），逼出 LLM 的语义泛化优势。
- **P1.5 报告**：一键串联全流程，输出 CSV / MD / JSON 对比。

---

## 四、开集路由判定流程（最核心的「为什么需要 LLM」）

```mermaid
flowchart TD
    T["测试样本"] --> F["提取 A3 融合特征<br/>1536 维"]
    F --> H["OODHead<br/>计算与各已知类原型距离"]
    H --> D{"最小距离 ≤ 阈值？"}
    D -- "是 · known" --> A3["A3 分类器<br/>预测 benign / known_DDoS"]
    D -- "否 · unknown" --> A0["A0 冻结 Qwen<br/>语义推理<br/>正常流量 / DDoS 攻击"]
    A3 --> M["Macro-F1 / Unknown F1<br/>Unknown Recall / Leak Rate"]
    A0 --> M
```

**说明**：这正是「制造 LLM 用武之地」的机制——A3 在闭集很强但对「没见过的攻击」只会误判成已知类；OOD 头把疑似未知的样本分流给 LLM 做语义推理，从而把 Unknown F1 拉起来。Phase 1 的核心假设就是「开集+少样本下 A0 路由版显著高于 A3-only」。

---

## 五、自动化实验流程（run_openworld_experiment.py）

```mermaid
flowchart TD
    S0["run_openworld_experiment.py"] --> S1["Stage1 训练 A3 Backbone（可选）"]
    S0 --> S1b["Stage1b 训练 A0 LLM（可选）"]
    S1 --> S2["Stage2 对每个 k 循环"]
    S1b --> S2
    S2 --> S2a["2a 训练 OOD 头"]
    S2 --> S2b["2b 运行 OOD 路由评估"]
    S2a --> S2b
    S2b --> S3["Stage3 生成汇总报告"]
    S3 --> R1["openworld_runs_*.csv 明细"]
    S3 --> R2["openworld_fewshot.csv 按 k 对比"]
    S3 --> R3["summary_*.md 可读报告"]
    S3 --> R4["metrics_*.json 结构化指标"]
```

---

## 六、关键文件对照表

| 阶段 | 核心文件 | 作用 |
|------|----------|------|
| P1.1 | `scripts/make_openset_split.py` | 构造开集划分 |
| P1.2 | `src/model_architectures/ood_head.py` + `scripts/train_ood_head.py` | OOD 检测头（类原型距离） |
| P1.3 | `scripts/run_ood_routing.py` | 已知→A3 / 未知→A0 路由评估 |
| P1.4 | `scripts/make_fewshot.py` | 少样本数据（k=5/10/20） |
| P1.5 | `scripts/run_openworld_experiment.py` + `scripts/report_generator.py` | 自动化实验 + 报告 |
| 复用 | `src/model_architectures/multi_modal_model.py` | 含 `extract_fusion_features()` |
| 复用 | `split_data/dataset_1/split_0` | 规范闭集基准 |

---

## 七、核心评估指标（不再报单一 accuracy）

| 指标 | 含义 |
|------|------|
| Macro-F1 | 三分类（benign/known/unknown）宏平均 F1 |
| **Unknown F1** | 把 unknown 当正类的 F1（**核心指标**） |
| Unknown Recall | 未知 DDoS 被正确检测的比例 |
| Unknown Leak Rate | 未知 DDoS 被误判为 known 的比例 |
| Benign / known DDoS Recall | 各自召回率 |

**一句话结论**：Phase 1 用「开集 + 少样本」把任务难度抬高，让 A3-only 在未知攻击上暴露短板，再用 OOD 路由把未知分流给冻结 Qwen 做语义推理，从而用 A0 vs A3 的差距给出「LLM 有用」的硬证据。
