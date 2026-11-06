# Triage 协议（方向 2 · v1）

> 版本：v1 @ 2026-11-06 ｜ 适用：`dataset_200 / split_openset_0` ｜ 状态：**只建协议与标注表，不做评测**
> 上游：`plan/2026-11-05-G1收口与triage.md` §2.3、§3.3-6；G1 已终局（`docs/claims.md` C-25 = `未支持`）
> 相关既有结论：`C-11` 检索远优于 LLM 自选（61/72 vs 40/72，p=0.000324）、`C-13` LLM 不能直派 ATT&CK（两步式后 0/22、0/25）、`C-16`/`C-17` 解释质量只过自动指标

## 1. 任务定义

输入一条**已被 OOD 头判为 unknown** 的流量样本（79 维统计特征；文本描述可用性取决于划分），要求产出一份可直接交给安全分析员处置的 triage 记录，**固定五个输出槽**：

| 槽 | 名称 | 要求 | 不达标的样子 |
|---|---|---|---|
| S1 | 行为概括 | ≤2 句，只写从特征可观测到的行为（连接打点、载荷长度分布、端口/协议形态等），不写推测 | 复述特征名、没有行为语义 |
| S2 | 候选假设 | 2–3 条，每条标 `high/med/low` 置信，并写明"与最接近的已知类的差异点" | 只给 1 条、或三条都同义 |
| S3 | ATT&CK 候选 | 1–2 个技术号，**允许并鼓励显式 abstain**（写明"证据不足以定位"） | 无依据硬派编号 |
| S4 | 下一步动作 | ≥1 条可执行动作（抓同源 N 分钟流量、补某类日志、比对某已知家族指纹、隔离观察窗口） | "继续监控"这类空话 |
| S5 | 证据边界 | 说明结论依赖哪些证据、不覆盖什么 | 缺失，或与 S1–S4 无关 |

判定 unknown 的依据（`ood_score`、阈值、head id）由系统侧记录，**不算分析员要写的内容**。

## 2. 四个臂

| 臂 | 输入 | 生成方式 | 允许引用证据 | 允许 abstain |
|---|---|---|---|---|
| **A0 模板** | head 输出 + 分数 | 固定模板填空（"样本被判为未知，得分 X，建议人工复核"） | 否 | 不适用 |
| **A1 检索-only** | 行为词元 → CTI 检索 | 规则拼装 `build_cti_explanation` 的 top-k 证据，不调 LLM | 是（带出处） | 是（无命中即 abstain） |
| **A2 LLM-only** | 行为词元/特征摘要 | 单步生成五槽，不给 CTI | 否 | 是 |
| **A3 LLM+RAG** | 行为词元 + top-k CTI 证据 | 现行两步式：先判证据是否足够，再选编号/生成 | 是（带出处） | 是（第一步即 abstain） |

`A0` 是下限基线，`A1` 是"不用 LLM 的上限"，`A2` 与 `A3` 的差即为**检索增益**，`A3` 与 `A1` 的差即为**LLM 增益**。

## 3. 指标

**主指标（逐样本人工打，5 点尺）**
- `actionability`：一名值班分析员能否照着 S4 立刻行动
- `sufficiency`：S1–S5 是否足以支撑"这是一类新东西"的判断

**次指标（可自动或半自动统计）**
- 槽覆盖率：五槽齐备率（分母 20）
- abstain 校准：该 abstain 的样本里 abstain 的比例 / 不该 abstain 却 abstain 的比例
- 引用可验证率：所引 ATT&CK 编号在检索证据里确有出处
- 耗时：每样本端到端秒数（含检索与生成）

**判定门（G2，留给下一阶段）**
`A3` 的 `actionability` 比 `max(A0, A1)` 高 **≥1 分（5 点尺）**，且配对检验 `p < 0.05` → 方向 2 正；否则记负，并沿用"LLM 在解释层无增益"的表述。20 条**不足以**过这个门，只用于校准尺子和估方差。

## 4. 打 分 尺（actionability）

| 分 | 锚点 |
|---|---|
| 5 | 按 S4 执行后能直接证实/排除一类新行为，动作含具体对象与时间窗 |
| 4 | 动作可执行，但需分析员补一步（如自行选端口范围） |
| 3 | 动作方向正确但缺对象/窗口，需较多补充 |
| 2 | 只能得到"再收集更多数据"这类泛化建议 |
| 1 | 无可执行动作，或动作与 S1–S2 矛盾 |

## 5. 打 分 尺（sufficiency）

| 分 | 锚点 |
|---|---|
| 5 | S1–S5 自洽、边界清楚，可直接写进工单 |
| 4 | 有 1 处小缺口（如 S2 只有 2 条） |
| 3 | 主要结论成立，但 S3 无依据或 S5 缺失 |
| 2 | 槽位残缺，结论依赖外部知识 |
| 1 | 五槽基本为空或互相矛盾 |

## 6. 标注流程

1. 双标 20 条（`labeler` 记 `L1`/`L2`），**先独立打，再对表**。
2. 算 `actionability` 与 `sufficiency` 的 Cohen's κ。
3. `κ < 0.4` → 改 §4/§5 锚点，重标这 20 条；`κ ≥ 0.4` 才允许扩量。
4. 每扩 50 条抽 10% 复核，防止尺子漂移。

## 7. 标注表字段

`data/annotations/triage_202611.csv`

| 字段 | 说明 |
|---|---|
| `sample_id` | `ds{dataset}_s{split}_{全局下标}`，全局下标由 `recover_global_indices` 反查得出 |
| `source` | `genuine`（`unknown_origin=1`，真实留出类）或 `simulated`（`=2`，分布内已知类改标） |
| `family` | **粗粒度、逐样本不可分**：`heldout_web_or_other`（真实留出类为 Web Attack / Other Attack，见 `split_0/label_mapping.json` 的 10/11）或 `known_relabeled`（模拟未知，原类不可恢复）。不要据此评价"猜中了哪个家族"，标签在开集划分里已被改写为 `unknown` |
| `five_slots` | 五槽内容，格式 `S1=… \|\| S2=… \|\| S3=… \|\| S4=… \|\| S5=…`；**建表阶段留空** |
| `labeler` | `L1` / `L2` |
| `actionability` | 1–5，建表阶段留空 |
| `sufficiency` | 1–5，建表阶段留空 |
| `notes` | 抽样来源与 origin 说明 |

## 8. 数据来源与可复现性

- **参考配置（固定）**：`ood_reports/ood_routing_20261105_205104.json` —— backbone `model_5022`（`A0*`，seed 1011）、`ood_head_97`；该配置是 11-05 那批里 `tpr@fpr=1%` 最高的（0.0878 组内），且 `routing_log` 里被真正判 unknown 的 `simulated` 有 30 条，够抽 10 条。其它报告不够（如 `model_5001` 只有 2 条 simulated），所以不能随便换报告。
- 抽样脚本：`scripts/export_triage_samples.py --routing_report ood_reports/ood_routing_20261105_205104.json --n 20 --seed 42`
- 抽样口径：只取 `routing_log`（516 条）里**真正被判 unknown** 的样本；按 `unknown_origin` 分层各取 10 条（`genuine`/`simulated`）。516 条里 267 条是 `unknown_origin=0`（已知类被误报，不属于 triage 对象），已排除。
- 下标反查自检：脚本用 `recover_global_indices` 反查 516/516 全成功，并校验 `origin=0 ⇔ 标签非 unknown`；不一致即中止（本次通过）。
- 若 `scripts/export_triage_samples.py` 缺失，等价做法是 `scripts/generate_unknown_explanations.py --n_samples 20 --seed 42 --dry_run` 的抽样逻辑，再手工导出（本条为回退说明）
- 复现命令：`/root/miniconda3/bin/python scripts/export_triage_samples.py --routing_report ood_reports/ood_routing_20261105_205104.json --n 20 --seed 42` → 20 行（10 genuine + 10 simulated），已落盘 `data/annotations/triage_202611.csv`

## 9. 证据边界（写进报告时必须带）

- 20 条只用于**校准尺子**，任何"LLM 更好/更差"的结论都不允许基于这 20 条。
- `simulated` 样本的特征仍来自已知类，`A2`/`A3` 可能"认出来"原类，这不等于识别未知；解读时要分开看 `source`。
- `family` 是粗粒度标签（`heldout_web_or_other` / `known_relabeled`），**逐样本家族不可恢复**（开集划分把 Label 改写成 `unknown=10`，npz 与 CSV 都不保留原值），因此不要做"分家族准确率"。
- 不评测检测指标（G1 已终局），不新增检测变体。
- `A1` 的检索库与 `A3` 同一套 CTI 文档，所以 `A3` 相对 `A1` 的增益只可能是"组织与措辞"，不可能是"多知道事实"。
- 留出类为 Web Attack / Other Attack，只有单数据集 `dataset_200`，结论不能外推到其他数据集。
