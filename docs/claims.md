# 结论登记（claims.md）

> 本文件是**对外结论的唯一登记处**。写摘要、汇报、论文、周报之前先在这里核对状态；
> 引用任何结论都必须带 `C-xx` 编号与状态，并附产物路径。
> 产物是否可用以 `artifacts/ledger.csv` 为准：只有 `active` 的产物能支撑结论，
> `deprecated` 只能作为历史证据。

## 0. 状态定义与引用规则

| 状态 | 定义 | 允许的表述 |
|---|---|---|
| **已证实** | 有预注册口径、可复现产物，且有统计检验/置信区间支持该方向 | 可直接陈述，但必须同时给出效应量与区间 |
| **未支持** | 检验已做，但方向不支持、或两组统计上不可区分；**不等于"相反结论成立"** | 只能写"不支持 / 不可区分"，禁止反向断言 |
| **待检验** | 尚无预注册口径或产物 | 只能写"待检验"，不得进入摘要与标题 |

补充规则：

- 描述性事实（如"闭集已到天花板"）标 `已证实（描述性）`，不附带统计显著性含义。
- 统一口径：`dataset_200 / split_openset_0`，`test_samples=2040`（已知 1655 + 未知 385 = 真实 222 + 模拟 163）。
  跨数据集、多未知家族的结论必须另立条目，不得沿用本口径的数字。
- 比值型指标一律同时给分子/分母（72 条子集中 1 条 ≈ 1.39pp）。

## 1. 闭集与融合设计

| ID | 结论 | 状态 | 证据（数值｜产物） | 边界 |
|---|---|---|---|---|
| C-01 | 闭集分类已到天花板，4 个 A0\* 变体无可区分差距 | 已证实（描述性） | accuracy 0.9915–0.9944（seed 42/123/456）｜`test_reports/report_*.json`（26 份 active）、`docs/stage_conclusion_20261027.md` §2 | 仅 `dataset_200 / split_0` 闭集口径 |
| C-02 | A3（BERT+数值、无 LLM）闭集比 A0\* 系低 1–2pp | 已证实（描述性） | 0.9703 / 0.9846 / 0.9813 vs 0.9923（A0\*_no_num）、0.9934（A0\*）｜`saved_models/model_3001`–`3003`、`docs/stage_conclusion_20261103.md` §3 | 未调参、未调阈值（按红线） |
| C-03 | 数值模态在 OOD 检测上带来增益 | **未支持** | concat 与纯文本：`auroc_genuine` −2.36pp [−3.65, −1.11]，但 `tpr@fpr=1%` +3.55pp [+0.99, +5.93]、`unknown_recall_simulated` +5.11pp [+1.66, +8.96]，两口径方向相反｜`results/2026-10-28/bootstrap_ood_ci_20261028.json`、`results/2026-10-28/ood_multiseed_summary_20261028.json` | 不得表述为"数值模态无效"，只能说无变体双口径同时占优 |
| C-04 | 纯文本（无数值）在 OOD 上整体更优 | **未支持** | 同上：纯文本仅赢 AUROC 一侧，低 FPR 与模拟未知召回均落后｜`results/2026-10-28/bootstrap_ood_ci_20261028.json` | 与 C-03 互为镜像，两条都只能写"无单一排序结论" |

## 2. LLM 是否提升检测（A3 对照）

| ID | 结论 | 状态 | 证据（数值｜产物） | 边界 |
|---|---|---|---|---|
| C-05 | LLM 主干能提升 OOD 检测 | **未支持** | A3 − A0\*_no_num：`auroc_genuine` −0.28pp [−1.41, 0.71]、`tpr@fpr=1%` −2.16pp [−4.03, 0.00]、`unknown_recall_simulated` +0.82pp [−1.72, 3.25]，三项含 seed CI 全跨零｜`results/2026-11-03/bootstrap_ood_ci_with_a3.json`、`docs/stage_conclusion_20261103.md` §5–§6 | 仅 A3 vs A0\*_no_num 可比；A3 vs A0\* 不适用"无差异"表述 |
| C-06 | A3 与 A0\*（concat）是 trade-off，不是单向占优 | 已证实 | `auroc_genuine` A3 +2.08pp [1.16, 2.99]；`tpr@fpr=1%` A3 −5.71pp [−7.50, −3.15]；`unknown_recall_simulated` A3 −4.29pp [−8.17, −0.77]｜`docs/stage_conclusion_20261103.md` §5、`results/2026-11-03/bootstrap_ood_ci_with_a3.json` | 两组 CI 均不跨零且方向相反 |
| C-07 | 只报 AUROC 会高估 A3 的部署价值 | 已证实（方法学） | 同一批报告：AUROC 上 A3 占优、低 FPR 操作点上 A3 明显落后｜`results/2026-11-03/ood_multiseed_with_a3.json` | 对外必须 AUROC 与 `tpr@fpr=1%` 成对给出 |
| C-08 | 低 FPR 与模拟未知召回是共同短板 | 已证实（描述性） | `tpr@fpr=1%` 0.027–0.084；`unknown_recall_simulated` 0.039–0.127（A0\*_attn 最高 0.1268）｜`docs/stage_conclusion_20261027.md` §3、`docs/stage_conclusion_20261103.md` §4 | 按红线不调 `distance_type` / `temperature` / 阈值来遮掩 |
| C-09 | 混合 `unknown_recall`（真实未知 + 模拟未知）可当作真实 OOD 能力 | **未支持** | 台账中 144 份 OOD 报告因旧划分缺 `unknown_origin` 仍为 `deprecated`；活跃开集划分才可分解 genuine/simulated｜`artifacts/ledger.csv`（`kind=split`、`kind=ood_report`）、`artifacts/README.md` | 引用旧报告必须标注"deprecated，混合口径" |
| C-10 | 同一 backbone 上换 head seed 会明显改变结论 | **未支持** | head 极差：`auroc_genuine` ≤0.00014（≤0.014pp），多数工作点指标极差为 0｜`docs/stage_conclusion_20261027.md` §3.1、`results/2026-10-28/ood_head_seed_summary.csv` | 例外：`A0*_add` 的 `unknown_recall_simulated` head 极差 0.00613，单头数值需带 seed 波动 |
| C-25 | **G1 终局**：LLM 主干（`A0*` 系）相对最佳 no-LLM 变体在 `tpr@fpr=1%` 上无可检测增益 | **未支持** | `best_llm = A0*` 0.0878 ± 0.0079（n=5）vs `best_no_llm = A3` 0.0691 ± 0.0920（n=5）：`A0* − A3` = +1.87pp，配对 t-CI [−8.85, +12.59] 跨零、bootstrap（B=4000，sample+seed）p=0.549；未过“≥2pp 且 CI 不重叠”门槛，也未确认方向相反｜`results/2026-11-03/ood_multiseed_fair_encoder_n5.json`、`bootstrap_ood_ci_fair_encoder_n5.json`、`ood_fair_encoder_table_n5.md` §2 | 仅 `dataset_200 / split_openset_0`、单 Web Attack 家族。**本阶段不再新增检测变体**：`A3_ft`（BERT 联合微调）未参评——训练时的文本特征是离线预算的 `bert` 向量，`BertEncoder.forward` 带 `@torch.no_grad()`，`save_pretrained` 也不落 BERT 权重，加 `trainable` 开关只会得到一个与 `A3` 数值等价的假变体；`A0*_cti` 因 `dataset_200` 无文本列未做 |

## 3. ATT&CK 映射与检索

| ID | 结论 | 状态 | 证据（数值｜产物） | 边界 |
|---|---|---|---|---|
| C-11 | 检索显著优于 LLM 自选技术 | 已证实 | 同一批 72 条：检索 61/72（0.8472）vs LLM 40/72（0.5556），并集 67/72（0.9306），`both=34` / `rag_rescues=27` / `rag_hurts=6` / `neither=5`，McNemar 精确检验 p=0.000324｜`explanation_results/eval_mapping_hybrid_20261027_094918.json`、`docs/stage_conclusion_20261027.md` §4 | 检索不进 known 判类路径，避免污染闭集指标 |
| C-12 | 检索 84.9% 在更大候选库上依然成立 | **未支持** | 7 条库 0.8494（家族 0.9505）→ 10 条 0.4220（家族 0.9471）→ 22 条 0.3331（家族 0.7490）｜`results/2026-10-28/cti_kb_scale_{a,b,c}.json`、`cti_kb_scale_summary_20261028.json` | 对外必须连知识库规模一起报；不得称"开放候选集上的检索能力" |
| C-13 | LLM 可直接指派 ATT&CK 技术（含 unknown 路径） | **未支持** | unknown 路径 98/100 返回 `null`；单步 abstain prompt 后仍 100/100 null 且无理由（`explanation_results/unknown_explanations_abstain_20261103.json`）；两步式 `null_with_reason_rate=1.0` 但 step1 100/100 判证据不足，LLM 映射仍 0/22、0/25（`explanation_results/unknown_explanations_twostep_20261105.json`）；模拟未知组 LLM 0/25、检索 top-1 1/25、top-3 21/25｜`docs/stage_conclusion_20261027.md` §5、`explanation_results/unknown_explanations_model2011_ood22_20261028_093615.json` | 展示形态定为"检索 top-3 映射 + LLM 叙述" |

## 4. 解释质量

| ID | 结论 | 状态 | 证据（数值｜产物） | 边界 |
|---|---|---|---|---|
| C-14 | 解释一致性 0.56 反映解释质量差 | **未支持** | 同一份 JSON 换尺子：`strict_v1` 0.56、`family_v2` 0.81，人工抽检 20/20 与 `family_v2` 一致；0.56 主因是词表过窄｜`explanation_results/eval_model2000_20261026_100154.json`（v2/v3 版本）｜`docs/stage_conclusion_20261027.md` §5 | 真正跑偏的是只复述特征、不给类别结论的样本，需单独统计 |
| C-15 | 解释中的数字不编造（`unverified_number_rate ≤ 0.05`） | **未支持** | legacy 0.20 → v2 0.13 → v3 0.11（11/100），关卡 0.05 未达；剩余为 1.5B 模型的真实量级改写（如 `6.329e+07` 写成 `6.329亿`）｜`explanation_results/eval_model2000_20261026_100154_rule_v3.json` | 唯一有效手段是改 prompt，但一致性会从 0.81 掉到 0.70，净负收益，已回退 |
| C-16 | unknown 路径 LLM 解释可满足引用率关卡 | 已证实 | 100 条：`reference_rate` 1.00（≥0.90 PASS）、`unverified_number_rate` 0.09、空解释 0｜`docs/stage_conclusion_20261027.md` §5（C1）、`explanation_results/unknown_explanations_model2011_ood22_20261028_093615.json` | 抽样按 `unknown_origin` 分层；路由回归 `routed` 与基线逐位一致 |
| C-17 | LLM 解释比模板解释更有用（人工可感知价值） | **未支持** | 20 条 unknown 样本盲标（L1+L2 两轮取均值，`A0` 模板 vs `A3` LLM+RAG）：`actionability` 3.00 ± 0.00 vs 2.925 ± 0.977，配对差 **−0.075**、Wilcoxon `p=0.555`（10 胜 0 平 10 负）；`sufficiency` 3.00 vs 4.50，差 +1.50、`p=0.0001`（20 胜 0 平 0 负）｜`results/2026-11-06/triage/summary_kappa.json`、`blind_pack.csv`、`blind_scores_L1/L2.csv` | 主指标 `actionability` 无差异；`sufficiency` 增益来自 S2/S4/S5 填得更满，**S3（ATT&CK）两臂皆空**（`A3` 20/20 abstain），不含技术定位能力；κ=0.748/0.757 为**重测信度**（单人，非标注者间）；n=20，仅 `dataset_200 / split_openset_0`；详见 **C-26** |

| ID | 结论 | 状态 | 证据（数值｜产物） | 边界 |
|---|---|---|---|---|
| C-26 | **方向 2 终局**：LLM+RAG 解释层相对模板/检索在 triage `actionability` 上无可检测增益 | **未支持** | 四臂中 `A0`/`A3` 已完成人工盲评（`A1` 未标，不影响判定：`max(A0,A1) ≥ A0 = 3.00 > A3 = 2.925`，差值上限 −0.075 < 门槛 +1.00）：`A3 − A0` = **−0.075**、`p=0.555`，未过"≥1 分且 p<0.05"门；κ（`actionability` 0.748 / `sufficiency` 0.757，n=40，重测）达标后才出此判定｜`results/2026-11-06/triage/summary_kappa.json`、`gen_all4.jsonl`（80 行）、`blind_pack.csv` | 仅 `dataset_200 / split_openset_0`、n=20、`qwen2.5-1.5b`、检索 `k=3`。**不再扩量**：扩量只会把 −0.075 估得更准，不会翻正。abstain 退化：`A0` 20/20、`A1` 0/20、`A2` 20/20、`A3` 20/20，除 `A1` 外三臂恒 100%，`abstain` 校准在本批无区分度 |
| C-27 | 检索-only（BM25 + 规则拼装、不开 LLM）的 triage 卡片可作为 unknown 处置交付形态 | **不确定** | 客观面已达标：`cards_all516.csv` 516 行 / `cards_true249.csv` 249 行，五槽非空率 1.00、`S3` 覆盖 1.00、abstain 0/516、单卡 BM25 检索 p50 **0.12 ms** / p95 **0.34 ms**（CPU、无模型加载），`top1` 复算一致 **516/516**；置信 high 81 / med 90 / low 345（真未知 249 条中 high 仅 9）；`--mode arms` 复现 A0 3.00 / A3 2.925、`p=0.5552` 与 C-26 逐位一致｜`results/2026-11-10/triage_delivery/`（`cards_all516.csv`、`cards_true249.csv`、`sample_cards.md`、`metrics.json`、`summary_arms.json`）、`scripts/triage_card.py` | **人评未完成**：`blind_pack_a1a2.csv` 38 行未标，`A1−A3` 的 `actionability` 无法计算，判定门落"不确定"，不写"可交付/不足"。`S3` 覆盖 100% 是结构性的（BM25 恒返回 k=3），**不等于映射正确**；真未知 249 条无 ATT&CK 真值；部署视图 516 条里 267 条是已知类误报（`Bot` 159/159 全被判 unknown），不得合并成"未知召回" |

## 5. 泛化与工程结论

| ID | 结论 | 状态 | 证据（数值｜产物） | 边界 |
|---|---|---|---|---|
| C-18 | 结论可推广到多未知家族 / 跨数据集 | **待检验** | 当前只留出 1 个 Web Attack 家族，未做 CIC-IDS2017→UNSW-NB15、时间切分、少样本 | — |
| C-19 | 公平编码器对照（≥4 变体 × 3 seeds，控制 batch/epoch/pooling/适配）会改变现有判读 | 已证实（本数据/本实现） | 5 变体 backbone 级：`A0*` `tpr@fpr=1%` 0.0878 ± 0.0079（n=5）vs `A3` 0.0691 ± 0.0920（n=5），差 +1.87pp、CI 跨零；`A0*` vs `A3_adapter` +5.61pp [+3.02, +8.20]，但 AUROC −3.52pp [−6.71, −0.32]；G1 判定**不确定**（未过 ≥2pp 且 CI 不重叠）｜`results/2026-11-03/ood_multiseed_fair_encoder_n5.json`、`bootstrap_ood_ci_fair_encoder_n5.json`、`ood_fair_encoder_table_n5.md` | 仅单 Web Attack 家族；`A0*_last`/`A0_frozen` 仍 n=3；判定按 `best_llm − best_no_llm`，`best_no_llm = A3`；G1 终局判定见 **C-25** |
| C-20 | 语义原型在 `auroc_simulated` 或 `tpr@fpr=1%` 上带来 ≥2pp 提升 | **待检验** | 尚无原型实验产物 | — |
| C-21 | 成本-精度 Pareto 能决定 LLM 该放哪一层 | **待检验** | 尚无延迟/显存/每千样本成本测量 | — |
| C-22 | `llm_pooling` 由 mean 改 last 能改善 OOD | **未支持** | `A0*_last − A0*`：AUROC +0.04pp、`tpr@fpr=1%` −1.47pp、recsim +1.23pp，n=3 全跨零；二者统计不可区分｜`results/2026-11-03/ood_multiseed_fair_encoder_n5.json`、`bootstrap_ood_ci_fair_encoder_n5.json` | 只测 `last` 一处改动，未测 attention/加权 pooling |
| C-23 | 冻结 LLM 主体（`A0_frozen`）与 LoRA 是 trade-off | 已证实 | `A0_frozen − A0*`：AUROC +3.47pp [+2.96, +3.99]、`tpr@fpr=1%` −2.68pp [−5.29, −0.08]、recsim −7.57pp [−10.74, −4.39]（n=3，t-CI）｜`results/2026-11-03/ood_fair_encoder_table_n5.md` | n=3，未补 seed；训练成本仅约 `A0*_last` 的 1/3 |
| C-24 | A3 加 `text_adapter` 能提升 OOD | **未支持** | `A3_adapter − A3`：AUROC +0.60pp、`tpr@fpr=1%` −3.74pp、recsim −1.72pp，n=5 全部跨零；闭集 known_acc 0.9403 vs 0.9253｜`results/2026-11-03/ood_fair_encoder_table_n5.md` | adapter 只加在冻结 BERT 之后，未做 joint finetune |
| C-28 | **机制优先的负结果论文：E0–E8 预注册** | **部分证实** | 预注册见 §5.1（2026-11-19 跑数前登记）；结果回填见 §5.2：E0 0.9855、E8.1 R² 0.498/0.691、E8.2 `I`≈0（HGB −0.020）、E8.4 旧产物不可采信（同估计器 Δ≈0）、E3 0.1281（未支持）、E4 Δ=0、E5 0.1462、E6 未达标（均未支持）；E7 描述性；E8.3 文本翻转 2.42%（通过；阳性对照 66.2% 推翻「不读文本」解读，E8.4 仍待检验） | 单数据集（`dataset_200 / split_openset_0`）、单模型（`qwen2.5-1.5b` + `bert`）、**无人类评测**；门槛锁定后不得事后调整 |

### 5.1 C-28 预注册：机制优先的负结果论文（2026-11-19，跑数前登记）

> 依据：`00-必读-执行总纲-基于LLM的恶意流量识别研究-下一步方案.md` v1.1 §5。
> **本节在 E0/E8 跑数前写入。跑完只能按此判「达标 / 未达标」，不得修改门槛迁就结果。**

**统一口径**：`dataset_200 / split_openset_0`；test 2040（已知 1655 + 未知 385 = 真实 222 + 模拟 163）；编码器全程冻结（未在本数据集训练）；距离度量 / α / τ 只在 val 上选，test 只评估一次；少样本每类从 train 均匀抽 k，5 个采样种子，报 mean ± sd。

| 实验 | 指标 | 预注册门槛 | 判读规则 |
|---|---|---|---|
| E0 轻量基线（79 维 → MLP / XGBoost） | 闭集 acc / macro-F1；OOD 用 max-softmax | 描述性，不设门槛（为「LLM 值不值 60×」提供锚点） | 只报数值 |
| E1 闭集多模态基线 | acc / macro-F1 | 复述 0.970–0.993，**不重跑** | 只引用 |
| E2 主干对照 A3 vs A0\* | auroc_genuine / tpr@fpr=1% | 复述 C-25，**不重跑** | 只引用 |
| E3 零样本语义原型（BERT） | acc / macro-F1 | `acc ≥ 0.50` | 未达标 → `未支持`，停零样本路线 |
| E4 少样本 4 来源 × k∈{0,1,5,10,20}（BERT，5 seed） | acc 曲线 | `P_mix − P_shot ≥ 3pp @ k≤5` 且配对 bootstrap 95% CI 不跨零 | 达标 → `已证实` |
| E5 同上（Qwen 1536） | 同上 | 同 E4；无 GPU 窗口则降级为 500 条子集或仅 BERT 空间（摘要注明「描述由 LLM 生成、由 BERT 编码」） | 未完成不影响论文成立 |
| E6 开集（零样本 / 语义热启动 / 随机初始化） | AUROC、tpr@fpr=1% | `tpr@fpr=1%` +2pp 且 bootstrap CI（B=4000）不跨零 | 未达标只报描述性 |
| E7 消融（描述来源 / α / 距离 / 原型条数） | — | 描述性，不设门槛 | — |
| E8 模态信息审计 | R²、`I(text;y\|features)`、扰动敏感率 | `I ≈ 0` 且扰动敏感率 < 0.2 | 达标 → 「文本无独立信息」（`已证实`） |

**判读三态（互斥）**：达标 → `已证实`，摘要可出现，须附效应量与区间；未达标 → `未支持`，只写「不支持 / 不可区分」，**禁止反向断言**；没跑 → `待检验`，不得进摘要。

**边界（写作时不得删除）**：

1. 单数据集 / 单平台（CSE-CIC-IDS2017 `dataset_200`），**不得称「跨数据集泛化」**；
2. 单模型（`qwen2.5-1.5b` 本地权重 + `bert`），**不得外推到 GPT-4 级或 API 模型**；
3. **无人类评测**：禁止任何 `actionability` / `sufficiency` 的正向表述；
4. E1/E2 只复述，沿用 C-25「本阶段不再新增检测变体」。

**降级分支（按顺序触发）**：E8 判「文本无独立信息」→ 第 5 章定稿、第 6 章降级为「边界内的尝试」，不再追加变体；E3 `acc < 0.30` → 停零样本路线，主线切 `P_mix`（k=1/5）；GPU 窗口始终拿不到 → E5 用 500 条子集或只做 BERT 空间；E6 无增益 → 保留描述性表格，不写「改善」。

### 5.2 C-28 结果回填（2026-11-20，按 5.1 门槛判读，门槛未改）

| 实验 | 预注册门槛 | 实测 | 判定 | 产物 |
|---|---|---|---|---|
| E0 轻量基线 | 描述性 | 79 维 MLP：acc **0.9855**、macro-F1 **0.9858**；OOD AUROC_genuine 0.8323、`tpr@fpr=1%` 0.0450 | 只报数值 | `results/2026-11-19/e0_baseline/metrics.json` |
| E8.1 文本↔特征可预测性 | 描述性 | 79→768：R² **0.498**（方差加权）；768→79：R² **0.691** | 只报数值（文本高度冗余） | `results/2026-11-19/e8_audit/metrics.json` |
| E8.2 条件互信息 | `I ≈ 0` | logreg：`I(text;y\|features)` = 0.121 nats CI95 [0.077, 0.180]；HGB 复核：**−0.020** nats CI95 [−0.041, −0.004] | **I 项通过**（线性结果判估计器伪影，无正信息增量） | `results/2026-11-19/e8_audit/metrics.json`、`e8_audit_hgb/metrics.json` |
| E8.3 反事实扰动 | 文本侧翻转率 < 0.20 | A0*（`saved_models/model_2013`）原 acc **0.9939**；同类 donor 替换 20 个 `TEXT_FEATURES`：文本侧 **2.42%** CI95 [1.69, 3.14]，特征侧 **3.38%** CI95 [2.54, 4.23]；两侧均不变 **94.4%**；差值 CI [−0.18pp, +2.00pp] 跨零 | **门槛通过、解读修正**：同类内扰动不改变判定；阳性对照（跨类 66.2%）证明文本分支在读内容 → 不支持「模型不读文本」，文本无额外信息由 E8.1/E8.2 承担 | `results/2026-11-20/e83_perturbation/metrics.json` |
| E8.3 阳性对照（跨类 donor） | 描述性（不入门槛） | 文本侧翻转 **66.2%** CI95 [63.9, 68.3]、特征侧 **10.1%** CI95 [8.6, 11.5]，特征−文本 CI [−58.9pp, −53.3pp] | **扰动有效**：文本分支对内容高度敏感，E8.3 原「模型不读文本」解读不成立 | `results/2026-11-20/e83_perturbation_cross/metrics.json` |
| E8.4 去文本消融 | 描述性（不设门槛） | 旧产物同 seed 配对 11 对：Δacc = **+20.4pp**（sd 7.4，CI95 [19.8, 21.1]）；但 no_text 检查点闭集仅 0.54–0.71（<< E0 0.9855），属 2026-09/10 旧批次，跨批次混淆。同估计器去文本对照：logreg 0.9674→0.9873（+2.0pp）、HGB 0.9958→0.9897（−0.6pp） | **待检验**（旧产物不可采信；同估计器对照 Δ≈0，方向与 H1 一致） | `results/2026-11-20/e84_text_ablation/metrics.json` |
| E8 检查1 确定性复核 | 描述性 | 逐条字符串精确比对：**9362/9362 一致**（train 6362 / val 960 / test 2040），非法标记 0；文本按 `:.4g` 格式化，为**有损**编码（test 2040 条仅 1872 种文本，碰撞 8.24%；train 11.30%） | 文本＝f(20 字段)，**R≈0 属构造性必然**；文本信息量严格少于 20 字段 | `results/2026-11-20/e8_check1_determinism/metrics.json` |
| E3 零样本语义原型（BERT） | acc ≥ 0.50 | test acc **0.1281**、macro-F1 0.050（距离 cosine，val 选） | **未支持**（未达标） | `results/2026-11-20/e3_e4_bert/metrics.json`、`results/2026-11-20/prototypes/prototypes.json`、`emb/bert_protos.npz` |
| E4 少样本 4 来源 × k × 5 seed（BERT） | `P_mix − P_shot ≥ 3pp @ k≤5` 且 CI 不跨零 | P_shot：k=1 **0.4481**、k=5 **0.5078**、k=10 **0.5190**、k=20 **0.5175**；P_llm 0.1281；P_rand 0.09–0.12；val 选 α*=0 → Δ = 0.0000，CI95 [0, 0] | **未支持**（语义原型对少样本无增益） | `results/2026-11-20/e3_e4_bert/metrics.json`、`curve.csv` |
| E5 少样本（Qwen 空间，1536） | 同 E4 | 零样本 **0.1462**；P_shot：k=1 **0.1782**、k=5 **0.1861**、k=10 **0.2268**、k=20 **0.2329**；α* 贡献 k=1 +0.44pp（CI[−0.64,+1.49]）、k=5 +0.12pp、k≥10 为 0 | **未支持**（负结论在 Qwen 空间复现，且弱于 BERT） | `results/2026-11-20/e5_qwen/metrics.json`、`emb/qwen_*.npz|npy` |
| E6 开集三组（原型距离） | `tpr@fpr=1%` +2pp 且 CI 不跨零 | BERT：AUROC 0.7407→**0.7584**，`tpr@fpr=1%` 0.0000→**0.0045**（CI[0.00,1.35]），P_rand 3.6% 更高；Qwen：AUROC 0.6139→**0.4643**，`tpr` 均为 0 | **未支持**（语义热启动不改善开集；距离尺度不可比） | `results/2026-11-20/e6_bert/metrics.json`、`e6_qwen/metrics.json` |
| E7 原型构造消融 | 描述性 | 零样本 acc：BERT 0.0961–0.1281、Qwen 0.1027–0.1462；描述子集 1/2/3 条、原型 1/2/3 条、cosine/euclidean 全接近随机 | 只报数值（**描述来源不是瓶颈**） | `results/2026-11-20/e7_ablation/{bert,qwen}.json` |

**回填说明**：E3/E4/E5/E6 全部未达标 → 第 6 章按「边界内的尝试（负）」定稿，语义原型不进入主线；E7 表明描述来源不是瓶颈。E8.3 完成（文本翻转 2.42%，门槛通过）→ E8 整体记「部分证实」（E8.4 仍缺可采信产物）。**检查1 完成（9362/9362 一致、`:.4g` 有损编码）→ 文本＝f(20 字段) 确认，R≈0 属构造性必然，「全 79 字段文本正例」由可选升为必需（判据刻度）。** G1/G2 于 2026-11-20 在 GPU 窗口完成（Qwen2.5-1.5B，4000 条 + 30 条描述，< 5 min）。

### 5.3 C-29 预注册与回填：E9 容量阶梯（2026-11-20）

> 门槛在《执行总纲》v1.2（2026-11-19）§5 中已固定，本条目为**补登**，非事后设定。

**门槛**：`tpr@fpr=1%` 较 BERT **+2pp** 且 bootstrap CI 不跨零 → 「容量有效」；否则「容量无效」。闭集 acc 不作判据。

**结果**（Qwen2.5-7B，3584 维，免训练原型/NCM 同口径）：
- 零样本闭集 **0.1130**；P_shot：k=1 **0.1960**、k=5 **0.1908**、k=10 **0.2028**、k=20 **0.2202**
- P_mix 最佳 k=10、α=0.25：**+0.88pp** CI95 [0.30, 1.44]（< 3pp）
- 开集：AUROC **0.4767**、`tpr@fpr=1%` **0.0**；三档 AUROC **0.741(BERT) → 0.614(1.5B) → 0.477(7B)**
- 判读：Δtpr = 0 → **容量无效**；且随规模单调下降，与 H2（信息上界）一致

**边界**：只覆盖 BERT-base → Qwen2.5-7B；不得外推到 API 模型或更大规模。
**产物**：`results/2026-11-20/e9_qwen7b/metrics.json`、`emb/qwen7b_protos.npy`、`emb/qwen7b_texts.npz`

## 6. 产物索引

| 用途 | 路径 | 台账状态 |
|---|---|---|
| 台账（产物有效性唯一判据） | `artifacts/ledger.csv` | active（本文件引用基准） |
| A3 对照阶段结论 | `docs/stage_conclusion_20261103.md` | 文档 |
| OOD 三 seed 阶段结论 | `docs/stage_conclusion_20261027.md` | 文档 |
| A3 backbone | `saved_models/model_3001`–`model_3003` | active |
| A3 OOD 头 | `saved_ood_heads/ood_head_39`–`ood_head_47` | active |
| A3 路由报告（9 份） | `ood_reports/ood_routing_20261103_0940*`–`0942*` | active |
| A0\* 路由报告 | `ood_reports/ood_routing_*.json` | 明细见台账 |
| A3 汇总 / bootstrap | `results/2026-11-03/ood_multiseed_with_a3.json`、`bootstrap_ood_ci_with_a3.json` | — |
| 2026-10-28 汇总 / bootstrap | `results/2026-10-28/ood_multiseed_summary_20261028.json`、`bootstrap_ood_ci_20261028.json` | — |
| 公平编码器对照（n=5 补 seed） | `results/2026-11-03/ood_multiseed_fair_encoder_n5.json`、`bootstrap_ood_ci_fair_encoder_n5.json`、`ood_fair_encoder_table_n5.md` | — |
| 知识库规模敏感性 | `results/2026-10-28/cti_kb_scale_{a,b,c,c_behavior,c_strip}.json` | — |
| 映射对比（McNemar） | `explanation_results/eval_mapping_hybrid_20261027_094918.json` | active |
| 解释一致性（strict/v2/v3） | `explanation_results/eval_model2000_20261026_100154{,_rule_v2,_rule_v3}.json` | active |
| 检索命中率 | `explanation_results/eval_retrieval_20261026_105616.json` | active |
| unknown 路径解释 | `explanation_results/unknown_explanations_model2011_ood22_20261028_093615.json` | active |
| 检索-only 交付卡片（C-27） | `results/2026-11-10/triage_delivery/`（`cards_all516.csv`、`cards_true249.csv`、`sample_cards.md`、`metrics.json`、`summary_arms.json`）、`scripts/triage_card.py` | — |

## 7. 变更记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-11-03 | v1.0 | 建表：21 条结论（已证实 7 / 未支持 9 / 待检验 5）按闭集、LLM 检测、映射检索、解释质量、泛化五组登记 |
| 2026-11-05 | v1.1 | 公平编码器对照收口：C-19 由“待检验”升为“已证实（本数据/本实现）”，G1 判“不确定”；新增 C-22（last pooling 未支持）、C-23（A0_frozen trade-off 已证实）、C-24（text_adapter 未支持）；C-13 补两步式证据 |
| 2026-11-10 | v1.3 | 方向 2（unknown triage）收口：新增 C-26（四臂 `actionability` 无可检测增益，`A3−A0` −0.075、p=0.555）；C-17 由 `待检验` 改为 `未支持`（首个人工盲评证据，κ 0.748/0.757 为重测信度）；更正 `eval_triage.py` 的 κ 标签（函数实为不加权 Cohen's κ，原打印误写"线性加权"） |
| 2026-11-06 | v1.2 | G1 **终局判定**固化为 C-25（`未支持`：+1.87pp、t-CI [−8.85, +12.59] 跨零，未过 ≥2pp 门槛）；C-19 边界加指向 C-25；明确本阶段不再新增检测变体，`A3_ft`（文本特征离线预算，加开关只会得到与 A3 等价的假变体）与 `A0*_cti`（无文本列）不参评 |
| 2026-11-10 | v1.4 | 检索-only 交付形态收口：新增 C-27（**不确定**——客观指标全达标，`A1−A3` 人评 38 行未标，判定门落"不确定"）；新增 `scripts/triage_card.py` 与 `results/2026-11-10/triage_delivery/`；`eval_triage.py` 增 `--mode arms`（单轮臂间比较，**不输出 κ**，A1/A2 缺分写 `missing` 不插补）；补 C-26 所在表头（原 C-26 为缺表头的孤立行） |
| 2026-11-19 | v1.5 | **预注册 C-28**（E0–E8 口径、门槛、判读规则与边界，依据执行总纲 v1.1 §5；跑数前登记）；新增 §5.1 预注册细则；边界明写单数据集 / 单 1.5B 模型 / 无人类评测 |

| 2026-11-20 | v1.6 | C-28 结果回填（新增 §5.2）：E0/E8.1/E8.2 完成；E8.4 判「待检验」（旧 no_text 检查点跨批次混淆，同估计器对照 Δ≈0）；E3 acc 0.1281 **未支持**、E4 Δ=0 **未支持**（门槛未改）；新增 `scripts/build_semantic_prototypes.py`、`scripts/eval_fewshot_prototypes.py`、`scripts/audit_modality_info.py` 的 `e84` 分支 |
| 2026-11-20 | v1.7 | E8.3 结果回填：A0*(model_2013) 文本侧翻转 2.42% CI[1.69,3.14]、特征侧 3.38% CI[2.54,4.23]（同类 donor、20 字段对齐），门槛 <0.20 **通过**；E8 整体判「部分证实」（E8.4 旧产物仍不可采信）；C-28 状态 `待检验`→**部分证实**；新增 `scripts/e83_perturbation.py`（总纲 v1.3 G4） |
| 2026-11-20 | v1.8 | E8.3 阳性对照（跨类 donor）回填：文本侧 66.2% CI[63.9,68.3] vs 特征侧 10.1%；修正 E8.3 判读为「门槛通过但解读需修正」——不支持「模型不读文本」，文本无额外信息仍由 E8.1/E8.2 承担；`scripts/e83_perturbation.py` 增 `--donor {same,cross}` |
| 2026-11-20 | v1.9 | E9 容量阶梯完成并回填（§5.3 C-29）：Qwen2.5-7B 零样本闭集 0.1130、开集 AUROC 0.4767 / tpr@fpr=1% 0.0（BERT 0.741 / 1.5B 0.614 / 7B 0.477）→ **容量无效**；`scripts/cache_text_embeddings.py` 支持任意维度编码器（3584 维）；新增 `emb/qwen7b_*`、`results/2026-11-20/e9_qwen7b/` |
| 2026-11-20 | v1.9 | §5.2 新增「E8 检查1 确定性复核」行：9362/9362 逐条一致、`:.4g` 有损编码（test 碰撞 8.24%）→ 文本＝f(20 字段)，R≈0 属构造性必然；新增 `scripts/check_text_determinism.py` 与该产物；E8.4 产物加机器可读标记 `trustworthy:false / usable_for_claim:false / verdict:待检验`；`e83_perturbation.py` 的 gate 改为按 donor 角色输出（阳性对照标 `applicable:false`，不再出现误导性 `pass:false`）；两份 e83 产物按新 schema 重跑，数值逐位复现 |

> 新增结论必须先定状态再写数值；状态升级（未支持 → 已证实）需要新的预注册口径与产物，
> 并在此表追加变更行。
