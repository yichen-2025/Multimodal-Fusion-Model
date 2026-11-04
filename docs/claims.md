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

## 3. ATT&CK 映射与检索

| ID | 结论 | 状态 | 证据（数值｜产物） | 边界 |
|---|---|---|---|---|
| C-11 | 检索显著优于 LLM 自选技术 | 已证实 | 同一批 72 条：检索 61/72（0.8472）vs LLM 40/72（0.5556），并集 67/72（0.9306），`both=34` / `rag_rescues=27` / `rag_hurts=6` / `neither=5`，McNemar 精确检验 p=0.000324｜`explanation_results/eval_mapping_hybrid_20261027_094918.json`、`docs/stage_conclusion_20261027.md` §4 | 检索不进 known 判类路径，避免污染闭集指标 |
| C-12 | 检索 84.9% 在更大候选库上依然成立 | **未支持** | 7 条库 0.8494（家族 0.9505）→ 10 条 0.4220（家族 0.9471）→ 22 条 0.3331（家族 0.7490）｜`results/2026-10-28/cti_kb_scale_{a,b,c}.json`、`cti_kb_scale_summary_20261028.json` | 对外必须连知识库规模一起报；不得称"开放候选集上的检索能力" |
| C-13 | LLM 可直接指派 ATT&CK 技术（含 unknown 路径） | **未支持** | unknown 路径 98/100 返回 `null`；模拟未知组 LLM 0/25、检索 top-1 1/25、top-3 21/25｜`docs/stage_conclusion_20261027.md` §5、`explanation_results/unknown_explanations_model2011_ood22_20261028_093615.json` | 展示形态定为"检索 top-3 映射 + LLM 叙述" |

## 4. 解释质量

| ID | 结论 | 状态 | 证据（数值｜产物） | 边界 |
|---|---|---|---|---|
| C-14 | 解释一致性 0.56 反映解释质量差 | **未支持** | 同一份 JSON 换尺子：`strict_v1` 0.56、`family_v2` 0.81，人工抽检 20/20 与 `family_v2` 一致；0.56 主因是词表过窄｜`explanation_results/eval_model2000_20261026_100154.json`（v2/v3 版本）｜`docs/stage_conclusion_20261027.md` §5 | 真正跑偏的是只复述特征、不给类别结论的样本，需单独统计 |
| C-15 | 解释中的数字不编造（`unverified_number_rate ≤ 0.05`） | **未支持** | legacy 0.20 → v2 0.13 → v3 0.11（11/100），关卡 0.05 未达；剩余为 1.5B 模型的真实量级改写（如 `6.329e+07` 写成 `6.329亿`）｜`explanation_results/eval_model2000_20261026_100154_rule_v3.json` | 唯一有效手段是改 prompt，但一致性会从 0.81 掉到 0.70，净负收益，已回退 |
| C-16 | unknown 路径 LLM 解释可满足引用率关卡 | 已证实 | 100 条：`reference_rate` 1.00（≥0.90 PASS）、`unverified_number_rate` 0.09、空解释 0｜`docs/stage_conclusion_20261027.md` §5（C1）、`explanation_results/unknown_explanations_model2011_ood22_20261028_093615.json` | 抽样按 `unknown_origin` 分层；路由回归 `routed` 与基线逐位一致 |
| C-17 | LLM 解释比模板解释更有用（人工可感知价值） | **待检验** | 无人工盲评、无忠实性/反事实检验｜— | 0.81/0.11 属自动指标，不足以支撑"有用" |

## 5. 泛化与工程结论

| ID | 结论 | 状态 | 证据（数值｜产物） | 边界 |
|---|---|---|---|---|
| C-18 | 结论可推广到多未知家族 / 跨数据集 | **待检验** | 当前只留出 1 个 Web Attack 家族，未做 CIC-IDS2017→UNSW-NB15、时间切分、少样本 | — |
| C-19 | 公平编码器对照（≥4 变体 × 3 seeds，控制 batch/epoch/pooling/适配）会改变现有判读 | **待检验** | A3 与 A0\* 的训练/适配尚未完全对齐 | 负结果可能被质疑"没调好" |
| C-20 | 语义原型在 `auroc_simulated` 或 `tpr@fpr=1%` 上带来 ≥2pp 提升 | **待检验** | 尚无原型实验产物 | — |
| C-21 | 成本-精度 Pareto 能决定 LLM 该放哪一层 | **待检验** | 尚无延迟/显存/每千样本成本测量 | — |

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
| 知识库规模敏感性 | `results/2026-10-28/cti_kb_scale_{a,b,c,c_behavior,c_strip}.json` | — |
| 映射对比（McNemar） | `explanation_results/eval_mapping_hybrid_20261027_094918.json` | active |
| 解释一致性（strict/v2/v3） | `explanation_results/eval_model2000_20261026_100154{,_rule_v2,_rule_v3}.json` | active |
| 检索命中率 | `explanation_results/eval_retrieval_20261026_105616.json` | active |
| unknown 路径解释 | `explanation_results/unknown_explanations_model2011_ood22_20261028_093615.json` | active |

## 7. 变更记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-11-03 | v1.0 | 建表：21 条结论（已证实 7 / 未支持 9 / 待检验 5）按闭集、LLM 检测、映射检索、解释质量、泛化五组登记 |

> 新增结论必须先定状态再写数值；状态升级（未支持 → 已证实）需要新的预注册口径与产物，
> 并在此表追加变更行。
