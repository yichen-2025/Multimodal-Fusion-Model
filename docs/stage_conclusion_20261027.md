# 阶段结论：解释质量补强 + OOD 三 seed（2026-10-27）

> 口径：`dataset_200` / `split_openset_0`，`test_samples=2040`，`metric_scope=genuine+simulated`。
> 代码锚点：`2b2bb8b`（A 阶段脚本）+ `773ad56`（台账）+ 本次文档提交。
> 所有数字均可由第 6 节命令在云端仓库复现（`run.py` 包装，`/root/miniconda3/bin/python`）。

---

## 1. 结论摘要

| 问题 | 结论 | 关键数字 |
|---|---|---|
| 数值模态是否带来增益 | 无变体在 AUROC 与低 FPR 两个口径上同时优于纯文本，结论维持"无增益"，但也不宣称纯文本更优 | 闭集 concat +0.24pp（配对 t≈1.9, df=2，不显著）；OOD AUROC concat 0.9370±0.0020 vs 纯文本 0.9606±0.0087，而 `tpr@fpr=1%` 0.0840 vs 0.0485 方向相反 |
| MITRE 映射该用 LLM 还是检索 | 检索显著更优，LLM 只负责叙述 | 同一批 72 条：LLM 40/72、检索 61/72、并集 67/72；McNemar 精确检验 p=0.000324 |
| 解释一致性 0.56 是否等于质量差 | 否，原规则是词面误杀 | 家族级口径 0.81，strict 口径 0.56，人工抽检 20 条一致 |
| 解释中的数字是否编造 | 校验器侧误判已基本修完，剩余确为模型量级改写 | 未验证 0.20→0.13→0.11（legacy/v2/v3 三套口径）；A3 关卡 0.05 未达 |
| unknown 路径解释该怎么做 | LLM 只负责叙述，技术映射取检索 top-3，不依赖模型自选编号 | 100 条引用率 1.00（关卡 ≥0.90）；模型 98/100 返回 null，检索 top-3 命中 21/25（模拟未知）与 20/22（已知类误判） |

---

## 2. 闭集三 seed（沿用，未改动）

| 变体 | seed 42 | 123 | 456 | mean ± std |
|---|---|---|---|---|
| `A0*_no_num` | 0.9940 | 0.9915 | 0.9903 | 0.9919 ± 0.0018 |
| `A0*`(concat) | 0.9940 | 0.9946 | 0.9946 | 0.9944 ± 0.0003 |
| `A0*_add` | 0.9946 | 0.9940 | 0.9903 | 0.9930 ± 0.0023 |
| `A0*_attn` | 0.9927 | 0.9909 | 0.9909 | 0.9915 ± 0.0010 |

## 3. OOD 三 seed（本次补齐）

每个变体 3 个 backbone seed，每头独立训练（head seed 固定 42，`cosine` + `temperature=0.1`）。数值取各自 `ood_routing` 报告的 `routed.*`；`±` 后为 **样本标准差（ddof=1, n=3）**，不是总体标准差：

| 变体 | head 与 backbone | `auroc_genuine` 三个 seed | mean ± sd | `tpr@fpr=1%` mean ± sd | `unknown_recall_simulated` mean ± sd | `known_accuracy` mean |
|---|---|---|---|---|---|---|
| `A0*_no_num` | 10→2000, 15→2001, 16→2002 | 0.9507 / 0.9671 / 0.9641 | **0.9606 ± 0.0087** | 0.0485 ± 0.0091 | 0.0388 ± 0.0035 | 0.9380 |
| `A0*`(concat) | 14→2013, 17→2004, 18→2005 | 0.9370 / 0.9390 / 0.9351 | **0.9370 ± 0.0020** | 0.0840 ± 0.0065 | 0.0900 ± 0.0071 | 0.9200 |
| `A0*_add` | 12→2006, 19→2007, 20→2012 | 0.9360 / 0.9668 / 0.9797 | **0.9608 ± 0.0225** | 0.0840 ± 0.0065 | 0.1063 ± 0.0522 | 0.9074 |
| `A0*_attn` | 13→2009, 21→2010, 22→2011 | 0.9372 / 0.9218 / 0.9139 | **0.9243 ± 0.0118** | 0.0580 ± 0.0084 | 0.1268 ± 0.0465 | 0.8673 |

**解读（两个口径方向相反，必须同时报，不能只报 AUROC）**：
- AUROC 口径：纯文本 0.9606 高于 concat 0.9370（差 2.35pp，约为 concat 组内 sd 的 12 倍），与闭集"无增益"方向一致；add 均值持平但 sd 大 2.6 倍；attn 最低。
- 低 FPR 操作点口径：`tpr@fpr=1%` 与 `unknown_recall_simulated` 上 concat（0.0840 / 0.0900）反而高于纯文本（0.0485 / 0.0388），差 3.5pp / 5.1pp，约为组内 sd 的 5–7 倍。路由实际用的是阈值判定，这一口径更接近部署行为。
- 合并结论：**没有任何变体在两个口径上同时优于纯文本**；既不能宣称"数值模态有增益"，也不能宣称"纯文本更优"。concat 的 `known_accuracy`（0.9200）同样低于纯文本（0.9380）。
**口径提醒**：sd 只反映 backbone seed 方差（head seed 固定 42），不能当作全流程方差；跨变体比较时两组 sd 不等价，差值判读需谨慎。`A0*_add` 的 seed 456 用 2012（2008 已 invalid 归档）。

## 4. 映射路径对比（A1）

同一份解释结果 `explanations_model2000_20261026_100154.json` 的 72 条攻击样本（`true_mitre_ids` 非空）：

| 路径 | 命中 | 速率 |
|---|---|---|
| LLM 自选 | 40/72 | 0.5556 |
| 检索 top-1 | 61/72 | 0.8472 |
| 并集 | 67/72 | 0.9306 |

- `both_hit=34`、`rag_rescues=27`、`rag_hurts=6`、`neither=5`，不一致条数 33；
- **McNemar 精确检验 p=0.000324（双侧）**，检索优于 LLM 显著；
- 分类别看：LLM 在 PortScan/FTP-Patator/SSH-Patator/Bot 上完全失效（7、8、9 类各 0/8），检索在 7/8 类 8/8、9 类 5/8；检索唯一明显劣势是 GoldenEye（4/8 vs LLM 8/8）。
- 结论：映射层应走检索，LLM 只写叙述；**检索不进入 known 判类路径**，避免污染闭集指标。

> 更正：早期汇报中的 `28/7`、`union=68/72` 为笔误，产物与逐条明细均为 `27/6`、`67/72`（`eval_mapping_hybrid_20261027_094918.json`）。

## 5. 解释质量（A2/A3）

**A2 一致性（同一份 JSON，只换尺子）**

| 口径 | rate | 说明 |
|---|---|---|
| `strict_v1` | 0.56 | 要求点出具体类别名，历史可比 |
| `family_v2` | **0.81** | 允许家族级表述（DoS 洪水 / 扫描 / 暴力破解 / 否定式） |

人工抽检 20 条：新规则与人工判定 20/20 一致。原 0.56 的主因是词面表过窄（class 2 只认 `goldeneye`、class 9 只认 `bot/僵尸`），不是解释整体跑偏；真正跑偏的是只复述特征、不给类别结论的样本（如 154/358/384/670/699/757/882/953）。

**A3 数字引用（未达关卡，记录在案）**

| 口径 | reference_rate | unverified_number_rate | 未验证样本 |
|---|---|---|---|
| legacy | 0.98 | 0.20 | 20/100 |
| v2（剔技术编号 + 科学计数法归一 + 亿/万展开） | 0.98 | 0.13 | 13/100 |
| **v3（v2 + 按 token 归组：数量级只认归一值、物理单位原值与归一值都认）** | 0.98 | **0.11** | 11/100 |
| prompt 实验（禁止单位换算，已回退） | 1.00 | 0.08 | 8/100 |

v3 只修"换算正确但校验器只认字面"的误判（`3.272e+06` 微秒写成 `3.272 秒`、`7.309e+06` 字节/秒写成 `7.309M字节/秒`），救回 5 个 token、2 个样本；剩余 22 个 token 是**真量级改写**：`6.329e+07` 写成 `6.329亿`（12 个 token 中的主体）、`1.84e+06` 微秒写成 `1.84万微秒`、`2.03e+04` 包/秒写成 `2.03K包/秒`、`5.602e+06` 微秒写成 `5分钟`；另有 2 个取整/区间表述（`64-122字节` 取整成 64 对原文 `64.25`）和 1 个单位语义错误（`6.378 Mbps` 对原文 `6.378e+04` 字节/秒）。
**关卡 0.05 未达成**：v3 已把校验器侧的可修误判压到 5 个 token，剩余项全部落在 1.5B 模型的量级改写能力上；唯一有效手段是改 prompt，但实测一致性从 0.81 掉到 0.70、也只能到 0.08，净负收益，已回退（prompt 实验产物在 `_archive/`）。

**C1 unknown 路径接入 LLM 解释（100 条，seed=42，路由报告 `ood_routing_20261027_084942`）**

| 项 | 数字 | 关卡 |
|---|---|---|
| `reference_rate`（v3） | 1.00（100/100） | ≥0.90 **PASS** |
| `unverified_number_rate`（v3） | 0.09（9/100） | 记录 |
| 空解释 | 0 | — |
| 路由回归 | `routed` 与 `a3_baseline` 逐位一致，`routing_log` 458=458 | **PASS** |

抽样按 `unknown_origin` 分三层：被误判为未知的已知类 25（总体 226）、真实未知 50（206）、模拟未知 25（26）。未验证率分组为 0.12 / 0.08 / 0.08，说明差异来自样本类型而非提示词泄漏。

**口径修正**：`routing_log` 的 `sample_idx` 是 batch 内下标、不是全局下标，必须用 `ood_score` 与 `all_ood_scores` 做 float32 精确匹配反查全局下标（458/458 命中，行为词元复核 20/20 一致）。直接按 `sample_idx` 索引 CSV 会把 routed 组成误判成"全部为已知类"，C1 抽样与命中率都会失效。

**映射结果：LLM 拒绝指派技术，可信映射来自检索**

| 分组 | LLM 映射 | 检索 top-1 | 检索 top-3 |
|---|---|---|---|
| 模拟未知（真值为合并类 3 个技术） | 0/25 | 1/25 | 21/25 |
| 已知类被误判为未知（真值为原始类） | 0/22 | 13/22 | 20/22 |

模型对 98/100 明确返回 `null`，理由一致：样本既被判为未知，就不应再指派某个已知技术。真实未知组按红线 #6 不计算命中。因此 unknown 路径的展示形态定为"检索 top-3 + LLM 解释文本"：模拟未知组 top-1 只有 4%（真值是 Web 暴力破解集合，BM25 多召回 DoS 类），靠 top-3 才到 84%；已知类误判组 top-1 已有 59%。

## 6. 已知短板（不通过调阈值掩盖）

- `unknown_recall_simulated=0.0859`、`tpr_at_fpr_1=0.0909`：模拟 unknown 与低 FPR 区间是短板，按红线不调 `distance_type`/`temperature`/`ood_threshold`。
- unknown 路径的映射仍取检索 top-3 而非 LLM 自选：C1 已验证 LLM 能生成合规解释（引用率 1.00），但模型对 98/100 不给技术编号，故形成"检索映射 + LLM 叙述"的混合形态；模板解释保留为兜底。
- `unverified_number_rate=0.11`（v3 口径，v2 为 0.13）高于 0.05 关卡，属模型能力上限。
- OOD 的 AUROC 与低 FPR 操作点结论相反（见第 3 节），因此 OOD 侧只保留"无变体双口径优于纯文本"的表述，不给单一排序结论。
- 72 条子集方差大：一条样本≈1.39pp，所有比例须同时报分子/分母。

## 7. 复现命令

```bash
# R1 映射对比（含 McNemar）
/root/miniconda3/bin/python run.py /root/miniconda3/bin/python scripts/eval_mapping_hybrid.py \
  --result explanation_results/explanations_model2000_20261026_100154.json --dataset_id 200 --split_id 0

# R2 解释质量（一致性 + 引用率；一次输出 v3/v2/legacy 三套口径）
/root/miniconda3/bin/python run.py /root/miniconda3/bin/python scripts/eval_explanations.py \
  --result explanation_results/explanations_model2000_20261026_100154.json \
  --output explanation_results/eval_model2000_20261026_100154_rule_v3.json

# R3 检索命中率
/root/miniconda3/bin/python run.py /root/miniconda3/bin/python scripts/eval_retrieval.py \
  --backbone_model_id 2000 --dataset_id 200 --split_id 0

# R4 缺失的 OOD 头（若 head 15..22 不存在）
/root/miniconda3/bin/python run.py /root/miniconda3/bin/python scripts/train_ood_head.py \
  --model_id 2001 --ood_id 15 --dataset_id 200 --split_id 0 \
  --variant 'A0*_no_num' --distance_type cosine --temperature 0.1 --epochs 10 --seed 42
/root/miniconda3/bin/python run.py /root/miniconda3/bin/python scripts/run_ood_routing.py \
  --backbone_model_id 2001 --ood_id 15 --dataset_id 200 --split_id 0 --backbone_variant 'A0*_no_num'

# R6 unknown 路径 LLM 解释（C1）
/root/miniconda3/bin/python run.py /root/miniconda3/bin/python scripts/generate_unknown_explanations.py \
  --routing_report ood_reports/ood_routing_20261027_084942.json --n_samples 100 --seed 42

# R5 口径与台账
/root/miniconda3/bin/python run.py /root/miniconda3/bin/python scripts/audit_artifacts.py --check
/root/miniconda3/bin/python run.py /root/miniconda3/bin/python scripts/verify_caliber.py
```

## 8. 产物索引

| 产物 | 路径 |
|---|---|
| 解释原文（A1/A2/A3 输入） | `explanation_results/explanations_model2000_20261026_100154.json` |
| 映射对比（含 p 值） | `explanation_results/eval_mapping_hybrid_20261027_094918.json` |
| 解释质量评估（v3 主口径，内含 v2/legacy 对照） | `explanation_results/eval_model2000_20261026_100154_rule_v3.json` |
| 解释质量评估（v2，历史对照） | `explanation_results/eval_model2000_20261026_100154_rule_v2.json` |
| 检索评估 | `explanation_results/eval_retrieval_20261026_105616.json` |
| unknown 路径 LLM 解释（C1，active） | `explanation_results/unknown_explanations_model2011_ood22_20261028_093615.json` |
| OOD 路由报告 | `ood_reports/ood_routing_20261027_0829..0849.json`（8 份） |
| OOD 头 | `saved_ood_heads/ood_head_15..22` |
| 台账 | `artifacts/ledger.csv` |
| 归档清单 | `_archive/MANIFEST.jsonl` |

