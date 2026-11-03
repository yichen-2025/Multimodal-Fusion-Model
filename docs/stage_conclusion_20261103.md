# 阶段结论 2026-11-03：A3 对照实验（dataset_200 / split_openset_0）

> 分支 `exp/explain-rag`，基线提交 `4b729d6`；本文不改写 `docs/stage_conclusion_20261027.md`，只追加 A3 对照结论。
> 上游计划：`plan/2026-11-02.md` 阶段 A/B。

## 1. 口径

| 项 | 值 |
|---|---|
| 数据集 | `dataset_200` / `split_openset_0` |
| 训练 / 验证 | 6362 / 按划分文件 |
| 测试 | 2040 = 已知 1655 + 未知 385（unknown = label 10） |
| 闭集 test | 910 条已知样本（混淆矩阵口径） |
| OOD 指标 | `auroc_genuine` / `tpr@fpr=1%` / `unknown_recall_simulated` / `known_accuracy` / `unknown_recall_genuine` |
| 统计 | backbone 级 n=3，mean ± sd(ddof=1)；每个 backbone 另有 head seed 42/43/44 三个头 |
| 训练配置 | A3 与 A0* 系同配置，10 epoch / batch 64 / lr 5e-4 / cosine 距离 / T=0.1 |

## 2. 本阶段做了什么

1. **阶段 A（元数据修复）**：`utils/caliber.py` 新增 `read_model_variant()`；`scripts/run_ood_routing.py` 改为「显式参数优先，否则从模型 `config.txt` 读取」，报告新增 `variant_source`。
2. **回填**：重跑 2000/10、2006/12、2009/13、2013/14 四份误标报告。四份报告指标与旧报告**逐位一致**（例：2000/10 `auroc_genuine=0.950679`、`tpr@fpr1=0.038961`），仅元数据由错误的 `A3` 修正为真实变体。
3. **归档**：6 份误标文件移入 `_archive/ood_reports/`，`_archive/MANIFEST.jsonl` 追加 6 行；`ood_reports/` 中 `dataset_id=200 & backbone_variant="A3"` 的 live 报告数已为 0。其中 2003/11 因 backbone 已被台账判为 invalid（checkpoint 不含 LoRA）不再重跑，直接归档。
4. **阶段 B（A3 对照）**：训练 3 个 A3 backbone（`model_3001/3002/3003`，seed 42/123/456）、9 个 OOD 头（`ood_id 39–47`）、9 份路由报告。

## 3. 闭集结果（910 条已知测试样本）

| 模型 | variant | seed | accuracy | macro-F1 |
|---|---|---|---|---|
| model_3001 | A3 | 42 | 0.9703 | 0.9701 |
| model_3002 | A3 | 123 | 0.9846 | 0.9846 |
| model_3003 | A3 | 456 | 0.9813 | 0.9813 |
| model_2000 | A0*_no_num | 42 | 0.9923 | 0.9923 |
| model_2013 | A0* | 42 | 0.9934 | 0.9934 |

A3 闭集比 A0* 系低约 1–2 个百分点（0.970–0.985 vs 0.992–0.993），未改变「闭集已到天花板」的判断。

## 4. OOD 主表（backbone 级，n=3，ddof=1）

| variant | auroc_genuine | tpr@fpr=1% | unknown_recall_simulated | known_accuracy | unknown_recall_genuine |
|---|---|---|---|---|---|
| **A3**（BERT 文本 + 数值，无 LLM） | **0.9578 ± 0.0143** | **0.0268 ± 0.0221** | **0.0470 ± 0.0255** | **0.9271 ± 0.0091** | **0.9264 ± 0.0231** |
| A0*_no_num（纯 LLM 文本） | 0.9606 ± 0.0088 | 0.0485 ± 0.0091 | 0.0389 ± 0.0035 | 0.9380 ± 0.0062 | 0.9610 ± 0.0338 |
| A0*（LLM 文本 + 数值，concat） | 0.9371 ± 0.0019 | 0.0840 ± 0.0065 | 0.0900 ± 0.0071 | 0.9200 ± 0.0061 | 0.9489 ± 0.0383 |

同 backbone 跨 head seed 的离散度极小（A3 `auroc` head_max_range = 0.0007），组间方差由 backbone seed 主导（3002 明显偏低）。

## 5. 对比表（diff = A3 − 基线，单位 pp）

| 对比 | 指标 | diff | sd_A3 | sd_基线 | max倍数 | pooled倍数 |
|---|---|---|---|---|---|---|
| A3 − A0* | auroc_genuine | **+2.08** | 0.0143 | 0.0019 | 1.45 | 2.03 |
| A3 − A0* | tpr@fpr=1% | **−5.71** | 0.0221 | 0.0065 | 2.59 | 3.51 |
| A3 − A0* | unknown_recall_simulated | −4.29 | 0.0255 | 0.0071 | 1.68 | 2.29 |
| A3 − A0*_no_num | auroc_genuine | −0.28 | 0.0143 | 0.0088 | 0.20 | 0.24 |
| A3 − A0*_no_num | tpr@fpr=1% | −2.16 | 0.0221 | 0.0091 | 1.00 | 1.30 |
| A3 − A0*_no_num | unknown_recall_simulated | +0.82 | 0.0255 | 0.0035 | 0.30 | 0.40 |

### 5.1 配对 bootstrap（B=4000，sample + backbone seed 双重采样）

`scripts/bootstrap_ood_ci.py`，报告级配对重采样，结果见 `results/2026-11-03/bootstrap_ood_ci_with_a3.json`。

| 对比 | 指标 | diff | 样本级 95% CI | 含 seed 95% CI |
|---|---|---|---|---|
| A3 − A0* | auroc_genuine | +2.08pp | [1.16, 2.99] | [0.33, 3.62] |
| A3 − A0* | tpr@fpr=1% | −5.71pp | [−7.50, −3.15] | [−8.55, −2.39] |
| A3 − A0* | unknown_recall_simulated | −4.29pp | [−8.17, −0.77] | [−9.11, 0.00] |
| A3 − A0*_no_num | auroc_genuine | −0.28pp | [−1.41, 0.71] | [−2.24, 1.53] |
| A3 − A0*_no_num | tpr@fpr=1% | −2.16pp | [−4.03, 0.00] | [−5.01, 1.26] |
| A3 − A0*_no_num | unknown_recall_simulated | +0.82pp | [−1.72, 3.25] | [−2.86, 4.58] |

## 6. 判定（按计划 B5 预注册规则）

规则回顾：

| 条件 | 判定 |
|---|---|
| 双口径 diff 均负且两口径 max倍数 > 2 | 主假设成立 |
| 方向相反，或任一 max倍数 ≤ 2 | 无显著差异 |
| 双口径 diff 均正且 max倍数 > 2 | 主假设被否定 |

实际数据：

- **A3 vs A0***：`auroc_genuine` **+2.08pp（A3 更高）**、`tpr@fpr=1%` **−5.71pp（A3 更低）** → **两口径方向相反**。
- **A3 vs A0*_no_num**：`auroc_genuine` −0.28pp（0.20×sd）、`tpr@fpr=1%` −2.16pp（1.00×sd） → **均未达 2× sd**。

**判定：落第二行——无显著差异。** 该判定的适用范围是 **A3 vs A0\*_no_num**（三项指标 CI 全跨零）。**A3 vs A0\* 不适用「无差异」表述**：两组 CI 均不跨零且方向相反，应写作 trade-off（AUROC 更高、低误报召回更低），不可简写为「两组均无显著差异」。

配对 bootstrap 进一步细化了这个判定：

- **A3 与 A0*_no_num**：三项指标差值 95% CI 全部跨零（含 seed 重采样后区间更宽，`auroc` [−2.24, 1.53]），统计上不可区分。
- **A3 与 A0***：`auroc_genuine` **+2.08pp [1.16, 2.99]**、`tpr@fpr=1%` **−5.71pp [−7.50, −3.15]**，两个区间都不跨零但**方向相反**——这是明确的 trade-off，不是「A3 双口径占优」，因此仍不构成主假设成立。

据此确定的论文主张：

> LLM 主干在 OOD 检测上不带来显著增益；LLM 的价值在于未知样本的解释能力，而非分类/检测本身。

具体表述要点：

1. A3（BERT 文本 + 数值）在 `auroc_genuine` 上与纯 LLM 文本（A0*_no_num）基本持平（0.9578 vs 0.9606），甚至高于 LLM+数值 concat（A0*，0.9371）。
2. A3 在 `tpr@fpr=1%` 上弱于两个 A0* 组（0.0268 vs 0.0485 / 0.0840），说明高 AUROC 并不等于低误报率下的召回——**单看 AUROC 会高估 A3**。
3. 因此主贡献应移到「可解释性 + 未知解释」方向（Phase 2/3），而不是「LLM 主干提升分类」。
4. 阶段 A 的元数据修复不改任何指标，只消除「dataset_200 存在 A3 模型」的错误印象。

## 7. 已知限制

1. 仅 3 个 backbone seed，组内方差由单个 seed（3002）主导；结论的置信度受限于 n=3。
2. A3 的 `BertEncoder` 虽被构造但不参与前向（使用 npz 预计算 768 维向量），因此 A3 的「文本编码器」是固定特征，不与 A0* 的 LoRA 微调 LLM 等价——这是设计差异，不是实现缺陷。
3. A3 闭集低 1–2pp，可能来自 10 epoch 下的容量差异，未做额外调参（按红线不调阈值/温度）。
4. 所有指标仍来自同分布随机切分，未做分布外推检验。

## 8. 产物索引

| 类型 | 路径 |
|---|---|
| A3 backbone | `saved_models/model_3001`、`model_3002`、`model_3003` |
| A3 OOD 头 | `saved_ood_heads/ood_head_39` … `ood_head_47` |
| A3 路由报告 | `ood_reports/ood_routing_20261103_094055.json` … `ood_routing_20261103_094213.json`（9 份） |
| 回填报告 | `ood_reports/ood_routing_20261103_092412/092454/092545/092627.json`（4 份） |
| 汇总 | `results/2026-11-03/ood_multiseed_with_a3.json` |
| 归档 | `_archive/ood_reports/`（6 份）+ `_archive/MANIFEST.jsonl`（+6 行） |
