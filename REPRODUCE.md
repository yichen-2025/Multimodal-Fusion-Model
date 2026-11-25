# 复现指南（Stage 1 · claims v2.7）

本指南给出**四条主结论的可执行复现步骤**，并标明每条结论能达到的复现等级。所有命令都在仓库根目录执行。

> 冻结基线：分支 `exp/explain-rag`，claims `v2.7`（tag `claims-v2.7-frozen`）。
> 复现时**不要覆盖** `results/2026-*` 下的任何文件，一律输出到新目录（如 `results/2026-11-25/<exp>/`）。

## 0. 复现等级一览

| 结论 | 内容 | 等级 | 需要 GPU | 本次是否实测 |
|---|---|---|---|---|
| C2 | 文本独立增量 ≤ 约 6%（高度冗余） | **A 精确**（数值逐位一致） | 否 | ✅ 已实测，diff = 0.0 |
| C3 | 语义原型在零样本/少样本上未达门槛 | **A 精确** | 否 | ✅ BERT、Qwen1.5B 已实测，diff = 0.0（Qwen7B 未跑） |
| C1 | LLM 编码器相对 BERT 无 OOD 增益 | **B 近似**（同 seed 集合下点位一致；CI 由 5 个 seed 决定） | 否（R1）；重训需 GPU（R2） | ✅ 配对差已复核 |
| C4 | 成本-精度前沿在 79 维 MLP 饱和 | **C 仅比值**（绝对延迟依赖硬件） | 是 | ⛔ 本机无 GPU，未实测 |

等级定义：
- **A**：同一数据、同一环境、同 seed → 指标逐位一致（容差 1e-9）。
- **B**：同 seed 集合 → 点位一致；CI/显著性只在同一 seed 集合内可比，换 seed 允许方向不变、数值变化。
- **C**：只承诺**相对比值**在容差内（C4：±25%）；绝对毫秒数不可跨机器比较。

## 1. 一次性准备

### 1.1 代码

```bash
git clone https://github.com/yichen-2025/Multimodal-Fusion-Model.git
cd Multimodal-Fusion-Model
git log -1 --oneline
```

> 冻结基线为 tag `stage-1-frozen`（`4241d96`）；该 tag 之后的提交只改文档，不影响复现结果。

### 1.2 环境（锁定的实测版本）

| 组件 | 版本 |
|---|---|
| Python | 3.12.3 |
| torch | 2.8.0+cu128 |
| scikit-learn | 1.9.0 |
| numpy | 2.3.2 |

```bash
python -V
python -c "import torch, sklearn, numpy; print(torch.__version__, sklearn.__version__, numpy.__version__)"
```

> `requirements.txt` 使用 `>=`，直接安装可能拉到更新版本，导致 C1/C4 漂移。**C2/C3 对版本不敏感**（纯 numpy/sklearn 确定性计算），C1 重训与 C4 必须固定上表版本。

### 1.3 数据与权重（不入库，走 Release）

| 资产 | 路径 | 大小 | 用于 | 来源 |
|---|---|---|---|---|
| 开集划分 | `split_data/dataset_200/split_openset_0/` | 19 MB（tar.gz；解压 40 MB） | C1–C3 | Release 附件 `split_openset_0.tar.gz` |
| BERT 文本嵌入/原型 | `emb/bert_protos.npz` | 已入库 | C3 | git |
| Qwen 文本嵌入/原型 | `emb/qwen_texts.npz`、`emb/qwen_protos.npy` | 已入库 | C3 | git |
| Qwen7B | `emb/qwen7b_texts.npz`、`emb/qwen7b_protos.npy` | 已入库（56 MB） | C3 | git |
| 路由报告 | `ood_reports/` | 6.8 MB（231 个 JSON） | C1 分数级重算 | Release 附件 `ood_reports.tar.gz` |
| OOD 头 | `saved_ood_heads/` | 9.6 MB（86 个） | C1 重训/重评 | Release 附件 `saved_ood_heads.tar.gz` |
| 预训练权重 | `models/bert`、`models/qwen2.5-1.5b` | — | 仅重算文本嵌入/延迟时需要 | `tools/download_*.py` |

> **发布状态**：三个附件已发布到 Release [`stage-1-frozen`](https://github.com/yichen-2025/Multimodal-Fusion-Model/releases/tag/stage-1-frozen)（2026-11-25），校验和以 Release 页与 `SHA256SUMS` 为准；实例 `/root/autodl-tmp/release_assets/` 保留副本。

校验（Release 页面的 `SHA256SUMS` 与本表一致）：

```
7a8b2e86fb1fada3b942f8fd65336133bf36f450402b2bfd093b5f9aff92eaa0  split_openset_0.tar.gz
ef522fd2dafc8170432adde2be3f67e9424088859c3a8b29c866b48e8934fb66  ood_reports.tar.gz
23c838e8439b266726f438fc34bfc023d2db6a6c36ba0902bf54bcc6e2bb7708  saved_ood_heads.tar.gz
```

```bash
sha256sum -c SHA256SUMS
tar -xzf split_openset_0.tar.gz -C split_data/dataset_200/
python -c "import numpy as np; z=np.load('split_data/dataset_200/split_openset_0/train.npz'); print(len(z['labels']))"
# 期望：train 6362 / val 960 / test 2040；已知类 10 + 未知类 Web Attack
```

## 2. 逐条复现

### 2.1 C2 — 模态信息审计（等级 A）

```bash
python scripts/audit_modality_info.py \
  --split_dir split_data/dataset_200/split_openset_0 \
  --out results/2026-11-25/repro_e8_audit \
  --parts e81,e82 --estimator logreg --boot 500 --seed 42
```

期望（`metrics.json`）：

| 字段 | 冻结值 | 容差 |
|---|---|---|
| `e82_conditional_mi.I_text_y_given_features_nats` | `0.12120589775385715` | 1e-9 |
| `e82_conditional_mi.I_text_y_given_features_bootstrap_ci95` | `[0.07700005384992734, 0.1797980301921368]` | 1e-9 |
| `e81_predictability.features_to_text.test.r2_variance_weighted` | `0.4980844656904194` | 1e-9 |
| `e82` 三臂准确率（features / text / both） | `0.9674 / 0.9819 / 0.9873` | 1e-9 |

判定：`I` 只支持「文本高度冗余，独立增量约 5.7%（≤6%）」。**不得**写成 `I ≈ 0` 或「文本无独立信息」（见 claims C-28）。

### 2.2 C3 — 语义原型（等级 A）

BERT 空间（文本嵌入在 split npz 内）：

```bash
python scripts/eval_fewshot_prototypes.py \
  --split_dir split_data/dataset_200/split_openset_0 \
  --protos emb/bert_protos.npz \
  --out results/2026-11-25/repro_e3_e4_bert \
  --tag repro-check --boot 2000 --seed 42
```

期望：`e3_zero_shot.test_acc = 0.12809667673716013`、`test_macro_f1 = 0.050007723760494074`、`E3/E4 gate = false`（容差 1e-9）。

Qwen1.5B 空间：

```bash
python scripts/eval_fewshot_prototypes.py \
  --split_dir split_data/dataset_200/split_openset_0 \
  --protos emb/qwen_protos.npy \
  --text_npz emb/qwen_texts.npz \
  --out results/2026-11-25/repro_e5_qwen --tag repro-check --boot 2000 --seed 42
```

Qwen7B 空间：把 `--protos` 换成 `emb/qwen7b_protos.npy`、`--text_npz` 换成 `emb/qwen7b_texts.npz`。
两者期望值与 `results/2026-11-20/{e5_qwen,e9_qwen7b}/metrics.json` 逐位一致（Qwen1.5B 已实测；Qwen7B 需要更多内存，本轮未跑）。

> 三个空间 × `k∈{1,5,10,20}` 的完整曲线见各次输出 `curve.csv`；门槛为「增益 ≥ +3pp」，实际全部未通过。

### 2.3 C1 — LLM vs BERT 的 OOD 对照（等级 B）

**R1（推荐，无需 GPU）**：用已入库的 5-seed 汇总复算配对差。

```bash
python scripts/verify_c1_pair.py \
  --points results/2026-11-03/ood_multiseed_fair_encoder_n5.json \
  --a "A0*" --b A3 --metric tpr_at_fpr_1
```

期望输出（`--head_seed 42` 为默认值，n5 汇总口径）：

```
配对 seed: [42, 123, 456, 789, 1011]
A0* tpr_at_fpr_1 = 8.7792 ± 0.7878 (n=5)
A3 tpr_at_fpr_1 = 6.9091 ± 9.1990 (n=5)
diff = +1.8701pp, t-CI95 [-8.85, +12.59], 跨零=True
```

**R2（可选，需 GPU）**：分数级重算（231 份路由报告 + OOD 头）。

```bash
python scripts/bootstrap_ood_ci.py --n_boot 4000 --seed 20261028 \
  --out results/2026-11-25/repro_bootstrap_ood_ci.json
```

注意：该脚本复现的是 **sample 级重采样**口径（`bootstrap_ood_ci_with_a3.json`），与 C-25 的 **n5 配对**口径（`bootstrap_ood_ci_fair_encoder_n5.json`）不同，两者不可混用。C-25 判据以 n5 配对为准：`A0* 0.0878±0.0079` vs `A3 0.0691±0.0920`，+1.87pp，p=0.549。

### 2.4 C4 — 成本-精度前沿（等级 C）

需要 GPU + `models/bert`、`models/qwen2.5-1.5b`、`saved_models/model_2013`：

```bash
python scripts/bench_latency.py \
  --split_dir split_data/dataset_200/split_openset_0 \
  --out results/2026-11-25/repro_c4_latency \
  --n_small 1000 --n_llm 200
```

期望：**只比比值**——LLM 路径相对 79 维 MLP 约 `1040×` 延迟、换 `+0.85pp` 精度。绝对毫秒数随 GPU/驱动变化，容差 ±25%，不得跨机器比较绝对值。

## 3. 本次实测对照（2026-11-25，AutoDL，CPU）

| 结论 | 冻结值 | 本次复现值 | 差值 | 判定 |
|---|---|---|---|---|
| C2 `I(text;y\|features)` | 0.12120589775385715 | 0.12120589775385715 | **0.0** | A 通过 |
| C2 CI95 | [0.07700005, 0.17979803] | 同左 | **0.0** | A 通过 |
| C3 BERT `E3 test_acc` | 0.12809667673716013 | 0.12809667673716013 | **0.0** | A 通过 |
| C3 BERT `test_macro_f1` | 0.050007723760494074 | 0.050007723760494074 | **0.0** | A 通过 |
| C3 Qwen1.5B `E3 test_acc` | 0.14622356495468278 | 0.14622356495468278 | **0.0** | A 通过 |
| C3 Qwen1.5B `test_macro_f1` | 0.07922903846543791 | 0.07922903846543791 | **0.0** | A 通过 |
| C1 配对差 | +1.8701pp，CI [−8.85, +12.59] | +1.8701pp，CI [−8.85, +12.59] | **0.0** | B 通过 |
| C4 延迟比 | ~1040× | 未测（无 GPU） | — | 待补 |

C2/C3 的复现产物位于 `results/2026-11-25/repro_e8_audit/`、`repro_e3_e4_bert/`（本机实测，不入库）。

**数据包完整性验收（2026-11-25）**：把 `split_openset_0.tar.gz` 解压到全新目录后重跑 C2/C3，5 个数据文件（`train/val/test.npz`、`label_mapping.json`、`train_scaler.npy`）的 sha256 与仓库内原件**全部一致**，指标 diff 仍为 0.0；`ood_reports.tar.gz` 解出 231 份、`saved_ood_heads.tar.gz` 解出 86 个目录。

## 4. 判定与容差

1. **A 级**：全部关键字段 `|diff| ≤ 1e-9`。差一点先查数据（npz 行序/类别映射），再查 sklearn 版本。
2. **B 级**：点位必须一致；换 seed 后只要求「CI 是否跨零」的方向不变，不要求数值复现。
3. **C 级**：只比比值，容差 ±25%。
4. 任何级别的**结论方向翻转**（例如 CI 由跨零变为不跨零且越过 +2pp 门槛）都视为复现失败，需先排查数据哈希与环境，再讨论写法。
5. 不允许为了让数字对上而改 `results/` 下的冻结产物、阈值或 `claims.md`。

## 5. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `FileNotFoundError: split_data/...` | 未解压 Release 数据包 | 按 §1.3 放置，比对 sha256 |
| C2 数值完全不同 | npz 被重新生成（行序变了） | 用 Release 原始 npz，不要跑 `make_openset_split.py` |
| C3 报缺少 `text_embeddings` | 用了 `--text_npz` 的 Qwen 命令跑 BERT | BERT 不需要 `--text_npz` |
| C1 配对差不是 1.87pp | 用了 sample 级 bootstrap 口径 | 确认读的是 `ood_multiseed_fair_encoder_n5.json` |
| C4 延迟差一个量级 | CPU 模式 / 不同 GPU | 只在同型号 GPU 上比值；CPU 下不要报 C4 |
| 下载模型失败 | 无法访问 HF | `export HF_ENDPOINT=https://hf-mirror.com` |

## 附录 A — C1 配对差的等价片段

`scripts/verify_c1_pair.py` 的计算逻辑（只取 `head_seed=42`，按 backbone seed 配对）：

```python
import json, math, statistics
d = json.load(open("results/2026-11-03/ood_multiseed_fair_encoder_n5.json"))
by = {}
for p in d["points"]:
    if p["head_seed"] != 42:
        continue
    by.setdefault(p["variant"], {})[p["backbone_seed"]] = p["tpr_at_fpr_1"]
seeds = sorted(set(by["A0*"]) & set(by["A3"]))
diffs = [(by["A0*"][s] - by["A3"][s]) * 100 for s in seeds]
mean, sd = statistics.mean(diffs), statistics.stdev(diffs)
ci = (mean - 2.776 * sd / math.sqrt(len(diffs)), mean + 2.776 * sd / math.sqrt(len(diffs)))
print(round(mean, 4), [round(x, 2) for x in ci])   # 1.8701 [-8.85, 12.59]
```

## 附录 B — 不入库资产清单（Release 发布项）

| 打包项 | 内容 | 必要性 | 状态 |
|---|---|---|---|
| `split_openset_0.tar.gz` | `train/val/test.npz` + `label_mapping.json` + `train_scaler.npy` + 原始 csv | C1–C3 必需 | ✅ 已发布 |
| `ood_reports.tar.gz` | 231 份 `ood_routing_*.json` | C1 分数级重算必需 | ✅ 已发布 |
| `saved_ood_heads.tar.gz` | 86 个 OOD 头 | C1 重评/重训 | ✅ 已发布 |
| `SHA256SUMS` | 上列所有文件的校验和 | 全部 | ✅ 已发布 |

权重（BERT / Qwen2.5-1.5B / Qwen7B）不打包，用 `tools/download_*.py` 从 HF 拉取。
