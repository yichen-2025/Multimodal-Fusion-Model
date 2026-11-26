# explain-rag 分支复现方式

> **从 0 开始复现。** 本文档假定你在一台**全新的机器**上执行 `git clone`，仓库里只有源码和少量结果 JSON：所有数据集划分、模型权重、OOD 检测头、日志、解释产物都**不在**版本库里（均被 `.gitignore` 排除），必须按本文从头重跑。
>
> 目标分支：`exp/explain-rag`（远端存在 `origin/exp/explain-rag`）
> 参考 HEAD：`03b37fd`（2026-11-25 核对；编制本文件时为 `4b729d6`）
> 文档编写日期：2026-10-28；最近核对：2026-11-25
>
> **与其他复现文档的分工**：本文是"新机器从 0 重建"的完整流程（含数据链、12 个 backbone、OOD 头、解释）。
> 复现入口是 [`REPRODUCE.md`](../REPRODUCE.md)；本文只作设计与背景说明，不提供复现步骤。口径必须一致：`dataset_200 / split_openset_0`。
> **解释器**：非交互 SSH 下 `python` 不在 PATH，本文命令里的 `python` 请替换为 `/root/miniconda3/bin/python`。
> **GPU**：当前实例在无卡模式下 `torch.cuda.is_available()` 为 `False`，第 3–6 节需先切 GPU 模式。

---

## 零、先搞清楚"哪些能复现、哪些必须自己造"

| 内容 | 是否在 git 中 | 复现方式 |
|---|---|---|
| 源码、脚本、配置 | ✅ 98 个受控文件 | `git clone` 即得 |
| `results/2026-10-28/*.json`、`artifacts/ledger.csv`、`docs/stage_conclusion_20261027.md` | ✅ | 作为**对照标准**，不是运行依赖 |
| CIC-IDS2018 原始 CSV | ❌（仅一个 DDoS 文件被历史上 track 过） | 自行下载 7 个 CSV 放入 `data_processing/` |
| `processed_dataset/`、`split_data/` | ❌ | 第 2 节重建 |
| `models/qwen2.5-1.5b`、`models/bert` | ❌（`models/` 被忽略） | 第 1 节下载（约 4.5GB） |
| `saved_models/model_*` | ❌ | 第 3 节重训 |
| `saved_ood_heads/`、`ood_reports/`、`test_reports/` | ❌ | 第 4–5 节重跑 |
| `explanation_results/` | ❌ | 第 6 节重跑 |
| `run_history/`、`logs/` | ❌ | 运行时自动生成 |

**一句话：源码开箱即用，数据与权重全部自备，训练必须重跑。**

---

## 一、环境准备

### 1.1 克隆指定分支

```bash
git clone -b exp/explain-rag <repo_url> Multimodal-Fusion-Model
cd Multimodal-Fusion-Model
```

### 1.2 Python 与依赖

原机实测环境（建议对齐，否则数值会有小幅漂移）：

| 项 | 版本 |
|---|---|
| Python | 3.12.3（`/root/miniconda3/bin/python`） |
| torch | 2.8.0+cu128 |
| transformers | 5.16.1 |
| peft | 0.20.0 |
| scikit-learn | 1.9.0 |
| GPU | RTX 4080 SUPER 32GB |

```bash
/root/miniconda3/bin/pip install -r requirements.txt
# requirements.txt 只给了下限，严格复现请显式钉版本：
#   poetry/pip install "torch==2.8.0" "transformers==5.16.1" "peft==0.20.0"
```

**注意**：本机 `python` 不在非交互 SSH 的 PATH 里，历史命令全部使用绝对路径 `/root/miniconda3/bin/python`。你自己跑时要么用绝对路径，要么先 `source ~/.bashrc`。

### 1.3 下载基座模型

```bash
mkdir -p models
# Qwen2.5-1.5B：放到 models/qwen2.5-1.5b（约 2.9GB）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir models/qwen2.5-1.5b
# BERT：放到 models/bert（约 1.6GB）
huggingface-cli download google-bert/bert-base-uncased --local-dir models/bert
```

> 若追求与原机**逐位一致**，直接从原机 `scp -P 25411 root@connect.weste.seetacloud.com:/root/autodl-tmp/Multimodal-Fusion-Model/models ./` 拷贝，比重新下载更可靠（HF 上的小版本差异会影响 hidden states）。SSH 端口以 AutoDL 控制台为准，实例重启后可能变化。

---

## 二、数据准备（P0）

### 2.1 下载 CIC-IDS2018

需要官方 7 个 CSV（含 `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`），全部放入 `data_processing/`。

### 2.2 清洗 → 提取子集 → 模态分离 → 开集划分

以下四条是**原机 run_history 中记录的真实命令**（`dataset_200` 就是全部结论所依据的主数据集）：

```bash
# ① 清洗，产出 processed_dataset/processed_dataset.csv + label_mapping.json
/root/miniconda3/bin/$PY scripts/data_cleaning.py

# ② 平衡采样出主数据集 dataset_200（9999 条上限，10 类）
/root/miniconda3/bin/$PY scripts/extract_subset.py \
  --mode balanced --total_samples 9999 --dataset_id 200 --random_state 42

# ③ 模态分离：70/10/20，产出 split_data/dataset_200/split_0/*.npz
/root/miniconda3/bin/$PY scripts/split_modality.py \
  --dataset_id 200 --input_csv processed_dataset/dataset_200.csv \
  --test_size 0.2 --val_size 0.1 --random_state 42

# ④ 开集划分：留出类 10，模拟未知 10%，val 保留 50 条真 OOD
/root/miniconda3/bin/$PY scripts/make_openset_split.py \
  --dataset_id 200 --source_split_id 0 --hold_out_classes 10 \
  --simulated_unknown_ratio 0.1 --val_unknown_count 50 --allow_mixed_calibration
```

验收：`split_data/dataset_200/split_openset_0/` 下 test 集 **2040 条**（其中 163 条模拟未知、222 条真 OOD），`label_mapping.json` 为 10 类。

> 参数名坑：是 `--simulated_unknown_ratio`，不是 `--unknown_ratio`；是 `--k_values`，不是 `--k_per_class`。

### 2.3 输入合法性自检（必跑）

```bash
/root/miniconda3/bin/$PY scripts/validate_inputs.py --dataset_id 200
```

必须全部通过：数值列无 NaN/Inf、文本无非法前缀、无重复输入组、训练/测试无重叠。**这一步不通过就不要往下训**——历史上正是"输入塌缩"让 43.6% 的样本向量重复，导致整批模型作废。

---

## 三、训练 12 个 backbone（P1）

### 3.1 变体与规模

4 个变体 × 3 个 backbone seed（42 / 43 / 44），共 12 个模型，原机 ID 为 `model_2000`–`model_2011`：

| 变体 | 含义 |
|---|---|
| `A0*` | LLM(Qwen+LoRA) + 数值，concat 融合 |
| `A0*_no_num` | 仅文本（LLM 分支） |
| `A0*_add` | 加性融合 |
| `A0*_attn` | 注意力融合 |

### 3.2 单条训练命令模板（来自 run_history）

```bash
/root/miniconda3/bin/$PY scripts/train.py \
  --variant "A0*" --dataset_id 200 --split_id 0 --openset \
  --model_id 2004 --seed 123 --epochs 10 --lr 5e-4
```

要点：

- `--openset` 必须加，否则加载的是闭集划分；
- `--model_id` 不填则自动递增，建议显式指定以便对照台账；
- **若改动过 `src/model_architectures/fusion_projection.py`，必须加 `--force_retrain`**，否则旧权重维度不兼容；
- 实际超参以 `saved_models/model_<id>/config.txt` 为准（不同批次 epochs 有 10 和 15 两种，config 是唯一真相）。

### 3.3 逐个跑完 12 个

```bash
for V in "A0*_no_num" "A0*" "A0*_add" "A0*_attn"; do
  for S in 42 43 44; do
    /root/miniconda3/bin/$PY scripts/train.py \
      --variant "$V" --dataset_id 200 --split_id 0 --openset \
      --seed $S --epochs 10 --lr 5e-4
  done
done
```

单模型实测约 2650s（RTX 4080 SUPER），12 个约 9 小时。建议在 `tmux` 里跑，本地可断线。

> 若只做**冒烟验证**：先跑 `--variant A3`（纯 MLP，几分钟）确认数据管线通，再跑 LLM 变体。

---

## 四、OOD 检测头与路由（P2）

### 4.1 训练检测头（每个 backbone × 多个 head seed）

```bash
/root/miniconda3/bin/$PY scripts/train_ood_head.py \
  --model_id 2010 --ood_id 36 --dataset_id 200 --split_id 0 \
  --variant "A0*_attn" --distance_type cosine --temperature 0.1 --epochs 10 --seed 44
```

- 每个 backbone 训练 3 个 head seed（42/43/44），单次约 120s
- 全量：12 backbone × 3 head seed ≈ 36 个 head（原机 `ood_head_0`–`ood_head_38`，含历史无效项）

### 4.2 路由评估

```bash
/root/miniconda3/bin/$PY scripts/run_ood_routing.py \
  --backbone_model_id 2000 --ood_id 14 --dataset_id 200 --split_id 0
```

产物在 `ood_reports/`。开集指标必须**分三组报**：`auroc_genuine`、`tpr@fpr=1%`、`unknown_recall_simulated`，不允许只报 overall accuracy。

### 4.3 汇总与置信区间

```bash
/root/miniconda3/bin/$PY scripts/summarize_ood_multiseed.py
/root/miniconda3/bin/$PY scripts/bootstrap_ood_ci.py      # B=4000，配对重采样
```

对照标准：`results/2026-10-28/ood_multiseed_summary_20261028.json`、`bootstrap_ood_ci_20261028.json`。

### 4.4 闭集评估

```bash
/root/miniconda3/bin/$PY scripts/test_model.py --model_id 2000 --dataset_id 200 --split_id 0
```

对照标准：known_accuracy 应落在 **0.986–0.995** 区间（纯文本 0.9919、concat 0.9944、add 0.9930、attn 0.9915）。

---

## 五、解释层（Phase 2 B1）

```bash
# 5.1 生成解释（分层抽样 100 条）
/root/miniconda3/bin/$PY scripts/generate_explanations.py \
  --backbone_model_id 2000 --dataset_id 200 --split_id 0 \
  --n_samples 100 --sampling stratified

# 5.2 解释质量评估
/root/miniconda3/bin/$PY scripts/eval_explanations.py --result explanation_results/<生成的文件名>.json

# 5.3 映射路径对比（LLM vs 检索，同批 72 条）
/root/miniconda3/bin/$PY scripts/eval_mapping_hybrid.py --result explanation_results/<同上>.json
```

对照标准：

| 指标 | 期望值 |
|---|---|
| 引用率 | 0.98（C1 unknown 路径 1.00） |
| 数字未验证率（v3） | ≈ 0.11 |
| MITRE 命中（LLM 自选） | 40 / 72 |
| 检索命中 | 61 / 72，并集 67 / 72，McNemar p=0.000324 |

**关键约束**：prompt 中不得出现"预测类别的正确 MITRE 编号"，否则映射指标失效。

---

## 六、CTI-RAG 检索（Phase 2 B2）

```bash
# 7 条基线库
/root/miniconda3/bin/$PY scripts/eval_retrieval.py \
  --backbone_model_id 2000 --kb a --tag a

# 10 条（+父技术）/ 22 条（+混淆项）
/root/miniconda3/bin/$PY scripts/eval_retrieval.py \
  --backbone_model_id 2000 --kb b --tag b
/root/miniconda3/bin/$PY scripts/eval_retrieval.py \
  --backbone_model_id 2000 --kb c --tag c

# 三臂汇总 + 敏感性对照
/root/miniconda3/bin/$PY scripts/summarize_cti_kb_scale.py
```

对照标准（1474 条攻击样本）：

| 候选库 | strict hit@1 |
|---|---|
| 7 条 | 0.8494（1252/1474） |
| 10 条 | 0.422 |
| 22 条 | 0.333（家族级 0.749） |

检索是确定性的（BM25 + 固定词表），**这三组数字应当在数据完全一致时逐位复现**，是判断"数据管线是否真的对齐"最灵敏的探针。

### 6.1 unknown 路径接 LLM 解释

```bash
/root/miniconda3/bin/$PY scripts/generate_unknown_explanations.py \
  --routing_report ood_reports/<路由报告>.json --n_samples 100 --seed 42
```

对照标准：引用率 1.00（关卡 0.90），100/100 非空。

---

## 七、P0 截断诊断（可选，验证口径一致性）

```bash
/root/miniconda3/bin/$PY scripts/diag_text_truncation.py
```

对照标准：2040 条中标签变化 **3 条**，100 条解释子集变化 **0 条**，`known_accuracy` 0.99456 → 0.99396。

---

## 八、产物治理校验（每次落盘后必跑）

```bash
/root/miniconda3/bin/$PY scripts/verify_caliber.py
/root/miniconda3/bin/$PY scripts/audit_artifacts.py --check
```

预期输出 `[PASS]`。新机器上线后 `artifacts/ledger.csv` 引用的模型路径大部分不存在，**首次必须跑 `--check` 的全量重建或先归档空目录**，否则会出现大量悬空引用——这属于预期现象，不代表复现失败。

---

## 九、复现成功的判定标准

不要追求逐位相同，按下面的容差判断：

| 指标 | 容差 | 判定 |
|---|---|---|
| 数据规模 | test 恰为 2040 条 | 必须精确匹配 |
| 检索 hit@1（7 条库） | 0.8494 | 必须精确匹配（确定性） |
| 闭集 known_accuracy | 0.986–0.995 | 落区间即可 |
| genuine AUROC | 0.93–0.95 | 落区间即可 |
| `tpr@fpr=1%` | 0.09 附近 | 允许 ±0.03 |
| 映射命中 | 61/72 ± 3 | 允许小幅漂移 |
| 解释未验证率 | 0.08–0.15 | 允许漂移 |

若检索 hit@1 不相等，**先查数据管线**（是否漏了 `--mode balanced`、`--hold_out_classes 10`、`--random_state 42`），而不是查检索代码。

---

## 十、常见坑

| 现象 | 原因 | 处理 |
|---|---|---|
| `python: command not found` | 非交互 SSH 未加载 conda | 用 `/root/miniconda3/bin/python` 绝对路径 |
| 训练报维度不匹配 | 改过 `fusion_projection.py` | 加 `--force_retrain` |
| LoRA 参数加载不到 | 旧 checkpoint 的 `state_dict` 缺陷（已修） | 重训，不要复用 `model_2003` / `model_2008` |
| 开集 MRR/AUROC 为 0 | 旧划分缺真实 OOD 标签（`unknown_origin=-1`） | 用 `make_openset_split.py` 重新生成，不要手改 |
| 检索全 0 分 | 直接把 ATT&CK 原文塞进库（无行为词元） | 必须走 `knowledge_base.py` 的 keywords 标注 |
| 解释里的数字乱编 | prompt 未约束、或数字校验器版本不对 | 用 v3 校验口径，阈值 0.05 已确认不可达 |
| 台账校验大量失败 | 新机器无 `saved_models/`、`ood_reports/` | 预期现象；重训后再校验 |

---

## 十一、最小复现路径（只想验证可行性）

若你只需要确认"这套代码能跑通"，而不需要复现论文级结论：

```bash
# 1. 数据
$PY scripts/data_cleaning.py
$PY scripts/extract_subset.py --mode balanced --total_samples 2000 --dataset_id 200 --random_state 42
$PY scripts/split_modality.py --dataset_id 200 --input_csv processed_dataset/dataset_200.csv --test_size 0.2 --val_size 0.1 --random_state 42
$PY scripts/make_openset_split.py --dataset_id 200 --source_split_id 0 --hold_out_classes 10 --simulated_unknown_ratio 0.1 --val_unknown_count 50 --allow_mixed_calibration

# 2. 只训一个 A0*，只训一个 head，跑一次路由
$PY scripts/train.py --variant "A0*" --dataset_id 200 --split_id 0 --openset --epochs 2 --lr 5e-4
$PY scripts/train_ood_head.py --model_id <上一步ID> --dataset_id 200 --split_id 0 --variant "A0*" --distance_type cosine
$PY scripts/run_ood_routing.py --backbone_model_id <上一步ID> --ood_id <head ID> --dataset_id 200 --split_id 0

# 3. 解释 + 检索
$PY scripts/generate_explanations.py --backbone_model_id <ID> --dataset_id 200 --split_id 0 --n_samples 20
$PY scripts/eval_retrieval.py --backbone_model_id <ID> --kb a --tag smoke
```

此路径的**指标不具可比性**（样本量、epoch 都不足），只证明链路可通。

---

## 十二、复现清单（打勾用）

- [ ] clone `exp/explain-rag` 分支
- [ ] 依赖版本对齐（torch 2.8 / transformers 5.x / peft 0.20）
- [ ] `models/qwen2.5-1.5b`、`models/bert` 就位
- [ ] 7 个 CIC-IDS2018 CSV 就位
- [ ] `dataset_200` + `split_openset_0` 生成，test=2040
- [ ] `validate_inputs` 全通过
- [ ] 12 个 backbone 训练完成，config 可查
- [ ] 36 个 OOD head + 路由报告
- [ ] 闭集 known_accuracy 落在 0.986–0.995
- [ ] 检索三臂 hit@1 = 0.849 / 0.422 / 0.333
- [ ] 解释 100 条 + 评估（引用率 0.98）
- [ ] unknown 路径解释（引用率 1.00）
- [ ] `verify_caliber` + `audit_artifacts --check` 通过
