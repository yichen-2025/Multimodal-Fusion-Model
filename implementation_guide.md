# Implementation Guide

从零开始实现多模态融合网络流量分类模型的完整指南。

---

## Overview

本项目实现基于多模态融合的网络流量恶意检测模型，支持：
- **闭集分类**：BENIGN / DDoS 二分类
- **消融实验**：对比 A0/A1/A2/A3 四种变体
- **开集检测**：识别未知攻击流量（OOD Detection）
- **少样本学习**：在极少训练数据下验证模型泛化能力

### 核心概念

| 概念 | 说明 |
|------|------|
| **数据集ID (`dataset_id`)** | 标识一个数据集子集 |
| **划分ID (`split_id`)** | 同一数据集的不同 train/val/test 划分 |
| **模型ID (`model_id`)** | 一次训练产出的模型 |
| **OOD头ID (`ood_id`)** | 一次OOD检测头训练产出 |

### 消融变体

| 变体 | 数值模态 | 文本模态 | LLM | 说明 |
|------|---------|---------|-----|------|
| A0 | ✓ | ✓ | ✓ | 完整多模态+LLM |
| A1 | ✓ | ✗ | ✓ | 仅数值+LLM |
| A2 | ✗ | ✓ | ✓ | 仅文本+LLM |
| A3 | ✓ | ✓ | ✗ | 纯MLP分类（无LLM） |

---

## Prerequisites

### 1. 环境要求

- **Python**: 3.10+
- **GPU**: 推荐 NVIDIA GPU，显存 ≥ 16GB（训练LLM时需要）
- **操作系统**: Windows / Linux / macOS 均可

### 2. 安装依赖

```bash
# 创建虚拟环境
conda create -n multimodal-fusion python=3.10 -y
conda activate multimodal-fusion

# 安装依赖（使用清华镜像加速）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 下载预训练模型

```bash
python tools/download_bert.py    # BERT编码器（~400MB）
python tools/download_qwen.py    # Qwen2.5-1.5B LLM（~3GB）
```

### 4. 准备原始数据

将 CSV 格式的原始数据集放入 `data_processing/` 目录。数据集需包含：
- `Label` 列：`BENIGN`（正常流量）或 `DDoS`（恶意流量）
- 特征列：`Destination Port`, `Bwd Packet Length Mean`, `Avg Bwd Segment Size`, `Bwd Packet Length Max`, `Bwd Packet Length Std`, `URG Flag Count`, `Packet Length Mean`, `Average Packet Size`, `Packet Length Std`

---

## 执行方式

项目通过 `main.py` 统一管理所有步骤。查看所有可用步骤：

```bash
python main.py --list
```

执行单个步骤：

```bash
python main.py --step <步骤名> [参数...]
```

查看某步骤的详细帮助：

```bash
python main.py --help --step <步骤名>
```

---

## Part 1: 闭集分类流程

从原始数据到完成闭集分类的完整流程。

### Step 1: 数据清洗

```bash
python main.py --step clean
```

**输入**: `data_processing/*.csv`
**输出**: `processed_dataset/processed_dataset.csv`
**说明**: 处理缺失值、异常值，标签编码（BENIGN→0, DDoS→1）

### Step 2: 提取数据子集

```bash
# 默认提取5000条样本，自动分配 dataset_id
python main.py --step subset

# 或指定参数
python main.py --step subset --dataset_id 1 --num_samples 5000 --seed 42
```

**输出**: 
- `processed_dataset/dataset_1.csv`
- `processed_dataset/subset_1_scaled_features.npy`
- `processed_dataset/subset_1_labels.npy`

> **记录 `dataset_id`**，后续步骤需要使用。

### Step 3: 模态分离与数据划分

```bash
# 将数据拆分为 train/val/test 并生成 BERT 嵌入
python main.py --step split --dataset_id 1
```

**输出**: `split_data/dataset_1/split_0/`
- `train.npz` — 训练集（特征 + BERT嵌入 + 标签）
- `val.npz` — 验证集
- `test.npz` — 测试集
- `train_scaler.npy` — 标准化器参数

> **记录 `split_id`**，后续步骤需要使用。

### Step 4: 训练模型

```bash
# 训练 A3 变体（无LLM，快速）
python main.py --step train --dataset_id 1 --split_id 0 --variant A3

# 训练 A0 变体（完整模型，需要GPU）
python main.py --step train --dataset_id 1 --split_id 0 --variant A0
```

**输出**: `saved_models/model_X/`
- `pytorch_model.bin`
- `config.txt`
- `loss_logs/`

> **记录 `model_id`**，测试时需要使用。

### Step 5: 测试模型

```bash
python main.py --step test --dataset_id 1 --split_id 0 --model_id <model_id>
```

**输出**: `test_reports/report_X.json`
- Accuracy, Precision, Recall, F1
- 混淆矩阵

### Step 6: 运行消融实验（可选）

一次性训练并评估所有变体：

```bash
python main.py --step ablation --dataset_id 1 --split_id 0
```

**输出**: `ablation_results/ablation_results_时间戳.csv`

---

## Part 2: 开集检测 + 少样本学习 (Phase 1)

证明 LLM 在开集/少样本场景下的独特优势。

### 核心思路

```
训练数据: [BENIGN + known_DDoS]  ← 已知类
测试数据: [BENIGN + known_DDoS + unknown_DDoS]  ← 含未知类

路由策略:
  已知样本 → A3 分类器 → benign / known_DDoS
  未知样本 → A0 (LLM) → 语义推理 → "正常" / "DDoS攻击"
```

### Step P1.1: 构造开集数据

从闭集数据中分离"已知DDoS"和"未知DDoS"：

```bash
python main.py --step openset --dataset_id 1 --unknown_ratio 0.3
```

**输出**: `split_data/dataset_1/split_openset_0/`
- `train.npz` — [BENIGN + known_DDoS]（不含未知）
- `val.npz` — [BENIGN + known_DDoS]
- `test.npz` — [BENIGN + known_DDoS + unknown_DDoS]（含未知）

### Step P1.2: 构造少样本数据

为每个已知类抽取 k 条样本用于训练：

```bash
# k=5: 每类仅5条训练样本
python main.py --step fewshot --dataset_id 1 --split_id 0 --k_per_class 5

# k=10
python main.py --step fewshot --dataset_id 1 --split_id 0 --k_per_class 10

# k=20
python main.py --step fewshot --dataset_id 1 --split_id 0 --k_per_class 20
```

**输出**: `split_fewshot_k_0/` 系列目录
- `train.npz` — 仅 2k 条样本（k benign + k known_DDoS）
- `val.npz` — 不变
- `test.npz` — 不变

### Step P1.3: 训练 OOD 检测头

OOD头基于A3融合特征的类原型距离来检测未知样本。

```bash
# 使用全量数据训练OOD头
python main.py --step ood-train --model_id <A3_model_id> --dataset_id 1 --split_id 0

# 使用少样本训练OOD头
python main.py --step ood-train --model_id <A3_model_id> --dataset_id 1 --split_id 0 --fewshot_k 5
python main.py --step ood-train --model_id <A3_model_id> --dataset_id 1 --split_id 0 --fewshot_k 10
python main.py --step ood-train --model_id <A3_model_id> --dataset_id 1 --split_id 0 --fewshot_k 20
```

**输出**: `saved_ood_heads/ood_X/`
- 原型向量权重
- 阈值配置

> **记录 `ood_id`**，评估时需要使用。

### Step P1.4: OOD 路由评估

运行开集路由评估，对比 A3-only vs A3+OOD+LLM：

```bash
python main.py --step ood-eval \
    --model_id <A3_model_id> \
    --ood_id <ood_id> \
    --llm_model_id <A0_model_id> \
    --dataset_id 1 \
    --split_id 0 \
    --fewshot_k 5
```

对每个 k 值重复运行。

**输出**: `ood_reports/`
- 各配置下的详细评估指标

### Step P1.5: 一键完整实验

如需自动化运行整个流程：

```bash
python main.py --step experiment \
    --backbone_model_id <A3_model_id> \
    --llm_model_id <A0_model_id> \
    --dataset_id 1 \
    --k_values "5,10,20,None"
```

此命令将自动：
1. 训练 OOD 检测头（对每个 k 值）
2. 运行 OOD 路由评估
3. 生成汇总报告

### Step P1.6: 生成汇总报告

```bash
python main.py --step report
```

**输出**: `test_reports/` 和 `ood_reports/`
- `openworld_runs_{timestamp}.csv` — 每次运行明细
- `openworld_fewshot.csv` — 按k值汇总对比
- `openworld_experiment_summary_{ts}.md` — 可读Markdown报告
- `openworld_metrics_{timestamp}.json` — 结构化指标

---

## 评估指标说明

| 指标 | 说明 |
|------|------|
| Macro-F1 | 三分类（benign/known/unknown）宏平均F1 |
| **Unknown F1** | 将unknown视为正类的F1（**核心指标**） |
| Unknown Recall | 未知DDoS被正确检测的比例 |
| Unknown Leak Rate | 未知DDoS被误判为known的比例 |
| Benign Recall | 正常流量召回率 |
| Known DDoS Recall | 已知DDoS召回率 |

---

## 常见场景速查

### 场景A: 从零开始（闭集）

```bash
python main.py --step clean
python main.py --step subset --dataset_id 1
python main.py --step split --dataset_id 1
python main.py --step train --dataset_id 1 --split_id 0 --variant A3
python main.py --step test --dataset_id 1 --split_id 0 --model_id <id>
```

### 场景B: 运行完整消融实验

```bash
python main.py --step clean
python main.py --step subset --dataset_id 1
python main.py --step split --dataset_id 1
python main.py --step ablation --dataset_id 1 --split_id 0
```

### 场景C: 已有模型，仅跑开集评估

```bash
# 假设已有: model_9 (A3), model_5 (A0)
python main.py --step openset --dataset_id 1 --unknown_ratio 0.3
python main.py --step fewshot --dataset_id 1 --k_per_class 5
python main.py --step ood-train --model_id 9 --dataset_id 1 --fewshot_k 5
python main.py --step ood-eval --model_id 9 --ood_id <ood> --llm_model_id 5 --fewshot_k 5
python main.py --step report
```

### 场景D: 一键完整开集实验

```bash
python main.py --step experiment --backbone_model_id 9 --llm_model_id 5 --k_values "5,10,20,None"
```

---

## 文件结构

```
Multimodal-Fusion-Model/
├── main.py                        # 统一入口（推荐）
├── IMPLEMENTATION_GUIDE.md        # 本文档
├── README.md                      # 项目概览
│
├── scripts/                       # 运行脚本
│   ├── data_cleaning.py           #   P0 数据清洗
│   ├── extract_subset.py          #   P0 子集提取
│   ├── split_modality.py          #   P0 模态分离
│   ├── train.py                   #   P0 模型训练
│   ├── test_model.py              #   P0 模型测试
│   ├── run_ablation.py            #   P0 消融实验
│   ├── make_openset_split.py      #   P1 开集划分
│   ├── train_ood_head.py          #   P1 OOD头训练
│   ├── run_ood_routing.py         #   P1 OOD路由评估
│   ├── make_fewshot.py            #   P1 少样本构造
│   ├── run_openworld_experiment.py # P1 一键实验
│   └── report_generator.py        #   P1 报告生成
│
├── src/model_architectures/       # 模型架构
│   ├── multi_modal_model.py       #   多模态融合模型
│   ├── ood_head.py                #   OOD检测头
│   ├── bert_encoder.py            #   BERT编码器
│   ├── numeric_encoder.py         #   数值编码器
│   └── fusion_projection.py       #   融合投影层
│
├── src/data/data_loader.py        # 数据加载
├── utils/log_utils.py             # 日志工具
├── tools/                         # 模型下载工具
│
├── data_processing/               # 原始数据（需手动放入）
├── processed_dataset/             # 清洗后数据（自动生成）
├── split_data/                    # 划分后数据（自动生成）
├── saved_models/                  # 训练模型（自动生成）
├── saved_ood_heads/               # OOD检测头（自动生成）
├── ablation_results/              # 消融实验结果（自动生成）
├── logs/                          # 操作日志（自动生成）
├── test_reports/                  # 测试报告（自动生成）
└── ood_reports/                   # OOD评估报告（自动生成）
```

---

## 输出目录说明

| 目录 | 内容 | 创建时机 |
|------|------|----------|
| `processed_dataset/` | 清洗后的数据和子集 | Step 1-2 |
| `split_data/` | train/val/test 划分数据 | Step 3 |
| `saved_models/` | 训练好的模型权重 | Step 4 |
| `saved_ood_heads/` | OOD检测头权重 | Step P1.3 |
| `ablation_results/` | 消融实验对比结果 | 消融实验后 |
| `test_reports/` | 模型测试报告 | Step 5 |
| `ood_reports/` | OOD路由评估报告 | Step P1.4 |
| `logs/` | 所有操作的日志记录 | 每步执行时 |

---

## Troubleshooting

### CUDA 不可用

```bash
python -c "import torch; print(torch.cuda.is_available())"
# 如果返回 False，检查 CUDA 驱动版本或使用 CPU 版本 PyTorch
```

### 显存不足

- 减小 `--batch_size` 参数
- 使用梯度累积 `--grad_accum`
- 训练时关闭 `--verbose` 减少内存开销

### 内存不足

- 减小 `--num_samples` 减少数据量
- 关闭其他占用内存的程序

### 日志文件查看

```bash
# 查看最近的训练日志
cat logs/training/log_*.json | python -m json.tool

# 查看 OOD 训练指标
cat logs/ood_training/log_*.json | python -m json.tool
```