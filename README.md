# 多模态融合网络流量分类模型

基于多模态融合的网络流量恶意检测模型，结合统计特征模态和文本描述模态进行流量分类，并支持消融实验以验证各模态的有效性。项目当前已拓展至**开集未知攻击检测**与**少样本学习**场景，旨在证明LLM在复杂场景下的独特优势。

## 项目简介

本项目实现了一个多模态融合模型，用于网络流量恶意检测。模型将网络流量数据转换为两种模态：

1. **统计特征模态**：9维数值特征（如包长度均值、端口号等）
2. **文本描述模态**：768维BERT语义嵌入

通过多模态融合技术，将两种模态的特征进行融合，结合LLM（Qwen2.5）进行分类。项目同时提供消融实验功能，可对比不同模态组合的性能。

### 核心特性

- **多模态融合**：数值特征与文本语义特征的高效融合
- **消融实验**：支持A0/A1/A2/A3四种变体对比
- **开集检测 (OOD)**：通过原型距离检测未知DDoS攻击
- **LLM路由**：未知样本交由LLM进行语义推理判定
- **少样本学习**：支持k=5/10/20等少样本训练配置

## 消融实验变体

| 变体 | 数值模态 | 文本模态 | LLM | 描述 |
|------|---------|---------|-----|------|
| A0 | ✓ | ✓ | ✓ | 全模型（数值+文本+LLM） |
| A1 | ✓ | ✗ | ✓ | 仅数值+LLM（无文本） |
| A2 | ✗ | ✓ | ✓ | 仅文本+LLM（无数值） |
| A3 | ✓ | ✓ | ✗ | 无LLM（纯MLP分类） |

## 目录结构

```
Multimodal-Fusion-Model/
├── main.py                        # 主入口文件
├── README.md                      # 项目说明文档
├── implementation_guide.md          # 实施指南（从零执行，含正确命令）
├── 原理图.png                      # 项目原理图
├── 数据集处理流程图.png             # 数据处理流程图
├── .gitignore                     # Git忽略规则
│
├── scripts/                       # 运行脚本
│   ├── data_cleaning.py           # 数据清洗脚本
│   ├── extract_subset.py          # 数据集子集提取脚本
│   ├── split_modality.py          # 模态分离与数据集划分脚本
│   ├── train.py                   # 模型训练脚本（支持消融变体）
│   ├── test_model.py              # 模型测试脚本
│   ├── run_ablation.py            # 消融实验运行器
│   ├── make_openset_split.py      # [P1.1] 构造开集数据集划分
│   ├── train_ood_head.py          # [P1.3] 训练OOD检测头
│   ├── run_ood_routing.py         # [P1.4] OOD路由评估
│   ├── make_fewshot.py            # [P1.2] 构造少样本数据
│   ├── report_generator.py       # [P1.5] 结果汇总报告生成器
│   └── run_openworld_experiment.py # [P1.5] 开集少样本全流程实验
│
├── tools/                         # 工具脚本
│   ├── download_bert.py           # BERT模型下载脚本
│   └── download_qwen.py           # Qwen模型下载脚本
│
├── src/                           # 核心源码
│   ├── data/
│   │   └── data_loader.py         # 数据加载模块
│   └── model_architectures/       # 模型架构
│       ├── bert_encoder.py        # BERT文本编码器
│       ├── numeric_encoder.py     # 数值特征编码器
│       ├── fusion_projection.py   # 特征融合投影层
│       ├── multi_modal_model.py   # 多模态融合模型
│       └── ood_head.py            # OOD检测头（基于原型距离）
│
├── utils/                         # 工具模块
│   └── log_utils.py               # 日志记录工具
│
├── tests/                         # 测试套件
│   └── test_project.py            # 项目测试
│
├── docs/                          # 文档
│   ├── theory.md                  # 项目原理说明（旧架构，仅供参考）
│   ├── 代码结构与数据流.md          # 文件功能 + 数据流动图
│   └── Phase1_流程说明.md          # Phase 1 流程说明
│
├── data_processing/               # 原始数据集（需手动放入）
├── processed_dataset/             # 处理后数据（自动生成）
├── split_data/                    # 划分后数据（自动生成）
├── saved_models/                  # 训练模型（自动生成）
├── ablation_results/              # 消融实验结果（自动生成）
├── logs/                          # 操作日志（自动生成）
└── test_reports/                  # 测试报告（自动生成）
```

## 快速开始

### 1. 环境配置

#### 1.1 创建 conda 虚拟环境

```bash
# 创建 Python 3.10 虚拟环境（推荐）
conda create -n multimodal-fusion python=3.10 -y

# 激活虚拟环境
conda activate multimodal-fusion
```

#### 1.2 安装 PyTorch

PyTorch 体积较大（GPU 版约 2.5GB），官方下载源在国内可能较慢。以下提供多种镜像方案加速下载。

##### 方案一：使用上海交大 PyTorch 镜像（推荐）

```bash
# CUDA 12.1
pip install torch>=2.0.0 --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu121

# CUDA 12.4
# pip install torch>=2.0.0 --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu124

# CUDA 11.8
# pip install torch>=2.0.0 --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu118

# CPU 版本
# pip install torch>=2.0.0 --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cpu
```

##### 方案二：清华源 + PyTorch 官方源混合

同时从清华镜像获取依赖包，从 PyTorch 官方源获取 CUDA 轮子：

```bash
# CUDA 12.1
pip install torch>=2.0.0 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  --prefer-binary
```

##### 方案三：PyTorch 官方源（速度较慢）

```bash
# CUDA 12.1
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu121
```

验证 GPU 是否可用：

```bash
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

#### 1.3 安装其余依赖（使用国内镜像加速）

```bash
# 使用清华镜像源安装 requirements.txt 中的依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

其他可选国内镜像源：

| 镜像源 | 地址 |
|--------|------|
| 清华大学 | `https://pypi.tuna.tsinghua.edu.cn/simple` |
| 上海交通大学 | `https://mirror.sjtu.edu.cn/pypi/web/simple` |
| 阿里云 | `https://mirrors.aliyun.com/pypi/simple` |
| 中科大 | `https://pypi.mirrors.ustc.edu.cn/simple` |
| 豆瓣 | `https://pypi.douban.com/simple` |

如需永久设置镜像源，可在 pip 配置文件中添加：

```bash
# Windows: %APPDATA%\pip\pip.ini
# Linux/Mac: ~/.config/pip/pip.conf
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 下载预训练模型

项目已内置 HuggingFace 镜像加速（`hf-mirror.com`），无需额外配置。

```bash
python tools/download_bert.py    # 下载 BERT 模型（约 400MB）
python tools/download_qwen.py    # 下载 Qwen2.5-1.5B 模型（约 3GB）
```

如需手动指定镜像源，可通过环境变量设置：

```bash
# 设置 HuggingFace 镜像（项目脚本默认已使用）
set HF_ENDPOINT=https://hf-mirror.com
```

### 3. 准备数据集

> **数据集下载说明**：本项目实际使用的是 **CSE-CIC-IDS2018（Friday DDoS）** 与 **UNSW-NB15** 两个数据集，评估 BENIGN / DDoS 的开集多模态融合。数据集体积较大，请自行下载后放入 `data_processing/`：
>
> - CSE-CIC-IDS2018：https://www.unb.ca/cic/datasets/ids-2018.html
> - UNSW-NB15：https://research.unsw.edu.au/projects/unsw-nb15-dataset
>
> 数据预处理（清洗 / 子集 / 模态分离）会自动完成格式统一，详见下方「运行项目」。

将原始数据集（CSV格式）放入 `data_processing/` 目录，数据集需包含：
- `Label` 列：值为 `BENIGN`（正常流量）或 `DDoS`（恶意流量）
- 特征列：`Destination Port`, `Bwd Packet Length Mean`, `Avg Bwd Segment Size`, `Bwd Packet Length Max`, `Bwd Packet Length Std`, `URG Flag Count`, `Packet Length Mean`, `Average Packet Size`, `Packet Length Std`

### 4. 运行项目

所有步骤通过统一入口 `main.py` 的 `--run <配置名>` 执行（配置定义见 `configs/run_configs.py`）。先查看可用配置：

```bash
python main.py --list
```

从零跑通闭集（P0）流程：

```bash
python main.py --run clean
python main.py --run subset --dataset_id 1
python main.py --run split  --dataset_id 1
# 训练 A3（无 LLM，纯 MLP，最快）→ 记录终端打印的 model_id
python main.py --run train  --dataset_id 1 --split_id 0 --variant A3
# 训练 A0（含冻结 Qwen，需 GPU）→ 记录 model_id
python main.py --run train  --dataset_id 1 --split_id 0 --variant A0
# 测试 A3
python main.py --run test   --dataset_id 1 --split_id 0 --model_id <A3_model_id>
# 一次性跑完 A0–A3 消融
python main.py --run ablation --dataset_id 1 --split_id 0
```

> 提示：`model_id` 为训练后 `saved_models/` 下的真实目录名（如 `model_9`）。`dataset_id` / `split_id` 建议全程固定（如 1 / 0）。
> 完整命令参数与开集（P1）流程见 `implementation_guide.md`。

## Phase 1: 开集未知攻击检测 + 少样本学习

Phase 1 旨在制造LLM用武之地，通过对比 A3(无LLM) 与 A0(有LLM) 在开集/少样本场景下的性能差距，证明LLM在复杂场景下的独特价值。

### 核心架构：OOD检测头 (OOD Head)

OOD检测头基于**类原型距离**来检测未知样本，实现开集检测能力。

**架构流程**：
```
输入: A3融合特征 (batch, 1536维)
  │
  ├── 计算与每个类原型（可学习向量）的距离
  ├── 距离 → 温度缩放 → softmax → 已知类概率
  ├── 最小距离 > 自适应阈值? → 标记为 unknown
  │
输出: {distances, scores, unknown_mask, pred_labels}
```

**关键设计**：
- **可训练原型**：每个已知类一个可学习的原型向量（`nn.Parameter`）
- **距离度量**：支持欧氏距离/余弦距离/马氏距离
- **自适应阈值**：在验证集上搜索最优F1对应的阈值
- **Center-Loss**：训练时拉近同类特征与原型的距离

### 开集路由机制

```
测试样本
  │
  ├── 提取A3融合特征
  ├── OODHead 判定
  │
  ├── known (distance ≤ threshold)
  │   └── A3分类器 → 预测 benign / known_DDoS
  │
  └── unknown (distance > threshold)
      └── A0 (Qwen LLM) → 语义推理 → "正常流量" or "DDoS攻击"
```

### 执行步骤

> 以下均通过 `main.py --run` 调用（避免直接调用底层脚本）。`<A3_model_id>`、`<A0_model_id>`、`<ood_id>` 为前序步骤产出的真实目录名。

#### Step 1 — 构造开集数据

```bash
python main.py --run openset --dataset_id 1 --source_split_id 0 --unknown_ratio 0.3
```

产物：`split_data/dataset_1/split_openset_0/`（训练/验证集含已知类，测试集含已知+未知）

#### Step 2 — 构造少样本数据

为每个已知类抽取 k 条样本：

```bash
python main.py --run fewshot --dataset_id 1 --source_split_id 0 --k_per_class 5
python main.py --run fewshot --dataset_id 1 --source_split_id 0 --k_per_class 10
python main.py --run fewshot --dataset_id 1 --source_split_id 0 --k_per_class 20
```

产物：`split_data/dataset_1/split_fewshot_<k>_0/`

#### Step 3 — 训练 OOD 检测头

冻结 A3 backbone，仅训练类原型：

```bash
python main.py --run ood_train --model_id <A3_model_id> --dataset_id 1 --split_id 0 --variant A3 --fewshot_k 5
python main.py --run ood_train --model_id <A3_model_id> --dataset_id 1 --split_id 0 --variant A3 --fewshot_k 10
python main.py --run ood_train --model_id <A3_model_id> --dataset_id 1 --split_id 0 --variant A3 --fewshot_k 20
```

产物：`saved_ood_heads/ood_<id>/`（原型向量 + 阈值）

#### Step 4 — 运行 OOD 路由评估

```bash
python main.py --run ood_eval \
    --backbone_model_id <A3_model_id> \
    --ood_id <ood_id> \
    --llm_model_id <A0_model_id> \
    --dataset_id 1 --split_id 0 --fewshot_k 5
```

> 注意：`--llm_model_id` 必须显式传入才会调用 Qwen；不传（默认 None）即 A3-only 基线，用于对照。

产物：`ood_reports/ood_routing_<时间戳>.json`（Unknown F1 / Macro-F1 / Recall / Leak Rate）

#### Step 5 — 一键完整实验

```bash
python main.py --run experiment \
    --backbone_model_id <A3_model_id> \
    --llm_model_id <A0_model_id> \
    --dataset_id 1 --split_id 0 \
    --k_values "5,10,20,None"
```

自动完成：对每个 k 训 OOD 头 → 路由评估 → 生成汇总。

#### Step 6 — 生成汇总报告

```bash
python main.py --run report
```

产物：
- `openworld_fewshot.csv` 按 k 值汇总对比
- `openworld_experiment_summary_<ts>.md` 可读 Markdown 报告
- `openworld_metrics_<timestamp>.json` 结构化指标

### 核心评估指标

| 指标 | 说明 |
|------|------|
| Macro-F1 | 三分类（benign/known/unknown）宏平均F1 |
| **Unknown F1** | 将unknown视为正类的F1（**核心指标**） |
| Unknown Recall | 未知DDoS被正确检测的比例 |
| Unknown Leak Rate | 未知DDoS被误判为known的比例 |
| Benign Recall | 正常流量召回率 |
| Known DDoS Recall | 已知DDoS召回率 |

## 主键体系

项目采用三级主键体系管理数据和模型：

| 主键 | 作用 | 示例 |
|------|------|------|
| `dataset_id` | 标识数据集子集 | `dataset_0.csv`, `dataset_1.csv` |
| `split_id` | 标识同一数据集的不同划分 | `split_data/dataset_0/split_0/` |
| `model_id` | 标识不同的训练结果 | `saved_models/model_0/` |

## 注意事项

### 硬件要求

- **GPU**：推荐 NVIDIA GPU（Ampere架构及以上），支持bfloat16混合精度训练
- **显存**：至少16GB（训练Qwen2.5-1.5B模型）
- **内存**：至少16GB（处理大数据集）

### 数据格式

- 原始数据集：CSV格式，包含`Label`列（值为`BENIGN`或`DDoS`）
- 支持的特征列：`Destination Port`, `Bwd Packet Length Mean`, `Avg Bwd Segment Size`, `Bwd Packet Length Max`, `Bwd Packet Length Std`, `URG Flag Count`, `Packet Length Mean`, `Average Packet Size`, `Packet Length Std`

### 日志系统

所有操作都会自动生成日志文件：

| 日志类型 | 保存位置 | 内容 |
|---------|---------|------|
| 子集提取 | `logs/subset/log_X.json` | 时间、数据集ID、样本数 |
| 数据划分 | `logs/split/log_X.json` | 时间、数据集ID、划分ID |
| 模型训练 | `logs/training/log_X.json` | 时间、模型ID、训练参数 |
| 模型测试 | `test_reports/report_X.json` | 时间、模型ID、评估指标 |
| 开集划分 | `logs/openset_split/` | 开集划分配置与样本统计 |
| OOD训练 | `logs/ood_training/` | OOD头训练过程与指标 |
| OOD路由 | `logs/ood_routing/` | OOD路由评估结果 |
| 少样本划分 | `logs/fewshot_split/` | 少样本划分配置与样本统计 |
| 完整实验 | `logs/experiment/` | 完整实验运行记录 |

### 测试脚本

运行项目测试套件验证所有功能：

```bash
python -m pytest tests/test_project.py -v
```

## 项目原理

本项目采用多模态融合架构，将网络流量数据的两种模态进行融合分类：

1. **统计特征模态**：从网络流量中提取9维数值特征（如包长度均值、端口号等），通过数值编码器进行处理
2. **文本描述模态**：将流量特征转换为自然语言描述，使用预训练BERT模型提取768维语义嵌入
3. **融合投影层**：将两种模态的特征进行融合，通过投影层映射到统一的特征空间
4. **LLM分类**：融合特征输入Qwen2.5 LLM，利用其推理能力进行流量分类
5. **消融实验**：通过控制各模态的启用状态，验证各组件对分类性能的贡献
6. **开集检测 (OOD)**：基于类原型距离检测未知攻击样本，实现开集识别能力
7. **LLM路由**：未知样本交由LLM进行语义推理，已知样本走高效的A3分类器
8. **少样本学习**：在少量训练样本（k=5/10/20）下验证模型的泛化能力

模型架构参考 `原理图.png`，数据处理流程参考 `数据集处理流程图.png`。