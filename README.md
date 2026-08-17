# 多模态融合网络流量分类模型

基于多模态融合的网络流量恶意检测模型，结合统计特征模态和文本描述模态进行流量分类，并支持消融实验以验证各模态的有效性。

## 项目简介

本项目实现了一个多模态融合模型，用于网络流量恶意检测。模型将网络流量数据转换为两种模态：

1. **统计特征模态**：9维数值特征（如包长度均值、端口号等）
2. **文本描述模态**：768维BERT语义嵌入

通过多模态融合技术，将两种模态的特征进行融合，结合LLM（Qwen2.5）进行分类。项目同时提供消融实验功能，可对比不同模态组合的性能。

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
│   └── run_ablation.py            # 消融实验运行器
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
│       └── multi_modal_model.py   # 多模态融合模型
│
├── utils/                         # 工具模块
│   └── log_utils.py               # 日志记录工具
│
├── tests/                         # 测试套件
│   └── test_project.py            # 项目测试
│
├── docs/                          # 文档
│   └── theory.md                  # 项目原理说明
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

将原始数据集（CSV格式）放入 `data_processing/` 目录，数据集需包含：
- `Label` 列：值为 `BENIGN`（正常流量）或 `DDoS`（恶意流量）
- 特征列：`Destination Port`, `Bwd Packet Length Mean`, `Avg Bwd Segment Size`, `Bwd Packet Length Max`, `Bwd Packet Length Std`, `URG Flag Count`, `Packet Length Mean`, `Average Packet Size`, `Packet Length Std`

### 4. 运行项目

打开 `main.py`，按顺序取消注释执行各步骤：

```python
# 步骤1：数据清洗
from scripts.data_cleaning import main as run_data_cleaning
run_data_cleaning()

# 步骤2：提取子集
success, dataset_id = extract_subset(num_samples=5000, random_state=42)

# 步骤3：模态分离
dataset_id = 0
split_modality(dataset_id=dataset_id, test_size=0.2, random_state=42)

# 步骤4：消融实验（全变体）
run_ablation(variants=["A0","A1","A2","A3"], dataset_id=0, split_id=0, repeat=1, seed=42)

# 步骤5：绘制消融实验结果
results_df = pd.read_csv("ablation_results/ablation_results_时间戳.csv")
plot_f1_comparison(results_df)
```

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

模型架构参考 `原理图.png`，数据处理流程参考 `数据集处理流程图.png`。