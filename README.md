# 多模态融合网络流量分类模型

基于多模态融合的网络流量恶意检测模型，结合统计特征模态和文本描述模态进行流量分类。

## 项目简介

本项目实现了一个多模态融合模型，用于网络流量恶意检测。模型将网络流量数据转换为两种模态：

1. **统计特征模态**：9维数值特征（如包长度均值、端口号等）
2. **文本描述模态**：768维BERT语义嵌入

通过多模态融合技术，将两种模态的特征进行融合，实现更准确的流量分类。

## 目录结构

```
多模态融合2/
├── main.py                    # 主入口文件（推荐运行方式）
├── data_cleaning.py           # 数据清洗脚本
├── extract_subset.py          # 数据集子集提取脚本
├── split_modality.py          # 模态分离与数据集划分脚本
├── train.py                   # 模型训练脚本
├── test_model.py              # 模型测试脚本
├── test_project.py            # 项目测试套件
├── download_bert.py           # BERT模型下载脚本
├── download_qwen.py           # Qwen模型下载脚本
│
├── src/                       # 核心源码
│   ├── data/
│   │   └── data_loader.py     # 数据加载模块
│   └── model_architectures/   # 模型架构
│       ├── bert_encoder.py    # BERT文本编码器
│       ├── numeric_encoder.py # 数值特征编码器
│       ├── fusion_projection.py # 特征融合投影层
│       └── multi_modal_model.py # 多模态融合模型
│
├── utils/
│   └── log_utils.py           # 日志记录工具
│
├── docs/                      # 文档
│   └── theory.md              # 项目原理说明
│
├── data_processing/           # 原始数据集（需手动放入）
├── processed_dataset/         # 处理后数据（自动生成）
├── split_data/                # 划分后数据（自动生成）
├── saved_models/              # 训练模型（自动生成）
├── logs/                      # 操作日志（自动生成）
└── test_reports/              # 测试报告（自动生成）
```

## 快速开始

### 1. 安装依赖

```bash
pip install torch transformers datasets scikit-learn pandas numpy pytest matplotlib
```

### 2. 下载预训练模型

```bash
python download_bert.py
python download_qwen.py
```

### 3. 准备数据集

将原始数据集（CSV格式）放入 `data_processing/` 目录，数据集需包含：
- `Label` 列：值为 `BENIGN`（正常流量）或 `DDoS`（恶意流量）
- 特征列：`Destination Port`, `Bwd Packet Length Mean`, `Avg Bwd Segment Size`, `Bwd Packet Length Max`, `Bwd Packet Length Std`, `URG Flag Count`, `Packet Length Mean`, `Average Packet Size`, `Packet Length Std`

### 4. 运行项目

打开 `main.py`，按顺序取消注释执行各步骤：

```python
# 步骤1：数据清洗
from data_cleaning import main as run_data_cleaning
run_data_cleaning()

# 步骤2：提取子集
success, dataset_id = extract_subset(num_samples=5000, random_state=42)

# 步骤3：模态分离
dataset_id = 0
split_modality(dataset_id=dataset_id, test_size=0.2, random_state=42)

# 步骤4：模型训练
train_model(model_path="./models/qwen2.5-1.5b", dataset_id=0, split_id=0, num_train_epochs=3)

# 步骤5：模型测试
result = test_model(model_id=0, dataset_id=0, split_id=0, verbose=True)

# 步骤6：绘制loss曲线
plot_loss_curve(model_id=0)
```

## 运行步骤详解

### 步骤1：数据清洗与预处理

**操作**：取消 `main.py` 中步骤1的注释

```python
from data_cleaning import main as run_data_cleaning
run_data_cleaning()
```

**运行**：`python main.py`

**输入**：`data_processing/` 目录下的CSV文件

**输出**：`processed_dataset/processed_dataset.csv`

**处理内容**：缺失值处理、异常值检测、标签编码（BENIGN→0, DDoS→1）

### 步骤2：提取数据集子集

**操作**：取消 `main.py` 中步骤2的注释

```python
success, dataset_id = extract_subset(
    num_samples=5000,
    # dataset_id=0,  # 可选：指定数据集ID
    random_state=42
)
```

**运行**：`python main.py`

**参数说明**：
- `num_samples`: 提取样本数量（默认5000）
- `dataset_id`: 数据集ID（默认自动递增）
- `random_state`: 随机种子（默认42）

**输出**：
- `processed_dataset/dataset_X.csv`（子集数据）
- `processed_dataset/subset_X_scaled_features.npy`（标准化特征）
- `processed_dataset/subset_X_labels.npy`（标签）

### 步骤3：模态分离与数据集划分

**操作**：取消 `main.py` 中步骤3的注释

```python
dataset_id = 0
split_modality(
    dataset_id=dataset_id,
    # split_id=0,  # 可选：指定划分ID
    test_size=0.2,
    random_state=42
)
```

**运行**：`python main.py`

**参数说明**：
- `dataset_id`: 数据集ID（默认0）
- `split_id`: 划分ID（默认自动递增）
- `test_size`: 测试集比例（默认0.2）

**输出**：`split_data/dataset_X/split_Y/`
- `train.npz`（训练集：统计特征 + BERT嵌入 + 标签）
- `test.npz`（测试集：统计特征 + BERT嵌入 + 标签）

### 步骤4：模型训练

**操作**：取消 `main.py` 中步骤4的注释

```python
train_model(
    model_path="./models/qwen2.5-1.5b",
    # model_id=0,  # 可选：指定模型ID
    dataset_id=0,
    split_id=0,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=3
)
```

**运行**：`python main.py`

**参数说明**：
- `model_path`: LLM模型路径（默认`./models/qwen2.5-1.5b`）
- `model_id`: 模型ID（默认自动递增）
- `learning_rate`: 学习率（默认1e-4）
- `num_train_epochs`: 训练轮数（默认3）

**输出**：`saved_models/model_X/`
- `pytorch_model.bin`（模型参数）
- `config.txt`（训练配置）
- `loss_logs/loss_log.csv`（训练loss记录）

### 步骤5：模型测试

**操作**：取消 `main.py` 中步骤5的注释

```python
result = test_model(
    model_id=0,
    dataset_id=0,
    split_id=0,
    verbose=True
)
```

**运行**：`python main.py`

**输出**：
- 控制台打印评估指标（准确率、精确率、召回率、F1值）
- `test_reports/report_X.json`（测试报告）

### 步骤6：绘制loss曲线

**操作**：取消 `main.py` 中步骤6的注释

```python
plot_loss_curve(model_id=0)
```

**运行**：`python main.py`

**效果**：弹出窗口显示训练loss变化曲线

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
python -m pytest test_project.py -v
```

## 项目原理

详细的项目原理说明请参考 `docs/theory.md`。
