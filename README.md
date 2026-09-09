# 多模态融合网络流量分类模型

基于多模态融合的网络流量恶意检测模型，结合统计特征模态和文本描述模态进行流量分类，并支持消融实验以验证各模态的有效性。项目当前已拓展至**开集未知攻击检测**与**少样本学习**场景，旨在证明LLM在复杂场景下的独特优势。

## 项目简介

本项目实现了一个多模态融合模型，用于网络流量恶意检测。模型将网络流量数据转换为两种模态：

1. **统计特征模态**：9维数值特征（如包长度均值、端口号等），经数值编码器处理。
2. **文本描述模态**：将流量特征转换为中文自然语言描述。基线变体 **A3** 用 BERT 提取 768 维语义嵌入；含 LLM 的变体（A0/A0\*/A0_frozen/A0\*_no_num）改用 **Qwen2.5-1.5B 作文本编码器**（对真实中文文本做 mean pooling，输出 1536 维文本向量）。

通过多模态融合技术，在 **LLM 外部**完成数值与文本特征的融合后送入分类头。LLM 在此**不再充当分类器**，而是作为「文本语义编码器」；BERT 仅在 A3 基线中充当文本编码器。项目提供闭集消融实验，对比不同模态组合与 LLM 用法的有效性。

### 核心特性

- **多模态融合**：数值特征与文本语义特征在 LLM 外部高效融合
- **LLM 文本编码器（路线 B）**：Qwen2.5-1.5B 冻结底座 + 可选 LoRA（q/v，r=8）作为文本编码器，A0\* 变体启用
- **闭集消融**：支持 A3 / A0 / A0\* / A0_frozen / A0\*_no_num / A0\*_no_text 六变体对比
- **开集检测 (OOD)**：基于类原型距离检测未知攻击（研究中，见 Phase 1）
- **少样本学习**：支持 k=5/10/20 等少样本训练配置

## 闭集消融变体（6 变体故事线）

> 架构已修正（路线 B）：当 `use_llm=True` 时，LLM 文本编码路径优先，BERT 分支被绕过（即便 `use_bert=True` 也不参与）；融合在 LLM 外部完成。下表「文本来源」以实际生效路径为准。

| 变体 | 数值 | 文本来源 | LLM / LoRA | 角色定位 |
|------|------|---------|-----------|---------|
| A3 | ✓ | BERT（768维） | ✗ | **基线**：数值+BERT，纯 MLP 分类 |
| A0 | ✓ | LLM 文本编码（1536维） | ✓ / 全冻结 | 旧用法对照（无 LoRA，BERT 闲置） |
| A0\* | ✓ | LLM 文本编码（1536维） | ✓ / **LoRA** | ★ 新方案核心：LLM 作文本编码器 + 领域适配 |
| A0_frozen | ✓ | LLM 文本编码（1536维） | ✓ / 全冻结 | LoRA 对照组（与 A0\* 比 LoRA 增益） |
| A0\*_no_num | ✗ | LLM 文本编码（1536维） | ✓ / LoRA | 模态贡献：仅文本 |
| A0\*_no_text | ✓ | 无 | ✗ | 模态贡献：仅数值 |

六变体回答四个递进问题：① LLM 作文本编码器行不行（A0\* vs A3）；② 旧用法是「放错位置」而非「LLM 没用」（A0\* vs A0）；③ LoRA 领域适配有无用（A0\* vs A0_frozen）；④ 数值/文本各自贡献（A0\* / A0\*_no_num / A0\*_no_text）。

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

> **数据集下载说明**：本项目实际使用的是 **CSE-CIC-IDS2017** 与 **UNSW-NB15** 两个数据集，评估多分类（12 类）的开集多模态融合。数据集体积较大，请自行下载后放入 `data_processing/`：
>
> - CSE-CIC-IDS2017：https://www.unb.ca/cic/datasets/ids-2017.html
> - UNSW-NB15：https://research.unsw.edu.au/projects/unsw-nb15-dataset
>
> 数据预处理（清洗 / 子集 / 模态分离）会自动完成格式统一，详见下方「运行项目」。

将原始数据集（CSV格式）放入 `data_processing/` 目录，数据集需包含：
- `Label` 列：取值见 `processed_dataset/label_mapping.json`，为 **12 类**（BENIGN + 11 种攻击，如 DoS Hulk / DDoS / PortScan / Bot / Web Attack 等）
- 特征列：`Destination Port`, `Bwd Packet Length Mean`, `Avg Bwd Segment Size`, `Bwd Packet Length Max`, `Bwd Packet Length Std`, `URG Flag Count`, `Packet Length Mean`, `Average Packet Size`, `Packet Length Std`

### 4. 运行项目

所有步骤通过统一入口 `main.py` 的 `--run <配置名>` 执行（配置定义见 `configs/run_configs.py`）。先查看可用配置：

```bash
python main.py --list
```

从零跑通闭集（P0）流程：

```bash
python main.py --run clean
python main.py --run subset --dataset_id 3
python main.py --run split  --dataset_id 3
# 训练 A3（无 LLM，纯 MLP，最快）→ 记录终端打印的 model_id
python main.py --run train  --dataset_id 3 --split_id 0 --variant A3
# 训练 A0*（含冻结 Qwen + LoRA，需 GPU）→ 记录 model_id
python main.py --run train  --dataset_id 3 --split_id 0 --variant A0*
# 测试 A3
python main.py --run test   --dataset_id 3 --split_id 0 --model_id <A3_model_id>
# 一次性跑闭集消融（默认覆盖 A3 / A0* / A0_frozen；全六变体加 --variants）
python main.py --run ablation --dataset_id 3 --split_id 0
```

> 提示：`model_id` 为训练后 `saved_models/` 下的真实目录名（如 `model_9`）。`dataset_id` / `split_id` 建议全程固定。
> **数据集 ID 说明**：当前分支 `split_data/` 含 `dataset_0/2/3/4/5`（多分类，12/15 类），旧分支的 `dataset_1` 已不存在；请以 `split_data/` 下实际存在的 ID 为准，推荐 **dataset_3（15 类，最难）** 为闭集主基准、**dataset_0（12 类）** 为复现。
> 完整命令参数与开集（P1）流程见 `implementation_guide.md`。

## Phase 1: 开集未知攻击检测 + 少样本学习（研究中，尚未跑通）

> **状态（2026-09-03）**：本阶段对应项目 Phase 1 主假设——「开集（未知攻击）场景下，以 LLM 作文本编码器的主干（A0\*）融合特征，比无 LLM 主干（A3）具有更好的分布外（OOD）可分性，从而在 **Unknown F1 / AUROC** 上显著更优」。该假设**尚未被有效验证**：2026-08-28 的一次探索性运行因 OOD 脚本集成缺陷（`unknown_f1=0.0`）已失效，结论须重做。
>
> **方法调整（重要）**：原计划中「未知样本交由 LLM 生成式路由做语义推理」的设计存在**结构性缺陷**（闭集 `argmax` 永远无法输出 `unknown`，见下方 BUG D），已决定**放弃生成式路由**，改为更干净、可直接支撑假设的方案：**所有主干共用同一个原型距离 OOD 头，公平对比「主干特征质量」**（A3 vs A0_frozen vs A0\*）。

### 核心架构：共享 OOD 检测头（原型距离）

OOD 检测头基于**类原型距离**检测未知样本；同一套头分别接在不同主干的融合特征上，比较的是「主干特征本身的 OOD 可分性」而非检测算法。

```
任一主干 (A3 / A0_frozen / A0*) 的融合特征 (batch, D维)
  │
  ├── OOD 头：计算与每个已知类原型的 distance
  ├── distance → 温度缩放 → softmax → 已知类概率
  ├── 最小 distance > 自适应阈值? → 标记为 unknown
  │
输出: {distances, scores, unknown_mask, pred_labels}
```

**关键设计**：
- **可训练原型**：每个已知类一个可学习的原型向量（`nn.Parameter`）
- **距离度量**：支持欧氏距离/余弦距离/马氏距离
- **自适应阈值**：在验证集上搜索最优 F1 对应的阈值
- **Center-Loss**：训练时拉近同类特征与原型的距离
- **OOD 来源**：用 `make_openset_split.py` 的**留出类别**策略（真实分布偏移），而非随机打标

### ⚠️ 当前执行阻塞（OOD 流水线脚本待修，来自阶段规划）

| BUG | 位置 | 问题 | 后果 |
|-----|------|------|------|
| A | `train_ood_head.py` 165–168/177；`run_ood_routing.py` 320–324/357 | `collate_fn(use_llm=False)` 且只传 `(stat, bert)`，**未传 `input_ids`** | A0\*/A0_frozen 的 LLM 文本分支取不到 input_ids，文本张量退化为**全零**，OOD 流水线里实际只用数值特征（根因） |
| B | 两脚本 `variant_configs` 仅含 A0/A1/A2/A3 | 缺 A0\*/A0_frozen/A0\*_no_num/A0\*_no_text | 无法对比 A0\* vs A3 |
| C | 两脚本 DataLoader 全 `use_llm=False` | 无 input_ids 产出 | 与 BUG A 同源 |
| D | `run_ood_routing.py` 把 unknown 交给 `llm_model.predict()`（闭集 argmax） | 只能输出 0..K-1 已知类，**永不输出 unknown** | 生成式路由结构不可行 → 已决定放弃，改 OOD 头特征质量对比 |

> 上述脚本修复（P0 修集成，约 0.5–1 天）完成后，方可按下方步骤执行；08-28 的 `openworld_*` 旧结果不可引用。

### 执行步骤（待脚本修复后）

> 通过 `main.py --run` 调用。`<A3_id>`、`<A0*_id>`、`<ood_id>` 为前序步骤真实目录名。主战场用 **dataset_3（15 类，最难）**，复现用 dataset_0（12 类）。

#### Step 1 — 构造开集数据（留出类别→unknown）

```bash
python main.py --run openset --dataset_id 3 --source_split_id 0 --unknown_ratio 0.3
```

产物：`split_data/dataset_3/split_openset_0/`（train/val 不含未知，test 含 `unknown`）

#### Step 2 — 构造少样本数据

```bash
python main.py --run fewshot --dataset_id 3 --source_split_id 0 --k_per_class 5
python main.py --run fewshot --dataset_id 3 --source_split_id 0 --k_per_class 10
python main.py --run fewshot --dataset_id 3 --source_split_id 0 --k_per_class 20
```

产物：`split_data/dataset_3/split_fewshot_<k>_0/`

#### Step 3 — 训练 OOD 检测头（在 A3 / A0_frozen / A0\* 主干上各训一个）

```bash
# 先确保对应闭集主干已训好存档
python main.py --run ood_train --model_id <A3_id>       --dataset_id 3 --split_id 0 --variant A3        --fewshot_k 5
python main.py --run ood_train --model_id <A0*_id>      --dataset_id 3 --split_id 0 --variant A0*       --fewshot_k 5
python main.py --run ood_train --model_id <A0_frozen_id> --dataset_id 3 --split_id 0 --variant A0_frozen --fewshot_k 5
```

产物：`saved_ood_heads/ood_<id>/`（原型向量 + 阈值）

#### Step 4 — 评测（OOD 头特征质量对比）

```bash
python main.py --run ood_eval \
    --backbone_model_id <A0*_id> \
    --ood_id <ood_id> \
    --dataset_id 3 --split_id 0 --fewshot_k 5
```

产物：`ood_reports/ood_routing_<时间戳>.json`（Unknown F1 / AUROC / Macro-F1 / Recall）

#### Step 5 — 生成汇总报告

```bash
python main.py --run report
```

产物：
- `openworld_fewshot.csv` 按 k 值 / 主干汇总对比
- `openworld_experiment_summary_<ts>.md` 可读 Markdown 报告
- `openworld_metrics_<timestamp>.json` 结构化指标

### 核心评估指标

| 指标 | 说明 |
|------|------|
| Macro-F1 | 已知类 + unknown 宏平均 F1 |
| **Unknown F1** | 将 unknown 视为正类的 F1（**核心指标**） |
| Unknown Recall | 未知攻击被正确检测的比例 |
| **AUROC** | 阈值无关的 OOD 可分性（推荐主报） |
| Known Macro-F1 | 确认「开集不损闭集」 |

## 主键体系

项目采用三级主键体系管理数据和模型：

| 主键 | 作用 | 示例 |
|------|------|------|
| `dataset_id` | 标识数据集子集 | `dataset_0.csv`, `dataset_3.csv` |
| `split_id` | 标识同一数据集的不同划分 | `split_data/dataset_0/split_0/` |
| `model_id` | 标识不同的训练结果 | `saved_models/model_0/` |

## 注意事项

### 硬件要求

- **GPU**：推荐 NVIDIA GPU（Ampere架构及以上），支持bfloat16混合精度训练
- **显存**：至少16GB（训练Qwen2.5-1.5B模型）
- **内存**：至少16GB（处理大数据集）

### 数据格式

- 原始数据集：CSV格式，包含`Label`列（取值为 12 类，详见 `processed_dataset/label_mapping.json`）
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

本项目采用「LLM 作文本编码器 + 外部融合」的多模态架构，将网络流量数据的两种模态进行融合分类：

1. **统计特征模态**：从网络流量中提取 9 维数值特征（如包长度均值、端口号等），通过数值编码器处理
2. **文本描述模态**：将流量特征转换为中文自然语言描述；A3 基线用 BERT 提取 768 维语义嵌入，含 LLM 变体用 Qwen2.5-1.5B 对真实文本做 mean pooling 得到 1536 维文本向量
3. **融合投影层**：在 LLM **外部**将数值与文本特征拼接融合，投影到统一特征空间
4. **LLM 文本编码器（非分类器）**：Qwen2.5-1.5B 冻结底座 + 可选 LoRA（A0\* 启用），仅负责把中文文本编码成向量，分类由后续分类头完成
5. **闭集消融**：通过控制数值/文本/LLM/LoRA 的开关（6 变体），验证各组件对分类性能的贡献
6. **开集检测 (OOD)**：基于类原型距离检测未知攻击样本，实现开集识别（研究中）
7. **少样本学习**：在少量训练样本（k=5/10/20）下验证模型泛化能力

模型架构参考 `原理图.png`，数据处理流程参考 `数据集处理流程图.png`。