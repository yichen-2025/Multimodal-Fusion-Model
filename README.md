# 多模态融合网络流量分类模型

基于多模态融合的网络流量恶意检测模型，结合统计特征模态和文本描述模态进行流量分类，并支持消融实验以验证各模态的有效性。项目当前聚焦**开集未知攻击检测**场景，并已完成「LLM 主干能否带来开集增益」的受控检验。四条主结论已冻结为 `docs/claims.md` v2.7（tag `claims-v2.7-frozen`），逐实验证据见 `results/2026-11-20/SUMMARY-2026-11-20.md`，论文初稿见 `docs/第1章-绪论.md` ～ `docs/第9章-结论.md`。

## 项目简介

本项目实现了一个多模态融合模型，用于网络流量恶意检测。模型将网络流量数据转换为两种模态：

1. **统计特征模态**：79 维数值特征（如包长度均值、端口号等），经数值编码器处理。
2. **文本描述模态**：将流量特征转换为中文自然语言描述。基线变体 **A3** 用 BERT 提取 768 维语义嵌入；含 LLM 的变体（A0/A0\*/A0_frozen/A0\*_no_num）改用 **Qwen2.5-1.5B 作文本编码器**（对真实中文文本做 mean pooling，输出 1536 维文本向量）。

通过多模态融合技术，在 **LLM 外部**完成数值与文本特征的融合后送入分类头。LLM 在此**不再充当分类器**，而是作为「文本语义编码器」；BERT 仅在 A3 基线中充当文本编码器。项目提供闭集消融实验，对比不同模态组合与 LLM 用法的有效性。

### 核心特性

- **多模态融合**：数值特征与文本语义特征在 LLM 外部高效融合
- **LLM 文本编码器（路线 B）**：Qwen2.5-1.5B 冻结底座 + 可选 LoRA（q/v，r=8）作为文本编码器，A0\* 变体启用
- **闭集消融**：支持 A3 / A0 / A0\* / A0_frozen / A0\*_no_num / A0\*_no_text 六变体对比
- **开集检测 (OOD)**：基于类原型距离检测未知攻击（见 Phase 1）

## 研究结论（4 条，已冻结 2026-11-22）

**统一口径**：单数据集 **CSE-CIC-IDS2017**（`dataset_200 / split_openset_0`），10 个已知类 + 留出 `Web Attack` 作为未知；79 维数值特征 + 20 字段中文模板文本；编码器为 BERT-base（A3）与 Qwen2.5-1.5B（A0\*）。

| 贡献 | 结论 | 状态 | 主要产物 |
|---|---|---|---|
| **C1** | LLM 作文本编码器，相对 BERT 无可检测的 OOD 增益 | 未支持（+1.87pp，95% CI [−8.85, +12.59] 跨零） | `results/2026-11-03/` |
| **C2** | 文本是数值特征的模板化重编码，独立增量未检出超过约 **6%** | 部分证实（去文本会掉点：冗余 ≠ 无效） | `results/2026-11-19/e8_audit*`、`results/2026-11-20/{e10_*,e11_*}` |
| **C3** | LLM 语义原型在零样本 / 少样本 / 开集上均未达预注册门槛 | 未支持（三个编码器空间 × k∈{1,5,10,20} 全负） | `results/2026-11-20/{e3_e4_bert,e5_qwen,e6_*,e9_qwen7b}` |
| **C4** | 成本-精度前沿在 79 维 MLP 处饱和（约 1040× 延迟换 +0.85pp） | 已证实（描述性） | `results/2026-11-20/c4_latency/` |

口径纪律：AUROC 以未知为正类；α/τ 只在 val 上选；负结论只能写「未支持 / 不可区分」，不得反向断言。逐条编号、边界与状态以 [`docs/claims.md`](docs/claims.md) 为唯一登记处。

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
│   ├── report_generator.py       # [P1.5] 结果汇总报告生成器
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
│   ├── dirs.py                    # 必需目录统一创建（ensure_dirs）
│   └── log_utils.py               # 日志记录工具
│
├── tests/                         # 测试套件
│   └── test_project.py            # 项目测试
│
├── docs/                          # 文档
│   ├── commands.md                # 命令模板（参数说明+产物，直接复制用）
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

> **口径（2026-11-24）**：本仓库当前只使用 **CSE-CIC-IDS2017**（`dataset_200`），10 个已知类 + 留出 `Web Attack` 作为未知类，79 维数值特征 + 20 字段中文模板文本。此前文档中的「CSE-CIC-IDS2017 + UNSW-NB15、12 类」属 2026-08 旧口径，已废弃。
>
> 原始 CSV（约 1.1 GB）、处理后的 `split_data/`（约 350 MB）与预训练权重（BERT / Qwen2.5-1.5B）均**不入库**。最小复现数据包（`split_openset_0/{train,val,test}.npz`）与权重获取方式将随 GitHub Release + `REPRODUCE.md` 提供（整理中）。

将 CSE-CIC-IDS2017 原始 CSV 放入 `data_processing/`（官方下载：https://www.unb.ca/cic/datasets/ids-2017.html），然后依次运行：

```bash
python scripts/data_cleaning.py
python scripts/extract_subset.py        # 子集（目标 dataset_200；参数见 --help）
python scripts/split_modality.py        # 生成 split_data/dataset_200/
python scripts/make_openset_split.py    # 生成 split_openset_0（留出 Web Attack）
```

具体参数以各脚本 `--help` 为准（`docs/commands.md` 为 2026-08 快照，随复现整理一并更新）。

数据约定：
- `Label` 列：10 个已知类（BENIGN、DoS Hulk、DoS GoldenEye、DoS slowloris、DoS Slowhttptest、DDoS、PortScan、FTP-Patator、SSH-Patator、Bot），`Web Attack` 留出为未知；类名映射见 `saved_ood_heads/ood_head_39/config.json` 的 `label_mapping`
- 数值特征：**79 维**（`split_*` 中的 `scaled_features`）
- 文本：由 20 个字段生成中文模板（`utils/flow_features.py::generate_text_description`），**不是**手工挑选的 9 列

### 4. 运行项目

所有步骤通过**直接脚本 CLI** 运行（`python scripts/xxx.py --参数`），不再使用统一入口。完整命令模板见 [`docs/commands.md`](docs/commands.md)。

从零跑通闭集（P0）流程：

```bash
python scripts/data_cleaning.py
python scripts/extract_subset.py --dataset_id 3
python scripts/split_modality.py --dataset_id 3
# 训练 A3（无 LLM，纯 MLP，最快）→ 记录终端打印的 model_id
python scripts/train.py --dataset_id 3 --split_id 0 --variant A3
# 训练 A0*（含冻结 Qwen + LoRA，需 GPU）→ 记录 model_id
python scripts/train.py --dataset_id 3 --split_id 0 --variant A0*
# 测试 A3
python scripts/test_model.py --dataset_id 3 --split_id 0 --model_id <A3_model_id>
# 一次性跑闭集消融（默认覆盖 A3 / A0* / A0_frozen；全六变体加 --variants）
python scripts/run_ablation.py --dataset_id 3 --split_id 0
```

> 提示：`model_id` 为训练后 `saved_models/` 下的真实目录名（如 `model_9`）。`dataset_id` / `split_id` 建议全程固定。
> **数据集 ID 说明**：当前分支 `split_data/` 含 `dataset_0/2/3/4/5`（多分类，12/15 类），旧分支的 `dataset_1` 已不存在；请以 `split_data/` 下实际存在的 ID 为准，推荐 **dataset_3（15 类，最难）** 为闭集主基准、**dataset_0（12 类）** 为复现。
> 完整命令参数与开集（P1）流程见 `implementation_guide.md`。

## Phase 1: 开集未知攻击检测

> **更新（2026-11-03）**：A3 对照实验已在 `dataset_200` / `split_openset_0` 上完成（3 主干 × 3 头），结论为「A3 与 A0\*_no_num 统计上不可区分；与 A0\* 呈 AUROC 更高 / 低误报召回更低的 trade-off」，详见 `docs/stage_conclusion_20261103.md`。本节其余内容（dataset_3 命令、08-28 失效运行、脚本 BUG 清单）是 2026-09-03 的历史记录，与当前口径不一致，待重写。

> **状态（2026-09-03，历史）**：本阶段对应项目 Phase 1 主假设——「开集（未知攻击）场景下，以 LLM 作文本编码器的主干（A0\*）融合特征，比无 LLM 主干（A3）具有更好的分布外（OOD）可分性，从而在 **Unknown F1 / AUROC** 上显著更优」。写下这段时该假设**尚未被有效验证**（2026-08-28 的一次探索性运行因 OOD 脚本集成缺陷 `unknown_f1=0.0` 已失效）；**该假设已于 2026-11-03 用 A3 对照检验，结论见上方更新与 `docs/stage_conclusion_20261103.md`**。
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

### 历史记录：OOD 流水线集成缺陷（2026-09 已修复，仅留档）

| BUG | 位置 | 问题 | 后果 |
|-----|------|------|------|
| A | `train_ood_head.py` 165–168/177；`run_ood_routing.py` 320–324/357 | `collate_fn(use_llm=False)` 且只传 `(stat, bert)`，**未传 `input_ids`** | A0\*/A0_frozen 的 LLM 文本分支取不到 input_ids，文本张量退化为**全零**，OOD 流水线里实际只用数值特征（根因） |
| B | 两脚本 `variant_configs` 仅含 A0/A1/A2/A3 | 缺 A0\*/A0_frozen/A0\*_no_num/A0\*_no_text | 无法对比 A0\* vs A3 |
| C | 两脚本 DataLoader 全 `use_llm=False` | 无 input_ids 产出 | 与 BUG A 同源 |
| D | `run_ood_routing.py` 把 unknown 交给 `llm_model.predict()`（闭集 argmax） | 只能输出 0..K-1 已知类，**永不输出 unknown** | 生成式路由结构不可行 → 已决定放弃，改 OOD 头特征质量对比 |

> 上述缺陷已于 2026-09 修复。截至 2026-11 已产出 43 个 backbone、86 个 OOD 头与 237 份路由报告，C1（C-25）的 A3 对照即在此基础之上完成。下方「执行步骤」为历史命令；现行命令见 `docs/commands.md` 与 `how_to_run.md`。

### 执行步骤（历史命令，保留用于追溯）

> 通过直接脚本 CLI 调用。`<A3_id>`、`<A0*_id>`、`<ood_id>` 为前序步骤真实目录名。**注意：本节命令对应 2026-09 的 dataset_3（15 类）/ dataset_0（12 类）旧口径，现行主口径为 `dataset_200` / `split_openset_0`（10 类 + Web Attack 未知），开集流程见 `docs/commands.md`。**

#### Step 1 — 构造开集数据（留出类别→unknown）

```bash
python scripts/make_openset_split.py --dataset_id 3 --source_split_id 0 --simulated_unknown_ratio 0.3
```

产物：`split_data/dataset_3/split_openset_0/`（train/val 不含未知，test 含 `unknown`）

#### Step 2 — 训练 OOD 检测头（在 A3 / A0_frozen / A0\* 主干上各训一个）

```bash
# 先确保对应闭集主干已训好存档
python scripts/train_ood_head.py --model_id <A3_id>       --dataset_id 3 --split_id 0 --variant A3
python scripts/train_ood_head.py --model_id <A0*_id>      --dataset_id 3 --split_id 0 --variant A0*
python scripts/train_ood_head.py --model_id <A0_frozen_id> --dataset_id 3 --split_id 0 --variant A0_frozen
```

产物：`saved_ood_heads/ood_<id>/`（原型向量 + 阈值）

#### Step 3 — 评测（OOD 头特征质量对比）

```bash
python scripts/run_ood_routing.py \
    --backbone_model_id <A0*_id> \
    --ood_id <ood_id> \
    --dataset_id 3 --split_id 0
```

产物：`ood_reports/ood_routing_<时间戳>.json`（Unknown F1 / AUROC / Macro-F1 / Recall）

#### Step 4 — 生成汇总报告

```bash
python scripts/report_generator.py
```

产物：
- `openworld_summary.csv` 按主干汇总对比
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

- 原始数据集：CSV 格式，包含 `Label` 列；当前口径为 **10 个已知类 + 1 个留出未知类**（Web Attack），映射见 `saved_ood_heads/ood_head_39/config.json`
- 数值特征：**79 维**；文本由 20 个字段生成（`utils/flow_features.py::TEXT_FEATURES` / `generate_text_description`），长度上限 256

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
| 完整实验 | `logs/experiment/` | 完整实验运行记录 |

### 测试脚本

运行项目测试套件验证所有功能：

```bash
python -m pytest tests/test_project.py -v
```

## 项目原理

本项目采用「LLM 作文本编码器 + 外部融合」的多模态架构，将网络流量数据的两种模态进行融合分类：

1. **统计特征模态**：从网络流量中提取 79 维数值特征（如包长度均值、端口号等），通过数值编码器处理（历史文档曾写作“9 维”，属旧口径，与实际特征表不符）
2. **文本描述模态**：将流量特征转换为中文自然语言描述；A3 基线用 BERT 提取 768 维语义嵌入，含 LLM 变体用 Qwen2.5-1.5B 对真实文本做 mean pooling 得到 1536 维文本向量
3. **融合投影层**：在 LLM **外部**将数值与文本特征拼接融合，投影到统一特征空间
4. **LLM 文本编码器（非分类器）**：Qwen2.5-1.5B 冻结底座 + 可选 LoRA（A0\* 启用），仅负责把中文文本编码成向量，分类由后续分类头完成
5. **闭集消融**：通过控制数值/文本/LLM/LoRA 的开关（6 变体），验证各组件对分类性能的贡献
6. **开集检测 (OOD)**：基于类原型距离检测未知攻击样本，实现开集识别（研究中）

模型架构参考 `原理图.png`，数据处理流程参考 `数据集处理流程图.png`。
## 适用边界

- **少样本线已检验且为负**：2026-11-20 在 BERT / Qwen1.5B / Qwen7B 三个空间做了 k∈{1,5,10,20} × 5 seed，语义原型增益未达 +3pp 预注册门槛（C-20 未支持），见 `results/2026-11-20/`。
- 主结论来自**单一数据集**（`dataset_200` / `split_openset_0`）、**单一留出家族**（Web Attack）、单模板文本；不得外推为「通用未知检测」或「跨数据集泛化」。A3 对照的置信度受 backbone seed n=5 限制。
- **未做人类评测**：解释层只报客观指标，不写 `actionability` / `sufficiency` 的正向结论。
- 扩展项 E12（第二数据集）/ E13（非 Qwen 家族）在 `docs/claims.md` 登记为 `待检验`，不影响上述四条。

## 生成物台账

`saved_models/`、`saved_ood_heads/`、`test_reports/`、`ood_reports/` 中哪些产物有效、哪些仅为历史证据，统一以 [artifacts/README.md](artifacts/README.md) 与 `artifacts/ledger.csv` 为准。审查代码或引用历史指标前请先读台账。
