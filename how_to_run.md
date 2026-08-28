# 如何运行本项目

## 快速开始

```bash
# 1. 在 configs/run_configs.py 中修改 RUN_CONFIG 字典里的参数并保存
# 2. 在命令行执行：
python main.py --run <配置键名>

# 3. 也可以在配置键名后追加参数来覆盖 RUN_CONFIG 中的值：
python main.py --run <配置键名> --<参数名> <值>
```

## 示例

```bash
# 运行训练，参数来自 RUN_CONFIG["train"]
python main.py --run train

# 运行训练，但覆盖 dataset_id 和 learning_rate
python main.py --run train --dataset_id 10 --learning_rate 0.001

# 运行数据清洗
python main.py --run clean

# 运行子集提取
python main.py --run subset

# 运行消融实验
python main.py --run ablation

# 运行完整的开集实验
python main.py --run experiment
```

## 工作原理

### 配置

所有运行配置定义在 `configs/run_configs.py` 的 `RUN_CONFIG` 字典中。每个条目结构如下：

```python
RUN_CONFIG = {
    "train": {                              # <-- 配置键名（--run 使用此名称）
        "step": "train",                    # <-- 映射到 main.py 中的步骤
        "params": {                         # <-- 传递给函数的参数
            "variant": "A0",
            "dataset_id": 2,
            "learning_rate": 5e-4,
            "num_train_epochs": 10,
            # ... 更多参数
        }
    },
}
```

### 命令行参数覆盖

在配置键名后追加 `--<参数名> <值>` 会覆盖 `RUN_CONFIG["<配置键名>"]["params"]` 中对应的参数值。未覆盖的参数仍使用 `RUN_CONFIG` 中的值。

| 命令格式 | 示例 | 效果 |
|---|---|---|
| `python main.py --run <键名>` | `python main.py --run train` | 使用 `RUN_CONFIG["train"]["params"]` 的全部参数 |
| `python main.py --run <键名> --<参数> <值>` | `python main.py --run train --dataset_id 10 --learning_rate 0.001` | 在 `RUN_CONFIG` 参数基础上，用命令行值覆盖指定参数 |

### 列出所有可用配置

```bash
python main.py --list
```

该命令会按阶段分组打印所有配置键名及其默认参数。

### 配置键名参考

| 键名 | 阶段 | 说明 |
|---|---|---|
| `clean` | P0 数据预处理 | 数据清洗 |
| `subset` | P0 数据预处理 | 从清洗后的数据中提取子集 |
| `split` | P0 数据预处理 | 模态分离（train/val/test） |
| `train` | P0 模型训练 | 训练多模态融合模型 |
| `test` | P0 模型训练 | 测试已训练的模型 |
| `ablation` | P0 模型训练 | 运行消融实验 |
| `openset` | P1 开集检测 | 构造开集数据集 |
| `fewshot` | P1 少样本 | 构造少样本数据集 |
| `ood_train` | P1 少样本 | 训练 OOD 检测头 |
| `ood_eval` | P1 少样本 | OOD 路由评估 |
| `experiment` | P1 少样本 | 一键完整实验 |
| `report` | P1 少样本 | 生成汇总报告 |

> **注意：** 配置文件中还包含 `*_default` 条目（如 `train_default`、`subset_default`），它们作为参考基准。实际运行时请修改不带 `_default` 后缀的键（如 `train`、`subset`）。