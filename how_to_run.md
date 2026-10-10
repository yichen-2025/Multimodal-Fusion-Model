# 如何运行本项目

本项目所有步骤均通过**直接脚本 CLI** 运行，不再使用 `main.py --run` 统一入口。

## 快速开始

在项目根目录下直接执行脚本：

```bash
python scripts/<脚本名>.py --参数
```

例如：

```bash
# 数据清洗
python scripts/data_cleaning.py

# 提取子集
python scripts/extract_subset.py --total_samples 5000 --dataset_id 0

# 模态分离
python scripts/split_modality.py --dataset_id 0

# 训练 A3
python scripts/train.py --variant A3 --dataset_id 0 --split_id 0

# 测试
python scripts/test_model.py --model_id <id> --dataset_id 0 --split_id 0
```

查看某个脚本支持的所有参数：

```bash
python scripts/<脚本名>.py --help
```

## 完整命令模板

常用命令模板（含参数说明与产物）见 **[`docs/commands.md`](docs/commands.md)**，可直接复制使用。

## 运行方式说明

- 所有脚本均自带独立 CLI（`argparse`），不依赖任何统一入口。
- 脚本运行时会自动创建所需目录（通过 `utils/dirs.py` 的 `ensure_dirs()`）。
- `<id>` 表示前序步骤生成的真实数字 ID（如 `saved_models/model_9` 中的 `9`）。
- 含 LLM 的变体（如 `A0*`）需 GPU；`A3`（无 LLM，纯 MLP）可在 CPU 上快速验证。
