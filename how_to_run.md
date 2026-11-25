# 如何运行本项目

本项目所有步骤均通过**直接脚本 CLI** 运行，不再使用 `main.py --run` 统一入口。

> **解释器路径**：连云端实例时通常是非交互 SSH，`python` 不在 PATH。先 `export PY=/root/miniconda3/bin/python`，下文示例中的 `python` 一律替换成 `$PY`。
> **与复现文档的关系**：本文只讲"怎么运行"；论文四条主结论的数值复现见 [`REPRODUCE.md`](REPRODUCE.md)，全部 35 条结论的逐条验证见 [`docs/复现-全结论验证指南-20261125.md`](docs/复现-全结论验证指南-20261125.md)。
> **GPU 前置**：`$PY -c "import torch;print(torch.cuda.is_available())"` 为 `False` 时（无卡模式），只能跑数据预处理、A3 与 CPU 重算；含 LLM 的变体与延迟测量必须先切 GPU 模式。

## 重要：所有实验命令须经 `run.py` 执行

为了记录每次实验运行的元信息（命令、耗时、状态、产物说明等），**所有实验命令必须套一层 `$PY run.py` 前缀**（`$PY=/root/miniconda3/bin/python`）：

```bash
$PY run.py $PY <真实命令>
```

`run.py` 会：
- 自动记录命令内容、执行日期、运行时间、执行状态、退出码、git commit
- 根据命令注册表（`utils/command_registry.py`）填充该命令的产物说明与附加备注
- 仅在发生运行时异常（Traceback）时记录报错信息，命令行参数错误不计入
- 将记录追加到 `run_history/YYYY-MM-DD.jsonl`
- 透传真实命令的退出码（不影响调用方判断）

## 快速开始

在项目根目录下通过 `run.py` 执行脚本：

```bash
export PY=/root/miniconda3/bin/python
$PY run.py $PY scripts/<脚本名>.py --参数
```

例如：

```bash
# 数据清洗
$PY run.py $PY scripts/data_cleaning.py

# 提取子集（主口径见 REPRODUCE.md §1.3：--mode balanced --total_samples 9999 --dataset_id 200）
$PY run.py $PY scripts/extract_subset.py --total_samples 5000 --dataset_id 0

# 模态分离
$PY run.py $PY scripts/split_modality.py --dataset_id 0

# 训练 A3
$PY run.py $PY scripts/train.py --variant A3 --dataset_id 0 --split_id 0

# 测试
$PY run.py $PY scripts/test_model.py --model_id <id> --dataset_id 0 --split_id 0
```

查看某个脚本支持的所有参数：

```bash
$PY run.py $PY scripts/<脚本名>.py --help
```

> 说明：`run.py` 只是日志包装器，不影响数值。`REPRODUCE.md` 中的命令为了可直接粘贴而没有套 `run.py`；若希望每次运行都落 `run_history/`，按本节加外层即可。两处的脚本参数必须一致，不要把 `dataset_0/3`（旧口径）与 `dataset_200`（主口径）混用。

## 完整命令模板

常用命令模板（含参数说明与产物）见 **[`docs/commands.md`](docs/commands.md)**，可直接复制使用；但该文件是 2026-08 快照，主口径（`dataset_200 / split_openset_0`）以 `REPRODUCE.md` 为准。

## 运行方式说明

- 所有脚本均自带独立 CLI（`argparse`），不依赖任何统一入口。
- 脚本运行时会自动创建所需目录（通过 `utils/dirs.py` 的 `ensure_dirs()`）。
- `<id>` 表示前序步骤生成的真实数字 ID（如 `saved_models/model_9` 中的 `9`）。
- 含 LLM 的变体（如 `A0*`）需 GPU；`A3`（无 LLM，纯 MLP）可在 CPU 上快速验证。
- **所有实验命令必须通过 `$PY run.py $PY scripts/...` 执行**，以便自动记录到 `run_history/`。直接执行脚本会绕过记录机制。
