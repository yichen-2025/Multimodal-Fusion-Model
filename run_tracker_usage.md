# Run Tracker — 运行命令追踪器

## 这个模块做什么

每次通过 `python run.py ...` 执行 Python 脚本时，追踪器会自动往 `run_history/YYYY-MM-DD.jsonl` 里追加一行 JSON，共 14 个字段：

| 字段 | 类型 | 示例 |
|------|------|------|
| `id` | UUID | `550e8400-e29b-41d4-a716-446655440000` |
| `timestamp` | ISO 8601 | `2026-10-08T20:15:32` |
| `duration_seconds` | float | `154.23` |
| `duration_human` | string | `2m34s` |
| `command` | string | `python scripts/train.py --lr 1e-4` |
| `working_dir` | string | `D:\...\Multimodal-Fusion-Model` |
| `exit_code` | int | `0` |
| `status` | string | `success` / `error` / `failed` / `interrupted` / `system_error` |
| `error_message` | string \| null | `ModuleNotFoundError: No module named 'faiss'` |
| `extracted_metrics` | object \| null | `{"AUROC": 0.873, "uF1": 0.812}` |
| `stdout_tail` | string | stdout 最后 50 行 |
| `stderr_tail` | string \| null | stderr 最后 20 行（仅失败时） |
| `git_branch` | string | `feature/run-tracker` |
| `git_commit` | string | `4f2d909` |

核心设计思路：

- **零侵入**：不改任何现有脚本。Wrapper 只是站在 `python` 前面替你多做了一层事。
- **stdout/stderr 实时透传**：终端输出和直接跑脚本看起来一模一样。只在结尾多一行 `[run.py] ✓ 已记录...`。
- **subprocess 退出码保真**：`run.py` 返回和真实脚本完全相同的 exit_code，CI/自动化不会因此中断。
- **预检防噪**：文件不存在、语法错误、`py_compile` 失败在 `subprocess.Popen` 之前就拦截，不会污染 `run_history/`。
- **并发安全**：`threading.Lock` 保护 jsonl 写入；`flush + fsync` 保证崩溃安全。
- **全标准库**：不需要 `pip install` 任何东西。

---

## 怎么用

### 基本用法

```powershell
# 之前（不追踪）
python scripts/train.py --variant A0_add --seed 23 --lr 1e-4

# 现在（自动追踪）
python run.py scripts/train.py --variant A0_add --seed 23 --lr 1e-4
```

就是把 `python` 换成 `python run.py`，其他参数一个不动。

### 跳过语法预检

```powershell
python run.py --no-check scripts/train.py ...
```

跳过轻量的 `py_compile` 预检（每次省大约 50ms）。

### 成功时终端看到什么

```
# ... 脚本自身的一切输出，和之前完全一样 ...

[run.py] ✓ 已记录  状态=success  耗时=2m34s  退出码=0
```

### 失败时终端看到什么

```
Traceback (most recent call last):
  File "scripts/train.py", line 5, in <module>
    import nonexistent_module
ModuleNotFoundError: No module named 'nonexistent_module'

[run.py] ✓ 已记录  状态=error  耗时=0.23s  退出码=1
```

### Ctrl+C 中断时终端看到什么

```
# ... 部分训练输出 ...

[run.py] 进程被用户中断
[run.py] ✓ 已记录  状态=interrupted  耗时=10m23s  退出码=-1
```

注意：哪怕手动 kill 了，**已跑的时间照样记**。对长脚本特别有用——以后看历史时知道"上次跑到第 15 epoch 花了 10 分钟"。

### 文件不存在时（预检拦截，run_history/ 不生成任何东西）

```
python run.py scripts/nonexistent.py

[run.py] 错误：文件不存在 → scripts/nonexistent.py
```

---

## 指标自动提取（Layer 2 — 正则，不需要改任何代码）

追踪器对完整 stdout 跑一组正则，命中的值存入 `extracted_metrics`。它是**尽力而为**——没命中的脚本就得到空 `{}`，`stdout_tail` 永远兜底。

### 目前覆盖的指标

| 指标 | 正则模式 | 哪些脚本会打印它 |
|------|---------|----------------|
| AUROC | `(?i)\bAUROC[^\d]*([\d.]+)` | `run_multiseed_ood.py`、`run_ood_routing.py`、`run_openworld_experiment.py`、`analyze_multiseed_20260921.py` |
| uF1 | `(?i)\bu[ _-]?f1[^\d]*([\d.]+)` | `run_multiseed_ood.py`、`run_ood_routing.py`、`test_model.py` |
| unknown_f1 | `(?i)unknown[_ ]?f1[^\d]*([\d.]+)` | 同上 |
| macro_f1 | `(?i)(?:macro[_ -]?f1\|macro_f1)[^\d]*([\d.]+)` | 所有 OOD + 消融 + 测试脚本 |
| known_accuracy | `(?i)known[_ -]?accuracy[^\d]*([\d.]+)` | `run_multiseed_ood.py`、`test_model.py` |
| unknown_leak_rate | `(?i)(?:unknown[_ -]?leak\|leak(?:age)?[_ -]?rate)[^\d]*([\d.]+)` | `run_multiseed_ood.py`、`test_model.py` |
| Accuracy | `(?i)(?:整体)?accuracy[^\d]*([\d.]+)` | `run_ablation.py`、`test_model.py` |
| Precision / Recall | `precision` / `(?:unknown[_ -]?recall\|recall)` | `run_ablation.py`、`test_model.py`、`run_ood_routing.py` |
| F1（无修饰） | `(?i)(?<!macro[_ -])(?<!unknown[_ -])\bf1[^\d]*([\d.]+)` | `train.py`、`run_ablation.py` |
| Loss | `(?i)\bloss[^\d]*([\d.]+)` | `train_ood_head.py` |
| FAIL / WARN | `FAIL[^\d]*(\d+)` / `WARN[^\d]*(\d+)` | `verify_bugfixes.py` |

### 一个需要注意的点

正则匹配 stdout 里**第一次出现**的值。如果一个脚本多次打印同一个指标（比如 `run_multiseed_ood.py` 为每个 seed × variant 组合打印一行 AUROC），那么 `extracted_metrics` 里存的是第一组 seed 的值，不是最后的汇总平均值。

**想看汇总怎么办？** `stdout_tail` 永远保留最后 50 行，汇总表一定在那里。

---

## 指标提取（Layer 3 — 脚本配合，可选）

如果你想要 100% 可靠的提取、不想靠正则猜，可以在常跑的脚本里加一行显式标记：

```python
# 在 run_multiseed_ood.py 汇总表打印完之后加这一行
print(f"[FINAL_RESULT] mean_auroc={mean_auroc:.4f} mean_uF1={mean_uf1:.4f} "
      f"mean_macroF1={mean_macro_f1:.4f} mean_leak={mean_leak:.4f}")
```

追踪器会**先检查 `[FINAL_RESULT]`**，没找到才跑 Layer 2 的正则。找到了就直接解析 `key=value key=value ...`，这些值就是权威结果。

**Layer 3 完全可选**。哪个脚本跑得多就给哪个加一行，一次一行，不常跑的脚本永远可以不改。

---

## status 值说明

| status | 什么情况下 | exit_code | error_message |
|--------|-----------|-----------|---------------|
| `success` | 正常退出 | `0` | null |
| `error` | 环境 / 语法问题 | 非零 | ✅ 有 |
| `failed` | 运行时异常（OOM、空指针等） | 非零 | ✅ 有 |
| `interrupted` | 用户按了 Ctrl+C | `-1` | null |
| `system_error` | `OSError`（文件权限、路径过长） | `-1` | ✅ 有 |

`error` 和 `failed` 的区别：如果 stderr 里出现 `SyntaxError`、`argparse`、`ModuleNotFoundError`、`ImportError` → 归为 `error`；其他运行时异常 → 归为 `failed`。

---

## 日志文件存在哪里

```
Multimodal-Fusion-Model/
├── run.py                          ← Wrapper 入口
├── utils/
│   └── run_tracker.py              ← 核心追踪逻辑
└── run_history/                    ← 自动创建，已在 .gitignore 里
    ├── 2026-10-08.jsonl            ← 每天一个文件
    ├── 2026-10-09.jsonl
    └── ...
```

每行一条 JSON。只追加，不修改、不删除已有记录。

---

## 怎么查看运行历史

### 查看今天最新一条

```powershell
python -c "import json; lines=open('run_history/2026-10-08.jsonl', encoding='utf-8').readlines(); print(json.dumps(json.loads(lines[-1]), indent=2, ensure_ascii=False))"
```

### 用 pandas 读今天所有记录

```python
import pandas as pd

df = pd.read_json("run_history/2026-10-08.jsonl", lines=True)
print(df[['timestamp', 'command', 'status', 'duration_human', 'exit_code']])
```

### 合并多个日期的记录

```python
from pathlib import Path
import pandas as pd

frames = []
for path in sorted(Path("run_history").glob("2026-*.jsonl")):
    frames.append(pd.read_json(path, lines=True))

df = pd.concat(frames, ignore_index=True)
```

### 快速看错误率

```python
print(df['status'].value_counts())
# success        12
# failed          2
# error           1
# interrupted     1
```

### 快速看所有 OOD 实验的 AUROC 均值

```python
ood_runs = df[df['command'].str.contains('run_multiseed_ood|run_ood_routing')]
aurocs = ood_runs['extracted_metrics'].dropna().apply(lambda m: m.get('AUROC'))
print(aurocs.describe())
```

---

## 鲁棒性保障

| 风险 | 应对方式 |
|------|---------|
| 多个终端同时写入 jsonl | `threading.Lock` 串行化 |
| 写入过程中断电 | `flush + os.fsync` 保证落盘；JSONL 只追加，最多最后一行被截断 |
| 磁盘空间不足 | 写入前用 `shutil.disk_usage` 检查，剩余 < 100MB 时警告 |
| 单日日志文件过大 | 单个 jsonl 超过 50MB 时警告 |
| `run_tracker.py` 自身 import 失败 | 整个追踪逻辑包在 try/except 里；真实命令照常执行，只是跳过追踪 |
| 中文路径 + Windows | 所有 subprocess 调用显式传 `encoding='utf-8', errors='replace'` |
| 用户忘了加 `--no-check` | 预检本身就很轻（<100ms），内核还会缓存编译结果 |

---

## 刻意不做的事

| 不做 | 原因 |
|------|------|
| 记录每个 epoch / 每个 batch | 追踪器定位是**命令级**，不是**训练级**。训练过程已经有 `logs/` 和 `utils/log_utils.py` 里的 `save_log()` 覆盖了 |
| 为每个脚本写专属解析器 | 16 个脚本 × 格式频繁变动 = 维护爆炸。正则 + Layer 3 足够了 |
| 追踪非 Python 命令 | Wrapper 目标是 Python 实验脚本；扩展到任意 shell 命令可以但不在本次范围内 |
| 用 LLM 解析 stdout | 慢、贵、不可复现。正则虽然土但快、零成本、确定性强 |
| 用 SQLite 代替 JSONL | JSONL 直接在 pandas 里读、无额外依赖、追加写入天然崩溃安全 |

---

## 本次改动清单

| 操作 | 文件 | 行数 |
|------|------|------|
| 修改 | `.gitignore` | +3（加 `run_history/`） |
| 新建 | `run.py` | ~155 |
| 新建 | `utils/run_tracker.py` | ~170 |
| **合计** | | **+328 行，0 个现有脚本被修改** |
