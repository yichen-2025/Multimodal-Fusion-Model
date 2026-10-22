# 生成物台账（artifacts/）

> 本目录是仓库内所有实验生成物的唯一判据。审查代码、引用结论、继续实验之前，先读 `ledger.csv`。

## 为什么需要它

`run_history/`、`logs/`、`test_reports/report_*.json` 只记录 id（model_id / ood_id /
backbone_model_id），不记录产物状态；而 `saved_models/` 里曾经同时躺着 10 类、11 类、2 类
三种标签空间的模型。直接读目录会把历史实验当成当前基线。

## 状态定义

| status | 含义 | 位置 |
|---|---|---|
| active | 当前有效，可作为结论依据 | 原位 |
| deprecated | 历史实验 / 旧口径 / 重复运行，仅作证据 | 模型与头在 `_archive/`，报告原地保留 |
| invalid | 引用悬空或无法解析 | 模型与头在 `_archive/` |

## 当前 active 产物（2026-10-22 生成）

- 模型：`model_102`（11 类闭集基线，report_15）；`model_1000`–`model_1011`（融合/LLM 对比，report_1–12）
- 报告：`report_1`–`report_13`、`report_15`
- 划分：`split_data/dataset_0/split_0`（闭集）、`split_openset_0`（开集）、`split_fewshot_*`
- OOD 头：**无 active**。144 份 `ood_reports` 全部基于缺少 `unknown_origin` 的旧划分，
  unknown 指标是"真实 OOD + 模拟 unknown"混合口径，不能当作真实 OOD 能力。

## 已隔离产物

244 个产物已移入 `_archive/`，明细见 `_archive/MANIFEST.jsonl`（原路径、原因、大小、内容摘要）。
**移动不是删除，可逆**：`mv _archive/<相对路径> <原相对路径>` 即可还原。

`report_0`（dataset_100 子集冒烟实验，accuracy 0.027）与 `report_14`（report_13 的重复运行）
未被移动，因为在 `reports_index.csv` 中有对应行；它们只在台账里标记为 deprecated。

## 日志没有被修改

`reports_index.csv`、`run_history/`、`logs/`、`ood_reports/` 全部保持原样。要解释"某个 id 为什么
找不到文件"，查 `ledger.csv` 的 `archive_path` 列或 `_archive/MANIFEST.jsonl`。

## 命令

```bash
python scripts/audit_artifacts.py            # dry-run：打印状态统计与待隔离清单
python scripts/audit_artifacts.py --check    # 只读一致性校验，失败退出码 1
python scripts/audit_artifacts.py --apply    # 重新生成台账并隔离新产生的废弃产物
```

## 规则

- 引用旧结果时必须标注"deprecated，混合口径"。
- `_archive/` 下的内容不是当前结论依据。
- 新实验产物若不应成为 active，跑 `--apply` 前先确认台账判定是否符合预期。
