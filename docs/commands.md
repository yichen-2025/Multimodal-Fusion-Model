# 命令模板

> 本项目所有步骤均通过**直接脚本 CLI** 运行，不再使用 `main.py --run` 统一入口。
> 所有命令均需在项目根目录下执行（`python scripts/xxx.py ...`）。
> 参数名已对照各脚本 `--help` 实测校验，可直接复制使用。

---

## 通用约定

- `<id>` 表示前序步骤生成的真实数字 ID（如 `saved_models/model_9` 中的 `9`）。
- `--dataset_id` / `--split_id` 建议在同一条实验链路中保持固定。
- 数据集 ID 以 `split_data/` 下实际存在的目录为准（常见：`dataset_0` 12 类、`dataset_3` 15 类）。
- LLM 模型路径默认 `--model_path /root/autodl-tmp/Multimodal-Fusion-Model/models/qwen2.5-1.5b`（按需替换）。

---

## P0 数据预处理

### 1. 数据清洗

```bash
python scripts/data_cleaning.py
```

- 参数：`--use_merged`（默认开启，合并为 12 类）/ `--no_merged`（使用原始 15 类）。
- 产物：`processed_dataset/processed_dataset.csv`、`processed_dataset/label_mapping.json`。

### 2. 提取子集

```bash
python scripts/extract_subset.py --total_samples 5000 --dataset_id 0 --random_state 42
```

- 参数：
  - `--total_samples`：目标样本数（与 `--ratio` 二选一）。
  - `--ratio`：保留比例（如 `0.1`）。
  - `--mode`：`stratified`（默认）/ `random`。
  - `--dataset_id`：输出数据集 ID。
  - `--per_class`：自定义每类样本数（JSON 字符串）。
- 产物：`processed_dataset/dataset_<id>.csv`。

### 3. 模态分离

```bash
python scripts/split_modality.py --dataset_id 0 --test_size 0.2 --val_size 0.1
```

- 参数：`--dataset_id`、`--split_id`（默认自动递增）、`--test_size`、`--val_size`、`--random_state`、`--input_csv`。
- 产物：`split_data/dataset_<id>/split_<id>/{train,val,test}.npz`。

---

## P0 模型训练

### 4. 训练融合模型

```bash
python scripts/train.py --variant A3 --dataset_id 0 --split_id 0 --epochs 5 --lr 5e-4
```

- 关键参数：
  - `--variant`：`A0` / `A1` / `A2` / `A3` / `A0*` / `A0_frozen` 等。
  - `--dataset_id`、`--split_id`、`--model_id`（默认自动递增）。
  - `--epochs`、`--lr`、`--batch_size`、`--gradient_accumulation`。
  - `--openset`：加载开集划分（`split_openset_*`）。
  - `--model_path`：LLM 路径（含 LLM 的变体需要）。
- 产物：`saved_models/model_<id>/`。
- 注意：变体含 LLM（如 `A0*`）需 GPU；`A3`（无 LLM，纯 MLP）最快。

### 5. 测试模型

```bash
python scripts/test_model.py --model_id <id> --dataset_id 0 --split_id 0
```

- 参数：`--model_id`（或 `--saved_model_path`）、`--dataset_id`、`--split_id`、`--variant`（写入报告区分）、`--openset`、`--no_save_report`。
- 产物：`test_reports/report_<id>.json`。

### 6. 消融实验

```bash
python scripts/run_ablation.py --variants A3 A0* A0_frozen --dataset_id 0 --split_id 0
```

- 参数：`--variants`（空格分隔多个变体）、`--dataset_id`、`--split_id`、`--repeat`、`--seed`、`--plot`。
- 产物：`ablation_results/*.csv`。

---

## P1 开集 / 少样本

### 7. 构造开集划分

```bash
python scripts/make_openset_split.py --dataset_id 0 --source_split_id 0 --simulated_unknown_ratio 0.3
```

- 参数：
  - `--dataset_id`、`--source_split_id`、`--output_split_id`（默认自动递增）。
  - `--simulated_unknown_ratio`：模拟未知类比例（**注意参数名，不是 `--unknown_ratio`**）。
  - `--hold_out_classes`：完全留出的类别 ID 列表（如 `10 11`）。
  - `--no_default_holdout`：不使用默认留出类。
- 产物：`split_data/dataset_<id>/split_openset_<id>/`。

### 8. 构造少样本划分

```bash
python scripts/make_fewshot.py --dataset_id 0 --source_split_id 0 --k_values 5
```

- 参数：
  - `--dataset_id`、`--source_split_id`、`--output_split_id`。
  - `--k_values`：少样本 k 值，**逗号分隔字符串**（如 `5`、`5,10,20`），不是 `--k_per_class`。
  - `--no_remaining`：不生成剩余样本作为验证/测试。
- 产物：`split_data/dataset_<id>/split_fewshot_<k>_<id>/`。

### 9. 训练 OOD 检测头

```bash
python scripts/train_ood_head.py --model_id <backbone_id> --dataset_id 0 --split_id 0 --variant A3 --fewshot_k 5
```

- 参数：`--model_id`（backbone 模型 ID）、`--ood_id`（默认自动递增）、`--dataset_id`、`--split_id`、`--variant`、`--fewshot_k`、`--distance_type`、`--lr`、`--epochs`、`--batch_size`。
- 产物：`saved_ood_heads/ood_<id>/`。

### 10. OOD 路由评估

```bash
python scripts/run_ood_routing.py --backbone_model_id <id> --ood_id <id> --dataset_id 0 --split_id 0 --fewshot_k 5
```

- 参数（`--backbone_model_id` 与 `--ood_id` 为必填）：
  - `--backbone_model_id`、`--ood_id`、`--llm_model_id`（可选，不填则跳过 LLM）。
  - `--dataset_id`、`--split_id`、`--fewshot_k`（指定则加载 `split_fewshot_<k>`，否则用 `split_openset`）。
  - `--backbone_variant`、`--llm_variant`、`--no_compare_baseline`、`--quiet`。
- 产物：`ood_reports/` 下评测报告。

### 11. 一键开集实验

```bash
python scripts/run_openworld_experiment.py --dataset_id 0 --split_id 0 --k_values 5,10,20,None --backbone_variant A3 --llm_variant A0*
```

- 参数：
  - `--dataset_id`、`--split_id`。
  - `--k_values`：逗号分隔 k 值列表（`None` 表示全量），如 `5,10,20,None`。
  - `--backbone_variant`、`--llm_variant`、`--model_path`。
  - `--backbone_model_id` / `--llm_model_id`：指定已有模型则跳过对应训练。
  - `--skip_backbone_train`、`--skip_ood_train`、`--skip_llm`、`--no_report`。
- 注意：**该脚本没有 `--fewshot_k` 参数**，请使用 `--k_values`。
- 产物：训练 backbone + OOD 头 + `test_reports/openworld_*.csv/.md/.json`。

### 12. 生成汇总报告

```bash
python scripts/report_generator.py
```

- 参数：`--output_dir`（输出目录，默认 `test_reports/`）。
- 产物：`test_reports/openworld_fewshot.csv` + summary `.md` + `.json`。

---

## 常见错误参数名对照

| 错误写法 | 正确写法 | 所在脚本 |
|---|---|---|
| `--unknown_ratio` | `--simulated_unknown_ratio` | `make_openset_split.py` |
| `--k_per_class` | `--k_values` | `make_fewshot.py` |
| `--fewshot_k`（用于一键实验） | `--k_values` | `run_openworld_experiment.py` |
