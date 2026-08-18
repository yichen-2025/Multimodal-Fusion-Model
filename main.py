"""
多模态融合模型 — 统一入口

用法:
  python main.py --list                     列出所有步骤
  python main.py --help                     查看详细帮助
  python main.py --step <步骤名> [参数...]   执行指定步骤

示例:
  python main.py --step clean
  python main.py --step train --dataset_id 1 --variant A3
  python main.py --step openset --dataset_id 1 --unknown_ratio 0.3
  python main.py --step ood-train --model_id 9 --fewshot_k 5
  python main.py --step experiment --backbone_model_id 9 --k_values "5,10,20,None"
"""

import os
import sys
import argparse
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
sys.path.insert(0, PROJECT_ROOT)


def create_necessary_directories():
    directories = [
        "processed_dataset",
        "split_data",
        "saved_models",
        "saved_ood_heads",
        "logs/subset",
        "logs/split",
        "logs/training",
        "logs/openset_split",
        "logs/ood_training",
        "logs/ood_routing",
        "logs/fewshot_split",
        "logs/experiment",
        "ood_reports",
        "test_reports",
        "ablation_results",
    ]
    created_dirs = []
    for dir_path in directories:
        full_path = os.path.join(PROJECT_ROOT, dir_path)
        if not os.path.exists(full_path):
            os.makedirs(full_path, exist_ok=True)
            created_dirs.append(dir_path)
    if created_dirs:
        print("创建了以下目录：")
        for d in created_dirs:
            print(f"  - {d}")


def _parse_k_values(k_values_str):
    if k_values_str is None:
        return None
    result = []
    for k_str in k_values_str.split(","):
        k_str = k_str.strip()
        if k_str.lower() == "none":
            result.append(None)
        else:
            result.append(int(k_str))
    return result


def _parse_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes", "y")
    return bool(val)


STEP_CONFIGS = [
    {
        "id": "clean",
        "stage": "P0 数据预处理",
        "desc": "数据清洗：读取原始CSV，处理异常值和缺失值",
        "import": ("scripts.data_cleaning", "main"),
        "defaults": {},
        "param_map": {},
        "example": "python main.py --step clean",
    },
    {
        "id": "subset",
        "stage": "P0 数据预处理",
        "desc": "提取子集：从清洗后的数据中抽取指定数量的样本",
        "import": ("scripts.extract_subset", "extract_subset"),
        "defaults": {"num_samples": 5000, "random_state": 42},
        "param_map": {
            "dataset_id": "dataset_id",
            "num_samples": "num_samples",
            "random_state": "random_state",
        },
        "example": "python main.py --step subset --dataset_id 1 --num_samples 5000",
    },
    {
        "id": "split",
        "stage": "P0 数据预处理",
        "desc": "模态分离：将数据拆分为train/val/test并分离数值特征与文本特征",
        "import": ("scripts.split_modality", "split_modality"),
        "defaults": {"test_size": 0.2, "val_size": 0.1, "random_state": 42},
        "param_map": {
            "dataset_id": "dataset_id",
            "split_id": "split_id",
            "test_size": "test_size",
            "val_size": "val_size",
            "random_state": "random_state",
        },
        "example": "python main.py --step split --dataset_id 1",
    },
    {
        "id": "train",
        "stage": "P0 模型训练",
        "desc": "训练多模态融合模型（支持A0/A1/A2/A3变体）",
        "import": ("scripts.train", "train_model"),
        "defaults": {
            "variant": "A3",
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1e-4,
            "num_train_epochs": 3,
            "seed": 42,
        },
        "param_map": {
            "model_id": "model_id",
            "dataset_id": "dataset_id",
            "split_id": "split_id",
            "variant": "variant",
            "batch_size": "per_device_train_batch_size",
            "grad_accum": "gradient_accumulation_steps",
            "lr": "learning_rate",
            "epochs": "num_train_epochs",
            "seed": "seed",
        },
        "example": "python main.py --step train --dataset_id 1 --variant A3",
    },
    {
        "id": "test",
        "stage": "P0 模型训练",
        "desc": "测试已训练的模型",
        "import": ("scripts.test_model", "test_model"),
        "defaults": {"verbose": True, "save_report": True},
        "param_map": {
            "dataset_id": "dataset_id",
            "split_id": "split_id",
            "model_id": "model_id",
            "verbose": "verbose",
        },
        "example": "python main.py --step test --dataset_id 1 --model_id 3",
    },
    {
        "id": "ablation",
        "stage": "P0 模型训练",
        "desc": "运行消融实验（对比A0/A1/A2/A3）",
        "import": ("scripts.run_ablation", "run_ablation"),
        "defaults": {"variants": ["A0", "A1", "A2", "A3"], "repeat": 1, "seed": 42},
        "param_map": {
            "dataset_id": "dataset_id",
            "split_id": "split_id",
            "seed": "seed",
        },
        "example": "python main.py --step ablation --dataset_id 1",
    },
    {
        "id": "openset",
        "stage": "P1 开集检测",
        "desc": "构造开集数据集：将部分DDoS样本标记为未知类",
        "import": ("scripts.make_openset_split", "make_openset_split"),
        "defaults": {"unknown_ratio": 0.3, "random_state": 42},
        "param_map": {
            "dataset_id": "dataset_id",
            "split_id": "source_split_id",
            "output_split_id": "output_split_id",
            "unknown_ratio": "unknown_ratio",
            "random_state": "random_state",
        },
        "example": "python main.py --step openset --dataset_id 1 --unknown_ratio 0.3",
    },
    {
        "id": "fewshot",
        "stage": "P1 少样本",
        "desc": "构造少样本数据集：每已知类抽取k条训练样本",
        "import": ("scripts.make_fewshot", "make_fewshot_split"),
        "defaults": {"k_per_class": 5, "random_state": 42},
        "param_map": {
            "dataset_id": "dataset_id",
            "split_id": "source_split_id",
            "k_per_class": "k_per_class",
            "random_state": "random_state",
        },
        "example": "python main.py --step fewshot --dataset_id 1 --k_per_class 5",
    },
    {
        "id": "ood-train",
        "stage": "P1 少样本",
        "desc": "训练OOD检测头：基于A3融合特征学习类原型",
        "import": ("scripts.train_ood_head", "train_ood_head"),
        "defaults": {
            "variant": "A3",
            "distance_type": "cosine",
            "temperature": 1.0,
            "learning_rate": 1e-3,
            "num_epochs": 10,
            "batch_size": 64,
            "seed": 42,
        },
        "param_map": {
            "model_id": "model_id",
            "ood_id": "ood_id",
            "dataset_id": "dataset_id",
            "split_id": "split_id",
            "variant": "variant",
            "fewshot_k": "fewshot_k",
            "distance_type": "distance_type",
            "temperature": "temperature",
            "lr": "learning_rate",
            "epochs": "num_epochs",
            "batch_size": "batch_size",
            "seed": "seed",
        },
        "example": "python main.py --step ood-train --model_id 9 --dataset_id 1 --fewshot_k 5",
    },
    {
        "id": "ood-eval",
        "stage": "P1 少样本",
        "desc": "OOD路由评估：A3处理已知样本，LLM处理未知样本",
        "import": ("scripts.run_ood_routing", "run_ood_routing"),
        "defaults": {"backbone_variant": "A3", "llm_variant": "A0", "batch_size": 32},
        "param_map": {
            "model_id": "backbone_model_id",
            "ood_id": "ood_id",
            "llm_model_id": "llm_model_id",
            "dataset_id": "dataset_id",
            "split_id": "split_id",
            "fewshot_k": "fewshot_k",
            "batch_size": "batch_size",
        },
        "example": "python main.py --step ood-eval --model_id 9 --ood_id 1 --fewshot_k 5",
    },
    {
        "id": "experiment",
        "stage": "P1 少样本",
        "desc": "一键完整实验：训练OOD头 + 对所有k值评估 + 生成报告",
        "import": ("scripts.run_openworld_experiment", "run_openworld_experiment"),
        "defaults": {"k_values": [None], "seed": 42},
        "param_map": {
            "dataset_id": "dataset_id",
            "split_id": "split_id",
            "k_values": "k_values",
            "backbone_model_id": "backbone_model_id",
            "llm_model_id": "llm_model_id",
            "seed": "seed",
        },
        "example": 'python main.py --step experiment --backbone_model_id 9 --k_values "5,10,20,None"',
    },
    {
        "id": "report",
        "stage": "P1 少样本",
        "desc": "生成汇总报告（基于已有日志和评估结果）",
        "import": ("scripts.report_generator", "generate_experiment_summary"),
        "defaults": {},
        "param_map": {},
        "example": "python main.py --step report",
    },
]


def get_step_config(step_id):
    for cfg in STEP_CONFIGS:
        if cfg["id"] == step_id:
            return cfg
    return None


def list_steps():
    current_stage = None
    for cfg in STEP_CONFIGS:
        if cfg["stage"] != current_stage:
            current_stage = cfg["stage"]
            print(f"\n{'─' * 60}")
            print(f"  [{current_stage}]")
            print(f"{'─' * 60}")
        print(f"  {cfg['id']:<16} {cfg['desc']}")
        print(f"  {'':<16} 示例: {cfg['example']}")
        print()


def print_step_help(step_id):
    cfg = get_step_config(step_id)
    if cfg is None:
        print(f"未知步骤: {step_id}")
        return
    print(f"步骤: {cfg['id']}")
    print(f"阶段: {cfg['stage']}")
    print(f"描述: {cfg['desc']}")
    print(f"等价函数: {cfg['import'][1]}()")
    print(f"默认参数: {json.dumps(cfg['defaults'], ensure_ascii=False)}")
    print(f"示例: {cfg['example']}")
    print(f"\n可用覆盖参数:")
    print(f"  --dataset_id       数据集ID")
    print(f"  --split_id         划分ID")
    print(f"  --model_id         模型ID")
    print(f"  --variant          变体 (A0/A1/A2/A3)")
    print(f"  --fewshot_k        少样本k值")
    print(f"  --k_values         k值列表 (如 \"5,10,20,None\")")
    print(f"  --unknown_ratio    未知类比例")
    print(f"  --lr               学习率")
    print(f"  --epochs           训练轮数")
    print(f"  --batch_size       Batch大小")


def build_call_string(func_name, kwargs):
    parts = []
    for k, v in kwargs.items():
        if isinstance(v, str):
            parts.append(f"{k}='{v}'")
        else:
            parts.append(f"{k}={v}")
    return f"{func_name}({', '.join(parts)})"


def main():
    parser = argparse.ArgumentParser(
        description="多模态融合模型 — 统一入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
步骤示例:
  python main.py --step clean
  python main.py --step train --dataset_id 1 --variant A3
  python main.py --step openset --dataset_id 1 --unknown_ratio 0.3
  python main.py --step fewshot --dataset_id 1 --k_per_class 5
  python main.py --step ood-train --model_id 9 --dataset_id 1 --fewshot_k 5
  python main.py --step ood-eval --model_id 9 --ood_id 1 --fewshot_k 5
  python main.py --step experiment --backbone_model_id 9 --k_values "5,10,20,None"
  python main.py --step report
        """,
    )

    parser.add_argument("--step", type=str, default=None,
                        help=f"执行指定步骤。可选值: {', '.join(c['id'] for c in STEP_CONFIGS)}")
    parser.add_argument("--list", action="store_true",
                        help="列出所有可用步骤")

    # 通用参数
    parser.add_argument("--dataset_id", type=int, default=None, help="数据集ID")
    parser.add_argument("--split_id", type=int, default=None, help="划分ID")
    parser.add_argument("--model_id", type=int, default=None, help="模型ID")
    parser.add_argument("--ood_id", type=int, default=None, help="OOD检测头ID")
    parser.add_argument("--llm_model_id", type=int, default=None, help="LLM模型ID")
    parser.add_argument("--backbone_model_id", type=int, default=None, help="Backbone模型ID")
    parser.add_argument("--variant", type=str, default=None, help="变体 (A0/A1/A2/A3)")
    parser.add_argument("--backbone_variant", type=str, default=None, help="Backbone变体")
    parser.add_argument("--llm_variant", type=str, default=None, help="LLM变体")
    parser.add_argument("--fewshot_k", type=int, default=None, help="少样本k值")
    parser.add_argument("--k_values", type=str, default=None, help="k值列表 (如 '5,10,20,None')")
    parser.add_argument("--k_per_class", type=int, default=None, help="每类少样本数量")
    parser.add_argument("--unknown_ratio", type=float, default=None, help="未知类比例")
    parser.add_argument("--distance_type", type=str, default=None, choices=["euclidean", "cosine", "mahalanobis"], help="OOD距离度量")
    parser.add_argument("--temperature", type=float, default=None, help="OOD温度系数")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch大小")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--num_samples", type=int, default=None, help="子集样本数量")
    parser.add_argument("--test_size", type=float, default=None, help="测试集比例")
    parser.add_argument("--val_size", type=float, default=None, help="验证集比例")
    parser.add_argument("--grad_accum", type=int, default=None, help="梯度累积步数")
    parser.add_argument("--verbose", type=str, default=None, help="详细输出 (true/false)")
    parser.add_argument("--output_split_id", type=int, default=None, help="输出划分ID")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")

    args = parser.parse_args()

    if args.list:
        list_steps()
        return

    if args.step is None:
        parser.print_help()
        print("\n提示: 使用 --list 查看所有可用步骤，使用 --step <名称> 执行指定步骤。")
        return

    cfg = get_step_config(args.step)
    if cfg is None:
        print(f"错误: 未知步骤 '{args.step}'")
        print(f"可用步骤: {', '.join(c['id'] for c in STEP_CONFIGS)}")
        return

    print("=" * 60)
    print(f"执行步骤: {cfg['id']}  [{cfg['stage']}]")
    print(f"描述: {cfg['desc']}")
    print("=" * 60)

    create_necessary_directories()

    # 构建参数：以 defaults 为基础，用 args 中的值覆盖
    kwargs = dict(cfg["defaults"])

    for arg_name, kwarg_name in cfg["param_map"].items():
        arg_val = getattr(args, arg_name, None)
        if arg_val is not None:
            kwargs[kwarg_name] = arg_val

    # 特殊处理 k_values
    if "k_values" in cfg["param_map"]:
        if args.k_values is not None:
            kwargs["k_values"] = _parse_k_values(args.k_values)
        elif cfg["id"] == "experiment" and "k_values" not in kwargs:
            kwargs["k_values"] = [None]

    # 特殊处理 verbose
    if "verbose" in cfg["param_map"] and args.verbose is not None:
        kwargs["verbose"] = _parse_bool(args.verbose)

    # 特殊处理 num_samples 为 None 时的默认值
    if cfg["id"] == "subset" and args.num_samples is not None:
        kwargs["num_samples"] = args.num_samples

    # 特殊处理 dataset_id: 默认使用配置中的默认值，若无则用 0
    if "dataset_id" in cfg["param_map"] and "dataset_id" not in kwargs:
        kwargs["dataset_id"] = 0

    # 特殊处理 split_id: 默认用 0
    if "split_id" in cfg["param_map"] and "split_id" not in kwargs:
        kwargs["split_id"] = 0

    # 打印等价函数调用
    func_name = cfg["import"][1]
    call_str = build_call_string(func_name, kwargs)
    print(f"\n# 等价调用:")
    print(f"# {call_str}")
    print()

    # 动态导入并执行
    module_name, func_name = cfg["import"]
    import importlib
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)

    # 对于有返回值的函数，打印结果摘要
    result = func(**kwargs)

    # 打印结果摘要
    if result is not None:
        print("\n" + "-" * 40)
        print("执行结果摘要:")
        if isinstance(result, tuple):
            for i, item in enumerate(result):
                if isinstance(item, (int, float, str, bool)):
                    print(f"  [{i}]: {item}")
                elif isinstance(item, dict):
                    for k, v in item.items():
                        if isinstance(v, (int, float, str, bool)):
                            print(f"  [{i}].{k}: {v}")
                else:
                    print(f"  [{i}]: {type(item).__name__}")
        elif isinstance(result, dict):
            for k, v in result.items():
                if isinstance(v, (int, float, str, bool)):
                    print(f"  {k}: {v}")
                elif isinstance(v, dict):
                    print(f"  {k}: (dict with keys: {list(v.keys())[:5]})")
        elif isinstance(result, (int, float, str, bool)):
            print(f"  值: {result}")
        else:
            print(f"  类型: {type(result).__name__}")

    print("\n" + "=" * 60)
    print(f"步骤 [{cfg['id']}] 执行完成。")
    print("=" * 60)


if __name__ == "__main__":
    main()