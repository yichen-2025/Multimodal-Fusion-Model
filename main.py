"""
多模态融合模型 — 统一入口

用法:
  python main.py --list                     列出所有可用配置
  python main.py --run <配置名>              执行指定配置

示例:
  python main.py --run clean
  python main.py --run subset
  python main.py --run train
  python main.py --run ood_train
  python main.py --run experiment
"""

import os
import sys
import argparse
import importlib
import inspect

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
sys.path.insert(0, PROJECT_ROOT)

from configs.run_configs import RUN_CONFIG


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


STEP_CONFIGS = [
    {
        "id": "clean",
        "stage": "P0 数据预处理",
        "desc": "数据清洗：读取原始CSV，处理异常值和缺失值",
        "import": ("scripts.data_cleaning", "main"),
    },
    {
        "id": "subset",
        "stage": "P0 数据预处理",
        "desc": "提取子集：从清洗后的数据中抽取指定数量的样本",
        "import": ("scripts.extract_subset", "extract_subset"),
    },
    {
        "id": "split",
        "stage": "P0 数据预处理",
        "desc": "模态分离：将数据拆分为train/val/test并分离数值特征与文本特征",
        "import": ("scripts.split_modality", "split_modality"),
    },
    {
        "id": "train",
        "stage": "P0 模型训练",
        "desc": "训练多模态融合模型（支持A0/A1/A2/A3变体）",
        "import": ("scripts.train", "train_model"),
    },
    {
        "id": "test",
        "stage": "P0 模型训练",
        "desc": "测试已训练的模型",
        "import": ("scripts.test_model", "test_model"),
    },
    {
        "id": "ablation",
        "stage": "P0 模型训练",
        "desc": "运行消融实验（对比A0/A1/A2/A3）",
        "import": ("scripts.run_ablation", "run_ablation"),
    },
    {
        "id": "openset",
        "stage": "P1 开集检测",
        "desc": "构造开集数据集：将部分DDoS样本标记为未知类",
        "import": ("scripts.make_openset_split", "make_openset_split"),
    },
    {
        "id": "fewshot",
        "stage": "P1 少样本",
        "desc": "构造少样本数据集：每已知类抽取k条训练样本",
        "import": ("scripts.make_fewshot", "make_fewshot_split"),
    },
    {
        "id": "ood-train",
        "stage": "P1 少样本",
        "desc": "训练OOD检测头：基于A3融合特征学习类原型",
        "import": ("scripts.train_ood_head", "train_ood_head"),
    },
    {
        "id": "ood-eval",
        "stage": "P1 少样本",
        "desc": "OOD路由评估：A3处理已知样本，LLM处理未知样本",
        "import": ("scripts.run_ood_routing", "run_ood_routing"),
    },
    {
        "id": "experiment",
        "stage": "P1 少样本",
        "desc": "一键完整实验：训练OOD头 + 对所有k值评估 + 生成报告",
        "import": ("scripts.run_openworld_experiment", "run_openworld_experiment"),
    },
    {
        "id": "report",
        "stage": "P1 少样本",
        "desc": "生成汇总报告（基于已有日志和评估结果）",
        "import": ("scripts.report_generator", "generate_experiment_summary"),
    },
]


def get_step_config(step_id):
    for cfg in STEP_CONFIGS:
        if cfg["id"] == step_id:
            return cfg
    return None


def list_runs():
    default_runs = {k: v for k, v in RUN_CONFIG.items() if k.endswith("_default")}
    custom_runs = {k: v for k, v in RUN_CONFIG.items() if not k.endswith("_default")}

    for section_title, runs in [
        ("第一部分：默认配置（标准参数，供参考）", default_runs),
        ("第二部分：自定义配置（修改这些即可）", custom_runs),
    ]:
        print(f"\n{'═' * 60}")
        print(f"  {section_title}")
        print(f"{'═' * 60}")
        current_stage = None
        for run_name, run_cfg in runs.items():
            step_id = run_cfg["step"]
            cfg = get_step_config(step_id)
            stage = cfg["stage"] if cfg else "未知阶段"
            if stage != current_stage:
                current_stage = stage
                print(f"\n  [{current_stage}]")
            desc = cfg["desc"] if cfg else "(未知步骤)"
            params_str = ", ".join(f"{k}={v}" for k, v in run_cfg.get("params", {}).items())
            if len(params_str) > 80:
                params_str = params_str[:77] + "..."
            print(f"  {run_name:<24} {desc}")
            if params_str:
                print(f"  {'':<24} 参数: {params_str}")
        print()


def build_call_string(func_name, kwargs):
    parts = []
    for k, v in kwargs.items():
        if isinstance(v, str):
            parts.append(f"{k}='{v}'")
        else:
            parts.append(f"{k}={v}")
    return f"{func_name}({', '.join(parts)})"


def _get_func_params(module_name, func_name):
    """获取函数的参数列表（名称、默认值、类型），返回 (参数字典, 函数对象)。"""
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    sig = inspect.signature(func)
    params = {}
    for name, param in sig.parameters.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            # **kwargs 存在，接受任意参数
            return None, func
        default = param.default if param.default is not inspect.Parameter.empty else None
        params[name] = {"default": default}
    return params, func


def _infer_arg_type(param_name, func_default, cfg_params):
    """推断命令行参数的类型，优先从函数默认值，其次从配置 params。"""
    if isinstance(func_default, bool):
        return "bool"
    if isinstance(func_default, int):
        return "int"
    if isinstance(func_default, float):
        return "float"
    # 函数默认值是 None 或 str 等，尝试从 run 配置中取类型线索
    if param_name in cfg_params:
        cfg_val = cfg_params[param_name]
        if isinstance(cfg_val, bool):
            return "bool"
        if isinstance(cfg_val, int):
            return "int"
        if isinstance(cfg_val, float):
            return "float"
    return "str"


def _parse_override_args(remaining_args, func_params, cfg_params=None):
    """解析命令行中额外的 --key value 参数，校验后返回覆盖字典。"""
    if cfg_params is None:
        cfg_params = {}

    extra_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)

    for name, info in func_params.items():
        default = info["default"]
        arg_type = _infer_arg_type(name, default, cfg_params)

        if arg_type == "bool":
            extra_parser.add_argument(f"--{name}", action="store_true", default=None, dest=name)
            extra_parser.add_argument(f"--no-{name}", action="store_false", dest=name, default=None)
        elif arg_type == "int":
            extra_parser.add_argument(f"--{name}", type=int, default=None)
        elif arg_type == "float":
            extra_parser.add_argument(f"--{name}", type=float, default=None)
        else:
            extra_parser.add_argument(f"--{name}", type=str, default=None)

    parsed, unknown = extra_parser.parse_known_args(remaining_args)

    unknown_names = [a[2:] for a in unknown if a.startswith("--")]
    if unknown_names:
        return None, unknown_names

    # 只保留命令行显式传入的值（非 None）
    overrides = {k: v for k, v in vars(parsed).items() if v is not None}
    return overrides, []


def main():
    parser = argparse.ArgumentParser(
        description="多模态融合模型 — 统一入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
可用配置:
  {', '.join(RUN_CONFIG.keys())}

示例:
  python main.py --run clean
  python main.py --run train
  python main.py --run train --dataset_id 10 --learning_rate 0.001
  python main.py --run subset
  python main.py --run experiment
        """,
    )

    parser.add_argument("--run", type=str, default=None,
                        help=f"执行指定配置。可选值: {', '.join(RUN_CONFIG.keys())}")
    parser.add_argument("--list", action="store_true",
                        help="列出所有可用配置")

    args, remaining = parser.parse_known_args()

    if args.list:
        list_runs()
        return

    if args.run is None:
        parser.print_help()
        print("\n提示: 使用 --list 查看所有可用配置，使用 --run <配置名> 执行指定配置。")
        return

    run_cfg = RUN_CONFIG.get(args.run)
    if run_cfg is None:
        print(f"错误: 未知配置 '{args.run}'")
        print(f"可用配置: {', '.join(RUN_CONFIG.keys())}")
        return

    step_id = run_cfg["step"]
    params = run_cfg.get("params", {})

    cfg = get_step_config(step_id)
    if cfg is None:
        print(f"错误: 未知步骤 '{step_id}'")
        return

    create_necessary_directories()

    print("=" * 60)
    print(f"执行配置: {args.run}")
    print(f"步骤: {cfg['id']} [{cfg['stage']}]")
    print(f"描述: {cfg['desc']}")
    print("=" * 60)

    func_name = cfg["import"][1]
    call_str = build_call_string(func_name, params)
    print(f"\n# 等价调用:")
    print(f"# {call_str}")
    print()

    module_name, func_name = cfg["import"]

    # 获取函数签名，支持命令行覆盖参数
    func_params_result, func = _get_func_params(module_name, func_name)

    if func_params_result is not None and remaining:
        overrides, unknown = _parse_override_args(remaining, func_params_result, cfg_params=params)
        if unknown:
            print(f"错误: 函数 '{func_name}' 没有定义以下参数: {', '.join(unknown)}")
            print(f"该函数支持的参数: {', '.join(func_params_result.keys())}")
            return
        if overrides:
            print(f"命令行覆盖参数: {overrides}")
            params = {**params, **overrides}

    result = func(**params)

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
    print(f"配置 [{args.run}] 执行完成。")
    print("=" * 60)


if __name__ == "__main__":
    main()