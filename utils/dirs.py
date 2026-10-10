"""项目必需目录的统一创建工具。

原 main.py 中的 create_necessary_directories() 被抽到此处，
各脚本在 __main__ 开头调用 ensure_dirs()。
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NECESSARY_DIRS = [
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


def ensure_dirs():
    """创建项目运行所需的全部目录（已存在则跳过）。"""
    for d in NECESSARY_DIRS:
        os.makedirs(os.path.join(PROJECT_ROOT, d), exist_ok=True)
