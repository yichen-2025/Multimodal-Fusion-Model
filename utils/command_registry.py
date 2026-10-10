"""命令注册表。

为每个命令声明差异化属性（生成/改动的文件、附加说明），
供 CommandRecord 填充。新增命令只需在此追加一条，无需改 run.py。
"""

COMMAND_REGISTRY = {
    "data_cleaning": {
        "generated_files": [
            "processed_dataset/processed_dataset.csv",
            "processed_dataset/label_mapping.json",
        ],
        "modified_files": [],
        "notes": "",
    },
    "extract_subset": {
        "generated_files": [
            "processed_dataset/dataset_<id>.csv",
            "processed_dataset/subset_<id>_scaled_features.npy",
        ],
        "modified_files": [],
        "notes": "",
    },
    "split_modality": {
        "generated_files": [
            "split_data/dataset_<id>/split_<id>/{train,val,test}.npz",
        ],
        "modified_files": [],
        "notes": "首次运行加载 BERT 较慢",
    },
    "train": {
        "generated_files": [
            "saved_models/model_<id>/ 目录（含模型权重 .pt 与配置 config.txt）",
        ],
        "modified_files": [
            "logs/training/index.csv（追加一行训练摘要）",
        ],
        "notes": "若修改了 src/model_architectures/fusion_projection.py，须加 --force_retrain（旧权重维度不兼容）",
    },
    "test_model": {
        "generated_files": [
            "test_reports/report_<id>.json（含 Acc/P/R/F1 与混淆矩阵）",
        ],
        "modified_files": [
            "logs/testing/index.csv（追加一行测试摘要）",
        ],
        "notes": "",
    },
    "run_ablation": {
        "generated_files": [
            "ablation_results/*.csv + 对比图",
        ],
        "modified_files": [],
        "notes": "耗时长（多变体 x 多 seed）",
    },
    "make_openset_split": {
        "generated_files": [
            "split_data/dataset_<id>/split_openset_<id>/",
        ],
        "modified_files": [],
        "notes": "test 集含 unknown 类",
    },
    "make_fewshot": {
        "generated_files": [
            "split_data/dataset_<id>/split_fewshot_<k>_<id>/",
        ],
        "modified_files": [],
        "notes": "依赖已有 openset 划分",
    },
    "train_ood_head": {
        "generated_files": [
            "saved_ood_heads/ood_<id>/（原型 + 阈值）",
        ],
        "modified_files": [],
        "notes": "依赖已训 backbone",
    },
    "run_ood_routing": {
        "generated_files": [
            "ood_reports/ 下评测报告",
        ],
        "modified_files": [
            "logs/ood_routing/index.csv",
        ],
        "notes": "",
    },
    "run_openworld_experiment": {
        "generated_files": [
            "test_reports/openworld_*.csv/.md/.json",
            "训练 backbone + OOD 头（saved_models/ 与 saved_ood_heads/）",
        ],
        "modified_files": [],
        "notes": "默认 --llm_variant A0，须显式指定 A0*",
    },
    "report_generator": {
        "generated_files": [
            "test_reports/openworld_fewshot.csv + summary .md",
        ],
        "modified_files": [],
        "notes": "",
    },
    "run_multiseed_ood": {
        "generated_files": [
            "ood_multiseed_results/*.csv",
        ],
        "modified_files": [],
        "notes": "耗时长",
    },
}

# 未注册命令的默认模板
DEFAULT_COMMAND_META = {
    "generated_files": [],
    "modified_files": [],
    "notes": "未在 command_registry 中注册的命令",
}
