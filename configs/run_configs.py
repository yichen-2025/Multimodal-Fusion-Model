"""
运行配置字典

分为两类：
  1. 默认配置（xxx_default）：展示各步骤的标准默认参数，作为参考基准
  2. 自定义配置（xxx）：开发者修改这些即可，放在文件末尾方便查找

格式：{配置名: {step: 步骤ID, params: {参数字典}}}
注意：params 中的 key 必须与对应函数的参数名完全一致。
      标注 [自动生成] 的参数通常不需要手动设置（为 None 时自动递增）。
"""

RUN_CONFIG = {
    # ═══════════════════════════════════════════════════════
    # 第一部分：默认配置（展示标准参数，供参考）
    # ═══════════════════════════════════════════════════════

    # —— P0 数据预处理 ——
    "clean": {
        "step": "clean",
        "params": {}
    },
    "subset_default": {
        "step": "subset",
        "params": {
            "total_samples": 5000,
            "ratio": None,
            "mode": "stratified",
            "per_class": None,
            "dataset_id": 0,
            "random_state": 42
        }
    },
    "split_default": {
        "step": "split",
        "params": {
            "dataset_id": 0,
            "split_id": None,
            "test_size": 0.2,
            "val_size": 0.1,
            "random_state": 42,
            "input_csv": None
        }
    },

    # —— P0 模型训练 ——
    "train_default": {
        "step": "train",
        "params": {
            "variant": "A3",
            "dataset_id": 0,
            "split_id": 0,
            "model_path": None,
            "output_dir": None,
            "save_path": None,
            "model_id": None,
            "learning_rate": 5e-4,
            "num_train_epochs": 5,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "classifier_type": "mlp",
            "use_temperature": True,
            "dropout_rate": 0.1,
            "use_focal_loss": False,
            "focal_gamma": 2.0,
            "use_prototype_learning": False,
            "prototype_temperature": 0.07,
            "prototype_loss_weight": 0.1,
            "use_augmentation": False,
            "aug_noise_std": 0.01,
            "aug_prob": 0.5,
            "use_numeric": None,
            "use_bert": None,
            "use_llm": None,
            "fusion_type": "concat",
            "bert_trainable": False,
            "num_classes": None,
            "use_class_weights": True,
            "use_weighted_sampler": True,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "seed": 42
        }
    },
    "test_default": {
        "step": "test",
        "params": {
            "dataset_id": 0,
            "split_id": 0,
            "model_id": 0,
            "llm_model_path": None,
            "verbose": True,
            "save_report": True
        }
    },
    "ablation_default": {
        "step": "ablation",
        "params": {
            "variants": ["A0", "A1", "A2", "A3"],
            "dataset_id": 0,
            "split_id": 0,
            "model_path": None,
            "output_csv": None,
            "repeat": 1,
            "seed": 42
        }
    },

    # —— P1 开集检测 ——
    "openset_default": {
        "step": "openset",
        "params": {
            "dataset_id": 0,
            "source_split_id": 0,
            "output_split_id": None,
            "unknown_ratio": 0.3,
            "random_state": 42,
            "hold_out_classes": None
        }
    },
    "fewshot_default": {
        "step": "fewshot",
        "params": {
            "dataset_id": 0,
            "source_split_id": 0,
            "k_per_class": 5,
            "output_split_id": None,
            "include_remaining_as_test": True,
            "random_state": 42
        }
    },
    "ood_train_default": {
        "step": "ood-train",
        "params": {
            "model_id": 0,
            "ood_id": None,
            "dataset_id": 0,
            "split_id": 0,
            "model_path": None,
            "variant": "A3",
            "fewshot_k": 5,
            "distance_type": "cosine",
            "temperature": 1.0,
            "learning_rate": 1e-3,
            "num_epochs": 10,
            "batch_size": 64,
            "seed": 42
        }
    },
    "ood_eval_default": {
        "step": "ood-eval",
        "params": {
            "backbone_model_id": 0,
            "ood_id": 0,
            "llm_model_id": None,
            "dataset_id": 0,
            "split_id": 0,
            "model_path": None,
            "backbone_variant": "A3",
            "llm_variant": "A0",
            "fewshot_k": 5,
            "batch_size": 32,
            "verbose": True,
            "compare_baseline": True
        }
    },
    "experiment_default": {
        "step": "experiment",
        "params": {
            "dataset_id": 0,
            "split_id": 0,
            "k_values": [5, 10, 20, None],
            "backbone_model_id": 0,
            "llm_model_id": None,
            "model_path": None,
            "backbone_variant": "A3",
            "llm_variant": "A0",
            "distance_type": "cosine",
            "temperature": 1.0,
            "seed": 42,
            "train_batch_size": 2,
            "train_grad_accum": 4,
            "train_lr": 1e-4,
            "train_epochs": 3,
            "ood_lr": 1e-3,
            "ood_epochs": 10,
            "ood_batch_size": 64,
            "eval_batch_size": 32,
            "skip_backbone_train": False,
            "skip_ood_train": False,
            "skip_llm": False,
            "generate_report": True
        }
    },
    "report_default": {
        "step": "report",
        "params": {
            "output_dir": None
        }
    },

    # ═══════════════════════════════════════════════════════
    # 第二部分：自定义配置（开发者修改这些，参数已预置默认值）
    # ═══════════════════════════════════════════════════════

    # —— P0 数据预处理 ——
    "subset": {
        "step": "subset",
        "params": {
            "total_samples": 5000,
            "ratio": None,
            "mode": "stratified",
            "per_class": None,
            "dataset_id": 0,
            "random_state": 42
        }
    },
    "split": {
        "step": "split",
        "params": {
            "dataset_id": 0,
            "split_id": None,
            "test_size": 0.2,
            "val_size": 0.1,
            "random_state": 42,
            "input_csv": None
        }
    },

    # —— P0 模型训练 ——
    "train": {
        "step": "train",
        "params": {
            "variant": "A0",
            "dataset_id": 0,
            "split_id": 0,
            "model_path": None,
            "output_dir": None,
            "save_path": None,
            "model_id": None,
            "learning_rate": 5e-4,
            "num_train_epochs": 10,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "classifier_type": "mlp",
            "use_temperature": True,
            "dropout_rate": 0.1,
            "use_focal_loss": False,
            "focal_gamma": 2.0,
            "use_prototype_learning": False,
            "prototype_temperature": 0.07,
            "prototype_loss_weight": 0.1,
            "use_augmentation": False,
            "aug_noise_std": 0.01,
            "aug_prob": 0.5,
            "use_numeric": None,
            "use_bert": None,
            "use_llm": None,
            "fusion_type": "concat",
            "bert_trainable": False,
            "num_classes": None,
            "use_class_weights": True,
            "use_weighted_sampler": True,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "seed": 42
        }
    },
    "test": {
        "step": "test",
        "params": {
            "dataset_id": 0,
            "split_id": 0,
            "model_id": 0,
            "llm_model_path": None,
            "verbose": True,
            "save_report": True
        }
    },
    "ablation": {
        "step": "ablation",
        "params": {
            "variants": ["A3", "A0*", "A0_frozen"],
            "dataset_id": 0,
            "split_id": 0,
            "model_path": None,
            "output_csv": None,
            "repeat": 1,
            "seed": 42
        }
    },

    # —— P1 开集检测 ——
    "openset": {
        "step": "openset",
        "params": {
            "dataset_id": 0,
            "source_split_id": 0,
            "output_split_id": None,
            "unknown_ratio": 0.3,
            "random_state": 42,
            "hold_out_classes": None
        }
    },
    "fewshot": {
        "step": "fewshot",
        "params": {
            "dataset_id": 0,
            "source_split_id": 0,
            "k_per_class": 5,
            "output_split_id": None,
            "include_remaining_as_test": True,
            "random_state": 42
        }
    },
    "ood_train": {
        "step": "ood-train",
        "params": {
            "model_id": 0,
            "ood_id": None,
            "dataset_id": 0,
            "split_id": 0,
            "model_path": None,
            "variant": "A3",
            "fewshot_k": 5,
            "distance_type": "cosine",
            "temperature": 1.0,
            "learning_rate": 1e-3,
            "num_epochs": 10,
            "batch_size": 64,
            "seed": 42
        }
    },
    "ood_eval": {
        "step": "ood-eval",
        "params": {
            "backbone_model_id": 0,
            "ood_id": 0,
            "llm_model_id": None,
            "dataset_id": 0,
            "split_id": 0,
            "model_path": None,
            "backbone_variant": "A3",
            "llm_variant": "A0",
            "fewshot_k": 5,
            "batch_size": 32,
            "verbose": True,
            "compare_baseline": True
        }
    },
    "experiment": {
        "step": "experiment",
        "params": {
            "dataset_id": 0,
            "split_id": 0,
            "k_values": [5, 10, 20, None],
            "backbone_model_id": 0,
            "llm_model_id": None,
            "model_path": None,
            "backbone_variant": "A3",
            "llm_variant": "A0",
            "distance_type": "cosine",
            "temperature": 1.0,
            "seed": 42,
            "train_batch_size": 2,
            "train_grad_accum": 4,
            "train_lr": 1e-4,
            "train_epochs": 3,
            "ood_lr": 1e-3,
            "ood_epochs": 10,
            "ood_batch_size": 64,
            "eval_batch_size": 32,
            "skip_backbone_train": False,
            "skip_ood_train": False,
            "skip_llm": False,
            "generate_report": True
        }
    },
    "report": {
        "step": "report",
        "params": {
            "output_dir": None
        }
    },
}