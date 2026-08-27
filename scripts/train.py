import torch
import sys
import argparse
import os
import time
import math
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from torch.utils.data import WeightedRandomSampler
from transformers import Trainer, TrainingArguments, AutoTokenizer, TrainerCallback, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.model_architectures.multi_modal_model import MultiModalFusionModel, FocalLoss
from src.data.data_loader import generate_mock_data, load_real_data, load_split_data, collate_fn
from utils.log_utils import save_log


def augmentation_collate_fn(batch, tokenizer=None, max_length=128,
                            use_numeric=True, use_bert=True, use_llm=True,
                            noise_std=0.01, augmentation_prob=0.5):
    """
    带数据增强的批处理函数
    
    功能：在标准collate_fn基础上，对数值特征添加随机高斯噪声增强
    
    Args:
        batch: 样本列表
        tokenizer: LLM tokenizer
        max_length: 文本最大长度
        use_numeric: 是否使用数值模态
        use_bert: 是否使用文本模态
        use_llm: 是否使用LLM
        noise_std: 高斯噪声标准差
        augmentation_prob: 应用增强的概率
        
    Returns:
        dict: 整理后的batch数据
    """
    # 先使用标准collate_fn处理
    result = collate_fn(batch, tokenizer, max_length, use_numeric, use_bert, use_llm)
    
    # 对数值特征进行数据增强
    if use_numeric and noise_std > 0:
        stat_tensor = result["stat_tensor"]
        batch_size = stat_tensor.shape[0]
        
        # 随机决定哪些样本应用增强
        mask = torch.rand(batch_size, 1) < augmentation_prob
        noise = torch.randn_like(stat_tensor) * noise_std
        stat_tensor = stat_tensor + mask.float() * noise
        
        result["stat_tensor"] = stat_tensor
    
    return result


# 消融实验变体配置映射
VARIANT_CONFIGS = {
    # 基础模态消融
    "A0": {"use_numeric": True,  "use_bert": True,  "use_llm": True,  "fusion_type": "concat", "bert_trainable": False},
    "A1": {"use_numeric": True,  "use_bert": False, "use_llm": True,  "fusion_type": "concat", "bert_trainable": False},
    "A2": {"use_numeric": False, "use_bert": True,  "use_llm": True,  "fusion_type": "concat", "bert_trainable": False},
    "A3": {"use_numeric": True,  "use_bert": True,  "use_llm": False, "fusion_type": "concat", "bert_trainable": False},
    # 第三阶段改进消融（在A0基础上）
    "B0": {"use_focal_loss": False, "use_prototype_learning": False, "use_augmentation": False},
    "B1": {"use_focal_loss": True,  "use_prototype_learning": False, "use_augmentation": False},
    "B2": {"use_focal_loss": False, "use_prototype_learning": True,  "use_augmentation": False},
    "B3": {"use_focal_loss": False, "use_prototype_learning": False, "use_augmentation": True},
    "B4": {"use_focal_loss": True,  "use_prototype_learning": True,  "use_augmentation": False},
    "B5": {"use_focal_loss": True,  "use_prototype_learning": False, "use_augmentation": True},
    "B6": {"use_focal_loss": False, "use_prototype_learning": True,  "use_augmentation": True},
    "B7": {"use_focal_loss": True,  "use_prototype_learning": True,  "use_augmentation": True},
}

VARIANT_DESCRIPTIONS = {
    "A0": "全模型（数值+文本+LLM）",
    "A1": "仅数值+LLM（无文本）",
    "A2": "仅文本+LLM（无数值）",
    "A3": "无LLM（纯MLP分类）",
    "B0": "基线模型（无第三阶段改进）",
    "B1": "仅Focal Loss",
    "B2": "仅原型对比学习",
    "B3": "仅数据增强",
    "B4": "Focal Loss + 原型对比学习",
    "B5": "Focal Loss + 数据增强",
    "B6": "原型对比学习 + 数据增强",
    "B7": "全部改进（Focal + 原型 + 增强）",
}


class TrainerWithSampler(Trainer):
    """
    支持自定义 train_sampler 的 Trainer（兼容 transformers>=5.x 移除 train_sampler 参数的变更）。
    通过重写 _get_train_sampler 方法注入加权采样器。
    """

    def __init__(self, *args, train_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_train_sampler = train_sampler

    def _get_train_sampler(self, train_dataset=None):
        if self._custom_train_sampler is not None:
            return self._custom_train_sampler
        return super()._get_train_sampler(train_dataset)


class LossLoggerCallback(TrainerCallback):
    """
    自定义回调，用于记录训练过程中的loss值
    """
    
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.loss_log = []
        os.makedirs(log_dir, exist_ok=True)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """在日志记录时调用"""
        if logs is not None and 'loss' in logs:
            self.loss_log.append({
                'step': state.global_step,
                'epoch': state.epoch,
                'loss': logs['loss']
            })
    
    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时保存loss日志"""
        if self.loss_log:
            df = pd.DataFrame(self.loss_log)
            loss_file = os.path.join(self.log_dir, "loss_log.csv")
            df.to_csv(loss_file, index=False)
            print(f"\nLoss日志已保存到 {loss_file}")
            
            json_file = os.path.join(self.log_dir, "loss_log.json")
            df.to_json(json_file, orient='records', indent=2)
            print(f"Loss日志(JSON格式)已保存到 {json_file}")


class PerClassMetricsCallback(TrainerCallback):
    """
    每类指标追踪回调，在每次评估时记录每类的precision/recall/f1
    """
    
    def __init__(self, log_dir, num_classes, label_names=None):
        self.log_dir = log_dir
        self.num_classes = num_classes
        self.label_names = label_names or {i: str(i) for i in range(num_classes)}
        self.metrics_log = []
        os.makedirs(log_dir, exist_ok=True)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """评估时记录每类指标"""
        eval_output = kwargs.get('eval_output', None)
        if eval_output is not None and hasattr(eval_output, 'predictions'):
            preds = eval_output.predictions
            labels = eval_output.label_ids
            predictions = np.argmax(preds, axis=-1) if preds.ndim > 1 else preds
            
            per_class_precision = precision_score(labels, predictions, average=None, zero_division=0)
            per_class_recall = recall_score(labels, predictions, average=None, zero_division=0)
            per_class_f1 = f1_score(labels, predictions, average=None, zero_division=0)
            
            entry = {
                'epoch': state.epoch,
                'step': state.global_step,
            }
            for i in range(self.num_classes):
                name = self.label_names.get(i, str(i))
                entry[f'precision_{i}_{name}'] = per_class_precision[i] if i < len(per_class_precision) else 0.0
                entry[f'recall_{i}_{name}'] = per_class_recall[i] if i < len(per_class_recall) else 0.0
                entry[f'f1_{i}_{name}'] = per_class_f1[i] if i < len(per_class_f1) else 0.0
            
            self.metrics_log.append(entry)
    
    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时保存每类指标"""
        if self.metrics_log:
            df = pd.DataFrame(self.metrics_log)
            metrics_file = os.path.join(self.log_dir, "per_class_metrics.csv")
            df.to_csv(metrics_file, index=False)
            print(f"\n每类指标日志已保存到 {metrics_file}")


def generate_confusion_matrix(model, eval_dataset, device, save_path, label_names=None, tokenizer=None, use_llm=True):
    """
    生成并保存混淆矩阵可视化
    
    Args:
        model: 训练好的模型
        eval_dataset: 评估数据集（HuggingFace Dataset，包含 stat, bert, label, text 字段）
        device: 计算设备
        save_path: 保存路径
        label_names: 类别名称映射
        tokenizer: LLM tokenizer（用于文本编码）
        use_llm: 是否使用LLM
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    batch_size = 32
    num_samples = len(eval_dataset)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch = eval_dataset[start_idx:end_idx]
            
            stat_tensor = torch.tensor(batch['stat'], dtype=torch.float32).to(device)
            bert_tensor = torch.tensor(batch['bert'], dtype=torch.float32).to(device)
            labels = batch['label']
            
            input_ids = None
            attention_mask = None
            if use_llm and tokenizer is not None and 'text' in batch:
                texts = batch['text']
                if isinstance(texts, list):
                    encodings = tokenizer(
                        texts,
                        padding=True,
                        truncation=True,
                        max_length=128,
                        return_tensors="pt"
                    )
                    input_ids = encodings["input_ids"].to(device)
                    attention_mask = encodings["attention_mask"].to(device)
            
            outputs = model(stat_tensor, bert_tensor, input_ids=input_ids, 
                          attention_mask=attention_mask)
            logits = outputs['logits']
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    num_classes = max(max(all_labels), max(all_preds)) + 1
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    
    os.makedirs(save_path, exist_ok=True)
    
    cm_file = os.path.join(save_path, "confusion_matrix.csv")
    cm_df = pd.DataFrame(cm)
    if label_names:
        cm_df.columns = [label_names.get(i, str(i)) for i in range(num_classes)]
        cm_df.index = [label_names.get(i, str(i)) for i in range(num_classes)]
    cm_df.to_csv(cm_file)
    print(f"混淆矩阵已保存到 {cm_file}")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    if label_names:
        tick_marks = list(range(num_classes))
        tick_labels = [label_names.get(i, str(i)) for i in range(num_classes)]
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(tick_labels, fontsize=8)
    
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    plot_file = os.path.join(save_path, "confusion_matrix.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵图已保存到 {plot_file}")
    
    per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    print("\n各类别性能:")
    for i in range(num_classes):
        name = label_names.get(i, str(i)) if label_names else str(i)
        recall = per_class_recall[i] if i < len(per_class_recall) else 0.0
        f1 = per_class_f1[i] if i < len(per_class_f1) else 0.0
        print(f"  类别 {i} ({name}): Recall={recall:.4f}, F1={f1:.4f}")
    
    return cm


def get_next_model_id(base_dir=None):
    """获取下一个可用的模型ID（自动递增）"""
    if base_dir is None:
        base_dir = os.path.join(PROJECT_ROOT, "saved_models")
    os.makedirs(base_dir, exist_ok=True)
    
    max_id = -1
    for f in os.listdir(base_dir):
        if f.startswith("model_") and os.path.isdir(os.path.join(base_dir, f)):
            try:
                idx = int(f.replace("model_", ""))
                if idx > max_id:
                    max_id = idx
            except ValueError:
                pass
    
    return max_id + 1


def compute_class_weights(dataset, num_classes):
    """
    计算类别权重（反比于类别频率）
    
    Args:
        dataset: 训练数据集
        num_classes: 类别数量
    
    Returns:
        torch.Tensor: 类别权重张量
    """
    labels = [s['label'] for s in dataset]
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    
    class_weights = []
    for i in range(num_classes):
        if class_counts[i] > 0:
            weight = total_samples / (num_classes * class_counts[i])
        else:
            weight = 1.0
        class_weights.append(weight)
    
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    print(f"\n类别权重统计:")
    for i in range(num_classes):
        print(f"  类别 {i}: 样本数={class_counts[i]}, 权重={class_weights[i]:.4f}")
    
    return class_weights


def create_weighted_sampler(dataset, num_classes):
    """
    创建加权随机采样器
    
    Args:
        dataset: 训练数据集
        num_classes: 类别数量
    
    Returns:
        WeightedRandomSampler: 加权采样器
    """
    labels = [s['label'] for s in dataset]
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    
    class_weights = []
    for i in range(num_classes):
        if class_counts[i] > 0:
            weight = total_samples / (num_classes * class_counts[i])
        else:
            weight = 1.0
        class_weights.append(weight)
    
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    print(f"\n已创建WeightedRandomSampler，样本数={len(sample_weights)}")
    return sampler


def train_model(model_path=None, 
                output_dir=None, 
                save_path=None,
                model_id=None,
                dataset_id=0,
                split_id=0,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=2,
                learning_rate=5e-4,
                num_train_epochs=5,
                variant=None,
                use_numeric=None,
                use_bert=None,
                use_llm=None,
                fusion_type="concat",
                bert_trainable=False,
                seed=42,
                num_classes=None,
                use_class_weights=True,
                use_weighted_sampler=True,
                warmup_ratio=0.1,
                weight_decay=0.01,
                max_grad_norm=1.0,
                classifier_type="mlp",
                use_temperature=True,
                dropout_rate=0.1,
                use_focal_loss=False,
                focal_gamma=2.0,
                use_prototype_learning=False,
                prototype_temperature=0.07,
                prototype_loss_weight=0.1,
                use_augmentation=False,
                aug_noise_std=0.01,
                aug_prob=0.5):
    """
    训练多模态融合模型
    
    Args:
        model_path (str): LLM模型路径
        output_dir (str): Trainer临时输出目录
        save_path (str): 训练后模型参数保存目录
        model_id (int): 模型ID
        dataset_id (int): 数据集ID
        split_id (int): 划分ID
        per_device_train_batch_size (int): batch大小
        gradient_accumulation_steps (int): 梯度累积步数
        learning_rate (float): 学习率
        num_train_epochs (int): 训练轮数
        variant (str): 变体名 A0/A1/A2/A3
        use_numeric (bool): 是否使用数值模态
        use_bert (bool): 是否使用文本模态
        use_llm (bool): 是否使用LLM
        fusion_type (str): 融合策略
        bert_trainable (bool): BERT是否可训练
        seed (int): 随机种子
        num_classes (int): 分类类别数（None则自动从数据推断）
        use_class_weights (bool): 是否使用类别加权损失
        use_weighted_sampler (bool): 是否使用加权采样器
        warmup_ratio (float): 预热比例
        weight_decay (float): 权重衰减
        max_grad_norm (float): 梯度裁剪范数
        classifier_type (str): 分类器类型
        use_temperature (bool): 是否使用温度缩放
        dropout_rate (float): MLP分类器的dropout率
        use_focal_loss (bool): 是否使用Focal Loss
        focal_gamma (float): Focal Loss的聚焦参数
        use_prototype_learning (bool): 是否使用类别原型对比学习
        prototype_temperature (float): 原型对比学习的温度参数
        prototype_loss_weight (float): 原型对比学习损失的权重
        use_augmentation (bool): 是否使用数值特征数据增强
        aug_noise_std (float): 数据增强高斯噪声标准差
        aug_prob (float): 数据增强应用概率
        
    Returns:
        MultiModalFusionModel: 训练完成的模型
    """
    start_time = time.time()

    # 如果指定了变体，从配置映射中获取参数
    if variant is not None:
        if variant not in VARIANT_CONFIGS:
            raise ValueError(f"未知变体: {variant}. 可用变体为: {list(VARIANT_CONFIGS.keys())}")
        config = VARIANT_CONFIGS[variant]
        
        # A系列：基础模态消融
        if variant.startswith('A'):
            use_numeric = config['use_numeric']
            use_bert = config['use_bert']
            use_llm = config['use_llm']
            fusion_type = config['fusion_type']
            bert_trainable = config['bert_trainable']
        # B系列：第三阶段改进消融（使用A0作为基础配置）
        elif variant.startswith('B'):
            base_config = VARIANT_CONFIGS['A0']
            use_numeric = base_config['use_numeric']
            use_bert = base_config['use_bert']
            use_llm = base_config['use_llm']
            fusion_type = base_config['fusion_type']
            bert_trainable = base_config['bert_trainable']
            
            # 应用B系列的改进配置
            if 'use_focal_loss' in config:
                use_focal_loss = config['use_focal_loss']
            if 'use_prototype_learning' in config:
                use_prototype_learning = config['use_prototype_learning']
            if 'use_augmentation' in config:
                use_augmentation = config['use_augmentation']
    else:
        # 使用显式传入的参数，默认全部开启
        if use_numeric is None:
            use_numeric = True
        if use_bert is None:
            use_bert = True
        if use_llm is None:
            use_llm = True

    # 无LLM变体不需要GPU强制检查
    if use_llm:
        if not torch.cuda.is_available():
            print("=" * 60)
            print("警告: 未检测到GPU (CUDA)！")
            print("当前将使用CPU运行，模型训练速度会非常慢。")
            print("您的机器有NVIDIA驱动(566.24)但PyTorch是CPU版本。")
            print("")
            print("如需安装GPU版PyTorch，请运行:")
            print("  pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124")
            print("由于是非交互式运行，自动继续CPU模式...")
            print("=" * 60)
    
    if model_path is None:
        model_path = os.path.join(PROJECT_ROOT, "models", "qwen2.5-1.5b")
    
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "models")
    
    if model_id is None:
        model_id = get_next_model_id()
    
    if save_path is None:
        save_path = os.path.join(PROJECT_ROOT, "saved_models", f"model_{model_id}")
    
    variant_desc = VARIANT_DESCRIPTIONS.get(variant, "自定义配置") if variant else "自定义配置"
    
    print(f"\n训练参数:")
    print(f"  - 变体: {variant} ({variant_desc})" if variant else f"  - 变体: 自定义")
    print(f"  - 模型ID: {model_id}")
    print(f"  - 数据集ID: {dataset_id}")
    print(f"  - 划分ID: {split_id}")
    print(f"  - 数值模态: {'开启' if use_numeric else '关闭'}")
    print(f"  - 文本模态: {'开启' if use_bert else '关闭'}")
    print(f"  - LLM: {'开启' if use_llm else '关闭'}")
    print(f"  - 融合策略: {fusion_type}")
    print(f"  - BERT可训练: {'是' if bert_trainable else '否'}")
    print(f"  - 模型保存路径: {save_path}")

    train_dataset = load_split_data(data_dir=os.path.join(PROJECT_ROOT, "split_data"), data_type="train", 
                                    dataset_id=dataset_id, split_id=split_id)
    if train_dataset is None:
        print("未找到划分训练数据，将使用完整处理后的数据...")
        train_dataset = load_real_data(data_dir=os.path.join(PROJECT_ROOT, "processed_data"))
        if train_dataset is None:
            print("未找到完整处理后的数据，将使用模拟数据进行测试...")
            train_dataset = generate_mock_data(200)
    
    if num_classes is None:
        all_labels = [s['label'] for s in train_dataset]
        num_classes = max(all_labels) + 1
        print(f"自动检测类别数: {num_classes}")
    
    val_dataset = load_split_data(data_dir=os.path.join(PROJECT_ROOT, "split_data"), data_type="val",
                                  dataset_id=dataset_id, split_id=split_id)
    if val_dataset is None:
        print("未找到验证集(val)数据，尝试使用测试集(test)作为验证集...")
        val_dataset = load_split_data(data_dir=os.path.join(PROJECT_ROOT, "split_data"), data_type="test",
                                      dataset_id=dataset_id, split_id=split_id)
    if val_dataset is None:
        print("警告：未找到任何验证集数据，将跳过训练过程中的评估。")

    # 计算类别权重（用于加权损失函数）
    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(train_dataset, num_classes)
    
    # 创建加权采样器
    sampler = None
    if use_weighted_sampler:
        sampler = create_weighted_sampler(train_dataset, num_classes)

    # 初始化模型（需在 num_classes 确定后）
    model = MultiModalFusionModel(
        llm_model_path=model_path,
        use_numeric=use_numeric,
        use_bert=use_bert,
        use_llm=use_llm,
        fusion_type=fusion_type,
        bert_trainable=bert_trainable,
        num_classes=num_classes,
        class_weights=class_weights,
        classifier_type=classifier_type,
        use_temperature=use_temperature,
        dropout_rate=dropout_rate,
        use_focal_loss=use_focal_loss,
        focal_gamma=focal_gamma,
        use_prototype_learning=use_prototype_learning,
        prototype_temperature=prototype_temperature,
        prototype_loss_weight=prototype_loss_weight
    )
    
    # 无LLM时不需要tokenizer
    tokenizer = None
    if use_llm:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    def custom_collate(batch):
        if use_augmentation:
            return augmentation_collate_fn(
                batch, 
                tokenizer=tokenizer,
                use_numeric=use_numeric,
                use_bert=use_bert,
                use_llm=use_llm,
                noise_std=aug_noise_std,
                augmentation_prob=aug_prob
            )
        else:
            return collate_fn(
                batch, 
                tokenizer=tokenizer,
                use_numeric=use_numeric,
                use_bert=use_bert,
                use_llm=use_llm
            )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            'accuracy': accuracy_score(labels, predictions),
            'macro_precision': precision_score(labels, predictions, average='macro', zero_division=0),
            'macro_recall': recall_score(labels, predictions, average='macro', zero_division=0),
            'macro_f1': f1_score(labels, predictions, average='macro', zero_division=0),
            'weighted_precision': precision_score(labels, predictions, average='weighted', zero_division=0),
            'weighted_recall': recall_score(labels, predictions, average='weighted', zero_division=0),
            'weighted_f1': f1_score(labels, predictions, average='weighted', zero_division=0),
        }

    # 无LLM变体使用float32，不需要bf16
    steps_per_epoch = math.ceil(len(train_dataset) / (per_device_train_batch_size * gradient_accumulation_steps))
    total_training_steps = steps_per_epoch * num_train_epochs
    warmup_steps = int(warmup_ratio * total_training_steps)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        bf16=use_llm and torch.cuda.is_available(),  # CPU上不使用bf16
        fp16=False,  # 禁用fp16避免CPU警告
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        eval_strategy="epoch" if val_dataset is not None else "no",
        save_strategy="epoch",
        load_best_model_at_end=val_dataset is not None,
        metric_for_best_model="eval_macro_f1" if val_dataset is not None else None,
        greater_is_better=True,
        save_total_limit=3,
        seed=seed,
        data_seed=seed,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
    )

    # 创建Loss日志目录
    loss_log_dir = os.path.join(save_path, "loss_logs")
    
    # 加载标签映射（用于混淆矩阵和每类指标）
    try:
        from config.label_config import MERGED_LABEL_NAMES
        label_names = MERGED_LABEL_NAMES
    except ImportError:
        label_names = {i: str(i) for i in range(num_classes)}
    
    callbacks = [LossLoggerCallback(loss_log_dir)]
    callbacks.append(PerClassMetricsCallback(loss_log_dir, num_classes, label_names))
    # 仅当存在验证集时启用早停
    if val_dataset is not None:
        if use_llm:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
        else:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))
    
    trainer_kwargs = {
        'model': model,
        'args': training_args,
        'train_dataset': train_dataset,
        'eval_dataset': val_dataset,
        'data_collator': custom_collate,
        'compute_metrics': compute_metrics,
        'callbacks': callbacks,
    }
    
    # 如果使用加权采样器，添加到Trainer
    if sampler is not None:
        trainer_kwargs['train_sampler'] = sampler
        print("使用WeightedRandomSampler进行类别平衡采样")
    
    trainer = TrainerWithSampler(**trainer_kwargs)

    print("Starting training...")
    trainer.train()
    print("Training finished!")
    
    duration_seconds = time.time() - start_time
    
    print(f"\n保存模型到 {save_path}...")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    
    # 生成混淆矩阵
    if val_dataset is not None:
        print("\n生成混淆矩阵...")
        try:
            generate_confusion_matrix(
                model=model,
                eval_dataset=val_dataset,
                device=model.device,
                save_path=save_path,
                label_names=label_names,
                tokenizer=tokenizer,
                use_llm=use_llm
            )
        except Exception as e:
            print(f"混淆矩阵生成失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存配置信息
    with open(os.path.join(save_path, "config.txt"), "w") as f:
        f.write(f"model_id: {model_id}\n")
        f.write(f"variant: {variant}\n")
        f.write(f"variant_desc: {variant_desc}\n")
        f.write(f"dataset_id: {dataset_id}\n")
        f.write(f"split_id: {split_id}\n")
        f.write(f"model_path: {model_path}\n")
        f.write(f"use_numeric: {use_numeric}\n")
        f.write(f"use_bert: {use_bert}\n")
        f.write(f"use_llm: {use_llm}\n")
        f.write(f"fusion_type: {fusion_type}\n")
        f.write(f"bert_trainable: {bert_trainable}\n")
        f.write(f"num_classes: {num_classes}\n")
        f.write(f"per_device_train_batch_size: {per_device_train_batch_size}\n")
        f.write(f"gradient_accumulation_steps: {gradient_accumulation_steps}\n")
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"num_train_epochs: {num_train_epochs}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"use_class_weights: {use_class_weights}\n")
        f.write(f"use_weighted_sampler: {use_weighted_sampler}\n")
        f.write(f"warmup_ratio: {warmup_ratio}\n")
        f.write(f"weight_decay: {weight_decay}\n")
        f.write(f"max_grad_norm: {max_grad_norm}\n")
        f.write(f"classifier_type: {classifier_type}\n")
        f.write(f"use_temperature: {use_temperature}\n")
        f.write(f"dropout_rate: {dropout_rate}\n")
        f.write(f"use_focal_loss: {use_focal_loss}\n")
        f.write(f"focal_gamma: {focal_gamma}\n")
        f.write(f"use_prototype_learning: {use_prototype_learning}\n")
        f.write(f"prototype_temperature: {prototype_temperature}\n")
        f.write(f"prototype_loss_weight: {prototype_loss_weight}\n")
        f.write(f"use_augmentation: {use_augmentation}\n")
        f.write(f"aug_noise_std: {aug_noise_std}\n")
        f.write(f"aug_prob: {aug_prob}\n")
        f.write(f"duration_seconds: {duration_seconds}\n")
    
    print(f"模型配置已保存到 {os.path.join(save_path, 'config.txt')}")
    
    # 计算可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    log_data = {
        'model_id': model_id,
        'variant': variant,
        'variant_desc': variant_desc,
        'dataset_id': dataset_id,
        'split_id': split_id,
        'use_numeric': use_numeric,
        'use_bert': use_bert,
        'use_llm': use_llm,
        'fusion_type': fusion_type,
        'bert_trainable': bert_trainable,
        'num_classes': num_classes,
        'learning_rate': learning_rate,
        'epochs': num_train_epochs,
        'batch_size': per_device_train_batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'model_path': model_path,
        'save_path': os.path.abspath(save_path),
        'loss_log_path': os.path.abspath(loss_log_dir),
        'duration_seconds': duration_seconds,
        'trainable_params': trainable_params,
        'seed': seed,
        'use_class_weights': use_class_weights,
        'use_weighted_sampler': use_weighted_sampler,
        'warmup_ratio': warmup_ratio,
        'weight_decay': weight_decay,
        'max_grad_norm': max_grad_norm,
        'classifier_type': classifier_type,
        'use_temperature': use_temperature,
        'dropout_rate': dropout_rate,
        'use_focal_loss': use_focal_loss,
        'focal_gamma': focal_gamma,
        'use_prototype_learning': use_prototype_learning,
        'prototype_temperature': prototype_temperature,
        'prototype_loss_weight': prototype_loss_weight,
        'use_augmentation': use_augmentation,
        'aug_noise_std': aug_noise_std,
        'aug_prob': aug_prob
    }
    log_id = save_log('training', log_data)
    print(f"\n模型训练日志已保存: logs/training/log_{log_id}.json")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练多模态融合模型（支持消融实验）")
    parser.add_argument("--model_path", type=str, default=None, help="LLM模型路径")
    parser.add_argument("--model_id", type=int, default=None, help="模型ID（默认自动递增）")
    parser.add_argument("--dataset_id", type=int, default=0, help="数据集ID")
    parser.add_argument("--split_id", type=int, default=0, help="划分ID")
    parser.add_argument("--batch_size", type=int, default=4, help="每个设备的batch大小")
    parser.add_argument("--gradient_accumulation", type=int, default=2, help="梯度累积步数")
    parser.add_argument("--lr", type=float, default=5e-4, help="学习率")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--variant", type=str, default=None, help="消融变体 A0/A1/A2/A3")
    parser.add_argument("--use_numeric", action='store_true', default=None, help="使用数值模态")
    parser.add_argument("--use_bert", action='store_true', default=None, help="使用文本模态")
    parser.add_argument("--use_llm", action='store_true', default=None, help="使用LLM")
    parser.add_argument("--fusion_type", type=str, default="concat", help="融合策略 concat/add/attention")
    parser.add_argument("--bert_trainable", action='store_true', default=False, help="BERT可训练")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num_classes", type=int, default=None, help="分类类别数（默认自动检测）")
    parser.add_argument("--disable_class_weights", action='store_true', help="禁用类别加权损失")
    parser.add_argument("--disable_weighted_sampler", action='store_true', help="禁用加权采样器")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="预热比例")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪范数")
    parser.add_argument("--classifier_type", type=str, default="mlp", help="分类器类型: linear 或 mlp")
    parser.add_argument("--disable_temperature", action='store_true', help="禁用温度缩放")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="MLP分类器dropout率")
    parser.add_argument("--use_focal_loss", action='store_true', help="使用Focal Loss")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal Loss聚焦参数")
    parser.add_argument("--use_prototype_learning", action='store_true', help="使用类别原型对比学习")
    parser.add_argument("--prototype_temperature", type=float, default=0.07, help="原型对比学习温度参数")
    parser.add_argument("--prototype_loss_weight", type=float, default=0.1, help="原型对比学习损失权重")
    parser.add_argument("--use_augmentation", action='store_true', help="使用数值特征数据增强")
    parser.add_argument("--aug_noise_std", type=float, default=0.01, help="数据增强高斯噪声标准差")
    parser.add_argument("--aug_prob", type=float, default=0.5, help="数据增强应用概率")
    args = parser.parse_args()

    train_model(
        model_path=args.model_path,
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        split_id=args.split_id,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        variant=args.variant,
        use_numeric=args.use_numeric,
        use_bert=args.use_bert,
        use_llm=args.use_llm,
        fusion_type=args.fusion_type,
        bert_trainable=args.bert_trainable,
        seed=args.seed,
        num_classes=args.num_classes,
        use_class_weights=not args.disable_class_weights,
        use_weighted_sampler=not args.disable_weighted_sampler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        classifier_type=args.classifier_type,
        use_temperature=not args.disable_temperature,
        dropout_rate=args.dropout_rate,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        use_prototype_learning=args.use_prototype_learning,
        prototype_temperature=args.prototype_temperature,
        prototype_loss_weight=args.prototype_loss_weight,
        use_augmentation=args.use_augmentation,
        aug_noise_std=args.aug_noise_std,
        aug_prob=args.aug_prob
    )
