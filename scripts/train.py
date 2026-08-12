import torch
import sys
import argparse
import os
import time
import pandas as pd
import numpy as np
from transformers import Trainer, TrainingArguments, AutoTokenizer, TrainerCallback, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.model_architectures.multi_modal_model import MultiModalFusionModel
from src.data.data_loader import generate_mock_data, load_real_data, load_split_data, collate_fn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, PROJECT_ROOT)
from utils.log_utils import save_log


# 消融实验变体配置映射
VARIANT_CONFIGS = {
    "A0": {"use_numeric": True,  "use_bert": True,  "use_llm": True,  "fusion_type": "concat", "bert_trainable": False},
    "A1": {"use_numeric": True,  "use_bert": False, "use_llm": True,  "fusion_type": "concat", "bert_trainable": False},
    "A2": {"use_numeric": False, "use_bert": True,  "use_llm": True,  "fusion_type": "concat", "bert_trainable": False},
    "A3": {"use_numeric": True,  "use_bert": True,  "use_llm": False, "fusion_type": "concat", "bert_trainable": False},
}

VARIANT_DESCRIPTIONS = {
    "A0": "全模型（数值+文本+LLM）",
    "A1": "仅数值+LLM（无文本）",
    "A2": "仅文本+LLM（无数值）",
    "A3": "无LLM（纯MLP分类）",
}


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


def train_model(model_path=None, 
                output_dir=None, 
                save_path=None,
                model_id=None,
                dataset_id=0,
                split_id=0,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                learning_rate=1e-4,
                num_train_epochs=3,
                variant=None,
                use_numeric=None,
                use_bert=None,
                use_llm=None,
                fusion_type="concat",
                bert_trainable=False,
                seed=42):
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
        
    Returns:
        MultiModalFusionModel: 训练完成的模型
    """
    start_time = time.time()

    # 如果指定了变体，从配置映射中获取参数
    if variant is not None:
        if variant not in VARIANT_CONFIGS:
            raise ValueError(f"未知变体: {variant}. 可用变体为: {list(VARIANT_CONFIGS.keys())}")
        config = VARIANT_CONFIGS[variant]
        use_numeric = config['use_numeric']
        use_bert = config['use_bert']
        use_llm = config['use_llm']
        fusion_type = config['fusion_type']
        bert_trainable = config['bert_trainable']
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
            print("请确认是否继续...")
            print("=" * 60)
            choice = input("输入 'y' 继续使用CPU，输入其他键退出: ")
            if choice.strip().lower() != 'y':
                raise RuntimeError("用户选择终止：未检测到GPU。请检查CUDA环境或安装GPU版PyTorch。")
    
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

    # 初始化模型
    model = MultiModalFusionModel(
        llm_model_path=model_path,
        use_numeric=use_numeric,
        use_bert=use_bert,
        use_llm=use_llm,
        fusion_type=fusion_type,
        bert_trainable=bert_trainable
    )
    
    train_dataset = load_split_data(data_dir=os.path.join(PROJECT_ROOT, "split_data"), data_type="train", 
                                    dataset_id=dataset_id, split_id=split_id)
    if train_dataset is None:
        print("未找到划分训练数据，将使用完整处理后的数据...")
        train_dataset = load_real_data(data_dir=os.path.join(PROJECT_ROOT, "processed_data"))
        if train_dataset is None:
            print("未找到完整处理后的数据，将使用模拟数据进行测试...")
            train_dataset = generate_mock_data(200)
    
    val_dataset = load_split_data(data_dir=os.path.join(PROJECT_ROOT, "split_data"), data_type="val",
                                  dataset_id=dataset_id, split_id=split_id)
    
    # 无LLM时不需要tokenizer
    tokenizer = None
    if use_llm:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    def custom_collate(batch):
        return collate_fn(batch, 
                         tokenizer=tokenizer,
                         use_numeric=use_numeric,
                         use_bert=use_bert,
                         use_llm=use_llm)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'f1': f1_score(labels, predictions, zero_division=0),
        }

    # 无LLM变体使用float32，不需要bf16
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        bf16=use_llm,  # 无LLM时使用float32
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=3,
        seed=seed,
        data_seed=seed,
    )

    # 创建Loss日志目录
    loss_log_dir = os.path.join(save_path, "loss_logs")
    
    callbacks = [LossLoggerCallback(loss_log_dir)]
    # 无LLM变体收敛快，减小早停patience
    if use_llm:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
    else:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=custom_collate,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    print("Starting training...")
    trainer.train()
    print("Training finished!")
    
    duration_seconds = time.time() - start_time
    
    print(f"\n保存模型到 {save_path}...")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    
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
        f.write(f"per_device_train_batch_size: {per_device_train_batch_size}\n")
        f.write(f"gradient_accumulation_steps: {gradient_accumulation_steps}\n")
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"num_train_epochs: {num_train_epochs}\n")
        f.write(f"seed: {seed}\n")
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
        'learning_rate': learning_rate,
        'epochs': num_train_epochs,
        'batch_size': per_device_train_batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'model_path': model_path,
        'save_path': os.path.abspath(save_path),
        'loss_log_path': os.path.abspath(loss_log_dir),
        'duration_seconds': duration_seconds,
        'trainable_params': trainable_params,
        'seed': seed
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
    parser.add_argument("--batch_size", type=int, default=2, help="每个设备的batch大小")
    parser.add_argument("--gradient_accumulation", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--variant", type=str, default=None, help="消融变体 A0/A1/A2/A3")
    parser.add_argument("--use_numeric", action='store_true', default=None, help="使用数值模态")
    parser.add_argument("--use_bert", action='store_true', default=None, help="使用文本模态")
    parser.add_argument("--use_llm", action='store_true', default=None, help="使用LLM")
    parser.add_argument("--fusion_type", type=str, default="concat", help="融合策略 concat/add/attention")
    parser.add_argument("--bert_trainable", action='store_true', default=False, help="BERT可训练")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
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
        seed=args.seed
    )
