import torch
import sys
import argparse
import os
import time
import pandas as pd
from transformers import Trainer, TrainingArguments, AutoTokenizer, TrainerCallback
from src.model_architectures.multi_modal_model import MultiModalFusionModel
from src.data.data_loader import generate_mock_data, load_real_data, load_split_data, collate_fn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.log_utils import save_log


class LossLoggerCallback(TrainerCallback):
    """
    自定义回调，用于记录训练过程中的loss值
    
    功能：在每个训练步骤结束后记录loss值，保存到loss_log.csv文件中
    
    Args:
        log_dir (str): loss日志保存目录
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
            
            # 同时保存为JSON格式，方便后续读取
            json_file = os.path.join(self.log_dir, "loss_log.json")
            df.to_json(json_file, orient='records', indent=2)
            print(f"Loss日志(JSON格式)已保存到 {json_file}")


def get_next_model_id(base_dir="./saved_models"):
    """获取下一个可用的模型ID（自动递增）"""
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


def train_model(model_path="./models/qwen2.5-1.5b", 
                output_dir="./models", 
                save_path=None,
                model_id=None,
                dataset_id=0,
                split_id=0,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                learning_rate=1e-4,
                num_train_epochs=3):
    """
    训练多模态融合模型
    
    功能：使用Hugging Face Trainer进行模型训练，包括数据加载、训练参数配置和训练执行
    
    训练流程：
    1. 初始化多模态融合模型
    2. 生成模拟训练数据（或加载真实数据）
    3. 配置训练参数（batch大小、学习率、训练轮数等）
    4. 创建Trainer并执行训练
    5. 保存训练后的模型参数
    6. 返回训练完成的模型
    
    Args:
        model_path (str): LLM模型路径或名称，默认使用本地Qwen2.5-1.5B
        output_dir (str): Trainer临时输出目录，默认保存到./models
        save_path (str): 训练后模型参数保存目录，默认使用自动生成的路径
        model_id (int): 模型ID（默认自动递增）
        dataset_id (int): 数据集ID，默认0
        split_id (int): 划分ID，默认0
        per_device_train_batch_size (int): 每个设备的batch大小
        gradient_accumulation_steps (int): 梯度累积步数
        learning_rate (float): 学习率
        num_train_epochs (int): 训练轮数
        
    Returns:
        MultiModalFusionModel: 训练完成的模型
    """
    start_time = time.time()
    
    if model_id is None:
        model_id = get_next_model_id()
    
    if save_path is None:
        save_path = os.path.join("./saved_models", f"model_{model_id}")
    
    print(f"\n训练参数:")
    print(f"  - 模型ID: {model_id}")
    print(f"  - 数据集ID: {dataset_id}")
    print(f"  - 划分ID: {split_id}")
    print(f"  - 模型保存路径: {save_path}")

    model = MultiModalFusionModel(llm_model_path=model_path)
    
    train_dataset = load_split_data(data_dir="split_data", data_type="train", 
                                    dataset_id=dataset_id, split_id=split_id)
    if train_dataset is None:
        print("Split train data not found, using full processed data...")
        train_dataset = load_real_data(data_dir="processed_data")
        if train_dataset is None:
            print("Real data not found, using mock data for testing...")
            train_dataset = generate_mock_data(200)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    def custom_collate(batch):
        return collate_fn(batch, tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False
    )

    # 创建Loss日志目录
    loss_log_dir = os.path.join(save_path, "loss_logs")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=custom_collate,
        callbacks=[LossLoggerCallback(loss_log_dir)]
    )

    print("Starting training...")
    trainer.train()
    print("Training finished!")
    
    duration_seconds = time.time() - start_time
    
    print(f"\n保存模型到 {save_path}...")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    
    with open(os.path.join(save_path, "config.txt"), "w") as f:
        f.write(f"model_id: {model_id}\n")
        f.write(f"dataset_id: {dataset_id}\n")
        f.write(f"split_id: {split_id}\n")
        f.write(f"model_path: {model_path}\n")
        f.write(f"per_device_train_batch_size: {per_device_train_batch_size}\n")
        f.write(f"gradient_accumulation_steps: {gradient_accumulation_steps}\n")
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"num_train_epochs: {num_train_epochs}\n")
        f.write(f"duration_seconds: {duration_seconds}\n")
    
    print(f"模型配置已保存到 {os.path.join(save_path, 'config.txt')}")
    
    log_data = {
        'model_id': model_id,
        'dataset_id': dataset_id,
        'split_id': split_id,
        'learning_rate': learning_rate,
        'epochs': num_train_epochs,
        'batch_size': per_device_train_batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'model_path': model_path,
        'save_path': os.path.abspath(save_path),
        'loss_log_path': os.path.abspath(loss_log_dir),
        'duration_seconds': duration_seconds
    }
    log_id = save_log('training', log_data)
    print(f"\n模型训练日志已保存: logs/training/log_{log_id}.json")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练多模态融合模型")
    parser.add_argument("--model_path", type=str, default="./models/qwen2.5-1.5b", help="LLM模型路径")
    parser.add_argument("--model_id", type=int, default=None, help="模型ID（默认自动递增）")
    parser.add_argument("--dataset_id", type=int, default=0, help="数据集ID")
    parser.add_argument("--split_id", type=int, default=0, help="划分ID")
    parser.add_argument("--batch_size", type=int, default=2, help="每个设备的batch大小")
    parser.add_argument("--gradient_accumulation", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    args = parser.parse_args()

    train_model(
        model_path=args.model_path,
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        split_id=args.split_id,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        num_train_epochs=args.epochs
    )