import pandas as pd
import numpy as np
import os
import sys
import argparse
import torch
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.model_architectures.bert_encoder import BertEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.log_utils import save_log

BASE_INPUT_DIR = "./processed_dataset"
BASE_OUTPUT_DIR = "./split_data"
BERT_MODEL_PATH = "./models/bert"
TEST_SIZE = 0.2

SELECTED_FEATURES = [
    "Bwd Packet Length Mean",
    "Avg Bwd Segment Size",
    "Bwd Packet Length Max",
    "Bwd Packet Length Std",
    "Destination Port",
    "URG Flag Count",
    "Packet Length Mean",
    "Average Packet Size",
    "Packet Length Std"
]


def get_next_split_id(dataset_id):
    """获取下一个可用的split ID（自动递增）"""
    dataset_split_dir = os.path.join(BASE_OUTPUT_DIR, f"dataset_{dataset_id}")
    os.makedirs(dataset_split_dir, exist_ok=True)
    
    if not os.path.exists(dataset_split_dir):
        return 0
    
    files = os.listdir(dataset_split_dir)
    max_id = -1
    for f in files:
        if f.startswith("split_") and os.path.isdir(os.path.join(dataset_split_dir, f)):
            try:
                idx = int(f.replace("split_", ""))
                if idx > max_id:
                    max_id = idx
            except ValueError:
                pass
    
    return max_id + 1


def generate_text_description(row):
    parts = []
    parts.append(f"目标端口是{int(row['Destination Port'])}")
    parts.append(f"反向包平均长度{row['Bwd Packet Length Mean']:.1f}字节")
    parts.append(f"反向段平均大小{row['Avg Bwd Segment Size']:.1f}字节")
    parts.append(f"反向包最大长度{int(row['Bwd Packet Length Max'])}字节")
    parts.append(f"反向包长度标准差{row['Bwd Packet Length Std']:.1f}")
    parts.append(f"URG标志计数{int(row['URG Flag Count'])}")
    parts.append(f"包平均长度{row['Packet Length Mean']:.1f}字节")
    parts.append(f"平均包大小{row['Average Packet Size']:.1f}字节")
    parts.append(f"包长度标准差{row['Packet Length Std']:.1f}")
    
    return "。".join(parts) + "。"


def split_modality(dataset_id=0, split_id=None, test_size=TEST_SIZE, random_state=42):
    start_time = time.time()
    print("=" * 60)
    print("模态分离脚本")
    print("=" * 60)

    input_csv = os.path.join(BASE_INPUT_DIR, f"dataset_{dataset_id}.csv")
    
    if split_id is None:
        split_id = get_next_split_id(dataset_id)
    
    output_dir = os.path.join(BASE_OUTPUT_DIR, f"dataset_{dataset_id}", f"split_{split_id}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n数据集ID: {dataset_id}")
    print(f"划分ID: {split_id}")

    print("\n1. 加载数据集子集...")
    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip()
    print(f"数据集: {df.shape[0]}行, {df.shape[1]}列")

    print("\n2. 统计特征模态（连续特征）...")
    X = df[SELECTED_FEATURES]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    print(f"  - 统计特征维度: {X_scaled.shape}")

    print("\n3. 文本描述模态（离散特征）...")
    if "text_description" not in df.columns:
        print("  - 生成文本描述...")
        df['text_description'] = df.apply(generate_text_description, axis=1)
    text_descriptions = df['text_description'].tolist()
    print(f"  - 文本描述数量: {len(text_descriptions)}")

    print("\n4. 提取BERT文本嵌入...")
    bert_encoder = BertEncoder(local_model_path=BERT_MODEL_PATH)
    bert_encoder.eval()
    with torch.no_grad():
        bert_embeddings = bert_encoder(text_descriptions)
    bert_embeddings = bert_embeddings.cpu().numpy().astype(np.float32)
    print(f"  - BERT嵌入维度: {bert_embeddings.shape}")

    print("\n5. 提取标签...")
    labels = df['Label'].values.astype(np.int64)
    label_counts = np.unique(labels, return_counts=True)
    print(f"  - 标签分布: {dict(zip(label_counts[0], label_counts[1]))}")

    print("\n6. 数据集划分（训练集/测试集）...")
    X_train, X_test, bert_train, bert_test, y_train, y_test, df_train, df_test = train_test_split(
        X_scaled, bert_embeddings, labels, df, test_size=test_size, random_state=random_state, stratify=labels
    )
    print(f"  - 训练集: {len(X_train)}个样本")
    print(f"  - 测试集: {len(X_test)}个样本")

    print("\n7. 保存模态分离数据（合并为npz格式）...")
    
    np.savez(os.path.join(output_dir, "train.npz"),
             scaled_features=X_train,
             text_embeddings=bert_train,
             labels=y_train)
    print(f"  - train.npz: {X_train.shape[0]}样本（含scaled_features, text_embeddings, labels）")
    
    df_train.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
    print(f"  - train_data.csv: {df_train.shape[0]}行（文本描述备份）")
    
    np.savez(os.path.join(output_dir, "test.npz"),
             scaled_features=X_test,
             text_embeddings=bert_test,
             labels=y_test)
    print(f"  - test.npz: {X_test.shape[0]}样本（含scaled_features, text_embeddings, labels）")
    
    df_test.to_csv(os.path.join(output_dir, "test_data.csv"), index=False)
    print(f"  - test_data.csv: {df_test.shape[0]}行（文本描述备份）")
    
    np.save(os.path.join(output_dir, "train_scaler.npy"), {
        'mean': scaler.mean_,
        'std': scaler.scale_
    })
    print(f"  - train_scaler.npy: 标准化器参数")

    duration_seconds = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("模态分离完成！")
    print(f"保存路径: {os.path.abspath(output_dir)}")
    print(f"耗时: {duration_seconds:.2f}秒")
    print("=" * 60)

    log_data = {
        'dataset_id': dataset_id,
        'split_id': split_id,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'test_size': test_size,
        'random_state': random_state,
        'output_dir': os.path.abspath(output_dir),
        'duration_seconds': duration_seconds
    }
    log_id = save_log('split', log_data)
    print(f"\n数据集划分日志已保存: logs/split/log_{log_id}.json")

    return dataset_id, split_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模态分离脚本，将数据集划分为训练集和测试集")
    parser.add_argument("--dataset_id", type=int, default=0, help="数据集ID（默认0）")
    parser.add_argument("--split_id", type=int, default=None, help="划分ID（默认自动递增）")
    parser.add_argument("--test_size", type=float, default=TEST_SIZE, help="测试集比例（默认0.2）")
    parser.add_argument("--random_state", type=int, default=42, help="随机种子（默认42）")
    args = parser.parse_args()

    success = split_modality(args.dataset_id, args.split_id, args.test_size, args.random_state)
    sys.exit(0 if success else 1)