import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.log_utils import save_log

INPUT_CSV = "./processed_dataset/processed_dataset.csv"
OUTPUT_DIR = "./processed_dataset"

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


def get_next_dataset_id():
    """获取下一个可用的数据集ID（自动递增）"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = os.listdir(OUTPUT_DIR)
    
    max_id = -1
    for f in files:
        if f.startswith("dataset_") and f.endswith(".csv"):
            try:
                idx = int(f.replace("dataset_", "").replace(".csv", ""))
                if idx > max_id:
                    max_id = idx
            except ValueError:
                pass
    
    return max_id + 1


def extract_subset(num_samples, dataset_id=None, random_state=42):
    print("=" * 60)
    print("提取数据集子集（保证正负样本平衡）")
    print("=" * 60)

    try:
        if dataset_id is None:
            dataset_id = get_next_dataset_id()
        print(f"\n数据集ID: {dataset_id}")

        print("\n1. 加载预处理数据...")
        df = pd.read_csv(INPUT_CSV)
        print(f"  原始数据量: {df.shape[0]}行")

        print("\n2. 查看原始标签分布...")
        label_counts = df['Label'].value_counts()
        print(f"  正常流量(0): {label_counts.get(0, 0)}")
        print(f"  恶意流量(1): {label_counts.get(1, 0)}")

        samples_per_class = num_samples // 2
        print(f"\n3. 按类别采样（每类{samples_per_class}个样本）...")

        df_benign = df[df['Label'] == 0]
        df_malicious = df[df['Label'] == 1]

        max_possible = min(len(df_benign), len(df_malicious))
        if samples_per_class > max_possible:
            print(f"  警告: 请求的样本数超过可用数据，每类最多{max_possible}个")
            samples_per_class = max_possible

        df_benign_sample = df_benign.sample(n=samples_per_class, random_state=random_state)
        df_malicious_sample = df_malicious.sample(n=samples_per_class, random_state=random_state)

        df_subset = pd.concat([df_benign_sample, df_malicious_sample]).reset_index(drop=True)
        df_subset = df_subset.sample(frac=1, random_state=random_state).reset_index(drop=True)

        print(f"  采样后数据量: {df_subset.shape[0]}行")
        print(f"  正常流量(0): {len(df_benign_sample)}")
        print(f"  恶意流量(1): {len(df_malicious_sample)}")

        print("\n4. 标准化数值特征...")
        X = df_subset[SELECTED_FEATURES]
        y = df_subset['Label']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        print("\n5. 保存子集数据...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        subset_csv_path = os.path.join(OUTPUT_DIR, f"dataset_{dataset_id}.csv")
        df_subset.to_csv(subset_csv_path, index=False)
        print(f"  - dataset_{dataset_id}.csv: {df_subset.shape[0]}行")

        subset_features_path = os.path.join(OUTPUT_DIR, f"subset_{dataset_id}_scaled_features.npy")
        np.save(subset_features_path, X_scaled.astype(np.float32))
        print(f"  - subset_{dataset_id}_scaled_features.npy: {X_scaled.shape}")

        subset_labels_path = os.path.join(OUTPUT_DIR, f"subset_{dataset_id}_labels.npy")
        np.save(subset_labels_path, y.values.astype(np.int64))
        print(f"  - subset_{dataset_id}_labels.npy: {y.shape}")

        subset_scaler_path = os.path.join(OUTPUT_DIR, f"subset_{dataset_id}_scaler.npy")
        np.save(subset_scaler_path, {
            'mean': scaler.mean_,
            'std': scaler.scale_
        })
        print(f"  - subset_{dataset_id}_scaler.npy: 标准化器参数")

        print("\n" + "=" * 60)
        print("子集提取完成！")
        print(f"  - 数据集ID: {dataset_id}")
        print(f"  - 总样本数: {len(df_subset)}")
        print(f"  - 特征维度: {X_scaled.shape[1]}")
        print(f"  - 保存路径: {os.path.abspath(OUTPUT_DIR)}")
        print("=" * 60)

        log_data = {
            'dataset_id': dataset_id,
            'total_samples': len(df_subset),
            'positive_samples': len(df_malicious_sample),
            'negative_samples': len(df_benign_sample),
            'num_samples_requested': num_samples,
            'random_state': random_state,
            'output_dir': os.path.abspath(OUTPUT_DIR)
        }
        log_id = save_log('subset', log_data)
        print(f"\n子集提取日志已保存: logs/subset/log_{log_id}.json")

        return True, dataset_id

    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取数据集子集，保证正负样本数量相等")
    parser.add_argument("--num_samples", type=int, default=5000, help="子集总样本数（默认5000，每类各占一半）")
    parser.add_argument("--dataset_id", type=int, default=None, help="数据集ID（默认自动递增）")
    parser.add_argument("--random_state", type=int, default=42, help="随机种子（默认42）")
    args = parser.parse_args()

    success, dataset_id = extract_subset(args.num_samples, args.dataset_id, args.random_state)
    sys.exit(0 if success else 1)