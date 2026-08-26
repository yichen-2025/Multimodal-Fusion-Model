import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, PROJECT_ROOT)
from utils.log_utils import save_log

INPUT_CSV = os.path.join(PROJECT_ROOT, "processed_dataset", "processed_dataset.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "processed_dataset")

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

LABEL_NAMES = {
    0: "BENIGN",
    1: "DoS Hulk",
    2: "DoS GoldenEye",
    3: "DoS slowloris",
    4: "DoS Slowhttptest",
    5: "DDoS",
    6: "PortScan",
    7: "FTP-Patator",
    8: "SSH-Patator",
    9: "Bot",
    10: "Web Attack - Brute Force",
    11: "Web Attack - XSS",
    12: "Web Attack - Sql Injection",
    13: "Infiltration",
    14: "Heartbleed",
}


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


def compute_class_counts(label_counts, total_samples, mode="stratified", per_class=None):
    """
    计算每个类别应采样的数量

    Args:
        label_counts: Series, 每个类别的样本数
        total_samples: int, 目标总样本数
        mode: str, 采样模式
            - "stratified": 按原始分布比例采样
            - "balanced": 每类采样相同数量（受稀有类别限制）
            - "custom": 按per_class指定数量采样
        per_class: dict, 每个类别的采样数量（仅custom模式使用）

    Returns:
        dict: {label_id: sample_count}
    """
    classes = sorted(label_counts.index.tolist())
    total_available = label_counts.sum()

    if mode == "stratified":
        ratios = label_counts / total_available
        raw_counts = (ratios * total_samples).apply(int)
        remainder = total_samples - raw_counts.sum()

        if remainder > 0:
            fractional_parts = (ratios * total_samples) - raw_counts
            extra_indices = fractional_parts.nlargest(remainder).index
            raw_counts[extra_indices] += 1

        return {c: max(1, raw_counts.get(c, 0)) for c in classes}

    elif mode == "balanced":
        if total_samples is not None and total_samples > 0:
            default_per_class = max(1, total_samples // len(classes))
        else:
            default_per_class = 100

        counts = {}
        for c in classes:
            available = label_counts.get(c, 0)
            counts[c] = min(default_per_class, available) if available > 0 else 0
        return counts

    elif mode == "custom":
        if per_class is None:
            raise ValueError("custom模式需要提供per_class参数")
        counts = {}
        for c in classes:
            available = label_counts.get(c, 0)
            requested = per_class.get(c, 0)
            counts[c] = min(requested, available)
        return counts

    else:
        raise ValueError(f"未知采样模式: {mode}")


def extract_subset(total_samples=None, ratio=None, mode="stratified",
                   per_class=None, dataset_id=None, random_state=42):
    """
    从processed_dataset.csv中提取子集，支持多分类分层采样

    Args:
        total_samples: 目标总样本数（与ratio二选一）
        ratio: 保留比例（如0.1表示保留10%数据）
        mode: 采样模式
            - "stratified": 按原始类别分布比例采样（推荐）
            - "balanced": 每类采样相同数量
            - "custom": 按per_class字典指定数量
        per_class: 每个类别的采样数量，如{0: 5000, 1: 3000}
        dataset_id: 输出数据集ID，None则自动递增
        random_state: 随机种子
    """
    print("=" * 60)
    print("多分类数据集子集提取工具")
    print("=" * 60)

    try:
        if not os.path.exists(INPUT_CSV):
            raise FileNotFoundError(f"输入文件不存在: {INPUT_CSV}")

        if total_samples is None and ratio is None:
            total_samples = 10000

        if ratio is not None:
            if not (0 < ratio <= 1):
                raise ValueError("ratio必须在(0, 1]范围内")
            print(f"\n1. 加载预处理数据（采样比例: {ratio*100:.1f}%）...")
            df = pd.read_csv(INPUT_CSV)
            total_samples = max(1, int(len(df) * ratio))
        else:
            print(f"\n1. 加载预处理数据（目标样本数: {total_samples}）...")
            df = pd.read_csv(INPUT_CSV)

        df.columns = df.columns.str.strip()
        print(f"  原始数据量: {df.shape[0]}行")

        print("\n2. 原始标签分布...")
        label_counts = df['Label'].value_counts().sort_index()
        for lbl, cnt in label_counts.items():
            name = LABEL_NAMES.get(int(lbl), str(lbl))
            pct = cnt / len(df) * 100
            print(f"  {int(lbl):2d}: {name:<30s} = {cnt:>8d} ({pct:5.1f}%)")

        if total_samples >= len(df):
            print(f"\n  请求的样本数({total_samples}) >= 总样本数({len(df)})，直接复制全部数据")
            df_subset = df.copy()
        else:
            print(f"\n3. 计算各类别采样数量（模式: {mode}）...")

            class_counts = compute_class_counts(label_counts, total_samples, mode, per_class)

            actual_total = sum(class_counts.values())
            print(f"  实际采样总数: {actual_total}")
            for lbl, cnt in sorted(class_counts.items()):
                name = LABEL_NAMES.get(int(lbl), str(lbl))
                avail = label_counts.get(lbl, 0)
                note = " [全部保留]" if cnt >= avail else ""
                print(f"  {int(lbl):2d}: {name:<30s} 采样 {cnt:>6d} / 可用 {avail:>8d}{note}")

            print(f"\n4. 执行分层采样...")
            sampled_dfs = []
            for lbl, cnt in class_counts.items():
                if cnt <= 0:
                    continue
                df_class = df[df['Label'] == lbl]
                n_sample = min(cnt, len(df_class))
                sampled = df_class.sample(n=n_sample, random_state=random_state)
                sampled_dfs.append(sampled)

            df_subset = pd.concat(sampled_dfs, ignore_index=True)
            df_subset = df_subset.sample(frac=1, random_state=random_state).reset_index(drop=True)

        print(f"\n5. 采样后标签分布...")
        subset_counts = df_subset['Label'].value_counts().sort_index()
        for lbl, cnt in subset_counts.items():
            name = LABEL_NAMES.get(int(lbl), str(lbl))
            pct = cnt / len(df_subset) * 100
            print(f"  {int(lbl):2d}: {name:<30s} = {cnt:>6d} ({pct:5.1f}%)")
        print(f"  总计: {len(df_subset)}行")

        print("\n6. 标准化数值特征...")
        X = df_subset[SELECTED_FEATURES]
        y = df_subset['Label']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"  特征矩阵: {X_scaled.shape}")

        print("\n7. 保存子集数据...")
        if dataset_id is None:
            dataset_id = get_next_dataset_id()

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        subset_csv_path = os.path.join(OUTPUT_DIR, f"dataset_{dataset_id}.csv")
        df_subset.to_csv(subset_csv_path, index=False)
        print(f"  - dataset_{dataset_id}.csv: {df_subset.shape[0]}行 (可直接用于split_modality.py)")

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
        print(f"  - 类别数: {len(np.unique(y))}")
        print(f"  - 保存路径: {os.path.abspath(OUTPUT_DIR)}")
        print(f"\n  下一步: 使用 split_modality.py --input_csv dataset_{dataset_id}.csv 进行模态分离")
        print("=" * 60)

        log_data = {
            'dataset_id': dataset_id,
            'total_samples': len(df_subset),
            'num_classes': len(np.unique(y)),
            'mode': mode,
            'original_total': len(df),
            'sampling_ratio': round(len(df_subset) / len(df), 4),
            'random_state': random_state,
            'class_distribution': {int(k): int(v) for k, v in subset_counts.items()},
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
    parser = argparse.ArgumentParser(
        description="多分类数据集子集提取 — 支持分层采样、平衡采样和自定义采样"
    )
    parser.add_argument("--total_samples", type=int, default=None,
                        help="目标总样本数（如10000），与--ratio二选一")
    parser.add_argument("--ratio", type=float, default=None,
                        help="保留比例（如0.1表示10%%），与--total_samples二选一")
    parser.add_argument("--mode", type=str, default="stratified",
                        choices=["stratified", "balanced", "custom"],
                        help="采样模式: stratified=按原始比例, balanced=每类均衡, custom=自定义数量")
    parser.add_argument("--per_class", type=str, default=None,
                        help="custom模式下每类采样数，格式: '0:5000,1:3000,2:500,...'")
    parser.add_argument("--dataset_id", type=int, default=None,
                        help="输出数据集ID（默认自动递增）")
    parser.add_argument("--random_state", type=int, default=42,
                        help="随机种子（默认42）")
    args = parser.parse_args()

    per_class_dict = None
    if args.per_class:
        per_class_dict = {}
        for pair in args.per_class.split(","):
            lbl, cnt = pair.strip().split(":")
            per_class_dict[int(lbl)] = int(cnt)

    if args.mode == "custom" and per_class_dict is None:
        print("错误: custom模式必须通过 --per_class 指定各类别采样数")
        sys.exit(1)

    success, dataset_id = extract_subset(
        total_samples=args.total_samples,
        ratio=args.ratio,
        mode=args.mode,
        per_class=per_class_dict,
        dataset_id=args.dataset_id,
        random_state=args.random_state
    )
    sys.exit(0 if success else 1)