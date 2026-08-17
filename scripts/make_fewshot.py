import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
from utils.log_utils import save_log

SPLIT_DATA_DIR = os.path.join(PROJECT_ROOT, "split_data")


def get_next_fewshot_id(dataset_id, k_per_class):
    dataset_dir = os.path.join(SPLIT_DATA_DIR, f"dataset_{dataset_id}")
    os.makedirs(dataset_dir, exist_ok=True)

    prefix = f"split_fewshot_{k_per_class}_"
    max_id = -1
    for f in os.listdir(dataset_dir):
        if f.startswith(prefix) and os.path.isdir(os.path.join(dataset_dir, f)):
            try:
                idx = int(f.replace(prefix, ""))
                if idx > max_id:
                    max_id = idx
            except ValueError:
                pass

    return max_id + 1


def load_npz(split_dir, data_type):
    npz_path = os.path.join(split_dir, f"{data_type}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"未找到文件: {npz_path}")
    return np.load(npz_path, allow_pickle=True)


def load_csv(split_dir, data_type):
    csv_path = os.path.join(split_dir, f"{data_type}_data.csv")
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


def make_fewshot_split(dataset_id=1,
                        source_split_id=0,
                        k_per_class=5,
                        output_split_id=None,
                        include_remaining_as_test=True,
                        random_state=42):
    """
    从开集划分构造少样本数据集

    策略：
    1. 从openset训练集（仅含已知类）中，每类抽取k个样本作为fewshot训练集
    2. 剩余的已知训练样本 → 可选择加入测试集（避免浪费数据）
    3. 验证集保持不变
    4. 测试集：原始测试集（含unknown）+ 可选加入剩余训练样本

    Args:
        dataset_id (int): 数据集ID
        source_split_id (int): 源openset划分ID
        k_per_class (int): 每已知类的少样本数量
        output_split_id (int): 输出划分ID，默认自动递增
        include_remaining_as_test (bool): 是否将剩余训练样本加入测试集
        random_state (int): 随机种子

    Returns:
        tuple: (dataset_id, k_per_class, output_split_id)
    """
    start_time = time.time()
    rng = np.random.RandomState(random_state)

    if output_split_id is None:
        output_split_id = get_next_fewshot_id(dataset_id, k_per_class)

    source_dir = os.path.join(SPLIT_DATA_DIR, f"dataset_{dataset_id}", f"split_openset_{source_split_id}")
    output_dir = os.path.join(SPLIT_DATA_DIR, f"dataset_{dataset_id}",
                               f"split_fewshot_{k_per_class}_{output_split_id}")

    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"源openset划分目录不存在: {source_dir}")

    print("=" * 60)
    print("少样本数据集构造")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  - 数据集ID: {dataset_id}")
    print(f"  - 源openset划分ID: {source_split_id}")
    print(f"  - 输出划分ID: {output_split_id}")
    print(f"  - 每类样本数k: {k_per_class}")
    print(f"  - 随机种子: {random_state}")
    print(f"  - 源目录: {source_dir}")
    print(f"  - 输出目录: {output_dir}")

    # ========== 1. 加载源数据 ==========
    print("\n" + "-" * 40)
    print("1. 加载openset源数据...")

    train_npz = load_npz(source_dir, "train")
    val_npz = load_npz(source_dir, "val")
    test_npz = load_npz(source_dir, "test")

    train_csv = load_csv(source_dir, "train")
    val_csv = load_csv(source_dir, "val")
    test_csv = load_csv(source_dir, "test")

    train_labels = train_npz['labels']
    val_labels = val_npz['labels']
    test_labels = test_npz['labels']

    num_known_classes = len(np.unique(train_labels))

    print(f"  - 训练集: {len(train_labels)}样本")
    for c in range(num_known_classes):
        cnt = int(np.sum(train_labels == c))
        print(f"    类{c}: {cnt}样本")
    print(f"  - 验证集: {len(val_labels)}样本")
    print(f"  - 测试集: {len(test_labels)}样本")
    for lbl, cnt in zip(*np.unique(test_labels, return_counts=True)):
        print(f"    类{lbl}: {cnt}样本")

    # ========== 2. 少样本采样 ==========
    print("\n" + "-" * 40)
    print(f"2. 每类抽取 {k_per_class} 个样本...")

    train_keep_idx = []
    train_remaining_idx = []

    for c in range(num_known_classes):
        class_idx = np.where(train_labels == c)[0]
        if len(class_idx) < k_per_class:
            print(f"  警告: 类{c}只有{len(class_idx)}样本，少于k={k_per_class}")
            keep = class_idx
        else:
            keep = rng.choice(class_idx, size=k_per_class, replace=False)
        rem = np.setdiff1d(class_idx, keep)

        train_keep_idx.extend(keep.tolist())
        train_remaining_idx.extend(rem.tolist())

    train_keep_idx = np.array(train_keep_idx, dtype=int)
    train_remaining_idx = np.array(train_remaining_idx, dtype=int)

    print(f"  - fewshot训练集: {len(train_keep_idx)}样本 (每类{k_per_class})")
    print(f"  - 剩余训练样本: {len(train_remaining_idx)}样本")

    # ========== 3. 构造新划分 ==========
    print("\n" + "-" * 40)
    print("3. 构造少样本划分...")

    # 新训练集：fewshot样本
    new_train_stat = train_npz['scaled_features'][train_keep_idx]
    new_train_bert = train_npz['text_embeddings'][train_keep_idx]
    new_train_labels = train_labels[train_keep_idx].copy()

    # 验证集：保持不变
    new_val_stat = val_npz['scaled_features']
    new_val_bert = val_npz['text_embeddings']
    new_val_labels = val_labels.copy()

    # 测试集：原始测试 + 可选剩余训练样本
    if include_remaining_as_test and len(train_remaining_idx) > 0:
        new_test_stat = np.concatenate([
            test_npz['scaled_features'],
            train_npz['scaled_features'][train_remaining_idx]
        ])
        new_test_bert = np.concatenate([
            test_npz['text_embeddings'],
            train_npz['text_embeddings'][train_remaining_idx]
        ])
        new_test_labels = np.concatenate([
            test_labels,
            train_labels[train_remaining_idx]
        ])
    else:
        new_test_stat = test_npz['scaled_features']
        new_test_bert = test_npz['text_embeddings']
        new_test_labels = test_labels.copy()

    # 打印新划分统计
    print(f"\n  新训练集: {len(new_train_labels)}样本")
    for c in range(num_known_classes):
        cnt = int(np.sum(new_train_labels == c))
        print(f"    类{c}: {cnt}样本")

    print(f"  新验证集: {len(new_val_labels)}样本")
    for c in range(num_known_classes):
        cnt = int(np.sum(new_val_labels == c))
        print(f"    类{c}: {cnt}样本")

    print(f"  新测试集: {len(new_test_labels)}样本")
    for lbl, cnt in zip(*np.unique(new_test_labels, return_counts=True)):
        name = {0: "BENIGN", 1: "known_DDoS", 2: "unknown_DDoS"}.get(lbl, str(lbl))
        print(f"    {name}({lbl}): {cnt}样本")

    # ========== 4. 构造CSV ==========
    print("\n" + "-" * 40)
    print("4. 构造CSV文本描述...")

    def gather_csv_rows(csv_df, idx_list, labels):
        if csv_df is None:
            return None
        rows = csv_df.iloc[idx_list].copy()
        rows['Label'] = labels
        return rows

    new_train_csv = gather_csv_rows(train_csv, train_keep_idx, new_train_labels)
    new_val_csv = gather_csv_rows(val_csv, np.arange(len(val_labels)), new_val_labels)

    # 测试集CSV需要拼接
    if include_remaining_as_test and len(train_remaining_idx) > 0 and test_csv is not None:
        test_csv_part = test_csv.copy()
        test_csv_part['Label'] = test_labels

        remaining_csv_part = train_csv.iloc[train_remaining_idx].copy()
        remaining_csv_part['Label'] = train_labels[train_remaining_idx]

        new_test_csv = pd.concat([test_csv_part, remaining_csv_part], ignore_index=True)
    elif test_csv is not None:
        new_test_csv = test_csv.copy()
        new_test_csv['Label'] = test_labels
    else:
        new_test_csv = None

    # ========== 5. 保存 ==========
    print("\n" + "-" * 40)
    print("5. 保存少样本划分数据...")

    os.makedirs(output_dir, exist_ok=True)

    np.savez(os.path.join(output_dir, "train.npz"),
             scaled_features=new_train_stat,
             text_embeddings=new_train_bert,
             labels=new_train_labels)
    print(f"  - train.npz: {len(new_train_labels)}样本")

    np.savez(os.path.join(output_dir, "val.npz"),
             scaled_features=new_val_stat,
             text_embeddings=new_val_bert,
             labels=new_val_labels)
    print(f"  - val.npz: {len(new_val_labels)}样本")

    np.savez(os.path.join(output_dir, "test.npz"),
             scaled_features=new_test_stat,
             text_embeddings=new_test_bert,
             labels=new_test_labels)
    print(f"  - test.npz: {len(new_test_labels)}样本")

    if new_train_csv is not None:
        new_train_csv.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
        print(f"  - train_data.csv: {len(new_train_csv)}行")
    if new_val_csv is not None:
        new_val_csv.to_csv(os.path.join(output_dir, "val_data.csv"), index=False)
        print(f"  - val_data.csv: {len(new_val_csv)}行")
    if new_test_csv is not None:
        new_test_csv.to_csv(os.path.join(output_dir, "test_data.csv"), index=False)
        print(f"  - test_data.csv: {len(new_test_csv)}行")

    # 复制scaler
    scaler_path = os.path.join(source_dir, "train_scaler.npy")
    if os.path.exists(scaler_path):
        import shutil
        shutil.copy2(scaler_path, os.path.join(output_dir, "train_scaler.npy"))
        print(f"  - train_scaler.npy: 已复制")

    duration_seconds = time.time() - start_time

    # ========== 6. 日志 ==========
    print("\n" + "-" * 40)
    print("6. 记录日志...")

    log_data = {
        'dataset_id': dataset_id,
        'source_split_id': source_split_id,
        'output_split_id': output_split_id,
        'k_per_class': k_per_class,
        'include_remaining': include_remaining_as_test,
        'random_state': random_state,
        'num_known_classes': num_known_classes,
        'train_total': len(new_train_labels),
        'train_per_class': {int(c): int(np.sum(new_train_labels == c)) for c in range(num_known_classes)},
        'val_total': len(new_val_labels),
        'test_total': len(new_test_labels),
        'test_per_class': {int(lbl): int(cnt) for lbl, cnt in zip(*np.unique(new_test_labels, return_counts=True))},
        'output_dir': os.path.abspath(output_dir),
        'duration_seconds': round(duration_seconds, 3)
    }

    log_id = save_log('fewshot_split', log_data)
    print(f"  - 日志已保存: logs/fewshot_split/log_{log_id}.json")

    print("\n" + "=" * 60)
    print(f"少样本数据集构造完成！")
    print(f"  - 输出目录: {os.path.abspath(output_dir)}")
    print(f"  - 耗时: {duration_seconds:.2f}秒")
    print("=" * 60)

    return dataset_id, k_per_class, output_split_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="构造少样本数据集 — 从开集划分中每已知类抽取k个样本作训练集"
    )
    parser.add_argument("--dataset_id", type=int, default=1,
                        help="数据集ID（默认1）")
    parser.add_argument("--source_split_id", type=int, default=0,
                        help="源openset划分ID（默认0）")
    parser.add_argument("--k_values", type=str, default="5,10,20",
                        help="k值列表，逗号分隔（默认5,10,20）")
    parser.add_argument("--output_split_id", type=int, default=None,
                        help="输出划分ID（默认自动递增，仅单个k时生效）")
    parser.add_argument("--no_remaining", action="store_true",
                        help="不将剩余训练样本加入测试集")
    parser.add_argument("--random_state", type=int, default=42,
                        help="随机种子（默认42）")
    args = parser.parse_args()

    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    for k in k_values:
        print(f"\n{'#'*60}")
        print(f"# 处理 k={k}")
        print(f"{'#'*60}")
        make_fewshot_split(
            dataset_id=args.dataset_id,
            source_split_id=args.source_split_id,
            k_per_class=k,
            output_split_id=args.output_split_id if len(k_values) == 1 else None,
            include_remaining_as_test=not args.no_remaining,
            random_state=args.random_state
        )
        print()

    print("\n全部k值处理完成！")