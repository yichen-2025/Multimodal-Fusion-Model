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
    15: "unknown",
}


def get_next_openset_split_id(dataset_id):
    """获取下一个可用的开集划分ID（自动递增）"""
    dataset_dir = os.path.join(SPLIT_DATA_DIR, f"dataset_{dataset_id}")
    os.makedirs(dataset_dir, exist_ok=True)

    max_id = -1
    for f in os.listdir(dataset_dir):
        if f.startswith("split_openset_") and os.path.isdir(os.path.join(dataset_dir, f)):
            try:
                idx = int(f.replace("split_openset_", ""))
                if idx > max_id:
                    max_id = idx
            except ValueError:
                pass

    return max_id + 1


def load_split_npz(split_dir, data_type):
    """加载指定划分的npz数据"""
    npz_path = os.path.join(split_dir, f"{data_type}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"未找到文件: {npz_path}")
    return np.load(npz_path, allow_pickle=True)


def load_split_csv(split_dir, data_type):
    """加载指定划分的CSV数据"""
    csv_path = os.path.join(split_dir, f"{data_type}_data.csv")
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


def make_openset_split(dataset_id=1,
                       source_split_id=0,
                       output_split_id=None,
                       unknown_ratio=0.3,
                       random_state=42,
                       hold_out_classes=None):
    """
    构造多分类开集数据集划分

    策略：
    1. 训练集/验证集：保留所有已知类别样本
    2. 测试集：随机将部分已知类样本标记为"未知类"(label=num_known_classes)
       模拟开集场景
    3. 可选 hold_out_classes：指定某些类别完全移出训练/验证，仅出现在测试集中

    Args:
        dataset_id (int): 数据集ID，默认1
        source_split_id (int): 源划分ID，默认0
        output_split_id (int): 输出划分ID，默认自动递增
        unknown_ratio (float): 未知类比例，默认0.3 (30%非BENIGN样本变为未知)
        random_state (int): 随机种子，默认42
        hold_out_classes (list): 要完全留出作为未知的类别ID列表

    Returns:
        tuple: (dataset_id, output_split_id)
    """
    start_time = time.time()

    if output_split_id is None:
        output_split_id = get_next_openset_split_id(dataset_id)

    source_dir = os.path.join(SPLIT_DATA_DIR, f"dataset_{dataset_id}", f"split_{source_split_id}")
    output_dir = os.path.join(SPLIT_DATA_DIR, f"dataset_{dataset_id}", f"split_openset_{output_split_id}")

    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"源划分目录不存在: {source_dir}")

    print("=" * 60)
    print("多分类开集数据集划分构造")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  - 数据集ID: {dataset_id}")
    print(f"  - 源划分ID: {source_split_id}")
    print(f"  - 输出划分ID: {output_split_id}")
    print(f"  - 未知类比例: {unknown_ratio}")
    print(f"  - 随机种子: {random_state}")
    print(f"  - 留出类别: {hold_out_classes}")
    print(f"  - 源目录: {source_dir}")
    print(f"  - 输出目录: {output_dir}")

    # ========== 1. 加载源数据 ==========
    print("\n" + "-" * 40)
    print("1. 加载源数据...")

    train_npz = load_split_npz(source_dir, "train")
    val_npz = load_split_npz(source_dir, "val")
    test_npz = load_split_npz(source_dir, "test")

    train_csv = load_split_csv(source_dir, "train")
    val_csv = load_split_csv(source_dir, "val")
    test_csv = load_split_csv(source_dir, "test")

    train_labels = train_npz['labels']
    val_labels = val_npz['labels']
    test_labels = test_npz['labels']

    all_labels = np.concatenate([train_labels, val_labels, test_labels])
    num_known_classes = int(max(all_labels)) + 1
    print(f"  - 已知类别数: {num_known_classes}")

    print(f"  - 训练集: {len(train_labels)}样本")
    for lbl, cnt in zip(*np.unique(train_labels, return_counts=True)):
        name = LABEL_NAMES.get(int(lbl), str(lbl))
        print(f"    {int(lbl)}: {name} = {cnt}")
    print(f"  - 验证集: {len(val_labels)}样本")
    print(f"  - 测试集: {len(test_labels)}样本")

    # ========== 2. 确定留出类别 ==========
    print("\n" + "-" * 40)
    print("2. 确定留出类别...")

    if hold_out_classes is None:
        hold_out_classes = []

    known_classes = [c for c in range(num_known_classes) if c not in hold_out_classes]
    print(f"  - 已知类别: {known_classes}")
    if hold_out_classes:
        print(f"  - 留出类别: {hold_out_classes}")
        for c in hold_out_classes:
            name = LABEL_NAMES.get(c, str(c))
            print(f"    {c}: {name}")

    # ========== 3. 分离已知/留出样本 ==========
    print("\n" + "-" * 40)
    print(f"3. 标记 {unknown_ratio*100:.0f}% 非BENIGN样本为未知类...")

    rng = np.random.RandomState(random_state)

    # 训练集：移除留出类别，保留已知类别
    train_keep_mask = np.isin(train_labels, known_classes)
    train_keep_idx = np.where(train_keep_mask)[0]

    # 验证集：同训练集处理
    val_keep_mask = np.isin(val_labels, known_classes)
    val_keep_idx = np.where(val_keep_mask)[0]

    # 测试集：已知样本 + 留出样本
    test_known_mask = np.isin(test_labels, known_classes)
    test_known_idx = np.where(test_known_mask)[0]
    test_holdout_mask = np.isin(test_labels, hold_out_classes)
    test_holdout_idx = np.where(test_holdout_mask)[0]

    # 从测试集的已知非BENIGN样本中标记unknown_ratio为未知
    test_benign_mask = test_labels == 0
    test_nonbenign_mask = test_known_mask & ~test_benign_mask
    test_nonbenign_idx = np.where(test_nonbenign_mask)[0]

    n_test_unknown = int(len(test_nonbenign_idx) * unknown_ratio)
    test_unknown_idx = rng.choice(test_nonbenign_idx, size=n_test_unknown, replace=False)
    test_known_keep_idx = np.setdiff1d(test_known_idx, test_unknown_idx)

    # 从训练集的非BENIGN样本中也标记一部分为未知（移至测试集）
    train_nonbenign_idx = np.where((train_labels != 0) & train_keep_mask)[0]
    n_train_unknown = int(len(train_nonbenign_idx) * unknown_ratio)
    train_unknown_idx = rng.choice(train_nonbenign_idx, size=n_train_unknown, replace=False)
    train_known_idx = np.setdiff1d(train_keep_idx, train_unknown_idx)

    # 从验证集的非BENIGN样本中也标记一部分为未知
    val_nonbenign_idx = np.where((val_labels != 0) & val_keep_mask)[0]
    n_val_unknown = int(len(val_nonbenign_idx) * unknown_ratio)
    val_unknown_idx = rng.choice(val_nonbenign_idx, size=n_val_unknown, replace=False)
    val_known_idx = np.setdiff1d(val_keep_idx, val_unknown_idx)

    print(f"  - 训练集: {len(train_known_idx)} known, {len(train_unknown_idx)} unknown(移至测试)")
    print(f"  - 验证集: {len(val_known_idx)} known, {len(val_unknown_idx)} unknown(移至测试)")
    print(f"  - 测试集: {len(test_known_keep_idx)} known, {len(test_unknown_idx)+len(test_holdout_idx)} unknown")

    # ========== 4. 构造新划分 ==========
    print("\n" + "-" * 40)
    print("4. 构造开集划分...")

    new_train_stat = train_npz['scaled_features'][train_known_idx]
    new_train_bert = train_npz['text_embeddings'][train_known_idx]
    new_train_labels = train_labels[train_known_idx].copy()

    new_val_stat = val_npz['scaled_features'][val_known_idx]
    new_val_bert = val_npz['text_embeddings'][val_known_idx]
    new_val_labels = val_labels[val_known_idx].copy()

    unknown_label = num_known_classes

    test_known_stat = test_npz['scaled_features'][test_known_keep_idx]
    test_known_bert = test_npz['text_embeddings'][test_known_keep_idx]
    test_known_labels = test_labels[test_known_keep_idx].copy()

    test_unknown_stat = test_npz['scaled_features'][test_unknown_idx]
    test_unknown_bert = test_npz['text_embeddings'][test_unknown_idx]
    test_unknown_labels = np.full(len(test_unknown_idx), unknown_label, dtype=np.int64)

    train_unknown_stat = train_npz['scaled_features'][train_unknown_idx]
    train_unknown_bert = train_npz['text_embeddings'][train_unknown_idx]
    train_unknown_labels = np.full(len(train_unknown_idx), unknown_label, dtype=np.int64)

    val_unknown_stat = val_npz['scaled_features'][val_unknown_idx]
    val_unknown_bert = val_npz['text_embeddings'][val_unknown_idx]
    val_unknown_labels = np.full(len(val_unknown_idx), unknown_label, dtype=np.int64)

    holdout_stat = test_npz['scaled_features'][test_holdout_idx]
    holdout_bert = test_npz['text_embeddings'][test_holdout_idx]
    holdout_labels = np.full(len(test_holdout_idx), unknown_label, dtype=np.int64)

    new_test_stat = np.concatenate([
        test_known_stat,
        test_unknown_stat,
        train_unknown_stat,
        val_unknown_stat,
        holdout_stat
    ])
    new_test_bert = np.concatenate([
        test_known_bert,
        test_unknown_bert,
        train_unknown_bert,
        val_unknown_bert,
        holdout_bert
    ])
    new_test_labels = np.concatenate([
        test_known_labels,
        test_unknown_labels,
        train_unknown_labels,
        val_unknown_labels,
        holdout_labels
    ])

    train_counts = np.unique(new_train_labels, return_counts=True)
    val_counts = np.unique(new_val_labels, return_counts=True)
    test_counts = np.unique(new_test_labels, return_counts=True)

    print(f"\n  新训练集: {len(new_train_labels)}样本")
    for lbl, cnt in zip(*train_counts):
        name = LABEL_NAMES.get(int(lbl), str(lbl))
        print(f"    {int(lbl)}: {name} = {cnt}")

    print(f"  新验证集: {len(new_val_labels)}样本")
    for lbl, cnt in zip(*val_counts):
        name = LABEL_NAMES.get(int(lbl), str(lbl))
        print(f"    {int(lbl)}: {name} = {cnt}")

    print(f"  新测试集: {len(new_test_labels)}样本")
    for lbl, cnt in zip(*test_counts):
        name = LABEL_NAMES.get(int(lbl), str(lbl))
        print(f"    {int(lbl)}: {name} = {cnt}")

    # ========== 5. 构造CSV文本描述 ==========
    print("\n" + "-" * 40)
    print("5. 构造CSV文本描述...")

    def gather_csv_rows(csv_df, idx_list, new_labels):
        if csv_df is None:
            return None
        rows = csv_df.iloc[idx_list].copy()
        rows['Label'] = new_labels
        return rows

    new_train_csv = gather_csv_rows(train_csv, train_known_idx, new_train_labels)
    new_val_csv = gather_csv_rows(val_csv, val_known_idx, new_val_labels)

    if test_csv is not None:
        test_csv_known = test_csv.iloc[test_known_keep_idx].copy()
        test_csv_known['Label'] = test_labels[test_known_keep_idx]

        test_csv_unknown = test_csv.iloc[test_unknown_idx].copy()
        test_csv_unknown['Label'] = unknown_label
        if 'text_description' in test_csv_unknown.columns:
            test_csv_unknown['text_description'] = test_csv_unknown['text_description'].apply(
                lambda x: f"[未知攻击] {x}" if pd.notna(x) else "[未知攻击]"
            )

        train_csv_unknown = train_csv.iloc[train_unknown_idx].copy()
        train_csv_unknown['Label'] = unknown_label
        if 'text_description' in train_csv_unknown.columns:
            train_csv_unknown['text_description'] = train_csv_unknown['text_description'].apply(
                lambda x: f"[未知攻击] {x}" if pd.notna(x) else "[未知攻击]"
            )

        val_csv_unknown = val_csv.iloc[val_unknown_idx].copy()
        val_csv_unknown['Label'] = unknown_label
        if 'text_description' in val_csv_unknown.columns:
            val_csv_unknown['text_description'] = val_csv_unknown['text_description'].apply(
                lambda x: f"[未知攻击] {x}" if pd.notna(x) else "[未知攻击]"
            )

        holdout_csv = test_csv.iloc[test_holdout_idx].copy()
        holdout_csv['Label'] = unknown_label
        if 'text_description' in holdout_csv.columns:
            holdout_csv['text_description'] = holdout_csv['text_description'].apply(
                lambda x: f"[未知攻击] {x}" if pd.notna(x) else "[未知攻击]"
            )

        new_test_csv = pd.concat([test_csv_known, test_csv_unknown,
                                   train_csv_unknown, val_csv_unknown,
                                   holdout_csv], ignore_index=True)
    else:
        new_test_csv = None

    # ========== 6. 保存 ==========
    print("\n" + "-" * 40)
    print("6. 保存开集划分数据...")

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
    if new_val_csv is not None:
        new_val_csv.to_csv(os.path.join(output_dir, "val_data.csv"), index=False)
    if new_test_csv is not None:
        new_test_csv.to_csv(os.path.join(output_dir, "test_data.csv"), index=False)

    scaler_path = os.path.join(source_dir, "train_scaler.npy")
    if os.path.exists(scaler_path):
        import shutil
        shutil.copy2(scaler_path, os.path.join(output_dir, "train_scaler.npy"))
        print(f"  - train_scaler.npy: 已复制")

    mapping_src = os.path.join(source_dir, "label_mapping.json")
    if os.path.exists(mapping_src):
        import shutil
        shutil.copy2(mapping_src, os.path.join(output_dir, "label_mapping.json"))
        print(f"  - label_mapping.json: 已复制")

    duration_seconds = time.time() - start_time

    # ========== 7. 日志记录 ==========
    print("\n" + "-" * 40)
    print("7. 记录日志...")

    log_data = {
        'dataset_id': dataset_id,
        'source_split_id': source_split_id,
        'output_split_id': output_split_id,
        'unknown_ratio': unknown_ratio,
        'random_state': random_state,
        'num_known_classes': num_known_classes,
        'hold_out_classes': hold_out_classes,
        'unknown_label': unknown_label,
        'train_total': len(new_train_labels),
        'val_total': len(new_val_labels),
        'test_total': len(new_test_labels),
        'test_known': int(np.sum(new_test_labels < unknown_label)),
        'test_unknown': int(np.sum(new_test_labels == unknown_label)),
        'output_dir': os.path.abspath(output_dir),
        'duration_seconds': round(duration_seconds, 3)
    }
    log_id = save_log('openset_split', log_data)
    print(f"  - 日志已保存: logs/openset_split/log_{log_id}.json")

    print("\n" + "=" * 60)
    print("开集数据集划分构造完成！")
    print(f"  - 输出目录: {os.path.abspath(output_dir)}")
    print(f"  - 耗时: {duration_seconds:.2f}秒")
    print("=" * 60)

    return dataset_id, output_split_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="构造多分类开集数据集划分 — 将部分非BENIGN样本标记为未知类，仅出现在测试集中"
    )
    parser.add_argument("--dataset_id", type=int, default=1, help="数据集ID（默认1）")
    parser.add_argument("--source_split_id", type=int, default=0, help="源划分ID（默认0）")
    parser.add_argument("--output_split_id", type=int, default=None, help="输出划分ID（默认自动递增）")
    parser.add_argument("--unknown_ratio", type=float, default=0.3, help="未知类比例（默认0.3）")
    parser.add_argument("--random_state", type=int, default=42, help="随机种子（默认42）")
    parser.add_argument("--hold_out_classes", type=int, nargs='*', default=None,
                        help="完全留出的类别ID列表（如 --hold_out_classes 13 14）")
    args = parser.parse_args()

    dataset_id, output_split_id = make_openset_split(
        dataset_id=args.dataset_id,
        source_split_id=args.source_split_id,
        output_split_id=args.output_split_id,
        unknown_ratio=args.unknown_ratio,
        random_state=args.random_state,
        hold_out_classes=args.hold_out_classes
    )
    sys.exit(0)