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
                       random_state=42):
    """
    构造开集数据集划分

    功能：从封闭集划分出发，将一部分DDoS样本标记为"未知类"(label=2)，
          这些未知样本仅出现在测试集中，训练/验证集不含未知类。

    Args:
        dataset_id (int): 数据集ID，默认1
        source_split_id (int): 源划分ID，默认0
        output_split_id (int): 输出划分ID，默认自动递增
        unknown_ratio (float): 未知类比例，默认0.3 (30% DDoS变为未知)
        random_state (int): 随机种子，默认42

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
    print("开集数据集划分构造")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  - 数据集ID: {dataset_id}")
    print(f"  - 源划分ID: {source_split_id}")
    print(f"  - 输出划分ID: {output_split_id}")
    print(f"  - 未知类比例: {unknown_ratio}")
    print(f"  - 随机种子: {random_state}")
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

    print(f"  - 训练集: {len(train_labels)}样本 (benign={np.sum(train_labels==0)}, DDoS={np.sum(train_labels==1)})")
    print(f"  - 验证集: {len(val_labels)}样本 (benign={np.sum(val_labels==0)}, DDoS={np.sum(val_labels==1)})")
    print(f"  - 测试集: {len(test_labels)}样本 (benign={np.sum(test_labels==0)}, DDoS={np.sum(test_labels==1)})")

    # ========== 2. 分离 benign 和 DDoS ==========
    print("\n" + "-" * 40)
    print("2. 分离 benign 与 DDoS 样本...")

    rng = np.random.RandomState(random_state)

    # 训练集
    train_benign_mask = train_labels == 0
    train_ddos_mask = train_labels == 1
    train_benign_idx = np.where(train_benign_mask)[0]
    train_ddos_idx = np.where(train_ddos_mask)[0]

    # 验证集
    val_benign_mask = val_labels == 0
    val_ddos_mask = val_labels == 1
    val_benign_idx = np.where(val_benign_mask)[0]
    val_ddos_idx = np.where(val_ddos_mask)[0]

    # 测试集
    test_benign_mask = test_labels == 0
    test_ddos_mask = test_labels == 1
    test_benign_idx = np.where(test_benign_mask)[0]
    test_ddos_idx = np.where(test_ddos_mask)[0]

    # ========== 3. 标记未知类 ==========
    print("\n" + "-" * 40)
    print(f"3. 标记 {unknown_ratio*100:.0f}% DDoS 为未知类...")

    # 训练集: 从DDoS中选出unknown_ratio标记为未知（将被移到测试集）
    n_train_unknown = int(len(train_ddos_idx) * unknown_ratio)
    train_unknown_idx = rng.choice(train_ddos_idx, size=n_train_unknown, replace=False)
    train_known_idx = np.setdiff1d(train_ddos_idx, train_unknown_idx)

    # 验证集: 同训练集处理
    n_val_unknown = int(len(val_ddos_idx) * unknown_ratio)
    val_unknown_idx = rng.choice(val_ddos_idx, size=n_val_unknown, replace=False)
    val_known_idx = np.setdiff1d(val_ddos_idx, val_unknown_idx)

    # 测试集: 从DDoS中选出unknown_ratio标记为未知（直接改标签）
    n_test_unknown = int(len(test_ddos_idx) * unknown_ratio)
    test_unknown_idx = rng.choice(test_ddos_idx, size=n_test_unknown, replace=False)
    test_known_idx = np.setdiff1d(test_ddos_idx, test_unknown_idx)

    print(f"  - 训练集 DDoS: {len(train_known_idx)} known, {len(train_unknown_idx)} unknown(移至测试集)")
    print(f"  - 验证集 DDoS: {len(val_known_idx)} known, {len(val_unknown_idx)} unknown(移至测试集)")
    print(f"  - 测试集 DDoS: {len(test_known_idx)} known, {len(test_unknown_idx)} unknown(标签→2)")

    # ========== 4. 构造新划分 ==========
    print("\n" + "-" * 40)
    print("4. 构造开集划分...")

    # 新训练集: benign + known DDoS (不含未知)
    new_train_idx = np.concatenate([train_benign_idx, train_known_idx])
    new_train_stat = train_npz['scaled_features'][new_train_idx]
    new_train_bert = train_npz['text_embeddings'][new_train_idx]
    new_train_labels = train_labels[new_train_idx].copy()

    # 新验证集: benign + known DDoS (不含未知)
    new_val_idx = np.concatenate([val_benign_idx, val_known_idx])
    new_val_stat = val_npz['scaled_features'][new_val_idx]
    new_val_bert = val_npz['text_embeddings'][new_val_idx]
    new_val_labels = val_labels[new_val_idx].copy()

    # 新测试集: benign + known DDoS(标签=1) + unknown DDoS(标签=2, 来自三处)
    # 先收集所有unknown样本的索引和来源
    # test集中的unknown直接在test_npz中取
    test_unknown_ddos_stat = test_npz['scaled_features'][test_unknown_idx]
    test_unknown_ddos_bert = test_npz['text_embeddings'][test_unknown_idx]
    test_unknown_ddos_labels = np.full(len(test_unknown_idx), 2, dtype=np.int64)

    # train中标记为unknown的从train_npz取
    train_unknown_ddos_stat = train_npz['scaled_features'][train_unknown_idx]
    train_unknown_ddos_bert = train_npz['text_embeddings'][train_unknown_idx]
    train_unknown_ddos_labels = np.full(len(train_unknown_idx), 2, dtype=np.int64)

    # val中标记为unknown的从val_npz取
    val_unknown_ddos_stat = val_npz['scaled_features'][val_unknown_idx]
    val_unknown_ddos_bert = val_npz['text_embeddings'][val_unknown_idx]
    val_unknown_ddos_labels = np.full(len(val_unknown_idx), 2, dtype=np.int64)

    # 合并测试集
    new_test_stat = np.concatenate([
        test_npz['scaled_features'][test_benign_idx],       # benign
        test_npz['scaled_features'][test_known_idx],        # known DDoS
        test_unknown_ddos_stat,                               # test unknown
        train_unknown_ddos_stat,                             # train→unknown
        val_unknown_ddos_stat                                # val→unknown
    ])
    new_test_bert = np.concatenate([
        test_npz['text_embeddings'][test_benign_idx],
        test_npz['text_embeddings'][test_known_idx],
        test_unknown_ddos_bert,
        train_unknown_ddos_bert,
        val_unknown_ddos_bert
    ])
    new_test_labels = np.concatenate([
        test_labels[test_benign_idx],                         # 0: benign
        test_labels[test_known_idx],                          # 1: known DDoS
        test_unknown_ddos_labels,                             # 2: unknown
        train_unknown_ddos_labels,                            # 2: unknown
        val_unknown_ddos_labels                               # 2: unknown
    ])

    # 打印新划分统计
    train_counts = np.unique(new_train_labels, return_counts=True)
    val_counts = np.unique(new_val_labels, return_counts=True)
    test_counts = np.unique(new_test_labels, return_counts=True)

    print(f"\n  新训练集: {len(new_train_labels)}样本")
    print(f"    - benign(0): {train_counts[1][train_counts[0]==0][0] if 0 in train_counts[0] else 0}")
    print(f"    - known DDoS(1): {train_counts[1][train_counts[0]==1][0] if 1 in train_counts[0] else 0}")

    print(f"  新验证集: {len(new_val_labels)}样本")
    print(f"    - benign(0): {val_counts[1][val_counts[0]==0][0] if 0 in val_counts[0] else 0}")
    print(f"    - known DDoS(1): {val_counts[1][val_counts[0]==1][0] if 1 in val_counts[0] else 0}")

    print(f"  新测试集: {len(new_test_labels)}样本")
    for lbl, cnt in zip(test_counts[0], test_counts[1]):
        label_name = {0: "benign", 1: "known DDoS", 2: "unknown DDoS"}.get(lbl, str(lbl))
        print(f"    - {label_name}({lbl}): {cnt}")

    # ========== 5. 构造CSV文本描述 ==========
    print("\n" + "-" * 40)
    print("5. 构造CSV文本描述...")

    def gather_csv_rows(csv_df, idx_list, new_labels, unknown_mask=None):
        """从原CSV中按索引收集行，并更新标签"""
        if csv_df is None:
            return None
        rows = csv_df.iloc[idx_list].copy()
        rows['Label'] = new_labels
        if unknown_mask is not None:
            # 为未知样本标注文本描述
            unknown_text_mask = unknown_mask[idx_list] if len(unknown_mask) > 0 else np.zeros(len(idx_list), dtype=bool)
            if 'text_description' in rows.columns:
                rows.loc[unknown_text_mask, 'text_description'] = \
                    rows.loc[unknown_text_mask, 'text_description'].apply(
                        lambda x: f"[未知攻击] {x}" if pd.notna(x) else "[未知攻击]"
                    )
        return rows

    # 训练集CSV
    new_train_csv = gather_csv_rows(train_csv, new_train_idx, new_train_labels)

    # 验证集CSV
    new_val_csv = gather_csv_rows(val_csv, new_val_idx, new_val_labels)

    # 测试集CSV — 需要拼接多来源
    if test_csv is not None:
        test_csv_benign = test_csv.iloc[test_benign_idx].copy()
        test_csv_benign['Label'] = test_labels[test_benign_idx]

        test_csv_known = test_csv.iloc[test_known_idx].copy()
        test_csv_known['Label'] = test_labels[test_known_idx]

        test_csv_unknown = test_csv.iloc[test_unknown_idx].copy()
        test_csv_unknown['Label'] = 2
        if 'text_description' in test_csv_unknown.columns:
            test_csv_unknown['text_description'] = test_csv_unknown['text_description'].apply(
                lambda x: f"[未知攻击] {x}" if pd.notna(x) else "[未知攻击]"
            )

        train_csv_unknown = train_csv.iloc[train_unknown_idx].copy()
        train_csv_unknown['Label'] = 2
        if 'text_description' in train_csv_unknown.columns:
            train_csv_unknown['text_description'] = train_csv_unknown['text_description'].apply(
                lambda x: f"[未知攻击] {x}" if pd.notna(x) else "[未知攻击]"
            )

        val_csv_unknown = val_csv.iloc[val_unknown_idx].copy()
        val_csv_unknown['Label'] = 2
        if 'text_description' in val_csv_unknown.columns:
            val_csv_unknown['text_description'] = val_csv_unknown['text_description'].apply(
                lambda x: f"[未知攻击] {x}" if pd.notna(x) else "[未知攻击]"
            )

        new_test_csv = pd.concat([test_csv_benign, test_csv_known, test_csv_unknown,
                                   train_csv_unknown, val_csv_unknown], ignore_index=True)
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

    # 保存CSV
    if new_train_csv is not None:
        new_train_csv.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
        print(f"  - train_data.csv: {len(new_train_csv)}行")
    if new_val_csv is not None:
        new_val_csv.to_csv(os.path.join(output_dir, "val_data.csv"), index=False)
        print(f"  - val_data.csv: {len(new_val_csv)}行")
    if new_test_csv is not None:
        new_test_csv.to_csv(os.path.join(output_dir, "test_data.csv"), index=False)
        print(f"  - test_data.csv: {len(new_test_csv)}行")

    # 复制scaler（与源划分一致）
    scaler_path = os.path.join(source_dir, "train_scaler.npy")
    if os.path.exists(scaler_path):
        import shutil
        shutil.copy2(scaler_path, os.path.join(output_dir, "train_scaler.npy"))
        print(f"  - train_scaler.npy: 已复制")

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
        'train_total': len(new_train_labels),
        'train_benign': int(train_counts[1][train_counts[0] == 0][0]) if 0 in train_counts[0] else 0,
        'train_known': int(train_counts[1][train_counts[0] == 1][0]) if 1 in train_counts[0] else 0,
        'val_total': len(new_val_labels),
        'val_benign': int(val_counts[1][val_counts[0] == 0][0]) if 0 in val_counts[0] else 0,
        'val_known': int(val_counts[1][val_counts[0] == 1][0]) if 1 in val_counts[0] else 0,
        'test_total': len(new_test_labels),
        'test_benign': int(test_counts[1][test_counts[0] == 0][0]) if 0 in test_counts[0] else 0,
        'test_known': int(test_counts[1][test_counts[0] == 1][0]) if 1 in test_counts[0] else 0,
        'test_unknown': int(test_counts[1][test_counts[0] == 2][0]) if 2 in test_counts[0] else 0,
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
        description="构造开集数据集划分 — 将部分DDoS样本标记为未知类，仅出现在测试集中"
    )
    parser.add_argument("--dataset_id", type=int, default=1, help="数据集ID（默认1）")
    parser.add_argument("--source_split_id", type=int, default=0, help="源划分ID（默认0）")
    parser.add_argument("--output_split_id", type=int, default=None, help="输出划分ID（默认自动递增）")
    parser.add_argument("--unknown_ratio", type=float, default=0.3, help="未知类比例（默认0.3）")
    parser.add_argument("--random_state", type=int, default=42, help="随机种子（默认42）")
    args = parser.parse_args()

    dataset_id, output_split_id = make_openset_split(
        dataset_id=args.dataset_id,
        source_split_id=args.source_split_id,
        output_split_id=args.output_split_id,
        unknown_ratio=args.unknown_ratio,
        random_state=args.random_state
    )
    sys.exit(0)