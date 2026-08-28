import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, PROJECT_ROOT)
from utils.log_utils import save_log

SPLIT_DATA_DIR = os.path.join(PROJECT_ROOT, "split_data")

# 12类合并后的标签名（与 label_config.py 的 MERGED_LABEL_MAPPING 一致）
LABEL_NAMES_MERGED = {
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
    10: "Web Attack",
    11: "Other Attack",
}

# 原始15类标签名（仅作参考）
LABEL_NAMES_RAW = {
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

LABEL_NAMES = LABEL_NAMES_MERGED

# 默认留出类别：选 Web Attack(10) + Other Attack(11) 两个合并类
# 理由：(1) 这两个本身就是小类/合并类，适合做 unknown
#       (2) 留出后已知类仍有 10 个，覆盖 DoS/DDoS/端口扫描/暴力破解/僵尸网络等多种攻击
#       (3) 留出类在特征分布上与已知类有真实差异
DEFAULT_HOLD_OUT = [10, 11]


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


def make_openset_split(dataset_id=0,
                       source_split_id=0,
                       output_split_id=None,
                       hold_out_classes=None,
                       simulated_unknown_ratio=0.1,
                       random_state=42,
                       use_default_holdout=True):
    """
    构造多分类开集数据集划分 —— 基于「留出类别」策略

    ===================================================================
    新策略（真正的开集划分）
    ===================================================================

    Phase 1: 真正留出类别 (Primary)
      - hold_out_classes 指定的类别 → 完全从训练集/验证集中移除
      - 测试集中这些类别的样本 → 全部标记为 unknown (label = num_known_classes)
      - 这些样本构成了「真实分布偏移」的 OOD，与训练分布完全不同

    Phase 2: 分布内模拟未知 (Optional, 次要)
      - 从测试集的已知非 BENIGN 样本中，再随机标记少量为 unknown
      - 这些样本特征分布与已知类完全相同，但被迫当 unknown
      - 作用：评估 OOD 头在 hard case 上的表现
      - 默认比例 0.1（10%），可设为 0 关闭

    与旧策略的本质区别：
      ❌ 旧：从已知类中随机标记样本为 unknown → 分布无变化 → OOD 头完全失败
      ✅ 新：留出类别从未参与训练 → 测试时才出现 → 真正的 OOD 场景

    ===================================================================

    Args:
        dataset_id (int): 数据集ID
        source_split_id (int): 源划分ID（如 split_0）
        output_split_id (int): 输出划分ID，默认自动递增
        hold_out_classes (list): 要完全留出的类别ID列表。
            若 use_default_holdout=True 且本参数为 None，使用 DEFAULT_HOLD_OUT。
        simulated_unknown_ratio (float): 分布内模拟 unknown 的比例（默认0.1）。
            设为 0 可关闭。
        random_state (int): 随机种子
        use_default_holdout (bool): hold_out_classes 为 None 时是否使用默认留出集

    Returns:
        tuple: (dataset_id, output_split_id)
    """
    start_time = time.time()

    # ========== 0. 确定留出类别 ==========
    if hold_out_classes is None and use_default_holdout:
        hold_out_classes = list(DEFAULT_HOLD_OUT)
    elif hold_out_classes is None:
        hold_out_classes = []

    if output_split_id is None:
        output_split_id = get_next_openset_split_id(dataset_id)

    source_dir = os.path.join(SPLIT_DATA_DIR, f"dataset_{dataset_id}", f"split_{source_split_id}")
    output_dir = os.path.join(SPLIT_DATA_DIR, f"dataset_{dataset_id}", f"split_openset_{output_split_id}")

    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"源划分目录不存在: {source_dir}")

    print("=" * 60)
    print("多分类开集数据集划分构造（留出类别策略）")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  - 数据集ID: {dataset_id}")
    print(f"  - 源划分ID: {source_split_id}")
    print(f"  - 输出划分ID: {output_split_id}")
    print(f"  - 留出类别 (hold_out_classes): {hold_out_classes}")
    print(f"  - 分布内模拟 unknown 比例: {simulated_unknown_ratio}")
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

    train_labels = train_npz['labels'].astype(np.int64)
    val_labels = val_npz['labels'].astype(np.int64)
    test_labels = test_npz['labels'].astype(np.int64)

    # 源数据的完整类别列表
    all_source_labels = np.concatenate([train_labels, val_labels, test_labels])
    all_source_classes = sorted(np.unique(all_source_labels).tolist())

    print(f"  - 源数据类别数: {len(all_source_classes)}")
    for c in all_source_classes:
        name = LABEL_NAMES.get(c, str(c))
        tr = int(np.sum(train_labels == c))
        va = int(np.sum(val_labels == c))
        te = int(np.sum(test_labels == c))
        print(f"    {c}: {name}  train={tr} val={va} test={te}")

    # ========== 2. 确定 known / hold-out 分割 ==========
    print("\n" + "-" * 40)
    print("2. 确定 known / hold-out 分割...")

    # 校验 hold_out_classes
    for c in hold_out_classes:
        if c not in all_source_classes:
            print(f"  警告: hold_out_classes 中的 {c} 在源数据中不存在，已忽略")
    hold_out_classes = [c for c in hold_out_classes if c in all_source_classes]

    known_classes = sorted([c for c in all_source_classes if c not in hold_out_classes])
    num_known_classes = len(known_classes)
    unknown_label = num_known_classes  # known = 0..K-1, unknown = K

    print(f"  - known_classes ({num_known_classes}个): {known_classes}")
    for c in known_classes:
        print(f"      {c}: {LABEL_NAMES.get(c, str(c))}")
    print(f"  - hold_out_classes ({len(hold_out_classes)}个): {hold_out_classes}")
    for c in hold_out_classes:
        print(f"      {c}: {LABEL_NAMES.get(c, str(c))}")
    print(f"  - unknown_label = {unknown_label}")

    # ========== 3. Phase 1: 真正留出类别 ==========
    print("\n" + "-" * 40)
    print("3. Phase 1 — 真正留出类别...")

    # 训练集：只保留 known_classes，完全不含 hold_out
    train_keep_mask = np.isin(train_labels, known_classes)
    train_keep_idx = np.where(train_keep_mask)[0]

    # 验证集：同训练集
    val_keep_mask = np.isin(val_labels, known_classes)
    val_keep_idx = np.where(val_keep_mask)[0]

    # 测试集 known 部分（known_classes 的原始标签保留）
    test_known_mask = np.isin(test_labels, known_classes)
    test_known_idx = np.where(test_known_mask)[0]

    # 测试集 hold-out 部分（标签改为 unknown_label）
    test_holdout_mask = np.isin(test_labels, hold_out_classes)
    test_holdout_idx = np.where(test_holdout_mask)[0]

    print(f"  - train known: {len(train_keep_idx)}")
    print(f"  - val known:   {len(val_keep_idx)}")
    print(f"  - test known (原始): {len(test_known_idx)}")
    print(f"  - test hold-out (真实OOD, 标签→{unknown_label}): {len(test_holdout_idx)}")

    # ========== 4. Phase 2: 分布内模拟 unknown ==========
    print("\n" + "-" * 40)
    print("4. Phase 2 — 分布内模拟 unknown (ratio={})".format(simulated_unknown_ratio))

    rng = np.random.RandomState(random_state)

    # 从 test known 样本中（排除 BENIGN=0），随机选 simulated_unknown_ratio 改为 unknown
    test_known_labels = test_labels[test_known_idx]
    test_known_nonbenign_mask = test_known_labels != 0
    test_known_nonbenign_idx_in_subset = np.where(test_known_nonbenign_mask)[0]

    simulated_idx_in_subset = np.array([], dtype=int)
    if simulated_unknown_ratio > 0 and len(test_known_nonbenign_idx_in_subset) > 0:
        n_sim = int(len(test_known_nonbenign_idx_in_subset) * simulated_unknown_ratio)
        if n_sim > 0:
            simulated_idx_in_subset = rng.choice(
                test_known_nonbenign_idx_in_subset, size=n_sim, replace=False
            )

    test_known_keep_idx = np.setdiff1d(
        np.arange(len(test_known_idx)), simulated_idx_in_subset
    )
    test_simulated_idx = test_known_idx[simulated_idx_in_subset]

    print(f"  - test known keep (保留known标签): {len(test_known_keep_idx)}")
    print(f"  - test simulated unknown (分布内改→{unknown_label}): {len(test_simulated_idx)}")

    # ========== 5. 构造新划分 ==========
    print("\n" + "-" * 40)
    print("5. 构造新开集划分...")

    # 训练集/验证集：只有 known，标签保持原样
    new_train_stat = train_npz['scaled_features'][train_keep_idx]
    new_train_bert = train_npz['text_embeddings'][train_keep_idx]
    new_train_labels = train_labels[train_keep_idx].copy()

    new_val_stat = val_npz['scaled_features'][val_keep_idx]
    new_val_bert = val_npz['text_embeddings'][val_keep_idx]
    new_val_labels = val_labels[val_keep_idx].copy()

    # 测试集：三部分拼接
    # (a) test known keep — 保留原标签
    test_known_keep_stat = test_npz['scaled_features'][test_known_idx[test_known_keep_idx]]
    test_known_keep_bert = test_npz['text_embeddings'][test_known_idx[test_known_keep_idx]]
    test_known_keep_labels = test_labels[test_known_idx[test_known_keep_idx]].copy()

    # (b) test simulated unknown — 标签改为 unknown_label
    test_sim_stat = test_npz['scaled_features'][test_simulated_idx]
    test_sim_bert = test_npz['text_embeddings'][test_simulated_idx]
    test_sim_labels = np.full(len(test_simulated_idx), unknown_label, dtype=np.int64)

    # (c) test hold-out — 标签改为 unknown_label (真正的 OOD)
    holdout_stat = test_npz['scaled_features'][test_holdout_idx]
    holdout_bert = test_npz['text_embeddings'][test_holdout_idx]
    holdout_labels = np.full(len(test_holdout_idx), unknown_label, dtype=np.int64)

    new_test_stat = np.concatenate([
        test_known_keep_stat,
        test_sim_stat,
        holdout_stat,
    ])
    new_test_bert = np.concatenate([
        test_known_keep_bert,
        test_sim_bert,
        holdout_bert,
    ])
    new_test_labels = np.concatenate([
        test_known_keep_labels,
        test_sim_labels,
        holdout_labels,
    ])

    # 打印新划分的统计
    print(f"\n  训练集 (known only): {len(new_train_labels)} 样本")
    for c in known_classes:
        cnt = int(np.sum(new_train_labels == c))
        print(f"    {c}: {LABEL_NAMES.get(c, str(c))} = {cnt}")

    print(f"  验证集 (known only): {len(new_val_labels)} 样本")
    for c in known_classes:
        cnt = int(np.sum(new_val_labels == c))
        print(f"    {c}: {LABEL_NAMES.get(c, str(c))} = {cnt}")

    print(f"  测试集: {len(new_test_labels)} 样本")
    for lbl, cnt in zip(*np.unique(new_test_labels, return_counts=True)):
        if lbl < unknown_label:
            name = LABEL_NAMES.get(int(lbl), str(lbl))
            src = "known"
        else:
            name = "unknown"
            # 拆分 simulated vs hold-out
            n_sim = len(test_sim_labels)
            n_ho = len(holdout_labels)
            src = "hold-out(OOD)+simulated={}+{}".format(n_ho, n_sim)
        print(f"    {lbl}: {name} = {cnt}  ({src})")

    # sanity check: 训练/验证集不应出现 hold_out 标签
    assert len(np.intersect1d(new_train_labels, hold_out_classes)) == 0, \
        "训练集不应包含 hold_out_classes！"
    assert len(np.intersect1d(new_val_labels, hold_out_classes)) == 0, \
        "验证集不应包含 hold_out_classes！"
    print("\n  ✅ sanity check 通过：train/val 不含 hold-out 类")

    # ========== 6. 构造 CSV 文本描述 ==========
    print("\n" + "-" * 40)
    print("6. 构造 CSV 文本描述...")

    def gather_csv_rows(csv_df, idx_list, new_labels):
        if csv_df is None:
            return None
        rows = csv_df.iloc[idx_list].copy()
        rows['Label'] = new_labels
        return rows

    new_train_csv = gather_csv_rows(train_csv, train_keep_idx, new_train_labels)
    new_val_csv = gather_csv_rows(val_csv, val_keep_idx, new_val_labels)

    if test_csv is not None:
        # (a) known keep 部分
        test_csv_a = test_csv.iloc[test_known_idx[test_known_keep_idx]].copy()
        test_csv_a['Label'] = test_labels[test_known_idx[test_known_keep_idx]]

        # (b) simulated unknown 部分
        test_csv_b = test_csv.iloc[test_simulated_idx].copy()
        test_csv_b['Label'] = unknown_label
        if 'text_description' in test_csv_b.columns:
            test_csv_b['text_description'] = test_csv_b['text_description'].apply(
                lambda x: f"[未知-模拟] {x}" if pd.notna(x) else "[未知-模拟]"
            )

        # (c) hold-out unknown 部分
        test_csv_c = test_csv.iloc[test_holdout_idx].copy()
        test_csv_c['Label'] = unknown_label
        if 'text_description' in test_csv_c.columns:
            test_csv_c['text_description'] = test_csv_c['text_description'].apply(
                lambda x: f"[未知-留出类] {x}" if pd.notna(x) else "[未知-留出类]"
            )

        new_test_csv = pd.concat([test_csv_a, test_csv_b, test_csv_c], ignore_index=True)
    else:
        new_test_csv = None

    # ========== 7. 保存 ==========
    print("\n" + "-" * 40)
    print("7. 保存开集划分数据...")

    os.makedirs(output_dir, exist_ok=True)

    np.savez(os.path.join(output_dir, "train.npz"),
             scaled_features=new_train_stat,
             text_embeddings=new_train_bert,
             labels=new_train_labels)
    print(f"  - train.npz: {len(new_train_labels)} 样本")

    np.savez(os.path.join(output_dir, "val.npz"),
             scaled_features=new_val_stat,
             text_embeddings=new_val_bert,
             labels=new_val_labels)
    print(f"  - val.npz: {len(new_val_labels)} 样本")

    np.savez(os.path.join(output_dir, "test.npz"),
             scaled_features=new_test_stat,
             text_embeddings=new_test_bert,
             labels=new_test_labels)
    print(f"  - test.npz: {len(new_test_labels)} 样本")

    if new_train_csv is not None:
        new_train_csv.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
    if new_val_csv is not None:
        new_val_csv.to_csv(os.path.join(output_dir, "val_data.csv"), index=False)
    if new_test_csv is not None:
        new_test_csv.to_csv(os.path.join(output_dir, "test_data.csv"), index=False)

    # 保存更新后的 label_mapping（只有 known 类）
    new_mapping = {str(c): LABEL_NAMES.get(c, str(c)) for c in known_classes}
    new_mapping[str(unknown_label)] = "unknown"
    with open(os.path.join(output_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(new_mapping, f, ensure_ascii=False, indent=2)
    print(f"  - label_mapping.json: 已更新（known={num_known_classes}, unknown={unknown_label}）")

    # 复制 scaler
    scaler_path = os.path.join(source_dir, "train_scaler.npy")
    if os.path.exists(scaler_path):
        import shutil
        shutil.copy2(scaler_path, os.path.join(output_dir, "train_scaler.npy"))
        print(f"  - train_scaler.npy: 已复制")

    duration_seconds = time.time() - start_time

    # ========== 8. 日志记录 ==========
    print("\n" + "-" * 40)
    print("8. 记录日志...")

    log_data = {
        'dataset_id': dataset_id,
        'source_split_id': source_split_id,
        'output_split_id': output_split_id,
        'strategy': 'hold_out_classes',
        'hold_out_classes': hold_out_classes,
        'known_classes': known_classes,
        'num_known_classes': num_known_classes,
        'unknown_label': unknown_label,
        'simulated_unknown_ratio': simulated_unknown_ratio,
        'random_state': random_state,
        'train_total': len(new_train_labels),
        'val_total': len(new_val_labels),
        'test_total': len(new_test_labels),
        'test_known_keep': int(np.sum(new_test_labels < unknown_label)),
        'test_unknown_total': int(np.sum(new_test_labels == unknown_label)),
        'test_unknown_holdout': len(holdout_labels),
        'test_unknown_simulated': len(test_sim_labels),
        'output_dir': os.path.abspath(output_dir),
        'duration_seconds': round(duration_seconds, 3)
    }
    log_id = save_log('openset_split', log_data)
    print(f"  - 日志已保存: logs/openset_split/log_{log_id}.json")

    print("\n" + "=" * 60)
    print("✅ 开集数据集划分构造完成（留出类别策略）！")
    print(f"  - known:   {known_classes} ({num_known_classes}类)")
    print(f"  - holdout: {hold_out_classes} ({len(hold_out_classes)}类，从未见过)")
    print(f"  - 输出目录: {os.path.abspath(output_dir)}")
    print(f"  - 耗时: {duration_seconds:.2f}秒")
    print("=" * 60)

    return dataset_id, output_split_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="构造多分类开集数据集划分（留出类别策略 — 真正的 OOD）"
    )
    parser.add_argument("--dataset_id", type=int, default=0, help="数据集ID（默认0）")
    parser.add_argument("--source_split_id", type=int, default=0, help="源划分ID（默认0）")
    parser.add_argument("--output_split_id", type=int, default=None, help="输出划分ID（默认自动递增）")
    parser.add_argument("--hold_out_classes", type=int, nargs='*', default=None,
                        help="完全留出的类别ID列表。默认自动使用 [10, 11] (Web Attack + Other Attack)")
    parser.add_argument("--simulated_unknown_ratio", type=float, default=0.1,
                        help="测试集中已知非BENIGN样本改为模拟unknown的比例（默认0.1，设为0关闭）")
    parser.add_argument("--no_default_holdout", action="store_true",
                        help="关闭默认留出集（需要手动指定 --hold_out_classes）")
    parser.add_argument("--random_state", type=int, default=42, help="随机种子（默认42）")
    args = parser.parse_args()

    dataset_id, output_split_id = make_openset_split(
        dataset_id=args.dataset_id,
        source_split_id=args.source_split_id,
        output_split_id=args.output_split_id,
        hold_out_classes=args.hold_out_classes,
        simulated_unknown_ratio=args.simulated_unknown_ratio,
        random_state=args.random_state,
        use_default_holdout=not args.no_default_holdout,
    )
    sys.exit(0)
