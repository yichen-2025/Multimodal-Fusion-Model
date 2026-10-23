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
from utils.dirs import ensure_dirs
from utils.log_utils import save_log
from utils.caliber import CaliberError

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
                       allow_mixed_calibration=False,
                       val_unknown_count=50,
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

    # 验证集：known 样本 + 少量 hold-out 样本(标签→unknown_label) 用于阈值校准
    val_keep_mask = np.isin(val_labels, known_classes)
    val_keep_idx = np.where(val_keep_mask)[0]

    # 从源 val 的 hold-out 样本中抽 val_unknown_count 个保留在新 val（作为真实 unknown）
    # 剩余的源 val hold-out → 移到 test 作为额外 OOD 样本
    if val_unknown_count < 0:
        raise ValueError(f'val_unknown_count 必须 >= 0，当前为 {val_unknown_count}')
    val_holdout_idx_all = np.where(np.isin(val_labels, hold_out_classes))[0]
    n_val_unknown = min(val_unknown_count, len(val_holdout_idx_all))
    if n_val_unknown > 0:
        rng_val = np.random.RandomState(random_state + 1)  # 独立种子
        val_unknown_idx = rng_val.choice(val_holdout_idx_all, size=n_val_unknown, replace=False)
        val_holdout_to_test = np.setdiff1d(val_holdout_idx_all, val_unknown_idx)
    else:
        val_unknown_idx = np.array([], dtype=int)
        val_holdout_to_test = np.array([], dtype=int)

    # 测试集 known 部分（known_classes 的原始标签保留）
    test_known_mask = np.isin(test_labels, known_classes)
    test_known_idx = np.where(test_known_mask)[0]

    # 测试集 hold-out 部分（标签改为 unknown_label）
    test_holdout_mask = np.isin(test_labels, hold_out_classes)
    test_holdout_idx = np.where(test_holdout_mask)[0]

    print(f"  - train known: {len(train_keep_idx)}")
    print(f"  - val known:   {len(val_keep_idx)}")
    print(f"  - val unknown (hold-out 移入, 标签→{unknown_label}): {len(val_unknown_idx)}")
    print(f"  - val holdout→test (额外 OOD): {len(val_holdout_to_test)}")
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

    # val unknown 部分（hold-out 移入，标签→unknown_label）
    if len(val_unknown_idx) > 0:
        val_unknown_stat = val_npz['scaled_features'][val_unknown_idx]
        val_unknown_bert = val_npz['text_embeddings'][val_unknown_idx]
        val_unknown_labels = np.full(len(val_unknown_idx), unknown_label, dtype=np.int64)

        new_val_stat = np.concatenate([new_val_stat, val_unknown_stat])
        new_val_bert = np.concatenate([new_val_bert, val_unknown_bert])
        new_val_labels = np.concatenate([new_val_labels, val_unknown_labels])

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

    # (d) 源 val hold-out 剩余部分 → 移入 test 作为额外 OOD
    val_holdout_extra_stat = val_npz['scaled_features'][val_holdout_to_test] if len(val_holdout_to_test) > 0 else np.empty((0, holdout_stat.shape[1] if holdout_stat.ndim > 1 else 0))
    val_holdout_extra_bert = val_npz['text_embeddings'][val_holdout_to_test] if len(val_holdout_to_test) > 0 else np.empty((0, holdout_bert.shape[1] if holdout_bert.ndim > 1 else 0))
    val_holdout_extra_labels = np.full(len(val_holdout_to_test), unknown_label, dtype=np.int64)

    new_test_stat = np.concatenate([
        test_known_keep_stat,
        test_sim_stat,
        holdout_stat,
        val_holdout_extra_stat,
    ]) if len(val_holdout_extra_stat) > 0 else np.concatenate([
        test_known_keep_stat,
        test_sim_stat,
        holdout_stat,
    ])
    new_test_bert = np.concatenate([
        test_known_keep_bert,
        test_sim_bert,
        holdout_bert,
        val_holdout_extra_bert,
    ]) if len(val_holdout_extra_bert) > 0 else np.concatenate([
        test_known_keep_bert,
        test_sim_bert,
        holdout_bert,
    ])
    new_test_labels = np.concatenate([
        test_known_keep_labels,
        test_sim_labels,
        holdout_labels,
        val_holdout_extra_labels,
    ])

    # ── unknown_origin: 0=known, 1=genuine_ood, 2=simulated_in_distribution ──
    # 后续所有脚本据此拆分 known / 真实OOD / 分布内模拟，禁止把两种 unknown 混算
    new_train_origin = np.zeros(len(new_train_labels), dtype=np.int64)  # 训练集全 known
    new_val_origin = np.concatenate([
        np.zeros(len(val_keep_idx), dtype=np.int64),             # val known
        np.full(len(val_unknown_idx), 1, dtype=np.int64),        # val unknown = 真实 hold-out
    ]) if len(val_unknown_idx) > 0 else np.zeros(len(val_keep_idx), dtype=np.int64)

    new_test_origin = np.concatenate([
        np.zeros(len(test_known_keep_labels), dtype=np.int64),   # (a) known keep
        np.full(len(test_sim_labels), 2, dtype=np.int64),        # (b) simulated
        np.full(len(holdout_labels), 1, dtype=np.int64),         # (c) 真实 hold-out
        np.full(len(val_holdout_extra_labels), 1, dtype=np.int64),  # (d) 来自 val 的真实 hold-out
    ])

    # B1：声明了模拟比例却未生成任何模拟 unknown（比例过小或无非 BENIGN 候选）
    if simulated_unknown_ratio > 0 and len(test_sim_labels) == 0:
        raise CaliberError(
            f"声明 simulated_unknown_ratio={simulated_unknown_ratio} 但未生成任何模拟 unknown，"
            f"请检查比例或候选样本"
        )

    # B2：val 与 test 的 unknown 构成必须同源，否则阈值校准与评估口径不一致
    n_val_sim = int(np.sum(new_val_origin == 2))
    n_test_sim = int(np.sum(new_test_origin == 2))
    if (n_val_sim > 0) != (n_test_sim > 0):
        if not allow_mixed_calibration:
            raise CaliberError(
                f"val 模拟 unknown={n_val_sim}，test 模拟 unknown={n_test_sim}："
                f"阈值校准口径与评估口径不一致。请让 val/test 同源，"
                f"或显式传 --allow_mixed_calibration 声明混合校准"
            )
        print("  ⚠️ B2: 已声明 allow_mixed_calibration，val/test 的 unknown 构成不同，"
              "所有下游报告须标注 mixed-calibration")

    # B3：unknown 标签不得与真实类别冲突（known 必须是连续 0..K-1）
    if known_classes != list(range(num_known_classes)):
        raise CaliberError(
            f"known_classes 不是连续的 0..{num_known_classes - 1}：{known_classes}，"
            f"unknown_label={unknown_label} 会与真实类别冲突"
        )

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

    # sanity check: 训练集不应出现 hold_out 标签
    assert len(np.intersect1d(new_train_labels, hold_out_classes)) == 0, \
        "训练集不应包含 hold_out_classes！"
    # 验证集：所有标签应为 known (0..unknown_label-1) 或 unknown_label
    assert np.all((new_val_labels < unknown_label) | (new_val_labels == unknown_label)), \
        "验证集标签异常：应只有 known 或 unknown_label"
    print(f"\n  ✅ sanity check 通过：train 不含 hold-out 类，val 含 {int(np.sum(new_val_labels == unknown_label))} 个 unknown 样本")

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

    # val CSV: known + unknown 两部分合并
    if val_csv is not None:
        val_known_csv = val_csv.iloc[val_keep_idx].copy()
        val_known_csv['Label'] = val_labels[val_keep_idx]
        if len(val_unknown_idx) > 0:
            val_unknown_csv = val_csv.iloc[val_unknown_idx].copy()
            val_unknown_csv['Label'] = unknown_label
            if 'text_description' in val_unknown_csv.columns:
                val_unknown_csv['text_description'] = val_unknown_csv['text_description'].apply(
                    lambda x: f"[未知-留出类] {x}" if pd.notna(x) else "[未知-留出类]"
                )
            new_val_csv = pd.concat([val_known_csv, val_unknown_csv], ignore_index=True)
        else:
            new_val_csv = val_known_csv
    else:
        new_val_csv = None

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

        # (d) 源 val hold-out 剩余 → 移入 test
        if len(val_holdout_to_test) > 0 and val_csv is not None:
            test_csv_d = val_csv.iloc[val_holdout_to_test].copy()
            test_csv_d['Label'] = unknown_label
            if 'text_description' in test_csv_d.columns:
                test_csv_d['text_description'] = test_csv_d['text_description'].apply(
                    lambda x: f"[未知-留出类-来自val] {x}" if pd.notna(x) else "[未知-留出类-来自val]"
                )
            new_test_csv = pd.concat([test_csv_a, test_csv_b, test_csv_c, test_csv_d], ignore_index=True)
        else:
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
             labels=new_train_labels,
             unknown_origin=new_train_origin)
    print(f"  - train.npz: {len(new_train_labels)} 样本")

    np.savez(os.path.join(output_dir, "val.npz"),
             scaled_features=new_val_stat,
             text_embeddings=new_val_bert,
             labels=new_val_labels,
             unknown_origin=new_val_origin,
             unknown_calibration_mixed=np.array([int(bool(allow_mixed_calibration))]))
    print(f"  - val.npz: {len(new_val_labels)} 样本")

    np.savez(os.path.join(output_dir, "test.npz"),
             scaled_features=new_test_stat,
             text_embeddings=new_test_bert,
             labels=new_test_labels,
             unknown_origin=new_test_origin,
             unknown_calibration_mixed=np.array([int(bool(allow_mixed_calibration))]))
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
        'val_unknown_count': val_unknown_count,
        'random_state': random_state,
        'train_total': len(new_train_labels),
        'val_total': len(new_val_labels),
        'test_total': len(new_test_labels),
        'test_known_keep': int(np.sum(new_test_labels < unknown_label)),
        'test_unknown_total': int(np.sum(new_test_labels == unknown_label)),
        'test_unknown_holdout': len(holdout_labels),
        'test_unknown_simulated': len(test_sim_labels),
        'test_unknown_genuine': int(np.sum(new_test_origin == 1)),
        'val_unknown_genuine': int(np.sum(new_val_origin == 1)),
        'unknown_origin_saved': True,
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
    ensure_dirs()
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
    parser.add_argument("--allow_mixed_calibration", action="store_true",
                        help="允许 val/test 的 unknown 构成不同（阈值校准与评估混合口径）")
    parser.add_argument("--val_unknown_count", type=int, default=50,
                        help="val 中保留的真实 hold-out unknown 数量（默认50，其余移入 test）")
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
        allow_mixed_calibration=args.allow_mixed_calibration,
        val_unknown_count=args.val_unknown_count,
        random_state=args.random_state,
        use_default_holdout=not args.no_default_holdout,
    )
    sys.exit(0)
