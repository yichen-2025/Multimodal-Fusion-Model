"""流量特征 schema 与文本模板的单一真相源。

本模块只负责四件事：
1. 定义数值分支与文本分支的列集合（消除 extract_subset / split_modality 各自的重复定义）；
2. 生成不含 label/origin 语义的文本描述；
3. 提供 NaN/inf fail-fast 断言；
4. 提供按数值特征去重（剔除跨标签重复组）。

约定：
- 数值分支 = 除 EXCLUDED_COLUMNS 以外的全部数值列（processed_dataset 实测 79 维）
- 文本分支 = TEXT_FEATURES，固定 20 个语义字段，统一 :.4g 格式化
- 文本中不得出现 label / origin / unknown / 未知 / "[" 等标记
"""

import numpy as np
import pandas as pd

# 文本分支统一最大长度：20 字段新模板实测 130-197 token，128 会截断末尾字段
TEXT_MAX_LENGTH = 256

# 标识符 / 标签 / 文本列，禁止进入数值分支，避免 ID 泄漏与目标泄漏
EXCLUDED_COLUMNS = {
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Timestamp",
    "Label",
    "text_description",
}

# 文本分支只描述语义明确、量纲稳定的字段
TEXT_FEATURES = [
    "Destination Port",
    "Protocol",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "Average Packet Size",
    "Down/Up Ratio",
    "SYN Flag Count",
    "ACK Flag Count",
    "PSH Flag Count",
    "URG Flag Count",
]

# 文本中禁止出现的标记（任何 label/origin 提示都算泄漏）
FORBIDDEN_TEXT_TOKENS = ("[", "未知", "unknown", "Unknown", "UNKNOWN")


def select_numeric_columns(df):
    """返回数值分支使用的列（保持 CSV 原始顺序）。存在非数值列即失败，禁止静默丢弃。"""
    columns = [c for c in df.columns if c not in EXCLUDED_COLUMNS]
    non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"存在未排除的非数值列: {non_numeric}")
    if not columns:
        raise ValueError("数值列为空，请检查输入 CSV 的 schema")
    return columns


def assert_finite(df, columns, split_name="input"):
    """NaN/inf fail-fast：数值必须可参与标准化与梯度计算。"""
    values = df[columns].to_numpy(dtype=np.float64)
    finite_cols = np.isfinite(values).all(axis=0)
    if not finite_cols.all():
        bad = [c for c, ok in zip(columns, finite_cols) if not ok]
        raise ValueError(f"{split_name}: 数值列含 NaN/inf: {bad}")
    return values


def generate_text_description(row):
    """按固定 20 字段模板生成文本描述。

    禁止 int() 截断、禁止 :.1f 舍入、禁止写入 label / origin 相关信息，
    否则数值不同而文本相同的样本对文本分支完全等价。
    """
    parts = [
        f"目标端口{int(row['Destination Port'])}",
        f"协议号{int(row['Protocol'])}",
        f"流持续{float(row['Flow Duration']):.4g}微秒",
        f"正向包数{float(row['Total Fwd Packets']):.4g}",
        f"反向包数{float(row['Total Backward Packets']):.4g}",
        f"正向字节{float(row['Total Length of Fwd Packets']):.4g}",
        f"反向字节{float(row['Total Length of Bwd Packets']):.4g}",
        f"流速{float(row['Flow Bytes/s']):.4g}字节每秒",
        f"包速{float(row['Flow Packets/s']):.4g}包每秒",
        f"流间隔均值{float(row['Flow IAT Mean']):.4g}微秒",
        f"流间隔标准差{float(row['Flow IAT Std']):.4g}",
        f"流间隔最大{float(row['Flow IAT Max']):.4g}微秒",
        f"正向间隔均值{float(row['Fwd IAT Mean']):.4g}微秒",
        f"反向间隔均值{float(row['Bwd IAT Mean']):.4g}微秒",
        f"平均包大小{float(row['Average Packet Size']):.4g}字节",
        f"下行上行比{float(row['Down/Up Ratio']):.4g}",
        f"SYN标志{float(row['SYN Flag Count']):.4g}",
        f"ACK标志{float(row['ACK Flag Count']):.4g}",
        f"PSH标志{float(row['PSH Flag Count']):.4g}",
        f"URG标志{float(row['URG Flag Count']):.4g}",
    ]
    return "。".join(parts) + "。"


_B36 = "0123456789abcdefghijklmnopqrstuvwxyz"


def _b36(n):
    """字段序号 → base36 短键（0→'0'，35→'z'，36→'10'）。"""
    s = ""
    while True:
        s = _B36[n % 36] + s
        n //= 36
        if n == 0:
            return s


def generate_compact_text(row, fields, value_format=".6g"):
    """位置键紧凑模板：'0=1.06e+07 1=8 2=5 …'（E10 全字段覆盖用）。

    键取字段在 `fields` 中的序号（base36），不写字段全名：实测全名或 4 字符缩写
    都会让 79 字段文本 100% 超过 512 token，位置键可压到 p99=413。
    既有 `generate_text_description` 的 20 字段模板保持不变。
    """
    parts = []
    for i, col in enumerate(fields):
        v = row.get(col)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if np.isnan(fv):
            continue
        parts.append(f"{_b36(i)}={fv:{value_format}}")
    return " ".join(parts)


def find_illegal_text(text_series):
    """返回含非法标记（label/origin 提示词）的布尔掩码。"""
    mask = pd.Series(False, index=text_series.index)
    text = text_series.astype(str)
    for token in FORBIDDEN_TEXT_TOKENS:
        mask = mask | text.str.contains(token, regex=False)
    return mask


def deduplicate_by_features(df, feature_columns, label_column="Label"):
    """按数值特征去重，并整组剔除跨标签重复组。

    步骤：
    1. 用 feature_columns 完全相等判定"同输入"分组；
    2. 组内出现多个 Label 的组，属于标签矛盾的数据缺陷，整组剔除；
    3. 剩余组内保留首行，消除重复流。

    Returns:
        (deduped_df, stats)
    """
    if not feature_columns:
        raise ValueError("feature_columns 为空，无法去重")
    if label_column not in df.columns:
        raise ValueError(f"缺少标签列: {label_column}")

    work = df.reset_index(drop=True)
    group_ids = work.groupby(
        list(feature_columns), sort=False, dropna=False
    ).ngroup().to_numpy()

    frame = pd.DataFrame({"gid": group_ids, "label": work[label_column].to_numpy()})
    n_labels_per_group = frame.groupby("gid")["label"].nunique(dropna=False)
    cross_gids = set(n_labels_per_group[n_labels_per_group > 1].index.tolist())

    cross_mask = pd.Series(group_ids).isin(cross_gids).to_numpy()
    kept = work[~cross_mask]
    deduped = kept.drop_duplicates(subset=list(feature_columns), keep="first")

    residual = deduped.groupby(
        list(feature_columns), sort=False, dropna=False
    )[label_column].nunique(dropna=False)
    if (residual > 1).any():
        raise RuntimeError("去重后仍存在跨标签重复组，schema 异常")

    stats = {
        "input_rows": int(len(work)),
        "unique_feature_groups": int(len(np.unique(group_ids))),
        "cross_label_groups": int(len(cross_gids)),
        "cross_label_rows": int(cross_mask.sum()),
        "duplicate_rows_removed": int(len(kept) - len(deduped)),
        "output_rows": int(len(deduped)),
    }
    return deduped.reset_index(drop=True), stats
