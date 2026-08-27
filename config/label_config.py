"""
统一标签映射配置
用于多分类任务的类别合并与标签映射，避免 data_cleaning.py 和 split_modality.py 之间的不一致。
"""

# ============================================================
# 原始15类标签映射（原始CSV中的标签 → 编码）
# ============================================================
ORIGINAL_LABEL_MAPPING = {
    "BENIGN": 0,
    "DoS Hulk": 1,
    "DoS GoldenEye": 2,
    "DoS slowloris": 3,
    "DoS Slowhttptest": 4,
    "DDoS": 5,
    "PortScan": 6,
    "FTP-Patator": 7,
    "SSH-Patator": 8,
    "Bot": 9,
    "Web Attack \x96 Brute Force": 10,
    "Web Attack \x96 XSS": 11,
    "Web Attack \x96 Sql Injection": 12,
    "Infiltration": 13,
    "Heartbleed": 14,
}

ORIGINAL_LABEL_NAMES = {v: k for k, v in ORIGINAL_LABEL_MAPPING.items()}

# ============================================================
# 合并后的12类标签映射（用于训练）
# ============================================================
# 合并规则：
#   - Web Attack 三个子类型 → 合并为 "Web Attack" (10)
#   - Infiltration + Heartbleed → 合并为 "Other Attack" (11)
# 合并阈值：样本数 < 50 的类别触发合并

MERGED_LABEL_MAPPING = {
    "BENIGN": 0,
    "DoS Hulk": 1,
    "DoS GoldenEye": 2,
    "DoS slowloris": 3,
    "DoS Slowhttptest": 4,
    "DDoS": 5,
    "PortScan": 6,
    "FTP-Patator": 7,
    "SSH-Patator": 8,
    "Bot": 9,
    "Web Attack": 10,
    "Other Attack": 11,
}

MERGED_LABEL_NAMES = {v: k for k, v in MERGED_LABEL_MAPPING.items()}

# ============================================================
# 原始编码 → 合并后编码 的映射规则
# ============================================================
# 将原始编码（0-14）映射到合并后的编码（0-11）
ORIGINAL_TO_MERGED = {
    0: 0,    # BENIGN → BENIGN
    1: 1,    # DoS Hulk → DoS Hulk
    2: 2,    # DoS GoldenEye → DoS GoldenEye
    3: 3,    # DoS slowloris → DoS slowloris
    4: 4,    # DoS Slowhttptest → DoS Slowhttptest
    5: 5,    # DDoS → DDoS
    6: 6,    # PortScan → PortScan
    7: 7,    # FTP-Patator → FTP-Patator
    8: 8,    # SSH-Patator → SSH-Patator
    9: 9,    # Bot → Bot
    10: 10,  # Web Attack - Brute Force → Web Attack
    11: 10,  # Web Attack - XSS → Web Attack
    12: 10,  # Web Attack - Sql Injection → Web Attack
    13: 11,  # Infiltration → Other Attack
    14: 11,  # Heartbleed → Other Attack
}

# ============================================================
# 合并阈值配置
# ============================================================
MIN_SAMPLES_THRESHOLD = 50  # 样本数低于此阈值的类别将被合并

# 需要合并的原始类别列表（编码）
MERGE_TARGET_ORIGINAL_CODES = {
    10, 11, 12,  # Web Attack 三个子类
    13, 14,      # Infiltration, Heartbleed
}

# 合并后的目标类别名称
MERGE_TARGET_NAMES = {
    10: "Web Attack",
    11: "Other Attack",
}


def normalize_label(label_str, use_merged=True):
    """
    标准化原始标签字符串，返回编码

    Args:
        label_str: 原始标签字符串
        use_merged: 是否使用合并后的12类编码（True）或原始15类编码（False）

    Returns:
        int or None: 对应的编码，无法识别时返回None
    """
    import pandas as pd

    if pd.isna(label_str):
        return None
    label_str = str(label_str).strip()

    mapping = MERGED_LABEL_MAPPING if use_merged else ORIGINAL_LABEL_MAPPING

    # 直接匹配
    if label_str in mapping:
        return mapping[label_str]

    # 特殊字符标准化后匹配
    normalized = label_str.replace('\x96', '-').replace('\x97', '-').replace('\x98', '-')
    normalized = normalized.replace('\x99', '-').replace('\u2013', '-').replace('\u2014', '-')

    for key, val in mapping.items():
        cleaned_key = key.replace('\x96', '-').replace('\x97', '-').replace('\u2013', '-').replace('\u2014', '-')
        if cleaned_key == normalized:
            return val

    # 模糊匹配 Web Attack 子类型
    if 'Web Attack' in label_str:
        if use_merged:
            return mapping["Web Attack"]
        else:
            if 'Brute Force' in label_str:
                return mapping["Web Attack \x96 Brute Force"]
            if 'XSS' in label_str:
                return mapping["Web Attack \x96 XSS"]
            if 'Sql Injection' in label_str or 'SQL' in label_str:
                return mapping["Web Attack \x96 Sql Injection"]

    return None