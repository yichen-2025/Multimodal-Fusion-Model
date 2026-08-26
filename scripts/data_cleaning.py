import pandas as pd
import numpy as np
import os
import json
import gc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_DIR = os.path.join(PROJECT_ROOT, "data_processing")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "processed_dataset")

LABEL_MAPPING = {
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

LABEL_NAMES = {v: k for k, v in LABEL_MAPPING.items()}

BENIGN_RATIO = 2.0
RANDOM_STATE = 42


def read_csv_safe(filepath):
    for encoding in ['utf-8', 'latin1', 'iso-8859-1']:
        try:
            df = pd.read_csv(filepath, encoding=encoding, low_memory=False)
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
    df = pd.read_csv(filepath, encoding='utf-8', low_memory=False, on_bad_lines='skip')
    return df


def clean_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    for col in df.columns:
        if df[col].dtype in ['float64', 'float32']:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(df[col].mean())

    df = df.dropna()
    return df


def normalize_label(label):
    if pd.isna(label):
        return None
    label_str = str(label).strip()

    if label_str in LABEL_MAPPING:
        return LABEL_MAPPING[label_str]

    normalized = label_str.replace('\x96', '-').replace('\x97', '-').replace('\x98', '-').replace('\x99', '-')
    normalized = normalized.replace('\u2013', '-').replace('\u2014', '-').replace('\u2015', '-')

    for key, val in LABEL_MAPPING.items():
        if key.replace('\x96', '-').replace('\x97', '-').replace('\u2013', '-').replace('\u2014', '-') == normalized:
            return val

    if 'Web Attack' in label_str:
        if 'Brute Force' in label_str:
            return LABEL_MAPPING["Web Attack \x96 Brute Force"]
        if 'XSS' in label_str:
            return LABEL_MAPPING["Web Attack \x96 XSS"]
        if 'Sql Injection' in label_str or 'SQL' in label_str:
            return LABEL_MAPPING["Web Attack \x96 Sql Injection"]

    return None


def main():
    print("=" * 60)
    print("多分类数据清洗与合并脚本")
    print("=" * 60)

    csv_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')])
    if not csv_files:
        print(f"错误: 在 {INPUT_DIR} 目录中未找到CSV文件")
        return

    print(f"\n发现 {len(csv_files)} 个CSV文件:")
    for f in csv_files:
        print(f"  - {f}")

    print("\n" + "-" * 40)
    print("Step 1: 读取并合并所有CSV文件...")

    all_dfs = []
    total_rows = 0
    for csv_file in csv_files:
        filepath = os.path.join(INPUT_DIR, csv_file)
        print(f"\n  读取: {csv_file}")
        df = read_csv_safe(filepath)
        df.columns = df.columns.str.strip()
        print(f"    原始: {df.shape[0]}行, {df.shape[1]}列")

        if 'Label' not in df.columns:
            print(f"    警告: 未找到Label列，跳过该文件")
            continue

        df_clean = clean_data(df)
        print(f"    清洗后: {df_clean.shape[0]}行")
        all_dfs.append(df_clean)
        total_rows += len(df_clean)
        del df, df_clean
        gc.collect()

    print(f"\n  合并后总行数: {total_rows}")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    gc.collect()

    print("\n" + "-" * 40)
    print("Step 2: 标签编码...")
    print(f"  标签映射表 ({len(LABEL_MAPPING)} 类):")
    for name, idx in sorted(LABEL_MAPPING.items(), key=lambda x: x[1]):
        print(f"    {idx}: {name}")

    combined_df['Label'] = combined_df['Label'].apply(normalize_label)
    valid_mask = combined_df['Label'].notna()
    dropped = (~valid_mask).sum()
    if dropped > 0:
        print(f"\n  警告: {dropped} 行无法识别的标签，已删除")
        combined_df = combined_df[valid_mask]

    combined_df['Label'] = combined_df['Label'].astype(int)
    label_counts = combined_df['Label'].value_counts().sort_index()
    print(f"\n  合并后标签分布:")
    for label_idx, count in label_counts.items():
        name = LABEL_NAMES.get(label_idx, str(label_idx))
        pct = count / len(combined_df) * 100
        print(f"    {label_idx}: {name:40s} = {count:>8d} ({pct:5.2f}%)")

    print("\n" + "-" * 40)
    print("Step 3: 类别均衡采样...")
    label_counts_dict = combined_df['Label'].value_counts()
    benign_count = label_counts_dict.get(0, 0)
    non_benign_max = label_counts_dict.drop(0).max() if len(label_counts_dict) > 1 else 0
    target_benign = int(non_benign_max * BENIGN_RATIO)

    print(f"  BENIGN原始数量: {benign_count:,}")
    print(f"  最大非BENIGN类: {non_benign_max:,}")
    print(f"  BENIGN采样比例: {BENIGN_RATIO}x (目标: {target_benign:,})")

    if target_benign < benign_count:
        print(f"  下采样BENIGN: {benign_count:,} -> {target_benign:,}")
        benign_df = combined_df[combined_df['Label'] == 0].sample(
            n=target_benign, random_state=RANDOM_STATE
        )
        non_benign_df = combined_df[combined_df['Label'] != 0]
        combined_df = pd.concat([benign_df, non_benign_df], ignore_index=True)
        del benign_df, non_benign_df
        gc.collect()
    else:
        print(f"  保留全部BENIGN样本")

    print(f"\n  均衡后标签分布:")
    label_counts_final = combined_df['Label'].value_counts().sort_index()
    for label_idx, count in label_counts_final.items():
        name = LABEL_NAMES.get(label_idx, str(label_idx))
        pct = count / len(combined_df) * 100
        print(f"    {label_idx}: {name:40s} = {count:>8d} ({pct:5.2f}%)")

    print(f"  总样本数: {len(combined_df):,}")

    print("\n" + "-" * 40)
    print("Step 4: 保存清洗后数据...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_csv = os.path.join(OUTPUT_DIR, "processed_dataset.csv")
    combined_df.to_csv(output_csv, index=False)
    print(f"  数据文件: {output_csv}")
    print(f"  行数: {combined_df.shape[0]:,}, 列数: {combined_df.shape[1]}")

    mapping_path = os.path.join(OUTPUT_DIR, "label_mapping.json")
    mapping_data = {
        "label_to_id": {name: idx for name, idx in LABEL_MAPPING.items()},
        "id_to_label": {str(idx): name for idx, name in LABEL_NAMES.items()},
        "num_classes": len(LABEL_MAPPING),
        "benign_ratio": BENIGN_RATIO,
        "total_samples": len(combined_df),
        "class_distribution": {
            str(int(k)): int(v) for k, v in label_counts_final.items()
        }
    }
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping_data, f, ensure_ascii=False, indent=2)
    print(f"  标签映射: {mapping_path}")

    print("\n" + "=" * 60)
    print("数据清洗与合并完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()