import pandas as pd
import numpy as np
import os
import json
import gc
import argparse
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# 导入统一标签配置
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
from config.label_config import (
    ORIGINAL_LABEL_MAPPING,
    ORIGINAL_LABEL_NAMES,
    MERGED_LABEL_MAPPING,
    MERGED_LABEL_NAMES,
    ORIGINAL_TO_MERGED,
    MIN_SAMPLES_THRESHOLD
)

INPUT_DIR = os.path.join(PROJECT_ROOT, "data_processing")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "processed_dataset")

BENIGN_RATIO = 2.0
RANDOM_STATE = 42
MIN_SAMPLES_PER_CLASS = 500
USE_SMOTE = True


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


def normalize_label(label, use_merged=True):
    """
    标准化原始标签字符串，返回编码
    使用统一配置中的映射规则，支持合并或不合并模式
    """
    import sys
    sys.path.insert(0, PROJECT_ROOT)
    from config.label_config import normalize_label as config_normalize
    return config_normalize(label, use_merged=use_merged)


def balance_with_smote(df, target_min_samples=500, random_state=42):
    """
    使用SMOTE过采样少数类 + 欠采样多数类来平衡数据集
    
    Args:
        df: 输入DataFrame（包含Label列和特征列）
        target_min_samples: 每个类别最少目标样本数
        random_state: 随机种子
    
    Returns:
        平衡后的DataFrame
    """
    print("\n  使用SMOTE过采样 + 欠采样进行类别均衡...")
    
    feature_cols = [col for col in df.columns if col != 'Label']
    X = df[feature_cols].copy()
    y = df['Label'].copy()
    
    label_counts = y.value_counts()
    print(f"  原始分布: {len(label_counts)} 个类别")
    
    min_class_count = label_counts.min()
    max_class_count = label_counts.max()
    print(f"  最少样本类: {min_class_count} 个样本")
    print(f"  最多样本类: {max_class_count} 个样本")
    
    if min_class_count >= target_min_samples:
        print(f"  所有类别样本数 >= {target_min_samples}，无需过采样")
        return df
    
    target_counts = {}
    for label in label_counts.index:
        count = label_counts[label]
        if count < target_min_samples:
            target_counts[label] = min(target_min_samples, count * 3)
        elif count > target_min_samples * 5:
            target_counts[label] = target_min_samples * 3
        else:
            target_counts[label] = count
    
    print(f"  目标分布: 最少 {min(target_counts.values())} 个样本, 最多 {max(target_counts.values())} 个样本")
    
    try:
        smote = SMOTE(
            sampling_strategy=target_counts,
            random_state=random_state,
            k_neighbors=min(min_class_count - 1, 5) if min_class_count > 1 else 1,
            n_jobs=-1
        )
        
        print("  执行SMOTE过采样...")
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        under_sampler = RandomUnderSampler(
            sampling_strategy={
                label: min(count, target_min_samples * 3)
                for label, count in y_resampled.value_counts().items()
            },
            random_state=random_state
        )
        
        print("  执行欠采样...")
        X_final, y_final = under_sampler.fit_resample(X_resampled, y_resampled)
        
        result_df = pd.DataFrame(X_final, columns=feature_cols)
        result_df['Label'] = y_final.values
        
        print(f"  均衡完成: {len(result_df)} 个样本")
        final_counts = result_df['Label'].value_counts().sort_index()
        for label_idx, count in final_counts.items():
            name = MERGED_LABEL_NAMES.get(label_idx, str(label_idx))
            print(f"    {label_idx}: {name} = {count}")
        
        return result_df
        
    except Exception as e:
        print(f"  SMOTE采样失败: {e}")
        print("  回退到简单的随机采样方案...")
        
        sampled_dfs = []
        for label in y.unique():
            class_df = df[df['Label'] == label]
            current_count = len(class_df)
            
            if current_count < target_min_samples:
                sampled = class_df.sample(
                    n=target_min_samples,
                    replace=True,
                    random_state=random_state
                )
            elif current_count > target_min_samples * 5:
                sampled = class_df.sample(
                    n=target_min_samples * 3,
                    random_state=random_state
                )
            else:
                sampled = class_df
            
            sampled_dfs.append(sampled)
        
        result_df = pd.concat(sampled_dfs, ignore_index=True)
        print(f"  简单采样完成: {len(result_df)} 个样本")
        return result_df


def main(use_merged=True):
    print("=" * 60)
    print("多分类数据清洗与合并脚本")
    print("=" * 60)
    print(f"标签模式: {'合并12类' if use_merged else '原始15类'}")
    print(f"合并阈值: MIN_SAMPLES_THRESHOLD = {MIN_SAMPLES_THRESHOLD}")

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
    if use_merged:
        mapping = MERGED_LABEL_MAPPING
        print(f"  标签映射表 ({len(mapping)} 类 - 已合并):")
    else:
        mapping = ORIGINAL_LABEL_MAPPING
        print(f"  标签映射表 ({len(mapping)} 类 - 原始):")
    for name, idx in sorted(mapping.items(), key=lambda x: x[1]):
        print(f"    {idx}: {name}")

    combined_df['Label'] = combined_df['Label'].apply(lambda x: normalize_label(x, use_merged=use_merged))
    valid_mask = combined_df['Label'].notna()
    dropped = (~valid_mask).sum()
    if dropped > 0:
        print(f"\n  警告: {dropped} 行无法识别的标签，已删除")
        combined_df = combined_df[valid_mask]

    combined_df['Label'] = combined_df['Label'].astype(int)
    label_counts = combined_df['Label'].value_counts().sort_index()
    label_names_lookup = MERGED_LABEL_NAMES if use_merged else ORIGINAL_LABEL_NAMES
    print(f"\n  标签分布 ({len(label_counts)} 类):")
    for label_idx, count in label_counts.items():
        name = label_names_lookup.get(label_idx, str(label_idx))
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

    if USE_SMOTE:
        print("\n  使用SMOTE进行类别平衡...")
        combined_df = balance_with_smote(
            combined_df,
            target_min_samples=MIN_SAMPLES_PER_CLASS,
            random_state=RANDOM_STATE
        )

    print(f"\n  均衡后标签分布:")
    label_counts_final = combined_df['Label'].value_counts().sort_index()
    for label_idx, count in label_counts_final.items():
        name = label_names_lookup.get(label_idx, str(label_idx))
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

    save_mapping = MERGED_LABEL_MAPPING if use_merged else ORIGINAL_LABEL_MAPPING
    save_names = MERGED_LABEL_NAMES if use_merged else ORIGINAL_LABEL_NAMES
    mapping_path = os.path.join(OUTPUT_DIR, "label_mapping.json")
    mapping_data = {
        "label_to_id": {name: idx for name, idx in save_mapping.items()},
        "id_to_label": {str(idx): name for idx, name in save_names.items()},
        "num_classes": len(save_mapping),
        "use_merged": use_merged,
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
    parser = argparse.ArgumentParser(description="多分类数据清洗与合并脚本")
    parser.add_argument("--use_merged", action='store_true', default=True,
                        help="使用合并后的12类标签（默认开启）")
    parser.add_argument("--no_merged", action='store_true', default=False,
                        help="使用原始15类标签（不合并）")
    args = parser.parse_args()
    main(use_merged=not args.no_merged)