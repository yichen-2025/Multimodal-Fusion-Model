import pandas as pd
import numpy as np
import os

INPUT_DIR = "./data_processing"
OUTPUT_DIR = "./processed_dataset"

def clean_data(df):
    df = df.copy()
    
    df.columns = df.columns.str.strip()
    
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32']:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(df[col].mean())
    
    df = df.dropna()
    
    return df

def main():
    print("=" * 60)
    print("数据清洗脚本")
    print("=" * 60)

    csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
    if not csv_files:
        print(f"错误: 在 {INPUT_DIR} 目录中未找到CSV文件")
        return
    
    data_file = os.path.join(INPUT_DIR, csv_files[0])
    print(f"\n1. 读取原始数据: {data_file}")
    df = pd.read_csv(data_file)
    print(f"原始数据: {df.shape[0]}行, {df.shape[1]}列")

    print("\n2. 查看原始标签分布...")
    if 'Label' in df.columns:
        label_counts = df['Label'].value_counts()
        print(f"标签分布:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}")

    print("\n3. 数据清洗...")
    df_clean = clean_data(df)
    print(f"清洗后数据: {df_clean.shape[0]}行, {df_clean.shape[1]}列")

    print("\n4. 标签编码 (BENIGN→0, DDoS→1)...")
    if 'Label' in df_clean.columns:
        df_clean['Label'] = df_clean['Label'].map({'BENIGN': 0, 'DDoS': 1})
        label_counts = df_clean['Label'].value_counts()
        print(f"标签分布: 正常流量(0)={label_counts.get(0, 0)}, 恶意流量(1)={label_counts.get(1, 0)}")

    print("\n5. 保存清洗后数据...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "processed_dataset.csv")
    df_clean.to_csv(output_path, index=False)
    print(f"  - 保存路径: {os.path.abspath(output_path)}")
    print(f"  - 数据量: {df_clean.shape[0]}行, {df_clean.shape[1]}列")

    print("\n" + "=" * 60)
    print("数据清洗完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()