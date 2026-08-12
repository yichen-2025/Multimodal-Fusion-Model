"""
多模态融合模型主入口文件
使用方式：取消注释对应步骤的代码，运行 python main.py

步骤说明：
1. 数据清洗（data_cleaning.py）
2. 提取子集（extract_subset.py）
3. 模态分离（split_modality.py）
4. 模型训练（train.py）
5. 模型测试（test_model.py）
6. 绘制loss曲线（plot_loss_curve）

注意：请按照顺序逐步取消注释执行，确保每一步完成后再执行下一步
"""

import os
import pandas as pd

from scripts.extract_subset import extract_subset
from scripts.split_modality import split_modality
from scripts.train import train_model
from scripts.test_model import test_model
from scripts.run_ablation import run_ablation, plot_f1_comparison


def create_necessary_directories():
    """
    创建项目所需的目录（被.gitignore忽略的目录）
    
    这些目录在首次运行时可能不存在，需要提前创建以避免FileNotFoundError
    
    创建的目录列表：
    - processed_dataset/: 处理后的数据
    - split_data/: 划分后的数据
    - saved_models/: 训练模型
    - logs/subset/: 子集提取日志
    - logs/split/: 数据划分日志
    - logs/training/: 模型训练日志
    - test_reports/: 测试报告
    """
    directories = [
        "processed_dataset",
        "split_data",
        "saved_models",
        "logs/subset",
        "logs/split",
        "logs/training",
        "test_reports"
    ]
    
    created_dirs = []
    for dir_path in directories:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            created_dirs.append(dir_path)
    
    if created_dirs:
        print("创建了以下目录：")
        for dir_path in created_dirs:
            print(f"  - {dir_path}")
    else:
        print("所有必要目录已存在")


def plot_loss_curve(model_id):
    """
    绘制训练loss曲线
    
    Args:
        model_id (int): 模型ID
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    # 读取loss日志
    df = pd.read_csv(f"./saved_models/model_{model_id}/loss_logs/loss_log.csv")
    
    # 绘制loss曲线
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['loss'], label='Training Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title(f'Model {model_id} Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    """
    主函数：按照步骤逐步执行
    
    使用方法：
    1. 取消注释第一步代码，运行 python main.py
    2. 第一步完成后，注释第一步，取消注释第二步
    3. 依次执行后续步骤
    """
    
    # 创建必要的目录（被.gitignore忽略的目录）
    create_necessary_directories()
    
    # ============================================
    # 步骤1：数据清洗
    # 输入：data_processing/ 目录下的CSV文件
    # 输出：processed_dataset/processed_dataset.csv
    # ============================================
    # from scripts.data_cleaning import main as run_data_cleaning
    # run_data_cleaning()
    
    # ============================================
    # 步骤2：提取数据集子集
    # 输入：processed_dataset/processed_dataset.csv
    # 输出：processed_dataset/dataset_X.csv
    # 参数：
    #   num_samples: 提取样本数量（默认5000）
    #   dataset_id: 数据集ID（默认自动递增）
    #   random_state: 随机种子（默认42）
    # ============================================
    # success, dataset_id = extract_subset(
    #     num_samples=5000,
    #     # dataset_id=0,  # 可选：指定数据集ID
    #     random_state=42
    # )
    # if success:
    #     print(f"成功提取子集，数据集ID: {dataset_id}")
    # else:
    #     print("提取子集失败")
    
    # ============================================
    # 步骤3：模态分离与数据集划分
    # 输入：processed_dataset/dataset_X.csv
    # 输出：split_data/dataset_X/split_Y/
    # 参数：
    #   dataset_id: 数据集ID（默认0）
    #   split_id: 划分ID（默认自动递增）
    #   test_size: 测试集比例（默认0.2）
    #   random_state: 随机种子（默认42）
    # ============================================
    # dataset_id = 0  # 与步骤2的dataset_id一致
    # split_modality(
    #     dataset_id=dataset_id,
    #     # split_id=0,  # 可选：指定划分ID
    #     test_size=0.2,
    #     random_state=42
    # )
    # print(f"成功完成模态分离，数据集ID: {dataset_id}")
    
    # ============================================
    # 步骤4：消融实验
    # 输入：split_data/dataset_X/split_Y/
    # 输出：ablation_results/ablation_results_时间戳.csv
    # 参数：
    #   variants: 变体列表（默认["A0","A1","A2","A3"]）
    #   dataset_id: 数据集ID（默认0）
    #   split_id: 划分ID（默认0）
    #   repeat: 重复次数（默认1）
    #   seed: 随机种子（默认42）
    # ============================================
    # run_ablation(
    #     variants=["A0","A1","A2","A3"],
    #     dataset_id=0,
    #     split_id=0,
    #     repeat=1,
    #     seed=42
    # )

    # ============================================
    # 步骤5：绘制消融实验结果
    # 输入：ablation_results/ablation_results_时间戳.csv
    # 输出：ablation_results/ablation_comparison.png
    # ============================================
    results_df = pd.read_csv("ablation_results/ablation_results_from_autodl.csv")
    plot_f1_comparison(results_df)
    


if __name__ == "__main__":
    main()