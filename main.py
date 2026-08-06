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

from extract_subset import extract_subset
from split_modality import split_modality
from train import train_model
from test_model import test_model


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
    # from data_cleaning import main as run_data_cleaning
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
    # 步骤4：模型训练
    # 输入：split_data/dataset_X/split_Y/train.npz
    # 输出：saved_models/model_Z/
    # 参数：
    #   model_path: LLM模型路径（默认./models/qwen2.5-1.5b）
    #   model_id: 模型ID（默认自动递增）
    #   dataset_id: 数据集ID（默认0）
    #   split_id: 划分ID（默认0）
    #   per_device_train_batch_size: 每个设备的batch大小（默认2）
    #   gradient_accumulation_steps: 梯度累积步数（默认4）
    #   learning_rate: 学习率（默认1e-4）
    #   num_train_epochs: 训练轮数（默认3）
    # ============================================
    # try:
    #     train_model(
    #         model_path="./models/qwen2.5-1.5b",
    #         # model_id=0,  # 可选：指定模型ID
    #         dataset_id=0,
    #         split_id=0,
    #         per_device_train_batch_size=2,
    #         gradient_accumulation_steps=4,
    #         learning_rate=1e-4,
    #         num_train_epochs=3
    #     )
    #     print("训练完成")
    # except Exception as e:
    #     print(f"训练出错：{e}")
    
    # ============================================
    # 步骤5：模型测试
    # 输入：split_data/dataset_X/split_Y/test.npz + saved_models/model_Z/
    # 输出：test_reports/report_W.json
    # 参数：
    #   model_id: 模型ID（与--saved_model_path二选一）
    #   saved_model_path: 模型路径（与--model_id二选一）
    #   dataset_id: 数据集ID（默认0）
    #   split_id: 划分ID（默认0）
    #   verbose: 是否打印详细信息（默认True）
    # ============================================
    # try:
    #     result = test_model(
    #         model_id=0,
    #         dataset_id=0,
    #         split_id=0,
    #         verbose=True
    #     )
    # except Exception as e:
    #     print(f"测试出错：{e}")
    
    # ============================================
    # 步骤6：绘制loss曲线（训练完成后执行）
    # 参数：
    #   model_id: 模型ID
    # ============================================
    # plot_loss_curve(model_id=0)


if __name__ == "__main__":
    main()