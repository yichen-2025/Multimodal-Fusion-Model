import torch
import sys
import argparse
import numpy as np
import os
import json
import time
from datetime import datetime
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import AutoTokenizer
from src.model_architectures.multi_modal_model import MultiModalFusionModel
from src.data.data_loader import load_real_data, generate_mock_data, load_split_data

REPORTS_DIR = "./test_reports"
INDEX_FILE = os.path.join(REPORTS_DIR, "reports_index.csv")


def evaluate_model(model, dataset, tokenizer, device):
    """
    在测试集上评估模型性能
    
    Args:
        model (MultiModalFusionModel): 训练好的多模态融合模型
        dataset (datasets.Dataset): 测试数据集
        tokenizer: LLM的tokenizer
        device (torch.device): 运行设备
        
    Returns:
        dict: 评估指标字典
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            stat_vector = torch.tensor(sample["stat"], dtype=torch.float32).unsqueeze(0).to(device)
            bert_tensor = torch.tensor(sample["bert"], dtype=torch.float32).unsqueeze(0).to(device)
            label = sample["label"]
            
            target_dtype = next(model.fusion_projection.parameters()).dtype
            stat_vector = stat_vector.to(dtype=target_dtype)
            bert_tensor = bert_tensor.to(dtype=target_dtype)
            
            outputs = model(stat_vector, bert_tensor)
            logits = outputs["logits"]
            pred = torch.argmax(logits, dim=1).item()
            
            all_preds.append(pred)
            all_labels.append(label)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels
    }


def print_evaluation_results(results):
    """
    打印评估结果
    
    Args:
        results (dict): 评估指标字典
    """
    print("=" * 60)
    print("模型评估结果")
    print("=" * 60)
    print(f"\n准确率 (Accuracy): {results['accuracy']:.4f}")
    print(f"精确率 (Precision): {results['precision']:.4f}")
    print(f"召回率 (Recall): {results['recall']:.4f}")
    print(f"F1分数 (F1 Score): {results['f1']:.4f}")
    
    print("\n混淆矩阵 (Confusion Matrix):")
    cm = results['confusion_matrix']
    print(f"              预测正常  预测恶意")
    print(f"实际正常      {cm[0][0]:>8d}    {cm[0][1]:>8d}")
    print(f"实际恶意      {cm[1][0]:>8d}    {cm[1][1]:>8d}")
    
    tp = cm[1][1]
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    
    print(f"\n分类详情:")
    print(f"  - 真阳性 (TP): {tp} (实际恶意，预测恶意)")
    print(f"  - 真阴性 (TN): {tn} (实际正常，预测正常)")
    print(f"  - 假阳性 (FP): {fp} (实际正常，预测恶意)")
    print(f"  - 假阴性 (FN): {fn} (实际恶意，预测正常)")
    
    print("\n" + "=" * 60)


def get_model_path(model_id, base_dir="./saved_models"):
    """根据模型ID获取模型路径"""
    return os.path.join(base_dir, f"model_{model_id}")


def get_next_report_id():
    """获取下一个可用的报告ID（自动递增）"""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    if not os.path.exists(INDEX_FILE):
        return 0
    
    try:
        df = pd.read_csv(INDEX_FILE)
        if 'report_id' in df.columns:
            max_id = df['report_id'].max()
            if pd.isna(max_id):
                return 0
            return int(max_id) + 1
    except Exception:
        pass
    
    return 0


def save_test_report(report_data):
    """
    保存测试报告
    
    功能：将测试报告保存到test_reports目录，包括：
    1. 详细报告保存为JSON文件（report_{report_id}.json）
    2. 索引信息保存到CSV文件（reports_index.csv）
    
    Args:
        report_data (dict): 测试报告数据字典
            必须包含字段：
                - report_id: int
                - timestamp: str (ISO格式)
                - model_id: int
                - dataset_id: int
                - split_id: int
                - test_samples: int
                - test_positive: int
                - test_negative: int
                - accuracy: float
                - precision: float
                - recall: float
                - f1: float
                - tp: int
                - tn: int
                - fp: int
                - fn: int
                - duration_seconds: float
            可选字段（用于扩展）：
                - model_path: str
                - llm_model_path: str
                - device: str
                - model_config: dict
                - ... (可任意扩展)
    
    Returns:
        int: 报告ID
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    report_id = report_data.get('report_id', get_next_report_id())
    report_data['report_id'] = report_id
    
    json_path = os.path.join(REPORTS_DIR, f"report_{report_id}.json")
    
    save_data = report_data.copy()
    if 'confusion_matrix' in save_data:
        cm = save_data.pop('confusion_matrix')
        if isinstance(cm, np.ndarray):
            save_data['confusion_matrix'] = cm.tolist()
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    csv_row = {
        'report_id': report_id,
        'timestamp': report_data.get('timestamp', ''),
        'model_id': report_data.get('model_id', ''),
        'dataset_id': report_data.get('dataset_id', ''),
        'split_id': report_data.get('split_id', ''),
        'test_samples': report_data.get('test_samples', ''),
        'test_positive': report_data.get('test_positive', ''),
        'test_negative': report_data.get('test_negative', ''),
        'accuracy': report_data.get('accuracy', ''),
        'precision': report_data.get('precision', ''),
        'recall': report_data.get('recall', ''),
        'f1': report_data.get('f1', ''),
        'tp': report_data.get('tp', ''),
        'tn': report_data.get('tn', ''),
        'fp': report_data.get('fp', ''),
        'fn': report_data.get('fn', ''),
        'duration_seconds': report_data.get('duration_seconds', '')
    }
    
    if os.path.exists(INDEX_FILE):
        try:
            df = pd.read_csv(INDEX_FILE)
        except PermissionError:
            temp_file = INDEX_FILE + ".tmp"
            if os.path.exists(temp_file):
                df = pd.read_csv(temp_file)
            else:
                df = pd.DataFrame(columns=['report_id', 'timestamp', 'model_id', 'dataset_id', 'split_id',
                                           'test_samples', 'test_positive', 'test_negative',
                                           'accuracy', 'precision', 'recall', 'f1',
                                           'tp', 'tn', 'fp', 'fn', 'duration_seconds'])
        df = pd.concat([df, pd.DataFrame([csv_row])], ignore_index=True)
    else:
        df = pd.DataFrame([csv_row])
    
    temp_file = INDEX_FILE + ".tmp"
    df.to_csv(temp_file, index=False)
    os.replace(temp_file, INDEX_FILE)
    
    return report_id


def test_model(dataset_id=0, split_id=0, model_id=0, 
               llm_model_path="./models/qwen2.5-1.5b", 
               verbose=True,
               save_report=True):
    """
    使用指定的数据集划分和训练参数测试模型
    
    功能：加载指定划分的测试集和指定训练参数的模型，进行评估并返回结果
    
    Args:
        dataset_id (int): 数据集ID，对应dataset_{dataset_id}.csv
        split_id (int): 划分ID，对应split_data/dataset_{dataset_id}/split_{split_id}/
        model_id (int): 模型ID，对应saved_models/model_{model_id}/
        llm_model_path (str): LLM基础模型路径
        verbose (bool): 是否打印详细信息
        save_report (bool): 是否保存测试报告
        
    Returns:
        dict: 评估指标字典，与evaluate_model返回结果相同，额外包含report_id字段
            {
                'accuracy': float,      # 准确率
                'precision': float,     # 精确率
                'recall': float,        # 召回率
                'f1': float,            # F1分数
                'confusion_matrix': np.ndarray,  # 混淆矩阵
                'predictions': list,    # 预测结果列表
                'labels': list,         # 真实标签列表
                'report_id': int        # 报告ID（如果保存了报告）
            }
    
    Example:
        >>> results = test_model(dataset_id=2, split_id=0, model_id=0)
        >>> print(results['accuracy'])
        0.95
        >>> print(results['report_id'])
        3
    """
    start_time = time.time()
    timestamp = datetime.now().isoformat()
    
    saved_model_path = get_model_path(model_id)
    
    if verbose:
        print(f"\n测试参数:")
        print(f"  - 数据集ID: {dataset_id}")
        print(f"  - 划分ID: {split_id}")
        print(f"  - 模型ID: {model_id}")
        print(f"  - 模型保存路径: {saved_model_path}")

    saved_model_exists = os.path.exists(os.path.join(saved_model_path, 'pytorch_model.bin'))
    
    if not saved_model_exists:
        raise FileNotFoundError(f"未找到训练后模型参数: {saved_model_path}")
    
    if verbose:
        print("=" * 60)
        print("加载训练后模型")
        print("=" * 60)
    
    model = MultiModalFusionModel.from_pretrained(
        llm_model_path=llm_model_path,
        save_dir=saved_model_path
    )
    
    device = next(model.llm.parameters()).device
    
    if verbose:
        print(f"模型设备: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path, local_files_only=True)
    
    if verbose:
        print("\n" + "=" * 60)
        print("加载测试数据")
        print("=" * 60)
    
    test_dataset = load_split_data(data_dir="split_data", data_type="test",
                                   dataset_id=dataset_id, split_id=split_id)
    
    if test_dataset is None:
        raise FileNotFoundError(f"未找到划分后的测试数据: split_data/dataset_{dataset_id}/split_{split_id}/")
    
    test_samples = len(test_dataset)
    test_labels = [sample['label'] for sample in test_dataset]
    test_positive = sum(1 for l in test_labels if l == 1)
    test_negative = test_samples - test_positive
    
    if verbose:
        print(f"测试样本数: {test_samples}")
        print(f"  - 恶意流量(1): {test_positive}")
        print(f"  - 正常流量(0): {test_negative}")
    
    if verbose:
        print("\n" + "=" * 60)
        print("开始评估...")
        print("=" * 60)
    
    results = evaluate_model(model, test_dataset, tokenizer, device)
    
    duration_seconds = time.time() - start_time
    
    cm = results['confusion_matrix']
    tp = int(cm[1][1])
    tn = int(cm[0][0])
    fp = int(cm[0][1])
    fn = int(cm[1][0])
    
    if verbose:
        print_evaluation_results(results)
        print(f"\n测试耗时: {duration_seconds:.2f}秒")
        print(f"测试完成！")
    
    if save_report:
        report_data = {
            'report_id': get_next_report_id(),
            'timestamp': timestamp,
            'model_id': model_id,
            'model_path': saved_model_path,
            'llm_model_path': llm_model_path,
            'dataset_id': dataset_id,
            'split_id': split_id,
            'test_samples': test_samples,
            'test_positive': test_positive,
            'test_negative': test_negative,
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'duration_seconds': duration_seconds,
            'device': str(device),
            'model_config': {}
        }
        
        report_id = save_test_report(report_data)
        results['report_id'] = report_id
        
        if verbose:
            print(f"\n测试报告已保存:")
            print(f"  - 报告ID: {report_id}")
            print(f"  - 索引文件: {INDEX_FILE}")
            print(f"  - 详细报告: {os.path.join(REPORTS_DIR, f'report_{report_id}.json')}")
    
    return results


def main():
    """
    测试主函数（命令行入口）
    
    功能：通过命令行参数指定数据集划分和模型，加载测试集进行评估
    """
    parser = argparse.ArgumentParser(description="测试多模态融合模型")
    parser.add_argument("--model_path", type=str, default="./models/qwen2.5-1.5b", help="LLM模型路径")
    parser.add_argument("--model_id", type=int, default=None, help="模型ID（用于从saved_models加载）")
    parser.add_argument("--saved_model_path", type=str, default=None, help="训练后模型参数路径（与model_id二选一）")
    parser.add_argument("--dataset_id", type=int, default=0, help="数据集ID")
    parser.add_argument("--split_id", type=int, default=0, help="划分ID")
    parser.add_argument("--no_save_report", action="store_true", help="不保存测试报告")
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    
    if args.saved_model_path is not None:
        SAVED_MODEL_PATH = args.saved_model_path
        model_id = None
    elif args.model_id is not None:
        SAVED_MODEL_PATH = get_model_path(args.model_id)
        model_id = args.model_id
    else:
        SAVED_MODEL_PATH = "./saved_model"
        model_id = None
    
    print(f"\n测试参数:")
    print(f"  - 模型ID: {model_id}")
    print(f"  - 数据集ID: {args.dataset_id}")
    print(f"  - 划分ID: {args.split_id}")
    print(f"  - 模型保存路径: {SAVED_MODEL_PATH}")

    saved_model_exists = os.path.exists(os.path.join(SAVED_MODEL_PATH, 'pytorch_model.bin'))
    
    if not saved_model_exists:
        print(f"错误：未找到训练后模型参数")
        print(f"请先运行 train.py 训练模型")
        return
    
    start_time = time.time()
    timestamp = datetime.now().isoformat()
    
    print("=" * 60)
    print("加载训练后模型")
    print("=" * 60)
    
    model = MultiModalFusionModel.from_pretrained(
        llm_model_path=MODEL_PATH,
        save_dir=SAVED_MODEL_PATH
    )
    
    device = next(model.llm.parameters()).device
    print(f"模型设备: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    
    print("\n" + "=" * 60)
    print("加载测试数据")
    print("=" * 60)
    
    test_dataset = load_split_data(data_dir="split_data", data_type="test",
                                   dataset_id=args.dataset_id, split_id=args.split_id)
    if test_dataset is None:
        print("未找到划分后的测试数据，使用完整数据进行测试...")
        test_dataset = load_real_data(data_dir="processed_data")
        if test_dataset is None:
            print("未找到真实数据，使用模拟数据进行测试...")
            test_dataset = generate_mock_data(100)
    
    test_samples = len(test_dataset)
    test_labels = [sample['label'] for sample in test_dataset]
    test_positive = sum(1 for l in test_labels if l == 1)
    test_negative = test_samples - test_positive
    
    print(f"测试样本数: {test_samples}")
    print(f"  - 恶意流量(1): {test_positive}")
    print(f"  - 正常流量(0): {test_negative}")
    
    print("\n" + "=" * 60)
    print("开始评估...")
    print("=" * 60)
    
    results = evaluate_model(model, test_dataset, tokenizer, device)
    
    duration_seconds = time.time() - start_time
    
    cm = results['confusion_matrix']
    tp = int(cm[1][1])
    tn = int(cm[0][0])
    fp = int(cm[0][1])
    fn = int(cm[1][0])
    
    print_evaluation_results(results)
    print(f"\n测试耗时: {duration_seconds:.2f}秒")
    
    if not args.no_save_report:
        report_data = {
            'report_id': get_next_report_id(),
            'timestamp': timestamp,
            'model_id': model_id,
            'model_path': SAVED_MODEL_PATH,
            'llm_model_path': MODEL_PATH,
            'dataset_id': args.dataset_id,
            'split_id': args.split_id,
            'test_samples': test_samples,
            'test_positive': test_positive,
            'test_negative': test_negative,
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'duration_seconds': duration_seconds,
            'device': str(device),
            'model_config': {}
        }
        
        report_id = save_test_report(report_data)
        
        print(f"\n测试报告已保存:")
        print(f"  - 报告ID: {report_id}")
        print(f"  - 索引文件: {INDEX_FILE}")
        print(f"  - 详细报告: {os.path.join(REPORTS_DIR, f'report_{report_id}.json')}")
    
    print(f"\n测试完成！")


if __name__ == "__main__":
    main()