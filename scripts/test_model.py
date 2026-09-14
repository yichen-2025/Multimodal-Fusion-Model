import torch
import sys
import argparse
import numpy as np
import os
import json
import time
from datetime import datetime
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from transformers import AutoTokenizer
from src.model_architectures.multi_modal_model import MultiModalFusionModel
from src.data.data_loader import load_real_data, generate_mock_data, load_split_data
from utils.open_set_eval import evaluate_open_set  # B8：统一评估口径

REPORTS_DIR = os.path.join(PROJECT_ROOT, "test_reports")
INDEX_FILE = os.path.join(REPORTS_DIR, "reports_index.csv")


def evaluate_model(model, dataset, tokenizer=None, device=None):
    """
    在测试集上评估模型性能
    
    Args:
        model: 训练好的多模态融合模型
        dataset: 测试数据集
        tokenizer: LLM的tokenizer（无LLM时可为None）
        device: 运行设备
        
    Returns:
        dict: 评估指标字典
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    if device is None:
        device = model.device
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            stat_vector = torch.tensor(sample["stat"], dtype=torch.float32).unsqueeze(0).to(device)
            bert_tensor = torch.tensor(sample["bert"], dtype=torch.float32).unsqueeze(0).to(device)
            label = sample["label"]
            
            target_dtype = next(model.fusion_projection.parameters()).dtype
            stat_vector = stat_vector.to(dtype=target_dtype)
            bert_tensor = bert_tensor.to(dtype=target_dtype)
            
            if model.use_llm and tokenizer is not None and "text" in sample:
                # LLM 分支：对真实文本做 tokenize，传 input_ids 和 attention_mask
                text = sample["text"] if sample["text"] else ""
                encodings = tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )
                input_ids = encodings["input_ids"].to(device)
                attention_mask = encodings["attention_mask"].to(device)
                outputs = model(stat_vector, bert_tensor, input_ids=input_ids, attention_mask=attention_mask)
            else:
                # 无 LLM 或无 tokenizer：只传 stat 和 bert
                outputs = model(stat_vector, bert_tensor)
            
            logits = outputs["logits"]
            pred = torch.argmax(logits, dim=1).item()
            
            all_preds.append(pred)
            all_labels.append(label)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
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
    """
    print("=" * 60)
    print("模型评估结果")
    print("=" * 60)
    print(f"\n准确率 (Accuracy): {results['accuracy']:.4f}")
    print(f"宏平均精确率 (Macro-Precision): {results['precision']:.4f}")
    print(f"宏平均召回率 (Macro-Recall): {results['recall']:.4f}")
    print(f"宏平均F1 (Macro-F1): {results['f1']:.4f}")
    
    print("\n混淆矩阵 (Confusion Matrix):")
    cm = results['confusion_matrix']
    print(cm)
    
    unique_labels = sorted(set(results['labels']))
    if len(unique_labels) <= 2:
        if len(unique_labels) == 2:
            tp = cm[1][1]
            tn = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            print(f"\n二分类详情:")
            print(f"  - 真阳性 (TP): {tp}")
            print(f"  - 真阴性 (TN): {tn}")
            print(f"  - 假阳性 (FP): {fp}")
            print(f"  - 假阴性 (FN): {fn}")
    else:
        print(f"\n多分类详情 ({len(unique_labels)}类):")
        for c in unique_labels:
            if c < len(cm) and cm[c].sum() > 0:
                correct = cm[c][c] if c < len(cm) and c < len(cm[c]) else 0
                total = cm[c].sum()
                print(f"  - 类{c}: {correct}/{total} = {correct/total:.4f}")
    
    print("\n" + "=" * 60)


def get_model_path(model_id, base_dir=None):
    """根据模型ID获取模型路径"""
    if base_dir is None:
        base_dir = os.path.join(PROJECT_ROOT, "saved_models")
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
        'duration_seconds': report_data.get('duration_seconds', ''),
        'variant': report_data.get('variant', ''),
        'use_numeric': report_data.get('use_numeric', ''),
        'use_bert': report_data.get('use_bert', ''),
        'use_llm': report_data.get('use_llm', ''),
        'fusion_type': report_data.get('fusion_type', ''),
        'bert_trainable': report_data.get('bert_trainable', ''),
        'trainable_params': report_data.get('trainable_params', ''),
        'llm_use_lora': report_data.get('llm_use_lora', ''),
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
                                           'tp', 'tn', 'fp', 'fn', 'duration_seconds',
                                           'variant', 'use_numeric', 'use_bert', 'use_llm',
                                           'fusion_type', 'bert_trainable', 'trainable_params', 'llm_use_lora'])
        df = pd.concat([df, pd.DataFrame([csv_row])], ignore_index=True)
    else:
        df = pd.DataFrame([csv_row])
    
    temp_file = INDEX_FILE + ".tmp"
    df.to_csv(temp_file, index=False)
    os.replace(temp_file, INDEX_FILE)
    
    return report_id


def test_model(dataset_id=0, split_id=0, model_id=0, 
               llm_model_path=None, 
               variant=None,
               llm_use_lora=None,
               verbose=True,
               save_report=True):
    """
    使用指定的数据集划分和训练参数测试模型
    """
    if llm_model_path is None:
        llm_model_path = os.path.join(PROJECT_ROOT, "models", "qwen2.5-1.5b")
    
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
        llm_model_path_or_save_dir=llm_model_path,
        save_dir=saved_model_path
    )
    
    device = model.device
    
    if verbose:
        print(f"模型设备: {device}")
        print(f"模型配置: use_numeric={model.use_numeric}, use_bert={model.use_bert}, use_llm={model.use_llm}")
    
    # 无LLM时不需要tokenizer
    tokenizer = None
    if model.use_llm:
        tokenizer = AutoTokenizer.from_pretrained(llm_model_path, local_files_only=True)
    
    if verbose:
        print("\n" + "=" * 60)
        print("加载测试数据")
        print("=" * 60)
    
    test_dataset = load_split_data(data_dir=os.path.join(PROJECT_ROOT, "split_data"), data_type="test",
                                   dataset_id=dataset_id, split_id=split_id)
    
    if test_dataset is None:
        raise FileNotFoundError(f"未找到划分后的测试数据: split_data/dataset_{dataset_id}/split_{split_id}/")
    
    test_samples = len(test_dataset)
    test_labels = [sample['label'] for sample in test_dataset]
    
    if verbose:
        print(f"测试样本数: {test_samples}")
        for lbl, cnt in zip(*np.unique(test_labels, return_counts=True)):
            print(f"  - 类{lbl}: {cnt}样本")
    
    if verbose:
        print("\n" + "=" * 60)
        print("开始评估...")
        print("=" * 60)
    
    results = evaluate_model(model, test_dataset, tokenizer, device)
    
    duration_seconds = time.time() - start_time
    
    cm = results['confusion_matrix']
    num_classes = len(cm)
    
    if verbose:
        print_evaluation_results(results)
        print(f"\n测试耗时: {duration_seconds:.2f}秒")
        print(f"测试完成！")
    
    # 计算可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
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
            'num_classes': num_classes,
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'duration_seconds': duration_seconds,
            'device': str(device),
            'model_config': {
                'use_numeric': model.use_numeric,
                'use_bert': model.use_bert,
                'use_llm': model.use_llm,
                'fusion_type': model.fusion_type,
                'bert_trainable': model.bert_trainable,
                'llm_use_lora': getattr(model, 'llm_use_lora', llm_use_lora)
            },
            'variant': variant,
            'llm_use_lora': getattr(model, 'llm_use_lora', llm_use_lora),
            'use_numeric': model.use_numeric,
            'use_bert': model.use_bert,
            'use_llm': model.use_llm,
            'fusion_type': model.fusion_type,
            'bert_trainable': model.bert_trainable,
            'trainable_params': trainable_params
        }
        
        report_id = save_test_report(report_data)
        results['report_id'] = report_id
        
        if verbose:
            print(f"\n测试报告已保存:")
            print(f"  - 报告ID: {report_id}")
            print(f"  - 索引文件: {INDEX_FILE}")
            print(f"  - 详细报告: {os.path.join(REPORTS_DIR, f'report_{report_id}.json')}")
    
    return results


def evaluate_open_set_model(model, dataset, tokenizer=None, device=None, num_known_classes=2):
    """
    开集三分类评估（含unknown类）

    Args:
        model: 训练好的模型
        dataset: 测试数据集（标签含0,1,...,num_known_classes-1为已知，>=num_known_classes为未知）
        tokenizer: LLM tokenizer
        device: 运行设备
        num_known_classes: 已知类别数

    Returns:
        dict: 开集评估指标
    """
    model.eval()
    all_preds = []
    all_labels = []

    if device is None:
        device = model.device

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            stat_vector = torch.tensor(sample["stat"], dtype=torch.float32).unsqueeze(0).to(device)
            bert_tensor = torch.tensor(sample["bert"], dtype=torch.float32).unsqueeze(0).to(device)
            label = sample["label"]

            if model.use_llm and tokenizer is not None and "text" in sample:
                text = sample["text"] if sample["text"] else ""
                encodings = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
                input_ids = encodings["input_ids"].to(device)
                attention_mask = encodings["attention_mask"].to(device)
                outputs = model(stat_vector, bert_tensor, input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = model(stat_vector, bert_tensor)

            logits = outputs["logits"]
            pred = torch.argmax(logits, dim=1).item()

            all_preds.append(pred)
            all_labels.append(label)

    true_arr = np.array(all_labels)
    pred_arr = np.array(all_preds)

    # B8：评估逻辑已抽到 utils.open_set_eval，这里统一调用
    results = evaluate_open_set(
        true_arr, pred_arr, num_known_classes,
        include_routing_stats=False  # test_model 不需要路由统计字段
    )

    # 保留模型原始输出供下游分析
    results['predictions'] = all_preds
    results['labels'] = all_labels

    return results


def print_open_set_results(results):
    """打印开集评估结果"""
    print("=" * 60)
    print("开集评估结果")
    print("=" * 60)
    print(f"\n整体准确率: {results['accuracy']:.4f}")
    print(f"Macro-F1: {results['macro_f1']:.4f}")

    label_names = {
        0: "BENIGN", 1: "DoS Hulk", 2: "DoS GoldenEye", 3: "DoS slowloris",
        4: "DoS Slowhttptest", 5: "DDoS", 6: "PortScan", 7: "FTP-Patator",
        8: "SSH-Patator", 9: "Bot", 10: "Web Attack - Brute Force",
        11: "Web Attack - XSS", 12: "Web Attack - Sql Injection",
        13: "Infiltration", 14: "Heartbleed"
    }

    if 'known_accuracy' in results:
        print(f"\n已知类准确率: {results['known_accuracy']:.4f}")
        for key, val in results.items():
            if key.endswith('_recall') and key.startswith('class_'):
                c = int(key.replace('class_', '').replace('_recall', ''))
                name = label_names.get(c, f"class_{c}")
                print(f"  {name}({c}) 召回率: {val:.4f}")

    if 'unknown_recall' in results:
        print(f"\n未知类召回率: {results['unknown_recall']:.4f}")
        print(f"未知类F1: {results['unknown_f1']:.4f}")
        print(f"未知类泄漏率: {results['unknown_leak_rate']:.4f}")

    print("\n混淆矩阵:")
    cm = np.array(results['confusion_matrix'])
    print(cm)
    print("\n" + "=" * 60)


def main():
    """
    测试主函数（命令行入口）
    """
    parser = argparse.ArgumentParser(description="测试多模态融合模型（支持消融实验）")
    parser.add_argument("--model_path", type=str, default=None, help="LLM模型路径")
    parser.add_argument("--model_id", type=int, default=None, help="模型ID（用于从saved_models加载）")
    parser.add_argument("--saved_model_path", type=str, default=None, help="训练后模型参数路径")
    parser.add_argument("--dataset_id", type=int, default=0, help="数据集ID")
    parser.add_argument("--split_id", type=int, default=0, help="划分ID")
    parser.add_argument("--openset", action="store_true", help="加载开集划分（split_openset）")
    parser.add_argument("--no_save_report", action="store_true", help="不保存测试报告")
    parser.add_argument("--variant", type=str, default=None, help="变体名称（如 A0*, A0_frozen, A3），写入报告用于区分")
    args = parser.parse_args()

    if args.model_path is None:
        MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "qwen2.5-1.5b")
    else:
        MODEL_PATH = args.model_path
    
    if args.saved_model_path is not None:
        SAVED_MODEL_PATH = args.saved_model_path
        model_id = None
    elif args.model_id is not None:
        SAVED_MODEL_PATH = get_model_path(args.model_id)
        model_id = args.model_id
    else:
        SAVED_MODEL_PATH = os.path.join(PROJECT_ROOT, "saved_model")
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
    
    device = model.device
    print(f"模型设备: {device}")
    print(f"模型配置: use_numeric={model.use_numeric}, use_bert={model.use_bert}, use_llm={model.use_llm}")
    
    tokenizer = None
    if model.use_llm:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    
    print("\n" + "=" * 60)
    print("加载测试数据")
    print("=" * 60)
    
    test_dataset = load_split_data(data_dir=os.path.join(PROJECT_ROOT, "split_data"), data_type="test",
                                   dataset_id=args.dataset_id, split_id=args.split_id,
                                   openset=args.openset)
    if test_dataset is None:
        print("未找到划分后的测试数据，使用完整数据进行测试...")
        test_dataset = load_real_data(data_dir=os.path.join(PROJECT_ROOT, "processed_data"))
        if test_dataset is None:
            print("未找到真实数据，使用模拟数据进行测试...")
            test_dataset = generate_mock_data(100)
    
    test_samples = len(test_dataset)
    test_labels = [sample['label'] for sample in test_dataset]
    
    print(f"测试样本数: {test_samples}")
    for lbl, cnt in zip(*np.unique(test_labels, return_counts=True)):
        print(f"  - 类{lbl}: {cnt}样本")
    
    print("\n" + "=" * 60)
    print("开始评估...")
    print("=" * 60)
    
    results = evaluate_model(model, test_dataset, tokenizer, device)
    
    duration_seconds = time.time() - start_time
    
    cm = results['confusion_matrix']
    
    print_evaluation_results(results)
    print(f"\n测试耗时: {duration_seconds:.2f}秒")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
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
            'num_classes': len(set(test_labels)),
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'duration_seconds': duration_seconds,
            'device': str(device),
            'model_config': {
                'use_numeric': model.use_numeric,
                'use_bert': model.use_bert,
                'use_llm': model.use_llm,
                'fusion_type': model.fusion_type,
                'bert_trainable': model.bert_trainable
            },
            'variant': args.variant,
            'use_numeric': model.use_numeric,
            'use_bert': model.use_bert,
            'use_llm': model.use_llm,
            'fusion_type': model.fusion_type,
            'bert_trainable': model.bert_trainable,
            'llm_use_lora': getattr(model, 'llm_use_lora', None),
            'trainable_params': trainable_params
        }
        
        report_id = save_test_report(report_data)
        
        print(f"\n测试报告已保存:")
        print(f"  - 报告ID: {report_id}")
        print(f"  - 索引文件: {INDEX_FILE}")
        print(f"  - 详细报告: {os.path.join(REPORTS_DIR, f'report_{report_id}.json')}")
    
    print(f"\n测试完成！")


if __name__ == "__main__":
    main()
