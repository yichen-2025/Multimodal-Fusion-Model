import torch
import sys
import os
import time
import gc
import argparse
import csv
import numpy as np
import pandas as pd
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, PROJECT_ROOT)

from scripts.train import train_model, VARIANT_CONFIGS, VARIANT_DESCRIPTIONS, get_next_model_id
from scripts.test_model import test_model


def _free_gpu(label=""):
    """
    强制释放 GPU 显存（Windows WDDM 模式下 empty_cache 不可靠，
    必须先把模型显式 .to('cpu') 再 del，确保张量引用归零）。
    
    Args:
        label (str): 调试标签，打印时显示释放了多少显存
    """
    if not torch.cuda.is_available():
        return
    before = torch.cuda.memory_reserved() / 1024 ** 2
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    after = torch.cuda.memory_reserved() / 1024 ** 2
    freed = before - after
    if abs(freed) > 1:  # 有实质变化才打印
        tag = f"[{label}] " if label else ""
        print(f"  {tag}显存 {before:.0f} → {after:.0f} MB (释放 {freed:.0f} MB)")


def run_ablation(variants=None, 
                 dataset_id=0, 
                 split_id=0, 
                 model_path=None,
                 output_csv=None,
                 repeat=1,
                 seed=42):
    """
    运行消融实验
    
    Args:
        variants (list): 变体列表，默认 ["A0", "A1", "A2", "A3"]
        dataset_id (int): 数据集ID
        split_id (int): 划分ID
        model_path (str): LLM模型路径
        output_csv (str): 输出CSV文件路径
        repeat (int): 重复次数
        seed (int): 随机种子
        
    Returns:
        pd.DataFrame: 消融实验结果表
    """
    if variants is None:
        variants = ["A0", "A1", "A2", "A3"]
    
    if model_path is None:
        model_path = os.path.join(PROJECT_ROOT, "models", "qwen2.5-1.5b")
    
    if output_csv is None:
        output_dir = os.path.join(PROJECT_ROOT, "ablation_results")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = os.path.join(output_dir, f"ablation_results_{timestamp}.csv")
    
    print("=" * 80)
    print("消融实验运行器")
    print("=" * 80)
    print(f"\n实验配置:")
    print(f"  - 变体列表: {variants}")
    print(f"  - 数据集: dataset_{dataset_id}/split_{split_id}")
    print(f"  - 重复次数: {repeat}")
    print(f"  - 随机种子: {seed}")
    print(f"  - 输出文件: {output_csv}")
    if torch.cuda.is_available():
        total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"  - GPU: {torch.cuda.get_device_name(0)} ({total_mb:.0f} MB)")
    print("=" * 80)
    
    all_results = []
    prev_model = None  # 保存上一个变体的 model 引用用于释放
    
    for variant in variants:
        if variant not in VARIANT_CONFIGS:
            print(f"\n跳过未知变体: {variant}")
            continue

        # ── 释放上一个变体遗留的 GPU 显存 ──
        # 关键修复（Windows WDDM 6GB 显存）：
        # 旧代码：只 gc.collect()+empty_cache()，WDDM 下不可靠。
        # 新代码：先显式 model.to('cpu') 把 LLM 权重搬离 GPU，
        # 再 del + gc + empty_cache，确保碎片也被释放。
        if prev_model is not None:
            try:
                prev_model.to('cpu')
            except Exception:
                pass
            del prev_model
            prev_model = None
        _free_gpu(f"变体切换→{variant}")

        config = VARIANT_CONFIGS[variant]
        description = VARIANT_DESCRIPTIONS.get(variant, "未知")
        
        for rep in range(repeat):
            current_seed = seed + rep * 1000
            
            print(f"\n{'=' * 80}")
            print(f"运行变体 {variant}: {description} (第 {rep+1}/{repeat} 次)")
            print(f"{'=' * 80}")
            
            try:
                # 1. 训练
                print(f"\n[1/2] 开始训练...")
                model_id = get_next_model_id()
                save_path = os.path.join(PROJECT_ROOT, "saved_models", f"model_{model_id}")
                
                train_start = time.time()
                model = train_model(
                    model_path=model_path,
                    model_id=model_id,
                    dataset_id=dataset_id,
                    split_id=split_id,
                    variant=variant,
                    seed=current_seed
                )
                train_duration = time.time() - train_start
                
                # 获取可训练参数量
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                print(f"\n训练完成: {train_duration:.1f}秒, 可训练参数: {trainable_params:,}")
                
                # ⚠️ 关键：在调 test_model 前，先把 train_model 返回的 model 的
                # LLM 从 GPU 搬去 CPU！否则 test_model.from_pretrained() 又加载
                # 一个新 LLM 到 GPU，两份 Qwen (~6GB) 会把 6GB 显卡挤爆 → 
                # 触发 OOM fallback → 训练/测试速度骤降到 CPU 级别。
                if torch.cuda.is_available():
                    model.to('cpu')
                    _free_gpu("训练后→测试前（释放训练模型 LLM）")
                
                # 保存引用，供变体切换时再次释放
                prev_model = model
                
                # 2. 测试
                print(f"\n[2/2] 开始测试...")
                test_start = time.time()
                results = test_model(
                    dataset_id=dataset_id,
                    split_id=split_id,
                    model_id=model_id,
                    llm_model_path=model_path,
                    variant=variant,
                    llm_use_lora=config['llm_use_lora'],
                    save_report=True
                )
                test_duration = time.time() - test_start
                
                # 测试完立即释放 test_model 内部加载的 LLM
                _free_gpu("测试后（释放 test_model 内部 LLM）")
                
                # 3. 收集结果
                result_row = {
                    'variant': variant,
                    'description': description,
                    'use_numeric': config['use_numeric'],
                    'use_bert': config['use_bert'],
                    'use_llm': config['use_llm'],
                    'fusion_type': config['fusion_type'],
                    'bert_trainable': config['bert_trainable'],
                    'llm_use_lora': config['llm_use_lora'],
                    'accuracy': results['accuracy'],
                    'precision': results['precision'],
                    'recall': results['recall'],
                    'f1': results['f1'],
                    'train_seconds': train_duration,
                    'test_seconds': test_duration,
                    'trainable_params': trainable_params,
                    'seed': current_seed,
                    'model_id': model_id,
                    'report_id': results.get('report_id', None),
                    'timestamp': datetime.now().isoformat()
                }
                
                all_results.append(result_row)
                
                print(f"\n结果汇总:")
                print(f"  - Accuracy:  {results['accuracy']:.4f}")
                print(f"  - Precision: {results['precision']:.4f}")
                print(f"  - Recall:    {results['recall']:.4f}")
                print(f"  - F1 Score:  {results['f1']:.4f}")
                print(f"  - 训练耗时:  {train_duration:.1f}秒")
                print(f"  - 测试耗时:  {test_duration:.1f}秒")
                
            except Exception as e:
                print(f"\n变体 {variant} 运行失败: {e}")
                import traceback
                traceback.print_exc()
                
                # 记录失败结果
                all_results.append({
                    'variant': variant,
                    'description': description,
                    'use_numeric': config['use_numeric'],
                    'use_bert': config['use_bert'],
                    'use_llm': config['use_llm'],
                    'fusion_type': config['fusion_type'],
                    'bert_trainable': config['bert_trainable'],
                    'llm_use_lora': config['llm_use_lora'],
                    'accuracy': None,
                    'precision': None,
                    'recall': None,
                    'f1': None,
                    'train_seconds': None,
                    'test_seconds': None,
                    'trainable_params': None,
                    'seed': current_seed,
                    'model_id': None,
                    'report_id': None,
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                })
                continue
    
    # 保存结果到CSV
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n{'=' * 80}")
        print(f"消融实验完成！")
        print(f"结果已保存到: {output_csv}")
        print(f"{'=' * 80}")
        
        # 打印汇总表
        print("\n消融实验结果汇总:")
        print("=" * 80)
        
        # 如果有重复实验，计算均值
        if repeat > 1:
            summary = df.groupby('variant').agg({
                'accuracy': ['mean', 'std'],
                'precision': ['mean', 'std'],
                'recall': ['mean', 'std'],
                'f1': ['mean', 'std'],
                'train_seconds': ['mean']
            }).round(4)
            print(summary.to_string())
        else:
            display_cols = ['variant', 'description', 'accuracy', 'precision', 'recall', 'f1', 'train_seconds']
            print(df[display_cols].to_string(index=False))
        
        print("=" * 80)
        
        return df
    
    return None


def plot_f1_comparison(results_df, output_dir=None):
    """
    生成F1对比柱状图
    
    Args:
        results_df (pd.DataFrame): 结果数据
        output_dir (str): 输出目录
    """
    try:
        import matplotlib.pyplot as plt
        
        
        if output_dir is None:
            output_dir = os.path.join(PROJECT_ROOT, "ablation_results")
            os.makedirs(output_dir, exist_ok=True)
        
        # 按变体分组计算均值
        if 'seed' in results_df.columns and len(results_df) > len(results_df['variant'].unique()):
            agg_df = results_df.groupby('variant').agg({
                'f1': 'mean',
                'accuracy': 'mean',
                'precision': 'mean',
                'recall': 'mean'
            }).reset_index()
        else:
            agg_df = results_df
        
        # 创建柱状图
        fig, ax = plt.subplots(figsize=(12, 6))
        
        variants = agg_df['variant'].tolist()
        x = np.arange(len(variants))
        width = 0.2
        
        ax.bar(x - width*1.5, agg_df['accuracy'], width, label='Accuracy')
        ax.bar(x - width*0.5, agg_df['precision'], width, label='Precision')
        ax.bar(x + width*0.5, agg_df['recall'], width, label='Recall')
        ax.bar(x + width*1.5, agg_df['f1'], width, label='F1')
        
        ax.set_xlabel('Variant')
        ax.set_ylabel('Score')
        ax.set_title('Ablation Experiment Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(variants)
        ax.legend()
        
        # 根据数据动态调整Y轴范围，从0开始以完整展示所有指标
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(output_dir, f"ablation_comparison_{timestamp}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        print(f"\n对比图已保存到: {plot_path}")
        
    except ImportError:
        print("\n警告: 未安装matplotlib，跳过生成图表。")
        print("请运行: pip install matplotlib")


def main():
    parser = argparse.ArgumentParser(description="消融实验运行器")
    parser.add_argument("--variants", nargs="+", default=["A0", "A1", "A2", "A3"],
                       help="变体列表，如 A0 A1 A2 A3")
    parser.add_argument("--dataset_id", type=int, default=0, help="数据集ID")
    parser.add_argument("--split_id", type=int, default=0, help="划分ID")
    parser.add_argument("--model_path", type=str, default=None, help="LLM模型路径")
    parser.add_argument("--output_csv", type=str, default=None, help="输出CSV路径")
    parser.add_argument("--repeat", type=int, default=1, help="重复次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--plot", action="store_true", help="生成F1对比图")
    args = parser.parse_args()
    
    results_df = run_ablation(
        variants=args.variants,
        dataset_id=args.dataset_id,
        split_id=args.split_id,
        model_path=args.model_path,
        output_csv=args.output_csv,
        repeat=args.repeat,
        seed=args.seed
    )
    
    if results_df is not None and args.plot:
        plot_f1_comparison(results_df)


if __name__ == "__main__":
    main()
