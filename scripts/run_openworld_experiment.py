import os
import sys
import time
import json
import argparse
import numpy as np
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from utils.log_utils import save_log, get_next_log_id


VARIANT_CONFIGS = {
    "A0": {"use_numeric": True,  "use_bert": True,  "use_llm": True,  "fusion_type": "concat", "bert_trainable": False},
    "A1": {"use_numeric": True,  "use_bert": False, "use_llm": True,  "fusion_type": "concat", "bert_trainable": False},
    "A2": {"use_numeric": False, "use_bert": True,  "use_llm": True,  "fusion_type": "concat", "bert_trainable": False},
    "A3": {"use_numeric": True,  "use_bert": True,  "use_llm": False, "fusion_type": "concat", "bert_trainable": False},
}


def run_openworld_experiment(
    dataset_id=1,
    split_id=0,
    k_values=None,
    backbone_model_id=None,
    llm_model_id=None,
    model_path=None,
    backbone_variant="A3",
    llm_variant="A0",
    distance_type="cosine",
    temperature=1.0,
    seed=42,
    train_batch_size=2,
    train_grad_accum=4,
    train_lr=1e-4,
    train_epochs=3,
    ood_lr=1e-3,
    ood_epochs=10,
    ood_batch_size=64,
    eval_batch_size=32,
    skip_backbone_train=False,
    skip_ood_train=False,
    skip_llm=False,
    generate_report=True
):
    """
    运行完整的开放世界融合模型实验

    流程：
    1. 训练 A3 backbone（如果未指定已有模型ID）
    2. 对每个k值：
       a. 训练 OOD 检测头
       b. 运行 OOD 路由评估
    3. 生成汇总报告

    Args:
        dataset_id: 数据集ID
        split_id: 划分ID（对应split_openset）
        k_values: 少样本k值列表，如 [5, 10, 20, None]（None表示全量数据）
        backbone_model_id: 已有A3模型ID（跳过训练）
        llm_model_id: 已有A0模型ID（跳过训练）
        model_path: LLM模型路径
        backbone_variant: backbone变体（默认A3）
        llm_variant: LLM变体（默认A0）
        distance_type: OOD距离度量
        temperature: OOD温度系数
        seed: 随机种子
        train_batch_size: 训练batch size
        train_grad_accum: 梯度累积步数
        train_lr: 训练学习率
        train_epochs: 训练轮数
        ood_lr: OOD头学习率
        ood_epochs: OOD头训练轮数
        ood_batch_size: OOD头训练batch size
        eval_batch_size: 评估batch size
        skip_backbone_train: 跳过backbone训练
        skip_ood_train: 跳过OOD头训练
        skip_llm: 跳过LLM推理
        generate_report: 生成汇总报告

    Returns:
        dict: 完整实验结果
    """
    start_time = time.time()

    if k_values is None:
        k_values = [None]

    if model_path is None:
        model_path = os.path.join(PROJECT_ROOT, "models", "qwen2.5-1.5b")

    experiment_logs = []
    all_results = {}

    print("=" * 80)
    print("开放世界融合模型实验")
    print("=" * 80)
    print(f"  数据集: dataset_{dataset_id}")
    print(f"  开集划分: split_openset_{split_id}")
    print(f"  k值列表: {k_values}")
    print(f"  Backbone: {backbone_variant}")
    print(f"  LLM: {llm_variant} (skip={skip_llm})")
    print(f"  OOD距离: {distance_type}")
    print(f"  种子: {seed}")
    print("=" * 80)

    # ========== Stage 1: 训练 Backbone (A3) ==========
    if backbone_model_id is None and not skip_backbone_train:
        print("\n" + "=" * 80)
        print("[Stage 1] 训练 A3 Backbone 模型")
        print("=" * 80)

        from scripts.train import train_model, get_next_model_id

        backbone_model_id = get_next_model_id()
        stage_start = time.time()

        model = train_model(
            model_path=model_path,
            model_id=backbone_model_id,
            dataset_id=dataset_id,
            split_id=split_id,
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=train_grad_accum,
            learning_rate=train_lr,
            num_train_epochs=train_epochs,
            variant=backbone_variant,
            seed=seed
        )

        stage_duration = time.time() - stage_start
        experiment_logs.append({
            'stage': 'backbone_train',
            'model_id': backbone_model_id,
            'variant': backbone_variant,
            'status': 'completed',
            'duration_seconds': round(stage_duration, 3)
        })
        print(f"  Backbone训练完成: model_{backbone_model_id}, 耗时 {stage_duration:.1f}s")

    elif backbone_model_id is not None:
        print(f"\n[Stage 1] 使用已有Backbone模型: model_{backbone_model_id}")
    else:
        print(f"\n[Stage 1] 跳过Backbone训练（skip_backbone_train=True）")

    # ========== Stage 1b: 训练 LLM (A0) ==========
    if llm_model_id is None and not skip_llm:
        print("\n" + "=" * 80)
        print("[Stage 1b] 训练 A0 LLM 模型")
        print("=" * 80)

        from scripts.train import train_model, get_next_model_id

        llm_model_id = get_next_model_id()
        stage_start = time.time()

        model = train_model(
            model_path=model_path,
            model_id=llm_model_id,
            dataset_id=dataset_id,
            split_id=split_id,
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=train_grad_accum,
            learning_rate=train_lr,
            num_train_epochs=train_epochs,
            variant=llm_variant,
            seed=seed
        )

        stage_duration = time.time() - stage_start
        experiment_logs.append({
            'stage': 'llm_train',
            'model_id': llm_model_id,
            'variant': llm_variant,
            'status': 'completed',
            'duration_seconds': round(stage_duration, 3)
        })
        print(f"  LLM训练完成: model_{llm_model_id}, 耗时 {stage_duration:.1f}s")

    elif skip_llm:
        print("\n[Stage 1b] 跳过LLM训练（skip_llm=True）")
        llm_model_id = None
    else:
        print(f"\n[Stage 1b] 使用已有LLM模型: model_{llm_model_id}")

    # ========== Stage 2: 对每个k值训练OOD头 + 评估 ==========
    all_k_results = {}

    for k in k_values:
        k_label = "full" if k is None else str(k)
        print("\n" + "-" * 60)
        print(f"[Stage 2] k={k_label}")
        print("-" * 60)

        # 2a. 训练OOD头
        ood_id = None
        if not skip_ood_train:
            print(f"\n  [2a] 训练OOD检测头 (k={k_label})")
            from scripts.train_ood_head import train_ood_head

            ood_id = get_next_log_id("ood_training")
            stage_start = time.time()

            try:
                ood_head, ood_save_path = train_ood_head(
                    model_id=backbone_model_id,
                    ood_id=ood_id,
                    dataset_id=dataset_id,
                    split_id=split_id,
                    model_path=model_path,
                    variant=backbone_variant,
                    distance_type=distance_type,
                    temperature=temperature,
                    learning_rate=ood_lr,
                    num_epochs=ood_epochs,
                    batch_size=ood_batch_size,
                    seed=seed,
                    fewshot_k=k
                )

                stage_duration = time.time() - stage_start
                experiment_logs.append({
                    'stage': 'ood_train',
                    'k': k_label,
                    'ood_id': ood_id,
                    'status': 'completed',
                    'duration_seconds': round(stage_duration, 3)
                })
                print(f"    OOD训练完成: ood_{ood_id}, 耗时 {stage_duration:.1f}s")

            except Exception as e:
                print(f"    OOD训练失败: {e}")
                experiment_logs.append({
                    'stage': 'ood_train',
                    'k': k_label,
                    'ood_id': ood_id,
                    'status': f'failed: {str(e)}',
                    'duration_seconds': round(time.time() - stage_start, 3)
                })
                continue
        else:
            print(f"  [2a] 跳过OOD训练（skip_ood_train=True）")
            ood_id = get_next_log_id("ood_training")

        # 2b. 运行OOD路由评估
        print(f"\n  [2b] 运行OOD路由评估 (k={k_label}, ood_id={ood_id})")
        from scripts.run_ood_routing import run_ood_routing

        try:
            stage_start = time.time()

            eval_result = run_ood_routing(
                backbone_model_id=backbone_model_id,
                ood_id=ood_id,
                llm_model_id=llm_model_id,
                dataset_id=dataset_id,
                split_id=split_id,
                model_path=model_path,
                backbone_variant=backbone_variant,
                llm_variant=llm_variant,
                batch_size=eval_batch_size,
                verbose=True,
                compare_baseline=True,
                fewshot_k=k
            )

            stage_duration = time.time() - stage_start

            # 收集关键指标
            routed = eval_result.get('routed_results', {})
            a3_baseline = eval_result.get('a3_baseline', {})

            k_result = {
                'k': k,
                'ood_id': ood_id,
                'routed_macro_f1': routed.get('macro_f1', 0),
                'routed_unknown_f1': routed.get('unknown_f1', 0),
                'routed_unknown_recall': routed.get('unknown_recall', 0),
                'routed_accuracy': routed.get('accuracy', 0),
                'routed_per_class_recall': {
                    name: info.get('recall', 0)
                    for name, info in routed.get('per_class', {}).items()
                },
                'a3_known_accuracy': a3_baseline.get('known_accuracy', 0),
                'a3_known_macro_f1': a3_baseline.get('known_macro_f1', 0),
                'a3_unknown_leak_rate': routed.get('unknown_leak_rate', 0),
                'report_path': eval_result.get('report_path', ''),
                'duration_seconds': stage_duration,
            }

            all_k_results[k_label] = k_result

            # 记录实验日志
            experiment_logs.append({
                'stage': 'ood_routing',
                'k': k_label,
                'ood_id': ood_id,
                'status': 'completed',
                'metrics': {
                    'macro_f1': k_result['routed_macro_f1'],
                    'unknown_f1': k_result['routed_unknown_f1'],
                    'unknown_recall': k_result['routed_unknown_recall'],
                },
                'duration_seconds': round(stage_duration, 3)
            })

            print(f"    评估完成: Macro-F1={k_result['routed_macro_f1']:.4f}, "
                  f"Unknown-F1={k_result['routed_unknown_f1']:.4f}")

        except Exception as e:
            print(f"    评估失败: {e}")
            experiment_logs.append({
                'stage': 'ood_routing',
                'k': k_label,
                'ood_id': ood_id,
                'status': f'failed: {str(e)}',
                'duration_seconds': round(time.time() - stage_start, 3)
            })

    # ========== Stage 3: 生成汇总报告 ==========
    total_duration = time.time() - start_time

    print("\n" + "=" * 80)
    print("[Stage 3] 生成汇总报告")
    print("=" * 80)

    if generate_report:
        try:
            from scripts.report_generator import generate_experiment_summary
            report_paths = generate_experiment_summary()
            print(f"  报告已生成:")
            for key, path in report_paths.items():
                print(f"    {key}: {path}")
        except Exception as e:
            print(f"  报告生成失败: {e}")

    # ========== 保存实验日志 ==========
    final_log_data = {
        'dataset_id': dataset_id,
        'split_id': split_id,
        'backbone_model_id': backbone_model_id,
        'ood_id': max([e.get('ood_id', 0) for e in experiment_logs], default=0),
        'llm_model_id': llm_model_id,
        'fewshot_k': str(k_values),
        'stage': 'complete',
        'status': 'completed',
        'metrics_json': json.dumps(all_k_results, ensure_ascii=False),
        'duration_seconds': round(total_duration, 3),
    }

    experiment_log_id = save_log('experiment', final_log_data)

    # 打印最终汇总
    print("\n" + "=" * 80)
    print("实验完成！")
    print("=" * 80)

    print(f"\n  配置:")
    print(f"    数据集: dataset_{dataset_id}")
    print(f"    开集划分: split_openset_{split_id}")
    print(f"    Backbone: model_{backbone_model_id} ({backbone_variant})")
    if llm_model_id:
        print(f"    LLM: model_{llm_model_id} ({llm_variant})")
    else:
        print(f"    LLM: 跳过")

    print(f"\n  各k值结果:")
    print(f"    {'k':>6} | {'Macro-F1':>10} | {'Unknown-F1':>10} | {'Unknown-Recall':>14} | {'Accuracy':>10}")
    print(f"    {'-' * 6}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 14}-+-{'-' * 10}")

    for k_label, result in sorted(all_k_results.items(), key=lambda x: x[0]):
        print(f"    {k_label:>6} | {result['routed_macro_f1']:>10.4f} | "
              f"{result['routed_unknown_f1']:>10.4f} | "
              f"{result['routed_unknown_recall']:>14.4f} | "
              f"{result['routed_accuracy']:>10.4f}")

    print(f"\n  总耗时: {total_duration:.2f}秒")
    print(f"  实验日志ID: {experiment_log_id}")

    all_results = {
        'experiment_log_id': experiment_log_id,
        'backbone_model_id': backbone_model_id,
        'llm_model_id': llm_model_id,
        'total_duration': total_duration,
        'per_k_results': all_k_results,
        'experiment_logs': experiment_logs,
    }

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="开放世界融合模型完整实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 完整实验：训练A3 + 训练OOD头 + 评估 (k=5,10,full)
  python scripts/run_openworld_experiment.py --k_values "5,10,None"

  # 仅用已有模型评估
  python scripts/run_openworld_experiment.py --backbone_model_id 3 --skip_backbone_train --k_values "5,10"

  # 跳过LLM（仅A3+OOD）
  python scripts/run_openworld_experiment.py --skip_llm --k_values "5,10,20"
        """
    )
    parser.add_argument("--dataset_id", type=int, default=1, help="数据集ID")
    parser.add_argument("--split_id", type=int, default=0, help="开集划分ID")
    parser.add_argument("--k_values", type=str, default="5,10,20,None",
                        help="少样本k值列表，逗号分隔（None表示全量数据）")
    parser.add_argument("--backbone_model_id", type=int, default=None,
                        help="已有Backbone模型ID（指定后跳过训练）")
    parser.add_argument("--llm_model_id", type=int, default=None,
                        help="已有LLM模型ID（指定后跳过训练）")
    parser.add_argument("--model_path", type=str, default=None, help="LLM模型路径")
    parser.add_argument("--backbone_variant", type=str, default="A3",
                        help="Backbone变体（A0/A1/A2/A3）")
    parser.add_argument("--llm_variant", type=str, default="A0",
                        help="LLM变体（A0/A1/A2/A3）")
    parser.add_argument("--distance_type", type=str, default="cosine",
                        choices=["euclidean", "cosine", "mahalanobis"],
                        help="OOD距离度量")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="OOD温度系数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--skip_backbone_train", action="store_true",
                        help="跳过Backbone训练")
    parser.add_argument("--skip_ood_train", action="store_true",
                        help="跳过OOD头训练")
    parser.add_argument("--skip_llm", action="store_true",
                        help="跳过LLM训练和推理")
    parser.add_argument("--no_report", action="store_true",
                        help="不生成汇总报告")

    # 训练参数
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--train_grad_accum", type=int, default=4)
    parser.add_argument("--train_lr", type=float, default=1e-4)
    parser.add_argument("--train_epochs", type=int, default=3)

    # OOD参数
    parser.add_argument("--ood_lr", type=float, default=1e-3)
    parser.add_argument("--ood_epochs", type=int, default=10)
    parser.add_argument("--ood_batch_size", type=int, default=64)

    # 评估参数
    parser.add_argument("--eval_batch_size", type=int, default=32)

    args = parser.parse_args()

    # 解析k_values
    k_values_str = args.k_values
    k_values = []
    for k_str in k_values_str.split(","):
        k_str = k_str.strip()
        if k_str.lower() == "none":
            k_values.append(None)
        else:
            k_values.append(int(k_str))

    run_openworld_experiment(
        dataset_id=args.dataset_id,
        split_id=args.split_id,
        k_values=k_values,
        backbone_model_id=args.backbone_model_id,
        llm_model_id=args.llm_model_id,
        model_path=args.model_path,
        backbone_variant=args.backbone_variant,
        llm_variant=args.llm_variant,
        distance_type=args.distance_type,
        temperature=args.temperature,
        seed=args.seed,
        train_batch_size=args.train_batch_size,
        train_grad_accum=args.train_grad_accum,
        train_lr=args.train_lr,
        train_epochs=args.train_epochs,
        ood_lr=args.ood_lr,
        ood_epochs=args.ood_epochs,
        ood_batch_size=args.ood_batch_size,
        eval_batch_size=args.eval_batch_size,
        skip_backbone_train=args.skip_backbone_train,
        skip_ood_train=args.skip_ood_train,
        skip_llm=args.skip_llm,
        generate_report=not args.no_report
    )