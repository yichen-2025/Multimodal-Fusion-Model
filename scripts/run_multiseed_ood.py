"""
多 seed × 多变体 OOD 开集实验运行器

流程（复用 run_ablation 风格）：
  对每个 (seed, variant):
    1. train_model()          → 训练 backbone（openset 划分）
    2. train_ood_head()       → 用冻结 backbone 训 OOD 头
    3. run_ood_routing()      → OOD 路由评估，抓 AUROC / uF1 / leak / macroF1
  4. 汇总 CSV + mean ± std 控制台表

示例：
  # 跑 seed 42 / 123 / 2024，变体 A3 + A0* + A0*_no_text
  python scripts/run_multiseed_ood.py \
    --variants A3 A0* A0*_no_text \
    --seeds 42 123 2024 \
    --model_path /root/autodl-tmp/models/qwen2.5-1.5b

  # 只跑 seed 123 的 A0*（单 seed 单变体，快速验证链路）
  python scripts/run_multiseed_ood.py \
    --variants A0* --seeds 123 \
    --model_path /root/autodl-tmp/models/qwen2.5-1.5b

依赖（必须已 P0 修复）：
  - scripts/train.py 有 --openset 参数
  - src/model_architectures/ood_head.py 已去 eval 扰动
  - split_openset_{split_id} 已重生成（含 val unknown）
"""

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

# 复用 run_ablation 的 GPU 释放逻辑
def _free_gpu(label=""):
    if not torch.cuda.is_available():
        return
    before = torch.cuda.memory_reserved() / 1024 ** 2
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    after = torch.cuda.memory_reserved() / 1024 ** 2
    freed = before - after
    if abs(freed) > 1:
        tag = f"[{label}] " if label else ""
        print(f"  {tag}显存 {before:.0f} -> {after:.0f} MB (释放 {freed:.0f} MB)")


# 延迟 import（避免脚本 --help 就触发 LLM import）
def _import_all():
    from scripts.train import (
        train_model, VARIANT_CONFIGS, VARIANT_DESCRIPTIONS, get_next_model_id
    )
    from scripts.train_ood_head import train_ood_head
    from scripts.run_ood_routing import run_ood_routing
    return (train_model, VARIANT_CONFIGS, VARIANT_DESCRIPTIONS,
            get_next_model_id, train_ood_head, run_ood_routing)


# ---------------------------------------------------------------------------
# 复用已有模型的辅助函数
# ---------------------------------------------------------------------------

def _parse_config_txt(path):
    """解析 train.py 保存的 config.txt（key: value 格式）为 dict，
    缺失字段返回 None（向后兼容老模型，比如没有 openset 的）。"""
    config = {}
    if not os.path.isfile(path):
        return config
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip()
                # 类型转换
                if val.lower() in ("true", "yes"):
                    config[key] = True
                elif val.lower() in ("false", "no"):
                    config[key] = False
                else:
                    try:
                        config[key] = int(val)
                    except ValueError:
                        try:
                            config[key] = float(val)
                        except ValueError:
                            config[key] = val
    except Exception:
        pass
    return config


def _find_existing_model(seed, variant, dataset_id, split_id, openset=False,
                         saved_models_dir=None):
    """
    扫描 saved_models/ 目录，找到与给定参数完全匹配的已训练模型。

    匹配规则：seed / variant / dataset_id / split_id 必须一致。
    openset 优先匹配完全一致的；老模型 config.txt 里没有 openset 字段
    （解析后为 None），如果 openset=False 也视为匹配。

    Returns:
        (model_id, config_dict) — 找到匹配
        (None, None)            — 没找到
    """
    if saved_models_dir is None:
        saved_models_dir = os.path.join(PROJECT_ROOT, "saved_models")

    if not os.path.isdir(saved_models_dir):
        return None, None

    matches = []
    for entry in os.listdir(saved_models_dir):
        if not entry.startswith("model_"):
            continue
        model_dir = os.path.join(saved_models_dir, entry)
        if not os.path.isdir(model_dir):
            continue
        cfg_path = os.path.join(model_dir, "config.txt")
        cfg = _parse_config_txt(cfg_path)
        if not cfg:
            continue  # 没有 config.txt 就跳过（不完整的模型目录）

        # 硬匹配字段
        if cfg.get("seed") != seed:
            continue
        if cfg.get("variant") != variant:
            continue
        if cfg.get("dataset_id") != dataset_id:
            continue
        if cfg.get("split_id") != split_id:
            continue

        # openset 兼容处理
        cfg_openset = cfg.get("openset")
        if cfg_openset is not None and cfg_openset != openset:
            continue
        # cfg_openset is None 说明是老模型（保存时没这个字段），仅当 openset=False 时视作兼容匹配

        try:
            model_id = int(entry.replace("model_", ""))
        except ValueError:
            continue
        matches.append((model_id, cfg))

    if not matches:
        return None, None

    # 多个匹配时，优先 openset 明确一致的；否则返回 model_id 最大的（最新训练的）
    exact = [m for m in matches if m[1].get("openset") == openset]
    chosen = exact[0] if exact else matches[0]
    if len(matches) > 1:
        chosen = max(matches, key=lambda m: m[0])  # model_id 最大
    return chosen


def _find_existing_ood_head(model_id, variant, dataset_id, split_id,
                            saved_ood_dir=None):
    """
    扫描 saved_ood_heads/ 目录，找到与给定参数完全匹配的已训练 OOD 头。

    Returns:
        ood_id (int) — 找到匹配
        None         — 没找到
    """
    if saved_ood_dir is None:
        saved_ood_dir = os.path.join(PROJECT_ROOT, "saved_ood_heads")

    if not os.path.isdir(saved_ood_dir):
        return None

    import json
    matches = []
    for entry in os.listdir(saved_ood_dir):
        if not entry.startswith("ood_head_"):
            continue
        ood_dir = os.path.join(saved_ood_dir, entry)
        if not os.path.isdir(ood_dir):
            continue
        cfg_path = os.path.join(ood_dir, "config.json")
        if not os.path.isfile(cfg_path):
            continue
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            continue

        if cfg.get("model_id") != model_id:
            continue
        if cfg.get("variant") != variant:
            continue
        if cfg.get("dataset_id") != dataset_id:
            continue
        if cfg.get("split_id") != split_id:
            continue

        try:
            ood_id = int(entry.replace("ood_head_", ""))
        except ValueError:
            continue
        matches.append(ood_id)

    if not matches:
        return None
    return max(matches)  # 取 ood_id 最大的（最新训练的）


def run_multiseed(variants=None,
                  seeds=None,
                  dataset_id=0,
                  split_id=0,
                  model_path=None,
                  epochs=10,
                  batch_size=4,
                  gradient_accumulation=2,
                  lr=5e-4,
                  output_csv=None,
                  reuse=True,
                  force_retrain=False):
    """
    多 seed OOD 开集实验

    Args:
        variants:   变体列表，默认 ["A3", "A0*", "A0*_no_text"]
        seeds:      seed 列表，默认 [42, 123, 2024]
        dataset_id: 默认 0
        split_id:   默认 0
        model_path: LLM 路径（AutoDL: /root/autodl-tmp/models/qwen2.5-1.5b）
        epochs:     backbone 训练轮数，默认 10（与历史 model_102/103 一致）
        batch_size: per_device_train_batch_size，默认 4
        gradient_accumulation: 默认 2
        lr:         learning_rate，默认 5e-4
        output_csv: 结果 CSV 路径，默认 OOD 实验结果目录
        reuse:      开启复用模式（默认 True）：已有匹配的 backbone/OOD 头就跳过训练
        force_retrain: 强制重训（默认 False）：忽略已有模型，全部重新训练
    """
    # --- 默认值 ---
    if variants is None:
        variants = ["A3", "A0*", "A0*_no_text"]
    if seeds is None:
        seeds = [42, 123, 2024]
    if model_path is None:
        model_path = os.path.join(PROJECT_ROOT, "models", "qwen2.5-1.5b")

    # --- import ---
    (train_model, VARIANT_CONFIGS, VARIANT_DESCRIPTIONS,
     get_next_model_id, train_ood_head, run_ood_routing) = _import_all()

    # --- 输出路径 ---
    if output_csv is None:
        out_dir = os.path.join(PROJECT_ROOT, "ood_multiseed_results")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = os.path.join(out_dir, f"multiseed_{ts}.csv")

    # --- 打印实验配置 ---
    print("=" * 80)
    print("多 seed × 多变体 OOD 开集实验")
    print("=" * 80)
    print(f"  variants:   {variants}")
    print(f"  seeds:      {seeds}")
    print(f"  dataset:    split_openset_{dataset_id}/{split_id}")
    print(f"  epochs:     {epochs}  batch: {batch_size}  grad_acc: {gradient_accumulation}  lr: {lr}")
    print(f"  model_path: {model_path}")
    reuse_tag = "FORCE RETRAIN" if force_retrain else ("ON" if reuse else "OFF")
    print(f"  复用模式:   {reuse_tag}  {'(已有匹配的 backbone/OOD 头会跳过训练)' if (reuse and not force_retrain) else ''}")
    if torch.cuda.is_available():
        mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"  GPU:        {torch.cuda.get_device_name(0)} ({mb:.0f} MB)")
    else:
        print("  GPU:        NOT DETECTED (CPU 模式会极慢)")
    print(f"  输出:       {output_csv}")
    print("=" * 80)

    all_rows = []
    prev_model = None  # 用于显存释放

    for seed in seeds:
        for variant in variants:
            # --- 显存释放（每次开始新一轮前） ---
            if prev_model is not None:
                try:
                    prev_model.to('cpu')
                except Exception:
                    pass
                del prev_model
                prev_model = None
            _free_gpu(f"seed={seed} variant={variant} 开始前")

            # --- 检查变体合法性 ---
            if variant not in VARIANT_CONFIGS:
                print(f"  [!] 跳过未知变体: {variant}")
                continue
            config = VARIANT_CONFIGS[variant]
            desc = VARIANT_DESCRIPTIONS.get(variant, variant)

            print(f"\n{'=' * 80}")
            print(f"▶ seed={seed}  variant={variant}  ({desc})")
            print(f"{'=' * 80}")

            row = {
                'seed': seed,
                'variant': variant,
                'description': desc,
                'use_llm': config['use_llm'],
                'use_bert': config['use_bert'],
                'use_numeric': config['use_numeric'],
                'status': 'pending',
                'model_id': None,
                'ood_id': None,
                'source_backbone': None,  # 'trained' / 'reused'
                'source_ood': None,       # 'trained' / 'reused'
                'auroc': None,
                'unknown_f1': None,
                'unknown_recall': None,
                'unknown_leak_rate': None,
                'known_accuracy': None,
                'macro_f1': None,
                'backbone_train_seconds': None,
                'ood_train_seconds': None,
                'routing_seconds': None,
                'error': None,
            }

            try:
                # ===== 1. 训练 backbone（支持复用） =====
                print(f"\n[1/3] 训练 backbone ...")
                existing_model_id = None
                if reuse and not force_retrain:
                    existing_model_id, _ = _find_existing_model(
                        seed=seed, variant=variant,
                        dataset_id=dataset_id, split_id=split_id, openset=True
                    )

                if existing_model_id is not None:
                    # --- 复用已有 backbone ---
                    row['model_id'] = existing_model_id
                    row['source_backbone'] = 'reused'
                    model_save_dir = os.path.join(PROJECT_ROOT, "saved_models", f"model_{existing_model_id}")
                    print(f"  ♻️  复用已有 model_{existing_model_id} ({model_save_dir})")
                    prev_model = None  # 复用模式下我们不持有 model 对象，让后续 train_ood_head 自己加载
                else:
                    # --- 从头训练 ---
                    row['model_id'] = get_next_model_id()
                    row['source_backbone'] = 'trained'
                    t0 = time.time()
                    model = train_model(
                        model_path=model_path,
                        model_id=row['model_id'],
                        dataset_id=dataset_id,
                        split_id=split_id,
                        variant=variant,
                        seed=seed,
                        num_train_epochs=epochs,
                        per_device_train_batch_size=batch_size,
                        gradient_accumulation_steps=gradient_accumulation,
                        learning_rate=lr,
                        openset=True,  # P0: 真正开集划分
                    )
                    row['backbone_train_seconds'] = time.time() - t0
                    print(f"  ✅ model_{row['model_id']} 训练完成 ({row['backbone_train_seconds']:.1f}s)")

                    # LLM 搬去 CPU（避免两份 LLM 同时在 GPU）
                    if torch.cuda.is_available():
                        model.to('cpu')
                        _free_gpu("backbone 训完→OOD头训练前")
                    prev_model = model  # 变体切换时再释放

                # ===== 2. 训练 OOD 头（支持复用） =====
                print(f"\n[2/3] 训练 OOD 头 ...")
                existing_ood_id = None
                if reuse and not force_retrain:
                    existing_ood_id = _find_existing_ood_head(
                        model_id=row['model_id'], variant=variant,
                        dataset_id=dataset_id, split_id=split_id
                    )

                if existing_ood_id is not None:
                    # --- 复用已有 OOD 头 ---
                    row['ood_id'] = existing_ood_id
                    row['source_ood'] = 'reused'
                    ood_save_dir = os.path.join(PROJECT_ROOT, "saved_ood_heads", f"ood_head_{existing_ood_id}")
                    print(f"  ♻️  复用已有 ood_head_{existing_ood_id} ({ood_save_dir})")
                else:
                    # --- 从头训练 ---
                    row['source_ood'] = 'trained'
                    t1 = time.time()
                    ood_head, ood_save_path = train_ood_head(
                        model_id=row['model_id'],
                        dataset_id=dataset_id,
                        split_id=split_id,
                        model_path=model_path,
                        variant=variant,
                        seed=seed,
                        num_epochs=10,
                        batch_size=64,
                    )
                    row['ood_train_seconds'] = time.time() - t1
                    # ood_save_path 形如 ".../saved_ood_heads/ood_head_5"
                    row['ood_id'] = int(os.path.basename(ood_save_path).replace('ood_head_', ''))
                    print(f"  ✅ ood_head_{row['ood_id']} 训练完成 ({row['ood_train_seconds']:.1f}s)")

                    _free_gpu("OOD 头训完→路由前")

                # ===== 3. 路由评估 =====
                print(f"\n[3/3] 路由评估 ...")
                t2 = time.time()
                result = run_ood_routing(
                    backbone_model_id=row['model_id'],
                    backbone_variant=variant,
                    ood_id=row['ood_id'],
                    dataset_id=dataset_id,
                    split_id=split_id,
                    model_path=model_path,
                    verbose=False,  # 静默模式，减少输出
                )
                row['routing_seconds'] = time.time() - t2

                # 收集指标
                routed = result['routed_results']
                row['auroc'] = result['auroc']
                row['unknown_f1'] = routed['unknown_f1']
                row['unknown_recall'] = routed['unknown_recall']
                row['unknown_leak_rate'] = routed['unknown_leak_rate']
                row['known_accuracy'] = routed.get('known_accuracy', 0)
                row['macro_f1'] = routed['macro_f1']
                row['status'] = 'ok'

                print(f"  ✅ AUROC={row['auroc']:.4f}  uF1={row['unknown_f1']:.4f}  leak={row['unknown_leak_rate']:.4f}  "
                      f"({row['routing_seconds']:.1f}s)")

            except Exception as e:
                row['status'] = 'fail'
                row['error'] = str(e)
                import traceback
                traceback.print_exc()
                print(f"  ❌ 失败: {e}")

            all_rows.append(row)

            # 每跑完一条就追加写 CSV（防止中途挂掉丢数据）
            _append_csv(output_csv, [row])

    # --- 完整 DataFrame ---
    df = pd.DataFrame(all_rows)

    # --- 控制台汇总 ---
    _print_summary(df, variants, seeds, output_csv)

    return df


def _append_csv(path, rows):
    """追加写 CSV（首次写 header，之后追加）"""
    df = pd.DataFrame(rows)
    write_header = not os.path.exists(path)
    df.to_csv(path, mode='a', header=write_header, index=False, encoding='utf-8-sig')


def _print_summary(df, variants, seeds, output_csv):
    ok = df[df['status'] == 'ok'].copy()
    if len(ok) == 0:
        print("\n[!] 没有成功的实验！")
        return

    print(f"\n{'=' * 80}")
    print(f"🎯 多 seed OOD 开集实验汇总（成功 {len(ok)}/{len(df)}）")
    print(f"{'=' * 80}")

    # --- 按 seed × variant 展开表 ---
    print("\n(1) 逐条结果（AUROC / uF1 / leak / macroF1）")
    print("-" * 106)
    hdr = f"{'seed':>6} {'variant':<12} {'AUROC':>8} {'uF1':>8} {'uRec':>8} {'leak':>8} {'known_acc':>10} {'macroF1':>8} {'model':>8} {'ood':>6} {'src':>10}"
    print(hdr)
    print("-" * 106)
    for _, r in ok.iterrows():
        src_bb = "R" if r.get('source_backbone') == 'reused' else "T"
        src_ood = "R" if r.get('source_ood') == 'reused' else "T"
        src = f"bb={src_bb} ood={src_ood}"
        print(f"{int(r['seed']):>6} {r['variant']:<12} {r['auroc']:>8.4f} {r['unknown_f1']:>8.4f} "
              f"{r['unknown_recall']:>8.4f} {r['unknown_leak_rate']:>8.4f} "
              f"{r['known_accuracy']:>10.4f} {r['macro_f1']:>8.4f} "
              f"{str(r['model_id']):>8} {str(r['ood_id']):>6} {src:>10}")

    # --- mean ± std 汇总 ---
    print(f"\n(2) mean ± std 跨 seed")
    print("-" * 96)
    metrics = ['auroc', 'unknown_f1', 'unknown_recall', 'unknown_leak_rate', 'macro_f1']
    # ddof=0 总体标准差：单 seed 时 n-1=0 不会再产生 NaN，直接返回 0
    mean_df = ok.groupby('variant')[metrics].mean().round(4)
    std_df = ok.groupby('variant')[metrics].agg(lambda x: x.std(ddof=0)).round(4)
    # 拼成 mean, std, mean, std ... 的宽表
    agg = pd.DataFrame(index=mean_df.index)
    for m in metrics:
        agg[f"{m}_mean"] = mean_df[m]
        agg[f"{m}_std"] = std_df[m]

    header = f"{'variant':<12}"
    for m in metrics:
        header += f"  {m+'_mean':>10}  {m+'_std':>8}"
    print(header)
    print("-" * 96)
    n_seeds = len(ok['seed'].unique())
    for variant in variants:
        if variant not in agg.index:
            continue
        row_str = f"{variant:<12}"
        for m in metrics:
            mean = agg.loc[variant, f"{m}_mean"]
            std = agg.loc[variant, f"{m}_std"]
            row_str += f"  {mean:>10.4f}  {std:>8.4f}"
        print(row_str)
    if n_seeds == 1:
        print(f"  提示：仅 1 个 seed，std 列为 0（总体标准差 ddof=0）。跑 3 seeds 就有真实方差了。")

    # --- 最终排序（AUROC 主指标） ---
    print(f"\n(3) AUROC 排序（主指标，阈值无关）")
    print("-" * 60)
    auroc_mean = ok.groupby('variant')['auroc'].mean().sort_values(ascending=False)
    rank = 1
    for variant, val in auroc_mean.items():
        std = ok[ok['variant'] == variant]['auroc'].std(ddof=0)  # ddof=0 避免单 seed 时 NaN
        bar = " ★" if rank == 1 else "  "
        print(f"  {rank}. {variant:<12} AUROC = {val:.4f} ± {std:.4f}{bar}")
        rank += 1

    # --- 关键对比 ---
    if 'A0*' in auroc_mean.index and 'A3' in auroc_mean.index:
        gap = auroc_mean['A0*'] - auroc_mean['A3']
        print(f"\n  A0* - A3 = {gap:.4f} (相对提升 {gap/auroc_mean['A3']*100:.1f}%)")
        conclusion = "✅ 主假设成立" if gap > 0 else "❌ 主假设不成立"
        print(f"  {conclusion}: A0* {'优于' if gap > 0 else '未优于'} A3")

    print(f"\n{'=' * 80}")
    print(f"结果已保存: {output_csv}")
    print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(
        description="多 seed × 多变体 OOD 开集实验运行器 (P2，支持复用已有模型)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整 P2: 3 seed × 3 variant（默认复用已有模型）
  python scripts/run_multiseed_ood.py \\
    --variants A3 "A0*" A0*_no_text --seeds 42 123 2024 \\
    --model_path /root/autodl-tmp/models/qwen2.5-1.5b

  # 强制全部重新训练（忽略已有模型）
  python scripts/run_multiseed_ood.py --force_retrain

  # 关闭复用（等同 --force_retrain 效果）
  python scripts/run_multiseed_ood.py --no_reuse

  # 单 seed 单变体快速验证
  python scripts/run_multiseed_ood.py \\
    --variants A0* --seeds 123 \\
    --model_path /root/autodl-tmp/models/qwen2.5-1.5b
        """,
    )
    parser.add_argument("--variants", nargs="+",
                        default=["A3", "A0*", "A0*_no_text"],
                        help="变体列表（默认 A3 A0* A0*_no_text）")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 123, 2024],
                        help="seed 列表（默认 42 123 2024）")
    parser.add_argument("--dataset_id", type=int, default=0)
    parser.add_argument("--split_id", type=int, default=0)
    parser.add_argument("--model_path", type=str, default=None,
                        help="LLM 模型路径（AutoDL: /root/autodl-tmp/models/qwen2.5-1.5b）")
    parser.add_argument("--epochs", type=int, default=10,
                        help="backbone 训练轮数（默认 10，与历史 model_102/103 一致）")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--no_reuse", action="store_true", default=False,
                        help="关闭复用，等同 --force_retrain")
    parser.add_argument("--force_retrain", action="store_true", default=False,
                        help="强制全部重新训练（忽略已有 backbone/OOD 头）")
    args = parser.parse_args()

    # --no_reuse 和 --force_retrain 等价
    force_retrain = args.force_retrain or args.no_reuse
    reuse = not force_retrain

    run_multiseed(
        variants=args.variants,
        seeds=args.seeds,
        dataset_id=args.dataset_id,
        split_id=args.split_id,
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        lr=args.lr,
        output_csv=args.output_csv,
        reuse=reuse,
        force_retrain=force_retrain,
    )


if __name__ == "__main__":
    main()
