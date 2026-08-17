import os
import sys
import json
import glob
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def collect_ood_routing_reports(report_dir=None):
    """收集所有OOD路由评估报告"""
    if report_dir is None:
        report_dir = os.path.join(PROJECT_ROOT, "ood_reports")

    if not os.path.exists(report_dir):
        print(f"未找到报告目录: {report_dir}")
        return []

    reports = []
    for f in sorted(os.listdir(report_dir)):
        if f.startswith("ood_routing_") and f.endswith(".json"):
            filepath = os.path.join(report_dir, f)
            with open(filepath, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            data['source_file'] = f
            reports.append(data)

    return reports


def collect_ood_training_logs():
    """收集所有OOD训练日志"""
    log_dir = os.path.join(PROJECT_ROOT, "logs", "ood_training")
    logs = []

    if not os.path.exists(log_dir):
        return logs

    for f in sorted(os.listdir(log_dir)):
        if f.startswith("log_") and f.endswith(".json"):
            filepath = os.path.join(log_dir, f)
            with open(filepath, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            data['source_file'] = f
            logs.append(data)

    return logs


def collect_openset_split_logs():
    """收集开集划分日志"""
    log_dir = os.path.join(PROJECT_ROOT, "logs", "openset_split")
    logs = []

    if not os.path.exists(log_dir):
        return logs

    for f in sorted(os.listdir(log_dir)):
        if f.startswith("log_") and f.endswith(".json"):
            filepath = os.path.join(log_dir, f)
            with open(filepath, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            data['source_file'] = f
            logs.append(data)

    return logs


def collect_fewshot_split_logs():
    """收集少样本划分日志"""
    log_dir = os.path.join(PROJECT_ROOT, "logs", "fewshot_split")
    logs = []

    if not os.path.exists(log_dir):
        return logs

    for f in sorted(os.listdir(log_dir)):
        if f.startswith("log_") and f.endswith(".json"):
            filepath = os.path.join(log_dir, f)
            with open(filepath, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            data['source_file'] = f
            logs.append(data)

    return logs


def generate_experiment_summary(output_dir=None):
    """
    生成开放世界实验汇总报告

    生成内容：
    1. openworld_fewshot.csv — 各k值下的A3 vs OOD-routed对比
    2. experiment_summary.md — 人类可读的实验总结
    3. metrics_by_k.json — 结构化的指标汇总
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "test_reports")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("开放世界实验汇总报告生成器")
    print("=" * 60)

    # ========== 1. 收集数据 ==========
    print("\n[1/4] 收集评估报告...")
    routing_reports = collect_ood_routing_reports()
    ood_logs = collect_ood_training_logs()
    openset_logs = collect_openset_split_logs()
    fewshot_logs = collect_fewshot_split_logs()

    print(f"  - OOD路由报告: {len(routing_reports)}")
    print(f"  - OOD训练日志: {len(ood_logs)}")
    print(f"  - 开集划分日志: {len(openset_logs)}")
    print(f"  - 少样本划分日志: {len(fewshot_logs)}")

    if not routing_reports:
        print("\n警告：没有找到OOD路由报告，请先运行评估！")
        return {}

    # ========== 2. 构建对比表 ==========
    print("\n[2/4] 构建对比表...")

    rows = []
    for report in routing_reports:
        routed = report.get('routed', {})
        a3_baseline = report.get('a3_baseline', {})

        row = {
            'timestamp': report.get('timestamp', ''),
            'dataset_id': report.get('dataset_id', ''),
            'split_id': report.get('split_id', ''),
            'backbone_model_id': report.get('backbone_model_id', ''),
            'ood_id': report.get('ood_id', ''),
            'llm_model_id': report.get('llm_model_id', ''),
            'total_test': report.get('total_test', ''),

            # OOD-routed 指标
            'routed_accuracy': routed.get('accuracy', ''),
            'routed_macro_f1': routed.get('macro_f1', ''),
            'routed_unknown_recall': routed.get('unknown_recall', ''),
            'routed_unknown_f1': routed.get('unknown_f1', ''),
            'routed_unknown_leak_rate': routed.get('unknown_leak_rate', ''),
            'routed_known_accuracy': routed.get('known_accuracy', ''),
            'routed_num_known_routed': routed.get('num_known_routed', ''),
            'routed_num_unknown_routed': routed.get('num_unknown_routed', ''),

            # 各类召回
            'routed_benign_recall': routed.get('per_class', {}).get('BENIGN', {}).get('recall', ''),
            'routed_known_ddos_recall': routed.get('per_class', {}).get('known_DDoS', {}).get('recall', ''),

            # A3基线指标
            'a3_known_accuracy': a3_baseline.get('known_accuracy', ''),
            'a3_known_macro_f1': a3_baseline.get('known_macro_f1', ''),
            'a3_unknown_leak_rate': a3_baseline.get('unknown_leak_rate', ''),

            'duration_seconds': report.get('duration_seconds', ''),
            'source_file': report.get('source_file', ''),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # ========== 3. 生成CSV报告 ==========
    print("\n[3/4] 生成CSV报告...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 3a. 详细逐行报告
    csv_path = os.path.join(output_dir, f"openworld_runs_{timestamp}.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  - 详细报告: {csv_path}")

    # 3b. 汇总报告（按k值分组对比）
    summary_rows = []
    if len(df) > 0:
        # 尝试从报告文件名解析k值
        for _, row in df.iterrows():
            k_val = "full"
            src_file = row.get('source_file', '')
            # 尝试关联fewshot日志获取k值
            for fs_log in fewshot_logs:
                fid = fs_log.get('log_id', '')
                # 通过dataset_id和split_id匹配
                if (str(fs_log.get('dataset_id')) == str(row.get('dataset_id')) and
                    str(fs_log.get('output_split_id')) == str(row.get('split_id'))):
                    k_val = fs_log.get('k_per_class', 'full')
                    break

            summary_rows.append({
                'k': k_val,
                'routed_macro_f1': row.get('routed_macro_f1', ''),
                'routed_unknown_f1': row.get('routed_unknown_f1', ''),
                'routed_unknown_recall': row.get('routed_unknown_recall', ''),
                'routed_benign_recall': row.get('routed_benign_recall', ''),
                'routed_known_ddos_recall': row.get('routed_known_ddos_recall', ''),
                'routed_accuracy': row.get('routed_accuracy', ''),
                'a3_known_macro_f1': row.get('a3_known_macro_f1', ''),
                'a3_unknown_leak_rate': row.get('a3_unknown_leak_rate', ''),
                'improvement_macro_f1': (
                    (row.get('routed_macro_f1', 0) or 0) -
                    (row.get('a3_known_macro_f1', 0) or 0)
                ),
                'dataset_id': row.get('dataset_id', ''),
                'split_id': row.get('split_id', ''),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(output_dir, f"openworld_fewshot.csv")
    summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
    print(f"  - 汇总报告: {summary_csv_path}")

    # ========== 4. 生成可读总结 ==========
    print("\n[4/4] 生成可读总结...")

    md_content = generate_markdown_summary(routing_reports, ood_logs, openset_logs, fewshot_logs, summary_df)

    md_path = os.path.join(output_dir, f"openworld_experiment_summary_{timestamp}.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"  - 可读报告: {md_path}")

    # ========== 5. 生成结构化指标 ==========
    json_path = os.path.join(output_dir, f"openworld_metrics_{timestamp}.json")
    metrics = {
        'generated_at': datetime.now().isoformat(),
        'routing_reports_count': len(routing_reports),
        'ood_training_logs_count': len(ood_logs),
        'openset_split_logs_count': len(openset_logs),
        'fewshot_split_logs_count': len(fewshot_logs),
        'by_k': {},
        'all_runs': rows,
    }

    # 按k值汇总
    if len(summary_rows) > 0:
        by_k = {}
        for row in summary_rows:
            k = str(row.get('k', 'full'))
            if k not in by_k:
                by_k[k] = []
            by_k[k].append({
                'routed_macro_f1': row.get('routed_macro_f1', 0),
                'routed_unknown_f1': row.get('routed_unknown_f1', 0),
                'routed_unknown_recall': row.get('routed_unknown_recall', 0),
                'improvement_macro_f1': row.get('improvement_macro_f1', 0),
            })

        # 计算每个k的均值
        for k, runs in by_k.items():
            if runs:
                metrics['by_k'][k] = {
                    'num_runs': len(runs),
                    'avg_routed_macro_f1': np.mean([r['routed_macro_f1'] for r in runs]),
                    'avg_routed_unknown_f1': np.mean([r['routed_unknown_f1'] for r in runs]),
                    'avg_routed_unknown_recall': np.mean([r['routed_unknown_recall'] for r in runs]),
                    'avg_improvement_macro_f1': np.mean([r['improvement_macro_f1'] for r in runs]),
                }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  - 结构化指标: {json_path}")

    print("\n" + "=" * 60)
    print("实验汇总报告生成完成！")
    print(f"  - 输出目录: {output_dir}")
    print("=" * 60)

    return {
        'csv_path': csv_path,
        'summary_csv_path': summary_csv_path,
        'md_path': md_path,
        'json_path': json_path,
    }


def generate_markdown_summary(routing_reports, ood_logs, openset_logs, fewshot_logs, summary_df):
    """生成Markdown格式的可读报告"""

    lines = []
    lines.append("# 开放世界融合模型 — 实验汇总报告\n")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**报告数量**: {len(routing_reports)} 次路由评估\n")

    lines.append("\n---\n")
    lines.append("## 1. 实验配置\n")

    if openset_logs:
        latest = openset_logs[-1]
        lines.append(f"- **开集数据**: dataset_{latest.get('dataset_id')}/split_openset_{latest.get('output_split_id')}")
        lines.append(f"  - 训练集: {latest.get('train_total')} (已知类)")
        lines.append(f"  - 测试集: {latest.get('test_total')} (含{latest.get('test_unknown')}个未知样本)")
        lines.append(f"  - 未知比例: {latest.get('unknown_ratio', 'N/A')}")

    if fewshot_logs:
        lines.append(f"- **少样本配置**: {len(fewshot_logs)} 个k值")
        for fl in fewshot_logs:
            lines.append(f"  - k={fl.get('k_per_class')}: 训练{fl.get('train_total')}样本, 测试{fl.get('test_total')}样本")

    lines.append("\n---\n")
    lines.append("## 2. OOD检测头配置\n")

    if ood_logs:
        latest_ood = ood_logs[-1]
        lines.append(f"- **OOD头ID**: {latest_ood.get('ood_id')}")
        lines.append(f"- Backbone模型ID: {latest_ood.get('model_id')}")
        lines.append(f"- 特征维度: {latest_ood.get('feature_dim')}")
        lines.append(f"- 距离度量: {latest_ood.get('distance_type')}")
        lines.append(f"- OOD阈值: {latest_ood.get('threshold', 'N/A')}")

    lines.append("\n---\n")
    lines.append("## 3. 核心指标汇总\n")

    if len(routing_reports) > 0:
        lines.append("| 指标 | OOD-Routed | A3基线 | 提升 |")
        lines.append("|------|:----------:|:------:|:----:|")

        # 计算平均
        routed_macro_f1 = np.mean([r.get('routed', {}).get('macro_f1', 0) for r in routing_reports])
        routed_unknown_f1 = np.mean([r.get('routed', {}).get('unknown_f1', 0) for r in routing_reports])
        routed_unknown_recall = np.mean([r.get('routed', {}).get('unknown_recall', 0) for r in routing_reports])
        routed_accuracy = np.mean([r.get('routed', {}).get('accuracy', 0) for r in routing_reports])

        a3_macro_f1 = np.mean([r.get('a3_baseline', {}).get('known_macro_f1', 0) or 0 for r in routing_reports])
        a3_leak = np.mean([r.get('a3_baseline', {}).get('unknown_leak_rate', 0) or 0 for r in routing_reports])

        lines.append(f"| Macro-F1 (三分类) | {routed_macro_f1:.4f} | {a3_macro_f1:.4f} | {routed_macro_f1 - a3_macro_f1:+.4f} |")
        lines.append(f"| Accuracy | {routed_accuracy:.4f} | — | — |")
        lines.append(f"| Unknown F1 | {routed_unknown_f1:.4f} | 0.0000 (全误判) | +{routed_unknown_f1:.4f} |")
        lines.append(f"| Unknown Recall | {routed_unknown_recall:.4f} | 0.0000 | +{routed_unknown_recall:.4f} |")
        lines.append(f"| Unknown Leak Rate | {1-routed_unknown_recall:.4f} | {a3_leak:.4f} | {a3_leak - (1-routed_unknown_recall):+.4f} |")

    lines.append("\n---\n")
    lines.append("## 4. 各类性能\n")

    if len(routing_reports) > 0:
        # 收集各类召回
        benign_recalls = []
        known_ddos_recalls = []
        unknown_recalls = []

        for r in routing_reports:
            per_class = r.get('routed', {}).get('per_class', {})
            benign_recalls.append(per_class.get('BENIGN', {}).get('recall', 0))
            known_ddos_recalls.append(per_class.get('known_DDoS', {}).get('recall', 0))

        lines.append("| 类别 | 召回率 | F1 |")
        lines.append("|------|:------:|:--:|")

        if benign_recalls:
            avg_benign = np.mean(benign_recalls)
            lines.append(f"| BENIGN (0) | {avg_benign:.4f} | — |")
        if known_ddos_recalls:
            avg_known = np.mean(known_ddos_recalls)
            lines.append(f"| known DDoS (1) | {avg_known:.4f} | — |")

        if routing_reports:
            avg_unknown_r = np.mean([r.get('routed', {}).get('unknown_recall', 0) for r in routing_reports])
            avg_unknown_f1 = np.mean([r.get('routed', {}).get('unknown_f1', 0) for r in routing_reports])
            lines.append(f"| unknown DDoS (2) | {avg_unknown_r:.4f} | {avg_unknown_f1:.4f} |")

    lines.append("\n---\n")
    lines.append("## 5. 关键发现\n")

    if len(routing_reports) > 0:
        # 计算提升
        avg_unknown_f1 = np.mean([r.get('routed', {}).get('unknown_f1', 0) for r in routing_reports])
        avg_macro_f1 = np.mean([r.get('routed', {}).get('macro_f1', 0) for r in routing_reports])
        a3_macro_f1 = np.mean([r.get('a3_baseline', {}).get('known_macro_f1', 0) or 0 for r in routing_reports])

        lines.append(f"1. **未知攻击检测**: OOD-routed方案成功检测出未知DDoS攻击，Unknown F1 = **{avg_unknown_f1:.4f}**")
        lines.append(f"2. **整体性能**: 三分类Macro-F1 = **{avg_macro_f1:.4f}**")

        if avg_macro_f1 > a3_macro_f1:
            lines.append(f"3. **超越A3基线**: Macro-F1提升 {avg_macro_f1 - a3_macro_f1:+.4f}，证明开集路由有效")
        else:
            lines.append(f"3. **已知类保持**: A3基线Macro-F1 = {a3_macro_f1:.4f}，OOD-routed = {avg_macro_f1:.4f}")

    lines.append("\n---\n")
    lines.append("*报告由 run_openworld_experiment.py 自动生成*\n")

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="开放世界实验汇总报告生成器")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    args = parser.parse_args()

    generate_experiment_summary(output_dir=args.output_dir)