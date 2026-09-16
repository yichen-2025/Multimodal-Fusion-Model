"""
diag_ood_scores.py —— 诊断 OOD 路由报告中的分数退化问题（D1）

用法：
    python scripts/diag_ood_scores.py                  # 扫描所有本地报告
    python scripts/diag_ood_scores.py --report PATH   # 只分析某一份报告
    python scripts/diag_ood_scores.py --recent 5       # 只分析最近 N 份报告

输出：
    - 分数唯一值占比（低 → 退化严重）
    - 常数聚集 top-N（值、计数、占比）
    - 退化样本是否集中在某些 label 上
    - 如果能加载 backbone → 进一步检查退化样本的融合特征范数
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

LABEL_NAMES = {
    0: "BENIGN",
    1: "DoS Hulk",
    2: "DoS GoldenEye",
    3: "DoS slowloris",
    4: "DoS Slowhttptest",
    5: "DDoS",
    6: "PortScan",
    7: "FTP-Patator",
    8: "SSH-Patator",
    9: "Bot",
    10: "Web Attack - Brute Force",
    11: "Web Attack - XSS",
    12: "Web Attack - Sql Injection",
    13: "Infiltration",
    14: "Heartbleed",
    15: "unknown",
}


def load_report(path):
    """加载一份 ood_routing_*.json 报告"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_scores(scores, labels, report_name=""):
    """
    分析一组 OOD 分数的退化程度

    Returns:
        dict: 诊断结果
    """
    scores_arr = np.array(scores, dtype=np.float64)
    labels_arr = np.array(labels, dtype=np.int64)

    total = len(scores_arr)

    # ---- 1. 唯一性分析 ----
    # 用 12 位小数近似（避免浮点数精度导致的假不同）
    scores_rounded = np.round(scores_arr, decimals=12)
    unique_vals, unique_counts = np.unique(scores_rounded, return_counts=True)
    num_unique = len(unique_vals)
    unique_ratio = num_unique / total

    # ---- 2. 常数聚集 ----
    # 按出现次数排序，找占比 > 1% 的常数聚集
    sorted_idx = np.argsort(-unique_counts)
    top_constants = []
    for i in sorted_idx[:10]:  # top 10
        val = unique_vals[i]
        cnt = int(unique_counts[i])
        pct = cnt / total * 100
        if pct >= 0.5:  # 只报告占比 ≥ 0.5% 的
            # 找到取这个值的样本的 label 分布
            mask = scores_rounded == val
            cluster_labels = labels_arr[mask]
            lbl_counts = Counter(cluster_labels.tolist())
            lbl_str = ", ".join(
                f"{LABEL_NAMES.get(l, str(l))}({l}):{c}"
                for l, c in lbl_counts.most_common(5)
            )
            top_constants.append({
                'value': float(val),
                'count': cnt,
                'pct': pct,
                'label_dist': lbl_str,
            })

    # ---- 3. 退化样本整体分布（用出现次数 > 均值 5 倍的当"聚集点"）----
    if num_unique > 0:
        threshold_pct = 5.0  # 任何值占比 > 5% → 视为退化聚集
        degenerate_mask = np.zeros(total, dtype=bool)
        for i in range(len(unique_vals)):
            if unique_counts[i] / total * 100 >= threshold_pct:
                degenerate_mask |= (scores_rounded == unique_vals[i])
        degenerate_count = int(degenerate_mask.sum())
        degenerate_pct = degenerate_count / total * 100

        # 退化样本的 label 分布
        if degenerate_count > 0:
            deg_labels = labels_arr[degenerate_mask]
            deg_lbl_counts = Counter(deg_labels.tolist())
        else:
            deg_lbl_counts = {}
    else:
        degenerate_count = 0
        degenerate_pct = 0.0
        deg_lbl_counts = {}

    return {
        'report_name': report_name,
        'total': total,
        'num_unique': num_unique,
        'unique_ratio': unique_ratio,
        'degenerate_count': degenerate_count,
        'degenerate_pct': degenerate_pct,
        'degenerate_label_dist': deg_lbl_counts,
        'top_constants': top_constants,
        'score_min': float(scores_arr.min()),
        'score_max': float(scores_arr.max()),
        'score_mean': float(scores_arr.mean()),
        'score_std': float(scores_arr.std()),
    }


def print_diagnostic(result):
    """格式化打印诊断结果"""
    print(f"\n{'='*70}")
    print(f"报告: {result['report_name']}")
    print(f"{'='*70}")

    print(f"\n  样本总数:           {result['total']}")
    print(f"  唯一值个数:         {result['num_unique']}")
    print(f"  唯一值占比:         {result['unique_ratio']:.4%}  {'✅ 正常' if result['unique_ratio'] > 0.95 else '⚠️ 退化!'}")
    print(f"  分数范围:           [{result['score_min']:.6f}, {result['score_max']:.6f}]")
    print(f"  分数 μ±σ:           {result['score_mean']:.6f} ± {result['score_std']:.6f}")

    print(f"\n  退化聚集 (占比>5%): {result['degenerate_count']} / {result['total']} "
          f"({result['degenerate_pct']:.1f}%)  "
          f"{'❌ 严重退化!' if result['degenerate_pct'] > 10 else ''}")
    if result['degenerate_label_dist']:
        print(f"  退化样本label分布:")
        for lbl, cnt in sorted(result['degenerate_label_dist'].items()):
            pct = cnt / max(1, result['degenerate_count']) * 100
            print(f"    {LABEL_NAMES.get(lbl, str(lbl))}({lbl}): {cnt} ({pct:.1f}%)")

    print(f"\n  常数聚集 Top-{min(10, len(result['top_constants']))}:")
    print(f"  {'值':>18s}  {'计数':>6s}  {'占比':>7s}  label分布")
    print(f"  {'-'*18}  {'-'*6}  {'-'*7}  {'-'*40}")
    for tc in result['top_constants'][:10]:
        flag = " ⚠️" if tc['pct'] >= 5 else (" ·" if tc['pct'] >= 1 else "")
        print(f"  {tc['value']:>18.12f}  {tc['count']:>6d}  {tc['pct']:>6.2f}%{flag}  "
              f"{tc['label_dist']}")


def scan_all_reports(report_dir, recent_n=None):
    """扫描 ood_reports 目录下所有 .json"""
    reports = [
        os.path.join(report_dir, f)
        for f in os.listdir(report_dir)
        if f.startswith('ood_routing_') and f.endswith('.json')
    ]
    reports.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    if recent_n is not None:
        reports = reports[:recent_n]
    return reports


def main():
    parser = argparse.ArgumentParser(description="诊断 OOD 报告中的分数退化问题 (D1)")
    parser.add_argument('--report', type=str, default=None,
                        help="只分析某一份报告的完整路径")
    parser.add_argument('--recent', type=int, default=None,
                        help="只分析最近 N 份报告")
    parser.add_argument('--all', action='store_true',
                        help="扫描所有本地报告")
    args = parser.parse_args()

    report_dir = os.path.join(PROJECT_ROOT, 'ood_reports')

    if args.report:
        reports = [args.report]
    else:
        reports = scan_all_reports(report_dir, recent_n=args.recent)

    if not reports:
        print(f"❌ 没找到任何报告。报告目录: {report_dir}")
        return

    print(f"📂 将分析 {len(reports)} 份报告")
    print(f"📂 报告目录: {report_dir}")

    all_results = []
    for rp in reports:
        if not os.path.exists(rp):
            print(f"  ⚠️  文件不存在，跳过: {rp}")
            continue
        try:
            report = load_report(rp)
            scores = report.get('all_ood_scores', [])
            labels = report.get('all_labels', [])
            if not scores or not labels:
                print(f"  ⚠️  {os.path.basename(rp)}: 无 all_ood_scores / all_labels")
                continue
            result = analyze_scores(scores, labels, os.path.basename(rp))
            print_diagnostic(result)
            all_results.append(result)
        except Exception as e:
            print(f"  ❌ {os.path.basename(rp)} 分析失败: {e}")

    # ---- 汇总 ----
    if all_results:
        print(f"\n{'#'*70}")
        print(f"  汇总: {len(all_results)} 份报告")
        print(f"{'#'*70}")
        degenerate_count = sum(1 for r in all_results if r['degenerate_pct'] > 5)
        print(f"  有严重退化 (>5%常数聚集): {degenerate_count} / {len(all_results)}")
        avg_unique_ratio = np.mean([r['unique_ratio'] for r in all_results])
        print(f"  平均唯一值占比:           {avg_unique_ratio:.4%}")

        # 列出最严重的
        if all_results:
            worst = min(all_results, key=lambda r: r['unique_ratio'])
            print(f"\n  🚨 退化最严重: {worst['report_name']}")
            print(f"     唯一值占比 {worst['unique_ratio']:.4%}, "
                  f"退化占比 {worst['degenerate_pct']:.1f}%")


if __name__ == "__main__":
    main()
