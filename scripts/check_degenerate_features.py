"""
check_degenerate_features.py —— 深入检查退化样本的融合特征

验证假设：同类样本文本描述相似 → LLM embedding 相同/极近 → normalize 后方向一致 → cosine 距离相同

用法：
    python scripts/check_degenerate_features.py --report ood_reports/ood_routing_20260910_135129.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

LABEL_NAMES = {
    0: "BENIGN", 1: "DoS Hulk", 2: "DoS GoldenEye", 3: "DoS slowloris",
    4: "DoS Slowhttptest", 5: "DDoS", 6: "PortScan", 7: "FTP-Patator",
    8: "SSH-Patator", 9: "Bot", 10: "Web Attack - Brute Force",
    11: "Web Attack - XSS", 12: "Web Attack - Sql Injection",
    13: "Infiltration", 14: "Heartbleed", 15: "unknown",
}


def find_matching_split(report_labels, dataset_id=1):
    """在 split_data 中找 test labels 匹配报告 labels 的 split"""
    split_base = os.path.join(PROJECT_ROOT, 'split_data', f'dataset_{dataset_id}')
    if not os.path.exists(split_base):
        print(f"  ⚠️  split 目录不存在: {split_base}")
        return None, None

    for d in sorted(os.listdir(split_base)):
        test_path = os.path.join(split_base, d, 'test.npz')
        if not os.path.exists(test_path):
            continue
        try:
            npz = np.load(test_path, allow_pickle=True)
            npz_labels = npz['labels'].astype(int)
            if len(npz_labels) == len(report_labels):
                # 检查 label 分布是否匹配
                report_dist = Counter(report_labels.tolist())
                npz_dist = Counter(npz_labels.tolist())
                if report_dist == npz_dist:
                    print(f"  ✅ 找到匹配 split: split_data/dataset_{dataset_id}/{d} "
                          f"(labels 长度={len(npz_labels)}, 分布一致)")
                    return test_path, npz
        except Exception as e:
            print(f"  ⚠️  读取失败 {d}: {e}")

    # 如果没精确匹配，打印所有 split 的 size
    print(f"\n  报告 labels 长度: {len(report_labels)}")
    for d in sorted(os.listdir(split_base)):
        test_path = os.path.join(split_base, d, 'test.npz')
        if os.path.exists(test_path):
            npz = np.load(test_path, allow_pickle=True)
            print(f"    split_data/dataset_{dataset_id}/{d}: test.npz labels={len(npz['labels'])}")

    return None, None


def analyze_within_class_similarity(npz, labels, emb_key='text_embeddings'):
    """分析同类样本的 embedding 余弦相似度"""
    if emb_key not in npz.files:
        print(f"\n  ⚠️  {emb_key} 不存在于 npz, 可用 keys: {list(npz.files)}")
        return

    emb = npz[emb_key].astype(np.float64)
    print(f"\n  {emb_key} shape: {emb.shape}, dtype: {emb.dtype}")

    unique_labels = np.unique(labels)
    print(f"\n{'='*70}")
    print(f"各类别 {emb_key} 内部相似度分析（余弦距离 1 - cos_sim）")
    print(f"{'='*70}")
    print(f"  {'类别':<35s} {'n':>5s} {'唯一值':>7s} {'均值cos距离':>12s} {'std':>10s} {'最小':>12s} {'最大':>12s}")
    print(f"  {'-'*35} {'-'*5} {'-'*7} {'-'*12} {'-'*10} {'-'*12} {'-'*12}")

    for lbl in sorted(unique_labels):
        mask = labels == lbl
        cls_emb = emb[mask]
        cls_total = len(cls_emb)

        # 同类内部的两两 cosine 距离（采样最多 200 个）
        if cls_total > 200:
            idx = np.random.choice(cls_total, 200, replace=False)
            cls_emb_sample = cls_emb[idx]
        else:
            cls_emb_sample = cls_emb

        # L2 normalize
        norms = np.linalg.norm(cls_emb_sample, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)  # 防零范数
        emb_norm = cls_emb_sample / norms

        # 两两 cosine 相似度 → 距离
        cos_sim = emb_norm @ emb_norm.T  # [N, N]
        dists = 1.0 - cos_sim  # [N, N]

        # 取非对角元素的上三角
        n = len(emb_norm)
        iu = np.triu_indices(n, k=1)
        pairwise_dists = dists[iu]

        # 同类样本是否唯一（round 到 8 位小数）
        rounded = np.round(cls_emb, decimals=8)
        unique_count = len(np.unique(rounded, axis=0))

        mean_d = np.mean(pairwise_dists) if len(pairwise_dists) > 0 else 0
        std_d = np.std(pairwise_dists) if len(pairwise_dists) > 0 else 0
        min_d = np.min(pairwise_dists) if len(pairwise_dists) > 0 else 0
        max_d = np.max(pairwise_dists) if len(pairwise_dists) > 0 else 0

        flag = ""
        if mean_d < 1e-6:
            flag = "  ❌ 同类embedding几乎完全相同!"
        elif mean_d < 1e-4:
            flag = "  ⚠️  同类embedding高度相似"

        print(f"  {LABEL_NAMES.get(int(lbl), str(lbl)):<35s}({int(lbl):2d}): "
              f"{cls_total:5d} {unique_count:7d} {mean_d:12.8f} {std_d:10.8f} "
              f"{min_d:12.8f} {max_d:12.8f}{flag}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    parser.add_argument('--dataset_id', type=int, default=None)
    args = parser.parse_args()

    # ---- 1. 加载报告 ----
    with open(args.report, 'r', encoding='utf-8') as f:
        report = json.load(f)

    scores = np.array(report['all_ood_scores'], dtype=np.float64)
    labels = np.array(report['all_labels'], dtype=np.int64)
    scores_rounded = np.round(scores, decimals=12)
    dataset_id = args.dataset_id if args.dataset_id else report.get('dataset_id', 1)

    print(f"报告: {os.path.basename(args.report)}")
    print(f"Backbone: model_{report['backbone_model_id']} ({report['backbone_variant']})")
    print(f"OOD head: ood_head_{report['ood_id']}")
    print(f"样本数: {len(scores)}, OOD 头距离类型: ", end="")

    ood_cfg_path = os.path.join(PROJECT_ROOT, 'saved_ood_heads',
                                f"ood_head_{report['ood_id']}", 'config.json')
    if os.path.exists(ood_cfg_path):
        with open(ood_cfg_path, 'r', encoding='utf-8') as f:
            oc = json.load(f)
        print(f"{oc.get('distance_type', '?')}, temperature={oc.get('temperature', '?')}, "
              f"num_known={oc.get('num_known_classes', '?')}")
    else:
        print("(OOD head config 未找到)")

    # ---- 2. 各类别 OOD 分数分布 ----
    print(f"\n{'='*70}")
    print("各类别 OOD 分数分布")
    print(f"{'='*70}")
    unique_labels = np.unique(labels)
    for lbl in sorted(unique_labels):
        mask = labels == lbl
        cls_scores = scores_rounded[mask]
        cls_unique = len(np.unique(cls_scores))
        cls_total = len(cls_scores)
        same_ratio = 1 - cls_unique / cls_total if cls_total > 0 else 0
        flag = " ❌ 同类分数几乎全相同!" if same_ratio > 0.5 else (" ⚠️" if same_ratio > 0.2 else "")
        print(f"  {LABEL_NAMES.get(int(lbl), str(lbl)):35s}({int(lbl):2d}): "
              f"n={cls_total:4d}, unique={cls_unique:4d} ({cls_unique/cls_total:.1%}){flag}")

    # ---- 3. 找匹配的 split ----
    print(f"\n{'='*70}")
    print("定位匹配的 split 数据")
    print(f"{'='*70}")
    test_path, npz = find_matching_split(labels, dataset_id)
    if npz is None:
        # 试 dataset_0
        test_path, npz = find_matching_split(labels, 0)

    if npz is not None:
        print(f"\n  test.npz keys: {list(npz.files)}")

        # ---- 4. 分析 text_embeddings 同类相似度 ----
        analyze_within_class_similarity(npz, labels, 'text_embeddings')

        # ---- 5. 如果有 scaled_features，也分析一下 ----
        if 'scaled_features' in npz.files:
            analyze_within_class_similarity(npz, labels, 'scaled_features')

        # ---- 6. 报告 label 分布 vs npz label 分布 ----
        print(f"\n{'='*70}")
        print("报告 labels 分布 vs npz labels 分布")
        print(f"{'='*70}")
        report_dist = Counter(labels.tolist())
        npz_dist = Counter(npz['labels'].astype(int).tolist())
        for lbl in sorted(set(list(report_dist.keys()) + list(npz_dist.keys()))):
            r_cnt = report_dist.get(lbl, 0)
            n_cnt = npz_dist.get(lbl, 0)
            match = "✅" if r_cnt == n_cnt else "❌"
            print(f"  {LABEL_NAMES.get(lbl, str(lbl)):30s}({lbl:2d}): report={r_cnt:4d}, npz={n_cnt:4d} {match}")
    else:
        print("\n  ❌ 没找到匹配的 split 数据，跳过 embedding 分析")


if __name__ == '__main__':
    main()
