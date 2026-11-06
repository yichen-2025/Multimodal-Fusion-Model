"""导出 triage 标注样例：只抽样建表，不调 LLM、不占 GPU。

口径见 docs/triage_protocol.md §8：只取 routing_log 里真正被判 unknown 的样本，
genuine(origin=1) 与 simulated(origin=2) 各半，排除 known_false_positive(origin=0)。
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.generate_unknown_explanations import (  # noqa: E402
    ORIGIN_NAMES,
    load_split,
    recover_global_indices,
)

FIELDS = ["sample_id", "source", "family", "five_slots", "labeler",
          "actionability", "sufficiency", "notes"]


def latest_report():
    files = sorted(glob.glob(os.path.join(PROJECT_ROOT, "ood_reports", "ood_routing_*.json")))
    if not files:
        raise SystemExit("ood_reports 下没有路由报告")
    return files[-1]


def main():
    ap = argparse.ArgumentParser(description="导出 triage 标注样例（不评测）")
    ap.add_argument("--routing_report", default=None, help="默认取最新一份路由报告")
    ap.add_argument("--n", type=int, default=20, help="总条数，真实/模拟各一半")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset_id", type=int, default=None)
    ap.add_argument("--split_id", type=int, default=None)
    ap.add_argument("--output", default=os.path.join("data", "annotations", "triage_202611.csv"))
    args = ap.parse_args()

    rep_path = args.routing_report or latest_report()
    if not os.path.isabs(rep_path):
        rep_path = os.path.join(PROJECT_ROOT, rep_path)
    report = json.load(open(rep_path, "r", encoding="utf-8"))
    print("路由报告:", os.path.basename(rep_path))

    dataset_id = args.dataset_id if args.dataset_id is not None else int(report["dataset_id"])
    split_id = args.split_id if args.split_id is not None else int(report["split_id"])

    labels, origins, texts, raw_labels = load_split(dataset_id, split_id)
    if origins is None:
        raise SystemExit("划分缺少 unknown_origin，无法区分真实/模拟未知")
    unknown_label = int(labels.max())

    gidx, unmatched = recover_global_indices(report)
    pool = [(g, e) for g, e in zip(gidx, report["routing_log"]) if g is not None]
    print("routing_log=%d 下标反查成功=%d 未匹配=%d" % (len(gidx), len(pool), unmatched))
    if unmatched > 0:
        print("警告：有 %d 条未反查到全局下标，已丢弃（不影响抽样口径）" % unmatched)

    # 下标反查自检：origin 与标签必须自洽（0=已知类误报，1/2=unknown 类）
    bad = [g for g, _ in pool
           if (int(origins[g]) == 0) == (int(labels[g]) == unknown_label)]
    if bad:
        raise SystemExit("下标反查不一致：%d 条 origin 与标签矛盾，中止" % len(bad))
    n_fp = sum(1 for g, _ in pool if int(origins[g]) == 0)
    print("其中 known_false_positive（origin=0，不入选）: %d 条" % n_fp)

    rng = np.random.RandomState(args.seed)
    per_genuine = args.n // 2
    per_sim = args.n - per_genuine
    rows = []
    for origin, want in ((1, per_genuine), (2, per_sim)):
        cand = sorted({g for g, _ in pool if int(origins[g]) == origin})
        take = min(want, len(cand))
        if take < want:
            print("警告：%s 可用 %d 条，少于目标 %d 条" % (ORIGIN_NAMES[origin], len(cand), want))
        if take == 0:
            continue
        picked = rng.choice(np.array(cand), size=take, replace=False).tolist()
        score_of = {g: e.get("ood_score") for g, e in pool}
        for g in sorted(int(x) for x in picked):
            # test_data.csv 的 Label 在开集划分里已被改写成 unknown(10)，原始家族不可逐样本恢复；
            # 真实未知的留出类是 Web Attack / Other Attack（见 split_0/label_mapping.json），
            # 模拟未知来自某个已知类。两者都只能给粗粒度标签，逐样本不可分。
            if origin == 1:
                fam = "heldout_web_or_other"
                note = "真实留出类（Web Attack / Other Attack，逐样本不可分）；unknown_origin=1"
            else:
                fam = "known_relabeled"
                note = "分布内已知类改标 unknown（原类不可逐样本恢复，特征仍属已知类）；unknown_origin=2"
            rows.append({
                "sample_id": "ds%d_s%d_%04d" % (dataset_id, split_id, g),
                "source": "genuine" if origin == 1 else "simulated",
                "family": fam,
                "five_slots": "",
                "labeler": "",
                "actionability": "",
                "sufficiency": "",
                "notes": "%s；ood_score=%s；报告=%s"
                         % (note, score_of.get(g), os.path.basename(rep_path)),
            })

    out_path = args.output if os.path.isabs(args.output) else os.path.join(PROJECT_ROOT, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    counts = {}
    for r in rows:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    print("写入:", out_path)
    print("条数:", len(rows), counts)
    for r in rows:
        print("  %-16s %-9s %s" % (r["sample_id"], r["source"], r["family"]))


if __name__ == "__main__":
    main()
