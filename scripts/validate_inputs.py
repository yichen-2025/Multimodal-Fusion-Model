# -*- coding: utf-8 -*-
"""阶段 2 输入验收：检查新数据集/开集划分是否满足建模前提。

检查项（任一失败 sys.exit(1)，不生成报告、不进入训练）：
1. 划分文件齐全（npz + *_data.csv）
2. 数值维度 == expected（默认 79），且各 split 一致
3. 数值列无 NaN/inf
4. 标签与 unknown_origin 语义一致（known→0 / unknown→1,2；禁止 -1 混入新划分）
5. 规模与来源构成（train/val/test；known/genuine/simulated）
6. 唯一性：数值唯一行、文本唯一行、数值+文本联合唯一行
7. 同输入组：最大组大小、跨标签组数（必须 0）、跨 origin 组数（必须 0）
8. 文本非法标记（"[" / 未知 / unknown）必须 0
9. train 与 test 完全重叠率 <= max_overlap（默认 0）

用法：
  python scripts/validate_inputs.py --dataset_id 200 --split_id 0
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from utils.flow_features import (  # noqa: E402
    find_illegal_text,
    select_numeric_columns,
)

SPLITS = ("train", "val", "test")


class Checker:
    def __init__(self):
        self.failures = []

    def check(self, name, ok, detail=""):
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            self.failures.append(f"{name}: {detail}")
        return ok

    def info(self, text):
        print(f"         {text}")


def resolve_split_dir(dataset_id, split_id, split_name=None):
    base = os.path.join(PROJECT_ROOT, "split_data", f"dataset_{dataset_id}")
    if split_name is None:
        openset = os.path.join(base, f"split_openset_{split_id}")
        split_name = f"split_openset_{split_id}" if os.path.exists(openset) else f"split_{split_id}"
    return os.path.join(base, split_name), split_name


def load_split(split_dir, name):
    npz_path = os.path.join(split_dir, f"{name}.npz")
    csv_path = os.path.join(split_dir, f"{name}_data.csv")
    if not os.path.exists(npz_path) or not os.path.exists(csv_path):
        raise FileNotFoundError(f"缺少 {npz_path} 或 {csv_path}")
    with np.load(npz_path, allow_pickle=True) as data:
        stat = np.asarray(data["scaled_features"], dtype=np.float64)
        labels = np.asarray(data["labels"], dtype=np.int64)
        origin = np.asarray(data["unknown_origin"], dtype=np.int64) if "unknown_origin" in data else None
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return stat, labels, origin, df


def main():
    parser = argparse.ArgumentParser(description="阶段 2 输入验收（失败即退出，禁止进入训练）")
    parser.add_argument("--dataset_id", type=int, required=True)
    parser.add_argument("--split_id", type=int, required=True)
    parser.add_argument("--split_name", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--expected_numeric_dim", type=int, default=79)
    parser.add_argument("--max_overlap", type=float, default=0.0,
                        help="train/test 完全重叠率上限（默认 0，即不允许重叠）")
    args = parser.parse_args()

    split_dir, split_name = resolve_split_dir(args.dataset_id, args.split_id, args.split_name)
    if not os.path.isdir(split_dir):
        print(f"划分目录不存在: {split_dir}")
        sys.exit(1)

    ck = Checker()
    print("=" * 70)
    print(f"输入验收: dataset_{args.dataset_id} / {split_name}")
    print("=" * 70)

    loaded = {}
    for name in SPLITS:
        print(f"\n[{name}]")
        try:
            stat, labels, origin, df = load_split(split_dir, name)
        except FileNotFoundError as e:
            ck.check(f"{name} 文件齐全", False, str(e))
            continue
        loaded[name] = (stat, labels, origin, df)
        ck.check(f"{name} 文件齐全", True, f"npz={stat.shape}, csv={df.shape[0]}行")
        ck.check(f"{name} npz/csv 行数一致", stat.shape[0] == len(df),
                 f"npz={stat.shape[0]} csv={len(df)}")
        ck.check(f"{name} 数值维度 == {args.expected_numeric_dim}", stat.shape[1] == args.expected_numeric_dim,
                 f"实际 {stat.shape[1]}")
        ck.check(f"{name} 数值无 NaN/inf", bool(np.isfinite(stat).all()),
                 "存在非法值" if not np.isfinite(stat).all() else "")

        if origin is None:
            ck.check(f"{name} 含 unknown_origin", False, "npz 缺少 unknown_origin 字段")
            continue
        bad_codes = sorted(set(np.unique(origin).tolist()) - {-1, 0, 1, 2})
        ck.check(f"{name} unknown_origin 取值合法", not bad_codes, f"非法值 {bad_codes}")
        legacy = int(np.sum(origin == -1))
        ck.check(f"{name} 无 legacy origin(-1)", legacy == 0,
                 f"{legacy} 条来源未知，新划分必须显式标注" if legacy else "")

        known_mask = labels < args.num_classes
        unk_mask = labels == args.num_classes
        ck.check(f"{name} 标签落在 0..{args.num_classes}",
                 bool(np.all((labels >= 0) & (labels <= args.num_classes))),
                 f"min={int(labels.min())} max={int(labels.max())}")
        ck.check(f"{name} known 样本 origin 全为 0", bool(np.all(origin[known_mask] == 0)),
                 f"异常 {int(np.sum(origin[known_mask] != 0))} 条")
        ck.check(f"{name} unknown 样本 origin ∈ {{1,2}}", bool(np.all(np.isin(origin[unk_mask], (1, 2)))),
                 f"异常 {int(np.sum(~np.isin(origin[unk_mask], (1, 2))))} 条")

        n_known = int(known_mask.sum())
        n_gen = int(np.sum(origin == 1))
        n_sim = int(np.sum(origin == 2))
        ck.info(f"总 {len(labels)} = known {n_known} + genuine {n_gen} + simulated {n_sim}")

        numeric_columns = select_numeric_columns(df)
        ck.check(f"{name} CSV 数值列数 == npz 维度", len(numeric_columns) == stat.shape[1],
                 f"CSV {len(numeric_columns)} vs npz {stat.shape[1]}")

        csv_text = df["text_description"] if "text_description" in df.columns else None
        if csv_text is None:
            ck.check(f"{name} 含 text_description", False, "CSV 缺少 text_description")
        else:
            n_num_unique = int(df[numeric_columns].drop_duplicates().shape[0])
            n_txt_unique = int(csv_text.nunique(dropna=False))
            joint = pd.concat([df[numeric_columns], csv_text.rename("text_description")], axis=1)
            n_joint_unique = int(joint.drop_duplicates().shape[0])
            ck.info(f"唯一性: 数值 {n_num_unique}/{len(df)} ({n_num_unique / len(df):.2%}), "
                    f"文本 {n_txt_unique}/{len(df)} ({n_txt_unique / len(df):.2%}), "
                    f"联合 {n_joint_unique}/{len(df)} ({n_joint_unique / len(df):.2%})")

            group_ids = df.groupby(list(numeric_columns), sort=False, dropna=False).ngroup().to_numpy()
            grouped = pd.DataFrame({"gid": group_ids, "label": labels, "origin": origin})
            n_labels = grouped.groupby("gid")["label"].nunique()
            n_origins = grouped.groupby("gid")["origin"].nunique()
            max_group = int(grouped["gid"].value_counts().max())
            ck.check(f"{name} 无跨标签同输入组", int((n_labels > 1).sum()) == 0,
                     f"{int((n_labels > 1).sum())} 组跨标签（最大同输入组 {max_group} 条）")
            ck.check(f"{name} 无跨 origin 同输入组", int((n_origins > 1).sum()) == 0,
                     f"{int((n_origins > 1).sum())} 组跨 origin")

            illegal = find_illegal_text(csv_text)
            ck.check(f"{name} 文本无 label/origin 标记", int(illegal.sum()) == 0,
                     f"{int(illegal.sum())} 条含非法标记" + (
                         f"，示例: {csv_text[illegal].iloc[0][:60]!r}" if illegal.any() else ""))

    if "train" in loaded and "test" in loaded:
        print("\n[跨 split]")
        train_df = loaded["train"][3]
        test_df = loaded["test"][3]
        numeric_columns = select_numeric_columns(train_df)
        train_keys = set(map(tuple, train_df[numeric_columns].to_numpy().tolist()))
        test_keys = set(map(tuple, test_df[numeric_columns].to_numpy().tolist()))
        overlap = len(train_keys & test_keys) / max(1, len(test_keys))
        ck.check("train/test 完全重叠率 <= max_overlap", overlap <= args.max_overlap,
                 f"{overlap:.4%} (上限 {args.max_overlap:.4%})")

        if "text_description" in train_df.columns and "text_description" in test_df.columns:
            train_text = set(train_df["text_description"].astype(str).tolist())
            test_text = set(test_df["text_description"].astype(str).tolist())
            text_overlap = len(train_text & test_text) / max(1, len(test_text))
            ck.info(f"文本完全重叠率: {text_overlap:.2%}（同源样本共享文本属正常，"
                    f"但数值重叠必须为 0）")

    print("\n" + "=" * 70)
    if ck.failures:
        print(f"验收失败：{len(ck.failures)} 项不通过")
        for f in ck.failures:
            print(f"  - {f}")
        print("禁止进入阶段 3。")
        sys.exit(1)

    print("验收通过：可进入阶段 3（训练代表模型）。")
    print("=" * 70)


if __name__ == "__main__":
    main()
