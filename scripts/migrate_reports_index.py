"""一次性迁移 reports_index.csv 到新口径 schema。

功能：
1. 补齐 INDEX_COLUMNS 中的缺失列（split_name / openset / num_classes / metric_scope）；
2. 从对应 report_{id}.json 反填 num_classes、openset、split_name、metric_scope；
3. 迁移前自动备份原文件到 reports_index.csv.bak（仅首次）。

注意：
- openset 依据 test_samples 推断（2040=开集 / 2000=闭集），无法推断的行（如历史 mock 行）留空；
- 旧开集报告一律标 metric_scope=mixed-unknown(deprecated)，禁止与真实 OOD 指标并列。
"""

import os
import sys
import json
import shutil

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from utils.caliber import INDEX_COLUMNS


def _backfill_row(df, i, report_json_path):
    """从单个报告 JSON 反填一行。"""
    if not os.path.exists(report_json_path):
        return df

    with open(report_json_path, encoding="utf-8") as f:
        data = json.load(f)

    # num_classes（保留报告原值，供 verify_caliber 校验与重跑判别）
    if not str(df.at[i, "num_classes"]).strip():
        df.at[i, "num_classes"] = str(data.get("num_classes", ""))

    # openset 依据 test_samples 推断
    ts = data.get("test_samples")
    openset = None
    if ts == 2040:
        openset = True
    elif ts == 2000:
        openset = False

    sid = data.get("split_id", 0)

    if openset is not None and not str(df.at[i, "openset"]).strip():
        df.at[i, "openset"] = "True" if openset else "False"

    if not str(df.at[i, "split_name"]).strip():
        if openset is True:
            df.at[i, "split_name"] = f"split_openset_{sid}"
        elif openset is False:
            df.at[i, "split_name"] = f"split_{sid}"

    if not str(df.at[i, "metric_scope"]).strip():
        if openset is True:
            df.at[i, "metric_scope"] = "mixed-unknown(deprecated)"
        elif openset is False:
            df.at[i, "metric_scope"] = "closed"

    return df


def main():
    idx_path = os.path.join(PROJECT_ROOT, "test_reports", "reports_index.csv")
    report_dir = os.path.join(PROJECT_ROOT, "test_reports")

    if not os.path.exists(idx_path):
        print("reports_index.csv 不存在，无内容可迁移")
        return

    df = pd.read_csv(idx_path, dtype=str)

    # 1. 补齐缺失列
    for col in INDEX_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # 2. 逐行反填
    for i in range(len(df)):
        rid = df.at[i, "report_id"] if "report_id" in df.columns else None
        if rid is None or not str(rid).strip():
            continue
        report_json = os.path.join(report_dir, f"report_{rid}.json")
        df = _backfill_row(df, i, report_json)

    # 3. 备份 + 写回（列顺序：INDEX_COLUMNS 在前，遗留列在后）
    bak = idx_path + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(idx_path, bak)

    ordered = [c for c in INDEX_COLUMNS] + [c for c in df.columns if c not in INDEX_COLUMNS]
    df = df[ordered]
    df.to_csv(idx_path, index=False, encoding="utf-8-sig")

    print(f"迁移完成，共 {len(df)} 行")
    print(f"  - 新增列: {[c for c in INDEX_COLUMNS if c not in pd.read_csv(bak, nrows=0, dtype=str).columns]}")
    print(f"  - 原文件已备份至: {bak}")


if __name__ == "__main__":
    main()