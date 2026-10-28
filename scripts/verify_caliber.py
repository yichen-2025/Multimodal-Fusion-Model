"""口径一致性校验（验收标准）。

对 test_reports/ 下的评估报告与 reports_index.csv 做 fail-fast 校验：
1. reports_index.csv 列 schema 是否完整（assert_index_schema）；
2. 每个报告 JSON 是否含口径字段（split_name / num_classes / openset 等）；
   口径修复前生成的 legacy 报告（缺 label_space 与 metric_scope）只告警，不硬失败；
3. open-set 报告是否含 unknown_recall_genuine；
4. 报告 num_classes 是否与模型 config.txt 一致；
5. reports_index.csv 是否出现同模型同划分同配置的重复行（警告）。

任一硬校验失败则退出码非 0。
"""

import os
import sys
import json

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from audit_artifacts import check_artifact_consistency
from utils.caliber import (
    CaliberError,
    REQUIRED_REPORT_FIELDS,
    assert_index_schema,
    read_model_num_classes,
)

# 口径修复后新增的两个字段。两者同时缺失 => 该报告由修复前的 test_model 生成，
# 属 legacy 报告：只告警跳过，避免历史文件让整个校验永久退出码非 0。
# 只要出现"部分缺失"（只缺其中一个或只缺部分字段），仍按硬错误处理，防止新报告漏字段。
LEGACY_MARKERS = ("label_space", "metric_scope")


def deprecated_report_ids():
    """台账中已标 deprecated 的报告 id（含 report_ 前缀），重复行检测时排除。"""
    path = os.path.join(PROJECT_ROOT, "artifacts", "ledger.csv")
    if not os.path.exists(path):
        return set()
    led = pd.read_csv(path, dtype=str).fillna("")
    sel = led[(led["kind"] == "test_report") & (led["status"] == "deprecated")]
    return set(sel["id"].astype(str).tolist())


def main():
    errors = []
    warnings = []

    report_dir = os.path.join(PROJECT_ROOT, "test_reports")
    idx_path = os.path.join(report_dir, "reports_index.csv")

    # 1. 索引 schema
    try:
        assert_index_schema(idx_path)
        print("[OK] reports_index.csv 列 schema 完整")
    except CaliberError as e:
        errors.append(str(e))

    # 2. 逐报告 JSON 校验
    report_names = sorted(
        f for f in os.listdir(report_dir)
        if f.startswith("report_") and f.endswith(".json")
    )
    for rname in report_names:
        rpath = os.path.join(report_dir, rname)
        try:
            with open(rpath, encoding="utf-8") as f:
                rep = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            errors.append(f"{rname}: 无法解析报告 JSON ({e})")
            continue

        missing = REQUIRED_REPORT_FIELDS - set(rep)
        is_legacy = all(k not in rep for k in LEGACY_MARKERS)
        if missing:
            if is_legacy:
                warnings.append(
                    f"{rname}: 旧口径(legacy)报告，缺 {sorted(missing)}，"
                    f"跳过全部口径校验；如需纳入结论请用当前 test_model 重跑"
                )
                continue
            else:
                errors.append(f"{rname}: 缺少口径字段 {sorted(missing)}")
                continue

        if not is_legacy and rep.get("openset") and "unknown_recall_genuine" not in rep:
            errors.append(f"{rname}: open-set 报告缺少 unknown_recall_genuine")

        mid = rep.get("model_id")
        if mid is not None and rep.get("num_classes") is not None:
            mdir = os.path.join(PROJECT_ROOT, "saved_models", f"model_{mid}")
            if os.path.exists(mdir):
                try:
                    cfg_nc = read_model_num_classes(mdir)
                    if int(rep["num_classes"]) != cfg_nc:
                        errors.append(
                            f"{rname}: 报告 num_classes={rep['num_classes']} != "
                            f"模型 config.txt num_classes={cfg_nc}"
                        )
                except CaliberError as e:
                    warnings.append(f"{rname}: {e}")

    # 3. 重复行检测（台账已标 deprecated 的行不再计入：重复运行是设计内行为）
    if os.path.exists(idx_path):
        df = pd.read_csv(idx_path, dtype=str)
        dup_keys = [c for c in ["model_id", "dataset_id", "split_id", "openset", "num_classes"] if c in df.columns]
        if dup_keys:
            dep = deprecated_report_ids()
            dup = df[df.duplicated(subset=dup_keys, keep=False)]
            if not dup.empty:
                active = dup[~dup["report_id"].astype(str).map(
                    lambda x: x in dep or f"report_{x}" in dep)]
                active_dup = active[active.duplicated(subset=dup_keys, keep=False)]
                if not active_dup.empty:
                    ids = sorted(active_dup["report_id"].astype(str).tolist())
                    warnings.append(f"reports_index.csv 重复行（同模型同划分同配置，均未标 deprecated）: report_id={ids}")

    # 4. 产物引用一致性：报告引用的 model/split 是否存在、active 产物是否在 live、台账是否自洽
    art_errors, art_warnings = check_artifact_consistency()
    errors.extend(art_errors)
    warnings.extend(art_warnings)

    print("=" * 60)
    if errors:
        print(f"[FAIL] 发现 {len(errors)} 个口径问题:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("[PASS] 所有口径硬校验通过")

    if warnings:
        print(f"[WARN] {len(warnings)} 个警告:")
        for w in warnings:
            print(f"  - {w}")
    print("=" * 60)

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()