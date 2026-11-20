"""检查 1：确定性复核（审计协议 §4.2 第 1 项）。

目的
----
验证文本分支是否严格等于 20 个数值字段的确定性函数：
    text == generate_text_description(row)  ?

为什么重要
----------
若 100% 一致，则「文本不含特征之外的信息」（R ≈ 0 / 条件互信息 ≈ 0）
在相当程度上是**构造性必然**，而不是关于 LLM 文本能力的经验发现。
论文中必须按「在本文本构造下」表述，不能写成「LLM 文本无信息」。
本检查同时是「全 79 字段文本正例」（R>0 刻度校准）的前置条件。

判读
----
- match_rate == 1.0          → 文本 = f(20 字段)，即构造性冗余
- match_rate < 1.0           → 存在模板外信息，需逐条定位差异来源
- n_distinct_text < n_distinct_tuple → 文本对 20 字段有损（碰撞），
  说明部分数值差异在文本侧不可辨，需在论文中声明

口径
----
split_openset_0 / train·val·test 三个 CSV，逐条字符串精确比对。

用法
----
    python scripts/check_text_determinism.py
产物
----
    results/2026-11-20/e8_check1_determinism/metrics.json
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.flow_features import (  # noqa: E402
    TEXT_FEATURES,
    find_illegal_text,
    generate_text_description,
)

SPLIT_DIR = "split_data/dataset_200/split_openset_0"
OUT_DIR = "results/2026-11-20/e8_check1_determinism"
SPLITS = ("train", "val", "test")


def check_split(name):
    path = os.path.join(SPLIT_DIR, f"{name}_data.csv")
    df = pd.read_csv(path)

    missing = [c for c in TEXT_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: 缺少文本字段 {missing}")
    if "text_description" not in df.columns:
        raise ValueError(f"{name}: 缺少 text_description 列")

    expected = [generate_text_description(row) for _, row in df.iterrows()]
    actual = df["text_description"].astype(str).tolist()

    n = len(df)
    n_match = sum(e == a for e, a in zip(expected, actual))
    mismatches = [
        {"row": i, "expected": e[:160], "actual": a[:160]}
        for i, (e, a) in enumerate(zip(expected, actual))
        if e != a
    ][:3]

    # 碰撞检查：不同 20 字段元组是否映射到同一文本（有损）
    tuples = df[list(TEXT_FEATURES)].round(10).astype(str).agg("|".join, axis=1)
    n_distinct_tuple = int(tuples.nunique())
    n_distinct_text = int(df["text_description"].astype(str).nunique())

    illegal = find_illegal_text(df["text_description"])
    n_illegal = int(illegal.sum())

    return {
        "split": name,
        "path": path,
        "n_rows": n,
        "n_exact_match": n_match,
        "match_rate": n_match / n if n else None,
        "n_mismatch": n - n_match,
        "mismatch_examples": mismatches,
        "n_distinct_text": n_distinct_text,
        "n_distinct_field_tuple": n_distinct_tuple,
        "text_is_injective": n_distinct_text == n_distinct_tuple,
        "text_collision_rate": 1 - n_distinct_text / n if n else None,
        "n_illegal_token": n_illegal,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = [check_split(s) for s in SPLITS]

    total_rows = sum(r["n_rows"] for r in results)
    total_match = sum(r["n_exact_match"] for r in results)
    overall = total_match / total_rows if total_rows else None

    payload = {
        "experiment": "E8 检查1 确定性复核",
        "split_dir": SPLIT_DIR,
        "protocol": "text == generate_text_description(row)，逐条字符串精确比对（20 字段，:.4g）",
        "splits": results,
        "summary": {
            "n_rows_total": total_rows,
            "n_exact_match_total": total_match,
            "match_rate_overall": overall,
            "all_injective": all(r["text_is_injective"] for r in results),
            "n_illegal_token_total": sum(r["n_illegal_token"] for r in results),
        },
        "conclusion_rule": (
            "match_rate == 1.0 → 文本严格等于 20 字段的确定性函数（构造性冗余）；"
            "text_is_injective == False → 该函数为有损映射（:.4g 精度），"
            "文本信息量严格少于 20 字段。两者同时成立时，R≈0 是数据构造的"
            "直接后果，论文须按「本构造下」表述，并补 R>0 正例做刻度校准。"
        ),
        "interpretation": (
            "文本由 20 个数值字段按固定模板生成，因此「文本不含特征之外的标签信息」"
            "在很大程度上是数据构造的直接后果，不能外推为「LLM 文本本身无信息」。"
            "此外文本按 :.4g 格式化，是**有损**编码：多条 20 字段取值不同的样本"
            "会塌缩成同一文本，文本侧可辨信息严格少于数值侧。"
            "要获得判别力，必须补 R>0 正例（全 79 字段文本）作刻度校准。"
        ),
        "gate": {
            "determinism_match_rate_eq": 1.0,
            "pass": overall == 1.0,
        },
    }

    out_path = os.path.join(OUT_DIR, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print(f"[check1] 总条数 {total_rows}，精确一致 {total_match}，match_rate={overall}")
    for r in results:
        print(
            f"  {r['split']}: n={r['n_rows']} match={r['n_exact_match']} "
            f"({r['match_rate']:.6f}) distinct_text={r['n_distinct_text']} "
            f"distinct_tuple={r['n_distinct_field_tuple']} injective={r['text_is_injective']} "
            f"collision={r['text_collision_rate']:.4f} illegal={r['n_illegal_token']}"
        )
    print(f"[check1] 写入 {out_path}")


if __name__ == "__main__":
    main()
