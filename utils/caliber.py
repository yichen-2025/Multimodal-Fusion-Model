"""口径真相源模块。

只做两件事：
1. 从模型配置读取唯一的标签空间（num_classes）；
2. 提供 fail-fast 断言，口径不一致时直接抛异常，禁止静默继续。

约定（仅此一处定义，任何脚本不得自行解释）：
- label_space   = 模型 config.txt 中的 num_classes，取值 0..K-1
- known_accuracy   = 在 label < K 样本上，argmax 预测等于真实标签的比例（分类正确率）
- known_retention  = 在 label < K 样本上，预测仍落在 0..K-1 的比例（= 1 - 误拒率）
- unknown_recall   = 在 label >= K 样本上，预测 == K 的比例
- macro_f1_known   = 只在 label < K 样本、只在 0..K-1 标签集合上计算
"""

import os
import json

import numpy as np


class CaliberError(RuntimeError):
    """口径不一致时抛出，禁止静默继续。"""


def read_model_num_classes(model_dir: str) -> int:
    """从模型 config.txt 读取 num_classes。禁止从测试集反推。"""
    cfg = os.path.join(model_dir, "config.txt")
    if not os.path.exists(cfg):
        raise CaliberError(f"缺少模型配置: {cfg}，禁止从测试集反推 num_classes")
    with open(cfg, encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("num_classes:"):
                return int(line.split(":", 1)[1].strip())
    raise CaliberError(f"{cfg} 中未找到 num_classes")


def read_openset_flag(model_dir: str) -> bool:
    """读取模型 config.txt 中的 openset 标志（可选，缺省 False）。"""
    cfg = os.path.join(model_dir, "config.txt")
    if not os.path.exists(cfg):
        return False
    with open(cfg, encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("openset:"):
                val = line.split(":", 1)[1].strip().lower()
                return val in ("true", "1", "yes", "on")
    return False


def assert_classifier_dim(model, num_classes: int):
    """断言分类头维度与配置一致。"""
    out_dim = getattr(model, "num_classes", None)
    if out_dim is None:
        raise CaliberError("模型对象没有 num_classes 属性，拒绝推断")
    if int(out_dim) != int(num_classes):
        raise CaliberError(f"分类头维度 {out_dim} != 配置 num_classes {num_classes}")


def assert_label_space(labels, num_classes: int, allow_unknown: bool, split_name: str):
    """断言标签空间合法（闭集不允许 unknown，开集 unknown 必须恰好等于 K）。"""
    labels = np.asarray(labels)
    if labels.size == 0:
        raise CaliberError(f"{split_name}: 空标签")
    if labels.min() < 0:
        raise CaliberError(f"{split_name}: 出现负标签")
    max_label = int(labels.max())
    if not allow_unknown and max_label >= num_classes:
        raise CaliberError(
            f"{split_name}: 含 label={max_label} >= num_classes={num_classes}，"
            f"但当前模式未声明 open-set。请传 --openset 或改测 split_openset_*"
        )
    if allow_unknown and max_label >= num_classes + 1:
        raise CaliberError(
            f"{split_name}: unknown 标签应恰好等于 {num_classes}，实际 {max_label}"
        )


# ── 报告 schema 校验 ──
REQUIRED_REPORT_FIELDS = {
    "split_name", "num_classes", "label_space", "openset",
    "metric_scope", "unknown_origin_available",
}

INDEX_COLUMNS = [
    "report_id", "timestamp", "model_id", "dataset_id", "split_id",
    "split_name", "openset", "num_classes", "metric_scope",
    "test_samples", "accuracy", "precision", "recall", "f1",
    "duration_seconds", "variant",
    "use_numeric", "use_bert", "use_llm", "fusion_type",
    "bert_trainable", "trainable_params", "llm_use_lora",
]


def validate_report(report: dict):
    """报告落盘前校验口径字段，缺失即抛异常。"""
    missing = REQUIRED_REPORT_FIELDS - set(report)
    if missing:
        raise CaliberError(f"报告缺少口径字段: {sorted(missing)}")
    if report["openset"] and "unknown_recall_genuine" not in report:
        raise CaliberError("open-set 报告缺少 unknown_recall_genuine")
    if report.get("unknown_origin_available") is False and report["openset"]:
        report["metric_scope"] = "mixed-unknown(deprecated)"
        print("⚠️ 旧划分：所有 OOD 指标为混合口径，禁止与真实 OOD 结果并列")


def assert_index_schema(path: str):
    """校验 reports_index.csv 是否已迁移到新 schema，禁止混写。"""
    import pandas as pd

    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    missing = set(INDEX_COLUMNS) - set(df.columns)
    if missing:
        raise CaliberError(
            f"reports_index.csv 缺少列 {sorted(missing)}；"
            f"请先运行 scripts/migrate_reports_index.py 迁移，禁止混写"
        )