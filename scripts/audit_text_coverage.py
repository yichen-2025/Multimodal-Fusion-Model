"""E10 文本覆盖审计：从 split CSV 重建文本 → BERT 编码 → 写新 split 目录。

背景：既有 20 字段模板（`generate_text_description`）只覆盖 79 个数值列中的 20 个，
且检查1 已证明「文本 = f(20 字段)」。于是 E8 的 `I(text;y|features) ≈ 0` 可能只是
构造决定的必然，而不是数据的性质。本脚本构造两个对照目录：

- `--fields all79`   ：文本覆盖全部 79 个数值列，`scaled_features` 保持 79 维
                       → 检验冗余结论对字段覆盖是否鲁棒（上界是否被人为压低）。
- `--fields nontext59`：文本只覆盖 20 个 `TEXT_FEATURES` 之外的 59 个字段，
                       `scaled_features` 切到 20 维 → 人为正例，给 C2 判据装刻度。
- `--fields text20`   ：文本只覆盖 20 个 `TEXT_FEATURES`，`scaled_features` 保持 79 维，
                       且精度用 `.6g`（E8 原模板为 `.4g`，有损碰撞 8.24%）
                       → 与 full79 同模板同精度，隔离「字段覆盖」；与 E8 对照隔离「编码精度」。

编码口径与 `text_embeddings` 缓存一致：`BertEncoder`（`[CLS]`、无池化、无归一化）。
原目录只读，脚本启动即断言 `out_dir != split_dir`。

用法：
  python scripts/audit_text_coverage.py \
      --split_dir split_data/dataset_200/split_openset_0 \
      --out_dir   split_data/dataset_200/split_openset_0_full79 \
      --fields all79 --device cpu --max_length 512 --batch_size 32
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_architectures.bert_encoder import BertEncoder
from utils.flow_features import (
    TEXT_FEATURES,
    generate_compact_text,
    select_numeric_columns,
)

PARTS = ("train", "val", "test")
TEXT_COLUMN = "text_description"


def token_lengths(tokenizer, texts):
    """返回每条文本的 token 数（含特殊 token），用于截断率自检。"""
    lens = np.empty(len(texts), dtype=np.int64)
    for i, t in enumerate(texts):
        lens[i] = len(tokenizer(t, add_special_tokens=True)["input_ids"])
    return lens


def sha256_of_texts(texts):
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def resolve_fields(df, mode):
    """返回 (文本字段列表, 保留的 scaled_features 列索引或 None)。"""
    numeric = select_numeric_columns(df)
    if mode == "all79":
        return numeric, None
    missing = [c for c in TEXT_FEATURES if c not in numeric]
    if missing:
        raise ValueError(f"TEXT_FEATURES 不在数值列中: {missing}")
    keep_idx = [numeric.index(c) for c in TEXT_FEATURES]
    if mode == "text20":
        return [c for c in numeric if c in set(TEXT_FEATURES)], None
    text_fields = [c for c in numeric if c not in set(TEXT_FEATURES)]
    return text_fields, keep_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fields", required=True, choices=["all79", "nontext59", "text20"])
    ap.add_argument("--model_path", default="models/bert")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--limit_rows", type=int, default=0, help="仅冒烟用；0 表示全量")
    ap.add_argument("--value_format", default=".6g")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    split_dir = os.path.realpath(args.split_dir)
    out_dir = os.path.realpath(args.out_dir)
    if split_dir == out_dir:
        raise SystemExit("拒绝执行：out_dir 与原 split_dir 相同，原目录只读")
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # 1) 读 CSV，确定文本字段与保留的特征列
    frames = {p: pd.read_csv(os.path.join(split_dir, f"{p}_data.csv")) for p in PARTS}
    text_fields, keep_idx = resolve_fields(frames["train"], args.fields)
    if TEXT_COLUMN not in frames["train"].columns:
        raise SystemExit(f"CSV 缺少文本列 {TEXT_COLUMN}")

    # 2) 生成文本
    texts = {}
    for p in PARTS:
        df = frames[p]
        if args.limit_rows:
            df = df.iloc[: args.limit_rows]
        texts[p] = [
            generate_compact_text(row, text_fields, value_format=args.value_format)
            for _, row in df.iterrows()
        ]
        frames[p] = df

    # 3) 自检：确定性（抽 200 行重新生成并逐条比对）+ 截断率 + 非法标记
    n_det = 0
    n_det_ok = 0
    for p in PARTS:
        df = frames[p]
        k = min(200, len(df))
        idx = rng.choice(len(df), size=k, replace=False)
        for i in idx:
            again = generate_compact_text(
                df.iloc[i], text_fields, value_format=args.value_format
            )
            n_det += 1
            n_det_ok += int(again == texts[p][i])

    encoder = BertEncoder(
        local_model_path=args.model_path,
        batch_size=args.batch_size,
        force_cpu=(args.device == "cpu"),
    )
    tokenizer = encoder.get_tokenizer()

    tok_stats = {}
    trunc_total = 0
    rows_total = 0
    for p in PARTS:
        lens = token_lengths(tokenizer, texts[p])
        trunc = int((lens > args.max_length).sum())
        trunc_total += trunc
        rows_total += len(lens)
        tok_stats[p] = {
            "n": int(len(lens)),
            "p50": int(np.percentile(lens, 50)),
            "p95": int(np.percentile(lens, 95)),
            "p99": int(np.percentile(lens, 99)),
            "max": int(lens.max()) if len(lens) else 0,
            "over_max_length": trunc,
        }

    # 4) 编码并落盘 npz
    text_dim = None
    for p in PARTS:
        src = np.load(os.path.join(split_dir, f"{p}.npz"))
        emb = encoder(texts[p], max_length=args.max_length).numpy()
        if not np.isfinite(emb).all():
            raise SystemExit(f"{p}: 嵌入含 NaN/inf")
        if text_dim is None:
            text_dim = int(emb.shape[1])

        features = src["scaled_features"]
        labels = src["labels"]
        unknown_origin = src["unknown_origin"]
        calib = src["unknown_calibration_mixed"] if "unknown_calibration_mixed" in src.files \
            else None
        if args.limit_rows:
            n = len(texts[p])
            features, labels, unknown_origin = features[:n], labels[:n], unknown_origin[:n]
        if keep_idx is not None:
            features = features[:, keep_idx]

        out = {
            "scaled_features": features,
            "text_embeddings": emb,
            "labels": labels,
            "unknown_origin": unknown_origin,
        }
        if calib is not None:
            out["unknown_calibration_mixed"] = calib
        np.savez(os.path.join(out_dir, f"{p}.npz"), **out)

        df = frames[p].copy()
        df[TEXT_COLUMN] = texts[p]
        df.to_csv(os.path.join(out_dir, f"{p}_data.csv"), index=False)
        print(f"[{p}] n={len(df)} features={features.shape[1]} text={emb.shape}")

    # 5) scaler 原样复制（只作参考，pos59 的特征维度已不同）
    src_scaler = os.path.join(split_dir, "train_scaler.npy")
    if os.path.exists(src_scaler):
        shutil.copy2(src_scaler, os.path.join(out_dir, "train_scaler.npy"))
    label_map = os.path.join(split_dir, "label_mapping.json")
    if os.path.exists(label_map):
        with open(label_map, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        with open(os.path.join(out_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)

    # 6) manifest
    manifest = {
        "experiment": "E10 文本覆盖审计",
        "source_split_dir": split_dir,
        "out_dir": out_dir,
        "fields_mode": args.fields,
        "text_fields": text_fields,
        "n_text_fields": len(text_fields),
        "feature_dim": int(len(keep_idx)) if keep_idx is not None else 79,
        "text_dim": text_dim,
        "value_format": args.value_format,
        "template": "positional_key_base36 + '=' + value, single space separator",
        "tokenizer": args.model_path,
        "encoder": "BertEncoder([CLS]), 无池化/无归一化",
        "device": encoder.device.type,
        "dtype": "fp16" if encoder.device.type == "cuda" else "fp32",
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "limit_rows": args.limit_rows,
        "token_stats": tok_stats,
        "truncation_rate": float(trunc_total / rows_total) if rows_total else 0.0,
        "determinism": f"{n_det_ok}/{n_det}",
        "text_sha256": {p: sha256_of_texts(texts[p]) for p in PARTS},
        "rows": {p: int(len(texts[p])) for p in PARTS},
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps({k: manifest[k] for k in
                      ("fields_mode", "n_text_fields", "feature_dim", "text_dim",
                       "device", "max_length", "truncation_rate", "determinism")},
                     ensure_ascii=False))
    if manifest["truncation_rate"] > 0:
        raise SystemExit("截断率 > 0：文本 ≠ f(字段)，按计划 §6-R1 压缩模板后重跑")
    if text_dim != 768:
        raise SystemExit(f"文本维度异常: {text_dim} != 768")


if __name__ == "__main__":
    main()
