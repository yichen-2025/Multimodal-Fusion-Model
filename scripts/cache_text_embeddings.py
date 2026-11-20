"""GPU 窗口专用：Qwen2.5-1.5B 文本编码（G1/G2）。

对应执行总纲 §2.3 白名单：
  G1  流样本文本编码（test 2040 + 池 1000）→ emb/qwen_{test,pool}.npz
  G2  类描述编码（10 类 × 3 条 = 30 条）→ emb/qwen_protos.npy

编码口径：Qwen2.5 为 decoder LM，取「最后一个非 pad token」的 last_hidden_state，
fp16 推理，max_length=256，padding_side=left。所有文本用同一口径，保证原型与流样本同空间。

子命令：
  csv      从 *_data.csv 生成待编码清单（含标签顺序校验），并抽样训练池
  encode   编码清单 → npz（ids + emb），每 200 条落盘，支持 --resume
  protos   编码类描述 → emb/qwen_protos.npy

用法：
  python scripts/cache_text_embeddings.py csv --split_dir split_data/dataset_200/split_openset_0 \
      --out emb/texts_to_encode.csv --pool_size 1000
  python scripts/cache_text_embeddings.py encode --csv emb/texts_to_encode.csv \
      --out emb/qwen_texts.npz --resume
  python scripts/cache_text_embeddings.py protos --split_dir split_data/dataset_200/split_openset_0 \
      --out emb/qwen_protos.npy
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

NUM_KNOWN = 10
SAVE_EVERY = 200


def cmd_csv(args):
    rows = []
    for split, n_take in (("test", None), ("val", None), ("train", args.pool_size)):
        path = os.path.join(args.split_dir, f"{split}_data.csv")
        with open(path, encoding="utf-8", errors="ignore") as f:
            recs = list(csv.DictReader(f))
        d = np.load(os.path.join(args.split_dir, f"{split}.npz"))
        y = d["labels"].astype(np.int64)
        if len(recs) != len(y):
            raise SystemExit(f"{split}: csv {len(recs)} 行 != npz {len(y)} 行")
        y_csv = np.array([int(r["Label"]) for r in recs], dtype=np.int64)
        if not np.array_equal(y_csv, y):
            raise SystemExit(f"{split}: csv 标签顺序与 npz 不一致，拒绝构建清单")
        if n_take is None:
            idx = np.arange(len(recs))
        else:
            idx = np.sort(np.random.default_rng(args.seed).choice(
                len(recs), min(n_take, len(recs)), replace=False))
        for i in idx:
            rows.append((f"{split}_{i}", recs[i]["text_description"]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "text"])
        w.writerows(rows)
    print(f"[OK] 清单 {len(rows)} 条 -> {args.out}")


def load_model(model_path):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.float16)
    model = model.to("cuda").eval()
    return tok, model


def encode_texts(texts, tok, model, batch_size, max_length):
    import torch
    out = []
    t0 = time.time()
    for i in range(0, len(texts), batch_size):
        b = texts[i:i + batch_size]
        inp = tok(b, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length).to(model.device)
        with torch.no_grad():
            h = model(**inp).last_hidden_state
        last = inp["attention_mask"].sum(dim=1) - 1
        e = h[torch.arange(h.size(0), device=h.device), last]
        out.append(e.float().cpu().numpy())
        done = min(i + batch_size, len(texts))
        if (i // batch_size) % 5 == 0 or done == len(texts):
            print(f"  {done}/{len(texts)}  {time.time() - t0:.1f}s", flush=True)
    return np.vstack(out)


def cmd_encode(args):
    with open(args.csv, encoding="utf-8") as f:
        recs = list(csv.DictReader(f))
    ids = [r["sample_id"] for r in recs]
    texts = [r["text"] for r in recs]

    done_ids, done_emb = np.array([], dtype="<U32"), np.zeros((0, 0), np.float32)
    if args.resume and os.path.exists(args.out):
        z = np.load(args.out, allow_pickle=False)
        done_ids, done_emb = z["ids"], z["emb"]
        print(f"[resume] 已有 {len(done_ids)} 条")

    todo = [(i, t) for i, t in zip(ids, texts) if i not in set(done_ids.tolist())]
    print(f"[encode] 待编码 {len(todo)} 条，模型 {args.model_path}")
    if not todo:
        print("[OK] 无需编码")
        return

    tok, model = load_model(args.model_path)
    buf_ids, buf_emb = [], []
    for j in range(0, len(todo), SAVE_EVERY):
        chunk = todo[j:j + SAVE_EVERY]
        emb = encode_texts([t for _, t in chunk], tok, model,
                           args.batch_size, args.max_length)
        buf_ids += [i for i, _ in chunk]
        buf_emb.append(emb)
        # 增量落盘
        all_ids = np.r_[done_ids, np.array(buf_ids, dtype="<U32")]
        all_emb = np.vstack(buf_emb) if len(done_ids) == 0 else np.vstack([done_emb] + buf_emb)
        np.savez(args.out, ids=all_ids, emb=all_emb.astype(np.float32))
        print(f"[save] {len(all_ids)}/{len(ids)} -> {args.out}", flush=True)

    z = np.load(args.out, allow_pickle=False)
    assert len(z["ids"]) == len(ids), (len(z["ids"]), len(ids))
    assert not np.isnan(z["emb"]).any(), "存在 NaN"
    print(f"[OK] {z['emb'].shape} -> {args.out}")


def cmd_protos(args):
    from scripts.build_semantic_prototypes import build_descriptions, CLASS_NAMES

    info = build_descriptions()
    texts = []
    for name in CLASS_NAMES:
        texts += info[name]["descriptions"]

    tok, model = load_model(args.model_path)
    E = encode_texts(texts, tok, model, args.batch_size, args.max_length)
    E = E.reshape(NUM_KNOWN, 3, -1).mean(axis=1).astype(np.float32)
    assert E.shape[0] == NUM_KNOWN, E.shape
    assert not np.isnan(E).any()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, E)
    os.makedirs(args.meta_dir, exist_ok=True)
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    pair_cos = float((En @ En.T)[np.triu_indices(NUM_KNOWN, 1)].mean())
    with open(os.path.join(args.meta_dir, "qwen_protos_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "G2 Qwen 类描述编码",
            "model_path": args.model_path,
            "encoder": "Qwen2.5-1.5B AutoModel, last non-pad token, fp16, max_length=%d" % args.max_length,
            "n_classes": NUM_KNOWN,
            "n_desc_per_class": 3,
            "class_names": CLASS_NAMES,
            "pairwise_cosine_mean": pair_cos,
            "artifact": args.out,
        }, f, ensure_ascii=False, indent=2)
    print(f"[OK] Qwen 原型 {E.shape} -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("csv")
    p1.add_argument("--split_dir", required=True)
    p1.add_argument("--out", required=True)
    p1.add_argument("--pool_size", type=int, default=1000)
    p1.add_argument("--seed", type=int, default=42)

    p2 = sub.add_parser("encode")
    p2.add_argument("--csv", required=True)
    p2.add_argument("--out", required=True)
    p2.add_argument("--model_path", default="models/qwen2.5-1.5b")
    p2.add_argument("--batch_size", type=int, default=16)
    p2.add_argument("--max_length", type=int, default=256)
    p2.add_argument("--resume", action="store_true")

    p3 = sub.add_parser("protos")
    p3.add_argument("--split_dir", required=True)
    p3.add_argument("--out", default="emb/qwen_protos.npy")
    p3.add_argument("--meta_dir", default="results/2026-11-20/prototypes")
    p3.add_argument("--model_path", default="models/qwen2.5-1.5b")
    p3.add_argument("--batch_size", type=int, default=8)
    p3.add_argument("--max_length", type=int, default=256)

    args = ap.parse_args()
    {"csv": cmd_csv, "encode": cmd_encode, "protos": cmd_protos}[args.mode](args)


if __name__ == "__main__":
    main()
