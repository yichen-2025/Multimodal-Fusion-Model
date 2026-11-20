"""C4：成本-精度 Pareto 的延迟实测（batch=1，单流在线形态）。

测量项
  mlp_cpu1   : 79 维小 MLP（79→256→10），CPU、OMP_NUM_THREADS=1
  mlp_gpu1   : 同结构在 GPU 上（用于说明小模型的 GPU 开销主要是调度）
  bert_gpu1  : BERT-base 原始文本 → 池化向量（在线编码成本）
  a0_gpu1    : A0*（数值 + Qwen1.5B + LoRA）端到端，含 tokenize
  tok_only   : 仅 tokenizer 耗时（用于拆分）

口径
  - batch=1；warmup 后测 n 次，每次 torch.cuda.synchronize()；
  - 报 p50 / p95（ms）；
  - 同一实例、同一驱动，横轴可直接比较。

用法
  python scripts/bench_latency.py --split_dir split_data/dataset_200/split_openset_0 \
      --llm_path models/qwen2.5-1.5b --model_dir saved_models/model_2013 \
      --out results/2026-11-20/c4_latency
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def pct(xs, q):
    return float(np.percentile(xs, q))


def bench(fn, n, warmup, cuda):
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return {"n": n, "p50_ms": pct(ts, 50), "p95_ms": pct(ts, 95),
            "mean_ms": float(np.mean(ts))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--llm_path", default="models/qwen2.5-1.5b")
    ap.add_argument("--model_dir", default="saved_models/model_2013")
    ap.add_argument("--bert_path", default="models/bert")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--n_small", type=int, default=1000)
    ap.add_argument("--n_llm", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(42)

    d = np.load(os.path.join(args.split_dir, "test.npz"))
    df = pd.read_csv(os.path.join(args.split_dir, "test_data.csv"))
    X = d["scaled_features"].astype(np.float32)
    texts = df["text_description"].astype(str).tolist()
    x1 = torch.tensor(X[:1])
    t1 = texts[0]

    res = {"experiment": "C4 成本-精度 Pareto：延迟实测",
           "protocol": {"batch": 1, "warmup": 20, "max_length": args.max_length,
                        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu-only",
                        "notes": "单流在线形态；含 tokenize（a0/bert）；同一实例内比较"},
           "points": {}}

    # ---- 小 MLP（79→256→10）----
    torch.manual_seed(0)
    mlp = torch.nn.Sequential(torch.nn.Linear(79, 256), torch.nn.ReLU(), torch.nn.Linear(256, 10))

    def run_mlp(model, dev):
        with torch.no_grad():
            model(x1.to(dev))

    res["points"]["mlp_cpu1"] = bench(lambda: run_mlp(mlp, "cpu"), args.n_small, 20, False)
    if torch.cuda.is_available():
        mlp_gpu = mlp.cuda()
        res["points"]["mlp_gpu1"] = bench(lambda: run_mlp(mlp_gpu, "cuda"), args.n_small, 20, True)

    # ---- BERT 在线编码 ----
    from transformers import AutoModel, AutoTokenizer

    btok = AutoTokenizer.from_pretrained(args.bert_path, local_files_only=True)
    bmodel = AutoModel.from_pretrained(args.bert_path, local_files_only=True).eval()
    if torch.cuda.is_available():
        bmodel = bmodel.cuda()

    def run_bert():
        bt = btok([t1], return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_length)
        if torch.cuda.is_available():
            bt = bt.to("cuda")
        with torch.no_grad():
            out = bmodel(**bt)
        _ = out.last_hidden_state[:, -1]

    res["points"]["bert_gpu1" if torch.cuda.is_available() else "bert_cpu1"] = bench(
        run_bert, args.n_llm, 20, torch.cuda.is_available())
    del bmodel
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- A0*（数值 + Qwen1.5B + LoRA）端到端 ----
    from transformers import AutoTokenizer as AT
    from src.model_architectures.multi_modal_model import MultiModalFusionModel

    model = MultiModalFusionModel.from_pretrained(
        llm_model_path_or_save_dir=args.llm_path, save_dir=args.model_dir)
    model.eval()
    dev = model.device
    tok = AT.from_pretrained(args.llm_path, local_files_only=True)

    def run_a0():
        bt = tok([t1], return_tensors="pt", padding=True, truncation=True,
                 max_length=args.max_length).to(dev)
        with torch.no_grad():
            out = model(stat_tensor=x1.to(dev), bert_tensor=None,
                        input_ids=bt["input_ids"], attention_mask=bt["attention_mask"])
        _ = out["logits"]

    res["points"]["a0_gpu1" if torch.cuda.is_available() else "a0_cpu1"] = bench(
        run_a0, args.n_llm, 10, torch.cuda.is_available())

    def run_tok():
        tok([t1], return_tensors="pt", padding=True, truncation=True,
            max_length=args.max_length)

    res["points"]["tok_only"] = bench(run_tok, args.n_small, 20, False)

    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    for k, v in res["points"].items():
        print(f"{k:12s} p50={v['p50_ms']:7.3f} p95={v['p95_ms']:7.3f} (n={v['n']})")


if __name__ == "__main__":
    main()
