"""C4：成本-精度 Pareto 图（fig5）+ 表 7-1。

横轴：单样本 p50 延迟（ms，对数）；纵轴：test 已知类准确率。
每个点的准确率分别来自 E0、E3/E5 的 20-shot 与 A0* 的 E8.3 原始准确率。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BASE = os.path.join(ROOT, "results", "2026-11-20")
OUT = os.path.join(BASE, "c4_latency")


def load(*p):
    with open(os.path.join(ROOT, *p), encoding="utf-8") as f:
        return json.load(f)


lat = load("results", "2026-11-20", "c4_latency", "metrics.json")["points"]
acc_e0 = load("results", "2026-11-19", "e0_baseline", "metrics.json")["closed_set"]["accuracy"]
e3 = load("results", "2026-11-20", "e3_e4_bert", "metrics.json")
e5 = load("results", "2026-11-20", "e5_qwen", "metrics.json")
p_same = load("results", "2026-11-20", "e83_perturbation", "metrics.json")


def shot(d, k):
    return [r["acc"] for r in d["e4_few_shot"]["curve"] if r["source"] == "P_shot" and r["k"] == k][0]


pts = [
    ("MLP (79-d), CPU", lat["mlp_cpu1"]["p50_ms"], acc_e0, "o", "#3B6EA5"),
    ("MLP (79-d), GPU", lat["mlp_gpu1"]["p50_ms"], acc_e0, "o", "#7FA6C9"),
    ("BERT protos, 20-shot", lat["bert_gpu1"]["p50_ms"], shot(e3, 20), "s", "#C0504D"),
    ("Qwen1.5B protos, 20-shot", lat["a0_gpu1"]["p50_ms"], shot(e5, 20), "^", "#8C6D31"),
    ("A0* (num + Qwen1.5B)", lat["a0_gpu1"]["p50_ms"], p_same["original_acc"], "D", "#2E7D32"),
]

fig, ax = plt.subplots(figsize=(6.4, 3.8))
offsets = {"MLP (79-d), CPU": (0, 10, "center"), "MLP (79-d), GPU": (8, -16, "left"),
           "BERT protos, 20-shot": (8, 6, "left"), "Qwen1.5B protos, 20-shot": (-8, 8, "right"),
           "A0* (num + Qwen1.5B)": (-10, -14, "right")}
for name, x, y, mk, c in pts:
    ax.scatter(x, y, marker=mk, s=52, color=c, zorder=3, edgecolor="black", linewidth=0.5)
    dx, dy, ha = offsets[name]
    ax.annotate(f"{name}\n{x:.2f} ms / {y:.3f}", (x, y), textcoords="offset points",
                xytext=(dx, dy), fontsize=7, ha=ha)

ax.set_xscale("log")
ax.set_xlabel("Single-sample latency, p50 (ms, log scale)")
ax.set_ylabel("Known-class accuracy (test)")
ax.set_ylim(0.1, 1.18)
ax.set_xlim(0.02, 200)
ax.grid(alpha=0.25, which="both")
ax.set_title("Cost-accuracy frontier saturates at the 79-d MLP", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_pareto.png"), bbox_inches="tight")
plt.close(fig)

rows = "\n".join(
    f"| {n} | {x:.3f} | {y:.4f} | {x / lat['mlp_cpu1']['p50_ms']:.1f}× |" for n, x, y, _, _ in pts)
md = f"""# 第 7 章 · 成本-精度（C4）：延迟实测与 Pareto

测量条件：batch=1、warmup 20、max_length 256、
硬件 {load("results", "2026-11-20", "c4_latency", "metrics.json")["protocol"]["hardware"]}、
每次迭代 `torch.cuda.synchronize()`；n=1000（小模型）/ 200（LLM）。

## 表 7-1 单样本延迟与精度

| 方案 | p50 延迟 (ms) | 已知类准确率 | 相对 MLP(CPU) 倍数 |
|---|---|---|---|
{rows}

补充：A0* 的 p95 = {lat['a0_gpu1']['p95_ms']:.2f} ms；tokenizer 单独耗时 {lat['tok_only']['p50_ms']:.2f} ms；
MLP 的 CPU 测量为单线程（OMP_NUM_THREADS=1），是最保守的对照。
Qwen1.5B 原型与 A0* 共用同一编码器前向，故 p50 相同；差别在头部（余弦 vs 融合 MLP）。

## 读法

- 前沿在 **79 维 MLP**（0.035 ms，0.9855）处已经饱和：
  往右 1000 倍成本（A0*，{lat['a0_gpu1']['p50_ms']:.1f} ms）只换来 **+{(p_same['original_acc'] - acc_e0) * 100:.2f}pp** 绝对精度。
- 原型路线（BERT/Qwen 编码 + 余弦）的成本由**编码器前向**决定：
  BERT 20-shot 在 {lat['bert_gpu1']['p50_ms']:.1f} ms 上只有 {shot(e3, 20):.3f}，
  即「更贵且更差」，落在前沿的右下方。
- 结论（C4）：工程上应部署数值分支；LLM 分支只在无法获得标注且可接受 0.13–0.23 精度时才有讨论空间。

图：`fig5_pareto.png`
"""
with open(os.path.join(OUT, "c4_tables.md"), "w", encoding="utf-8") as f:
    f.write(md)
print("ok")
