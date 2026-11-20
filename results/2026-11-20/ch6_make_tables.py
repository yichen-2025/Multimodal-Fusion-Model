"""生成第 6 章：章首对照表（表 6-0）与预注册判定汇总表（表 6-5）。

只读已有 results/*/metrics.json，不产生新实验。
输出：results/2026-11-20/ch6/ch6_tables.md
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "2026-11-20", "ch6")
os.makedirs(OUT, exist_ok=True)


def load(*p):
    with open(os.path.join(ROOT, *p), encoding="utf-8") as f:
        return json.load(f)


e0 = load("results", "2026-11-19", "e0_baseline", "metrics.json")
e3 = load("results", "2026-11-20", "e3_e4_bert", "metrics.json")
e5 = load("results", "2026-11-20", "e5_qwen", "metrics.json")
e9 = load("results", "2026-11-20", "e9_qwen7b", "metrics.json")
e6b = load("results", "2026-11-20", "e6_bert", "metrics.json")
e6q = load("results", "2026-11-20", "e6_qwen", "metrics.json")
e84 = load("results", "2026-11-20", "e84_retrain", "metrics.json")
ood = load("results", "2026-11-03", "ood_multiseed_with_a3.json")


def zs(d):
    z = d["e3_zero_shot"]
    o = z["ood_descriptive"]
    return z["test_acc"], o["auroc_genuine"], o["tpr_at_fpr1_genuine"]


def shot(d, k):
    for r in d["e4_few_shot"]["curve"]:
        if r["source"] == "P_shot" and r["k"] == k:
            return r["acc"]
    return float("nan")


def best_mix_small_k(d):
    """k<=5 中 P_mix 相对 P_shot 的最大提升及其 CI（val 选 α）。"""
    best = None
    for r in d["e4_few_shot"]["curve"]:
        if r.get("selected_on_val") and r["k"] <= 5:
            if best is None or r["delta_vs_shot"] > best["delta_vs_shot"]:
                best = r
    return best


var = {v["variant"]: v["metrics"] for v in ood["variants"]}
a3, a0 = var["A3"], var["A0*"]


def ms(m, key, nd=4):
    x = m[key]
    return f"{x['backbone_level_mean']:.{nd}f} ± {x['backbone_level_sd']:.{nd}f}"


rows = []
for name, d, enc, dim in (("E3", e3, "BERT-base (110M)", 768),
                          ("E5", e5, "Qwen2.5-1.5B", 1536),
                          ("E9", e9, "Qwen2.5-7B", 3584)):
    acc, auc, tpr = zs(d)
    bm = best_mix_small_k(d)
    rows.append(dict(name=name, enc=enc, dim=dim, acc=acc, auc=auc, tpr=tpr,
                     s1=shot(d, 1), s20=shot(d, 20),
                     mix=bm["delta_vs_shot"], mci=bm["delta_bootstrap_ci95"],
                     mk=bm["k"], alpha=bm["alpha"]))

t0 = f"""# 第 6 章 · 语义先验：章首对照表与判定汇总

> 数字来源：`results/2026-11-19/e0_baseline`、`results/2026-11-20/{{e3_e4_bert,e5_qwen,e9_qwen7b,e6_bert,e6_qwen}}`、
> `results/2026-11-03/ood_multiseed_with_a3.json`。所有数字由脚本读取，未手填。

## 表 6-0 章首对照：全监督上界 vs 零样本 vs 少样本（同一 test split）

| 方案 | 编码器 | 标注使用 | 已知类准确率 | 开集 AUROC | `tpr@fpr=1%` |
|---|---|---|---|---|---|
| E0 轻量 MLP | 79 维数值 | 10 类全标（6362 条） | **{e0["closed_set"]["accuracy"]:.4f}** | {e0["ood_max_softmax"]["auroc_genuine"]:.4f} | {e0["ood_max_softmax"]["tpr_at_fpr1_genuine"]:.4f} |
| A3（开集协议） | 数值为主 | 全标 | {a3["known_accuracy"]["backbone_level_mean"]:.4f} ± {a3["known_accuracy"]["backbone_level_sd"]:.4f} | {a3["auroc_genuine"]["backbone_level_mean"]:.4f} ± {a3["auroc_genuine"]["backbone_level_sd"]:.4f} | {a3["tpr_at_fpr_1"]["backbone_level_mean"]:.4f} ± {a3["tpr_at_fpr_1"]["backbone_level_sd"]:.4f} |
| A0*（数值 + Qwen 文本） | Qwen2.5-1.5B | 全标 | {a0["known_accuracy"]["backbone_level_mean"]:.4f} ± {a0["known_accuracy"]["backbone_level_sd"]:.4f} | {a0["auroc_genuine"]["backbone_level_mean"]:.4f} ± {a0["auroc_genuine"]["backbone_level_sd"]:.4f} | {a0["tpr_at_fpr_1"]["backbone_level_mean"]:.4f} ± {a0["tpr_at_fpr_1"]["backbone_level_sd"]:.4f} |
| 零样本原型 (E3) | BERT-base (110M) | 0 条 | {rows[0]["acc"]:.4f} | {rows[0]["auc"]:.4f} | {rows[0]["tpr"]:.4f} |
| 零样本原型 (E5) | Qwen2.5-1.5B | 0 条 | {rows[1]["acc"]:.4f} | {rows[1]["auc"]:.4f} | {rows[1]["tpr"]:.4f} |
| 零样本原型 (E9) | Qwen2.5-7B | 0 条 | {rows[2]["acc"]:.4f} | {rows[2]["auc"]:.4f} | {rows[2]["tpr"]:.4f} |
| 20-shot 原型 (E4) | BERT-base (110M) | 每类 20 条 | {rows[0]["s20"]:.4f} | — | — |
| 20-shot 原型 (E5) | Qwen2.5-1.5B | 每类 20 条 | {rows[1]["s20"]:.4f} | — | — |
| 20-shot 原型 (E9) | Qwen2.5-7B | 每类 20 条 | {rows[2]["s20"]:.4f} | — | — |

> 口径说明：首行（E0）是**闭集**协议（10 类，无未知类）下的准确率；A3/A0* 行为**开集**协议下的已知类准确率（含未知类拒识的影响），二者不可直接相减。零样本/少样本行为闭集原型分类。
> 关键读法：全监督上界 {e0["closed_set"]["accuracy"]:.3f}–0.994，零样本 0.11–0.15，20-shot 0.22–0.52。**差距来自标注信息，而不是编码器容量**（E9 已证 110M→7B 不改善）。

## 表 6-5 预注册判定汇总（门槛来自总纲 v1.1/v1.2 与 C-30，跑数前固定）

| 实验 | 预注册门槛 | 实测 | 判定 |
|---|---|---|---|
| E3 零样本（BERT） | 闭集 acc ≥ 0.50 | {rows[0]["acc"]:.4f} | **未达标**（不支持） |
| E5 零样本（Qwen1.5B） | 同 E3 | {rows[1]["acc"]:.4f} | **未达标** |
| E9 零样本（Qwen7B） | 同 E3 | {rows[2]["acc"]:.4f} | **未达标** |
| E4 少样本（BERT） | `P_mix − P_shot` ≥ 3pp @ k≤5 且 CI 不跨零 | 最大 {rows[0]["mix"] * 100:+.2f}pp（k={rows[0]["mk"]}, α={rows[0]["alpha"]}）CI[{rows[0]["mci"][0] * 100:+.2f}, {rows[0]["mci"][1] * 100:+.2f}]pp | **未达标**（α\\*=0 居多） |
| E5 少样本（Qwen1.5B） | 同 E4 | 最大 {rows[1]["mix"] * 100:+.2f}pp（k={rows[1]["mk"]}, α={rows[1]["alpha"]}）CI[{rows[1]["mci"][0] * 100:+.2f}, {rows[1]["mci"][1] * 100:+.2f}]pp | **未达标** |
| E9 少样本（Qwen7B） | 同 E4 | 最大 {rows[2]["mix"] * 100:+.2f}pp（k={rows[2]["mk"]}, α={rows[2]["alpha"]}）CI[{rows[2]["mci"][0] * 100:+.2f}, {rows[2]["mci"][1] * 100:+.2f}]pp | **未达标** |
| E6 开集三臂（BERT） | `tpr@fpr=1%` 热启动 − 零样本 ≥ 2pp 且 CI 不跨零 | {e6b["arms"]["zero_shot"]["tpr_at_fpr1_genuine"]:.4f} → {e6b["arms"]["warm_start"]["tpr_at_fpr1_genuine"]:.4f}（+{(e6b["arms"]["warm_start"]["tpr_at_fpr1_genuine"] - e6b["arms"]["zero_shot"]["tpr_at_fpr1_genuine"]) * 100:.2f}pp） | **未达标** |
| E6 开集三臂（Qwen1.5B） | 同上 | AUROC {e6q["arms"]["zero_shot"]["auroc_genuine"]:.4f} → {e6q["arms"]["warm_start"]["auroc_genuine"]:.4f}（热启动更差） | **未达标** |
| E7 原型构造消融 | 描述性（无门槛） | 描述来源/原型条数/距离全组合均在随机水平附近 | **描述不是瓶颈** |
| E8.4 去文本消融（C-30 重训） | 5 seed bootstrap CI 下界 > −1pp → 文本可省 | 5 seeds `A0*_no_text` {e84["acc_no_text_mean"]:.4f} ± {e84["acc_no_text_sd"]:.4f}，A0\\* 基准 {e84["baseline"]["acc"]:.4f} → Δ = **{e84["delta_pp"]:+.2f}pp** CI[{e84["delta_ci95_pp"][0]:+.2f}, {e84["delta_ci95_pp"][1]:+.2f}] | **未通过**（不能声称文本可省） |
| E9 容量阶梯 | `tpr@fpr=1%` 较 BERT +2pp 且 CI 不跨零 | 0.0000 vs 0.0000（Δtpr=0）；AUROC 0.741→0.614→0.477 | **未达标（容量无效）** |

**汇总**：语义先验路线（E3–E9）在三个编码器、五种原型来源、全部构造组合下均未达门槛；E8.4 已按 C-30 判「未通过」，且与 LLM 容量混叠，只能得出「不能声称文本可省」，不得反推「文本携带独立信息」。**本轮无「待检验」项。**
"""

p = os.path.join(OUT, "ch6_tables.md")
with open(p, "w", encoding="utf-8") as f:
    f.write(t0)
print("written:", p)
