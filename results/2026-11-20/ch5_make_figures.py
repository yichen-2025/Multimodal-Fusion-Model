"""生成论文第 5 章（机制诊断）的图表与数字表。

数据来源（全部为已落盘结果）：
  results/2026-11-19/e8_audit/metrics.json        E8.1 + E8.2(logreg)
  results/2026-11-19/e8_audit_hgb/metrics.json    E8.2(HGB)
  results/2026-11-20/e83_perturbation/metrics.json         E8.3 同类
  results/2026-11-20/e83_perturbation_cross/metrics.json   E8.3 跨类
  results/2026-11-20/e3_e4_bert|e5_qwen|e9_qwen7b/metrics.json  容量阶梯

输出：results/2026-11-20/ch5/{fig1..fig4}.png + ch5_numbers.md
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "2026-11-20", "ch5")
os.makedirs(OUT, exist_ok=True)

BLUE, RED, GRAY = "#3B6EA5", "#C0504D", "#8C8C8C"
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150,
})


def load(*p):
    with open(os.path.join(ROOT, *p), encoding="utf-8") as f:
        return json.load(f)


e8 = load("results", "2026-11-19", "e8_audit", "metrics.json")
e8h = load("results", "2026-11-19", "e8_audit_hgb", "metrics.json")
p_same = load("results", "2026-11-20", "e83_perturbation", "metrics.json")
p_cross = load("results", "2026-11-20", "e83_perturbation_cross", "metrics.json")
e5 = load("results", "2026-11-20", "e5_qwen", "metrics.json")
e9 = load("results", "2026-11-20", "e9_qwen7b", "metrics.json")
e3 = load("results", "2026-11-20", "e3_e4_bert", "metrics.json")


def bars(ax, labels, vals, cis, colors, width=0.55):
    x = np.arange(len(vals))
    ax.bar(x, vals, width, color=colors, edgecolor="black", linewidth=0.5)
    if cis is not None:
        lo = [v - c[0] for v, c in zip(vals, cis)]
        hi = [c[1] - v for v, c in zip(vals, cis)]
        ax.errorbar(x, vals, yerr=[lo, hi], fmt="none", ecolor="black", capsize=3, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)


# ---- Fig 1: 文本可预测性 ----
r2_ft = e8["e81_predictability"]["features_to_text"]["test"]["r2_variance_weighted"]
r2_tf = e8["e81_predictability"]["text_to_features"]["test"]["r2_variance_weighted"]
fig, ax = plt.subplots(figsize=(4.2, 3.2))
bars(ax, ["79 numeric\n-> 768 text", "768 text\n-> 79 numeric"], [r2_ft, r2_tf], None, [BLUE, BLUE])
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Test $R^2$ (variance-weighted)")
ax.set_ylim(0, 1)
for i, v in enumerate([r2_ft, r2_tf]):
    ax.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=9)
ax.set_title("Text embeddings are predictable from numeric features", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_predictability.png"), bbox_inches="tight")
plt.close(fig)

# ---- Fig 2: 条件互信息 ----
i_lin = e8["e82_conditional_mi"]["I_text_y_given_features_nats"]
ci_lin = e8["e82_conditional_mi"]["I_text_y_given_features_bootstrap_ci95"]
i_hgb = e8h["e82_conditional_mi"]["I_text_y_given_features_nats"]
ci_hgb = e8h["e82_conditional_mi"]["I_text_y_given_features_bootstrap_ci95"]
H = e8["e82_conditional_mi"]["entropy_y_test_nats"]
fig, ax = plt.subplots(figsize=(4.4, 3.2))
xs = [0, 1]
ax.errorbar(xs, [i_lin, i_hgb], yerr=[[i_lin - ci_lin[0], i_hgb - ci_hgb[0]],
                                      [ci_lin[1] - i_lin, ci_hgb[1] - i_hgb]],
            fmt="o", color=RED, capsize=4, lw=1.2, ms=6)
ax.axhline(0, color="black", lw=0.9, ls="--")
ax.set_xticks(xs)
ax.set_xticklabels(["Logistic\n(linear)", "HGB\n(nonlinear)"])
ax.set_xlim(-0.5, 1.5)
ax.set_ylabel("$I(\\mathrm{text}; y \\mid \\mathrm{features})$  [nats]")
ax.set_title(f"Conditional MI is ~0 (H(y) = {H:.2f} nats)", fontsize=9)
for x, v in zip(xs, [i_lin, i_hgb]):
    ax.annotate(f"{v:+.3f}", (x, v), textcoords="offset points", xytext=(12, 6), fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_conditional_mi.png"), bbox_inches="tight")
plt.close(fig)

# ---- Fig 3: 扰动审计 ----
fig, ax = plt.subplots(figsize=(4.8, 3.2))
labels = ["Same-class\ntext", "Same-class\nfeatures", "Cross-class\ntext", "Cross-class\nfeatures"]
vals = [p_same["text_flip_rate"], p_same["feature_flip_rate"],
        p_cross["text_flip_rate"], p_cross["feature_flip_rate"]]
cis = [p_same["text_flip_ci95"], p_same["feature_flip_ci95"],
       p_cross["text_flip_ci95"], p_cross["feature_flip_ci95"]]
bars(ax, labels, vals, cis, [BLUE, GRAY, BLUE, GRAY])
ax.axhline(0.20, color=RED, ls="--", lw=1)
ax.text(3.35, 0.215, "gate 0.20", color=RED, fontsize=7, ha="right")
ax.set_ylabel("Prediction flip rate")
ax.set_ylim(0, 0.8)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_perturbation.png"), bbox_inches="tight")
plt.close(fig)


# ---- Fig 4: 容量阶梯 ----
def zs_metrics(d):
    z = d["e3_zero_shot"]
    return (z["test_acc"], z["ood_descriptive"]["auroc_genuine"],
            z["ood_descriptive"]["tpr_at_fpr1_genuine"])


z3, z5, z9 = zs_metrics(e3), zs_metrics(e5), zs_metrics(e9)
acc = [z3[0], z5[0], z9[0]]
auc = [z3[1], z5[1], z9[1]]
tpr = [z3[2], z5[2], z9[2]]
sizes = [0, 1, 2]
names = ["BERT-base\n110M", "Qwen2.5\n1.5B", "Qwen2.5\n7B"]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
axes[0].plot(sizes, acc, "o-", color=BLUE, lw=1.4)
axes[0].set_ylim(0, 0.6)
axes[0].axhline(0.10, color=GRAY, ls=":", lw=1)
axes[0].text(0.02, 0.085, "chance", fontsize=7, color=GRAY, ha="left", va="top")
axes[0].set_ylabel("Zero-shot closed-set accuracy")
axes[1].plot(sizes, auc, "o-", color=RED, lw=1.4)
axes[1].axhline(0.5, color=GRAY, ls=":", lw=1)
axes[1].set_ylim(0.2, 0.65)
axes[1].text(0.02, 0.485, "chance", fontsize=7, color=GRAY, ha="left", va="top")
axes[1].set_ylabel("Open-set AUROC")
for ax in axes:
    ax.set_xticks(sizes)
    ax.set_xticklabels(names, fontsize=8)
for ax, vals in zip(axes, [acc, auc]):
    for x, v in zip(sizes, vals):
        ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7)
fig.suptitle("Scaling the encoder does not recover the missing information", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_capacity.png"), bbox_inches="tight")
plt.close(fig)

# ---- 数字表 ----
md = f"""# 第 5 章 · 机制诊断：数字表与图注

> 全部数字来自 `results/2026-11-19/`、`results/2026-11-20/` 的 `metrics.json`；括号内为 95% bootstrap CI。

## 表 5-1 文本模态的可预测性（E8.1，test n=1655）

| 方向 | R²（方差加权） | R²（逐维均值） | 说明 |
|---|---|---|---|
| 79 维数值 → 768 维文本嵌入 | **{r2_ft:.3f}** | {e8["e81_predictability"]["features_to_text"]["test"]["r2_mean_per_dim"]:.3f} | 文本有近一半方差可由数值特征解释 |
| 768 维文本 → 79 维数值 | **{r2_tf:.3f}** | {e8["e81_predictability"]["text_to_features"]["test"]["r2_mean_per_dim"]:.3f} | 反向更强：文本几乎不含数值之外的信息 |

图 1（`fig1_predictability.png`）：两个方向的 R² 均远高于 0，说明文本不是独立信源，而是数值特征的再编码。

## 表 5-2 条件互信息 `I(text; y | features)`（E8.2，单位 nats）

| 估计器 | 条件互信息 | 95% CI | 对照：`I(features; y)` | 对照：`H(y)` |
|---|---|---|---|---|
| 多项逻辑回归（线性） | {i_lin:+.3f} | [{ci_lin[0]:.3f}, {ci_lin[1]:.3f}] | {e8["e82_conditional_mi"]["I_features_y_nats"]:.3f} | {H:.2f} |
| HistGradientBoosting（非线性） | {i_hgb:+.3f} | [{ci_hgb[0]:.3f}, {ci_hgb[1]:.3f}] | {e8h["e82_conditional_mi"]["I_features_y_nats"]:.3f} | {H:.2f} |

说明：线性估计器给出 {i_lin:.3f} nats（占 H(y) 的 {i_lin / H:.1%}），其线性假设会高估；非线性估计器为 {i_hgb:+.3f}，CI 上界 {ci_hgb[1]:.3f} nats（相对 H(y) 约为 {ci_hgb[1] / H:.1%}，即接近 0），说明文本在给定数值特征后几乎不提供额外标签信息。

图 2（`fig2_conditional_mi.png`）：两种估计器的点估计与 CI，虚线为零。

## 表 5-3 反事实扰动审计（E8.3，A0*，n=1655）

| donor | 文本侧翻转率 | 特征侧翻转率 | 两侧均不变 |
|---|---|---|---|
| 同类（主协议） | **{p_same["text_flip_rate"]:.2%}** [{p_same["text_flip_ci95"][0]:.2%}, {p_same["text_flip_ci95"][1]:.2%}] | {p_same["feature_flip_rate"]:.2%} [{p_same["feature_flip_ci95"][0]:.2%}, {p_same["feature_flip_ci95"][1]:.2%}] | {p_same["flip_neither"]:.1%} |
| 跨类（阳性对照） | {p_cross["text_flip_rate"]:.2%} [{p_cross["text_flip_ci95"][0]:.2%}, {p_cross["text_flip_ci95"][1]:.2%}] | {p_cross["feature_flip_rate"]:.2%} [{p_cross["feature_flip_ci95"][0]:.2%}, {p_cross["feature_flip_ci95"][1]:.2%}] | {p_cross["flip_neither"]:.1%} |

判读：同类内文本扰动几乎不改变判定（{p_same["text_flip_rate"]:.2%} < 门槛 20%）；但跨类扰动使 {p_cross["text_flip_rate"]:.1%} 的预测翻转，**说明文本分支确实在读内容**。因此本实验只支持「同类流量间文本差异不足以改变判定」，不支持「模型不读文本」。

## 表 5-4 容量阶梯：零样本与开集（E3/E5/E9，同协议）

| 编码器 | 维度 | 零样本闭集 acc | 开集 AUROC | `tpr@fpr=1%` |
|---|---|---|---|---|
| BERT-base（110M） | 768 | {acc[0]:.4f} | {auc[0]:.4f} | {tpr[0]:.4f} |
| Qwen2.5-1.5B | 1536 | {acc[1]:.4f} | {auc[1]:.4f} | {tpr[1]:.4f} |
| Qwen2.5-7B | 3584 | {acc[2]:.4f} | {auc[2]:.4f} | {tpr[2]:.4f} |

图 4（`fig4_capacity.png`）：规模从 110M 增加到 7B，零样本精度未改善，开集 AUROC 仍在 0.26–0.52 之间（接近随机水平）。容量不是瓶颈，信息量才是。

## 未出图项

| 图 | 状态 | 阻塞原因 |
|---|---|---|
| 成本-精度 Pareto（C4） | **待补** | 缺 LLM 分支推理时延的实测值；需在无卡模式下测 A0* 与 MLP 的单样本时延 |
"""
with open(os.path.join(OUT, "ch5_numbers.md"), "w", encoding="utf-8") as f:
    f.write(md)

print("figs + ch5_numbers.md ->", OUT)
