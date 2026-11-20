"""论文图 1 / 图 3 / 图 4（先于正文定稿）。

图 1  总体管线图：79 维数值 -> 20 字段模板 -> BERT 编码 -> 融合分类，下方挂 E8.1–E8.4 信息审计。
图 3  少样本曲线：k = 1/5/10/20，P_shot / P_mix(α*) / P_rand，P_llm 为水平基线（BERT、Qwen1.5B 双面板）。
图 4  开集 ROC 局部放大（FPR 0–10%），BERT-base / Qwen1.5B / Qwen7B 零样本原型分数 + 随机线。

数据：
  results/2026-11-20/{e3_e4_bert,e5_qwen}/curve.csv
  results/2026-11-20/paper_figs/roc_data.json（云端 /tmp/make_roc_data.py 重算，逐点 fpr/tpr）
  results/2026-11-19/e8_audit/metrics.json、e8_audit_hgb、2026-11-20/e83_perturbation{,_cross}、e84_retrain
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RES = os.path.join(ROOT, "results", "2026-11-20")
OUT = os.path.join(RES, "paper_figs")
os.makedirs(OUT, exist_ok=True)

BLUE, RED, GRAY, GREEN, GOLD = "#3B6EA5", "#C0504D", "#8C8C8C", "#2E7D32", "#8C6D31"
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 200,
})


def load(*p):
    with open(os.path.join(ROOT, *p), encoding="utf-8") as f:
        return json.load(f)


def read_curve(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "k": int(r["k"]),
                "source": r["source"],
                "alpha": None if r["alpha"] in ("", "None") else float(r["alpha"]),
                "acc": float(r["acc"]),
                "selected": bool(r["delta_vs_shot"]),
            })
    return rows


# ================= 图 1：总体管线图 =================
def fig1():
    e8 = load("results", "2026-11-19", "e8_audit", "metrics.json")
    e8h = load("results", "2026-11-19", "e8_audit_hgb", "metrics.json")
    ps = load("results", "2026-11-20", "e83_perturbation", "metrics.json")
    pc = load("results", "2026-11-20", "e83_perturbation_cross", "metrics.json")
    e84 = load("results", "2026-11-20", "e84_retrain", "metrics.json")
    r2_ft = e8["e81_predictability"]["features_to_text"]["test"]["r2_variance_weighted"]
    r2_tf = e8["e81_predictability"]["text_to_features"]["test"]["r2_variance_weighted"]
    i_lin = e8["e82_conditional_mi"]["I_text_y_given_features_nats"]
    i_hgb = e8h["e82_conditional_mi"]["I_text_y_given_features_nats"]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, text, fc="#FFFFFF", ec="#333333", fs=8.2, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.2",
                                    linewidth=1.0, edgecolor=ec, facecolor=fc, linestyle=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(p0, p1, ls="-", color="#333333", rad=0.0):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11,
                                     linewidth=1.0, color=color, linestyle=ls,
                                     connectionstyle=f"arc3,rad={rad}",
                                     shrinkA=2, shrinkB=2))

    # 主链
    box(2, 29, 19, 9, "79 维数值特征\n（标准化后）", fc="#EAF0F7")
    box(31, 33, 25, 8, "规则模板生成文本\n20 字段：端口 / 包长 /\n标志位 / 时长 …", fc="#FDF3E7")
    box(65, 33, 24, 8, "BERT-base（冻结）\n768 维文本嵌入", fc="#EAF0F7")
    box(31, 19, 25, 8, "融合投影（concat）\n79 ⊕ 768 → MLP", fc="#FFFFFF")
    box(65, 19, 24, 8, "10 类分类头\n+ 原型开集判定", fc="#FFFFFF")

    arrow((21.5, 33.5), (30.5, 36.5), rad=-0.15)
    arrow((56.5, 37), (64.5, 37))
    arrow((21.5, 33), (30.5, 24.5), rad=0.15)
    arrow((56.5, 23), (64.5, 23))
    arrow((44, 32.5), (44, 27.5))

    ax.text(2, 44.2, "(a) 前向管线：文本模态是数值特征的规则再编码", fontsize=9, weight="bold")

    # 审计带
    ax.add_patch(FancyBboxPatch((1.5, 1.2), 97, 13.6, boxstyle="round,pad=0.4,rounding_size=1.2",
                                linewidth=1.0, edgecolor="#7F7F7F", facecolor="#FAFAFA",
                                linestyle="--"))
    ax.text(3, 13.4, "(b) 信息审计（第 5 章，E8.1–E8.4）", fontsize=8.5, weight="bold", color="#444444")
    audits = [
        (3.5, f"E8.1 可预测性\nR²(feat→text)={r2_ft:.3f}\nR²(text→feat)={r2_tf:.3f}"),
        (27.5, f"E8.2 条件互信息\nlogreg {i_lin:+.3f} nats\nHGB {i_hgb:+.3f} nats"),
        (51.5, f"E8.3 扰动翻转率\n同类 {ps['text_flip_rate']:.2%}\n跨类 {pc['text_flip_rate']:.2%}"),
        (75.5, f"E8.4 去文本重训\nΔ = {e84['delta_pp']:+.2f} pp\nCI[{e84['delta_ci95_pp'][0]:.2f}, {e84['delta_ci95_pp'][1]:.2f}]"),
    ]
    for x, t in audits:
        box(x, 2.4, 21, 8.6, t, fc="#FFFFFF", ec="#BFBFBF", fs=7.4)

    arrow((11.5, 28.4), (11.5, 14.6), ls="--", color="#7F7F7F")
    # 文本分支走右侧走廊下行，避免穿过融合框、也不与文本→BERT 的实线重叠
    ax.plot([56.5, 60.7], [35.0, 35.0], color="#7F7F7F", lw=1.0, ls="--")
    arrow((60.7, 35.0), (60.7, 14.6), ls="--", color="#7F7F7F")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_pipeline.png"), bbox_inches="tight")
    plt.close(fig)


# ================= 图 3：少样本曲线 =================
def fig3():
    panels = [
        ("BERT-base (110M)", read_curve(os.path.join(RES, "e3_e4_bert", "curve.csv"))),
        ("Qwen2.5-1.5B", read_curve(os.path.join(RES, "e5_qwen", "curve.csv"))),
    ]
    ks = [1, 5, 10, 20]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=False)

    for ax, (name, rows) in zip(axes, panels):
        def series(source, selected_only=False, alpha=None):
            xs, ys = [], []
            for r in rows:
                if r["source"] != source:
                    continue
                if source == "P_mix":
                    if selected_only and not r["selected"]:
                        continue
                    if alpha is not None and r["alpha"] != alpha:
                        continue
                xs.append(r["k"])
                ys.append(r["acc"])
            return xs, ys

        ks_, shot = series("P_shot")
        _, mix = series("P_mix", selected_only=True)
        kr_, rand = series("P_rand")
        p_llm = [r["acc"] for r in rows if r["k"] == 0 and r["source"] == "P_llm"][0]
        p_rand0 = [r["acc"] for r in rows if r["k"] == 0 and r["source"] == "P_rand"][0]

        ax.plot(ks_, mix, "s--", color=RED, lw=1.2, ms=6, mfc="none", zorder=2,
                label=r"$P_{mix}$ ($\alpha^*$ selected on val)")
        ax.plot(ks_, shot, "o-", color=BLUE, lw=1.4, ms=4.5, zorder=3, label=r"$P_{shot}$")
        ax.plot([0] + kr_, [p_rand0] + rand, "^:", color=GRAY, lw=1.1, ms=4,
                label=r"$P_{rand}$")
        ax.axhline(p_llm, color=GREEN, lw=1.1, ls="-.",
                   label=rf"$P_{{llm}}$ (0-shot) = {p_llm:.3f}")

        # α* 标注：仅标出 α*>0 的位置；全为 0 时统一文字说明
        n_pos = 0
        for r in rows:
            if r["source"] == "P_mix" and r["selected"] and r["alpha"] > 0:
                ax.annotate(rf"$\alpha^*$={r['alpha']:.2f}", (r["k"], r["acc"]),
                            textcoords="offset points", xytext=(5, 6), fontsize=7, color=RED)
                n_pos += 1
        if n_pos == 0:
            ax.text(0.03, 0.94, r"$\alpha^*=0$ at all k", transform=ax.transAxes,
                    fontsize=7.5, color=RED)

        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.minorticks_off()
        ax.set_xlim(0.8, 26)
        ax.set_xlabel("Shots per class k")
        ax.set_ylabel("Closed-set accuracy (test)")
        ax.set_title(name, fontsize=9)
        ax.set_ylim(0.05, 0.60 if "BERT" in name else 0.32)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Few-shot prototypes: semantic priors add no measurable gain over shots",
                 fontsize=9)
    fig.legend(handles, labels, fontsize=7.2, frameon=False, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(OUT, "fig3_fewshot.png"), bbox_inches="tight")
    plt.close(fig)


# ================= 图 4：开集 ROC 局部 =================
def fig4():
    d = json.load(open(os.path.join(OUT, "roc_data.json"), encoding="utf-8"))
    order = [
        ("bert_base", "BERT-base (110M)", BLUE, "-"),
        ("qwen_1p5b", "Qwen2.5-1.5B", GOLD, "--"),
        ("qwen_7b", "Qwen2.5-7B", RED, "-."),
    ]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.plot([0, 0.10], [0, 0.10], color=GRAY, lw=0.9, ls=":", label="chance")

    for key, name, color, ls in order:
        v = d[key]
        fpr, tpr = np.array(v["fpr"]), np.array(v["tpr"])
        ax.plot(fpr, tpr, color=color, lw=1.5, ls=ls,
                label=f"{name}  (AUROC {v['auroc_detection_orientation_unknown_pos']:.3f})")

    ax.axvline(0.01, color=GRAY, lw=0.8, ls="--")
    ax.text(0.0115, 0.020, "FPR = 1%", fontsize=7, color=GRAY, rotation=90, va="bottom")
    ax.set_xlim(0, 0.10)
    ax.set_ylim(0, 0.07)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("False positive rate (known classes)")
    ax.set_ylabel("True positive rate (genuine unknown)")
    ax.legend(fontsize=6.8, frameon=False, loc="upper left")
    ax.set_title("Open-set ROC zoom: all encoders at TPR ≈ 0 at FPR = 1%", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_roc_zoom.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig3()
    fig4()
    print("saved ->", OUT)
