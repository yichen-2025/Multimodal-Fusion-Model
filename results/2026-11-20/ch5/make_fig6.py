"""图 6：E8.4 去文本消融（5 seeds no_text vs A0* 与 E0 参考线）。"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, "results", "2026-11-20", "ch5")

with open(os.path.join(ROOT, "results", "2026-11-20", "e84_retrain", "metrics.json"), encoding="utf-8") as f:
    m = json.load(f)

seeds = m["seeds"]
accs = [m["acc_no_text"][s] for s in seeds]
mean = m["acc_no_text_mean"]
base = m["baseline"]["acc"]
e0 = m["e0_reference"]["acc"]
lo = base + m["delta_ci95_pp"][0] / 100.0
hi = base + m["delta_ci95_pp"][1] / 100.0

fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.scatter(range(len(seeds)), accs, s=46, color="#3B6EA5", zorder=3,
           edgecolor="black", linewidth=0.5, label="A0*_no_text (per seed)")
ax.errorbar([len(seeds)], [mean], yerr=[[mean - lo], [hi - mean]], fmt="D",
            color="#1F4E79", ms=6, capsize=5, zorder=3, label="no_text mean, 95% CI")
ax.axhline(base, color="#2E7D32", lw=1.4, label=f"A0* baseline ({base:.4f})")
ax.axhline(e0, color="#8C6D31", lw=1.1, ls="--", label=f"E0 numeric MLP ({e0:.4f})")

ax.set_xticks(list(range(len(seeds))) + [len(seeds)])
ax.set_xticklabels([f"s{s}" for s in seeds] + ["mean"])
ax.set_ylabel("Known-class accuracy (test)")
ax.set_ylim(0.966, 0.999)
ax.grid(axis="y", alpha=0.25)
ax.legend(fontsize=7, loc="lower right", frameon=False)
ax.set_title(f"Removing the text branch: $\\Delta$ = {m['delta_pp']:.2f}pp "
             f"CI[{m['delta_ci95_pp'][0]:.2f}, {m['delta_ci95_pp'][1]:.2f}]", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig6_text_ablation.png"), bbox_inches="tight")
plt.close(fig)
print("saved fig6")
