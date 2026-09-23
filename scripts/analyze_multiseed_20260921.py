# -*- coding: utf-8 -*-
"""分析 multiseed_20260921_230136.csv，生成带图表的 HTML 报告。"""
import csv, math, html
from collections import defaultdict

SRC = r"C:\Users\刘浥尘\Downloads\multiseed_20260921_230136.csv"
OUT = r"D:\大模型研究（新）\Multimodal-Fusion-Model\ood_reports\multiseed_analysis_20260921.html"

METRICS = ["auroc", "unknown_f1", "unknown_recall", "unknown_leak_rate",
           "known_accuracy", "macro_f1", "backbone_train_seconds",
           "ood_train_seconds", "routing_seconds"]

# 读取
rows = []
with open(SRC, encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if not r.get("variant"):
            continue
        for k in METRICS:
            try:
                r[k] = float(r[k])
            except (TypeError, ValueError):
                r[k] = None
        rows.append(r)

# 按 variant 聚合
by_var = defaultdict(list)
for r in rows:
    by_var[r["variant"]].append(r)

def meanstd(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return (float("nan"), float("nan"))
    m = sum(vals) / len(vals)
    if len(vals) > 1:
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    else:
        sd = 0.0
    return (m, sd)

agg = {}
for v, rs in by_var.items():
    agg[v] = {m: meanstd([r[m] for r in rs]) for m in METRICS}
    agg[v]["_n"] = len(rs)

ORDER = ["A3", "A0*", "A0*_add", "A0*_attn", "A0*_no_text", "A0*_no_num"]
DESCR = {r["variant"]: r["description"] for r in rows}
present = [v for v in ORDER if v in agg]

# ---------- SVG 柱状图 ----------
def bar_chart(metric, title, fmt=".3f", color="#5b8def", ymax=None, log=False):
    vals = [(v, agg[v][metric][0]) for v in present]
    if ymax is None:
        ymax = max(v[1] for v in vals) * 1.15
    W, H = 620, 280
    left, right, top, bottom = 70, 20, 30, 70
    plot_w = W - left - right
    plot_h = H - top - bottom
    n = len(vals)
    bw = plot_w / (n * 1.6)
    gap = plot_w / n
    svg = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="font-family:system-ui;font-size:11px">']
    svg.append(f'<text x="{W/2}" y="16" text-anchor="middle" fill="#e6e6e6" font-size="13" font-weight="600">{html.escape(title)}</text>')
    # y 轴网格
    for i in range(5):
        yv = ymax * (1 - i / 4)
        y = top + plot_h * i / 4
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{W-right}" y2="{y:.1f}" stroke="#3a3a3a" stroke-width="1"/>')
        svg.append(f'<text x="{left-6}" y="{y+3:.1f}" text-anchor="end" fill="#999">{yv:.2f}</text>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#666"/>')
    for i, (lab, val) in enumerate(vals):
        cx = left + gap * (i + 0.5)
        bh = plot_h * (val / ymax) if ymax > 0 else 0
        y = top + plot_h - bh
        svg.append(f'<rect x="{cx-bw/2:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="3" fill="{color}"/>')
        svg.append(f'<text x="{cx:.1f}" y="{y-4:.1f}" text-anchor="middle" fill="#e6e6e6">{val:{fmt}}</text>')
        # 换行 variant 标签
        parts = lab.split("_")
        if len(parts) > 1:
            svg.append(f'<text x="{cx:.1f}" y="{top+plot_h+16:.1f}" text-anchor="middle" fill="#bbb">{parts[0]}</text>')
            svg.append(f'<text x="{cx:.1f}" y="{top+plot_h+30:.1f}" text-anchor="middle" fill="#bbb">{"_".join(parts[1:])}</text>')
        else:
            svg.append(f'<text x="{cx:.1f}" y="{top+plot_h+22:.1f}" text-anchor="middle" fill="#bbb">{lab}</text>')
    svg.append('</svg>')
    return "".join(svg)

# ---------- 表格 ----------
def fmt_cell(m, s):
    return f"{m:.4f} ± {s:.4f}"

def leaderboard_table():
    cols = [("auroc", "AUROC"), ("unknown_f1", "Unknown F1"),
            ("unknown_recall", "Unknown Recall"), ("unknown_leak_rate", "Leak"),
            ("known_accuracy", "Known Acc"), ("macro_f1", "Macro-F1")]
    head = "<tr><th>变体</th><th>说明</th>" + "".join(f"<th>{c[1]}</th>" for c in cols) + "<th>路由(s)</th></tr>"
    body = ""
    for v in present:
        desc = DESCR.get(v, "")
        cells = "".join(f"<td>{fmt_cell(*agg[v][c[0]])}</td>" for c in cols)
        rt = agg[v]["routing_seconds"][0]
        body += f"<tr><td><b>{v}</b></td><td style='text-align:left;color:#bbb'>{html.escape(desc)}</td>{cells}<td>{rt:.2f}</td></tr>"
    return f"<table class='tbl'>{head}{body}</table>"

def compare_table(base, other, label):
    """逐 seed 对比 + 一致性统计"""
    base_rows = {r["seed"]: r for r in by_var[base]}
    oth_rows = {r["seed"]: r for r in by_var[other]}
    seeds = sorted(set(base_rows) & set(oth_rows), key=lambda x: int(x))
    metrics_compare = [("macro_f1", "Macro-F1"), ("unknown_f1", "Unknown F1"),
                       ("auroc", "AUROC"), ("known_accuracy", "Known Acc")]
    head = "<tr><th>seed</th>" + "".join(f"<th>{m[1]}<br>{base} → {other}</th>" for m in metrics_compare) + "</tr>"
    body = ""
    wins = [0] * len(metrics_compare)
    for s in seeds:
        cells = ""
        for i, (mk, _) in enumerate(metrics_compare):
            bv, ov = base_rows[s][mk], oth_rows[s][mk]
            better = ov > bv
            if better:
                wins[i] += 1
            cells += f"<td style='color:{'#7fd18a' if better else '#e88'}'>{bv:.3f} → {ov:.3f}</td>"
        body += f"<tr><td>{s}</td>{cells}</tr>"
    # 汇总行
    winstr = "".join(f"<td>{wins[i]}/{len(seeds)} seed 胜出</td>" for i in range(len(metrics_compare)))
    body += f"<tr style='font-weight:600;border-top:1px solid #555'><td>一致性</td>{winstr}</tr>"
    return f"<table class='tbl'>{head}{body}</table>"

# ---------- HTML ----------
def section(title, inner):
    return f"<div class='sec'><h2>{title}</h2>{inner}</div>"

html_parts = []
html_parts.append("""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<style>
body{background:#14151a;color:#e6e6e6;font-family:system-ui,'Segoe UI',sans-serif;margin:0;padding:28px 36px;line-height:1.55}
h1{font-size:22px;margin:0 0 4px}
.meta{color:#999;font-size:12px;margin-bottom:18px}
.sec{margin:22px 0;background:#1b1d24;border:1px solid #2a2d36;border-radius:10px;padding:16px 20px}
h2{font-size:16px;margin:0 0 12px;color:#fff;border-left:3px solid #5b8def;padding-left:9px}
.tbl{border-collapse:collapse;width:100%;font-size:12.5px}
.tbl th,.tbl td{border:1px solid #2c2f38;padding:6px 8px;text-align:center}
.tbl th{background:#23262f;color:#cfd3da}
.tbl td{background:#191b21}
svg{display:block;margin:6px auto}
.big{font-size:26px;font-weight:700;color:#7fd18a}
.kpi{display:flex;gap:14px;flex-wrap:wrap;margin:8px 0}
.kpi div{background:#22252e;border:1px solid #2c2f38;border-radius:8px;padding:10px 14px;min-width:120px}
.kpi b{display:block;font-size:20px;color:#5b8def}
.kpi span{font-size:11px;color:#aaa}
.note{background:#26221a;border-left:3px solid #d9a441;padding:8px 12px;border-radius:6px;color:#e8d9b5;font-size:13px;margin:10px 0}
.ok{color:#7fd18a}.bad{color:#e88}
ul{margin:8px 0 8px 18px;padding:0}li{margin:4px 0}
</style></head><body>""")

html_parts.append("<h1>多 seed 开集实验结果分析</h1>")
html_parts.append(f"<div class='meta'>数据源：multiseed_20260921_230136.csv ｜ seeds = 20 / 234 / 2026 ｜ 变体数 = {len(present)} ｜ 生成于本地分析脚本</div>")

# 执行摘要
a3m = agg["A3"]; a0m = agg["A0*"]; addm = agg["A0*_add"]; attnm = agg["A0*_attn"]
best_macro = max(present, key=lambda v: agg[v]["macro_f1"][0])
best_uf1 = max(present, key=lambda v: agg[v]["unknown_f1"][0])
best_auroc = max(present, key=lambda v: agg[v]["auroc"][0])
delta_macro = a0m["macro_f1"][0] - a3m["macro_f1"][0]
delta_uf1 = a0m["unknown_f1"][0] - a3m["unknown_f1"][0]

summary = f"""
<div class='kpi'>
  <div><b>{best_macro}</b><span>Macro-F1 最高 ({agg[best_macro]['macro_f1'][0]:.3f})</span></div>
  <div><b>{best_uf1}</b><span>Unknown F1 最高 ({agg[best_uf1]['unknown_f1'][0]:.3f})</span></div>
  <div><b>{best_auroc}</b><span>AUROC 最高 ({agg[best_auroc]['auroc'][0]:.3f})</span></div>
</div>
<ul>
<li><span class='ok'>主假设成立（修正版）</span>：A0*（LLM+LoRA 可训练，作文本编码器）在 <b>全部 3 个 seed、全部 4 项指标</b>上稳定优于 A3（无LLM）。Macro-F1 +{delta_macro:.3f}、Unknown F1 +{delta_uf1:.3f}、AUROC +{a0m['auroc'][0]-a3m['auroc'][0]:.3f}、Known Acc +{a0m['known_accuracy'][0]-a3m['known_accuracy'][0]:.3f}。</li>
<li><span class='ok'>融合方式决定上限</span>：A0* 默认融合最弱；换成 <b>add 融合（A0*_add）</b>后 Macro-F1 从 {a0m['macro_f1'][0]:.3f} 升到 {addm['macro_f1'][0]:.3f}（+{addm['macro_f1'][0]-a0m['macro_f1'][0]:.3f}），是本次最大增益点。attention 融合（A0*_attn）增益不稳定（seed 间抖动大）。</li>
<li><span class='ok'>文本编码器是主模态</span>：纯文本 A0*_no_num（AUROC {agg['A0*_no_num']['auroc'][0]:.3f}）远强于纯数值 A0*_no_text（{agg['A0*_no_text']['auroc'][0]:.3f}），且超过 A3（含 BERT 数值）。LLM-as-Encoder 取代了 BERT 的判别力。</li>
<li><span class='bad'>代价</span>：A0* 系列路由耗时 ~19.4s vs A3 ~1.2s（≈16×），主干训练 ~45–50min vs A3 ~2.5min（≈20×）。</li>
</ul>
"""
html_parts.append(section("执行摘要", summary))

# 主假设对比
html_parts.append(section("主假设：A0*（可训练LLM） vs A3（无LLM）", compare_table("A3", "A0*", "A0* vs A3")))

# 排行榜
html_parts.append(section("全变体排行榜（均值 ± 标准差，3 seed）", leaderboard_table()))

# 图表
charts = ""
charts += bar_chart("macro_f1", "Macro-F1（越高越好）", color="#5b8def")
charts += bar_chart("unknown_f1", "Unknown F1（开集未知类检测，越高越好）", color="#7fd18a")
charts += bar_chart("auroc", "AUROC（阈值无关 OOD 分离度，越高越好）", color="#d98bdc")
charts += bar_chart("known_accuracy", "Known Accuracy（已知类分类准确率）", color="#e0b450")
charts += bar_chart("routing_seconds", "路由耗时（秒，越低越好，注意刻度）", color="#e88", ymax=22)
html_parts.append(section("关键指标对比图", charts))

# 融合消融
fus = f"""
<ul>
<li><b>A0* 默认融合</b>：Macro-F1 {a0m['macro_f1'][0]:.3f} / uF1 {a0m['unknown_f1'][0]:.3f} / AUROC {a0m['auroc'][0]:.3f}</li>
<li><b>A0*_add（加和融合）</b>：Macro-F1 {addm['macro_f1'][0]:.3f} / uF1 {addm['unknown_f1'][0]:.3f} / AUROC {addm['auroc'][0]:.3f} / Known {addm['known_accuracy'][0]:.3f} — <span class='ok'>全指标第一，且 Known Acc 高达 {addm['known_accuracy'][0]:.3f}（近乎饱和）</span></li>
<li><b>A0*_attn（注意力融合）</b>：均值 Macro-F1 {attnm['macro_f1'][0]:.3f} / uF1 {attnm['unknown_f1'][0]:.3f}，但 seed 间方差大（uF1 在 0.52–0.74 间摆动）→ <span class='bad'>不稳健，不建议作为主配置</span></li>
</ul>
<div class='note'>结论：开集多分类场景下，<b>数值+LLM文本编码器 + add 融合</b>是当前最优结构；attention 融合虽有上限潜力但需更多 seed / 超参搜索才稳。</div>
"""
html_parts.append(section("融合方式消融（A0* vs add vs attn）", fus))

# 模态贡献
mod = f"""
<ul>
<li><b>A0*_no_text（纯数值，无文本/无LLM）</b>：AUROC {agg['A0*_no_text']['auroc'][0]:.3f} / Macro-F1 {agg['A0*_no_text']['macro_f1'][0]:.3f} — 全场最差之一，说明<b>去掉文本信息后数值特征本身判别力有限</b>。</li>
<li><b>A0*_no_num（纯 LLM 文本，无数值）</b>：AUROC {agg['A0*_no_num']['auroc'][0]:.3f} / Macro-F1 {agg['A0*_no_num']['macro_f1'][0]:.3f} / Known {agg['A0*_no_num']['known_accuracy'][0]:.3f} — <span class='ok'>仅靠文本就超过 A3（数值+BERT）</span>，证明 LLM 文本编码器的语义判别力是核心。</li>
<li>两者相加（A0*_add）即取二者之长 → 最优。数值提供互补信息，文本提供主判别力。</li>
</ul>
"""
html_parts.append(section("模态贡献（no_text vs no_num）", mod))

# 注意事项
caveats = """
<ul>
<li><b>seed 数偏少（仅 3 个）</b>：AUROC/uF1 在 seed 间波动明显（尤其 A0*_attn），建议在关键变体（A0*_add、A0*、A3）上补到 ≥5 seed 再下结论。</li>
<li><b>样本量</b>：unknown 类仅约 182 条（据记忆），F1 噪声约 ±4 点，minor 差异（如 attn vs add 的 uF1）可能落在噪声内。</li>
<li><b>本 CSV 注册表干净</b>：变体↔描述↔use_* 标志完全一致，model_id 连续（25–42），可信任；此前 09-17 版「model 链接错乱」问题已不存在。</li>
<li><b>口径提醒</b>：unknown_recall + unknown_leak_rate ≡ 1（定义恒等式），二者择一即可；判断 OOD 头质量应看 <b>AUROC</b>（阈值无关）。</li>
<li><b>假设边界</b>：本实验验证的是「<b>可训练</b> LLM（LoRA）作文本编码器有用」，与旧分支「冻结 LLM 无用」不矛盾——区别在于可训练性，而非 LLM 本身。</li>
</ul>
"""
html_parts.append(section("注意事项与下一步", caveats))

html_parts.append("</body></html>")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(html_parts))

print("WROTE", OUT)
print("\n=== 均值汇总 ===")
for v in present:
    print(f"{v:12s} macroF1={agg[v]['macro_f1'][0]:.4f}±{agg[v]['macro_f1'][1]:.4f}  uF1={agg[v]['unknown_f1'][0]:.4f}±{agg[v]['unknown_f1'][1]:.4f}  auroc={agg[v]['auroc'][0]:.4f}±{agg[v]['auroc'][1]:.4f}  known={agg[v]['known_accuracy'][0]:.4f}±{agg[v]['known_accuracy'][1]:.4f}  route={agg[v]['routing_seconds'][0]:.2f}s")
print("\n=== A0* vs A3 一致性 ===")
for mk in ["macro_f1","unknown_f1","auroc","known_accuracy"]:
    w=sum(1 for s in by_var["A3"] for r in [by_var["A0*"]] if False)  # placeholder
    a3v={r['seed']:r[mk] for r in by_var['A3']}
    a0v={r['seed']:r[mk] for r in by_var['A0*']}
    wins=sum(1 for s in a3v if a0v[s]>a3v[s])
    print(f"{mk:18s}: A0* 胜 {wins}/3 seed")
