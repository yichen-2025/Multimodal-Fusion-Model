"""检索-only triage 卡片：路由报告 → 五槽卡片（CSV + 样例 md + metrics.json）。

只用 BM25 + 规则拼装，不加载 LLM、不导入 torch。五槽里的检索证据优先取报告里已存的
`cti_rag_explanation`（与 A3 臂同一份证据），另跑一次 `build_cti_explanation` 只为
计时与复算一致性校验，不改写已存证据。

口径见 docs/triage_protocol.md §交付形态。注意：routing_log 的 sample_idx 是 batch 内
下标（0..batch-1，可重复），对外 ID 用 recover_global_indices 反查出的全局下标。

用法：
  python scripts/triage_card.py --routing_report ood_reports/ood_routing_20261105_205104.json \
      --outdir results/2026-11-10/triage_delivery --limit 0
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

from src.explain.retriever import build_cti_explanation  # noqa: E402
from scripts.generate_unknown_explanations import (  # noqa: E402
    load_split,
    recover_global_indices,
)

CARD_FIELDS = ["sample_id", "batch_idx", "true_label", "pred_known_class", "unknown_origin",
               "ood_score", "S1", "S2", "S3", "S4", "S5", "confidence", "citations",
               "abstain", "n_candidates", "retrieve_ms", "assemble_ms"]

CONF_RULE = ("high = top1.score>=s67 且 (top1-top2)>=g67；med = top1.score>=s33 且间隔>=g33；"
             "其余 low。s33/s67、g33/g67 取本报告 516 条 BM25 分数的 33/67 分位，"
             "且分位在**去重取值**上计算（分数离散，避免截断在众数上）")


def _conf_thresholds(scores, gaps):
    us, ug = sorted(set(scores)), sorted(set(gaps))
    return {
        "s33_top1": round(float(np.percentile(us, 33)), 4),
        "s67_top1": round(float(np.percentile(us, 67)), 4),
        "g33_gap": round(float(np.percentile(ug, 33)), 4),
        "g67_gap": round(float(np.percentile(ug, 67)), 4),
        "n_distinct_top1": len(us),
        "n_distinct_gap": len(ug),
        "rule": CONF_RULE,
    }


def _confidence(score, gap, th):
    if score >= th["s67_top1"] and gap >= th["g67_gap"]:
        return "high"
    if score >= th["s33_top1"] and gap >= th["g33_gap"]:
        return "med"
    return "low"


def _first_sentence(text, limit=60):
    for sep in ("。", "；", "\n"):
        if sep in (text or ""):
            text = text.split(sep)[0]
            break
    return (text or "")[:limit].strip(" ；。")


def build_card(sample_id, batch_idx, rec, text, conf_th, k=3):
    """五槽卡片：S1 行为词元 / S2 候选 / S3 编号 / S4 下一步 / S5 证据边界。"""
    t0 = time.perf_counter()
    ev = rec.get("cti_rag_explanation") or {}
    if not ev:
        ev = build_cti_explanation(text, k=k)
    top_k = ev.get("top_k") or []
    tokens = ev.get("behavior_tokens") or []
    top = top_k[0] if top_k else None
    score = float(top["score"]) if top else float("nan")
    gap = (float(top_k[0]["score"]) - float(top_k[1]["score"])) if len(top_k) > 1 else 0.0
    conf = _confidence(score, gap, conf_th) if top else "low"

    s2 = ["%s %s（BM25 %.2f，%s）" % (e["id"], e["name"], e["score"],
                                     conf if i == 0 else "low")
          for i, e in enumerate(top_k[:3])]
    s4 = ("按 %s 的检测线索（%s）排查同类流量；并提取该来源近 1 小时全量流量复核端口/协议分布。"
          % (top["id"], _first_sentence(top.get("detection")))
          if top else "未检索到相近技术：提取该来源近 1 小时全量流量并转人工分析。")
    card = {
        "sample_id": sample_id,
        "batch_idx": batch_idx,
        "true_label": int(rec.get("true_label", -1)),
        "pred_known_class": ("" if rec.get("a3_pred_known_class") is None
                             else int(rec["a3_pred_known_class"])),
        "unknown_origin": "",
        "ood_score": round(float(rec.get("ood_score", float("nan"))), 4),
        "S1": "行为特征：%s。" % ("、".join(tokens[:6]) or "无可抽取的行为词元"),
        "S2": " / ".join(s2) or "未检索到候选技术（low）",
        "S3": ";".join([top["id"]] if top else []),
        "S4": s4,
        "S5": "仅 BM25 检索 top-%d 相似度，检索相似不等于归因；未验证样本真实性，"
              "不覆盖 ATT&CK 映射正确性。" % k,
        "confidence": conf,
        "citations": ";".join([e["id"] for e in top_k[:3]]),
        "abstain": (not bool(top_k)),
        "n_candidates": len(top_k),
    }
    card["assemble_ms"] = round((time.perf_counter() - t0) * 1000, 4)
    return card


def _rate(rows, key):
    if not rows:
        return 0.0
    return round(sum(1 for r in rows if str(r.get(key, "")).strip()) / len(rows), 4)


def _view_metrics(rows, th):
    conf = {c: sum(1 for r in rows if r["confidence"] == c) for c in ("high", "med", "low")}
    return {
        "n": len(rows),
        "slot_nonempty_rate": {s: _rate(rows, s) for s in ("S1", "S2", "S3", "S4", "S5")},
        "s3_coverage": _rate(rows, "S3"),
        "abstain_rate": round(sum(1 for r in rows if r["abstain"]) / len(rows), 4) if rows else 0.0,
        "confidence": conf,
        "true_label_dist": {str(k): sum(1 for r in rows if str(r["true_label"]) == str(k))
                            for k in sorted({r["true_label"] for r in rows})},
    }


def _latency(rows, key, skip_first=True):
    v = sorted(float(r[key]) for r in rows)
    if skip_first and len(v) > 1:
        v = v[1:]
    if not v:
        return {}
    return {"n": len(v), "p50": round(float(np.percentile(v, 50)), 4),
            "p95": round(float(np.percentile(v, 95)), 4), "max": round(v[-1], 4)}


def _write_csv(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CARD_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CARD_FIELDS})


def _sample_md(rows, n_true, n_false):
    pick = ([r for r in rows if r["true_label"] == 10][:n_true]
            + [r for r in rows if r["true_label"] != 10][:n_false])
    n_t = sum(1 for r in pick if r["true_label"] == 10)
    lines = ["# 检索-only triage 卡片样例", "",
             "> 取自部署视角：%d 张为真未知（`true_label=10`），%d 张为已知类误判（OOD 误报）。"
             % (n_t, len(pick) - n_t),
             "> 边界：S3 有编号只说明 BM25 返回了候选，**不等于映射正确**。", ""]
    for r in pick:
        pred = r["pred_known_class"]
        pred_txt = "unknown" if pred == "" else str(pred)
        lines += ["---", "",
                  "## %s（true_label=%s，pred_known=%s，ood_score=%s）"
                  % (r["sample_id"], r["true_label"], pred_txt, r["ood_score"]), "",
                  "| 槽 | 内容 |", "|---|---|",
                  "| S1 行为 | %s |" % r["S1"],
                  "| S2 候选 | %s |" % r["S2"],
                  "| S3 编号 | %s |" % (r["S3"] or "（空：abstain）"),
                  "| S4 下一步 | %s |" % r["S4"],
                  "| S5 边界 | %s |" % r["S5"],
                  "| 置信 | %s（候选 %d 条，abstain=%s） |"
                  % (r["confidence"], r["n_candidates"], r["abstain"]), ""]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="检索-only triage 卡片（BM25 + 规则，无 LLM）")
    ap.add_argument("--routing_report",
                    default="ood_reports/ood_routing_20261105_205104.json")
    ap.add_argument("--outdir", default="results/2026-11-10/triage_delivery")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 n 条 flagged（0=全部）")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--sample_n_true", type=int, default=3, help="样例 md 里真未知张数")
    ap.add_argument("--sample_n_false", type=int, default=2, help="样例 md 里误报张数")
    args = ap.parse_args()

    rep_path = (args.routing_report if os.path.isabs(args.routing_report)
                else os.path.join(PROJECT_ROOT, args.routing_report))
    report = json.load(open(rep_path, encoding="utf-8"))
    gidx, unmatched = recover_global_indices(report)
    labels, origins, texts, _ = load_split(int(report["dataset_id"]), int(report["split_id"]))
    log = report["routing_log"]
    print("routing_log=%d 全局下标匹配=%d 未匹配=%d | head=%s | 报告=%s"
          % (len(log), sum(1 for g in gidx if g is not None), unmatched,
             report.get("ood_id"), os.path.basename(rep_path)))

    # 置信阈值固定取全报告（516 条）分位数，不受 --limit 影响
    scores, gaps = [], []
    for e in log:
        top_k = (e.get("cti_rag_explanation") or {}).get("top_k") or []
        if top_k:
            scores.append(float(top_k[0]["score"]))
            gaps.append(float(top_k[0]["score"]) - float(top_k[1]["score"]) if len(top_k) > 1 else 0.0)
    conf_th = _conf_thresholds(scores, gaps)
    print("置信阈值:", json.dumps(conf_th, ensure_ascii=False))

    pair = [(g, e) for g, e in zip(gidx, log) if g is not None]
    if args.limit:
        pair = pair[:args.limit]

    rows, rerun_agree = [], 0
    for g, e in pair:
        text = texts[g]
        # 计时用：真跑一次检索（结果只用于一致性校验，不改写卡片内容）
        t0 = time.perf_counter()
        fresh = build_cti_explanation(text, k=args.k)
        r_ms = (time.perf_counter() - t0) * 1000
        stored = (e.get("cti_rag_explanation") or {}).get("top_k") or []
        if stored and fresh.get("top_id") == stored[0]["id"]:
            rerun_agree += 1
        card = build_card("ds%s_s%s_%d" % (report["dataset_id"], report["split_id"], g),
                          e.get("sample_idx"), e, text, conf_th, k=args.k)
        card["unknown_origin"] = int(origins[g]) if origins is not None else ""
        card["retrieve_ms"] = round(r_ms, 4)
        rows.append(card)

    outdir = os.path.join(PROJECT_ROOT, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    true_rows = [r for r in rows if r["true_label"] == 10]
    _write_csv(os.path.join(outdir, "cards_all%d.csv" % len(rows)), rows)
    _write_csv(os.path.join(outdir, "cards_true%d.csv" % len(true_rows)), true_rows)
    with open(os.path.join(outdir, "sample_cards.md"), "w", encoding="utf-8") as f:
        f.write(_sample_md(rows, args.sample_n_true, args.sample_n_false))

    metrics = {
        "routing_report": os.path.relpath(rep_path, PROJECT_ROOT).replace("\\", "/"),
        "ood_id": report.get("ood_id"),
        "n_routing_log": len(log),
        "n_global_matched": sum(1 for g in gidx if g is not None),
        "n_unmatched": unmatched,
        "n_cards_all": len(rows),
        "n_cards_true_unknown": len(true_rows),
        "rerun_top1_agree": rerun_agree,
        "rerun_top1_agree_rate": round(rerun_agree / len(rows), 4) if rows else 0.0,
        "confidence_thresholds": conf_th,
        "views": {"all": _view_metrics(rows, conf_th),
                  "true_unknown": _view_metrics(true_rows, conf_th)},
        "unknown_origin_dist": {str(k): sum(1 for r in rows if str(r["unknown_origin"]) == str(k))
                                for k in sorted({r["unknown_origin"] for r in rows},
                                                key=lambda x: str(x))},
        "latency_ms": {
            "retrieve_first_call": round(float(rows[0]["retrieve_ms"]), 4) if rows else None,
            "retrieve": _latency(rows, "retrieve_ms"),
            "assemble": _latency(rows, "assemble_ms"),
        },
        "caveats": [
            "S3 覆盖 100% 是结构性的：BM25 恒返回 k 条，只说明检索给了编号，不等于编号正确",
            "真未知 249 条无 ATT&CK 真值，不得据此声称映射正确率",
            "本视图含 %d 条已知类误判（OOD 误报），不合并成单一未知召回" % (len(rows) - len(true_rows)),
            "retrieve_ms 为本机 CPU 单条 BM25 查询耗时，不含模型加载",
        ],
    }
    with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)

    print("卡片 all=%d true_unknown=%d | top1 复算一致 %d/%d | retrieve_ms p50=%.3f p95=%.3f"
          % (len(rows), len(true_rows), rerun_agree, len(rows),
             metrics["latency_ms"]["retrieve"].get("p50", -1),
             metrics["latency_ms"]["retrieve"].get("p95", -1)))
    print("输出:", outdir)


if __name__ == "__main__":
    main()
