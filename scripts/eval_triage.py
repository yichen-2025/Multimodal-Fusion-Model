"""triage 四臂评测：gen（生成）/ pack（盲标打包）/ agree（κ 与主指标）。

口径见 docs/triage_protocol.md。只读特征与路由报告，不加载 backbone、不改 OOD 判定；
只有 A2/A3 两臂加载 LLM（models/qwen2.5-1.5b）。

用法：
  python scripts/eval_triage.py --mode gen  --arms A0_template,A1_retrieval,A2_llm_only,A3_llm_rag \
      --labels data/annotations/triage_202611.csv \
      --routing_report ood_reports/ood_routing_20261105_205104.json \
      --out results/2026-11-06/triage/gen_all4.jsonl
  python scripts/eval_triage.py --mode pack --in results/2026-11-06/triage/gen_kappa.jsonl \
      --arms A0_template,A3_llm_rag --outdir results/2026-11-06/triage
  python scripts/eval_triage.py --mode agree --key results/2026-11-06/triage/pack_key.json \
      --scores_l1 results/2026-11-06/triage/blind_scores_L1.csv \
      --scores_l2 results/2026-11-06/triage/blind_scores_L2.csv \
      --out results/2026-11-06/triage/summary.json
  python scripts/eval_triage.py --mode arms --key results/2026-11-06/triage/pack_key_all.json \
      --scores_l1 results/2026-11-06/triage/blind_scores_all_L1.csv \
      --scores_l2 results/2026-11-06/triage/blind_scores_L2.csv \
      --jsonl results/2026-11-06/triage/gen_all4.jsonl \
      --out results/2026-11-10/triage_delivery/summary_arms.json

arms 模式只做**臂间单轮比较**（均值 ± sd、配对差、Wilcoxon p、S3 覆盖/abstain/延迟），
不输出 κ：A1/A2 只有单轮打分，凑 L2 会伪造重测；信度仍由 `--mode agree` 负责。
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
    STEP1_SYSTEM_PROMPT,
    build_step1_prompt,
    build_user_prompt,
    generate_response,
    load_split,
    parse_sufficiency_response,
    recover_global_indices,
)

ARMS_ALL = ["A0_template", "A1_retrieval", "A2_llm_only", "A3_llm_rag"]
LLM_ARMS = ["A2_llm_only", "A3_llm_rag"]

# A3 的第二步沿用 build_user_prompt(assessment=)，只把输出 schema 换成五槽。
TRIAGE_STEP2_SYSTEM_PROMPT = """你是网络安全分析专家，请把一条被判为未知类的流量整理成可处置的 triage 记录。
严格输出 JSON，字段如下：
{"S1": "行为概括（<=2 句，只写可观测行为）",
 "S2": ["候选假设1（置信 high/med/low + 与最接近已知类的差异点）", "候选假设2", "候选假设3"],
 "S3": ["Txxxx"] 或 [],
 "S4": "下一步动作（可执行，含对象与时间窗）",
 "S5": "证据边界（依赖什么、不覆盖什么）",
 "abstain": true 或 false}
规则：S3 证据不足时给空数组并把 abstain 置 true，禁止编造候选之外的编号；不许复述特征名当行为概括；
S4 只写下一步动作（含对象与时间窗），S5 只写证据边界（依赖哪些证据、不覆盖什么），禁止把动作写进 S5。"""


def _parse_json(resp):
    cleaned = (resp or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned), ""
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1]), ""
            except json.JSONDecodeError as e:
                return None, str(e)[:200]
        return None, "no json found"


def _row(sample_id, arm, s1="", s2=None, s3=None, s4="", s5="",
         abstain=False, citations=None, latency=0.0, parse_ok=True, note=""):
    return {
        "sample_id": sample_id, "arm": arm,
        "S1": s1, "S2": s2 or [], "S3": s3 or [], "S4": s4, "S5": s5,
        "abstain": bool(abstain), "citations": citations or [],
        "latency_s": round(float(latency), 2), "parse_ok": bool(parse_ok), "note": note,
    }


def arm_template(sample_id, score, head_id):
    return _row(
        sample_id, "A0_template",
        s1="该流量被 OOD 头 %s 判定为未知类（ood_score=%s），统计特征整体偏离已知类分布。"
           % (head_id, "%.3f" % score if score is not None else "NA"),
        s2=["可能属于训练集未覆盖的行为类型（low：仅凭分数，无行为差异点）"],
        s3=[], s4="提取该来源近 1 小时全量流量，复核端口/协议分布后人工判断是否为新行为。",
        s5="只依赖 OOD 分数与统计特征，无外部情报，无法定位具体技术。",
        abstain=True, note="模板臂：不含检索与 LLM")


def arm_retrieval(sample_id, ev):
    """A1：只用检索证据按规则拼装，不调 LLM。BM25 恒返回 k 条，故实际不会 abstain。"""
    t0 = time.time()
    tokens = ev.get("behavior_tokens") or []
    top_k = ev.get("top_k") or []
    top = top_k[0] if top_k else None
    ids = [top["id"]] if top else []
    s2 = ["%s %s（BM25 %.2f，%s）" % (e["id"], e["name"], e["score"],
                                     "high" if i == 0 else "low")
          for i, e in enumerate(top_k[:3])]
    s4 = ("按 %s 的检测线索（%s）排查同类流量。" % (top["id"], (top.get("detection") or "")[:40])
          if top else "该样本未检索到相近技术，转人工分析并补采同源流量。")
    return _row(
        sample_id, "A1_retrieval",
        s1="行为特征：%s。" % ("、".join(tokens[:6]) or "无可抽取的行为词元"),
        s2=s2 or ["未检索到候选技术（low）"], s3=ids, s4=s4,
        s5="仅 BM25 检索 top-k，检索相似不等于归因；未验证样本真实性。",
        abstain=not bool(ids), citations=ids, latency=time.time() - t0,
        note="检索臂：规则拼装，无 LLM；BM25 恒返回 k 条" if top else "检索臂：无命中")


_ACTION_WORDS = ("查找", "检查", "监控", "抓取", "隔离", "复核", "排查", "补采", "提取", "部署")


def _s5_leak(s5):
    """S5 该写证据边界，若出现动作词则记为越槽（只统计，不改写内容）。"""
    return int(any(w in (s5 or "") for w in _ACTION_WORDS))


def _two_step(llm, tok, device, text, tokens, evidence, arm, sample_id, max_new_tokens,
              step1_max_new_tokens, extra_note=""):
    t0 = time.time()
    s1_prompt = build_step1_prompt(text + extra_note, tokens, evidence)
    r1 = generate_response(llm, tok, STEP1_SYSTEM_PROMPT, s1_prompt, device, step1_max_new_tokens)
    assess = parse_sufficiency_response(r1)
    u2 = build_user_prompt(text + extra_note, tokens, evidence, assessment=assess)
    r2 = generate_response(llm, tok, TRIAGE_STEP2_SYSTEM_PROMPT, u2, device, max_new_tokens)
    parsed, err = _parse_json(r2)
    if not parsed:
        return _row(sample_id, arm, s1="", s4="", s5="", abstain=True, latency=time.time() - t0,
                    parse_ok=False, note="parse_error: %s | raw=%s" % (err, (r2 or "")[:160]))
    s3 = parsed.get("S3") or []
    if isinstance(s3, str):
        s3 = [s3] if s3.strip() else []
    s2 = parsed.get("S2") or []
    if isinstance(s2, str):
        s2 = [s2]
    return _row(sample_id, arm,
                s1=parsed.get("S1", ""), s2=s2, s3=s3, s4=parsed.get("S4", ""),
                s5=parsed.get("S5", ""),
                abstain=bool(parsed.get("abstain", not bool(s3))) or not assess.get("sufficient", True),
                citations=s3, latency=time.time() - t0,
                note="step1_sufficient=%s；s5_action_leak=%d"
                     % (assess.get("sufficient"), _s5_leak(parsed.get("S5", ""))))


def mode_gen(args):
    samples = []
    with open(os.path.join(PROJECT_ROOT, args.labels), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            sid = r["sample_id"]
            if any(s["sample_id"] == sid for s in samples):
                continue
            g = int(sid.rsplit("_", 1)[1])
            samples.append({"sample_id": sid, "gidx": g})
    if args.n:
        samples = samples[:args.n]

    rep = json.load(open(os.path.join(PROJECT_ROOT, args.routing_report), encoding="utf-8"))
    gidx, _ = recover_global_indices(rep)
    score_of, ev_of = {}, {}
    for g, e in zip(gidx, rep["routing_log"]):
        if g is None:
            continue
        score_of[g] = e.get("ood_score")
        if e.get("cti_rag_explanation"):
            ev_of[g] = e["cti_rag_explanation"]  # 报告已预算的检索证据，省一次检索
    head_id = rep.get("ood_id")
    labels, origins, texts, _ = load_split(int(rep["dataset_id"]), int(rep["split_id"]))
    print("样本 %d 条 | 报告 %s | head=%s | 报告内证据 %d 条"
          % (len(samples), os.path.basename(args.routing_report), head_id, len(ev_of)))

    arms = [a for a in args.arms.split(",") if a]
    llm = tok = None
    if set(arms) & set(LLM_ARMS):
        from src.explain.generate_explanation import load_llm_for_generation
        # 与 scripts/generate_unknown_explanations.py:287 一致：None 时落到本地基座目录
        llm_path = args.llm_model_path or os.path.join(PROJECT_ROOT, "models", "qwen2.5-1.5b")
        print("加载 LLM:", os.path.basename(llm_path))
        llm, tok = load_llm_for_generation(llm_path, device=args.device)

    out_path = os.path.join(PROJECT_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_ok = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            g, sid = s["gidx"], s["sample_id"]
            text = texts[g]
            ev = ev_of.get(g) or build_cti_explanation(text, k=args.k)
            for arm in arms:
                if arm == "A0_template":
                    row = arm_template(sid, score_of.get(g), head_id)
                elif arm == "A1_retrieval":
                    row = arm_retrieval(sid, ev)
                elif arm in LLM_ARMS:
                    if arm == "A3_llm_rag":
                        evidence, note = ev.get("top_k") or [], ""
                    else:
                        evidence, note = [], "\n注意：本臂不使用任何检索证据。"
                    row = _two_step(llm, tok, args.device, text,
                                    ev.get("behavior_tokens") or [], evidence, arm, sid,
                                    args.max_new_tokens, args.step1_max_new_tokens, note)
                else:
                    raise SystemExit("未知臂: %s" % arm)
                if row["parse_ok"]:
                    n_ok += 1
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print("  %-16s %-13s ok=%s abstain=%s %.1fs"
                      % (sid, arm, row["parse_ok"], row["abstain"], row["latency_s"]))
    print("写入 %s | 成功 %d 行" % (out_path, n_ok))


def _slot_text(row):
    return " || ".join([
        "S1=%s" % row["S1"],
        "S2=%s" % " / ".join(row["S2"]),
        "S3=%s" % (",".join(row["S3"]) if row["S3"] else "abstain"),
        "S4=%s" % row["S4"],
        "S5=%s" % row["S5"],
    ])


def mode_pack(args):
    rows = [json.loads(l) for l in open(os.path.join(PROJECT_ROOT, args.inp), encoding="utf-8")]
    arms = [a for a in args.arms.split(",") if a]
    rows = [r for r in rows if r["arm"] in arms]
    by_sid = {}
    for r in rows:
        by_sid.setdefault(r["sample_id"], []).append(r)
    outdir = os.path.join(PROJECT_ROOT, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    key, pack = {}, []
    for i, (sid, rs) in enumerate(sorted(by_sid.items())):
        for j, r in enumerate(rs):
            pid = "p%03d" % (len(pack) + 1)  # 逐行递增，避免各样本臂数不等时 pid 撞车
            key[pid] = {"sample_id": sid, "arm": r["arm"]}
            pack.append({"pack_id": pid, "sample_id": sid,
                         "S1": r["S1"], "S2": " / ".join(r["S2"]),
                         "S3": ",".join(r["S3"]) if r["S3"] else "",
                         "S4": r["S4"], "S5": r["S5"],
                         "actionability": "", "sufficiency": "", "notes": ""})
    pack.sort(key=lambda x: rng.rand())  # 打乱顺序，避免相邻样本同臂
    blind = os.path.join(outdir, "blind_pack.csv")
    with open(blind, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pack[0].keys()))
        w.writeheader()
        w.writerows(pack)
    json.dump(key, open(os.path.join(outdir, "pack_key.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    for name in ("L1", "L2"):
        p = os.path.join(outdir, "blind_scores_%s.csv" % name)
        with open(p, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["pack_id", "actionability", "sufficiency", "notes"])
            w.writeheader()
            for r in pack:
                w.writerow({"pack_id": r["pack_id"], "actionability": "", "sufficiency": "", "notes": ""})
    print("盲标表 %s（%d 行，无臂名）\n打分表 blind_scores_L1/L2.csv（各 %d 行）\n对照表 pack_key.json"
          % (blind, len(pack), len(pack)))
    print("注意：pack_id 已与臂名解耦，请勿把 pack_key.json 交给标注者。")


def _weighted_kappa(a, b, k=5):
    a, b = np.asarray(a, int), np.asarray(b, int)
    if len(a) == 0:
        return float("nan")
    po = float((a == b).mean())
    pa = np.array([(a == i).mean() for i in range(1, k + 1)])
    pb = np.array([(b == i).mean() for i in range(1, k + 1)])
    pe = float(pa @ pb)
    pe = 0.0 if pe >= 0.999 else pe
    return (po - pe) / (1 - pe) if pe < 0.999 else float("nan")


def _read_scores(path):
    out = {}
    with open(os.path.join(PROJECT_ROOT, path), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if not (r["actionability"] or "").strip():
                continue
            out[r["pack_id"]] = (int(r["actionability"]), int(r["sufficiency"]))
    return out


def _read_scores_opt(path):
    """可选打分表：文件不存在时返回空字典（A1/A2 未填分是合法状态）。"""
    if not path or not os.path.exists(os.path.join(PROJECT_ROOT, path)):
        return {}
    return _read_scores(path)


def _contrast(a, b, ids, idx):
    """配对差（a-b）：idx=0 为 actionability，1 为 sufficiency。"""
    d = np.array([a[i][idx] - b[i][idx] for i in ids])
    out = {"n": len(ids), "mean_diff": round(float(d.mean()), 3) if len(d) else float("nan")}
    if len(d) > 1:
        out["sd"] = round(float(d.std(ddof=1)), 3)
    p = float("nan")
    if len(d) > 5 and np.any(d != 0):
        from scipy.stats import wilcoxon
        p = float(wilcoxon(d).pvalue)
    out["p"] = round(p, 4)
    return out


def _jsonl_by_arm(path):
    out = {}
    full = os.path.join(PROJECT_ROOT, path)
    if not os.path.exists(full):
        return out
    with open(full, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out.setdefault(r["arm"], []).append(r)
    return out


def mode_arms(args):
    """单轮臂间比较：四臂均值 ± sd、A1 vs 其余臂的配对差与 Wilcoxon p、
    S3 覆盖/abstain/延迟。**不输出 κ**：A1/A2 只有单轮打分，硬凑 L2 会伪造重测。"""
    key = json.load(open(os.path.join(PROJECT_ROOT, args.key), encoding="utf-8"))
    rounds, used = [], []
    for p in (args.scores_l1, args.scores_l2):
        s = _read_scores_opt(p)
        if s:
            rounds.append(s)
            used.append(os.path.basename(p))
    if not rounds:
        raise SystemExit("打分表都没填分（%s），无事可做" % args.scores_l1)

    per_arm, n_rounds, missing = {}, {}, {}
    for pid, meta in key.items():
        arm, sid = meta["arm"], meta["sample_id"]
        hit = [r[pid] for r in rounds if pid in r]
        if not hit:
            missing.setdefault(arm, []).append(pid)
            continue
        n_rounds[arm] = max(n_rounds.get(arm, 0), len(hit))
        per_arm.setdefault(arm, {})[sid] = (float(np.mean([h[0] for h in hit])),
                                            float(np.mean([h[1] for h in hit])))

    summary = {"mode": "arms", "kappa": None,
               "kappa_note": "单轮臂间比较不估计信度；κ 见 --mode agree",
               "n_pack_keys": len(key), "rounds_used": used,
               "missing": {k: sorted(v) for k, v in sorted(missing.items())},
               "arms": {}}
    for arm, d in sorted(per_arm.items()):
        a = np.array([v[0] for v in d.values()])
        s = np.array([v[1] for v in d.values()])
        summary["arms"][arm] = {
            "n": len(d), "n_rounds": n_rounds.get(arm, 1),
            "actionability_mean": round(float(a.mean()), 3),
            "actionability_sd": round(float(a.std(ddof=1)) if len(a) > 1 else 0.0, 3),
            "sufficiency_mean": round(float(s.mean()), 3),
            "sufficiency_sd": round(float(s.std(ddof=1)) if len(s) > 1 else 0.0, 3),
        }

    pairs = [("A1_retrieval", "A0_template"), ("A1_retrieval", "A2_llm_only"),
             ("A1_retrieval", "A3_llm_rag"), ("A3_llm_rag", "A0_template")]
    summary["contrasts"] = {}
    for x, y in pairs:
        if x in per_arm and y in per_arm:
            ids = sorted(set(per_arm[x]) & set(per_arm[y]))
            if ids:
                summary["contrasts"]["%s_minus_%s" % (x, y)] = {
                    "actionability": _contrast(per_arm[x], per_arm[y], ids, 0),
                    "sufficiency": _contrast(per_arm[x], per_arm[y], ids, 1)}

    gen = _jsonl_by_arm(args.jsonl)
    summary["evidence"] = {}
    for arm, sid_map in sorted(per_arm.items()):
        rs = [r for r in gen.get(arm, []) if r["sample_id"] in sid_map]
        if not rs:
            continue
        lat = sorted(float(r.get("latency_s", 0.0)) for r in rs)
        summary["evidence"][arm] = {
            "n": len(rs),
            "s3_coverage": round(sum(1 for r in rs if r.get("S3")) / len(rs), 4),
            "abstain_rate": round(sum(1 for r in rs if r.get("abstain")) / len(rs), 4),
            "latency_s_p50": round(float(np.percentile(lat, 50)), 2),
            "latency_s_p95": round(float(np.percentile(lat, 95)), 2),
        }
    summary["caveats"] = [
        "本模式不做信度估计，不输出 κ（A1/A2 为单轮打分）",
        "S3 覆盖率高是 BM25 恒返回 k 条的结构性结果，不等于映射正确",
        "缺失臂写入 missing，未做任何插补",
    ]

    out = os.path.join(PROJECT_ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(summary, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("写入", out)


def mode_agree(args):
    from scipy.stats import wilcoxon
    key = json.load(open(os.path.join(PROJECT_ROOT, args.key), encoding="utf-8"))
    l1, l2 = _read_scores(args.scores_l1), _read_scores(args.scores_l2)
    common = sorted(set(l1) & set(l2))
    if not common:
        raise SystemExit("两份打分表没有共同的 pack_id（是否没填分？）")
    ka = _weighted_kappa([l1[p][0] for p in common], [l2[p][0] for p in common])
    ks = _weighted_kappa([l1[p][1] for p in common], [l2[p][1] for p in common])
    print("共同 pack_id=%d | actionability κ=%.3f | sufficiency κ=%.3f（Cohen's κ，不加权；L1 vs L2）"
          % (len(common), ka, ks))

    per_arm = {}
    for p, (a1, s1) in l1.items():
        arm, sid = key[p]["arm"], key[p]["sample_id"]
        per_arm.setdefault(arm, {})[sid] = ((a1 + l2[p][0]) / 2.0, (s1 + l2[p][1]) / 2.0)
    summary = {"kappa_actionability": ka, "kappa_sufficiency": ks,
               "n_pack": len(common), "kappa_type": args.kappa_type, "arms": {}}
    for arm, d in sorted(per_arm.items()):
        a = np.array([v[0] for v in d.values()])
        s = np.array([v[1] for v in d.values()])
        summary["arms"][arm] = {"n": len(d), "actionability_mean": round(float(a.mean()), 3),
                                "actionability_sd": round(float(a.std(ddof=1)) if len(a) > 1 else 0.0, 3),
                                "sufficiency_mean": round(float(s.mean()), 3)}
    base = None
    if "A3_llm_rag" in per_arm:
        a3 = per_arm["A3_llm_rag"]
        for name, other in (("A0_template", "A0_template"), ("A1_retrieval", "A1_retrieval")):
            if other in per_arm:
                ids = sorted(set(a3) & set(per_arm[other]))
                d = np.array([a3[i][0] - per_arm[other][i][0] for i in ids])
                p = float(wilcoxon(d).pvalue) if len(d) > 5 and np.any(d != 0) else float("nan")
                summary.setdefault("contrasts", {})["A3_minus_" + name] = {
                    "n": len(ids), "mean_diff": round(float(d.mean()), 3), "p": round(p, 4)}
        if "A0_template" in per_arm and "A1_retrieval" in per_arm:
            ids = sorted(set(a3) & set(per_arm["A0_template"]) & set(per_arm["A1_retrieval"]))
            d = np.array([a3[i][0] - max(per_arm["A0_template"][i][0], per_arm["A1_retrieval"][i][0])
                          for i in ids])
            p = float(wilcoxon(d).pvalue) if len(d) > 5 and np.any(d != 0) else float("nan")
            summary.setdefault("contrasts", {})["A3_minus_max_A0_A1"] = {
                "n": len(ids), "mean_diff": round(float(d.mean()), 3), "p": round(p, 4)}
            base = d
    if base is not None and ka >= 0.4 and base.mean() >= 1.0 and summary["contrasts"]["A3_minus_max_A0_A1"]["p"] < 0.05:
        summary["verdict"] = "positive_preliminary"
    elif ka < 0.4:
        summary["verdict"] = "scale_unusable"
    else:
        summary["verdict"] = "negative_or_inconclusive"

    out = os.path.join(PROJECT_ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(summary, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("写入", out)


def main():
    ap = argparse.ArgumentParser(description="triage 四臂评测（gen/pack/agree/arms）")
    ap.add_argument("--mode", required=True, choices=["gen", "pack", "agree", "arms"],
                    help="arms=单轮臂间比较（不输出 κ）")
    ap.add_argument("--arms", default=",".join(ARMS_ALL))
    ap.add_argument("--labels", default="data/annotations/triage_202611.csv")
    ap.add_argument("--routing_report", default="ood_reports/ood_routing_20261105_205104.json")
    ap.add_argument("--out", default="results/2026-11-06/triage/gen_all4.jsonl")
    ap.add_argument("--outdir", default="results/2026-11-06/triage")
    ap.add_argument("--inp", dest="inp", default="results/2026-11-06/triage/gen_kappa.jsonl")
    ap.add_argument("--n", type=int, default=0, help="只跑前 n 条（0=全部）")
    ap.add_argument("--k", type=int, default=3, help="检索 top-k")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--llm_model_path", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=500)
    ap.add_argument("--step1_max_new_tokens", type=int, default=120)
    ap.add_argument("--key", default="results/2026-11-06/triage/pack_key.json")
    ap.add_argument("--scores_l1", default="results/2026-11-06/triage/blind_scores_L1.csv")
    ap.add_argument("--scores_l2", default="results/2026-11-06/triage/blind_scores_L2.csv")
    ap.add_argument("--kappa_type", default="retest", choices=["retest", "inter_annotator"],
                    help="无第二标注者时为 retest（重测信度）")
    ap.add_argument("--jsonl", default="results/2026-11-06/triage/gen_all4.jsonl",
                    help="arms 模式统计 S3 覆盖/abstain/延迟用的生成结果")
    args = ap.parse_args()
    {"gen": mode_gen, "pack": mode_pack, "agree": mode_agree,
     "arms": mode_arms}[args.mode](args)


if __name__ == "__main__":
    main()
