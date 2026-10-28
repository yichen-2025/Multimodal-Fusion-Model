"""C1: unknown 路径 LLM 语义解释生成（只读路由报告，不进入判类路径）。

用法：
    python run.py python scripts/generate_unknown_explanations.py \
        --routing_report ood_reports/ood_routing_20261027_084942.json \
        --n_samples 100 --seed 42

口径（与 known 路径 generate_explanations.py 对齐，差异处显式标注）：
1. 全局下标反查：routing_log 里的 sample_idx 是 batch 内下标（非全局），
   因此用 ood_score 与报告 all_ood_scores 做 float32 精确匹配反查全局下标，
   再用 true_label 校验；不修改 run_ood_routing.py，保证路由指标逐位不变。
2. 分层抽样：按 unknown_origin 分层（0=被误判为未知的已知类、1=真实未知、
   2=分布内模拟未知），配额默认 25%/50%/25%，不足时向其它组回填。
3. 数字口径：解释中的数字只允许来自样本自身 text_description；检索证据
   （top-k detection 文本含端口/速率等数字）只作定性参考，禁止引用其数字。
   引用率/未验证率复用 eval_explanations 的 v3 解析函数，指标定义与 known 路径一致。
4. 命中率：只对存在真值技术的分组统计。genuine（origin=1）按红线 #6 不计算命中；
   simulated（origin=2）按其合并类的技术集合命中；known-FP（origin=0）按其原始类命中。
5. 本脚本只读报告与划分文件，不重跑分类、不写回路由报告，登记为 explanation_eval。
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.explain.attack_mapping import get_label_name, get_true_mitre_ids
from src.explain.generate_explanation import (
    _normalize_mitre_id,
    load_llm_for_generation,
    parse_llm_response,
)
from src.explain.retriever import build_cti_explanation, extract_behavior_tokens
from scripts.eval_explanations import number_in, parse_number_tokens, parse_numbers

ORIGIN_NAMES = {0: "known_false_positive", 1: "genuine_ood", 2: "simulated_unknown"}

SYSTEM_PROMPT = """你是一名网络安全分析专家，负责为「被判为未知类」的网络流量生成解释。
该样本已被分布外（OOD）检测头判定为未知类，不属于任何已知攻击类。

要求：
1. 解释要具体引用「流量文本特征」中的字段（端口、包数、字节数、标志位、间隔等），
   并说明这些特征为何支持「未知/可疑」的判断；
2. 解释中出现的任何数字都必须能在「流量文本特征」原文中找到并照抄原值，
   禁止改写量级、换算单位，也禁止自行计算派生数值（如求平均、求和、做差）；
3. 禁止引用「检索证据」中出现的任何数字（端口号、速率、阈值等），只能定性引用其技术名称与行为描述；
4. 该样本已被判定为未知类，但仍需从检索证据给出的候选编号中选出最相近的一项写入
   mitre_technique_id，作为参考（不改变未知判定）；不得编造候选之外的编号，候选为空时填 null；
5. 输出严格为 JSON 格式，包含两个字段：explanation（中文解释）和 mitre_technique_id（ATT&CK 编号或 null）"""


def build_user_prompt(text_description, behavior_tokens, evidence):
    """构建 unknown 路径的 user prompt（证据只作定性参考）。"""
    lines = []
    for e in evidence:
        lines.append(f"  - {e['id']}: {e['name']}（行为：{e['detection']}）")
    ev_str = "\n".join(lines) if lines else "  - （无检索证据）"
    tok_str = "、".join(behavior_tokens) or "无"
    return f"""流量文本特征：
{text_description}

判定结果：未知类（不属于任何已知类）
规则抽取的行为特征词元：{tok_str}

检索到的相近 ATT&CK 技术（证据，仅供定性参考，禁止引用其中的数字）：
{ev_str}

请生成 JSON 格式的分析结果："""


def load_split(dataset_id, split_id, openset=True):
    """返回 (npz labels, unknown_origin, text_description 列表, CSV 原始 Label 列表)。"""
    split_name = f"split_openset_{split_id}" if openset else f"split_{split_id}"
    split_dir = os.path.join(PROJECT_ROOT, "split_data", f"dataset_{dataset_id}", split_name)
    npz = np.load(os.path.join(split_dir, "test.npz"), allow_pickle=True)
    labels = np.asarray(npz["labels"])
    origins = np.asarray(npz["unknown_origin"]) if "unknown_origin" in npz else None
    df = pd.read_csv(os.path.join(split_dir, "test_data.csv"))
    texts = df["text_description"].fillna("").tolist() if "text_description" in df.columns else [""] * len(labels)
    raw_labels = df["Label"].to_numpy() if "Label" in df.columns else labels
    return labels, origins, texts, raw_labels


def recover_global_indices(report):
    """用 ood_score 反查 routing_log 的全局下标，并返回未匹配条目数。"""
    scores = np.asarray(report.get("all_ood_scores", []), dtype=np.float32)
    all_labels = np.asarray(report.get("all_labels", []))
    used = set()
    out = []
    unmatched = 0
    for e in report.get("routing_log", []):
        s = np.float32(e["ood_score"])
        cand = np.where(scores == s)[0] if len(scores) else []
        pick = None
        for g in cand:
            g = int(g)
            if g in used:
                continue
            if len(all_labels) and int(all_labels[g]) != int(e["true_label"]):
                continue
            pick = g
            break
        if pick is None:
            unmatched += 1
            out.append(None)
        else:
            used.add(pick)
            out.append(pick)
    return out, unmatched


def stratified_quota(avail, n_samples, weights):
    """按可用量与权重分配配额，不足的份额回填到有余量的组。"""
    quota = {}
    for k, w in weights.items():
        quota[k] = min(avail.get(k, 0), int(round(n_samples * w)))
    left = n_samples - sum(quota.values())
    order = sorted(weights, key=lambda k: -weights[k])
    while left > 0:
        progressed = False
        for k in order:
            if left <= 0:
                break
            if quota[k] < avail.get(k, 0):
                quota[k] += 1
                left -= 1
                progressed = True
        if not progressed:
            break
    return quota


def main():
    ap = argparse.ArgumentParser(description="C1: unknown 路径 LLM 解释生成")
    ap.add_argument("--routing_report", required=True, help="ood_reports/ood_routing_<ts>.json")
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset_id", type=int, default=None, help="默认取报告里的 dataset_id")
    ap.add_argument("--split_id", type=int, default=None, help="默认取报告里的 split_id")
    ap.add_argument("--llm_model_path", type=str, default=None)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--dry_run", action="store_true", help="只抽样并打印构成，不加载模型")
    args = ap.parse_args()

    rep_path = os.path.join(PROJECT_ROOT, args.routing_report) \
        if not os.path.isabs(args.routing_report) else args.routing_report
    report = json.load(open(rep_path, "r", encoding="utf-8"))

    dataset_id = args.dataset_id if args.dataset_id is not None else int(report["dataset_id"])
    split_id = args.split_id if args.split_id is not None else int(report["split_id"])
    num_known_classes = 10
    labels, origins, texts, raw_labels = load_split(dataset_id, split_id)
    num_known_classes = int(labels.max())  # 开集划分里 unknown_label = K
    if origins is None:
        raise SystemExit("划分缺少 unknown_origin，无法按真实/模拟未知分层")

    gidx, unmatched = recover_global_indices(report)
    pool = [g for g in gidx if g is not None]
    print(f"routing_log={len(gidx)} 全局下标匹配={len(pool)} 未匹配={unmatched}")

    # 一致性自检：抽 20 条用文本重算行为词元，必须与报告里存的一致
    log_entries = report["routing_log"]
    checked = 0
    for g, e in zip(gidx, log_entries):
        if g is None:
            continue
        if extract_behavior_tokens(texts[g]) == e["cti_rag_explanation"]["behavior_tokens"]:
            checked += 1
        if checked >= 20:
            break
    if checked < 20:
        raise SystemExit(f"行为词元对齐失败（{checked}/20），下标反查不可信")

    avail = Counter(int(origins[g]) for g in pool)
    quota = stratified_quota(avail, min(args.n_samples, len(pool)), {2: 0.25, 1: 0.50, 0: 0.25})
    print("routed 组成:", {ORIGIN_NAMES[k]: v for k, v in sorted(avail.items())})
    print("抽样配额:", {ORIGIN_NAMES[k]: v for k, v in sorted(quota.items())})

    rng = np.random.RandomState(args.seed)
    picked = []
    for k in sorted(quota):
        cand = [g for g in pool if int(origins[g]) == k]
        take = min(quota[k], len(cand))
        if take:
            picked.extend(rng.choice(np.array(cand), size=take, replace=False).tolist())
    picked = sorted(int(x) for x in picked)

    if args.dry_run:
        print("dry_run 抽样下标:", picked[:20], "... 共", len(picked))
        return

    entry_of = {g: e for g, e in zip(gidx, log_entries) if g is not None}
    llm_model_path = args.llm_model_path or os.path.join(PROJECT_ROOT, "models", "qwen2.5-1.5b")
    print(f"加载基座 LLM: {os.path.basename(llm_model_path)}")
    llm_model, tokenizer = load_llm_for_generation(llm_model_path, device=args.device)

    results = []
    ref_by_group = defaultdict(lambda: [0, 0, 0])  # [有引用, 有未验证, 总数]
    for g in tqdm(picked, desc="LLM 生成（unknown）"):
        ent = entry_of[g]
        text = texts[g]
        evidence = ent["cti_rag_explanation"].get("top_k", [])
        behavior_tokens = extract_behavior_tokens(text)
        user_prompt = build_user_prompt(text, behavior_tokens, evidence)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            chat = (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                    f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n")
        inputs = tokenizer(chat, return_tensors="pt", truncation=True, max_length=1536).to(args.device)
        with torch.no_grad():
            out = llm_model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        parsed = parse_llm_response(resp, num_known_classes)
        explanation = parsed["explanation"]

        origin = int(origins[g])
        # 真值技术：simulated 用合并类（label=K）；known-FP 用 CSV 原始类；genuine 不计算
        if origin == 2:
            truth = sorted(get_true_mitre_ids(int(labels[g])))
        elif origin == 0:
            truth = sorted(get_true_mitre_ids(int(raw_labels[g])))
        else:
            truth = []

        src_tokens = parse_number_tokens(text)
        text_nums = sorted({x for tok in src_tokens for x in tok})
        exp_tokens = parse_number_tokens(explanation)
        cited = [t for t in exp_tokens if t & set(text_nums)]
        bad = [t for t in exp_tokens if not (t & set(text_nums))]
        v2_text = parse_numbers(text, version="v2")
        v2_bad = [x for x in parse_numbers(explanation, version="v2") if not number_in(v2_text, x)]

        bucket = ref_by_group[origin]
        bucket[2] += 1
        bucket[0] += 1 if cited else 0
        bucket[1] += 1 if bad else 0

        results.append({
            "sample_index": g,
            "true_label": int(labels[g]),
            "true_label_name": get_label_name(int(labels[g])),
            "raw_label": int(raw_labels[g]),
            "unknown_origin": origin,
            "origin_name": ORIGIN_NAMES.get(origin, str(origin)),
            "pred_label": num_known_classes,
            "pred_label_name": get_label_name(num_known_classes),
            "correct": int(labels[g]) == num_known_classes,
            "true_mitre_ids": truth,
            "predicted_mitre_id": parsed["mitre_technique_id"],
            "mapping_declared_null": ('"mitre_technique_id": "null"' in resp)
            or ('"mitre_technique_id": null' in resp),
            "explanation": explanation,
            "raw_response": resp,
            "text_description": text,
            "ood_score": float(ent["ood_score"]),
            "evidence": evidence,
            "n_numbers": len(exp_tokens),
            "n_verified": len(cited),
            "n_unverified": len(bad),
            "unverified_tokens": [sorted(t)[0] for t in bad[:8]],
            "v2_unverified_n": len(v2_bad),
        })

    n = len(results)
    referenced = sum(1 for r in results if r["n_verified"] > 0)
    unverified = sum(1 for r in results if r["n_unverified"] > 0)
    empty = sum(1 for r in results if not (r["explanation"] or "").strip())

    by_origin = {}
    for k, (ref, hall, tot) in sorted(ref_by_group.items()):
        by_origin[ORIGIN_NAMES.get(k, str(k))] = {
            "n": tot,
            "reference_rate": round(ref / tot, 4) if tot else None,
            "unverified_number_rate": round(hall / tot, 4) if tot else None,
        }

    hit = {}
    for k in sorted(ref_by_group):
        grp = [r for r in results if r["unknown_origin"] == k]
        if k == 1:
            hit[ORIGIN_NAMES[k]] = {"computed": False, "reason": "红线 #6：genuine 不计算命中率",
                                    "n": len(grp)}
            continue
        denom = [r for r in grp if r["true_mitre_ids"]]
        h = [r for r in denom if r["predicted_mitre_id"] in r["true_mitre_ids"]]
        # 检索基线：同一批样本上 top-1 / top-3 证据的集合命中，用于对比 LLM 映射
        h1 = [r for r in denom if r["evidence"] and r["evidence"][0]["id"] in r["true_mitre_ids"]]
        h3 = [r for r in denom if any(e["id"] in r["true_mitre_ids"] for e in r["evidence"][:3])]
        hit[ORIGIN_NAMES[k]] = {
            "computed": True,
            "n_group": len(grp),
            "hit": len(h),
            "denom": len(denom),
            "rate": round(len(h) / len(denom), 4) if denom else None,
            "retrieval_top1_hit": f"{len(h1)}/{len(denom)}" if denom else None,
            "retrieval_top3_hit": f"{len(h3)}/{len(denom)}" if denom else None,
            "note": "合并类按集合命中（predicted 属于真值集合即可）",
        }

    print("\n" + "=" * 60)
    print(f"生成完成：{n} 条；空解释 {empty} 条")
    print(f"引用率（v3）= {referenced}/{n} = {referenced / n:.4f}"
          f"（关卡 ≥0.90：{'PASS' if referenced / n >= 0.90 else 'FAIL'}）")
    print(f"未验证数字率（v3）= {unverified}/{n} = {unverified / n:.4f}")
    for k, v in by_origin.items():
        print(f"  - {k}: n={v['n']} 引用率={v['reference_rate']} 未验证={v['unverified_number_rate']}")
    for k, v in hit.items():
        print(f"  - 命中 {k}: {v}")

    out_path = args.output or os.path.join(
        PROJECT_ROOT, "explanation_results",
        f"unknown_explanations_model{report['backbone_model_id']}_ood{report['ood_id']}_"
        f"{time.strftime('%Y%m%d_%H%M%S')}.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "config": {
            "routing_report": os.path.relpath(rep_path, PROJECT_ROOT),
            "backbone_model_id": report["backbone_model_id"],
            "backbone_variant": report.get("backbone_variant"),
            "ood_id": report["ood_id"],
            "dataset_id": dataset_id,
            "split_id": split_id,
            "num_known_classes": num_known_classes,
            "n_samples": n,
            "seed": args.seed,
            "quota": {ORIGIN_NAMES[k]: v for k, v in sorted(quota.items())},
            "routed_population": {ORIGIN_NAMES[k]: v for k, v in sorted(avail.items())},
            "index_recovery": {"matched": len(pool), "unmatched": unmatched,
                               "method": "ood_score float32 精确匹配 + true_label 校验"},
            "llm_model_path": llm_model_path,
            "prompt_numeric_rule": "解释数字只能来自 text_description；检索证据禁止引用数字",
            "number_rule_version": "v3(unit_base+v2)",
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        },
        "summary": {
            "reference_rate": round(referenced / n, 4),
            "unverified_number_rate": round(unverified / n, 4),
            "gate_reference_rate_met": referenced / n >= 0.90,
            "empty_explanations": empty,
            "llm_declared_null_n": sum(1 for r in results if r["mapping_declared_null"]),
            "by_origin": by_origin,
            "mitre_hit": hit,
        },
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"结果保存至: {out_path}")


if __name__ == "__main__":
    main()
