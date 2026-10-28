"""生成物台账与隔离工具。

背景
----
saved_models / saved_ood_heads / test_reports / ood_reports 里同时存在三类东西：
当前可复现产物、历史实验产物、无效产物。run_history、logs/*.log 和 report_*.json
只记录 id（model_id / ood_id / backbone_model_id），不记录产物状态，因此直接删除
文件会让这些历史记录无法回溯。

本脚本的原则：不删除，只“标注 + 隔离”。
1. 扫描产物，按引用关系判定 status（active / deprecated / invalid）；
2. 生成唯一台账 artifacts/ledger.csv；
3. --apply 时把 deprecated / invalid 的 model、ood_head 移入 _archive/，并写
   _archive/MANIFEST.jsonl（原路径、摘要、原因），日志仍可解释；
   report / ood_report / split_data 一律不移动，保证 reports_index.csv 不悬空。

用法
----
  python scripts/audit_artifacts.py            # dry-run，打印统计与预览
  python scripts/audit_artifacts.py --apply    # 写台账 + 执行隔离
  python scripts/audit_artifacts.py --check    # 一致性校验，失败退出码 1
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "saved_models")
HEAD_DIR = os.path.join(PROJECT_ROOT, "saved_ood_heads")
REPORT_DIR = os.path.join(PROJECT_ROOT, "test_reports")
OOD_REPORT_DIR = os.path.join(PROJECT_ROOT, "ood_reports")
SPLIT_DIR = os.path.join(PROJECT_ROOT, "split_data")
EXPL_DIR = os.path.join(PROJECT_ROOT, "explanation_results")
ARCHIVE_DIR = os.path.join(PROJECT_ROOT, "_archive")
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts")
LEDGER_PATH = os.path.join(ARTIFACT_DIR, "ledger.csv")
MANIFEST_PATH = os.path.join(ARCHIVE_DIR, "MANIFEST.jsonl")

LEDGER_FIELDS = [
    "kind", "id", "path", "status", "reason", "superseded_by",
    "num_classes", "split_name", "size_bytes", "digest", "archive_path", "updated_at",
]

# 主数据集判定：模型数达到阈值的数据集视为主数据集，更少的视为子集/冒烟实验。
# 旧实现取"模型数最多"的唯一数据集，会把模型更少的新数据集误判为子集，
# 进而让 audit_artifacts.py --apply 把该数据集的有效模型移入 _archive。
MAIN_DATASET_MIN_MODELS = 5

# 人工状态覆盖：经复核确认的结论，不随报告引用关系自动判定，保证台账可重建且稳定。
FORCED_MODEL_STATUS = {
    2008: ("invalid",
           "checkpoint 不含 LoRA adapter（state_dict 缺陷），早停恢复不一致，"
           "闭集 known_acc 0.8508 不可用；非 add 变体不稳定",
           "model_2012"),
    2003: ("deprecated",
           "修复前训练的 concat 模型，checkpoint 不含 LoRA adapter，"
           "闭集 0.9861 / genuine AUROC 0.8502 有偏，不可作为 OOD 基准",
           "model_2013"),
}

# 人工状态覆盖（报告）：backbone 被覆盖为非 active 时，对应报告同步降级，避免台账自相矛盾。
FORCED_REPORT_STATUS = {
    29: ("deprecated",
         "backbone model_2003 为修复前训练（checkpoint 不含 LoRA adapter），"
         "闭集 0.9861 有偏，结果不可用",
         "report_37"),
    35: ("deprecated",
         "backbone model_2008 的 checkpoint 不含 LoRA adapter（state_dict 缺陷），"
         "闭集 0.8508 不可用",
         "report_36"),
}


def relpath(path):
    return os.path.relpath(path, PROJECT_ROOT).replace(os.sep, "/")


def resolve(rel):
    """返回产物的实际位置：优先 live，其次 _archive，都没有返回 None。"""
    live = os.path.join(PROJECT_ROOT, rel)
    if os.path.exists(live):
        return live
    archived = os.path.join(ARCHIVE_DIR, rel)
    if os.path.exists(archived):
        return archived
    return None


def numeric_children(prefix, directory):
    out = {}
    pattern = re.compile(r"^%s_(\d+)$" % re.escape(prefix))
    if not os.path.isdir(directory):
        return out
    for name in sorted(os.listdir(directory)):
        m = pattern.match(name)
        if m:
            out[int(m.group(1))] = os.path.join(directory, name)
    return out


def numeric_reports(directory):
    out = {}
    if not os.path.isdir(directory):
        return out
    for name in sorted(os.listdir(directory)):
        m = re.fullmatch(r"report_(\d+)\.json", name)
        if m:
            out[int(m.group(1))] = os.path.join(directory, name)
    return out


def dirsize(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def digest(path):
    """内容摘要：只对相对路径与文件大小做哈希，避免每次重算 2.7G 权重。"""
    h = hashlib.sha256()
    if os.path.isfile(path):
        h.update(os.path.basename(path).encode("utf-8"))
        h.update(str(os.path.getsize(path)).encode("ascii"))
        return h.hexdigest()
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for f in sorted(files):
            fp = os.path.join(root, f)
            h.update(os.path.relpath(fp, path).encode("utf-8"))
            h.update(str(os.path.getsize(fp)).encode("ascii"))
    return h.hexdigest()


def config_field(path, key):
    cfg = os.path.join(path, "config.txt")
    if not os.path.exists(cfg):
        return None
    try:
        for line in open(cfg, encoding="utf-8", errors="ignore"):
            if line.strip().startswith(key + ":"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def model_num_classes(path):
    value = config_field(path, "num_classes")
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def npz_keys(path):
    try:
        import numpy as np
        with np.load(path) as z:
            return set(z.files)
    except Exception:
        return set()


def split_candidates(dataset_id, split_id, openset=None, fewshot_k=None):
    ds_dir = os.path.join(SPLIT_DIR, "dataset_%s" % dataset_id)
    names = []
    if fewshot_k is not None:
        names.append("split_fewshot_%s_%s" % (fewshot_k, split_id))
    if openset is True:
        names.append("split_openset_%s" % split_id)
    elif openset is False:
        names.append("split_%s" % split_id)
    else:
        names += ["split_openset_%s" % split_id, "split_%s" % split_id]
    for name in names:
        p = os.path.join(ds_dir, name)
        if os.path.isdir(p):
            return p
    return None


def resolve_report_split(d):
    ds, si = d.get("dataset_id"), d.get("split_id")
    if ds is None or si is None:
        return None
    openset = d.get("openset")
    if openset is None:
        openset = (d.get("test_samples") or 0) > 2000
    return split_candidates(ds, si, openset=bool(openset))


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:  # noqa: BLE001
        return {}, str(e)


def scanned_children(prefix, live_dir, archive_dir):
    """同时扫描 live 与 _archive，返回 {id: {logical, real, archived}}。"""
    out = {}
    for i, path in numeric_children(prefix, live_dir).items():
        out[i] = {"logical": path, "real": path, "archived": False}
    for i, path in numeric_children(prefix, archive_dir).items():
        if i not in out:
            out[i] = {"logical": os.path.join(live_dir, "%s_%d" % (prefix, i)),
                      "real": path, "archived": True}
    return out


def collect():
    """扫描全部产物，返回台账行与待隔离清单（不移动任何文件）。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []

    reports, report_err = {}, {}
    for rid, path in sorted(numeric_reports(REPORT_DIR).items()):
        data, err = load_json(path)
        reports[rid] = data
        if err:
            report_err[rid] = err

    ood_reports = []
    for path in sorted(glob.glob(os.path.join(OOD_REPORT_DIR, "*.json"))):
        data, err = load_json(path)
        ood_reports.append((os.path.basename(path), data, err))

    heads = {}
    head_scan = scanned_children("ood_head", HEAD_DIR, os.path.join(ARCHIVE_DIR, "saved_ood_heads"))
    for hid, entry in sorted(head_scan.items()):
        data, err = load_json(os.path.join(entry["real"], "config.json"))
        heads[hid] = {"entry": entry, "config": data, "error": err}

    models = {}
    model_scan = scanned_children("model", MODEL_DIR, os.path.join(ARCHIVE_DIR, "saved_models"))
    for mid, entry in sorted(model_scan.items()):
        ds_raw = config_field(entry["real"], "dataset_id")
        models[mid] = {
            "entry": entry,
            "num_classes": model_num_classes(entry["real"]),
            "dataset_id": int(ds_raw) if (ds_raw or "").isdigit() else None,
        }
    model_dataset = {mid: info["dataset_id"] for mid, info in models.items()}
    ds_counts = {}
    for ds in model_dataset.values():
        if ds is not None:
            ds_counts[ds] = ds_counts.get(ds, 0) + 1
    main_datasets = {ds for ds, n in ds_counts.items() if n >= MAIN_DATASET_MIN_MODELS}

    # ---- 报告状态：先判无效，再判重复
    report_status, report_reason, report_super = {}, {}, {}
    signature = {}
    for rid in sorted(reports):
        d = reports[rid]
        if rid in report_err:
            report_status[rid] = "invalid"
            report_reason[rid] = "JSON 解析失败: %s" % report_err[rid]
            continue
        missing = [k for k in ("model_id", "dataset_id", "split_id", "test_samples")
                   if d.get(k) is None]
        if missing:
            report_status[rid] = "invalid"
            report_reason[rid] = "缺少字段 %s" % missing
            continue
        if resolve("saved_models/model_%s" % d["model_id"]) is None:
            report_status[rid] = "invalid"
            report_reason[rid] = "引用的 model_%s 不存在（live/_archive 均无）" % d["model_id"]
            continue
        if resolve_report_split(d) is None:
            report_status[rid] = "invalid"
            report_reason[rid] = ("引用的 dataset_%s / split_%s 不存在，"
                                  "该报告不可复现") % (d["dataset_id"], d["split_id"])
            continue
        mid_int = int(d["model_id"])
        if model_dataset.get(mid_int) is not None and model_dataset[mid_int] not in main_datasets:
            report_status[rid] = "deprecated"
            report_reason[rid] = ("非主数据集的子集/冒烟实验（dataset_%s，仅 %d 个模型使用），"
                                  "不可作为结论") % (model_dataset[mid_int],
                                                     ds_counts.get(model_dataset[mid_int], 0))
            continue
        report_status[rid] = "active"
        report_reason[rid] = ""
        signature.setdefault(
            (d["model_id"], d["dataset_id"], d["split_id"], d["test_samples"]), []
        ).append(rid)

    for key, rids in sorted(signature.items(), key=lambda kv: str(kv[0])):
        if len(rids) < 2:
            continue
        # 口径相同时以时间最新的一次为准：修复口径后的重跑应覆盖旧结果
        rids = sorted(rids, key=lambda r: reports[r].get("timestamp") or "",
                      reverse=True)
        canonical = rids[0]
        for rid in rids[1:]:
            if report_status.get(rid) == "active":
                report_status[rid] = "deprecated"
                report_reason[rid] = ("与 report_%s 同模型/同划分/同样本数，"
                                      "重复运行（以较新一次为准）；旧报告口径可能不同"
                                      % canonical)
                report_super[rid] = "report_%s" % canonical

    for rid, (f_status, f_reason, f_super) in FORCED_REPORT_STATUS.items():
        if rid in reports:
            report_status[rid] = f_status
            report_reason[rid] = f_reason
            report_super[rid] = f_super

    active_models = {int(reports[rid]["model_id"]) for rid in reports
                     if report_status.get(rid) == "active"}
    invalid_models = {int(reports[rid]["model_id"]) for rid in reports
                      if report_status.get(rid) == "invalid"}

    # ---- ood 报告状态：旧划分缺少 unknown_origin 一律降级
    ood_status, ood_reason, ood_head_refs = {}, {}, set()
    for name, d, err in ood_reports:
        if err:
            ood_status[name] = "invalid"
            ood_reason[name] = "JSON 解析失败: %s" % err
            continue
        mid = d.get("backbone_model_id")
        if mid is None or resolve("saved_models/model_%s" % mid) is None:
            ood_status[name] = "invalid"
            ood_reason[name] = "backbone model_id=%s 不存在" % mid
            continue
        split = split_candidates(d.get("dataset_id", 0), d.get("split_id", 0),
                                 openset=True, fewshot_k=d.get("fewshot_k"))
        if split is None:
            ood_status[name] = "invalid"
            ood_reason[name] = "引用的开集划分不存在"
            continue
        test_npz = os.path.join(split, "test.npz")
        if int(mid) in invalid_models or int(mid) not in active_models:
            ood_status[name] = "deprecated"
            ood_reason[name] = ("backbone model_%s 已被台账标为非 active（结果有偏/无效），"
                                "该报告随 backbone 降级" % mid)
        elif "unknown_origin" not in npz_keys(test_npz):
            ood_status[name] = "deprecated"
            ood_reason[name] = ("旧划分缺少 unknown_origin，指标为“真实 OOD + 模拟 "
                                "unknown”混合口径，不能当作真实 OOD 能力")
        else:
            ood_status[name] = "active"
            ood_reason[name] = ""
            if d.get("ood_id") is not None:
                ood_head_refs.add(int(d["ood_id"]))

    # ---- 台账行：报告
    for rid in sorted(reports):
        d = reports[rid]
        rows.append({
            "kind": "test_report", "id": "report_%d" % rid,
            "path": "test_reports/report_%d.json" % rid,
            "status": report_status.get(rid, "invalid"),
            "reason": report_reason.get(rid, ""),
            "superseded_by": report_super.get(rid, ""),
            "num_classes": d.get("num_classes", ""),
            "split_name": ("split_openset_%s" % d.get("split_id")) if (d.get("test_samples") or 0) > 2000 else ("split_%s" % d.get("split_id")),
            "size_bytes": dirsize(os.path.join(REPORT_DIR, "report_%d.json" % rid)),
            "digest": digest(os.path.join(REPORT_DIR, "report_%d.json" % rid)),
            "archive_path": "", "updated_at": now,
        })

    # ---- 台账行：模型
    for mid, info in sorted(models.items()):
        forced = FORCED_MODEL_STATUS.get(mid)
        if forced is not None:
            status, reason, super_by = forced
        elif mid in active_models:
            status, reason, super_by = "active", "被有效报告引用", ""
        elif mid in invalid_models:
            status, reason, super_by = "invalid", "仅被无效报告引用（其划分不存在）", ""
        elif info["dataset_id"] is not None and info["dataset_id"] not in main_datasets:
            status, reason, super_by = "deprecated", ("非主数据集 dataset_%s 的子集/冒烟实验"
                                            "（accuracy 近似随机），不可作为结论") % info["dataset_id"], ""
        else:
            status, reason, super_by = "deprecated", "未被任何有效报告引用，仅属历史实验或旧口径 ood_reports", ""
        entry = info["entry"]
        rows.append({
            "kind": "model", "id": "model_%d" % mid, "path": relpath(entry["logical"]),
            "status": status, "reason": reason, "superseded_by": super_by,
            "num_classes": info["num_classes"] if info["num_classes"] is not None else "",
            "split_name": "", "size_bytes": dirsize(entry["real"]),
            "digest": digest(entry["real"]),
            "archive_path": relpath(entry["real"]) if entry["archived"] else "",
            "updated_at": now,
        })

    # ---- 台账行：OOD 头
    for hid, info in sorted(heads.items()):
        cfg = info["config"]
        mid = cfg.get("model_id")
        if info["error"] or mid is None:
            status, reason = "invalid", "config.json 缺失或无法解析"
        elif resolve("saved_models/model_%s" % mid) is None:
            status, reason = "invalid", "引用的 backbone model_%s 不存在" % mid
        elif int(mid) in active_models and hid in ood_head_refs:
            status, reason = "active", "被有效 OOD 报告引用且 backbone 为 active"
        else:
            status, reason = "deprecated", "其 backbone 或引用报告已降级，无有效 OOD 报告使用"
        entry = info["entry"]
        rows.append({
            "kind": "ood_head", "id": "ood_head_%d" % hid, "path": relpath(entry["logical"]),
            "status": status, "reason": reason, "superseded_by": "",
            "num_classes": cfg.get("num_known_classes", ""), "split_name": "",
            "size_bytes": dirsize(entry["real"]), "digest": digest(entry["real"]),
            "archive_path": relpath(entry["real"]) if entry["archived"] else "",
            "updated_at": now,
        })

    # ---- 台账行：OOD 报告（原地保留，只标注）
    for name, d, err in ood_reports:
        path = os.path.join(OOD_REPORT_DIR, name)
        rows.append({
            "kind": "ood_report", "id": name, "path": relpath(path),
            "status": ood_status.get(name, "invalid"),
            "reason": ood_reason.get(name, ""), "superseded_by": "",
            "num_classes": "", "split_name": "split_openset_%s" % d.get("split_id", ""),
            "size_bytes": dirsize(path), "digest": digest(path),
            "archive_path": "", "updated_at": now,
        })

    # ---- 台账行：划分
    if os.path.isdir(SPLIT_DIR):
        for ds_name in sorted(os.listdir(SPLIT_DIR)):
            ds_path = os.path.join(SPLIT_DIR, ds_name)
            if not (ds_name.startswith("dataset_") and os.path.isdir(ds_path)):
                continue
            for sp_name in sorted(os.listdir(ds_path)):
                sp_path = os.path.join(ds_path, sp_name)
                if not (sp_name.startswith("split") and os.path.isdir(sp_path)):
                    continue
                test_npz = os.path.join(sp_path, "test.npz")
                if not os.path.exists(test_npz):
                    status, reason = "invalid", "缺少 test.npz"
                elif "unknown_origin" not in npz_keys(test_npz):
                    status, reason = "active", "旧划分：test.npz 无 unknown_origin，开集指标为混合口径（闭集不受影响）"
                else:
                    status, reason = "active", ""
                rows.append({
                    "kind": "split", "id": "%s/%s" % (ds_name, sp_name),
                    "path": relpath(sp_path), "status": status, "reason": reason,
                    "superseded_by": "", "num_classes": "", "split_name": sp_name,
                    "size_bytes": dirsize(sp_path), "digest": digest(sp_path),
                    "archive_path": "", "updated_at": now,
                })

    # ---- 台账行：解释结果（按 config.sampling 判定；随机抽样分母不可对齐全量）
    retrieval_runs = sorted(glob.glob(os.path.join(EXPL_DIR, "eval_retrieval_*.json")))
    latest_retrieval = os.path.basename(retrieval_runs[-1]) if retrieval_runs else ""
    for path in sorted(glob.glob(os.path.join(EXPL_DIR, "*.json"))):
        name = os.path.basename(path)
        if name.startswith("eval_retrieval_"):
            is_latest = name == latest_retrieval
            rows.append({
                "kind": "explanation_eval", "id": name, "path": relpath(path),
                "status": "active" if is_latest else "deprecated",
                "reason": ("CTI-RAG 离线检索评估（最新一次）：top-k 命中 true_mitre_ids 与每类 top-1，"
                           "不参与抽样口径判定") if is_latest
                          else "知识库关键词调优前的检索评估，已被较新一次取代",
                "superseded_by": "" if is_latest else latest_retrieval,
                "num_classes": "", "split_name": "",
                "size_bytes": dirsize(path), "digest": digest(path),
                "archive_path": "", "updated_at": now,
            })
            continue
        if name.startswith("eval_"):
            rows.append({
                "kind": "explanation_eval", "id": name, "path": relpath(path),
                "status": "active",
                "reason": "eval_explanations.py 生成的指标明细，不参与抽样口径判定",
                "superseded_by": "", "num_classes": "", "split_name": "",
                "size_bytes": dirsize(path), "digest": digest(path),
                "archive_path": "", "updated_at": now,
            })
            continue
        if name.startswith("unknown_explanations_"):
            unk_runs = sorted(glob.glob(os.path.join(EXPL_DIR, "unknown_explanations_*.json")))
            latest_unk = os.path.basename(unk_runs[-1]) if unk_runs else ""
            is_latest = name == latest_unk
            rows.append({
                "kind": "explanation_eval", "id": name, "path": relpath(path),
                "status": "active" if is_latest else "deprecated",
                "reason": ("unknown 路径 LLM 解释（最新一次）：按 unknown_origin 分层抽样，"
                           "命中率仅对 simulated 与 known-FP 统计") if is_latest
                          else "unknown 路径解释的早期迭代，已被较新一次取代",
                "superseded_by": "" if is_latest else latest_unk,
                "num_classes": "", "split_name": "",
                "size_bytes": dirsize(path), "digest": digest(path),
                "archive_path": "", "updated_at": now,
            })
            continue
        sampling = ""
        try:
            with open(path, encoding="utf-8") as f:
                sampling = (json.load(f).get("config") or {}).get("sampling", "")
        except Exception:
            sampling = ""
        if sampling == "stratified":
            status = "active"
            reason = ("按类比例分层抽样，各类配额记录在 config.per_class_sample_counts，"
                      "分母可对齐全量")
        else:
            status = "deprecated"
            reason = "随机抽样（未分层），各类样本数不稳定，分母不可与全量对齐"
        rows.append({
            "kind": "explanation", "id": name, "path": relpath(path),
            "status": status, "reason": reason,
            "superseded_by": "", "num_classes": "", "split_name": "",
            "size_bytes": dirsize(path), "digest": digest(path),
            "archive_path": "", "updated_at": now,
        })

    # ---- 台账行：同批实验的重复快照
    for path in sorted(glob.glob(os.path.join(REPORT_DIR, "openworld_*"))):
        rows.append({
            "kind": "openworld_snapshot", "id": os.path.basename(path),
            "path": relpath(path), "status": "deprecated",
            "reason": "同批实验的重复快照，正式结果以 report_1–12 为准",
            "superseded_by": "report_1..report_12", "num_classes": "", "split_name": "",
            "size_bytes": dirsize(path), "digest": digest(path),
            "archive_path": "", "updated_at": now,
        })

    # ---- 台账行：编辑器缓存
    cache = os.path.join(MODEL_DIR, ".ipynb_checkpoints")
    if os.path.isdir(cache):
        rows.append({
            "kind": "editor_cache", "id": ".ipynb_checkpoints",
            "path": relpath(cache), "status": "deprecated",
            "reason": "编辑器缓存，非实验产物", "superseded_by": "",
            "num_classes": "", "split_name": "", "size_bytes": dirsize(cache),
            "digest": digest(cache),
            "archive_path": (relpath(os.path.join(ARCHIVE_DIR, relpath(cache)))
                             if os.path.exists(os.path.join(ARCHIVE_DIR, relpath(cache))) else ""),
            "updated_at": now,
        })

    planned = [r for r in rows
               if r["kind"] in ("model", "ood_head", "editor_cache")
               and r["status"] in ("deprecated", "invalid")
               and resolve(r["path"]) == os.path.join(PROJECT_ROOT, r["path"])]
    return rows, planned


def apply_archive(planned):
    moved = []
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    for row in planned:
        src = os.path.join(PROJECT_ROOT, row["path"])
        dst = os.path.join(ARCHIVE_DIR, row["path"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        size, dig = dirsize(src), digest(src)
        shutil.move(src, dst)
        record = {
            "kind": row["kind"], "id": row["id"], "src": row["path"],
            "dst": relpath(dst), "status": row["status"], "reason": row["reason"],
            "size_bytes": size, "digest": dig,
            "moved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(MANIFEST_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        row["archive_path"] = relpath(dst)
        moved.append(record)
    return moved


def write_ledger(rows):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    with open(LEDGER_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["kind"], str(r["id"]))):
            writer.writerow({k: row.get(k, "") for k in LEDGER_FIELDS})


def check_artifact_consistency():
    """供 verify_caliber 调用的只读一致性校验。"""
    errors, warnings = [], []
    rows, _ = collect()
    status = {(r["kind"], r["id"]): r["status"] for r in rows}
    by_key = {(r["kind"], r["id"]): r for r in rows}

    idx_path = os.path.join(REPORT_DIR, "reports_index.csv")
    if os.path.exists(idx_path):
        with open(idx_path, encoding="utf-8-sig") as f:
            for rec in csv.DictReader(f):
                rid = "report_%s" % rec.get("report_id")
                row = by_key.get(("test_report", rid))
                if row is None:
                    errors.append("reports_index: %s 找不到对应报告文件" % rid)
                    continue
                if row["status"] == "invalid":
                    warnings.append("reports_index: %s 已被台账标记 invalid（%s）"
                                    % (rid, row["reason"]))
                if row["status"] == "deprecated":
                    warnings.append("reports_index: %s 已被台账标记 deprecated（%s）"
                                    % (rid, row["reason"]))

    active_sig = {}
    for row in rows:
        if row["kind"] != "test_report" or row["status"] != "active":
            continue
        key = row["path"]
        active_sig.setdefault(key, 0)
        active_sig[key] += 1

    for row in rows:
        if row["kind"] == "ood_head" and row["status"] == "invalid":
            errors.append("%s: %s" % (row["id"], row["reason"]))

    for row in rows:
        if row["kind"] == "test_report" and row["status"] == "active":
            if resolve(row["path"]) is None:
                errors.append("%s: 台账标记 active 但文件不存在" % row["id"])

    live_active = [r["id"] for r in rows if r["status"] == "active"
                   and r["kind"] in ("model", "ood_head")
                   and r["archive_path"]]
    for aid in live_active:
        errors.append("%s: 台账标记 active 却位于 _archive" % aid)

    for row in rows:
        if row["kind"] == "model" and row["status"] == "active" and row["num_classes"] == "":
            errors.append("%s: active 模型缺少 config.txt 的 num_classes" % row["id"])

    return errors, warnings


def print_stats(rows, planned):
    from collections import Counter
    counter = Counter((r["kind"], r["status"]) for r in rows)
    kinds = sorted({k for k, _ in counter})
    print("kind".ljust(20) + "active".rjust(8) + "deprecated".rjust(12) + "invalid".rjust(10))
    for kind in kinds:
        print(kind.ljust(20)
              + str(counter.get((kind, "active"), 0)).rjust(8)
              + str(counter.get((kind, "deprecated"), 0)).rjust(12)
              + str(counter.get((kind, "invalid"), 0)).rjust(10))


def main():
    parser = argparse.ArgumentParser(description="生成物台账与隔离工具")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--apply", action="store_true", help="写台账并执行隔离")
    group.add_argument("--check", action="store_true", help="只做一致性校验")
    args = parser.parse_args()

    if args.check:
        errors, warnings = check_artifact_consistency()
        for w in warnings:
            print("[WARN] %s" % w)
        if errors:
            print("[FAIL] %d 个产物一致性问题:" % len(errors))
            for e in errors:
                print("  - %s" % e)
            return 1
        print("[PASS] 产物引用关系一致")
        return 0

    rows, planned = collect()
    print_stats(rows, planned)
    planned_bytes = sum(r["size_bytes"] for r in planned)
    print("\n待隔离: %d 个产物, %.1f MB" % (len(planned), planned_bytes / 1e6))
    for row in planned[:10]:
        print("  - %-18s %-10s %s" % (row["id"], row["status"], row["reason"]))
    if len(planned) > 10:
        print("  ... 其余 %d 个见台账" % (len(planned) - 10))

    if not args.apply:
        print("\n[dry-run] 未写台账、未移动文件。加 --apply 执行。")
        return 0

    moved = apply_archive(planned)
    write_ledger(rows)
    print("\n[APPLY] 已隔离 %d 个产物，台账 %s" % (len(moved), LEDGER_PATH))
    return 0


if __name__ == "__main__":
    sys.exit(main())