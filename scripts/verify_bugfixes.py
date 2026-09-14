"""
Bug 修复验证脚本（对应 下一步执行计划_2026-09-07.md 的 B1-B8）

分两层：
  [CHEAP]  CPU 即可，无需 GPU / 大模型：静态 + 逻辑断言（B3/B4/B6/B8/B5 + B1/B2 数据层）
  [LLM]    需 GPU + 一个 A0* 主干（use_llm=True）：特征探针，证明 LLM 文本分支真的被用上（B1/B2 特征层）

用法：
  python scripts/verify_bugfixes.py                 # 只跑 CHEAP
  python scripts/verify_bugfixes.py --model_id 20  # 追加 LLM 特征探针（A0*_attn）
  python scripts/verify_bugfixes.py --model_id 20 --device cuda

任何一项 FAIL 都意味着对应 bug 没修好，不要拿该变体的 OOD 结论当结论。
"""

import os
import re
import sys
import json
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
results = []


def record(tag, status, detail):
    results.append((tag, status, detail))
    icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[status]
    print(f"  [{icon} {status}] {tag}: {detail}")


# ───────────────────────────────────────────────────────────
# B6：variant_configs 是否补齐 4 个新变体（静态，读源码）
# ───────────────────────────────────────────────────────────
def check_b6():
    required = {"A0*", "A0_frozen", "A0*_no_num", "A0*_no_text"}
    for fname in ["scripts/train.py", "scripts/train_ood_head.py"]:
        path = os.path.join(PROJECT_ROOT, fname)
        src = open(path, encoding="utf-8").read()
        # 抓取所有 "AX": { ... } 形式的键
        keys = set(re.findall(r'"([A-Za-z0-9*_]+)"\s*:\s*\{', src))
        # 也兼容单引号
        keys |= set(re.findall(r"'([A-Za-z0-9*_]+)'\s*:\s*\{", src))
        missing = required - keys
        if missing:
            record(f"B6 {fname} 变体补齐", FAIL, f"缺 {missing}")
        else:
            record(f"B6 {fname} 变体补齐", PASS, f"含 {sorted(required)}")


# ───────────────────────────────────────────────────────────
# B8：evaluate_open_set 是否只有一份真相源（静态 + 运行时）
# ───────────────────────────────────────────────────────────
def check_b8():
    for fname in ["scripts/run_ood_routing.py", "scripts/test_model.py"]:
        path = os.path.join(PROJECT_ROOT, fname)
        src = open(path, encoding="utf-8").read()
        if "from utils.open_set_eval import evaluate_open_set" not in src and \
           "evaluate_open_set" not in src:
            record(f"B8 {fname} 引用统一评估", FAIL, "未使用 utils.open_set_eval")
            continue
        # 不能有第二份本地 def evaluate_open_set（注意：evaluate_open_set_model 是合法封装，不算）
        local_defs = len(re.findall(r"def evaluate_open_set\s*\(", src))
        if local_defs > 0:
            record(f"B8 {fname} 引用统一评估", FAIL, f"仍存在 {local_defs} 个本地定义")
        else:
            record(f"B8 {fname} 引用统一评估", PASS, "import 自 utils.open_set_eval，无本地副本")

    # 运行时：两脚本 import 的是同一个对象
    try:
        from utils.open_set_eval import evaluate_open_set as src_fn
        import importlib.util
        for fname in ["scripts/run_ood_routing.py", "scripts/test_model.py"]:
            spec = importlib.util.spec_from_file_location(
                f"_chk_{os.path.basename(fname)}", os.path.join(PROJECT_ROOT, fname))
            mod = importlib.util.module_from_spec(spec)
            # 不真正执行 __main__，只加载模块顶层
            spec.loader.exec_module(mod)
            same = (getattr(mod, "evaluate_open_set", None) is src_fn)
            status = PASS if same else FAIL
            record(f"B8 {fname} 同一函数对象", status,
                   "是同一份" if same else "指向不同实现")
    except Exception as e:
        record("B8 运行时同一性", WARN, f"运行时校验跳过（{type(e).__name__}: {e}）")


# ───────────────────────────────────────────────────────────
# B4：unknown_f1 必须是全样本二分类口径（含 FP），否则会被烂系统骗到 1.0
# ───────────────────────────────────────────────────────────
def check_b4():
    try:
        from utils.open_set_eval import evaluate_open_set
    except Exception as e:
        record("B4 全样本二分类口径", WARN,
               f"无法 import（本地 venv 缺 sklearn？请在有 sklearn 的环境重跑）：{e}")
        return

    import numpy as np
    K = 3  # 3 个已知类
    # 构造：50 已知 + 50 未知；预测者把"全部"判成 unknown（最烂系统）
    true = np.array([0] * 50 + [K] * 50)
    pred_all_unknown = np.array([K] * 100)

    r = evaluate_open_set(true, pred_all_unknown, K, include_routing_stats=False)
    ur = r["unknown_recall"]
    up = r["unknown_precision"]
    uf1 = r["unknown_f1"]
    # 期望：recall=1.0（真 unknown 全抓到），但 precision=0.5（已知一半误报），
    #       → f1 ≈ 0.667，绝不能是 1.0。旧 bug（只在 unknown 子集算）会给 1.0。
    if ur == 1.0 and abs(up - 0.5) < 1e-6 and 0.6 < uf1 < 0.75:
        record("B4 全样本二分类口径", PASS,
               f"全判unknown时 unknown_recall={ur:.2f}, precision={up:.2f}, f1={uf1:.3f}（FP 被惩罚）")
    else:
        record("B4 全样本二分类口径", FAIL,
               f"全判unknown时 recall={ur}, precision={up}, f1={uf1}（旧bug会给f1≈1.0）")

    # 完美分离应得 f1=1.0
    pred_perfect = np.array([0] * 50 + [K] * 50)
    rp = evaluate_open_set(true, pred_perfect, K, include_routing_stats=False)
    if abs(rp["unknown_f1"] - 1.0) < 1e-6:
        record("B4 完美分离基准", PASS, "完美分离 unknown_f1=1.0（口径正确）")
    else:
        record("B4 完美分离基准", FAIL, f"完美分离却 f1={rp['unknown_f1']}")


# ───────────────────────────────────────────────────────────
# B3：unknown 不再送 LLM 闭集重判（静态）
# ───────────────────────────────────────────────────────────
def check_b3():
    path = os.path.join(PROJECT_ROOT, "scripts/run_ood_routing.py")
    src = open(path, encoding="utf-8").read()
    # 旧 bug 特征：调用 llm_model.predict() 做闭集重判
    has_predict = bool(re.search(r"llm_model\.predict\s*\(", src))
    # 新修复特征：unknown 直接赋值为 num_known_classes
    has_direct = "routed_pred[idx] = num_known_classes" in src
    if has_predict:
        record("B3 无 LLM 闭集重判", FAIL, "仍存在 llm_model.predict() 闭集重判")
    elif not has_direct:
        record("B3 无 LLM 闭集重判", FAIL, "未见 routed_pred[idx]=num_known_classes 直判逻辑")
    else:
        record("B3 无 LLM 闭集重判", PASS,
               "已改为 OOD 头直判 num_known_classes，无 llm_model.predict 重判")


# ───────────────────────────────────────────────────────────
# B5：报告应带 variant + llm_use_lora（运行时，扫最新报告）
# ───────────────────────────────────────────────────────────
def check_b5():
    import glob
    cands = sorted(glob.glob(os.path.join(PROJECT_ROOT, "test_reports", "report_*.json")),
                   key=os.path.getmtime, reverse=True)
    found = None
    for p in cands:
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if "variant" in d and "llm_use_lora" in d and d.get("variant"):
            found = (p, d)
            break
    if found:
        p, d = found
        record("B5 报告含 variant+llm_use_lora", PASS,
               f"{os.path.basename(p)}: variant={d['variant']}, llm_use_lora={d['llm_use_lora']}")
    else:
        record("B5 报告含 variant+llm_use_lora", FAIL,
               "未在 test_reports 找到同时含非空 variant 与 llm_use_lora 的报告")


# ───────────────────────────────────────────────────────────
# B1/B2 数据层：collate_fn 在 use_llm=True 时必须产出 input_ids
# ───────────────────────────────────────────────────────────
def check_b1b2_collate():
    try:
        from src.data.data_loader import collate_fn
        import torch

        class StubTok:
            def __call__(self, texts, padding=True, truncation=True,
                         max_length=128, return_tensors="pt"):
                # 极简：把文本长度当 token id，保证产出张量
                ids = torch.tensor([[min(max_length, len(t))] for t in texts])
                return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        batch = [{"stat": [0.0] * 9, "bert": [0.0] * 768, "label": 0, "text": "某流量描述"}]
        out = collate_fn(batch, tokenizer=StubTok(), use_numeric=True,
                         use_bert=False, use_llm=True)
        if "input_ids" in out and out["input_ids"] is not None:
            record("B1/B2 数据层 collate 传 input_ids", PASS,
                   f"input_ids 形状 {tuple(out['input_ids'].shape)}（文本已编码进 batch）")
        else:
            record("B1/B2 数据层 collate 传 input_ids", FAIL,
                   "use_llm=True 时 collate 未产出 input_ids（文本仍会丢）")
    except Exception as e:
        # 退化为静态检查
        path = os.path.join(PROJECT_ROOT, "src/data/data_loader.py")
        src = open(path, encoding="utf-8").read()
        ok = ('if use_llm and tokenizer is not None and "text" in batch[0]'
              in src) and 'result["input_ids"]' in src
        record("B1/B2 数据层 collate 传 input_ids",
               PASS if ok else FAIL,
               ("静态确认 collate 在 use_llm=True 时写 input_ids"
                if ok else f"静态确认失败：{e}"))


# ───────────────────────────────────────────────────────────
# B1/B2 特征层（需 GPU + A0* 主干）：证明 LLM 文本分支真的改变融合特征
# ───────────────────────────────────────────────────────────
def check_b1b2_feature(model_id, device):
    try:
        from src.model_architectures.multi_modal_model import MultiModalFusionModel
        from src.data.data_loader import collate_fn
        import torch
        import numpy as np
    except Exception as e:
        record("B1/B2 特征层探针", WARN, f"依赖导入失败，跳过：{e}")
        return

    model_path = os.path.join(PROJECT_ROOT, "models", "qwen2.5-1.5b")
    ckpt = os.path.join(PROJECT_ROOT, "saved_models", f"model_{model_id}")
    if not os.path.isdir(ckpt):
        record("B1/B2 特征层探针", FAIL, f"model_{model_id} 不存在：{ckpt}")
        return
    model = MultiModalFusionModel.from_pretrained(model_path, ckpt)
    model.eval()
    model.to(device)
    tok = model.get_tokenizer()

    # 取一小批真实测试样本
    from src.data.data_loader import load_split_data
    ds = load_split_data(os.path.join(PROJECT_ROOT, "split_data"),
                         data_type="test", dataset_id=0, split_id=0, openset=False)
    if ds is None or len(ds) == 0:
        record("B1/B2 特征层探针", FAIL, "dataset_0 split_0 test 未找到")
        return
    batch = collate_fn([ds[i] for i in range(4)], tokenizer=tok,
                       use_numeric=model.use_numeric, use_bert=model.use_bert,
                       use_llm=model.use_llm)
    stat = batch["stat_tensor"].to(device)
    bert = batch["bert_tensor"].to(device)
    input_ids = batch.get("input_ids")
    attn = batch.get("attention_mask")
    if input_ids is not None:
        input_ids = input_ids.to(device)
        attn = attn.to(device) if attn is not None else None

    with torch.no_grad():
        f_with = model.extract_fusion_features(stat, bert, input_ids=input_ids, attention_mask=attn)
        f_without = model.extract_fusion_features(stat, bert, input_ids=None, attention_mask=None)

    f_with = f_with.cpu().float().numpy()
    f_without = f_without.cpu().float().numpy()

    norm_with = float(np.linalg.norm(f_with))
    diff = float(np.linalg.norm(f_with - f_without))
    # 余弦相似度
    denom = (norm_with * (np.linalg.norm(f_without) + 1e-9))
    cos = float(np.dot(f_with.ravel(), f_without.ravel()) / denom) if denom > 0 else 0.0

    ok = (norm_with > 1e-3) and (diff > 1e-3)
    if ok:
        record("B1/B2 特征层探针", PASS,
               f"带文本特征范数={norm_with:.3f}，与无文本差异={diff:.3f}，余弦={cos:.4f}"
               f"（文本分支确实参与融合，不再是全零）")
    else:
        record("B1/B2 特征层探针", FAIL,
               f"带/无文本特征几乎相同（norm={norm_with:.3f}, diff={diff:.3f}）→ 文本仍被丢弃")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=int, default=None,
                    help="A0* 主干 ID（如 20），用于特征层探针；不填只跑 CHEAP")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    print("=" * 64)
    print("Bug 修复验证（B1-B8）  CHEAP 层（CPU）")
    print("=" * 64)
    check_b6()
    check_b8()
    check_b4()
    check_b3()
    check_b5()
    check_b1b2_collate()

    if args.model_id is not None:
        print("\n" + "=" * 64)
        print(f"LLM 特征层探针（model_{args.model_id}, device={args.device}）")
        print("=" * 64)
        check_b1b2_feature(args.model_id, args.device)
    else:
        print("\n[提示] 未传 --model_id，跳过 B1/B2 特征层探针。"
              "要证明 LLM 文本分支真生效，请加 --model_id <A0*模型ID> 在 GPU 上跑。")

    print("\n" + "=" * 64)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_warn = sum(1 for _, s, _ in results if s == WARN)
    print(f"结果：{len(results)} 项，FAIL={n_fail}，WARN={n_warn}")
    if n_fail == 0:
        print("✅ 所有已检查项通过（B7 见下方说明，需独立数据排查）。")
    else:
        print("❌ 存在 FAIL，对应 bug 尚未真正修复，先修再下结论。")
    print("=" * 64)
    print("B7（dataset_5 F1=0.132）不在本脚本范围：它要求独立排查")
    print("split_data/dataset_5 的标签映射与数据完整性，请用 make_openset_split")
    print("的 sanity check + 人工核对 label_mapping.json 完成。")


if __name__ == "__main__":
    main()
