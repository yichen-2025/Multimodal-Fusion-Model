"""Phase 2 B1: LLM 语义解释层。

功能：
1. 用 backbone 对流量样本做实时分类预测（支持 A3/BERT 和 A0*/LLM 两种模式）
2. 调用基座 LLM（Qwen2.5-1.5b）生成自然语言解释 + MITRE ATT&CK 技术映射
3. 批量生成并缓存结果，便于后续评估

设计原则：
- backbone 做分类（根据模型配置自动适配 use_llm / use_bert）
- 基座 Qwen 做生成（不使用分类 LoRA，避免生成能力被削弱）
- 离线批量生成 + JSON 缓存
"""

import os
import re
import json
import time
import argparse
from typing import Dict, List, Optional, Any, Tuple

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

from src.model_architectures.multi_modal_model import MultiModalFusionModel
from src.explain.attack_mapping import (
    get_label_name,
    get_all_mitre_techniques,
    get_true_mitre_ids,
)
from utils.caliber import read_model_num_classes

# src/explain/generate_explanation.py -> 上三级为项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==================== Prompt 模板 ====================

SYSTEM_PROMPT = """你是一名网络安全分析专家，擅长分析网络流量特征并识别攻击类型。
你的任务是：根据给定的流量文本特征和分类模型的预测结果，生成自然语言解释，并映射到 MITRE ATT&CK 技术编号。

要求：
1. 解释要具体引用流量特征中的关键字段（如端口、包长度、标志位等）
2. 解释要说明"为什么这些特征表明是该攻击类型"
3. 若预测类别为 BENIGN（正常流量），mitre_technique_id 必须为 null，解释应说明该流量为何是正常的
4. 若预测类别为攻击类，MITRE ATT&CK 编号必须从候选列表中选择，不要编造
5. 输出严格为 JSON 格式，包含两个字段：explanation（中文解释）和 mitre_technique_id（ATT&CK 编号，正常流量为 null）"""


def build_user_prompt(
    text_description: str,
    pred_label: int,
    pred_label_name: str,
) -> str:
    """构建单条样本的 user prompt（不泄漏 GT MITRE 提示）。"""
    # 候选 ATT&CK 技术列表
    techniques = get_all_mitre_techniques()
    tech_list_str = "\n".join(
        [f"  - {t['id']}: {t['name']}" for t in techniques]
    )

    prompt = f"""流量文本特征：
{text_description}

分类模型预测结果：{pred_label_name}（label={pred_label}）

候选 MITRE ATT&CK 技术列表：
{tech_list_str}

请生成 JSON 格式的分析结果："""
    return prompt


# ==================== 模型加载 ====================

def load_backbone_model(model_id: int, device: str = "cuda") -> Tuple[MultiModalFusionModel, Optional[Any]]:
    """加载 backbone 模型用于分类预测。

    Returns:
        (model, tokenizer): tokenizer 仅在 use_llm=True 时返回，否则为 None
    """
    model_path = os.path.join(PROJECT_ROOT, "saved_models", f"model_{model_id}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型目录不存在: {model_path}")

    llm_model_path = os.path.join(PROJECT_ROOT, "models", "qwen2.5-1.5b")
    model = MultiModalFusionModel.from_pretrained(llm_model_path, model_path)
    model.to(device)
    model.eval()

    # 获取 LLM tokenizer（仅 use_llm=True 时有意义，A0* 等 LLM 类模型需要）
    tokenizer = model.get_tokenizer() if model.use_llm else None
    if tokenizer is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_llm_for_generation(llm_model_path: str, device: str = "cuda"):
    """加载基座 LLM 用于文本生成（不加载任何 LoRA adapter）。"""
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path, local_files_only=True)
    # 确保有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        llm_model_path,
        torch_dtype=llm_dtype,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


# ==================== 数据加载 ====================

def load_test_samples(
    dataset_id: int = 0,
    split_id: int = 0,
    n_samples: Optional[int] = None,
    seed: int = 42,
    openset: bool = True,
) -> List[Dict[str, Any]]:
    """加载测试集样本（支持开集/闭集划分）。

    返回列表，每个元素包含 stat, bert, label, text。
    """
    split_name = f"split_openset_{split_id}" if openset else f"split_{split_id}"
    split_dir = os.path.join(
        PROJECT_ROOT, "split_data", f"dataset_{dataset_id}", split_name
    )
    npz_path = os.path.join(split_dir, "test.npz")
    csv_path = os.path.join(split_dir, "test_data.csv")

    npz_data = np.load(npz_path, allow_pickle=True)
    stat_features = npz_data["scaled_features"]
    bert_embeddings = npz_data["text_embeddings"]
    labels = npz_data["labels"]

    # 加载文本描述
    text_descriptions = [""] * len(labels)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if "text_description" in df.columns:
            text_descriptions = df["text_description"].tolist()

    samples = []
    for i in range(len(labels)):
        samples.append(
            {
                "index": i,
                "stat": stat_features[i].astype(np.float32),
                "bert": bert_embeddings[i].astype(np.float32),
                "label": int(labels[i]),
                "text": text_descriptions[i] if i < len(text_descriptions) else "",
            }
        )

    # 抽样
    if n_samples is not None and n_samples < len(samples):
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(samples), size=n_samples, replace=False)
        samples = [samples[i] for i in sorted(indices)]

    print(f"加载 {len(samples)} 条测试样本（dataset={dataset_id}, {split_name}）")
    return samples


# ==================== 预测 ====================

@torch.no_grad()
def predict_with_backbone(
    model: MultiModalFusionModel,
    samples: List[Dict[str, Any]],
    device: str = "cuda",
    batch_size: int = 64,
    tokenizer=None,
) -> List[int]:
    """用 backbone 批量预测样本类别。

    自动适配两种模式：
    - use_bert=True (A3): 直接使用预计算的 bert_embeddings
    - use_llm=True (A0*): 对文本分词后通过 LLM 编码，需传入 input_ids/attention_mask
    """
    use_llm = model.use_llm
    predictions = []
    for i in tqdm(range(0, len(samples), batch_size), desc="Backbone 预测"):
        batch = samples[i : i + batch_size]
        stat_tensor = torch.tensor(
            np.stack([s["stat"] for s in batch]), dtype=torch.float32
        ).to(device)
        bert_tensor = torch.tensor(
            np.stack([s["bert"] for s in batch]), dtype=torch.float32
        ).to(device)

        # LLM 类模型需要对文本分词，传入 input_ids/attention_mask
        input_ids = None
        attention_mask = None
        if use_llm and tokenizer is not None:
            texts = [s["text"] for s in batch]
            encodings = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

        result = model(stat_tensor, bert_tensor, input_ids=input_ids, attention_mask=attention_mask)
        logits = result["logits"]
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        predictions.extend(preds)
    return predictions


# ==================== LLM 生成 ====================

@torch.no_grad()
def generate_explanation(
    llm_model,
    tokenizer,
    text_description: str,
    pred_label: int,
    device: str = "cuda",
    max_new_tokens: int = 300,
) -> Dict[str, Any]:
    """调用 LLM 生成单条样本的解释。"""
    pred_label_name = get_label_name(pred_label)

    user_prompt = build_user_prompt(
        text_description=text_description,
        pred_label=pred_label,
        pred_label_name=pred_label_name,
    )

    # 构造对话消息
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Qwen2.5 的 chat template
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # fallback: 手动拼接
        text = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(
        device
    )

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )

    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # 解析 JSON
    result = parse_llm_response(response, pred_label)
    result["raw_response"] = response
    return result


def _normalize_mitre_id(value: Any) -> Optional[str]:
    """规范化 MITRE 编号：从 'T1110.001: xxx' 中提取纯编号 'T1110.001'。"""
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    # 匹配 T 开头，后跟数字和点的模式
    match = re.search(r"T\d{4}(?:\.\d{3})?", value)
    if match:
        return match.group(0)
    return None


def parse_llm_response(response: str, pred_label: int) -> Dict[str, Any]:
    """解析 LLM 返回的 JSON 响应。"""
    result = {"explanation": "", "mitre_technique_id": None}

    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    parsed = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                parsed = None

    if parsed is not None:
        result["explanation"] = parsed.get("explanation", "")
        result["mitre_technique_id"] = _normalize_mitre_id(parsed.get("mitre_technique_id"))
    else:
        result["explanation"] = response.strip()
        result["mitre_technique_id"] = None

    return result


# ==================== 主流程 ====================

def run_explanation_generation(
    backbone_model_id: int = 110,
    dataset_id: int = 0,
    split_id: int = 0,
    n_samples: Optional[int] = 100,
    llm_model_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    device: str = "cuda",
    seed: int = 42,
    openset: bool = True,
) -> str:
    """执行完整的解释生成流程。

    Returns:
        输出文件路径
    """
    if llm_model_path is None:
        llm_model_path = os.path.join(PROJECT_ROOT, "models", "qwen2.5-1.5b")

    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "explanation_results")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        output_dir, f"explanations_model{backbone_model_id}_{timestamp}.json"
    )

    # 1. 加载 backbone
    print(f"\n{'=' * 60}")
    print(f"[1/4] 加载 backbone: model_{backbone_model_id}")
    backbone, backbone_tokenizer = load_backbone_model(backbone_model_id, device=device)

    # 已知类数从模型 config.txt 读取（单一口径），用于剔除 unknown 的 MITRE 分母
    num_known_classes = read_model_num_classes(
        os.path.join(PROJECT_ROOT, "saved_models", f"model_{backbone_model_id}")
    )

    # 2. 加载测试样本
    print(f"\n[2/4] 加载测试样本（n_samples={n_samples}）")
    samples = load_test_samples(dataset_id, split_id, n_samples, seed, openset=openset)

    # 3. backbone 预测
    print(f"\n[3/4] Backbone 批量预测")
    predictions = predict_with_backbone(backbone, samples, device=device, tokenizer=backbone_tokenizer)
    for s, p in zip(samples, predictions):
        s["pred_label"] = p

    # 4. LLM 生成解释
    print(f"\n[4/4] LLM 生成解释（基座: {os.path.basename(llm_model_path)}）")
    llm_model, tokenizer = load_llm_for_generation(llm_model_path, device=device)

    results = []
    for i, sample in enumerate(tqdm(samples, desc="LLM 生成")):
        try:
            llm_result = generate_explanation(
                llm_model=llm_model,
                tokenizer=tokenizer,
                text_description=sample["text"],
                pred_label=sample["pred_label"],
                device=device,
            )
        except Exception as e:
            llm_result = {
                "explanation": f"[ERROR] {str(e)}",
                "mitre_technique_id": None,
                "raw_response": "",
            }

        true_label_name = get_label_name(sample["label"])
        pred_label_name = get_label_name(sample["pred_label"])
        true_mitre_ids = sorted(get_true_mitre_ids(sample["label"]))

        results.append(
            {
                "sample_index": sample["index"],
                "true_label": sample["label"],
                "true_label_name": true_label_name,
                "pred_label": sample["pred_label"],
                "pred_label_name": pred_label_name,
                "correct": sample["label"] == sample["pred_label"],
                "true_mitre_id": true_mitre_ids[0] if len(true_mitre_ids) == 1 else None,
                "true_mitre_ids": true_mitre_ids,
                "predicted_mitre_id": llm_result["mitre_technique_id"],
                "explanation": llm_result["explanation"],
                "raw_response": llm_result.get("raw_response", ""),
                "text_description": sample["text"],
            }
        )

    # 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "backbone_model_id": backbone_model_id,
                    "dataset_id": dataset_id,
                    "split_id": split_id,
                    "n_samples": n_samples,
                    "openset": openset,
                    "num_known_classes": num_known_classes,
                    "llm_model_path": llm_model_path,
                    "seed": seed,
                    "timestamp": timestamp,
                },
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 打印统计
    correct_count = sum(1 for r in results if r["correct"])
    # D3: unknown（label >= num_known_classes）不进 MITRE 分母；
    #      合并类用集合命中率（预测 ID 属于真值集合即可）
    mitre_match = sum(
        1
        for r in results
        if r["true_label"] < num_known_classes
        and len(r["true_mitre_ids"]) > 0
        and r["predicted_mitre_id"] in r["true_mitre_ids"]
    )
    mitre_total = sum(
        1
        for r in results
        if r["true_label"] < num_known_classes and len(r["true_mitre_ids"]) > 0
    )

    print(f"\n{'=' * 60}")
    print(f"生成完成！共 {len(results)} 条")
    print(f"  - Backbone 预测准确率: {correct_count}/{len(results)} = {correct_count/len(results):.2%}")
    if mitre_total > 0:
        print(f"  - MITRE 映射准确率: {mitre_match}/{mitre_total} = {mitre_match/mitre_total:.2%}")
    print(f"  - 结果保存至: {output_path}")

    # 释放 GPU 显存
    del backbone, llm_model
    torch.cuda.empty_cache()

    return output_path


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(description="Phase 2 B1: LLM 语义解释生成")
    parser.add_argument(
        "--backbone_model_id",
        type=int,
        default=110,
        help="backbone 模型 ID（默认 110）",
    )
    parser.add_argument("--dataset_id", type=int, default=0, help="数据集 ID")
    parser.add_argument("--split_id", type=int, default=0, help="划分 ID")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="抽样数量（None=全量，默认 100）",
    )
    parser.add_argument(
        "--llm_model_path",
        type=str,
        default=None,
        help="基座 LLM 路径（默认 models/qwen2.5-1.5b）",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="输出目录（默认 explanation_results/）"
    )
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--openset",
        dest="openset",
        action="store_true",
        default=True,
        help="加载开集划分 split_openset_*（默认）",
    )
    parser.add_argument(
        "--no-openset",
        dest="openset",
        action="store_false",
        help="加载闭集划分 split_*",
    )
    args = parser.parse_args()

    run_explanation_generation(
        backbone_model_id=args.backbone_model_id,
        dataset_id=args.dataset_id,
        split_id=args.split_id,
        n_samples=args.n_samples if args.n_samples > 0 else None,
        llm_model_path=args.llm_model_path,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        openset=args.openset,
    )


if __name__ == "__main__":
    main()
