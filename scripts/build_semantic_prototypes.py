"""构建类语义原型（BERT 空间，CPU 秒级）。

对应执行总纲 §4.3：P_llm = 类描述嵌入的均值。本脚本只做「确定性」的类描述生成：
类名 + MITRE 知识库检测线索 + 手工行为模板，不调用任何生成式 LLM 推理，
避免把「描述质量」变成不可复现因素。

编码方式与缓存文本向量严格一致：`src/model_architectures/bert_encoder.BertEncoder`
（取 [CLS]，不做池化、不做归一化）；归一化版本另存，供余弦距离使用。

产物：
  emb/bert_protos.npz         proto (10,768) 原始 CLS 均值；proto_norm (10,768) L2 归一化
  <out>/prototypes.json       每类 3 条描述 + KB 映射 + 编码元数据

用法：
  python scripts/build_semantic_prototypes.py \
      --split_dir split_data/dataset_200/split_openset_0 \
      --out results/2026-11-20/prototypes \
      --emb emb/bert_protos.npz
"""
import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.explain.knowledge_base import MITRE_KB  # noqa: E402

NUM_KNOWN = 10

# 标签顺序以 split 的 labels 为准：0..9
CLASS_NAMES = [
    "BENIGN", "DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest",
    "DDoS", "PortScan", "FTP-Patator", "SSH-Patator", "Bot",
]

# 类 -> KB 技术编号（None 表示正常流量，无对应攻击技术）
CLASS_KB = {
    "BENIGN": None,
    "DoS Hulk": "T1498.001",
    "DoS GoldenEye": "T1498.001",
    "DoS slowloris": "T1498.002",
    "DoS Slowhttptest": "T1498.002",
    "DDoS": "T1498.001",
    "PortScan": "T1046",
    "FTP-Patator": "T1110.001",
    "SSH-Patator": "T1110.001",
    "Bot": "T1071",
}

# 行为模板（d2）：描述攻击/正常行为的机理，不含标签编号
BEHAVIOR = {
    "BENIGN": "正常的业务访问流量，连接持续时间与包速率处于日常范围，双向字节数基本均衡，"
              "使用常见服务端口，不存在突发的高并发或重复登录行为。",
    "DoS Hulk": "针对 Web 服务的突发高并发请求攻击，包速率极高、并发连接数骤增，"
                "单条连接持续时间很短，目标端口为 80。",
    "DoS GoldenEye": "HTTP 洪水攻击，多个并发连接持续请求同一目标，连接保持时间中等，"
                     "包速率很高，目标端口为 80。",
    "DoS slowloris": "慢速 HTTP 攻击，客户端逐字节缓慢发送请求头，连接持续时间极长，"
                     "每条连接传输字节很少，大量半开连接堆积在服务端。",
    "DoS Slowhttptest": "慢速连接耗尽攻击，长期保持半开连接并缓慢发送数据，"
                        "连接持续时间长、包数少、包速率低。",
    "DDoS": "分布式拒绝服务攻击，多源并发向同一目标发送大流量，包速率与字节速率极高，"
            "源地址高度分散，目标带宽被饱和。",
    "PortScan": "端口扫描侦察，单一源在短时间内探测大量目标端口，单条连接持续时间极短，"
                "SYN 与 RST 报文频繁，单包字节数很少。",
    "FTP-Patator": "针对 FTP 服务的口令暴力破解，反复向 21 端口发起短连接并尝试登录，"
                   "连接时长与字节数都较小。",
    "SSH-Patator": "针对 SSH 服务的口令暴力破解，反复向 22 端口发起短连接并尝试登录，"
                   "连接时长短、包数少。",
    "Bot": "受控主机与命令控制服务器之间的周期性通信，长连接保持，固定间隔的心跳小包，"
           "使用非常规端口承载应用层协议。",
}

# 流统计形态模板（d3）：只描述可观测的统计形态
SHAPE = {
    "BENIGN": "流统计表现为中等持续时长、较低包速率、双向字节数均衡。",
    "DoS Hulk": "流统计表现为极短持续时长、极高包速率、单包字节数偏大。",
    "DoS GoldenEye": "流统计表现为较短持续时长、高包速率、并发连接密集。",
    "DoS slowloris": "流统计表现为极长持续时长、极低包速率、单包字节数很小。",
    "DoS Slowhttptest": "流统计表现为长持续时长、低包速率、连接数多而每连接字节少。",
    "DDoS": "流统计表现为极短持续时长、极高包速率与字节速率、源端口分散。",
    "PortScan": "流统计表现为极短持续时长、极少字节数、包速率高且目标端口连续。",
    "FTP-Patator": "流统计表现为短持续时长、小字节数、目标端口固定为 21。",
    "SSH-Patator": "流统计表现为短持续时长、小字节数、目标端口固定为 22。",
    "Bot": "流统计表现为长持续时长、低包速率、周期性小包。",
}


def kb_by_id():
    return {k["id"]: k for k in MITRE_KB}


def build_descriptions():
    kb = kb_by_id()
    out = {}
    for name in CLASS_NAMES:
        kbid = CLASS_KB[name]
        if kbid is None:
            d1 = f"{name}（正常流量）：无对应攻击技术，属于合法业务访问。"
        else:
            k = kb[kbid]
            d1 = f"{name} 类型的网络流量，对应攻击技术 {k['id']} {k['name']}：" \
                 f"{k['description']}{k['detection']}"
        d2 = BEHAVIOR[name]
        d3 = f"{name}：{SHAPE[name]}"
        out[name] = {"kb_id": kbid, "descriptions": [d1, d2, d3]}
    return out


def l2norm(a):
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.clip(n, 1e-12, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--emb", default="emb/bert_protos.npz")
    ap.add_argument("--bert_path", default="models/bert")
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    # 校验标签空间与描述顺序一致
    d = np.load(os.path.join(args.split_dir, "train.npz"))
    y = d["labels"]
    present = set(np.unique(y[y < NUM_KNOWN]).tolist())
    if present != set(range(NUM_KNOWN)):
        raise SystemExit(f"已知类标签不完整: {sorted(present)}")

    protos_info = build_descriptions()
    texts, owner = [], []
    for c, name in enumerate(CLASS_NAMES):
        for t in protos_info[name]["descriptions"]:
            texts.append(t)
            owner.append(c)

    from src.model_architectures.bert_encoder import BertEncoder
    enc = BertEncoder(local_model_path=args.bert_path, batch_size=args.batch_size, force_cpu=True)
    E = enc(texts).numpy().astype(np.float32)          # (30, 768)
    E = E.reshape(NUM_KNOWN, 3, -1).mean(axis=1)        # (10, 768)
    En = l2norm(E).astype(np.float32)

    os.makedirs(os.path.dirname(args.emb) or ".", exist_ok=True)
    np.savez(args.emb, proto=E, proto_norm=En,
             labels=np.arange(NUM_KNOWN, dtype=np.int64))
    os.makedirs(args.out, exist_ok=True)
    meta = {
        "experiment": "类语义原型（BERT，P_llm）",
        "split_dir": args.split_dir,
        "bert_path": args.bert_path,
        "encoder": "BertEncoder([CLS], 无池化/无归一化)；与 text_embeddings 缓存一致",
        "n_classes": NUM_KNOWN,
        "n_desc_per_class": 3,
        "class_names": CLASS_NAMES,
        "prototypes": protos_info,
        "proto_raw_norm_mean": float(np.linalg.norm(E, axis=1).mean()),
        "pairwise_cosine_mean": float(
            (En @ En.T)[np.triu_indices(NUM_KNOWN, 1)].mean()),
        "artifact": args.emb,
    }
    with open(os.path.join(args.out, "prototypes.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] 原型 {E.shape} -> {args.emb}")
    print(f"[OK] 元数据 -> {os.path.join(args.out, 'prototypes.json')}")
    print(f"类间平均余弦相似度 = {meta['pairwise_cosine_mean']:.4f}")


if __name__ == "__main__":
    main()
