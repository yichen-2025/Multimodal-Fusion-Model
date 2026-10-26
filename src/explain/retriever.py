"""CTI 知识检索器：特征规范化 + BM25 词法检索（后续可替换为 embedding）。

设计要点：
- 流量文本是"字段+数值"模板，直接 BM25 会被随机端口/字节数淹没（实测 top-1 仅 9%）；
  因此先把文本规范化成行为词元（端口、包大小、持续时间、包速率、标志位），再做 BM25。
- 行为阈值取自训练集统计（见 commit 说明），不在测试集上调参。
- 知识条目只描述技术行为，不绑定分类标签；检索不参与闭集判类。
"""

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from src.explain.knowledge_base import get_kb_entries

TOKEN_RE = re.compile(r"[a-z]+_[a-z0-9]+|[a-z][a-z\-]*|\d+")
NUM = r"-?\d+(?:\.\d+)?(?:e[+-]\d+)?"
CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    tokens = TOKEN_RE.findall(text)
    for seg in CJK_RE.findall(text):
        if len(seg) > 1:
            tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    return tokens


def _num(text: str, key: str) -> Optional[float]:
    m = re.search(key + NUM, text or "", re.I)
    return float(re.search(NUM, m.group(0)).group(0)) if m else None


def extract_behavior_tokens(text: str) -> List[str]:
    """把一条流量文本规范化为行为词元（供检索使用）。"""
    toks: List[str] = []
    port = _num(text, "目标端口")
    if port is not None:
        toks.append("port_%d" % int(port))

    size = _num(text, "平均包大小")
    if size is not None:
        toks.append("pkt_small" if size < 100 else ("pkt_mid" if size < 300 else "pkt_large"))

    dur = _num(text, "流持续")
    if dur is not None:
        if dur < 1e4:
            toks.append("dur_very_short")
        elif dur < 1e7:
            toks.append("dur_short")
        elif dur < 3e7:
            toks.append("dur_medium")
        else:
            toks.append("dur_long")

    pps = _num(text, "包速")
    if pps is not None:
        toks.append("pps_huge" if pps >= 1e3 else ("pps_high" if pps >= 10 else "pps_low"))

    if _num(text, "ACK标志") == 1:
        toks.append("ack_set")
    if _num(text, "SYN标志") == 1:
        toks.append("syn_set")
    return toks


def _doc_text(entry: Dict[str, object]) -> str:
    return " ".join([
        str(entry.get("name", "")),
        str(entry.get("description", "")),
        str(entry.get("detection", "")),
        " ".join(entry.get("keywords", [])),
    ])


class BM25Retriever:
    def __init__(self, entries: Optional[List[Dict[str, object]]] = None, k1: float = 1.5, b: float = 0.75):
        entries = entries or get_kb_entries()
        self.ids = [e["id"] for e in entries]
        self.tfs = [Counter(tokenize(_doc_text(e))) for e in entries]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = sum(self.lens) / max(1, len(self.lens))
        self.N = len(self.ids)
        self.k1, self.b = k1, b
        df = Counter()
        for tf in self.tfs:
            df.update(set(tf.keys()))
        self.idf = {t: math.log((self.N - n + 0.5) / (n + 0.5) + 1.0) for t, n in df.items()}

    def score_tokens(self, query_tokens: List[str]) -> List[Tuple[str, float]]:
        q = Counter(query_tokens)
        scores = []
        for mid, tf, dl in zip(self.ids, self.tfs, self.lens):
            s = 0.0
            for t, qn in q.items():
                f = tf.get(t, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += self.idf.get(t, 0.0) * (f * (self.k1 + 1) / denom) * qn
            scores.append((mid, s))
        return sorted(scores, key=lambda x: -x[1])

    def retrieve(self, text: str, k: int = 5) -> List[Tuple[str, float]]:
        """text 为流量文本特征；内部先规范化成行为词元。"""
        return self.score_tokens(extract_behavior_tokens(text))[:k]

    def top1(self, text: str) -> Optional[str]:
        ranked = self.retrieve(text, 1)
        return ranked[0][0] if ranked else None
