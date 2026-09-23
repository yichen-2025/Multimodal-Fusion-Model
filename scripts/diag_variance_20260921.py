# -*- coding: utf-8 -*-
"""定位 seed 方差来源：哪个指标摆、最差 seed 是否跨变体一致。"""
import csv, math
from collections import defaultdict

SRC = r"C:\Users\刘浥尘\Downloads\multiseed_20260921_230136.csv"
METRICS = ["auroc", "unknown_f1", "unknown_recall", "unknown_leak_rate",
           "known_accuracy", "macro_f1", "routing_seconds"]

rows = []
with open(SRC, encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if not r.get("variant"):
            continue
        for k in METRICS:
            try: r[k] = float(r[k])
            except: r[k] = None
        rows.append(r)

by_var = defaultdict(dict)
for r in rows:
    by_var[r["variant"]][int(r["seed"])] = r

ORDER = ["A3","A0*","A0*_add","A0*_attn","A0*_no_text","A0*_no_num"]
present = [v for v in ORDER if v in by_var]

def stats(vals):
    vals=[v for v in vals if v is not None]
    m=sum(vals)/len(vals)
    sd=math.sqrt(sum((v-m)**2 for v in vals)/(len(vals)-1)) if len(vals)>1 else 0
    return m,sd,(sd/m if m else 0)

print("=== 各变体指标方差（按 CV=std/mean 排序看谁最不稳）===")
for v in present:
    seeds=sorted(by_var[v])
    line=f"{v:12s} "
    for mk in ["macro_f1","unknown_f1","auroc","known_accuracy"]:
        m,sd,cv=stats([by_var[v][s][mk] for s in seeds])
        line+=f" {mk[:4]}={m:.3f}(CV{cv*100:.0f}%)"
    print(line)

print("\n=== 最差 seed 是否跨变体一致？（以 uF1 / known_acc / macro_f1 的最小值定位）===")
for mk in ["unknown_f1","known_accuracy","macro_f1"]:
    print(f"\n-- 指标 {mk} 的最小值所在 seed --")
    for v in present:
        seeds=sorted(by_var[v])
        vals={s:by_var[v][s][mk] for s in seeds}
        worst=min(vals,key=vals.get)
        best=max(vals,key=vals.get)
        print(f"  {v:12s}: 最差 seed={worst} ({vals[worst]:.3f})  最佳 seed={best} ({vals[best]:.3f})")

print("\n=== 跨度（最佳-最差）===")
for v in present:
    seeds=sorted(by_var[v])
    line=f"{v:12s} "
    for mk in ["unknown_f1","known_accuracy","macro_f1","auroc"]:
        vals=[by_var[v][s][mk] for s in seeds]
        line+=f" {mk[:4]}_span={max(vals)-min(vals):.3f}"
    print(line)
