# E8.4 去文本消融 —— **本目录的 Δ 不可采信**

**结论：这里的 +20.4pp 不得用于论证「文本有贡献」或「文本可省」。**

## 为什么不可采信

- `A0*` / `A0*_no_text` 检查点来自 **2026-09/10 旧批次**。
- 纯数值臂的闭集准确率仅 **0.54–0.71**，显著低于同一 split 上 E0 轻量 MLP 的 **0.9855**，
  疑为旧检查点/欠训练，属**跨批次混淆**。
- 因此 Δ = +20.4pp（sd 7.4，CI95 [19.8, 21.1]）只说明「两个旧检查点不可比」，
  不说明文本贡献。CI 再窄也不能救一个跨批次比较。

## 可用的替代证据（同估计器对照）

| 估计器 | `acc_features` | `acc_features_plus_text` | Δ |
|---|---|---|---|
| logreg | 0.9674 | 0.9873 | +2.0pp |
| HGB | 0.9958 | 0.9897 | −0.6pp |

产物：`results/2026-11-19/e8_audit_hgb/metrics.json`。

## 机器可读状态

`metrics.json` 内已标注：

```json
"trustworthy": false,
"usable_for_claim": false,
"verdict": "待检验",
"superseded_by": "results/2026-11-19/e8_audit_hgb/metrics.json（同估计器对照，Δ≈0）"
```

**任何汇总脚本/LLM 应先读这三个字段；为 false 时不得进入摘要或正文数字。**
