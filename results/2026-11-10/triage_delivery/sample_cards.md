# 检索-only triage 卡片样例

> 取自部署视角：3 张为真未知（`true_label=10`），2 张为已知类误判（OOD 误报）。
> 边界：S3 有编号只说明 BM25 返回了候选，**不等于映射正确**。

---

## ds200_s0_1658（true_label=10，pred_known=6，ood_score=1.0715）

| 槽 | 内容 |
|---|---|
| S1 行为 | 行为特征：port_593、pkt_small、dur_very_short、pps_huge。 |
| S2 候选 | T1046 Network Service Scanning（BM25 1.78，low） / T1110.001 Brute Force: Password Guessing（BM25 1.59，low） / T1071 Application Layer Protocol（BM25 0.57，low） |
| S3 编号 | T1046 |
| S4 下一步 | 按 T1046 的检测线索（单一源在短时间内对多个目标端口发起连接尝试，出现大量 SYN 与 RST，目标端口序列递增，单条连接持续时间极短且字节数）排查同类流量；并提取该来源近 1 小时全量流量复核端口/协议分布。 |
| S5 边界 | 仅 BM25 检索 top-3 相似度，检索相似不等于归因；未验证样本真实性，不覆盖 ATT&CK 映射正确性。 |
| 置信 | low（候选 3 条，abstain=False） |

---

## ds200_s0_1668（true_label=10，pred_known=9，ood_score=3.1292）

| 槽 | 内容 |
|---|---|
| S1 行为 | 行为特征：port_52941、pkt_small、dur_very_short、pps_huge、ack_set。 |
| S2 候选 | T1046 Network Service Scanning（BM25 1.78，low） / T1110.001 Brute Force: Password Guessing（BM25 1.59，low） / T1498.001 Network Denial of Service: Direct Network Flood（BM25 1.54，low） |
| S3 编号 | T1046 |
| S4 下一步 | 按 T1046 的检测线索（单一源在短时间内对多个目标端口发起连接尝试，出现大量 SYN 与 RST，目标端口序列递增，单条连接持续时间极短且字节数）排查同类流量；并提取该来源近 1 小时全量流量复核端口/协议分布。 |
| S5 边界 | 仅 BM25 检索 top-3 相似度，检索相似不等于归因；未验证样本真实性，不覆盖 ATT&CK 映射正确性。 |
| 置信 | low（候选 3 条，abstain=False） |

---

## ds200_s0_1685（true_label=10，pred_known=9，ood_score=0.9572）

| 槽 | 内容 |
|---|---|
| S1 行为 | 行为特征：port_8080、pkt_small、dur_short、pps_high。 |
| S2 候选 | T1071 Application Layer Protocol（BM25 4.52，high） / T1190 Exploit Public-Facing Application（BM25 1.27，low） / T1498.001 Network Denial of Service: Direct Network Flood（BM25 1.07，low） |
| S3 编号 | T1071 |
| S4 下一步 | 按 T1071 的检测线索（长连接长时间保持，出现固定间隔的心跳小包与周期性通信；用非常规端口承载 HTTP/DNS，常见目标端口包括 80、443）排查同类流量；并提取该来源近 1 小时全量流量复核端口/协议分布。 |
| S5 边界 | 仅 BM25 检索 top-3 相似度，检索相似不等于归因；未验证样本真实性，不覆盖 ATT&CK 映射正确性。 |
| 置信 | high（候选 3 条，abstain=False） |

---

## ds200_s0_5（true_label=0，pred_known=0，ood_score=0.6392）

| 槽 | 内容 |
|---|---|
| S1 行为 | 行为特征：port_80、pkt_small、dur_short、pps_low。 |
| S2 候选 | T1498.001 Network Denial of Service: Direct Network Flood（BM25 2.59，low） / T1498.002 Network Denial of Service: Service Exhaustion Flood（BM25 2.01，low） / T1071 Application Layer Protocol（BM25 1.72，low） |
| S3 编号 | T1498.001 |
| S4 下一步 | 按 T1498.001 的检测线索（短时间内目标收到极高的包速率与字节速率（可达数十万包每秒），出现大量并发连接，SYN 与 ACK 比例异常，源地址分散）排查同类流量；并提取该来源近 1 小时全量流量复核端口/协议分布。 |
| S5 边界 | 仅 BM25 检索 top-3 相似度，检索相似不等于归因；未验证样本真实性，不覆盖 ATT&CK 映射正确性。 |
| 置信 | low（候选 3 条，abstain=False） |

---

## ds200_s0_8（true_label=9，pred_known=9，ood_score=0.9445）

| 槽 | 内容 |
|---|---|
| S1 行为 | 行为特征：port_8080、pkt_small、dur_short、pps_low。 |
| S2 候选 | T1071 Application Layer Protocol（BM25 2.87，med） / T1498.001 Network Denial of Service: Direct Network Flood（BM25 1.83，low） / T1110.001 Brute Force: Password Guessing（BM25 1.28，low） |
| S3 编号 | T1071 |
| S4 下一步 | 按 T1071 的检测线索（长连接长时间保持，出现固定间隔的心跳小包与周期性通信；用非常规端口承载 HTTP/DNS，常见目标端口包括 80、443）排查同类流量；并提取该来源近 1 小时全量流量复核端口/协议分布。 |
| S5 边界 | 仅 BM25 检索 top-3 相似度，检索相似不等于归因；未验证样本真实性，不覆盖 ATT&CK 映射正确性。 |
| 置信 | med（候选 3 条，abstain=False） |

