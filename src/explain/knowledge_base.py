"""MITRE ATT&CK 技术知识条目（CTI-RAG 的最小知识库）。

只收录 dataset_200 标签空间实际用到的技术，字段含义：
    id          技术编号
    name        技术名称
    description 技术行为描述
    detection   可观测的检测线索（流量/主机侧），用于检索与解释引用
    platforms   适用平台
    keywords    检索增强词（口语/缩写/中文表述）

注意：本文件只描述"技术行为"，不绑定具体类别标签，避免把分类答案写进知识库。
keywords 中的行为词元（port_/pkt_/dur_/pps_/ack_/syn_）按"检测语义自洽"编写，
选型时在 train/val 上做一致性核对，test 仅用于报告（见 scripts/eval_retrieval.py）。
"""

from typing import Dict, List, Optional

MITRE_KB: List[Dict[str, object]] = [
    {
        "id": "T1498.001",
        "name": "Network Denial of Service: Direct Network Flood",
        "description": (
            "攻击者以大量并发流量直接淹没目标带宽或网络设备，使合法请求无法得到响应。"
        ),
        "detection": (
            "短时间内目标收到极高的包速率与字节速率（可达数十万包每秒），出现大量并发连接，"
            "SYN 与 ACK 比例异常，源地址分散；受害端带宽饱和、丢包率骤升。"
        ),
        "platforms": "Network",
        "keywords": ["flood", "high rate", "泛洪", "高包速率", "大流量", "并发连接", "SYN",
                      "port_80", "pkt_large", "ack_set", "dur_short", "pps_low"],
    },
    {
        "id": "T1498.002",
        "name": "Network Denial of Service: Service Exhaustion Flood",
        "description": (
            "攻击者通过长期占用服务连接与线程资源（慢速发送请求、保持半开连接）耗尽服务能力。"
        ),
        "detection": (
            "单条连接持续时间异常长（分钟级甚至小时级），每条连接传输字节很少，"
            "存在大量半开连接或慢速 HTTP 请求头，服务连接数与响应时间同步上升。"
        ),
        "platforms": "Network",
        "keywords": ["slowloris", "slowhttptest", "慢速", "长时间保持连接", "半开连接", "连接耗尽",
                      "port_80", "pkt_small", "pkt_mid", "dur_long", "dur_medium", "pps_low"],
    },
    {
        "id": "T1046",
        "name": "Network Service Scanning",
        "description": "枚举目标主机开放的端口与服务，为后续攻击做侦察准备。",
        "detection": (
            "单一源在短时间内对多个目标端口发起连接尝试，出现大量 SYN 与 RST，"
            "目标端口序列递增，单条连接持续时间极短且字节数很少。"
        ),
        "platforms": "Network",
        "keywords": ["portscan", "端口扫描", "多端口", "SYN 探测", "RST", "scan",
                      "dur_very_short", "pps_huge", "pkt_small"],
    },
    {
        "id": "T1110.001",
        "name": "Brute Force: Password Guessing",
        "description": "攻击者反复尝试常见口令登录认证服务（FTP/SSH/Telnet/Web 登录）以获取凭据。",
        "detection": (
            "同一源对认证服务端口（TCP 21 FTP、22 SSH、23 Telnet、3389 RDP、80/443 Web 登录）"
            "发起大量短连接与重复登录失败；连接持续时间短、双向字节数小且对称，登录尝试频率高。"
        ),
        "platforms": "Linux, Windows, macOS",
        "keywords": ["brute force", "暴力破解", "口令猜测", "登录失败", "ftp", "ssh", "patator",
                      "port_21", "port_22", "port_23", "port_3389",
                      "pkt_small", "pkt_mid", "pps_low", "pps_huge", "dur_long", "dur_very_short"],
    },
    {
        "id": "T1071",
        "name": "Application Layer Protocol",
        "description": "恶意程序使用常见应用层协议（HTTP/DNS/IRC 等）作为命令与控制通道，混入正常流量。",
        "detection": (
            "长连接长时间保持，出现固定间隔的心跳小包与周期性通信；用非常规端口承载 HTTP/DNS，"
            "常见目标端口包括 80、443、8080、6667。"
        ),
        "platforms": "Linux, Windows, macOS",
        "keywords": ["C2", "心跳", "周期性通信", "长连接", "bot", "IRC", "application layer",
                      "port_8080", "port_6667", "port_443", "pkt_small", "pps_high", "dur_short"],
    },
    {
        "id": "T1059.007",
        "name": "Command and Scripting Interpreter: JavaScript",
        "description": "通过 JavaScript 等脚本解释器执行恶意逻辑。",
        "detection": "HTTP 响应中出现脚本载荷，浏览器或文档进程发起异常外联，下载可疑脚本文件。",
        "platforms": "Linux, Windows, macOS",
        "keywords": ["javascript", "脚本", "下载脚本", "script"],
    },
    {
        "id": "T1190",
        "name": "Exploit Public-Facing Application",
        "description": "利用对外服务的公开漏洞获取访问权限或执行代码。",
        "detection": "对 Web 服务发送畸形或超长请求，单次请求即触发异常响应或崩溃，出现已知漏洞利用特征串。",
        "platforms": "Linux, Windows, macOS",
        "keywords": ["漏洞利用", "畸形请求", "exploit", "web 服务", "sql injection", "xss",
                      "port_80", "port_443", "port_8080"],
    },
]


def get_kb_entries() -> List[Dict[str, object]]:
    return MITRE_KB


def get_entry(mitre_id: str) -> Optional[Dict[str, object]]:
    for e in MITRE_KB:
        if e["id"] == mitre_id:
            return e
    return None
