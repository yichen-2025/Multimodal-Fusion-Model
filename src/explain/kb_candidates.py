"""CTI-RAG 候选知识库（扩库实验专用，A/B/C 三臂）。

本文件只服务于 ``scripts/eval_retrieval.py --kb`` 的候选集切换，不改动
``src/explain/knowledge_base.py`` 的 ``MITRE_KB`` 默认内容，也不进入
``run_ood_routing`` 的解释路径。

臂定义：
    A  原 7 条（逐字复制 MITRE_KB），用于复现基线 hit@1 = 1252/1474 = 0.8494
    B  A + 3 条父技术（T1498 / T1110 / T1059），description/detection 为子技术
       的泛化合并，keywords 为子技术 keywords 的并集（机械生成，不人工撰写）
    C  B + 12 条网络相关混淆技术。这些条目的 description / detection / keywords
       由本实验按 ATT&CK 语义撰写，属假设前提，结论中必须声明。

字段与 MITRE_KB 一致：id / name / description / detection / platforms / keywords。
"""

from typing import Dict, List

KB_A: List[Dict[str, object]] = [
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

# B 臂新增的父技术：keywords 为对应子技术 keywords 的并集（去重保序，机械生成）。
KB_B_PARENTS: List[Dict[str, object]] = [
    {
        "id": "T1498",
        "name": "Network Denial of Service",
        "description": (
            "攻击者耗尽目标网络带宽或服务连接资源，使合法用户无法访问服务；"
            "按作用对象可分为直接流量泛洪与服务资源耗尽两类。"
        ),
        "detection": (
            "目标端口出现异常高的包速率或并发连接数，或单条连接持续异常长而传输字节极少；"
            "两类形态共同表现为服务可用性下降、正常请求无法得到响应。"
        ),
        "platforms": "Network",
        "keywords": ["flood", "high rate", "泛洪", "高包速率", "大流量", "并发连接", "SYN",
                      "port_80", "pkt_large", "ack_set", "dur_short", "pps_low",
                      "slowloris", "slowhttptest", "慢速", "长时间保持连接", "半开连接", "连接耗尽",
                      "pkt_small", "pkt_mid", "dur_long", "dur_medium"],
    },
    {
        "id": "T1110",
        "name": "Brute Force",
        "description": "攻击者通过反复猜试凭据、口令或口令哈希来获取账号访问权限。",
        "detection": (
            "认证服务端口出现密集的失败登录尝试；按手法不同可表现为口令猜测、"
            "撞库或密码喷洒，单账号或跨账号的尝试频率异常升高。"
        ),
        "platforms": "Linux, Windows, macOS",
        "keywords": ["brute force", "暴力破解", "口令猜测", "登录失败", "ftp", "ssh", "patator",
                      "port_21", "port_22", "port_23", "port_3389",
                      "pkt_small", "pkt_mid", "pps_low", "pps_huge", "dur_long", "dur_very_short"],
    },
    {
        "id": "T1059",
        "name": "Command and Scripting Interpreter",
        "description": "攻击者滥用命令解释器或脚本运行时执行恶意逻辑。",
        "detection": "出现异常的命令行或脚本执行、脚本文件下载，以及解释器进程发起的异常外联。",
        "platforms": "Linux, Windows, macOS",
        "keywords": ["javascript", "脚本", "下载脚本", "script"],
    },
]

# C 臂新增的混淆技术：文本为本实验按 ATT&CK 语义撰写（假设前提）。
KB_C_EXTRA: List[Dict[str, object]] = [
    {
        "id": "T1595.002",
        "name": "Active Scanning: Vulnerability Scanning",
        "description": "攻击者扫描目标对外服务的漏洞特征，判断其可利用性。",
        "detection": (
            "单一源向 Web 或服务端口发送大量探测请求，请求路径异常或含已知漏洞特征串，"
            "响应码集中为 4xx/5xx；单条连接持续时间极短、字节数小、请求速率高。"
        ),
        "platforms": "Network",
        "keywords": ["vulnerability scan", "漏洞扫描", "探测请求", "nikto", "nessus",
                      "port_80", "port_443", "pkt_small", "dur_very_short", "pps_huge"],
    },
    {
        "id": "T1499.002",
        "name": "Endpoint Denial of Service: Service Exhaustion Flood",
        "description": "攻击者通过大量请求耗尽终端服务的连接与线程资源。",
        "detection": (
            "服务端连接数持续升高、响应时间变长，单连接字节少而持续时间长；"
            "CPU 与内存占用上升，但网络带宽未饱和。"
        ),
        "platforms": "Linux, Windows",
        "keywords": ["endpoint dos", "服务耗尽", "线程耗尽", "连接数上升",
                      "port_80", "pkt_small", "dur_long", "pps_low"],
    },
    {
        "id": "T1110.003",
        "name": "Brute Force: Password Spraying",
        "description": "使用少量常见口令对大量账号逐一尝试，规避账号锁定策略。",
        "detection": (
            "同一源对多个账号各尝试少量口令，失败次数分散在多个账号上，单账号失败次数低；"
            "认证端口出现规律性低频请求。"
        ),
        "platforms": "Linux, Windows",
        "keywords": ["password spraying", "密码喷洒", "少量口令", "多账号",
                      "port_22", "port_3389", "pkt_small", "pps_low", "dur_very_short"],
    },
    {
        "id": "T1110.004",
        "name": "Brute Force: Credential Stuffing",
        "description": "使用泄露的账号口令组合批量尝试登录。",
        "detection": (
            "大量不同账号在同一时段集中登录且失败率高，来源地址分散；请求间隔均匀、单连接字节小。"
        ),
        "platforms": "Linux, Windows",
        "keywords": ["credential stuffing", "撞库", "泄露凭据", "批量登录",
                      "port_443", "pkt_small", "pps_high", "dur_very_short"],
    },
    {
        "id": "T1071.004",
        "name": "Application Layer Protocol: DNS",
        "description": "使用 DNS 协议作为命令与控制通道。",
        "detection": (
            "出现高频或超长 DNS 查询、异常子域名与 TXT 记录请求，查询名熵值高；"
            "流量目的端口为 53。"
        ),
        "platforms": "Linux, Windows, macOS",
        "keywords": ["dns tunneling", "dns c2", "域名隧道", "超长域名",
                      "port_53", "pkt_small", "pps_high", "dur_short"],
    },
    {
        "id": "T1095",
        "name": "Non-Application Layer Protocol",
        "description": "使用非应用层协议（如 ICMP、RAW IP）承载通信，绕过应用层检测。",
        "detection": (
            "出现非常规协议流量或 ICMP 载荷异常增大，缺少标准应用层握手；目的端口不固定。"
        ),
        "platforms": "Network",
        "keywords": ["icmp", "raw socket", "非应用层", "自定义协议", "pkt_mid", "pps_low"],
    },
    {
        "id": "T1571",
        "name": "Non-Standard Port",
        "description": "在非默认端口上承载常见协议，以规避基于端口的过滤。",
        "detection": (
            "已知协议特征出现在非常规端口（如非 443 端口承载 TLS、8080 承载 SSH），"
            "端口与协议指纹不一致。"
        ),
        "platforms": "Linux, Windows",
        "keywords": ["non-standard port", "非常规端口", "端口复用",
                      "port_8080", "port_4444", "pkt_mid", "pps_low"],
    },
    {
        "id": "T1105",
        "name": "Ingress Tool Transfer",
        "description": "从外部下载工具或载荷到受害主机。",
        "detection": (
            "单次连接下行字节数显著大于上行，持续中等时长，目的为公网地址；"
            "随后出现新进程或文件写入。"
        ),
        "platforms": "Linux, Windows, macOS",
        "keywords": ["download", "下载载荷", "工具投放",
                      "port_80", "port_443", "pkt_large", "dur_medium", "pps_low"],
    },
    {
        "id": "T1048",
        "name": "Exfiltration Over Alternative Protocol",
        "description": "通过不同于命令与控制通道的协议外传数据。",
        "detection": (
            "上行字节数异常高于下行，目的地址陌生且固定，传输呈长时低速特征；"
            "可能使用 DNS、ICMP 等协议承载。"
        ),
        "platforms": "Linux, Windows, macOS",
        "keywords": ["exfiltration", "数据外传", "上行流量", "渗出",
                      "pkt_mid", "pps_low", "dur_long"],
    },
    {
        "id": "T1210",
        "name": "Exploitation of Remote Services",
        "description": "利用远程服务的漏洞获取访问权限并横向移动。",
        "detection": (
            "对 SMB、RDP 或数据库端口发起异常请求并伴随漏洞特征，成功后出现二次连接；"
            "连接字节中等、时长中等。"
        ),
        "platforms": "Linux, Windows",
        "keywords": ["smb exploit", "远程服务利用", "横向移动",
                      "port_445", "port_3389", "pkt_mid", "dur_medium"],
    },
    {
        "id": "T1133",
        "name": "External Remote Services",
        "description": "通过 VPN、远程桌面等对外远程服务获得初始访问。",
        "detection": (
            "外部地址对 VPN 或 RDP 端口建立长时间会话，登录时间异常（如非工作时段），"
            "会话内流量平稳。"
        ),
        "platforms": "Linux, Windows",
        "keywords": ["vpn", "远程桌面", "外部远程服务",
                      "port_3389", "port_443", "pkt_mid", "dur_long", "pps_low"],
    },
    {
        "id": "T1203",
        "name": "Exploitation for Client Execution",
        "description": "利用客户端软件漏洞触发代码执行。",
        "detection": (
            "客户端进程访问恶意站点后发起异常外联或下载，出现畸形文档或脚本响应；"
            "单次会话字节中等、时长较短。"
        ),
        "platforms": "Linux, Windows, macOS",
        "keywords": ["client exploit", "客户端利用", "浏览器漏洞", "恶意文档",
                      "port_80", "pkt_mid", "dur_short"],
    },
]

KB_B: List[Dict[str, object]] = KB_A + KB_B_PARENTS
KB_C: List[Dict[str, object]] = KB_B + KB_C_EXTRA

ARMS = {"a": KB_A, "b": KB_B, "c": KB_C}
ARM_LABELS = {
    "a": "A:原7条",
    "b": "B:7条+3父技术",
    "c": "C:10条+12混淆项",
}


def get_arm(name: str) -> List[Dict[str, object]]:
    key = (name or "").strip().lower()
    if key not in ARMS:
        raise ValueError("未知臂: %r（可选 a/b/c）" % name)
    return [dict(e) for e in ARMS[key]]
