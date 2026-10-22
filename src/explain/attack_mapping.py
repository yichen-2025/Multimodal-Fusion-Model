"""攻击类别 -> MITRE ATT&CK 技术映射表（MVP 初稿）。

后续可替换为更专业的版本（如参考 CICIDS2017 相关论文的标准映射）。
映射依据：CICIDS2017 数据集攻击类型 + MITRE ATT&CK 常识。
"""

from typing import Dict, Optional, List

# 标签名称（与 run_ood_routing.py 中 LABEL_NAMES 保持一致）
LABEL_NAMES: Dict[int, str] = {
    0: "BENIGN",
    1: "DoS Hulk",
    2: "DoS GoldenEye",
    3: "DoS slowloris",
    4: "DoS Slowhttptest",
    5: "DDoS",
    6: "PortScan",
    7: "FTP-Patator",
    8: "SSH-Patator",
    9: "Bot",
    10: "Web Attack - Brute Force",
    11: "Web Attack - XSS",
    12: "Web Attack - Sql Injection",
    13: "Infiltration",
    14: "Heartbleed",
    15: "unknown",
}

# 攻击类别 -> MITRE ATT&CK 技术
ATTACK_TO_MITRE: Dict[int, Optional[Dict[str, str]]] = {
    0: None,  # BENIGN 正常流量，无攻击技术
    1: {"id": "T1498.001", "name": "Network Denial of Service: Direct Network Flood"},
    2: {"id": "T1498.001", "name": "Network Denial of Service: Direct Network Flood"},
    3: {"id": "T1498.002", "name": "Network Denial of Service: Service Exhaustion Flood"},
    4: {"id": "T1498.002", "name": "Network Denial of Service: Service Exhaustion Flood"},
    5: {"id": "T1498.001", "name": "Network Denial of Service: Direct Network Flood"},
    6: {"id": "T1046", "name": "Network Service Scanning"},
    7: {"id": "T1110.001", "name": "Brute Force: Password Guessing"},
    8: {"id": "T1110.001", "name": "Brute Force: Password Guessing"},
    9: {"id": "T1071", "name": "Application Layer Protocol"},
    10: {"id": "T1110.001", "name": "Brute Force: Password Guessing"},
    11: {"id": "T1059.007", "name": "Command and Scripting Interpreter: JavaScript"},
    12: {"id": "T1190", "name": "Exploit Public-Facing Application"},
    13: {"id": "T1190", "name": "Exploit Public-Facing Application"},
    14: {"id": "T1190", "name": "Exploit Public-Facing Application"},
    15: None,  # unknown，无确定映射
}


def get_mitre_for_label(label: int) -> Optional[Dict[str, str]]:
    """根据标签获取 MITRE ATT&CK 技术映射。"""
    return ATTACK_TO_MITRE.get(label)


def get_label_name(label: int) -> str:
    """根据标签获取攻击类别名称。"""
    return LABEL_NAMES.get(label, f"class_{label}")


def get_all_mitre_techniques() -> List[Dict[str, str]]:
    """返回所有去重后的 MITRE 技术列表（用于 prompt 候选）。"""
    seen = set()
    techniques = []
    for mapping in ATTACK_TO_MITRE.values():
        if mapping is not None and mapping["id"] not in seen:
            seen.add(mapping["id"])
            techniques.append(mapping)
    return techniques
