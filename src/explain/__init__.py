"""Phase 2 B1: LLM 语义解释层。"""

from .attack_mapping import (
    LABEL_NAMES,
    ATTACK_TO_MITRE,
    get_label_name,
    get_mitre_for_label,
    get_all_mitre_techniques,
)
from .generate_explanation import (
    run_explanation_generation,
    generate_explanation,
    build_user_prompt,
)

__all__ = [
    "LABEL_NAMES",
    "ATTACK_TO_MITRE",
    "get_label_name",
    "get_mitre_for_label",
    "get_all_mitre_techniques",
    "run_explanation_generation",
    "generate_explanation",
    "build_user_prompt",
]
