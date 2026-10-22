"""Phase 2 B1: LLM 语义解释生成 CLI 入口。

用法：
    python run.py python scripts/generate_explanations.py \
        --backbone_model_id 110 \
        --dataset_id 0 \
        --split_id 0 \
        --n_samples 100

说明：
    - A3 backbone 做分类预测
    - 基座 Qwen2.5-1.5b 生成自然语言解释 + MITRE ATT&CK 映射
"""

import sys
import os

# 确保项目根目录在 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.explain.generate_explanation import main

if __name__ == "__main__":
    main()
