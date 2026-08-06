# 数据模块导出文件
# 功能：将所有数据处理函数统一导出，方便外部调用
# 使用方式：from src.data import generate_mock_data, load_real_data, collate_fn

from .data_loader import generate_mock_data, load_real_data, extract_text_embeddings, collate_fn
