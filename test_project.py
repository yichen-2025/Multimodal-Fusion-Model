import pytest
import pandas as pd
import numpy as np
import os
import shutil
import sys
import torch
import json

TEST_DATASET_ID = 999
TEST_SPLIT_ID = 999
TEST_MODEL_ID = 999

TEST_PROCESSED_DIR = "./processed_dataset"
TEST_SPLIT_DIR = "./split_data"
TEST_SAVED_MODELS_DIR = "./saved_models"
TEST_LOGS_DIR = "./logs"


@pytest.fixture(scope="module")
def test_clean_data():
    """生成测试用的模拟原始数据"""
    test_dir = "./test_temp_data"
    os.makedirs(test_dir, exist_ok=True)
    
    np.random.seed(42)
    n_samples = 2000
    n_features = 79
    
    data = {
        "Destination Port": np.random.randint(1, 65535, n_samples),
        "Bwd Packet Length Mean": np.random.randn(n_samples) * 100 + 500,
        "Avg Bwd Segment Size": np.random.randn(n_samples) * 50 + 200,
        "Bwd Packet Length Max": np.random.randint(100, 2000, n_samples),
        "Bwd Packet Length Std": np.random.randn(n_samples) * 50 + 30,
        "URG Flag Count": np.random.randint(0, 5, n_samples),
        "Packet Length Mean": np.random.randn(n_samples) * 100 + 400,
        "Average Packet Size": np.random.randn(n_samples) * 80 + 350,
        "Packet Length Std": np.random.randn(n_samples) * 60 + 40,
        "Label": np.random.choice(['BENIGN', 'DDoS'], n_samples, p=[0.5, 0.5]),
    }
    
    for i in range(n_features - 10):
        data[f"Feature_{i}"] = np.random.randn(n_samples)
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(test_dir, "test_data.csv"), index=False)
    
    yield test_dir
    
    shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def cleaned_data(test_clean_data):
    """执行数据清洗"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from data_cleaning import clean_data as dc_clean_data
    
    input_dir = test_clean_data
    output_dir = TEST_PROCESSED_DIR
    
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    data_file = os.path.join(input_dir, csv_files[0])
    df = pd.read_csv(data_file)
    
    df_clean = dc_clean_data(df)
    
    df_clean['Label'] = df_clean['Label'].map({'BENIGN': 0, 'DDoS': 1})
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "processed_dataset.csv")
    df_clean.to_csv(output_path, index=False)
    
    yield output_path
    
    if os.path.exists(output_path):
        os.remove(output_path)


@pytest.fixture(scope="module")
def extracted_subset(cleaned_data):
    """提取数据集子集"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from extract_subset import extract_subset
    
    success, dataset_id = extract_subset(
        num_samples=1000,
        dataset_id=TEST_DATASET_ID,
        random_state=42
    )
    
    assert success, "子集提取失败"
    
    subset_path = os.path.join(TEST_PROCESSED_DIR, f"dataset_{TEST_DATASET_ID}.csv")
    assert os.path.exists(subset_path), "子集文件不存在"
    
    yield subset_path, dataset_id
    
    files_to_remove = [
        os.path.join(TEST_PROCESSED_DIR, f"dataset_{TEST_DATASET_ID}.csv"),
        os.path.join(TEST_PROCESSED_DIR, f"subset_{TEST_DATASET_ID}_scaled_features.npy"),
        os.path.join(TEST_PROCESSED_DIR, f"subset_{TEST_DATASET_ID}_labels.npy"),
        os.path.join(TEST_PROCESSED_DIR, f"subset_{TEST_DATASET_ID}_scaler.npy"),
    ]
    for f in files_to_remove:
        if os.path.exists(f):
            os.remove(f)


@pytest.fixture(scope="module")
def split_data(extracted_subset):
    """划分训练集和测试集"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from split_modality import split_modality
    
    dataset_id = TEST_DATASET_ID
    split_modality(
        dataset_id=dataset_id,
        split_id=TEST_SPLIT_ID,
        test_size=0.2,
        random_state=42
    )
    
    split_dir = os.path.join(TEST_SPLIT_DIR, f"dataset_{dataset_id}", f"split_{TEST_SPLIT_ID}")
    assert os.path.exists(split_dir), "划分目录不存在"
    
    yield split_dir, dataset_id, TEST_SPLIT_ID
    
    split_path = os.path.join(TEST_SPLIT_DIR, f"dataset_{dataset_id}")
    if os.path.exists(split_path):
        shutil.rmtree(split_path, ignore_errors=True)


@pytest.fixture(scope="module")
def trained_model(split_data):
    """训练模型"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from train import train_model
    
    _, dataset_id, split_id = split_data
    
    model = train_model(
        model_path="./models/qwen2.5-1.5b",
        model_id=TEST_MODEL_ID,
        dataset_id=dataset_id,
        split_id=split_id,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        num_train_epochs=1
    )
    
    model_path = os.path.join(TEST_SAVED_MODELS_DIR, f"model_{TEST_MODEL_ID}")
    assert os.path.exists(os.path.join(model_path, "pytorch_model.bin")), "模型文件不存在"
    
    yield model, model_path
    
    if os.path.exists(model_path):
        shutil.rmtree(model_path, ignore_errors=True)


@pytest.fixture(scope="module", autouse=True)
def cleanup_logs():
    """自动清理测试生成的日志文件"""
    yield
    
    log_files_to_remove = []
    
    if os.path.exists(os.path.join(TEST_LOGS_DIR, "subset")):
        for f in os.listdir(os.path.join(TEST_LOGS_DIR, "subset")):
            if f.startswith("log_") and f.endswith(".json"):
                try:
                    with open(os.path.join(TEST_LOGS_DIR, "subset", f), 'r', encoding='utf-8') as fp:
                        log_data = json.load(fp)
                        if log_data.get('dataset_id') == TEST_DATASET_ID:
                            log_files_to_remove.append(os.path.join(TEST_LOGS_DIR, "subset", f))
                except:
                    pass
    
    if os.path.exists(os.path.join(TEST_LOGS_DIR, "split")):
        for f in os.listdir(os.path.join(TEST_LOGS_DIR, "split")):
            if f.startswith("log_") and f.endswith(".json"):
                try:
                    with open(os.path.join(TEST_LOGS_DIR, "split", f), 'r', encoding='utf-8') as fp:
                        log_data = json.load(fp)
                        if log_data.get('dataset_id') == TEST_DATASET_ID and log_data.get('split_id') == TEST_SPLIT_ID:
                            log_files_to_remove.append(os.path.join(TEST_LOGS_DIR, "split", f))
                except:
                    pass
    
    if os.path.exists(os.path.join(TEST_LOGS_DIR, "training")):
        for f in os.listdir(os.path.join(TEST_LOGS_DIR, "training")):
            if f.startswith("log_") and f.endswith(".json"):
                try:
                    with open(os.path.join(TEST_LOGS_DIR, "training", f), 'r', encoding='utf-8') as fp:
                        log_data = json.load(fp)
                        if log_data.get('model_id') == TEST_MODEL_ID:
                            log_files_to_remove.append(os.path.join(TEST_LOGS_DIR, "training", f))
                except:
                    pass
    
    for f in log_files_to_remove:
        if os.path.exists(f):
            os.remove(f)
    
    for log_type in ['subset', 'split', 'training']:
        index_file = os.path.join(TEST_LOGS_DIR, log_type, "index.csv")
        if os.path.exists(index_file):
            try:
                df = pd.read_csv(index_file)
                if log_type == 'subset':
                    df = df[df['dataset_id'] != TEST_DATASET_ID]
                elif log_type == 'split':
                    df = df[(df['dataset_id'] != TEST_DATASET_ID) | (df['split_id'] != TEST_SPLIT_ID)]
                elif log_type == 'training':
                    df = df[df['model_id'] != TEST_MODEL_ID]
                df.to_csv(index_file, index=False)
            except:
                pass


class TestDataCleaning:
    """测试数据清洗功能"""
    
    def test_clean_data_exists(self, test_clean_data):
        csv_files = [f for f in os.listdir(test_clean_data) if f.endswith('.csv')]
        assert len(csv_files) > 0, "原始数据文件不存在"
    
    def test_clean_data_output(self, cleaned_data):
        assert os.path.exists(cleaned_data), "清洗后数据文件不存在"
        
        df = pd.read_csv(cleaned_data)
        assert df.shape[0] > 0, "清洗后数据为空"
        assert 'Label' in df.columns, "缺少Label列"
        assert set(df['Label'].unique()) <= {0, 1}, "标签编码不正确"
        
        for col in df.columns:
            if df[col].dtype in ['float64', 'float32']:
                assert not df[col].isna().any(), f"{col}列存在缺失值"
                assert not np.isinf(df[col]).any(), f"{col}列存在Inf值"


class TestExtractSubset:
    """测试子集提取功能"""
    
    def test_subset_exists(self, extracted_subset):
        subset_path, _ = extracted_subset
        assert os.path.exists(subset_path), "子集文件不存在"
    
    def test_subset_balance(self, extracted_subset):
        subset_path, _ = extracted_subset
        df = pd.read_csv(subset_path)
        
        label_counts = df['Label'].value_counts()
        assert len(label_counts) == 2, "标签类别数量不正确"
        assert abs(label_counts[0] - label_counts[1]) <= 2, "正负样本不平衡"
    
    def test_subset_size(self, extracted_subset):
        subset_path, _ = extracted_subset
        df = pd.read_csv(subset_path)
        assert df.shape[0] == 1000, f"子集大小不正确，期望1000，实际{df.shape[0]}"
    
    def test_subset_log_generated(self, extracted_subset):
        """测试子集提取是否生成日志文件"""
        subset_log_dir = os.path.join(TEST_LOGS_DIR, "subset")
        assert os.path.exists(subset_log_dir), "子集日志目录不存在"
        
        log_files = [f for f in os.listdir(subset_log_dir) if f.startswith("log_") and f.endswith(".json")]
        assert len(log_files) > 0, "未生成子集日志文件"
        
        test_log_path = None
        for f in log_files:
            log_path = os.path.join(subset_log_dir, f)
            with open(log_path, 'r', encoding='utf-8') as fp:
                log_data = json.load(fp)
                if log_data.get('dataset_id') == TEST_DATASET_ID:
                    test_log_path = log_path
                    break
        
        assert test_log_path is not None, "未找到测试用的子集日志文件"
        
        with open(test_log_path, 'r', encoding='utf-8') as fp:
            log_data = json.load(fp)
        
        assert 'log_id' in log_data, "日志缺少log_id字段"
        assert 'timestamp' in log_data, "日志缺少timestamp字段"
        assert 'dataset_id' in log_data, "日志缺少dataset_id字段"
        assert 'total_samples' in log_data, "日志缺少total_samples字段"
        assert 'positive_samples' in log_data, "日志缺少positive_samples字段"
        assert 'negative_samples' in log_data, "日志缺少negative_samples字段"
        
        assert log_data['total_samples'] == 1000, f"日志中样本数不正确，期望1000，实际{log_data['total_samples']}"
        assert log_data['positive_samples'] == 500, f"日志中正样本数不正确，期望500，实际{log_data['positive_samples']}"
        assert log_data['negative_samples'] == 500, f"日志中负样本数不正确，期望500，实际{log_data['negative_samples']}"


class TestSplitModality:
    """测试模态分离功能"""
    
    def test_split_directory(self, split_data):
        split_dir, _, _ = split_data
        assert os.path.isdir(split_dir), "划分目录不是目录"
    
    def test_train_files(self, split_data):
        split_dir, _, _ = split_data
        assert os.path.exists(os.path.join(split_dir, "train.npz")), "train.npz不存在"
        assert os.path.exists(os.path.join(split_dir, "train_data.csv")), "train_data.csv不存在"
    
    def test_test_files(self, split_data):
        split_dir, _, _ = split_data
        assert os.path.exists(os.path.join(split_dir, "test.npz")), "test.npz不存在"
        assert os.path.exists(os.path.join(split_dir, "test_data.csv")), "test_data.csv不存在"
    
    def test_train_test_ratio(self, split_data):
        split_dir, _, _ = split_data
        
        train_data = np.load(os.path.join(split_dir, "train.npz"), allow_pickle=True)
        test_data = np.load(os.path.join(split_dir, "test.npz"), allow_pickle=True)
        
        train_size = len(train_data['labels'])
        test_size = len(test_data['labels'])
        
        expected_train_size = 800
        expected_test_size = 200
        
        assert train_size == expected_train_size, f"训练集大小不正确，期望{expected_train_size}，实际{train_size}"
        assert test_size == expected_test_size, f"测试集大小不正确，期望{expected_test_size}，实际{test_size}"
    
    def test_split_log_generated(self, split_data):
        """测试数据集划分是否生成日志文件"""
        _, dataset_id, split_id = split_data
        
        split_log_dir = os.path.join(TEST_LOGS_DIR, "split")
        assert os.path.exists(split_log_dir), "划分日志目录不存在"
        
        log_files = [f for f in os.listdir(split_log_dir) if f.startswith("log_") and f.endswith(".json")]
        assert len(log_files) > 0, "未生成划分日志文件"
        
        test_log_path = None
        for f in log_files:
            log_path = os.path.join(split_log_dir, f)
            with open(log_path, 'r', encoding='utf-8') as fp:
                log_data = json.load(fp)
                if log_data.get('dataset_id') == dataset_id and log_data.get('split_id') == split_id:
                    test_log_path = log_path
                    break
        
        assert test_log_path is not None, "未找到测试用的划分日志文件"
        
        with open(test_log_path, 'r', encoding='utf-8') as fp:
            log_data = json.load(fp)
        
        assert 'log_id' in log_data, "日志缺少log_id字段"
        assert 'timestamp' in log_data, "日志缺少timestamp字段"
        assert 'dataset_id' in log_data, "日志缺少dataset_id字段"
        assert 'split_id' in log_data, "日志缺少split_id字段"
        assert 'train_samples' in log_data, "日志缺少train_samples字段"
        assert 'test_samples' in log_data, "日志缺少test_samples字段"
        
        assert log_data['train_samples'] == 800, f"日志中训练集样本数不正确，期望800，实际{log_data['train_samples']}"
        assert log_data['test_samples'] == 200, f"日志中测试集样本数不正确，期望200，实际{log_data['test_samples']}"


class TestEncoders:
    """测试编码器功能"""
    
    def test_numeric_encoder(self, split_data):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from src.model_architectures.numeric_encoder import NumericEncoder
        
        encoder = NumericEncoder(input_dim=9)
        
        split_dir, _, _ = split_data
        train_data = np.load(os.path.join(split_dir, "train.npz"), allow_pickle=True)
        stat_features = train_data['scaled_features'][:10]
        
        stat_tensor = torch.tensor(stat_features, dtype=torch.float32)
        output = encoder(stat_tensor)
        
        assert output.shape == (10, 128), f"NumericEncoder输出形状不正确，期望(10, 128)，实际{output.shape}"
    
    def test_bert_encoder(self, split_data):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from src.model_architectures.bert_encoder import BertEncoder
        
        encoder = BertEncoder(local_model_path="./models/bert")
        encoder.eval()
        
        split_dir, _, _ = split_data
        df_train = pd.read_csv(os.path.join(split_dir, "train_data.csv"))
        texts = df_train['text_description'].tolist()[:5]
        
        with torch.no_grad():
            embeddings = encoder(texts)
        
        assert embeddings.shape == (5, 768), f"BERT编码器输出形状不正确，期望(5, 768)，实际{embeddings.shape}"


class TestModelTraining:
    """测试模型训练功能"""
    
    def test_model_exists(self, trained_model):
        _, model_path = trained_model
        assert os.path.exists(model_path), "模型目录不存在"
        assert os.path.exists(os.path.join(model_path, "pytorch_model.bin")), "模型参数文件不存在"
        assert os.path.exists(os.path.join(model_path, "config.txt")), "配置文件不存在"
    
    def test_model_config(self, trained_model):
        _, model_path = trained_model
        
        config_path = os.path.join(model_path, "config.txt")
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        assert f"dataset_id: {TEST_DATASET_ID}" in config_content, "配置文件缺少dataset_id"
        assert f"split_id: {TEST_SPLIT_ID}" in config_content, "配置文件缺少split_id"
        assert f"model_id: {TEST_MODEL_ID}" in config_content, "配置文件缺少model_id"
    
    def test_training_log_generated(self, trained_model):
        """测试模型训练是否生成日志文件"""
        _, model_path = trained_model
        
        training_log_dir = os.path.join(TEST_LOGS_DIR, "training")
        assert os.path.exists(training_log_dir), "训练日志目录不存在"
        
        log_files = [f for f in os.listdir(training_log_dir) if f.startswith("log_") and f.endswith(".json")]
        assert len(log_files) > 0, "未生成训练日志文件"
        
        test_log_path = None
        for f in log_files:
            log_path = os.path.join(training_log_dir, f)
            with open(log_path, 'r', encoding='utf-8') as fp:
                log_data = json.load(fp)
                if log_data.get('model_id') == TEST_MODEL_ID:
                    test_log_path = log_path
                    break
        
        assert test_log_path is not None, "未找到测试用的训练日志文件"
        
        with open(test_log_path, 'r', encoding='utf-8') as fp:
            log_data = json.load(fp)
        
        assert 'log_id' in log_data, "日志缺少log_id字段"
        assert 'timestamp' in log_data, "日志缺少timestamp字段"
        assert 'model_id' in log_data, "日志缺少model_id字段"
        assert 'dataset_id' in log_data, "日志缺少dataset_id字段"
        assert 'split_id' in log_data, "日志缺少split_id字段"
        assert 'learning_rate' in log_data, "日志缺少learning_rate字段"
        assert 'epochs' in log_data, "日志缺少epochs字段"
        assert 'duration_seconds' in log_data, "日志缺少duration_seconds字段"
        
        assert log_data['learning_rate'] == 1e-4, f"日志中学习率不正确，期望1e-4，实际{log_data['learning_rate']}"
        assert log_data['epochs'] == 1, f"日志中训练轮数不正确，期望1，实际{log_data['epochs']}"
        assert log_data['duration_seconds'] > 0, "日志中训练时间不正确"


class TestModelTesting:
    """测试模型测试功能"""
    
    def test_model_load(self, trained_model, split_data):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from src.model_architectures.multi_modal_model import MultiModalFusionModel
        
        _, model_path = trained_model
        _, dataset_id, split_id = split_data
        
        model = MultiModalFusionModel.from_pretrained(
            llm_model_path="./models/qwen2.5-1.5b",
            save_dir=model_path
        )
        
        assert model is not None, "模型加载失败"
    
    def test_model_inference(self, trained_model, split_data):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from src.model_architectures.multi_modal_model import MultiModalFusionModel
        from src.data.data_loader import load_split_data
        
        _, model_path = trained_model
        _, dataset_id, split_id = split_data
        
        model = MultiModalFusionModel.from_pretrained(
            llm_model_path="./models/qwen2.5-1.5b",
            save_dir=model_path
        )
        model.eval()
        
        test_dataset = load_split_data(
            data_dir="split_data",
            data_type="test",
            dataset_id=dataset_id,
            split_id=split_id
        )
        
        assert len(test_dataset) > 0, "测试数据集为空"
        
        sample = test_dataset[0]
        stat_vector = torch.tensor(sample["stat"], dtype=torch.float32).unsqueeze(0)
        bert_tensor = torch.tensor(sample["bert"], dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(stat_vector, bert_tensor)
        
        assert "logits" in outputs, "模型输出缺少logits"
        assert outputs["logits"].shape == (1, 2), f"logits形状不正确，期望(1, 2)，实际{outputs['logits'].shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])