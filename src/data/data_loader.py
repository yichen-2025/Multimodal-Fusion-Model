import torch
import numpy as np
import os
from datasets import Dataset
from ..model_architectures.bert_encoder import BertEncoder


def generate_mock_data(num_samples=100):
    """
    生成模拟数据（用于测试和调试）
    
    功能：创建用于测试多模态融合模型的模拟网络流量数据
    
    Args:
        num_samples (int): 生成的样本数量，默认为100
        
    Returns:
        datasets.Dataset: 包含模拟数据的Dataset对象，每个样本包含：
            - stat: 9维数值统计特征（随机正态分布）
            - bert: 768维BERT文本特征（随机正态分布）
            - label: 分类标签（0或1，随机生成）
            - text: 文本描述（固定测试文本）
    """
    data = []
    for _ in range(num_samples):
        # 生成9维数值统计特征（模拟流量统计数据）
        stat = np.random.randn(9).astype(np.float32)
        # 生成768维BERT特征（模拟文本嵌入）
        bert = np.random.randn(768).astype(np.float32)
        # 生成二分类标签（0=正常流量，1=恶意流量）
        label = np.random.randint(0, 2)
        # 生成固定的文本描述
        text = "This is a sample network traffic description for testing."
        data.append({
            "stat": stat,
            "bert": bert,
            "label": label,
            "text": text
        })
    # 转换为Hugging Face Dataset对象
    return Dataset.from_list(data)


def load_real_data(data_dir="processed_data"):
    """
    加载真实数据（用于实际训练和推理）
    
    功能：从磁盘加载预处理好的CICIDS2017网络流量数据
    
    Args:
        data_dir (str): 预处理数据目录，默认为"processed_data"
        
    Returns:
        datasets.Dataset or None: 包含真实数据的Dataset对象，加载失败返回None
    """
    try:
        # 加载数值统计特征（标准化后的9维特征）
        stat_features = np.load(f"{data_dir}/scaled_features.npy")
        # 加载BERT文本特征（768维嵌入）
        bert_embeddings = np.load(f"{data_dir}/text_embeddings.npy")
        # 加载标签（0=正常流量，1=恶意流量）
        labels = np.load(f"{data_dir}/labels.npy")
        
        # 尝试加载文本描述（可选）
        text_descriptions = None
        try:
            import pandas as pd
            df = pd.read_csv(f"{data_dir}/processed_data.csv")
            if "text_description" in df.columns:
                text_descriptions = df["text_description"].tolist()
        except Exception:
            pass

        # 构建数据列表
        data = []
        for i in range(len(stat_features)):
            item = {
                "stat": stat_features[i].astype(np.float32),    # 数值特征，转换为float32
                "bert": bert_embeddings[i].astype(np.float32),  # BERT特征，转换为float32
                "label": int(labels[i]),                        # 分类标签
            }
            # 如果有文本描述，添加到item中
            if text_descriptions is not None:
                item["text"] = text_descriptions[i]
            else:
                item["text"] = ""
            data.append(item)
            
        # 转换为Dataset对象
        dataset = Dataset.from_list(data)
        
        # 打印数据统计信息
        print(f"Loaded {len(dataset)} samples from {data_dir}")
        print(f"  - Stat features shape: {stat_features.shape}")
        print(f"  - BERT embeddings shape: {bert_embeddings.shape}")
        print(f"  - Labels: {np.unique(labels, return_counts=True)}")
        
        return dataset
        
    except FileNotFoundError as e:
        # 处理文件不存在的情况
        print(f"Error loading data: {e}")
        print(f"Please run preprocess_data.py first to generate processed data.")
        return None


def load_split_data(data_dir="split_data", data_type="train", dataset_id=0, split_id=0, openset=False, fewshot_k=None):
    """
    加载划分后的数据集（训练集或测试集）
    
    功能：从split_data目录加载按比例划分后的训练集或测试集数据
    支持通过dataset_id和split_id定位数据，优先加载npz格式（合并文件），兼容旧的npy格式
    
    Args:
        data_dir (str): 划分后数据目录，默认为"split_data"
        data_type (str): 数据类型，"train"表示训练集，"test"表示测试集
        dataset_id (int): 数据集ID，默认为0
        split_id (int): 划分ID，默认为0
        openset (bool): 是否加载开集划分（目录名为split_openset_{id}）
        fewshot_k (int): 少样本k值，指定后加载split_fewshot_{k}_{split_id}
        
    Returns:
        datasets.Dataset or None: 包含划分后数据的Dataset对象，加载失败返回None
    """
    try:
        if fewshot_k is not None:
            split_dir = os.path.join(data_dir, f"dataset_{dataset_id}",
                                      f"split_fewshot_{fewshot_k}_{split_id}")
        elif openset:
            split_dir = os.path.join(data_dir, f"dataset_{dataset_id}", f"split_openset_{split_id}")
        else:
            split_dir = os.path.join(data_dir, f"dataset_{dataset_id}", f"split_{split_id}")
        
        if os.path.exists(split_dir):
            npz_path = os.path.join(split_dir, f"{data_type}.npz")
            
            if os.path.exists(npz_path):
                print(f"Loading {data_type} data from npz file (dataset={dataset_id}, split={split_id})...")
                npz_data = np.load(npz_path, allow_pickle=True)
                stat_features = npz_data['scaled_features']
                bert_embeddings = npz_data['text_embeddings']
                labels = npz_data['labels']
            else:
                print(f"Error: No {data_type}.npz found in {split_dir}")
                return None
        else:
            npz_path = os.path.join(data_dir, f"{data_type}.npz")
            
            if os.path.exists(npz_path):
                print(f"Loading {data_type} data from npz file (old format)...")
                npz_data = np.load(npz_path, allow_pickle=True)
                stat_features = npz_data['scaled_features']
                bert_embeddings = npz_data['text_embeddings']
                labels = npz_data['labels']
            else:
                files = os.listdir(data_dir)
                prefix = None
                for f in files:
                    if f.startswith(data_type) and f.endswith("_scaled_features.npy"):
                        prefix = f.replace("_scaled_features.npy", "")
                        break
                
                if prefix is None:
                    print(f"Error: No {data_type} data found in {data_dir}")
                    print(f"Please run split_modality.py first to generate split data.")
                    return None
                
                stat_features = np.load(os.path.join(data_dir, f"{prefix}_scaled_features.npy"))
                bert_embeddings = np.load(os.path.join(data_dir, f"{prefix}_text_embeddings.npy"))
                labels = np.load(os.path.join(data_dir, f"{prefix}_labels.npy"))
        
        current_dir = split_dir if os.path.exists(split_dir) else data_dir
        text_descriptions = None
        try:
            import pandas as pd
            csv_path = os.path.join(current_dir, f"{data_type}_data.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if "text_description" in df.columns:
                    text_descriptions = df["text_description"].tolist()
        except Exception:
            pass

        data = []
        for i in range(len(stat_features)):
            item = {
                "stat": stat_features[i].astype(np.float32),
                "bert": bert_embeddings[i].astype(np.float32),
                "label": int(labels[i]),
            }
            if text_descriptions is not None:
                item["text"] = text_descriptions[i]
            else:
                item["text"] = ""
            data.append(item)
        
        dataset = Dataset.from_list(data)
        
        print(f"Loaded {len(dataset)} {data_type} samples from {current_dir}")
        print(f"  - Stat features shape: {stat_features.shape}")
        print(f"  - BERT embeddings shape: {bert_embeddings.shape}")
        print(f"  - Labels: {np.unique(labels, return_counts=True)}")
        
        return dataset
        
    except FileNotFoundError as e:
        print(f"Error loading split data: {e}")
        print(f"Please run split_modality.py first to generate split data.")
        return None


def extract_text_embeddings(text_descriptions, bert_model_name="bert-base-chinese"):
    """
    提取文本的BERT嵌入特征
    
    功能：使用BERT编码器将文本描述转换为语义嵌入向量
    
    Args:
        text_descriptions (list): 文本描述列表
        bert_model_name (str): BERT模型名称，默认使用bert-base-chinese
        
    Returns:
        np.ndarray: BERT嵌入特征数组，形状为 [num_samples, 768]
    """
    if not torch.cuda.is_available():
        print("=" * 60)
        print("警告: 未检测到GPU (CUDA)！")
        print("当前将使用CPU运行，BERT嵌入提取速度可能极慢。")
        print("请确认是否继续...")
        print("=" * 60)
        choice = input("输入 'y' 继续使用CPU，输入其他键退出: ")
        if choice.strip().lower() != 'y':
            raise RuntimeError("用户选择终止：未检测到GPU。请检查CUDA环境或安装GPU版PyTorch。")

    # 初始化BERT编码器
    bert_encoder = BertEncoder(bert_model_name=bert_model_name)
    bert_encoder.eval()  # 设置为评估模式

    # 在不计算梯度的上下文中提取特征（节省内存）
    with torch.no_grad():
        embeddings = bert_encoder(text_descriptions)

    # 将张量转换为numpy数组并返回
    return embeddings.cpu().numpy()


def collate_fn(batch, tokenizer=None, max_length=128, 
               use_numeric=True, use_bert=True, use_llm=True):
    """
    数据批处理函数（用于DataLoader）
    
    功能：将多个样本整理成一个batch，处理数值特征、文本特征和标签的对齐
    支持消融实验配置：可选择性地忽略数值模态、文本模态或LLM
    
    Args:
        batch (list): 样本列表，每个样本是包含stat、bert、label、text的字典
        tokenizer (optional): LLM的tokenizer，用于文本编码
        max_length (int): 文本最大长度，默认为128
        use_numeric (bool): 是否使用数值模态，False时返回零张量
        use_bert (bool): 是否使用文本模态，False时返回零张量
        use_llm (bool): 是否使用LLM，False时不返回input_ids和attention_mask
        
    Returns:
        dict: 整理后的batch数据
    """
    if isinstance(batch, list) and len(batch) > 0:
        if isinstance(batch[0], dict) and "stat" in batch[0]:
            pass
        elif isinstance(batch[0], dict):
            raise ValueError(f"Batch[0] keys: {batch[0].keys()}")
        elif isinstance(batch, dict):
            batch = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
        else:
            raise ValueError(f"Unexpected batch[0] type: {type(batch[0])}, batch type: {type(batch)}")
    else:
        raise ValueError(f"Unexpected batch format: {type(batch)}")
    
    batch_size = len(batch)
    
    # 根据配置处理数值特征
    if use_numeric:
        stats = torch.tensor([x["stat"] for x in batch], dtype=torch.float32)
    else:
        stats = torch.zeros(batch_size, 9, dtype=torch.float32)
    
    # 根据配置处理BERT特征
    if use_bert:
        berts = torch.tensor([x["bert"] for x in batch], dtype=torch.float32)
    else:
        berts = torch.zeros(batch_size, 768, dtype=torch.float32)
    
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)

    # 构建返回字典
    result = {
        "stat_tensor": stats,
        "bert_tensor": berts,
        "labels": labels
    }

    # 仅在使用LLM时生成input_ids和attention_mask
    if use_llm and tokenizer is not None and "text" in batch[0]:
        texts = [x["text"] for x in batch]
        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        result["input_ids"] = encodings["input_ids"]
        result["attention_mask"] = encodings["attention_mask"]

    return result
