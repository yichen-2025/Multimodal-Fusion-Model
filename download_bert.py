import os
from huggingface_hub import snapshot_download

MODEL_NAME = "bert-base-chinese"
LOCAL_DIR = "./models/bert"

os.makedirs(LOCAL_DIR, exist_ok=True)

print(f"正在从镜像网站下载 {MODEL_NAME} 到 {LOCAL_DIR}")
print("=" * 60)

snapshot_download(
    repo_id=MODEL_NAME,
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False,
    endpoint="https://hf-mirror.com"
)

print("=" * 60)
print(f"BERT模型已成功保存到 {os.path.abspath(LOCAL_DIR)}")

files = os.listdir(LOCAL_DIR)
print("目录内容:")
for f in files:
    size = os.path.getsize(os.path.join(LOCAL_DIR, f)) / (1024 * 1024)
    print(f"  {f}: {size:.2f} MB")