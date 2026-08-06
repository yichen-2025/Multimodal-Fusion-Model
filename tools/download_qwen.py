import os
import sys
from huggingface_hub import snapshot_download


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LOCAL_DIR = "./models/qwen2.5-1.5b"


def download_model():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    print(f"正在从镜像网站下载 {MODEL_NAME}")
    print(f"保存路径: {os.path.abspath(LOCAL_DIR)}")
    print("=" * 60)
    
    try:
        snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False,
            endpoint="https://hf-mirror.com",
            resume_download=True,
            max_workers=4
        )
        
        print("=" * 60)
        print(f"模型下载完成！")
        print(f"保存路径: {os.path.abspath(LOCAL_DIR)}")
        print("\n目录内容:")
        
        total_size = 0
        for f in os.listdir(LOCAL_DIR):
            f_path = os.path.join(LOCAL_DIR, f)
            if os.path.isfile(f_path):
                size = os.path.getsize(f_path) / (1024 * 1024)
                total_size += size
                print(f"  {f}: {size:.2f} MB")
        
        print(f"\n总大小: {total_size:.2f} MB")
        return True
        
    except Exception as e:
        print(f"\n下载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)