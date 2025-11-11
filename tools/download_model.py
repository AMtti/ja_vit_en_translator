# tools/download_model.py
from huggingface_hub import snapshot_download
from pathlib import Path

# 変更可: 保存先
TARGET_DIR = Path(__file__).resolve().parents[1] / "models" / "facebook" / "m2m100_418M"

if __name__ == "__main__":
    TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="facebook/m2m100_418M",
        local_dir=str(TARGET_DIR),
        local_dir_use_symlinks=False
    )
    print(f"Downloaded to: {TARGET_DIR}")