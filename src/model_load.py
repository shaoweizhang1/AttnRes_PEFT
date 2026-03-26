import os

from huggingface_hub import snapshot_download


MODEL_NAME = "Qwen/Qwen2.5-7B"
SAVE_DIR = "./model"


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=SAVE_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
    )


if __name__ == "__main__":
    main()
