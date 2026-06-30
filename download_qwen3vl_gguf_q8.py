"""Download Qwen3-VL-8B-Instruct GGUF Q8_0 + FP16 mmproj.

Run:
    pip install huggingface_hub
    python download_qwen3vl_gguf_q8.py
"""
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "Qwen/Qwen3-VL-8B-Instruct-GGUF"
OUT_DIR = Path("models/qwen3vl-8b-gguf")
FILES = [
    "Qwen3VL-8B-Instruct-Q8_0.gguf",
    "mmproj-Qwen3VL-8B-Instruct-F16.gguf",
]

OUT_DIR.mkdir(parents=True, exist_ok=True)

for filename in FILES:
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        local_dir=str(OUT_DIR),
        local_dir_use_symlinks=False,
    )
    print(path)