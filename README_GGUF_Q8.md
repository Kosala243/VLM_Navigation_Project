# Qwen3-VL-8B-Instruct GGUF Q8_0 update

This update keeps your navigation pipeline structure the same. Only `ModelWrapper` changes so it can talk to a local llama.cpp OpenAI-compatible server running the GGUF model.

## 1. Download the GGUF files

```bash
pip install huggingface_hub
python download_qwen3vl_gguf_q8.py
```

Expected files:

```text
models/qwen3vl-8b-gguf/Qwen3VL-8B-Instruct-Q8_0.gguf
models/qwen3vl-8b-gguf/mmproj-Qwen3VL-8B-Instruct-F16.gguf
```

## 2. Start llama.cpp server

```bash
./start_qwen3vl_q8_llama_server.sh
```

Because your current AMD GPU exposes only about 0.5 GB VRAM, this script defaults to CPU mode with `N_GPU_LAYERS=0`.

## 3. In another SSH/tmux terminal, run the pipeline

```bash
./run_with_gguf_q8.sh
```

Optional:

```bash
NAV_GOAL="B0.004" IMAGE_DIR="images/seq2" MAX_IMAGES=5 ./run_with_gguf_q8.sh
```

## 4. Full precision mode still works

Unset `MODEL_BACKEND` or set:

```bash
export MODEL_BACKEND=transformers
python run_navigation.py
```
