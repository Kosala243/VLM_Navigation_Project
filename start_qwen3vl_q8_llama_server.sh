#!/usr/bin/env bash
set -euo pipefail

# Folder containing:
#   Qwen3VL-8B-Instruct-Q8_0.gguf
#   mmproj-Qwen3VL-8B-Instruct-F16.gguf
QWEN_GGUF_DIR="${QWEN_GGUF_DIR:-models/qwen3vl-8b-gguf}"
MODEL_GGUF="${MODEL_GGUF:-$QWEN_GGUF_DIR/Qwen3VL-8B-Instruct-Q8_0.gguf}"
MMPROJ_GGUF="${MMPROJ_GGUF:-$QWEN_GGUF_DIR/mmproj-Qwen3VL-8B-Instruct-F16.gguf}"
HOST="${LLAMA_HOST:-0.0.0.0}"
PORT="${LLAMA_PORT:-8080}"
CTX_SIZE="${CTX_SIZE:-4096}"

# Your AMD server currently exposes only ~0.5 GB VRAM, so default to CPU.
# If you later get a real GPU build/node, increase this.
N_GPU_LAYERS="${N_GPU_LAYERS:-0}"

if ! command -v llama-server >/dev/null 2>&1; then
  echo "ERROR: llama-server not found. Install/build latest llama.cpp first."
  exit 1
fi

if [ ! -f "$MODEL_GGUF" ]; then
  echo "ERROR: model GGUF not found: $MODEL_GGUF"
  exit 1
fi

if [ ! -f "$MMPROJ_GGUF" ]; then
  echo "ERROR: mmproj GGUF not found: $MMPROJ_GGUF"
  exit 1
fi

exec llama-server \
  -m "$MODEL_GGUF" \
  --mmproj "$MMPROJ_GGUF" \
  --host "$HOST" \
  --port "$PORT" \
  -c "$CTX_SIZE" \
  -ngl "$N_GPU_LAYERS"
