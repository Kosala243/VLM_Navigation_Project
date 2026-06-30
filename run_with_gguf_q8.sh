#!/usr/bin/env bash
set -euo pipefail

export MODEL_BACKEND="llama_cpp_server"
export LLAMA_SERVER_URL="${LLAMA_SERVER_URL:-http://127.0.0.1:8080/v1}"
export LLAMA_MODEL_ID="${LLAMA_MODEL_ID:-Qwen3-VL-8B-Instruct-GGUF-Q8_0}"
export MODEL_TEMPERATURE="${MODEL_TEMPERATURE:-0.0}"
export MODEL_TOP_P="${MODEL_TOP_P:-1.0}"

python run_navigation.py
