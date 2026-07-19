#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-}"

if [[ -z "$PROFILE" || ! -f "$PROFILE" ]]; then
    echo "Usage: $0 <environment-profile>"
    exit 1
fi

set -a
source "$PROFILE"
set +a

MODEL_PATH="${MODELS_DIR}/${LLAMA_MODEL_FILE}"
MMPROJ_PATH="${MODELS_DIR}/${LLAMA_MMPROJ_FILE}"
SERVER_PATH="${LLAMA_ROCM_DIR}/llama-server"

for path in "$MODEL_PATH" "$MMPROJ_PATH" "$SERVER_PATH"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: required file not found: $path"
        exit 1
    fi
done

echo "Profile:       $PROFILE"
echo "llama-server:  $SERVER_PATH"
echo "Model:         $MODEL_PATH"
echo "Projector:     $MMPROJ_PATH"
echo "Context:       ${LLAMA_CTX_SIZE}"
echo "GPU layers:    ${LLAMA_N_GPU_LAYERS}"
echo
ls -lh "$SERVER_PATH" "$MODEL_PATH" "$MMPROJ_PATH"
