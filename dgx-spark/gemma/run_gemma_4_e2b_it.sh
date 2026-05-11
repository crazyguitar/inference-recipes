#!/usr/bin/env bash
# Recipe: serve google/gemma-4-E2B-it with vLLM.
# Dense ~2.3B effective (~5.1B w/ embeddings) Gemma-4 with text+image+audio.
# Based on https://github.com/crazyguitar/recipes/blob/main/Google/Gemma4.md,
# adapted for single-GPU GB10 (Grace-Blackwell unified memory).
set -euo pipefail

MODEL="${MODEL:-google/gemma-4-E2B-it}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_NAME="${SERVED_NAME:-gemma-4-e2b-it}"

TP_SIZE="${TP_SIZE:-1}"
# Native context goes up to 128K per the model card; 32K default keeps KV-cache
# modest on GB10. Bump for long-doc workloads.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
# Must be >= max_tokens_per_mm_item (2496 for Gemma-4 vision tower); chunked MM
# input is force-disabled for multimodal-bidirectional attention models, so each
# MM item must fit in one batch.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
# GB10 (and other Grace-Blackwell unified-memory parts) share VRAM with host
# processes, so the practical free fraction at startup is well under 0.9.
# Datacenter H100/H200 users can override with GPU_MEM_UTIL=0.90.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.65}"
DTYPE="${DTYPE:-bfloat16}"

# E2B model card lists Text, Image, Audio (no video). Drop audio if unused.
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\": 4, \"audio\": 1\}}"

# Hugging Face cache / token (export HF_TOKEN before running for gated repos).
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

# Enable parallel chunked downloads when hf_transfer is installed (10-20x faster
# than the default sequential downloader). Falls back silently if missing.
if python3 -c "import hf_transfer" 2>/dev/null; then
    export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
else
    echo "[recipe] hf_transfer not installed; downloads will be slow." \
         "Run: pip install hf_transfer" >&2
fi

exec vllm serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype "$DTYPE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT" \
    --async-scheduling
