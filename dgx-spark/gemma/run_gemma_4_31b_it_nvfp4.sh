#!/usr/bin/env bash
# Recipe: serve nvidia/Gemma-4-31B-IT-NVFP4 with vLLM.
# NVFP4 (NVIDIA FP4) post-training quant of google/gemma-4-31B-it via NVIDIA
# Model Optimizer. ~21B effective params after quant — fits on a single
# Hopper/Blackwell GPU. Tested on H100 per the model card; Blackwell adds
# native FP4 tensor-core paths.
# Model card: https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4
# NVIDIA's reference command uses TP=8 on H100; we default TP=1 for GB10.
# Based on https://github.com/crazyguitar/recipes/blob/main/Google/Gemma4.md,
# adapted for single-GPU GB10 (Grace-Blackwell unified memory).
set -euo pipefail

MODEL="${MODEL:-nvidia/Gemma-4-31B-IT-NVFP4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_NAME="${SERVED_NAME:-gemma-4-31b-it-nvfp4}"

TP_SIZE="${TP_SIZE:-1}"
# Native context goes up to 256K per the NVIDIA card; keep 32K default to
# bound KV-cache on GB10. Bump for long-doc / video workloads.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
# Must be >= max_tokens_per_mm_item (2496 for Gemma-4 vision tower); chunked MM
# input is force-disabled for multimodal-bidirectional attention models, so each
# MM item must fit in one batch.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
# GB10 (and other Grace-Blackwell unified-memory parts) share VRAM with host
# processes, so the practical free fraction at startup is well under 0.9.
# Datacenter H100/H200/B200 users can override with GPU_MEM_UTIL=0.90.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.65}"

# Gemma-4-31B-IT-NVFP4 accepts text + image + video (up to 60s @ 1fps per the
# NVIDIA card). Audio kept for parity with the 26B recipe; drop if unused.
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\": 4, \"video\": 1, \"audio\": 1\}}"

# NVIDIA Model Optimizer ships NVFP4 weights; pass --quantization modelopt
# explicitly per the model card (some vLLM versions do not auto-detect).
QUANTIZATION="${QUANTIZATION:-modelopt}"

# Hugging Face cache / token (export HF_TOKEN before running for gated repos).
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

# Enable parallel chunked downloads when hf_transfer is installed (10-20x faster
# than the default sequential downloader, and bypasses the httpx brotli-decoder
# bug that crashes vLLM's in-process downloader on partial transfers). Falls
# back to the slower path if missing.
if python3 -c "import hf_transfer" 2>/dev/null; then
    export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
else
    echo "[recipe] hf_transfer not installed; downloads will be slow and may" \
         "hit a brotli decoder bug. Run: pip install hf_transfer" >&2
fi

# Pre-download outside the engine. The hf CLI has retry/backoff and doesn't
# share vLLM's in-process httpx state, so a transient brotli decode error here
# is recoverable instead of fatal. Skipped if the model path is local.
if [[ "$MODEL" != /* && "$MODEL" != .* ]]; then
    echo "[recipe] Pre-downloading $MODEL into HF cache..." >&2
    for attempt in 1 2 3; do
        if hf download "$MODEL"; then
            break
        fi
        echo "[recipe] hf download attempt $attempt failed; retrying..." >&2
        sleep $((attempt * 5))
    done
fi

exec vllm serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size "$TP_SIZE" \
    --quantization "$QUANTIZATION" \
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
