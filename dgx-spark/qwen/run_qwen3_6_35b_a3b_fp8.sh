#!/usr/bin/env bash
# Recipe: serve Qwen/Qwen3.6-35B-A3B-FP8 with vLLM.
# MoE model (35B total / 3B active) shipped natively in FP8.
# Based on https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html
# and https://github.com/crazyguitar/recipes/blob/main/Qwen/Qwen3.5.md
# Adapted for single-GPU GB10 (Grace-Blackwell unified memory).
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_NAME="${SERVED_NAME:-qwen3.6-35b-a3b-fp8}"

TP_SIZE="${TP_SIZE:-1}"
# Qwen3's native context is 32768. Going higher requires YaRN rope scaling
# (applied automatically below). Larger windows cost KV-cache memory, so drop
# MAX_NUM_SEQS or GPU_MEM_UTIL if you OOM.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
NATIVE_CTX=32768

ROPE_ARGS=()
if (( MAX_MODEL_LEN > NATIVE_CTX )); then
    # ceil(MAX_MODEL_LEN / NATIVE_CTX) — YaRN factor must cover the extension.
    FACTOR=$(( (MAX_MODEL_LEN + NATIVE_CTX - 1) / NATIVE_CTX ))
    # vLLM exposes rope_scaling via --hf-overrides (merged into the HF config).
    ROPE_ARGS=(--hf-overrides "{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":${FACTOR}.0,\"original_max_position_embeddings\":${NATIVE_CTX}}}")
fi
# GB10 (and other Grace-Blackwell unified-memory parts) share VRAM with host
# processes, so the practical free fraction at startup is well under 0.9.
# Datacenter H100/H200 users can override with GPU_MEM_UTIL=0.90.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.65}"

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
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    "${ROPE_ARGS[@]}" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --async-scheduling
