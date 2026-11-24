#!/usr/bin/env bash
set -euo pipefail

# Serve fine-tuned DeepResearch policy checkpoint via vLLM OpenAI-compatible API
# Wire to DeepResearch via:
#   POLICY_MODEL_MODE=shadow   # (or canary / active)
#   POLICY_MODEL_ID=deepresearch-policy-v3
#   POLICY_MODEL_BASE_URL=http://127.0.0.1:8001/v1

MODEL_PATH="${POLICY_MODEL_PATH:-backend/evals/rl_checkpoints/policy_grpo}"
SERVED_NAME="${POLICY_MODEL_ID:-deepresearch-policy-v3}"
PORT="${POLICY_VLLM_PORT:-8001}"

exec python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --port "${PORT}" \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --dtype auto
