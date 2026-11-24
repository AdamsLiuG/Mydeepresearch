#!/usr/bin/env bash
set -euo pipefail

# The served name must match POLICY_MODEL_ID in the research service.
# Point POLICY_MODEL_BASE_URL at http://127.0.0.1:8001/v1 by default.
backend_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
checkpoint="${POLICY_MODEL_PATH:-${backend_dir}/evals/rl_checkpoints/policy_grpo}"
served_name="${POLICY_MODEL_ID:-deepresearch-policy-v3}"
host="${POLICY_VLLM_HOST:-127.0.0.1}"
port="${POLICY_VLLM_PORT:-8001}"
dtype="${POLICY_VLLM_DTYPE:-bfloat16}"

if [[ "${host}" != "127.0.0.1" && "${host}" != "::1" && -z "${VLLM_API_KEY:-}" ]]; then
  echo "VLLM_API_KEY is required for a non-loopback bind" >&2
  exit 2
fi
if [[ ! "${port}" =~ ^[1-9][0-9]{0,4}$ ]] || (( port > 65535 )); then
  echo "POLICY_VLLM_PORT must be a port from 1 to 65535" >&2
  exit 2
fi
if [[ "${checkpoint}" == -* || ! "${served_name}" =~ ^[a-zA-Z0-9._/-]+$ ]]; then
  echo "Invalid policy model name or path" >&2
  exit 2
fi
if [[ ! -d "${checkpoint}" ]]; then
  echo "Policy checkpoint directory missing: ${checkpoint}" >&2
  exit 2
fi

# An offline-trained adapter is mounted at startup. Disable remote adapter management.
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=False
if [[ -f "${checkpoint}/adapter_config.json" ]]; then
  base_model="${POLICY_VLLM_BASE_MODEL:-$(python3 -c '
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(config.get("base_model_name_or_path") or "Qwen/Qwen2.5-7B-Instruct")
' "${checkpoint}/adapter_config.json")}"
  if [[ "${base_model}" == -* ]]; then
    echo "Invalid policy base model name or path" >&2
    exit 2
  fi
  lora_module="$(python3 -c '
import json
import sys

print(json.dumps({"name": sys.argv[1], "path": sys.argv[2]}))
' "${served_name}" "${checkpoint}")"
  args=("${base_model}" --enable-lora --max-lora-rank 32 --lora-modules "${lora_module}")
else
  args=("${checkpoint}" --served-model-name "${served_name}")
fi
if [[ -f "${checkpoint}/chat_template.jinja" ]]; then
  args+=(--chat-template "${checkpoint}/chat_template.jinja")
fi
exec vllm serve "${args[@]}" --host "${host}" --port "${port}" --dtype "${dtype}" \
  --max-model-len 2048 --enable-prefix-caching --gpu-memory-utilization 0.85
