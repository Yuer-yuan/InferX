#!/usr/bin/env bash
set -euo pipefail
source /home/guokc/mypro/gnn-mount-sda1/InferX/env.sh
# Tuning defaults are explicit; the report records the final selected invocation.
inferx_engine=${1:-llama}
inferx_draft=${2:-0}
inferx_kv=${3:-q8_0}
inferx_ubatch=${4:-512}
inferx_context=${5:-65536}
inferx_concurrency=${6:-1}
inferx_cache=${7:-cold}
inferx_pmin=${8:-0}
[[ "$inferx_engine" == llama || "$inferx_engine" == ninfer ]]
[[ "$inferx_cache" == cold || "$inferx_cache" == warm ]]
for inferx_n in "$inferx_draft" "$inferx_ubatch" "$inferx_context" "$inferx_concurrency"; do
  [[ "$inferx_n" =~ ^[0-9]+$ ]]
done
inferx_used=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
if (( inferx_used > 512 )); then
  echo "GPU 1 already uses $inferx_used MiB; inspect process ownership first" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES=1
INFERX_ENGINE="$inferx_engine" /usr/bin/python3 - <<'PY'
import json, os
from pathlib import Path
root=Path(os.environ['INFERX_ROOT'])
name='qwen38-27b' if os.environ['INFERX_ENGINE']=='llama' else 'qwen38-ninfer'
manifest=json.loads((root/'models'/name/'manifest.json').read_text())
assert manifest['verified'], 'model artifact must be verified before serving'
path=root/'models'/name/manifest['filename']
assert path.stat().st_size==manifest['size'], 'model artifact size changed'
PY
if [[ "$inferx_engine" == llama ]]; then
  export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  inferx_args=()
  if (( inferx_draft > 0 )); then
    inferx_args+=(--spec-type draft-mtp --spec-draft-n-max "$inferx_draft" \
      --spec-draft-p-min "$inferx_pmin")
  fi
  if [[ "$inferx_cache" == cold ]]; then inferx_args+=(--cache-ram 0); fi
  exec taskset -c 28-35 "$INFERX_ROOT/source/llama-build-cuda126/bin/llama-server" \
    --model "$INFERX_ROOT/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf" \
    --alias qwen38-27b --host 127.0.0.1 --port 18081 \
    --parallel "$inferx_concurrency" --ctx-size "$inferx_context" \
    --gpu-layers all --split-mode none --fit off --flash-attn on \
    --cache-type-k "$inferx_kv" --cache-type-v "$inferx_kv" \
    --threads 8 --threads-batch 8 --batch-size 2048 --ubatch-size "$inferx_ubatch" \
    --metrics --slots --no-webui --threads-http 8 \
    --reasoning off --chat-template-kwargs '{"enable_thinking":false}' "${inferx_args[@]}"
else
  export LD_LIBRARY_PATH="$INFERX_ROOT/toolchains/cuda-12.8.1/lib:$INFERX_ROOT/toolchains/ninfer-deps/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  inferx_args=()
  if (( inferx_draft > 0 )); then
    inferx_args+=(--spec mtp --draft-tokens "$inferx_draft" --lm-head-draft)
  fi
  if [[ "$inferx_cache" == cold ]]; then inferx_args+=(--no-prefix-reuse); fi
  exec taskset -c 28-35 "$INFERX_ROOT/source/ninfer-build-sm89/apps/ninfer-serve" \
    "$INFERX_ROOT/models/qwen38-ninfer/qwen3_8_27b.ninfer" \
    --host 127.0.0.1 --port 18081 --model-id qwen38-27b \
    --max-context "$inferx_context" --kv-capacity "$inferx_context" \
    --max-concurrency "$inferx_concurrency" --prefill-chunk "$inferx_ubatch" \
    --kv-dtype "$inferx_kv" --no-thinking --greedy \
    --presence-penalty 0 --frequency-penalty 0 \
    --request-log-jsonl "${INFERX_REQUEST_LOG:-$INFERX_ROOT/tuning/ninfer-requests.jsonl}" "${inferx_args[@]}"
fi
