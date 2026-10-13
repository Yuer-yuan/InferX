#!/usr/bin/env bash
set -euo pipefail
source /home/guokc/mypro/gnn-mount-sda1/InferX/env.sh
inferx_gpu=${1:-0}
case "$inferx_gpu" in
  0) inferx_cpus=0-7; inferx_port=18080 ;;
  1) inferx_cpus=28-35; inferx_port=18081 ;;
  *) echo 'Expected GPU 0 or 1' >&2; exit 1 ;;
esac
/usr/bin/python3 - "$inferx_gpu" <<'PY'
import json, os, subprocess, sys
from pathlib import Path
root = Path(os.environ['INFERX_ROOT'])
manifest = json.loads((root / 'models/qwen38-27b/manifest.json').read_text())
assert manifest['verified'], 'GGUF must pass SHA256 verification before loading'
used = int(subprocess.check_output(['nvidia-smi', '-i', sys.argv[1],
    '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
if used > 512:
    raise RuntimeError(f'GPU {sys.argv[1]} already uses {used} MiB; inspect existing jobs first')
PY
export CUDA_VISIBLE_DEVICES="$inferx_gpu"
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
test -f "$INFERX_ROOT/source/llama-build-cuda126/READY"
exec taskset -c "$inferx_cpus" "$INFERX_ROOT/source/llama-build-cuda126/bin/llama-server" \
  --model "$INFERX_ROOT/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf" \
  --host 127.0.0.1 --port "$inferx_port" --parallel 8 --ctx-size 16384 \
  --gpu-layers all --split-mode none --fit off --flash-attn on \
  --cache-type-k f16 --cache-type-v f16 --threads 8 --threads-batch 8 \
  --batch-size 2048 --ubatch-size 512 --metrics --slots --no-webui --threads-http 8 \
  --reasoning off --chat-template-kwargs '{"enable_thinking":false}'
