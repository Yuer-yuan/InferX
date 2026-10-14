#!/usr/bin/env bash
set -euo pipefail
source /home/guokc/mypro/gnn-mount-sda1/InferX/env.sh
/usr/bin/python3 - <<'PY'
import json, os, subprocess
from pathlib import Path
root = Path(os.environ['INFERX_ROOT'])
assert json.loads((root/'models/qwen38-27b/manifest.json').read_text())['verified']
used = int(subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=memory.used',
                                  '--format=csv,noheader,nounits'], text=True).strip())
if used > 512:
    raise RuntimeError(f'GPU 1 already uses {used} MiB; inspect ownership first')
PY
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
test -f "$INFERX_ROOT/source/llama-build-cuda126/READY"
exec taskset -c 28-35 "$INFERX_ROOT/source/llama-build-cuda126/bin/llama-server" \
  --model "$INFERX_ROOT/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_M.gguf" \
  --host 127.0.0.1 --port 18081 --parallel 8 --ctx-size 65536 \
  --gpu-layers all --split-mode none --fit off --flash-attn on \
  --cache-type-k f16 --cache-type-v f16 --threads 8 --threads-batch 8 \
  --batch-size 2048 --ubatch-size 512 --metrics --slots --no-webui --threads-http 8 \
  --reasoning off --chat-template-kwargs '{"enable_thinking":false}'
