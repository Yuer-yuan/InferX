#!/usr/bin/env bash
# Source this file on vlm-server before installing, downloading or running.
set -euo pipefail
export INFERX_ROOT=/home/guokc/mypro/gnn-mount-sda1/InferX
inferx_mount=$(findmnt -n -o TARGET --target "$(dirname "$INFERX_ROOT")")
if [[ "$inferx_mount" != /home/guokc/mypro/gnn-mount-sda1 ]]; then
  echo 'Expected HDD mount is absent; refusing to write downloads.' >&2
  return 1 2>/dev/null || exit 1
fi
mkdir -p "$INFERX_ROOT"/{cache,tmp,models,data,runs,logs}
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$INFERX_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=30
export UV_CACHE_DIR="$INFERX_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$INFERX_ROOT/cache/uv-python"
export PIP_CACHE_DIR="$INFERX_ROOT/cache/pip"
export XDG_CACHE_HOME="$INFERX_ROOT/cache/xdg"
export TORCH_HOME="$INFERX_ROOT/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$INFERX_ROOT/cache/torchinductor"
export TRITON_CACHE_DIR="$INFERX_ROOT/cache/triton"
export VLLM_CACHE_ROOT="$INFERX_ROOT/cache/vllm"
export NUMBA_CACHE_DIR="$INFERX_ROOT/cache/numba"
export CUDA_CACHE_PATH="$INFERX_ROOT/cache/cuda"
export TMPDIR="$INFERX_ROOT/tmp"
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
