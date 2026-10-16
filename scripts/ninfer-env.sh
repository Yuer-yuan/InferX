#!/usr/bin/env bash
# Source with an explicit data prefix. Node aliases belong in the deployment inventory.
: "${INFERX_ROOT:?Set INFERX_ROOT to the absolute experiment data directory}"
export INFERX_ROOT
[[ "$INFERX_ROOT" = /* && "$INFERX_ROOT" != / ]] || return 1
if [[ -z ${INFERX_CUDA_ROOT:-} ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    inferx_nvcc=$(readlink -f "$(command -v nvcc)")
    INFERX_CUDA_ROOT=$(dirname "$(dirname "$inferx_nvcc")")
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    INFERX_CUDA_ROOT=$(readlink -f /usr/local/cuda)
  else
    echo 'Set INFERX_CUDA_ROOT or provide nvcc on PATH.' >&2
    return 1
  fi
fi
export INFERX_CUDA_ROOT
[[ -x "$INFERX_CUDA_ROOT/bin/nvcc" ]] || return 1
export PATH="$INFERX_ROOT/envs/build/bin:$INFERX_ROOT/tools:$INFERX_CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$INFERX_ROOT/envs/build/lib:$INFERX_CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME="$INFERX_ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DOWNLOAD_TIMEOUT=180
export HF_HUB_ETAG_TIMEOUT=60
export UV_CACHE_DIR="$INFERX_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$INFERX_ROOT/cache/uv-python"
export PIP_CACHE_DIR="$INFERX_ROOT/cache/pip"
export XDG_CACHE_HOME="$INFERX_ROOT/cache/xdg"
export TIKTOKEN_CACHE_DIR="$INFERX_ROOT/cache/tiktoken"
export MODELSCOPE_CACHE="$INFERX_ROOT/cache/modelscope"
export TORCH_HOME="$INFERX_ROOT/cache/torch"
export TRITON_CACHE_DIR="$INFERX_ROOT/cache/triton"
export CUDA_CACHE_PATH="$INFERX_ROOT/cache/cuda"
export TMPDIR="$INFERX_ROOT/tmp"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PKG_CONFIG_PATH="$INFERX_ROOT/envs/build/lib/pkgconfig:$INFERX_ROOT/envs/build/share/pkgconfig"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
if [[ ${INFERX_DIRECT_MIRRORS:-0} == 1 ]]; then
  export NO_PROXY="$NO_PROXY,hf-mirror.com,mirrors.tuna.tsinghua.edu.cn,pypi.tuna.tsinghua.edu.cn"
fi
export no_proxy="$NO_PROXY"
