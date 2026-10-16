#!/usr/bin/env bash
# Isolated build/evaluation prerequisites; never modifies the system Python or driver.
set -euo pipefail
inferx_prefix=${1:?Usage: bootstrap_ninfer.sh /absolute/data/prefix}
[[ "$inferx_prefix" = /* && "$inferx_prefix" != / ]]
export INFERX_ROOT="$inferx_prefix"
mkdir -p "$INFERX_ROOT"/{cache,tmp,envs,logs,models,runs,config,tools,source}
export MAMBA_ROOT_PREFIX="$INFERX_ROOT/cache/mamba"
export XDG_CACHE_HOME="$INFERX_ROOT/cache/xdg"
export TMPDIR="$INFERX_ROOT/tmp"
export PIP_CACHE_DIR="$INFERX_ROOT/cache/pip"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME="$INFERX_ROOT/cache/huggingface"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DOWNLOAD_TIMEOUT=180
export HF_HUB_ETAG_TIMEOUT=60
if [[ ${INFERX_DIRECT_MIRRORS:-0} == 1 ]]; then
  export NO_PROXY="${NO_PROXY:+$NO_PROXY,}hf-mirror.com,mirrors.tuna.tsinghua.edu.cn,pypi.tuna.tsinghua.edu.cn"
  export no_proxy="$NO_PROXY"
fi
inferx_mamba=${INFERX_MAMBA:-$INFERX_ROOT/tools/micromamba}
inferx_channel=${INFERX_CONDA_CHANNEL:-conda-forge}
inferx_index=${INFERX_PYPI_INDEX:-https://pypi.org/simple}
"$inferx_mamba" --version
"$inferx_mamba" --no-rc create --yes \
  --root-prefix "$MAMBA_ROOT_PREFIX" --prefix "$INFERX_ROOT/envs/build" \
  --override-channels --channel "$inferx_channel" \
  'python=3.12' 'gxx_linux-64=13' 'ffmpeg=6.1' 'libcurl>=8.5' \
  'cmake=3.31' ninja pkg-config git pip zlib
"$INFERX_ROOT/envs/build/bin/python" -m pip install \
  --index-url "$inferx_index" \
  'huggingface_hub[cli]==0.36.0' 'hf_transfer==0.1.9' \
  'aiohttp==3.11.11' 'pytest==8.2.2' 'tokenizers==0.22.1' 'uv==0.11.29'
"$inferx_mamba" --no-rc env export \
  --prefix "$INFERX_ROOT/envs/build" --explicit > "$INFERX_ROOT/config/build-conda-explicit.txt"
"$INFERX_ROOT/envs/build/bin/python" -m pip freeze > "$INFERX_ROOT/config/build-pip-freeze.txt"
"$INFERX_ROOT/envs/build/bin/hf" version
