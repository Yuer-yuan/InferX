#!/usr/bin/env bash
set -euo pipefail
export INFERX_ROOT=${1:?Usage: build_ninfer_portable.sh /absolute/data/prefix}
source "$INFERX_ROOT/config/ninfer-env.sh"
inferx_compiler="$INFERX_ROOT/envs/build/bin/x86_64-conda-linux-gnu"
inferx_arch=${INFERX_CUDA_ARCH:-89}
inferx_build="$INFERX_ROOT/build/ninfer-sm$inferx_arch"
inferx_cc=${CC:-$inferx_compiler-gcc}
inferx_cxx=${CXX:-$inferx_compiler-g++}
cmake -S "$INFERX_ROOT/source/ninfer-4090" -B "$inferx_build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$inferx_cc" \
  -DCMAKE_CXX_COMPILER="$inferx_cxx" \
  -DCMAKE_CUDA_COMPILER="$INFERX_CUDA_ROOT/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER="$inferx_cxx" \
  -DCUDAToolkit_ROOT="$INFERX_CUDA_ROOT" \
  -DCMAKE_CUDA_ARCHITECTURES="$inferx_arch" \
  -DCMAKE_PREFIX_PATH="$INFERX_ROOT/envs/build" \
  -DFETCHCONTENT_SOURCE_DIR_XGRAMMAR_SOURCE="$INFERX_ROOT/source/xgrammar_source-src" \
  -DNINFER_ENABLE_UI=OFF -DNINFER_BUILD_APPS=ON \
  -DNINFER_BUILD_BENCHMARKS=ON -DBUILD_TESTING=OFF
cmake --build "$inferx_build" --parallel "${INFERX_BUILD_JOBS:-4}"
"$inferx_build/apps/ninfer-serve" --help
