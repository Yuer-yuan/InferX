#!/usr/bin/env bash
# Full driver wrapper plus the author's dlopen shim for CUDA Runtime launches.
set -euo pipefail
source /home/guokc/mypro/gnn-mount-sda1/InferX/env.sh
inferx_source="$INFERX_ROOT/source/pd-baselines/MorphX"
inferx_cuda="$INFERX_ROOT/toolchains/cuda-12.8.1"
inferx_build="$INFERX_ROOT/pd-baselines/builds"
[[ $(git -C "$inferx_source" rev-parse HEAD) == 4404aed4c180ad83c31179e1ce08ae2234390853 ]]
taskset -c 0-3,56-59 /usr/bin/g++-13 -shared -std=c++20 -O3 -g -fPIC \
  -DSMSCHED_MAPPED_HOST_CONTROL "$inferx_source/runtime/src/cuda-llm.cpp" \
  -I"$inferx_source/include" -I"$inferx_cuda/include" \
  -I"$INFERX_ROOT/source/pd-baselines/gdrcopy/include" \
  -I"$INFERX_ROOT/pd-baselines/deps/usr/include" \
  -L"$inferx_cuda/lib" -L"$INFERX_ROOT/source/pd-baselines/gdrcopy/src" \
  -lcudart -lgdrapi -ldl -Wl,-soname,libcuda.so -o "$inferx_build/libcuda_mapped_full.so"
/usr/bin/gcc-13 -shared -std=gnu11 -O3 -fPIC "$inferx_source/runtime/src/preload.c" \
  -ldl -o "$inferx_build/libpreload.so"
sha256sum "$inferx_build/libcuda_mapped_full.so" "$inferx_build/libpreload.so"
