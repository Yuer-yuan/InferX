#!/usr/bin/env bash
# Full driver wrapper plus the author's dlopen shim for CUDA Runtime launches.
set -euo pipefail
source /home/guokc/mypro/gnn-mount-sda1/InferX/env.sh
inferx_source="$INFERX_ROOT/source/pd-baselines/MorphX"
inferx_cuda="$INFERX_ROOT/toolchains/cuda-12.8.1"
inferx_build="$INFERX_ROOT/pd-baselines/builds"
inferx_name=${INFERX_MORPHX_RUNTIME_NAME:-libcuda_mapped_full.so}
[[ "$inferx_name" == libcuda_mapped_full.so || "$inferx_name" == libcuda_mapped_default.so || "$inferx_name" == libcuda_mapped_threadsafe.so || "$inferx_name" == libcuda_mapped_capture.so ]]
inferx_flags=()
if [[ "$inferx_name" != libcuda_mapped_full.so ]]; then
  inferx_flags+=(-DSMSCHED_ALLOW_DEFAULT_STREAM)
fi
if [[ "$inferx_name" == libcuda_mapped_threadsafe.so || "$inferx_name" == libcuda_mapped_capture.so ]]; then
  inferx_flags+=(-DSMSCHED_THREADSAFE_PROFILE)
fi
if [[ "$inferx_name" == libcuda_mapped_capture.so ]]; then
  inferx_flags+=(-DSMSCHED_CAPTURE_BYPASS)
fi
inferx_temporary=$(mktemp -d "$inferx_build/.runtime-build.XXXXXX")
trap 'rm -rf "$inferx_temporary"' EXIT
[[ $(git -C "$inferx_source" rev-parse HEAD) == 4404aed4c180ad83c31179e1ce08ae2234390853 ]]
taskset -c 0-3,56-59 /usr/bin/g++-13 -shared -std=c++20 -O3 -g -fPIC \
  "${inferx_flags[@]}" \
  -DSMSCHED_MAPPED_HOST_CONTROL "$inferx_source/runtime/src/cuda-llm.cpp" \
  -I"$inferx_source/include" -I"$inferx_cuda/include" \
  -I"$INFERX_ROOT/source/pd-baselines/gdrcopy/include" \
  -I"$INFERX_ROOT/pd-baselines/deps/usr/include" \
  -I"$INFERX_ROOT/code-pd-baselines/scripts" \
  -L"$inferx_cuda/lib" -L"$INFERX_ROOT/source/pd-baselines/gdrcopy/src" \
  -lcudart -lgdrapi -ldl -Wl,-soname,libcuda.so -o "$inferx_temporary/$inferx_name"
/usr/bin/gcc-13 -shared -std=gnu11 -O3 -fPIC "$inferx_source/runtime/src/preload.c" \
  -ldl -o "$inferx_temporary/libpreload.so"
# Atomic replacement preserves any older mapping held by a live diagnostic.
mv "$inferx_temporary/$inferx_name" "$inferx_build/$inferx_name"
if ! cmp -s "$inferx_temporary/libpreload.so" "$inferx_build/libpreload.so"; then
  mv "$inferx_temporary/libpreload.so" "$inferx_build/libpreload.so"
fi
sha256sum "$inferx_build/$inferx_name" "$inferx_build/libpreload.so"
