#!/usr/bin/env bash
set -euo pipefail
source /home/guokc/mypro/gnn-mount-sda1/InferX/env.sh
TASK_BASE="$INFERX_ROOT/sota-validation"
TASK_CUDA="$INFERX_ROOT/toolchains/cuda-12.8.1"
TASK_SOURCE="$TASK_BASE/runtime-native-ex-v1"
TASK_EXPECTED=$(sha256sum "$INFERX_ROOT/pd-strategies/morphx-runtime-before.cpp" | cut -d ' ' -f1)
cd "$INFERX_ROOT/code-pd-baselines"
"$INFERX_ROOT/repo-venv/bin/python" -m scripts.prepare_morphx_validation --source "$INFERX_ROOT/source/pd-baselines/MorphX" --destination "$TASK_SOURCE" --expected-sha256 "$TASK_EXPECTED"
mkdir "$TASK_BASE/builds"
taskset -c 0-3,56-59 /usr/bin/g++-13 -shared -std=c++20 -O3 -g -fPIC \
  -DSMSCHED_MAPPED_HOST_CONTROL -DSMSCHED_ALLOW_DEFAULT_STREAM -DSMSCHED_THREADSAFE_PROFILE -DSMSCHED_CAPTURE_BYPASS \
  "$TASK_SOURCE/runtime/src/cuda-llm.cpp" -I"$TASK_SOURCE/include" -I"$TASK_CUDA/include" \
  -I"$INFERX_ROOT/source/pd-baselines/gdrcopy/include" -I"$INFERX_ROOT/pd-baselines/deps/usr/include" \
  -L"$TASK_CUDA/lib" -L"$INFERX_ROOT/source/pd-baselines/gdrcopy/src" \
  -lcudart -lgdrapi -ldl -Wl,-soname,libcuda.so -o "$TASK_BASE/builds/libcuda_native_ex_v1.so"
TASK_PASS="$INFERX_ROOT/pd-baselines/builds/morphx-pass/libsmsched.so"
for TASK_VARIANT in native transformed; do
  TASK_FLAGS=()
  if [[ "$TASK_VARIANT" == transformed ]]; then TASK_FLAGS+=("-fpass-plugin=$TASK_PASS"); fi
  taskset -c 0-3,56-59 /usr/bin/clang++-18 -std=c++17 --cuda-path="$TASK_CUDA" --cuda-gpu-arch=sm_89 \
    --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13 -Wno-unknown-cuda-version \
    -include "$INFERX_ROOT/code-pd-baselines/scripts/cuda18_compat.cuh" "${TASK_FLAGS[@]}" \
    "$INFERX_ROOT/code-pd-baselines/tests/fixtures/morphx_cooperative.cu" \
    -L"$TASK_CUDA/lib" -lcudart -ldl -pthread -o "$TASK_BASE/builds/cooperative_$TASK_VARIANT"
done
sha256sum "$TASK_BASE/builds/"*
