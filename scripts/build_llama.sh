#!/usr/bin/env bash
set -euo pipefail
source /home/guokc/mypro/gnn-mount-sda1/InferX/env.sh
mkdir -p "$INFERX_ROOT/source"
cd "$INFERX_ROOT/source"
inferx_source_sha=09b36dba235fcac180efff18514da7038e65d8e5b9e4aefa59238468abfdec12
if ! echo "$inferx_source_sha  llama-v0.6.0.tar.gz" | sha256sum -c - 2>/dev/null; then
  curl -fL --retry 5 -o llama-v0.6.0.tar.gz https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/tags/v0.6.0
fi
echo "$inferx_source_sha  llama-v0.6.0.tar.gz" | sha256sum -c -
sha256sum llama-v0.6.0.tar.gz > llama-source.sha256
echo d81235049384534c167caea52b85a694f6103d14 > llama-source.commit
if [[ ! -d llama.cpp-0.6.0 ]]; then tar -xzf llama-v0.6.0.tar.gz; fi
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cmake -S llama.cpp-0.6.0 -B llama-build-cuda126 \
  -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.6 \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build llama-build-cuda126 --target llama-server llama-bench -j 8
"$INFERX_ROOT/source/llama-build-cuda126/bin/llama-server" --version
/usr/bin/python3 - <<'PY'
import ctypes
v = ctypes.c_int()
ctypes.CDLL('libcudart.so.12').cudaRuntimeGetVersion(ctypes.byref(v))
assert v.value == 12060, v.value
print('CUDA runtime verified:', v.value)
PY
touch llama-build-cuda126/READY
