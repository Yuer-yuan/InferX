#!/usr/bin/env bash
set -euo pipefail
source /home/guokc/mypro/gnn-mount-sda1/InferX/env.sh
mkdir -p "$INFERX_ROOT/source"
cd "$INFERX_ROOT/source"
curl -fL --retry 5 -C - -o llama-v0.6.0.tar.gz https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/tags/v0.6.0
sha256sum llama-v0.6.0.tar.gz > llama-source.sha256
echo '09b36dba235fcac180efff18514da7038e65d8e5b9e4aefa59238468abfdec12  llama-v0.6.0.tar.gz' | sha256sum -c -
tar -xzf llama-v0.6.0.tar.gz
cmake -S llama.cpp-0.6.0 -B llama-build \
  -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build llama-build --target llama-server llama-bench -j 8
"$INFERX_ROOT/source/llama-build/bin/llama-server" --version
