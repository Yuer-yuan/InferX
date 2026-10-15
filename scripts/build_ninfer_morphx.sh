#!/usr/bin/env bash
set -euo pipefail
source /home/guokc/mypro/gnn-mount-sda1/InferX/env.sh
inferx_source="$INFERX_ROOT/source/pd-baselines/ninfer-morphx"
inferx_cuda="$INFERX_ROOT/toolchains/cuda-12.8.1"
inferx_deps="$INFERX_ROOT/toolchains/ninfer-deps"
inferx_pass="$INFERX_ROOT/pd-baselines/builds/morphx-pass/libsmsched.so"
[[ $(git -C "$inferx_source" rev-parse HEAD) == 5c60b7c9b455231795c09da21a9fbb6aa53f08e5 ]]
export PKG_CONFIG_PATH="$inferx_deps/usr/lib/x86_64-linux-gnu/pkgconfig"
export LD_LIBRARY_PATH="$inferx_cuda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$INFERX_ROOT/ninfer-build-venv/bin:$inferx_cuda/bin:$PATH"
python3 - "$inferx_source" <<'PY'
import subprocess,sys
from pathlib import Path
root=Path(sys.argv[1])
for relative in ['CMakeLists.txt','src/CMakeLists.txt']:
    p=root/relative
    original=subprocess.check_output(['git','show','HEAD:'+relative],cwd=root,text=True)
    ported=original.replace('$<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>',
        '$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:-Xptxas=-v>')
    ported=ported.replace('$<$<COMPILE_LANGUAGE:CUDA>:--split-compile=0>',
        '$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:--split-compile=0>')
    ported=ported.replace('$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>',
        '$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:-lineinfo>')
    legacy=ported
    if relative == 'src/CMakeLists.txt':
        # The register cap applies to the whole unit; use a compiler-matched control.
        ported+='''
if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
  set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/ops/launcher/gqa_attention_prefill.cu
    APPEND PROPERTY COMPILE_OPTIONS -Xcuda-ptxas --maxrregcount=120)
endif()
'''
    if p.read_text() not in (original,legacy,ported):
        raise RuntimeError(f'unexpected edits: {relative}')
    p.write_text(ported)
p=root/'src/ops/kernel/gqa_attention_prefill_i8.cuh'
original=subprocess.check_output(['git','show','HEAD:src/ops/kernel/gqa_attention_prefill_i8.cuh'],cwd=root,text=True)
ported=original.replace('#pragma once', '''#pragma once
#if defined(__clang__) && __clang_major__ <= 18
#define __maxnreg__(value)
#endif''',1)
if p.read_text() not in (original,ported):
    raise RuntimeError('unexpected gqa kernel edits')
p.write_text(ported)
p=root/'tests/targets/qwen3_6_27b/test_load_plan.cpp'
original=subprocess.check_output(['git','show','HEAD:tests/targets/qwen3_6_27b/test_load_plan.cpp'],cwd=root,text=True)
ported='#include <algorithm>\n'+original
if p.read_text() not in (original,ported):
    raise RuntimeError('unexpected load-plan test edits')
p.write_text(ported)
PY
cmake -S "$inferx_source" -B "$INFERX_ROOT/pd-baselines/builds/ninfer-morphx" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc-13 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CUDA_COMPILER=/usr/bin/clang++-18 \
  -DCUDAToolkit_ROOT="$inferx_cuda" -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared \
  -DCMAKE_CUDA_FLAGS="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13 -Wno-unknown-cuda-version -include $INFERX_ROOT/code-pd-baselines/scripts/cuda18_compat.cuh -fpass-plugin=$inferx_pass" \
  -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=OFF \
  -DNINFER_ENABLE_UI=OFF
taskset -c 0-3,56-59 cmake --build "$INFERX_ROOT/pd-baselines/builds/ninfer-morphx" --parallel 2
