#!/usr/bin/env bash
set -euo pipefail
source /home/guokc/mypro/gnn-mount-sda1/InferX/env.sh
inferx_source="$INFERX_ROOT/source/ninfer-4090"
inferx_cuda="$INFERX_ROOT/toolchains/cuda-12.8.1"
inferx_deps="$INFERX_ROOT/toolchains/ninfer-deps"
[[ $(git -C "$inferx_source" rev-parse HEAD) == 5c60b7c9b455231795c09da21a9fbb6aa53f08e5 ]]
# Ubuntu 24.04 development packages were extracted, not installed system-wide.
# Use system runtime libraries and isolated headers/pkg-config metadata.
export PKG_CONFIG_PATH="$inferx_deps/usr/lib/x86_64-linux-gnu/pkgconfig"
export LD_LIBRARY_PATH="$inferx_cuda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$INFERX_ROOT/ninfer-build-venv/bin:$inferx_cuda/bin:$PATH"
export CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 CUDAHOSTCXX=/usr/bin/g++-13
/usr/bin/python3 - <<'PY'
import os
from pathlib import Path
root=Path(os.environ['INFERX_ROOT'])/'toolchains/ninfer-deps'
libdir=root/'usr/lib/x86_64-linux-gnu'
for link in libdir.glob('*.so'):
    if link.is_symlink() and not link.exists():
        # The server already has matching Ubuntu runtime package versions.
        name=os.readlink(link)
        target=Path('/usr/lib/x86_64-linux-gnu')/name
        assert target.is_file(), f'missing runtime for {link}: {target}'
        (libdir/name).symlink_to(target)
for path in (root/'usr/lib/x86_64-linux-gnu/pkgconfig').glob('*.pc'):
    text=path.read_text().replace('prefix=/usr', f'prefix={root}/usr')
    text=text.replace('includedir=/usr/include', f'includedir={root}/usr/include')
    text=text.replace('libdir=/usr/lib', f'libdir={root}/usr/lib')
    path.write_text(text)
PY
cmake -S "$inferx_source" -B "$INFERX_ROOT/source/ninfer-build-sm89" -G Ninja \
  -U '*FFMPEG*' -U '*LIBCURL*' \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_CUDA_COMPILER="$inferx_cuda/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER="$CUDAHOSTCXX" \
  -DCUDAToolkit_ROOT="$inferx_cuda" -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=ON \
  -DNINFER_ENABLE_UI=OFF
taskset -c 0-15,56-71 cmake --build "$INFERX_ROOT/source/ninfer-build-sm89" --parallel 4
"$INFERX_ROOT/source/ninfer-build-sm89/apps/ninfer-serve" --help
