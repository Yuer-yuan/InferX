#!/usr/bin/env bash
# Port the pinned compiler pass to installed LLVM18; no driver/system install.
set -euo pipefail
source /home/guokc/mypro/gnn-mount-sda1/InferX/env.sh
inferx_morphx="$INFERX_ROOT/source/pd-baselines/MorphX"
inferx_build="$INFERX_ROOT/pd-baselines/builds/morphx-pass"
inferx_finish_barrier=${INFERX_MORPHX_FINISH_BARRIER:-0}
[[ "$inferx_finish_barrier" == 0 || "$inferx_finish_barrier" == 1 ]]
inferx_cmake_flags=()
if [[ "$inferx_finish_barrier" == 1 ]]; then
  inferx_build+="-finish-barrier"
  inferx_cmake_flags+=(-DCMAKE_CXX_FLAGS=-DSMSCHED_VIRTUAL_CTA_FINISH_BARRIER)
fi
[[ $(git -C "$inferx_morphx" rev-parse HEAD) == 4404aed4c180ad83c31179e1ce08ae2234390853 ]]
python3 - "$inferx_morphx" <<'PY'
import subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]); path=root/'smsched-pass/SMSched.cpp'
original=subprocess.check_output(['git','show','HEAD:smsched-pass/SMSched.cpp'],cwd=root,text=True)
# LLVM18 returns a string triple; captures(none) was represented as nocapture.
ported=original.replace('M.getTargetTriple().', 'Triple(M.getTargetTriple()).')
old='Attribute::getWithCaptureInfo(ctx, CaptureInfo::none())'
assert old in ported
ported=ported.replace(old,'Attribute::get(ctx, Attribute::NoCapture)')
old=', llvm::ThinOrFullLTOPhase Phase'
assert old in ported
ported=ported.replace(old,'')
legacy=ported
# LLVM18 CUDA kernels use nvvm.annotations instead of the PTX_Kernel calling
# convention used by the artifact's newer Clang. Normalize before either pass.
needle='\tfor (Function &F : M) {'
normalization='''\tif (auto *annotations = M.getNamedMetadata("nvvm.annotations")) {
        for (auto *entry : annotations->operands()) {
            if (entry->getNumOperands() < 3) continue;
            auto *kind = dyn_cast<MDString>(entry->getOperand(1));
            auto *value = dyn_cast<ValueAsMetadata>(entry->getOperand(0));
            if (kind && kind->getString() == "kernel" && value)
                if (auto *kernel = dyn_cast<Function>(value->getValue()))
                    kernel->setCallingConv(CallingConv::PTX_Kernel);
        }
    }
'''
assert needle in ported
ported=ported.replace(needle,normalization+needle,1)
normalized=ported
needle='\t\tBuilder.SetInsertPoint(blockFin);\n'
assert ported.count(needle)==1
ported=ported.replace(needle,needle+'''#ifdef SMSCHED_VIRTUAL_CTA_FINISH_BARRIER
        // A native CTA's shared storage remains live until all threads finish.
        // Do not let the leader overwrite it while another warp still uses it.
        Builder.CreateCall(barrier0);
#endif
''',1)
if path.read_text() not in (original,legacy,normalized,ported):
    raise RuntimeError('unexpected source edits; refusing to replace')
path.write_text(ported)
PY
cmake -S "$inferx_morphx/smsched-pass" -B "$inferx_build" -G Ninja \
  -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm -DCMAKE_BUILD_TYPE=Release "${inferx_cmake_flags[@]}"
taskset -c 0-3,56-59 cmake --build "$inferx_build" --parallel 2
git -C "$inferx_morphx" diff > "$inferx_build/source.patch"
sha256sum "$inferx_build/libsmsched.so"
if [[ "$inferx_finish_barrier" == 1 ]]; then
  exit 0
fi
inferx_cuda="$INFERX_ROOT/toolchains/cuda-12.8.1"
PATH="$inferx_cuda/bin:$PATH" /usr/bin/clang++-18 \
  --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13 --cuda-device-only \
  --cuda-path="$inferx_cuda" --cuda-gpu-arch=sm_89 -Wno-unknown-cuda-version \
  -O3 -fpass-plugin="$inferx_build/libsmsched.so" -S \
  "$INFERX_ROOT/code-pd-baselines/tests/fixtures/morphx_probe.cu" \
  -o "$INFERX_ROOT/pd-baselines/builds/morphx_probe.ptx"
rg '\.entry morphx_probe\(' "$INFERX_ROOT/pd-baselines/builds/morphx_probe.ptx"
rg '\.entry morphx_probe_pk\(' "$INFERX_ROOT/pd-baselines/builds/morphx_probe.ptx"
/usr/bin/g++-13 "$INFERX_ROOT/code-pd-baselines/tests/fixtures/morphx_probe.cpp" \
  -I"$inferx_cuda/include" -lcuda -o "$INFERX_ROOT/pd-baselines/builds/morphx_probe"
