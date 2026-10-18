063 full L31 fixed W2 fit completed in54.623 seconds. No new model, kernel build, task-quality request, or performance run was started. Owner132176/start46581875 and child132184/start46599920 exited naturally with returncode0; both were absent and the GPU compute list empty at the terminal check.

The candidate preserves059's four levels{-3,-1,+1,+3}, original FP8 logical weights, ordered F32 H512, G512, eight fixed energy-weighted Lloyd updates, final F32 scales, and FP64 GPTQ feedback with the same ten frozen training-only inverse-Hessian blocks. Training is the original8192 tokens; the4096 heldout tokens are first read after cache freeze. The full fit's original256-row subset matches all1,310,720059 codes and2560scale bits exactly. All178,257,920 full-layer codes pass canonical, S4, and u1's062 interleaved pack/unpack checks.

Heldout reconstruction results (relative L2, lower is better):

| Output/reference | Original025 W4 | Candidate W2 |
|---|---:|---:|
| Full projection / original FP8 logical gate weights |0.1421982464|0.2604890259|
| Full17408-channel SwiGLU / original FP8 logical gate weights |0.2134864728|0.3931352214|
| Full5120-channel FFN branch / FP8 gate followed by the same025 down |0.2606925881|0.3966499294|

W2 versus the original025 W4 FFN branch is0.3940207561. Training gives0.3691509733 for the same comparison. The fixed W2 representation therefore adds substantial reconstruction error. These are diagnostics, not a task-accuracy cutoff or a task failure result. No extrapolation from u1's synthetic speed to this fitted model is made.

All branches use represented FP64 products and sigmoid, a BF16 SwiGLU result, then the original F32 H512/A4 and fixed025 down weights. This is not the native group-FMA/exp rounding path. The FP8-gate reference shares the025 down matrix and is not a complete FP8 FFN. The branch excludes the unavailable residual buffer. Inputs are the frozen024 trajectory, not new W2-model activations. Peak Torch allocation was5,826,032,128B, reserved5,941,231,616B; these exclude external CUDA context memory.

Raw results and cache remain on ucloud2 at `/workspace/InferX/three-node-optimization/ucloud2/true-int2-full-L31-GPTQ-063/results/`:

| File | Bytes | SHA256 |
|---|---:|---|
|packed-2bit.bin|44,564,480|7656113c17b77bc898da6d8d5116463ae8378819b58ca67a04718a6bc05b5f39|
|interleaved-W2.bin|44,564,480|ef9409e20ef98cfe6378a75f3e03b9333dd4fdab4e254dc7d8c87d3eb515cf6e|
|scales.f32|1,392,640|ae722997df03db969bebb4dd387652e2a51ac1a3f34b7706b6437e99f169ff4b|

`canonical-manifest.json` and `u1-interleave-manifest.json` describe the two formats. The latter is062's per16-code interleave, not the new register probe's proposed K64 layout. Any later layout conversion can use frozen codes without refitting. No full64-layer W2 cache was created.

The local `small063.tgz` contains77 members and3,619,179B, SHA256 `b8e17bbc9956628754508b0870004c34d1a748c370b565f37326fc72f60ac346`. It includes source/command/runtime identities, original equivalence checks, manifests, all18 metric records and per-token squared-error/reference sums, first8-token branch outputs, and terminal evidence. A separate standard-library recomputation of all18 saved metric aggregates passes in `CPU-stats-review.json`. Large cache arrays stay on ucloud2. Historical `STOP-OVERRIDE.json` is retained; `RESUME-AUTHORIZATION.json` records the subsequent explicit restoration.
