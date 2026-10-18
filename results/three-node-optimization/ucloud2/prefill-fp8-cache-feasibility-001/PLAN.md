# Phase-private FP8 FFN gate-up cache: source feasibility and probe plan

**Goal:** Test whether a scalar-scaled E4M3 gate-up prefill operator can beat the current pipelined Q4A8 whole operator on SM89, with independently measured numerical and task-quality changes.

**Status:** CPU source/filesystem review only. No CUDA library was loaded, no CUDA API/context/GPU query, nvcc/build, operator execution, model edit, or extra inference request was performed. Original FP8-reference collector35195/start39983597 and service34365/start39960588 remain authoritative. Future implementation/build/GPU work requires root technical review and natural termination of the original300.

**Architecture:** Keep the original artifact, packed Q4 weights, all target decode and MTP paths. Own64 immutable, prefill-only E4M3 gate-up caches. Quantize each invocation's BF16 activation with one tensor-wide FP32 scalar, run TN cuBLASLt with FP32 accumulation and BF16 raw gate/up, then the existing FP32 SwiGLU formula and BF16 output. This is a new numerical profile.

**Tech stack:** Installed CUDA13.2.20260303 / cuBLAS13.3.0.5 ELF SONAME13, C++/CUDA, explicit SM89. SDK hashes/ELF identities are in `SDK-identity.json`; these identify files, not a tested runtime combination.

**Spec:** Coordinator request and frozen `root-prefill-FP8-cache-primary-feasibility-review.json`. Read-only shared `source/ninfer-ucloud1-pipeline-queue-grouped-v2` source235 reference; hashes/line excerpts of the actual bytes read are in `actual-source.json`. No peer SSH/contact.

## Global constraints and review focus

- Only64 FFN gate-up matrices, each34816×5120; preserve packed storage and original decode. No all-model cache, CPU offload, checkpoint change, framework switch or tuning grid.
- Probe only T256 and T2048; same450W/modified48GB4090. No GPU action is authorized by this document.
- Original32768 output budget/natural EOS/quality context65536 and quality KV327680 must survive capacity planning. No clipping to fit.
- Zero tensors, outliers, nonfinite values and underflowed scale/reciprocal must have explicit, tested behavior; do not filter bad values silently.
- Original logical gate/up order, group64/nibble indexing, scales, TN/leading dimensions and BF16 intermediate rounding must be tested separately.
- Cache publication must follow completion of on-load writes; scratch must obey the engine's nonblocking-stream and graph-address ownership rules.
- Whole-op measurement includes every activation reduction/scale/quantization/GEMM/SwiGLU and intermediate read/write. Bare GEMM is supplementary.
- One heuristic result per shape, fixed before timing; unsupported means preserved failure, not an algorithm search or a hidden fallback result.

## Source findings

`Variant::post_mixer` at variant.cpp:258 gates the A8 route on `TextPhase::Prefill && T>=256` and enabled `swiglu`. It bypasses the public A16-only wrapper. Its scratch and BF16 activation are scope-owned. `jg_pipeline_adapter.cuh:40` quantizes BF16 input then launches `rowsplit_tall_a8::launch<128,true>` for folded gate/up. The adapter chunks at4096; this proposal does not broaden the reviewed2048 prefill shape.

Physical Q4 rows are logical gate rows0..17407 followed by up rows17408..34815. Per row there are80 K-groups,32 code bytes and one binary16 scale per group. Low nibble is even-k, high nibble odd-k; signed code is `(nibble^8)-8`. The folded kernel's row mapping changes its compute tiles, not artifact row order (`rowsplit_tall_mma.cuh:298`).

`Q4G64_F16S` names **binary16 scales**, not a mandated half-rounded decoded weight. Current A8 uses signed codes directly in exact group int32 dots, then FP32 `FMA(FP32(weight_scale*activation_scale), dot, acc)` in group order. Its SwiGLU consumes FP32 gate/up accumulators and rounds only the final product to BF16. A16 MMA decodes weights to BF16; it is a distinct comparison, not a universal oracle.

For the proposed cache define `w=FP32(intcode*FP32(F16scale))`, without inserting a new FP16 cast. Integer×half is exactly representable in FP32 here. Confirm this interpretation of “same Q4group64F16 weights” during root review. If actual half rounding is desired, that must be an explicit different representation and oracle, not a silent implementation detail.

The current fused SwiGLU uses `x/(1+exp2.approx(-log2(e)*x))`, FP32 reciprocal RN and final BF16 RN (`math.cuh:26`). Candidate raw gate/up BF16 rounding is new, even if both input quantizers were disabled.

Root's actual profile assigns5531.050ms/15612.870ms mapped prefill kernel sum to persistent gate/up (35.426%;2495 launches,mean2.216854ms). This is a sum across shapes/capture, not an exclusive critical-path share or an isolated T2048 baseline. Registers255 alone do not prove spilling/occupancy. The primary bottleneck hypothesis is plausible; speedup and roofline remain unmeasured.

## Installed SDK and official format support

Installed files provide scalar A/B scale modes/pointers, `FAST_ACCUM`, alignment preferences, workspace limits and E4M3 conversion definitions. `SDK-identity.json` includes header SHA256, actual library SHA256/build IDs/SONAME, and version macros; no `cublasGetVersion` call was used.

CUDA13.2 documents scalar tensor-wide FP8 scaling for SM8.9+, with TN, FP32 compute/scales and16-byte alignment. Its vector/128-block modes require SM90, MX32 requires SM100+. The BF16 C/D type combination is listed. Thus the proposed descriptors are source-feasible on Ada; an actual heuristic/kernel result at either shape is **not yet known**. [CUDA13.2 cuBLAS](https://docs.nvidia.com/cuda/archive/13.2.0/cublas/index.html#narrow-precision-data-types-usage).

The official sample supplies descriptor/scale/one-heuristic setup, pinned at commit6b3fa100c2349f6e267893cf20456c6e39f5bb72 with byte hashes in `official-sample-identity.json`. Its sample output is FP8; our proposed BF16 output must use the documented BF16 descriptors and omit FP8 D scale/amax settings. Its default stream0 call must not be copied into the engine's nonblocking stream. [Pinned sample](https://github.com/NVIDIA/CUDALibrarySamples/tree/6b3fa100c2349f6e267893cf20456c6e39f5bb72/cuBLASLt/LtFp8Matmul).

## Exact formats and operation

See `formats.json` for machine-readable dimensions, byte strides and scratch. Use column-major descriptors to view existing token-major storage without a transpose kernel:

| Operand | Stored descriptor | Operation / physical layout |
|---|---|---|
| A | E4M3 K5120×N34816,lda5120 | transpose; row-major cached W[N,K] |
| B | E4M3 K5120×T,ldb5120 | non-transpose; token-major X8[T,K] |
| C/D | BF16 N34816×T,ld34816 | token-major raw gate/up[T,N]; beta0,C aliases D |
| Y | BF16 T×17408 | SiLU(Dgate in FP32)×Dup in FP32,final BF16 RN |

Matmul is `(m,n,k)=(34816,T,5120)`, alpha1/beta0, `CUBLAS_COMPUTE_32F`, `CUDA_R_32F` scales, default epilogue, `FAST_ACCUM=0`. C/D aliasing and heuristic validity are future probe gates; initialize D once before warmup. Set only A/B scalar scale pointers/modes, not non-narrow D scales. Allocate all buffers/scalars/workspace at256-byte alignment, advertise actual minimum16B alignment. Both shapes and all strides/layer offsets satisfy16B divisibility. No grouped-batch FP8 API or Hopper block-scale kernel is involved.

Per layer `sW=FP32(max(abs(w))/448)`, `invW=FP32(1/sW)`, `W8=RNE_SATFINITE_E4M3(FP32(w*invW))`. Scan/decode original packed weights twice on load (amax,then encode), directly into the final cache; avoid a full-model F16 staging allocation. Store actual scalar and reciprocal bits and per-layer zero/underflow/saturation statistics. Keep original packed arrays unchanged.

Per invocation reduce amax across the entire finite BF16[T,K], finalize device `sX/ invX` identically, encode X8, then matmul and SwiGLU. All-zero tensors use scale1/code0. NaN/Inf or a nonzero maximum yielding zero/nonfinite scale/reciprocal rejects the probe with raw evidence. Actual FP8 underflows follow RNE, with zero counts reported. No percentile scale/calibration/IFBench feedback. A conservative reduction is4096elements per partial, a final scalar kernel, then conversion: all three operations count in timing.

## Capacity and lifetime

`memory-budget.json` computes64×34816×5120=11,408,506,880B (10.625GiB), plus16KiB padded layer-scale metadata. T256 adds86,246,912B; T2048 adds220,211,456B conservative scratch, including raw gates142,606,336B, X8 10,485,760B, amax10,240B, scalar256B and fixed64MiB cuBLASLt workspace. Existing BF16 input/output are reported separately. Do not count BF16 gate/up as a free epilogue.

Root's node1 quality OFF free17,802,461,184B already includes quality KV327680. Cache leaves6,393,954,304B before new scratch; cache/scales/T2048 scratch leave6,173,726,464B. Do **not** subtract the7,059,013,632B KV131072→327680 expansion again. This is an estimate on a different process/node configuration, not an observed ucloud2 fit. Original packed weights remain resident. One full decoded matrix staging, if introduced, adds356,515,840B; it is unnecessary in the two-pass proposal.

Future whole-phase planning must account for GDN, attention, logits/draft, graph pools, load double-buffering, fragmentation and all overlapping lifetimes. Reuse one scratch region only under proved serialized stream ownership; otherwise add disjoint copies. Record actual startup and phase peaks before inference. Cache allocation on a 24GB unmodified4090 is outside this48GB proposal.

## Future single-operator probe preregistration (not executed)

**Proposed isolated files:** `bench/ops/fp8_gateup_probe.cu` owns the closed whole-op comparison; `tests/ops/fp8_gateup_numeric_oracle.cpp` owns independent CPU represented arithmetic and guards. No target runtime/model integration in the first probe. Exact names/interfaces are proposed, not existing code.

- [ ] Root reviews source semantics, precision gate and this prereg; original300 collector/scorer reaches terminal. Keep the current model resident until an explicit later owner-controlled cutover.
- [ ] Freeze the exact original layer0 gate-up packed codes/scales and deterministic seed42 BF16 activations; save hashes. Use identical operands for current pipelinedA8, originalA16 and candidate. No model weight edit or IFBench calibration. Also include sparse basis, dense signed values, zero, single outlier and E4M3 midpoint/subnormal cases; allocate only one matrix for this standalone probe.
- [ ] Query **one** heuristic result per shape with64MiB workspace cap and real alignment. Save returned count/status, algorithm ID/config/workspace/waves; `AlgoCheck` once, then freeze. If either shape is unsupported, preserve the failure and stop that shape without changing workspace/algorithm/precision. No timed candidate was selected from a grid.
- [ ] Honor the read-only source AGENTS.md build gate: any later build needs root authorization; any feature/critical integration needs its corresponding CTest plus all84 existing CTests, with zero failures. None runs in this source-only stage.
- [ ] Run independent numeric checks before timing. Record separate raw GEMM and SwiGLU errors, saturation/underflow and guard regions; fail on nonfinite, OOB or represented-arithmetic mismatch. Preserve actual encoded bytes and actual scalar bits.
- [ ] Run20 warmups then100 complete-operator timings per shape, seed42 fixed operands, in10 fixed alternating blocks of10 A8/candidate invocations; repeat values are synthetic timing, not task/performance qualification. Same explicit nonblocking stream, end-event synchronization only at sample boundary, no inner-stage sync. No profile in these timings. Report medians/P95/paired block bootstrap, plus first-use/on-load costs separately. Record A16 only as a numerical/whole-op comparator, not the competing optimized baseline.
- [ ] One separate diagnostic replay records reduction/conversion/GEMM/SwiGLU stream intervals and temporary-memory traffic estimates; GPU interval union and wall time remain distinct from kernel sums. No NCU counter-permission retry. Heuristic/SDK support is not actual kernel execution evidence.
- [ ] A useful first gate is ≥10% lower whole-op median at T2048, no >5% T256 regression, and passed represented oracle. These thresholds are proposed for root review now, before observing candidate timings. Failure rejects further model integration unless a new sourced mechanism is approved; no retuning from IFBench.

### Numeric oracle and separate error budgets

1. Independently decode signed Q4 nibbles and binary16 scales using host bit arithmetic, compute logical `w` in FP64, and validate the production FP32 exact products. Independently implement E4M3 finite values/RNE ties/subnormals/saturation (not an invocation of the production converter). Verify encoded cache/input bytes against this oracle, including actual FP32 scale/reciprocal rounding.
2. The **represented FP8 oracle** is FP64 dot of decoded E4M3 codes multiplied by the exact stored scalar values, followed by the candidate's declared BF16 gate/up rounding and mathematical FP64 SiLU/product/final BF16. Also compare SwiGLU using the *actual* BF16 GEMM gates, isolating GEMM-reduction error from activation implementation error. Values within the FP32 accumulation error envelope of a BF16 midpoint must report both possible adjacent roundings rather than claim bitwise parity.
3. For raw gate/up use a predeclared FP32 forward-error envelope based on `gamma_(K+8)*sum(abs(products))` plus0.5BF16ulp and output-rounding-boundary handling. This is a conservative arithmetic guard, not a task-quality tolerance. For final output propagate the actual gate/up envelope with |SiLU'|≤1.1 and |SiLU(g)|≤|g|, add BF16 rounding and a separately checked SiLU math residual (≤2e-6×max(1,|g|) before multiplication, proposed gate). Reject unexplained violations; do not loosen after results.
4. Compute additional errors versus the logical original-Q4/BF16-input mathematical oracle, versus the **represented group64 A8** FP64 oracle, and versus originalA16 (which has BF16 weight decoding). Report maxabs, normalizedL2, P99 absolute/relative excluding declared nearzero mask, raw gate/up BF16 cast contribution and zero/sign-change rates. These measure representation changes; passing FP8's own oracle never establishes oldA8/A16 equivalence. Do not copy existing A16 tolerance to FP8.

### Precision and end-to-end gates after a useful operator result

Keep original greedy/no-thinking/seed42, penalties0, native MTP setting and same inputs/IDs/budgets/context/cache state in an explicit cacheOFF/cacheON pair. First run the same six long objectives with max256/natural EOS, preserving all failures and raw native input IDs. Require no new failed objective relative to its paired control, and prove actual T>=256 gate-up-cache executions by layer/chunk, not a flag. Historical4/6 is context, not a reason to edit questions or expected answers.

The original300 ordinary IFBench prompts are short (27..258 rendered tokens): many/all requests can bypass this prefill path. The current NInfer300 has observed A8coverage0/300. Full300 scoring is useful whole-model task evidence but is not cache quality evidence without actual eligible execution. Do not pad/nonce the corpus and present it as original300. Root should freeze a separate held-out long-instruction cohort before execution if expanded active-path quality is required. Proposed noninferiority gate for an adequately active paired task cohort is a lower95% paired prompt-cluster CI above−2percentagepoints on prompt/instruction strict, with all original length/failure outputs included; confirm this criterion at root review, never tune it from results.

Only then consider a bounded original cold8192/512 sustained C5/C6/C8 test with real request replacement, actual batch/progress, unchanged450W/KV/config and separate physical timeline. Existing source235/bcdf profiles cannot predict end-to-end TPOT tails. No unchanged weak-candidate100/grid/fullIFBench is scheduled by this plan.

## Decision

**Source-feasible, measured benefit and capacity unresolved.** The documented SM89 scalar FP8 format and installed headers/libs support preparing this one operator. Doubled decoded weight bytes versus Q4, tensor-wide activation precision, extra reduction and142.6MB BF16 raw intermediate at T2048 can erase GEMM savings. There is insufficient evidence to reject or claim a win without the bounded whole-op probe. Deliver this package for root review; do not build or execute it now.
