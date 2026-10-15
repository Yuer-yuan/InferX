Reviewer: /root/sota_phase_final_review, gpt-6-astra/high, fresh read-only context, range1ccf655..a4303d2. Received2026-10-16. Full returned report:

### Strengths

- The report clearly distinguishes **MorphX-on-SGLang measurements from original MorphX efficacy**. CUDA-copy transport, SGLang admission, partial kernel coverage, and the launch-cap ablation are disclosed.
- Qualification evidence retains failed GQA and Bullet state-contract outcomes without converting them into performance claims.
- Mixed runs retain three independent repeat identities; matched solo runs are correctly described as one diagnostic repetition. Profiled latency is excluded.
- The normalizer reconstructs tools from SSE, compares frozen payloads, checks arrival schedules and observed runtime settings, and records source fingerprints.
- I independently verified all **249 capsule payload hashes**, all **six analysis-source fingerprints**, the comparison SHA, and the reported **300 formal + 100 diagnostic + 52 solo** request totals.
- The preservation/restoration evidence and reviewed controller paths consistently use owned execution and fresh service identity checks. The preexisting user files remain outside this change.

### Issues

#### Critical (Must Fix)

None found.

#### Important (Should Fix)

1. **Proc-address substitution still loses the requested function ABI.**
   - **Files:** `inferx/morphx_validation.py:137`, `tests/test_morphx_validation.py:118`; resulting runtime self-query branches are visible in `results/sota-validation/execution/morphx-runtime-before.cpp:1151` and preserved in the final source capsule.
   - The patch validates native resolution, then continues into unconditional substitutions. In particular, `cuGetProcAddress_v2("cuGetProcAddress", ..., 11030, ...)` can resolve the legacy four-argument entry natively, but the wrapper replaces it with five-argument `cuGetProcAddress_v2`. Calling that pointer through the requested legacy signature supplies no valid fifth argument; the wrapper can pass an indeterminate pointer to the driver for writing.
   - Conversely, the legacy wrapper substitutes itself for a request resolving the newer ABI. Its native legacy call cannot populate the newer status output.
   - NVIDIA documents the [legacy four-argument signature](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-driver-api/group__CUDA__DRIVER__ENTRY__POINT.html) and requires that the [returned function match the requested version’s ABI](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__DRIVER__ENTRY__POINT.html). Successful native validation alone does not preserve that contract.
   - **Fix:** Give legacy and v2 entry points their actual signatures and select the replacement according to the native resolved ABI. Add regressions exercising both self-query directions through correctly typed function pointers. The current test defines both native query functions and both wrappers with five arguments, hiding this mismatch.
   - This is a compatibility defect in the qualified runtime, **not evidence that the recorded model runs failed**. Preserve the measured binary and capsule as historical evidence when correcting it.

2. **The strict input gate accepts materially shorter inputs than the frozen lane.**
   - **File:** `inferx/sota_results.py:63`.
   - The validator accepts any long prompt count of at least `130000`, although this pinned comparison’s complete cold/resume inputs are `131060` and `131049`; similarly, short inputs only need to reach `24000` instead of the observed `24568`.
   - Exact request-payload equality does not establish that the server processed the entire payload. Server-side truncation or a tokenizer/configuration mismatch can therefore pass this gate while reducing measured work.
   - I executed the unchanged audit function in memory with its existing fixture, changing the long SSE usage and summary to `130000`; it still returned **25 strict requests**.
   - **Fix:** Validate against qualified expected prompt counts keyed by frozen prompt and tokenizer/checkpoint identity, and require corresponding solo/mixed observations to match. Add near-boundary regressions, rather than testing only a grossly shortened 4096-token input.
   - All archived comparison counts are currently exactly `24568`, `131049`, or `131060`, so this finding concerns the validator’s protection of subsequent comparisons; it does not invalidate the existing 452 responses.

#### Minor (Nice to Have)

None elevated.

### Recommendations

- Address the two findings in the authorized consolidated fix pass. Preserve old runtime artifacts and clearly distinguish any corrected runtime from the binary used for the existing measurements.
- Re-export the existing raw campaigns with the strengthened input gate; these checks should not require new GPU measurements.
- Continue treating transport validation and EB+ as subsequent research work. The present measurements support the scoped conclusion that this port did not improve the joint tradeoff.

### Declined to judge

- **Original MorphX performance ceiling:** The private author orchestration and GDRCopy transport were not reproduced; the report explicitly leaves original efficacy unknown.
- **Full BulletServe GPU efficacy:** The Task 3 ruling narrows this phase to the actual state-contract probe and explicit unqualified status. Full IPC/state/quantization integration remains outstanding.
- **EB+ implementation or efficacy:** The included documents describe future work; no implemented or measured EB+ claim is made.
- **General NInfer numerical qualification:** The failing GQA geometry is preserved as an unqualified branch, separate from the qualified SGLang workload.
- **Sustained arrivals and late full-context interference:** The frozen workload ends short arrivals at 46 seconds. The report acknowledges this limitation rather than extrapolating.
- **Physical bandwidth and instantaneous SM occupancy:** The recorded trace supports kernel-lifetime and copy observations, not these unavailable counters.
- **Portability beyond the pinned SM89/Triton/CUDA environment:** The adapters explicitly constrain their qualified environment; this phase does not promise a general CUDA interposer.
- **Runnability of superseded historical helpers as current commands:** The reproduction README expressly archives failed/superseded variants and identifies the qualified execution paths.
- **Preexisting dirty report and solo supplement:** These are user-owned work outside the reviewed range and were not attributed to this implementation.

### Assessment

**Ready to merge? With fixes.**

**Reasoning:** The phase’s measurement provenance and scoped conclusions are strong, and the recorded results remain supported. Correct the remaining proc-address ABI mismatch and strengthen the complete-input gate before treating these helpers as the foundation for subsequent comparisons.

Review was read-only. I did not rerun GPU work or the full suite; the recorded **254 passing tests** are executor evidence. A direct local Python import lacked `aiohttp`, so the narrow validator check used the unchanged function bodies extracted into memory. No checkout, service, or GPU state was changed.
