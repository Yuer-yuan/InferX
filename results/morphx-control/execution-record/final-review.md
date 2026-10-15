# Fresh final review — /root/morphx_control_final_review
Model: gpt-6-astra; reasoning high; fork none; read-only; no subagents.
Range:4801610..e46e31a; one final review, no re-review.

**Strengths**

- Reviewed the whole `4801610..e46e31a` range, including adapter code, tests, eight archived runtime trees, qualification artifacts, configurations, exports, selection, and report.
- Event recording follows the successful native patched launch on its stream; publication follows recording. Unrecorded/error states cannot qualify completion, and retirement requires a successful query.
- Exact D32 initialization preserves allocation → initialization → launch ordering. Profiler progress reads, `schedule_2`, live SM writes, and native/PTSZ dispatch remain intact.
- Independently checked all 185 capsule payload hashes, eight manifest inventories, analysis-source hashes, and export/selection hashes. Counts agree: 300 mixed, 52 solo, and 175 diagnostic responses.
- ABI fairness and NUMA placement rulings are justified. Fresh controls, interleaved repeats, pre-formal selection, single-solo limitations, and the unknown original-author ceiling are clearly documented.

**Critical**

None found.

**Important**

1. **Preparation does not verify the complete copied source inventory.** inferx/morphx_control.py:76

   The verifier checks only paths listed in `manifest['source_sha256']`, but line 88 recursively copies every file under `runtime` and `include`. An extra unlisted file therefore passes verification. For example, an added `include/cuda.h` can change compilation through the destination’s include search path despite every checked hash matching.

   This violates the exact-source/full-inventory preparation contract. Compare the complete copied file set against the manifest’s corresponding inventory before creating the destination, rejecting additions and omissions. Add a regression for an unlisted header and confirm rejection occurs before output creation.

   The committed capsules are internally consistent; this finding concerns future preparation accepting changed inputs, rather than evidence of altered measured binaries.

**Minor**

None requiring separate action.

**Recommendations**

Apply the preparation-guard correction and targeted regression in the authorized consolidated fix pass. Preserve existing measured binaries, capsules, and results: this correction need not change CUDA runtime behavior. Complete the remaining review/archive ledger entries afterward.

**Declined to judge**

- Unfinished EB+ implementation, calibration, and efficacy: separate phase and untracked candidate files outside this review range.
- Original-author MorphX optimality and universal performance ceilings: unavailable implementation/platform evidence; the report explicitly leaves these unknown.
- Sustained 128K concurrency, other quantizations/capacities, and broader workload generalization: outside the measured cohort and explicitly limited in the report.
- Inherited profile-only allocation lifetime policy: unchanged predecessor behavior; the new completion event is never created on that path.

**Assessment — Ready to merge: With fixes**

The runtime changes and recorded comparison support the report’s restrained conclusions. Fix the incomplete source-inventory guard before considering this phase complete; no other Critical or Important issue was established. No tests or GPU work were rerun during this read-only review.
