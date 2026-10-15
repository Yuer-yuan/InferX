# MorphX control-transport fairness validation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline task-by-task; one fresh final whole-change review, no implementer delegation.

**Goal:** Reduce and causally measure control-port overhead while retaining actual MorphX profiling and dynamic SM scheduling.

**Architecture:** Source-verified immutable variants of qualified CUDA-copy-v2. Replace only the agent integer initializer and/or completion-only DMA polling, leaving native dispatch, progress profiling, live bounds, engine/admission and compiled modules unchanged. Qualify before comparing full-input waves.

**Tech Stack:** Python/pytest, C++/CUDA12.8 runtime adapter, existing LLVM18/Marlin/SGLang0.5.19, frozen requests/owned GPU1 harness.

**Spec:** docs/superpowers/specs/2026-10-16-morphx-control-design.md

## Global Constraints

- Branch research/ninfer-mtp-tuning; preserve dirty prior report/untracked solo8files. All remote source/build/env/cache/raw on HDD /home/guokc/mypro/gnn-mount-sda1/InferX; source env.sh, reuse27B.
- GPU1 UUID GPU-4d9bef9d-631c-bb36-3ff4-589d6a184189 only; GPU0 intentionally stopped. Fresh previous terminal restoration, identity/argv/idle guard and signal-safe exact original256K/MTP7/C1/rk4v4-e8 restoration with both manifests. No foreign stops/global changes.
- Immutable v2 source/library/overlay/raw; fail on source mismatch or existing destination. New morphx-control result paths and named binaries/manifests.
- Same27B W4A16/FP8KV/16BF16SSM/131328KV/C4/MTP0/eager/primaryP-D/admission. Frozen24568short/131060cold/131049resume; .5RPS48s/longat16; fresh code+fourstrict24K warmups/group. No profile timing as formal efficacy.
- Original profiler progress/CI/wave computation, schedule_2/liveSMwrite and native/cooperative/extended/PTSZ launch behavior remain. Three transport variants fixedsplit96/mode16; then at most three policy diagnostics on selected transport(split60mode0,96mode0,96mode16), freeze before formal3repeats+ownserialC4solos.

## Review Focus

- Event recorded too early, unrecorded event queried as complete, or recycling before kernel physical completion must never silently qualify a run.
- Event error/record failure or missing real fill symbol must fail loudly without publishing invalid ownership or counting failed longs as efficacy.
- Profiler progress and live dynamic SM control must still execute; an event-only fake scheduler or always-unrestricted kernel cannot be labeled MorphX.
- Exact uint32 agent values and original/per-thread explicit stream ordering must survive the replacement, including non-byte-repeated values.
- Different quantization/capacity/cohort, single-solo variability and profiled perturbation must remain explicit; reduced engineering overhead is not original-SOTA optimality or paper novelty.

---

### Task 1: Source-qualified transport variants and actual model correctness

**Files:** Create inferx/morphx_control.py, scripts/morphx_control_ops.hpp, scripts/prepare_morphx_control.py, tests/test_morphx_control.py; new results/morphx-control/qualification and HDD immutable builds/helpers.

**Interfaces:** Consume exact v2 manifest/source; apply inferx.morphx_validation.upgrade_proc_address_queries first in the new tree, retain historical old binary, and requalify corrected ABI on GPU. `patch_control_runtime(source: str, *, device_fill: bool, event_completion: bool) -> str` rejects stale anchors; preparation verifies full source inventory before writing a fresh named tree/library. Small C++ control helper provides exact fill and recorded-event completion semantics; concrete native API table/layout is implementer-owned.

- [x] Write source-mismatch/overwrite regressions plus compiled asynchronous-driver behavioral oracles: initialize agents128/768/1024 on exact stream after allocation; event not-ready leaves resources live, ready frees once, errors fail; unrecorded queries rejected and record precedes publish. Assert profiler retains original progress path and native launch blocks unchanged. Run targeted tests RED.
- [x] Implement minimal checked source adapter/helper, preserving two synchronized progress/bound initializations and existing profiler; GREEN. No async-init publication race introduced. Event exists only on non-profile launches; cleanup on errors and record-after-success before pending publication.
- [x] Build three fresh libraries on HDD (fill,event,both); rehash source/binary manifests. Actual GPU vector and five Marlin geometries each profiling on/off; concurrent two-stream completion/SM oracle. Use existing owned controller and preserve failed outputs.
- [x] Full27B code/four24K/complete128K gates for selected plausible variants with actual runtime coverage; every measured variant must qualify. Fix demonstrated failures with RED→GREEN; unknown never becomes a performance point. Full CPU suite, exact service proof and commit.

Expected: actual mechanism-preserving, model-qualified transport variants or substantive unresolved boundary, no unmeasured efficacy claim.

### Task 2: Joint transport/policy comparison and next baseline decision

**Files:** configs/morphx-control.json; results/morphx-control exports/source/reproduction/figures; docs/reports/2026-10-16-morphx-control.md. Reuse existing strict normalizer; extend only with meaningful failing regressions if necessary.

**Interfaces:** Task1 qualified libraries feed unchanged selected-policy diagnostics and frozen full comparisons. Existing compiler/v2 data remain sourced historical references; newly measured control makes any claimed efficacy gain causal.

- [x] One complete cold diagnostic per qualified transport at96mode16; select jointly short/long/throughput. On selected transport diagnose60mode0,96mode0,96mode16, retaining explicit ablation names; freeze before formal data. Record actual progress-profile/SM writes and qualification/source hashes.
- [x] Run freshly matched compiler control and selected port, three independent cold/resume mixed repeats/group and same-C4 serial solos; no builds/tests on performance cores during timed waves. Exclude failures, no silent reruns. If no lane qualifies, preserve unknown with exact gates and proceed to EB+ rather than manufacture a score.
- [x] Export twice; verify strict tools/fullinput/arrivals/runtime/source/per-wave identities, lane-solo ratios, long/short/throughput/headroom. Optional separate timeline with stream mapping may diagnose changed DMA/overlap but cannot supply formal latency or sustained128K/roofline claims.
- [ ] Report engineering effect, remaining original-method uncertainty and defensible EB+ follow-up. Full CPU suite and exact service/foreign8files/runtime inventory verification; commit.
- [ ] One fresh whole-change reviewer with Review Focus verbatim; re-grade, one Important/Critical RED→GREEN fix pass, no rereview. Archive ledger/helpers/review/tests, delete only own scratch, retain branch/no merge/push.

Expected: fairer adapted-MorphX comparison and evidence-based next baseline, not a universal novelty claim.

Execution clarification: include the freshly built ABI-only libcuda_query_abi_v4.so as a qualification/diagnostic control for all three transport variants. The three-variant limit refers to the treatments, not this causal control. No old ABI/library/source is overwritten.

Hardware correction before timing: GPU1 is NUMA1 (CPU28-55/84-111), whereas author-default agent CPU0 is NUMA0 and overlaps the prior client set. Add only a checked/logged SMSCHED_AGENT_CPU host-binding override in fresh immutable local-v2 builds; keep original default when unset. Use agent36, unchanged server28-35, client40-43/96-99 (disjoint physical cores). Requalify code/four24K/full128K for all four new libraries before any efficacy measurement; preserve old builds/results. Original device kernels, progress profiler, schedule_2 and launch logic stay byte-identical outside the CPU binding helper.
