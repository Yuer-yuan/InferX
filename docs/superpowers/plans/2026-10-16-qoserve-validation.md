# QoServe adaptation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline. No implementer delegation; one fresh whole-phase final review.

**Goal:** Adapt and measure actual public QoServe on the27B/4090 workload, compare it with a strong simple policy, and ground the continuing research decision.

**Architecture:** Extract pinned author policy mathematics; map ordering/chunk/relegation and full-input commitments into the existing native hybrid scheduler. Collect independent completed-engine-step latency profiles and caller QoS budgets; reuse owned lifecycle and frozen full-input traffic.

**Tech Stack:** Python/AST/pytest/stdlib unittest, pinned vLLM0.31/Torch2.13cu129, existing native27B INT4 kernels and owned restoration.

**Spec:** docs/superpowers/specs/2026-10-16-qoserve-validation-design.md

## Global Constraints

- Existing authorized `research/ninfer-mtp-tuning` checkout; preserve the exact8user-file SHA contract; no merge/push or changing frozen EB artifacts.
- GPU1 `GPU-4d9bef9d-631c-bb36-3ff4-589d6a184189` only; GPU0 stopped; no QoServe GPU until earlier owned campaigns are terminal/restored and current exact identity/argv/idle guards pass.
- All remote source/env/build/cache/raw on8876 HDD `/home/guokc/mypro/gnn-mount-sda1/InferX`; reuse27B/hf-mirror if needed; no global GPU/driver/clock changes.
- Same model revision91bd022d5b49442a868bc35008f6c21e1860edfa/native FP8 KV+BF16conv+FP32GDN/98blocks/block1568/C4/MTP0/eager/sync/context131328. QoServe paired configurations budget4096/cap0; simple control cap128.
- Public author commit72339db5b120614033b317880ad3a05a5e4c9415; preserve source math/branches, original0.008priority,0.995EMA,15output prediction,1.2guard and chunk candidates. Only explicitly recorded hardware/engine mappings.
- Full131049cold/131038resume and24×24557short .5RPS/48s,longat16s, code+4warm+100prime per fresh group. No failed/truncated long efficacy score; no future-output oracle.
- Exact original256K/MTP7/C1/rk4v4-e8 service plus BOTH manifests restored on every exit. Preserve all failures/source/provenance; no concurrent GPU controller.

## Review Focus

- Native whole-input checking is not physical reservation: full commitments/prefix sharing/CoW must not over-admit or mutate ownership merely by querying capacity.
- Paused or relegated requests retain queue/state ownership, original age and C4 accounting; no rejection, hidden request or silently enabled decoder preemption.
- Actual completed model-step geometry/latency must train the two-bin predictor; nominal sizes, stale cache counts, delayed worker sampling, censorship or unseen contexts cannot masquerade as calibration.
- QoS metadata and clocks must survive transport/resume, common budgets must be frozen without mixed outcomes, and SSE chunk gaps must not become per-token TBT.
- A measured lane must execute the intended policy, retain all late/completed requests, use independent waves/own matched solos, and restore the exact original service even on interruption.

---

### Task 1: Pinned author policy and source contracts

**Files:** Create `inferx/qoserve_policy.py`, `tests/test_qoserve_policy.py`; source evidence `results/qoserve-validation/author-source-v1.*`, supplemental author timing file/hash and compatibility notes.

**Interfaces:** Consume verified source capsule/manifest and timing source. Produce `load_author_policy(capsule:Path, manifest:dict, coefficients:dict|None=None)->type`, importable stdlib-only author methods; `policy_manifest(capsule:Path,manifest:dict,coefficients:dict|None=None)->dict` retains exact method/body SHA and explicit coefficient/native mappings. Missing deployment coefficients must not default to A100 silently.

- [ ] Write failing contracts for changed source bytes, source license, the original <=512 predictor boundary, exact normal/low-memory chunk searches, nonpositive slack, hybrid priority/drop ordering and cumulative TBT deadline with literal expected results.
- [ ] Observe correct RED, then extract original methods and inject only coefficient tables/native capacity access explicitly. Preserve original A100 tables for source-reference tests, reject unqualified deployment use.
- [ ] Test eager-relegation-once/requeue and throughput EMA against hand-derived source traces; record bridge mappings rather than claim whole old dispatcher unchanged.
- [ ] Run full local CPU suite (dedicated native checks remain separately gated); verify source manifest and user SHA contract; commit. No timed-server changes.

### Task 2: Native dispatcher and capacity commitments

**Files:** Create `inferx/qoserve_validation.py`, `scripts/prepare_qoserve_validation.py`, `tests/test_qoserve_native.py`; extend native engine configuration only where required by a failing launch contract.

**Interfaces:** Consume Task1 policy and pinned native scheduler/allocator source. Produce `prepare_adapter(author_root:Path,native_scheduler:Path,destination:Path)->dict` and importable `QoServeScheduler`, `CompatibleNativeScheduler`, `ObservedNativeScheduler`, with source-bound runtime/policy/token evidence. Adapter API records actual classes, cap, pool, precisions and profiling mode.

- [ ] Write real-native CPU contracts for full-input joint commitments, pure capacity queries, shared prefix/CoW, paused ownership/C4, abort/release, small128token progress and no-starvation/no silent decoder preemption. These run only after remote timed work ends.
- [ ] Write metadata/clock regressions: unknown-output prediction15, original arrival retained after resume, one wall/monotonic mapping, exact TBT deadline, late request requeues once and still completes. Do not derive model-token intervals from SSE.
- [ ] Observe actual RED in the pinned environment; implement ordering/eligibility/token-quota/full-commitment boundaries around native allocator/state handling. Compare native schedule AST modifications and extracted author hashes.
- [ ] Run all native contracts and full generic suite; prepare immutable HDD overlay, preserve source bytes and commit. No GPU efficacy from CPU contracts.

### Task 3: Independent real-model calibration and qualification

**Files:** Create `inferx/qoserve_profile.py`, `scripts/calibrate_qoserve_model.py`, `scripts/run_qoserve_validation.py`, `configs/qoserve-validation.json`, `tests/test_qoserve_profile.py`; calibration/runtime evidence under HDD `qoserve-validation`.

**Interfaces:** Consume Task2 actual scheduler/native Request/Output and shared `run_with_restoration`/`OwnedEngine`. Produce identity-bound per-engine-step rows, `fit_profile(rows,holdout,plan)->dict` and common caller contract file; actual full-model qualification before measured policy use.

- [ ] RED for geometry/identity/timing mismatches, bin edge513, too few eligible rows, invalid/nonfinite predictions, holdout contamination,5s-filter fallback provenance, impossible SLO oracle/transport fields and stale restoration prerequisites.
- [ ] Implement actual schedule-entry→native-update-exit timing with real SchedulerOutput features; no custom worker or extra CUDA synchronization. Run predeclared128/512/2048/2552×0/1/3decoder training rollouts and two disjoint all-candidate-cycling0/2decoder holdouts only after preceding owned GPU work terminates/restores.
- [ ] Preserve original five-feature LinearRegression/80:20/seed42/edge/cutoff rules and explicit bounded5s compatibility fallback; require >=100trainingrowsperbin, >=30heldoutrowsperbin/>=5perdeployedcandidate and >=95% disjoint1.2-guard coverage separatelyperbin/candidate. Keep censoring/excluded geometry/rank/error records.
- [ ] Obtain independent native24short serial +3cold +3resume calibration solos; freeze1.5×shortTTFT/modelTBT,1.25×longTTLT and4×noninteractiveTBT budgets. Only numeric QoS tags change payload; exact model inputs unchanged.
- [ ] Qualify code/24K/128K tools, actual policy ordering/chunk/relegation, native runtime/capacity and original restoration. Full suites, source/provenance inventory, commit. Do not fit formal outcomes.

### Task 4: Strong controls, formal comparisons and research judgment

**Files:** Create `scripts/compare_qoserve_model.py`, `scripts/analyze_qoserve_comparison.py`, `tests/test_qoserve_results.py`, `docs/reports/2026-10-16-qoserve-validation.md`; immutable configs/raw/exports/reproduction evidence.

**Interfaces:** Consume qualified profile/contracts/overlay. Reuse frozen workload content, strict SSE/tool/count checks and owned lifecycle. Produce independently normalized joint results, common-SLO goodput with the spec's exact engine/client predicates, practical joint target and a scoped research decision.

- [ ] RED for altered model payload, missing metadata/history/mode evidence, token-deadline versus SSE-gap confusion, tier1extraTBTrequirements, hidden/missing join IDs, incomplete/refused long, late-request exclusion, mismatched own-solo denominators, source/runtime tampering and pooled repetitions. Keep existing EB/SGLang normalizers intact.
- [ ] Run five bounded diagnostic settings (plain-nativeTruecap0, compatible-nativeTruecap0, plain-nativeTruecap128, plain-nativeFalsecap128, fullQoServe), each cold/resume complete. Freeze compatible native/strongestqualifiedsimple128/QoServe before3fresh mixed repeats and each lane's C4 serial-client solos. All groups receive code+4warm+100real prime.
- [ ] Export twice/byte-compare, independent-wave medians and own ratios plus absolute costs; separate engine-model-token TBT from client TTFT/e2e/chunk gaps. Record compatibility/contract/performance limitations.
- [ ] Assess all qualified SOTA/simple/PD evidence: whether material joint loss is avoidable and whether the intended practical gap remains. Prepare the next falsifiable mechanism design if justified; outer goal additionally requires its actual subsequent attempt.
- [ ] Full suites, actual exact restoration and user SHA checks, commit. One fresh whole-phase review with Review Focus verbatim, one Important/Critical TDD fix pass, no rereview; archive own ledger/rulings/deferred minors before own scratch deletion. Keep branch.
