# MorphX and Advanced Single-GPU Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement inline task-by-task; one fresh final review, no implementer delegation.

**Goal:** Determine which actual open-source mechanisms can improve the measured27B/128K short/long interference relative to isolated execution, prioritizing MorphX.

**Architecture:** Reuse owned GPU1 controller, frozen payloads and existing pinned toolchains. Audit candidate mechanisms, isolate targeted runtime corrections, qualify numerical/model behavior, then benchmark qualified configurations and preserve unqualified outcomes.

**Tech Stack:** Python/pytest, CUDA12.8/LLVM18 C++ MorphX port, existingNInfer/SGLang0.5.19, public BulletServe source.

**Spec:** docs/superpowers/specs/2026-10-15-sota-validation-design.md

## Global Constraints

- Existing research/ninfer-mtp-tuning checkout; preserve dirty prior report and solo-comparison files. New files/results use sota-validation scope.
- GPU1 only, GPU0 intentionally stopped; HDD /home/guokc/mypro/gnn-mount-sda1/InferX; source env.sh, reuse27B; hf-mirror for any required weights.
- Owned identity/idle/stopped-GPU0 guard and signal-safe exact256K/MTP7/both-manifest restoration; no global driver/package/permission changes.
- Full128K/C4/MTP0/eager; exact frozen24Kshort/.5RPS48s/longat16s; per-engine matched weights/precision/capacities. Do not compare different quantizations as isolated scheduler effects.
- Custom compilation and missing integration are explicitly authorized; preserve original mechanism, pursue source-backed corrections, and do not stop merely at absent native hybrid support. Only qualified complete long workloads yield efficacy statistics.

## Review Focus

- Cooperative/extended launch attributes, explicit streams and per-thread variants must preserve native CUDA launch semantics rather than silently discard attributes.
- Native-bypassed or unmodified kernels and an unchanged admission policy must never be mislabeled full MorphXPD coverage.
- Isolated baselines must match actual payload/cache/config observations; one solo wave is not three independent repetitions, and API cached counters are not physical work.
- Old libraries/overlays/raw data and the separately created solo supplement must remain unchanged; foreign or busy processes must never be stopped.
- Numerical/model failures, incompatible hybrid state or duplicated memory must be retained as unqualified outcomes, not presented as throughput or SOTA failure.

---

### Task 1: Pin candidates and audit the new isolated reference

**Files:** New results/sota-validation/source-audit.json, isolated-reference-audit.json and HDD source cache; no prior report edits.

**Interfaces:** Existing solo scope/audit/comparison source hashes, frozen raw specifications and startup manifests; pinned author git commits and exact files.

- [x] Rehash all solo audit/scope sources and verify78strict responses, originalC4/precision/config, serial dispatch and actual cache observations; verify current original service/idle/GPU0state read-only.
- [x] Pin/read MorphXPD orchestrator, launch dispatch and compiler/runtime contract; inspect Bullet hybrid dispatch, INT4, CUDA/MPS and weights/pools. Save source bytes/SHA/URLs; classify direct runnable, bounded-adaptation, or unqualified with concrete reasons.
- [x] Record scope/cross-engine/cache caveats; run existing full CPU suite in HDD repo-venv and commit audit artifacts.

Expected: supported candidate selection and audited solo reference, without treating public logs as this hardware's result.

### Task 2: MorphX launch semantics and actual correctness qualification

**Files:** New inferx/morphx_validation.py, scripts/prepare_morphx_validation.py, tests/test_morphx_validation.py, tests/fixtures/morphx_cooperative.cu only as needed; HDD immutable runtime/source/build/raw.

**Interfaces:** A source-anchored launch-attribute correction preserves native attributed launches and records native/transformed coverage; preparation rejects unexpected source or overwrite. Reuse run_owned_command/run_with_restoration, original idle guard and existing affectedCTest binaries; add a narrow qualifier launcher if existing flags cannot express an immutable library.

- [x] Write RED for source-anchor rejection, no old-file overwrite, attribute-preserving dispatch and explicit coverage; implement smallest demonstrated correction, GREEN. Do not rewrite unaffected runtime behavior.
- [x] Build fresh named runtime onHDD; run actual independent cooperative integer oracle and original affected GDN/GQA/Graph numerical tests with compiler-matched native/original-port/corrected-port. Inspect outputs and preserve all failures. Investigate root-backed corrections and custom builds with regressions; no speculative successful-probe claims.
- [x] If numerical qualification passes, run hook-off and hook-on full27B code/four24Ktools/128K gates with actual coverage. Establish actual author P/D orchestration path before any efficacy claim; otherwise preserve exact unqualified gate and explanatory source/runtime evidence.
- [x] Full CPU suite, exact restoration and hashes; commit actual adaptation/probe evidence.

Expected: qualified actual MorphXPD path or demonstrated compatibility boundary with efficacy unknown, never an invalid performance comparison.

### Task 3: Bounded second-baseline qualification and runnable comparisons

**Files:** New configs/sota-validation.json, results/sota-validation/qualification and HDD unique campaigns; extend only necessary existing controller/schema interfaces with RED→GREEN.

**Interfaces:** Task1 candidate audit + Task2 qualified execution/coverage. Same-engine control/advanced cases and lane-specific solo baselines use exact frozen requests and observed runtime capacities. No papers-only method receives synthetic measurements.

- [x] For Bullet, perform actual CPU/GPU compatibility/model gates and adapt missing hybrid/state paths where feasible; missing native class alone is not a terminal boundary. No stockGPU0script/globalMPS invocation.
- [x] Run one complete diagnostic per qualified advanced setting, at most3settings/mechanism; choose using short/long/throughput jointly. Freeze selection before3fresh mixed repeats and matched isolated diagnostic. Keep oldSGPD/simple references separate byengine/config.
- [x] If no new advanced path qualifies, finish all justified custom-build/adaptation qualification work and retain an explicit unknown efficacy result rather than invent another framework; preserve already-qualifiedSGPD evidence as the advanced measured reference.
- [x] Audit complete longs/tools/runtime/source/input hashes; verify exact original service and manifests, run full CPU suite and commit measured/failed evidence.

Expected: actual qualified comparison data where possible and reproducible boundaries elsewhere; no conclusion from truncated, refused or stalled long work.

### Task 4: Normalize slowdown, report and preserve

**Files:** New docs/reports/2026-10-15-sota-validation.md and results/sota-validation comparison/export/reproduction artifacts.

**Interfaces:** Strict raw exports produce continuous short/long slowdown and throughput/headroom/coverage, retaining independent repeat identities and excluded lanes.

- [ ] Export qualified raw twice, verify determinism/source hashes, calculate lane-specific solo/mixed ratios and joint criteria; if no qualified new lane, deterministic compatibility summaries retain failures and existing-reference provenance instead.
- [ ] Write direct measured answer with candidate/code/hardware distinctions and all unknowns; verify reproduction commands and source/binary inventories.
- [ ] Full suite and commit; recordTask4 completion before final review.

Expected: evidence-based answer to whether tested SOTA solves this workload, without a universal novelty or optimality claim.

## Final review and finish

- [ ] One fresh whole-change read-only reviewer; re-grade findings/declined items, oneImportant/Critical RED→GREEN fix pass, fullsuite and fresh service verification.
- [ ] Archive ledger/helpers/review/tests, close plan, delete only this scratch, retain branch/no merge/push; preserve prior dirty/user files.
