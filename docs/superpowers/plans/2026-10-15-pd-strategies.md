# Simple Admission and Open-source PD Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure matched simple admission policies and adapt actual SGLang PD-Multiplexing against the27B/128K interference workload.

**Architecture:** Reuse the owned GPU1 campaign/controller and frozen workloads. Isolate SGLang Python-source overlays on HDD, test admission accounting and hybrid-model split execution, then compare raw responses with full correctness/capacity gates.

**Tech Stack:** Python/pytest, SGLang0.5.19/Torch2.13/CUDA12.9, existing INT4 model, CUDA Green Context.

**Spec:** docs/superpowers/specs/2026-10-15-pd-strategies-design.md

## Global Constraints

- Existing research/ninfer-mtp-tuning checkout; retain prior data and notes.
- GPU1 only; HDD root /home/guokc/mypro/gnn-mount-sda1/InferX; source env.sh; hf-mirror for any models.
- Verified original process/idle checks and finally restore both256K/MTP7 manifests; no driver/global-package changes.
- Same27B INT4/C4/MTP0/FP8KV/BF16SSM; full131328 context budget; frozen24Kshort/128Klong/.5RPS/48s/16sarrival groups.
- Same actual token-pool limit across causal comparisons; gate correctness and full long completion before throughput comparisons.

## Review Focus

- Failed admission after budget reservation must restore accounting without dropping the ongoing chunk or corrupting GDN state.
- A cache miss may make a short candidate too large for remaining budget; record it rather than attributing every failure to the same cause.
- Split execution must reset residual at interval0 and preserve it thereafter, normalize/project exactly once, and reject unsupported model execution modes.
- CUDA Green Context initialization and duplicate attention workspaces may invalidate the original pool; startup arguments alone are not capacity proof.
- Interrupted/failed adapted lanes must leave no child processes and must restore the exact original service and manifests.

---

### Task 1: Instrumented simple-policy overlays and controller configuration

**Files:** Create inferx/sglang_pd.py, scripts/prepare_pd_strategies.py, tests/test_sglang_pd.py, configs/pd-strategies.json; modify inferx/pd_baselines.py and tests/test_pd_baselines.py.

**Interfaces:** `cap_chunk_budget(adder, req, original, cap)` invokes original admission with a temporary per-request cap and restores remaining round budget; `prepare_overlay(source:Path, destination:Path, policy:str, cap:int|None)->dict` copies and patches source under a new owned directory and writes hashes. Engine adapter emits native cache-report/max-total-tokens/PD-config flags from explicit fields.

- [x] Write failing tests for budget conservation, no-wait control, exhausted budget, exceptions, original capacity rejection, and valid128/256 caps; test explicit flags, HDD-only PD-config and no accidental GPU override.
- [x] Run RED; implement smallest policy hooks and source-anchored atomic overlay preparation with equal instrumentation for control.
- [x] GREEN targeted and full repository suite; CPU-validate real installed PrefillAdder accounting and preserve source hashes; commit.

Expected: instrumented independent overlays, verified unchanged no-wait behavior and no installed-package mutations.

### Task 2: Actual PD-Multiplexing Qwen adaptation and qualification

**Files:** Extend inferx/sglang_pd.py, scripts/prepare_pd_strategies.py, tests/test_sglang_pd.py, configs/pd-strategies.json; record results/pd-strategies/compatibility on HDD and locally.

**Interfaces:** `qwen_split_prefill(self,input_ids,positions,forward_batch,split_interval,input_embeds=None)` matches ModelRunner's real split entry and text-only wrapper hierarchy. Overlay contains narrowly scoped Qwen/backend/topology corrections as demonstrated necessary; manifests enumerate all deviations from upstream.

- [x] Write failing interval/residual/interleaving/final-logits/unsupported-mode tests, run RED, implement adapter, run GREEN.
- [x] Use real installed ModelRunner CPU entry to validate call compatibility, then owned GPU1 Green Context and full27B code/tool/full128K probes. Record native split/stream evidence and every failure.
- [x] If whole-input prefill cannot fit, qualify a separately labeled512-token-chunk PD adapter with stash/exclude/stream handoff and its reserve-only ablation. Resolve demonstrated hybrid dispatch/topology/capacity issues with targeted regressions; qualify matched actual pool or document why none fits. Preserve exact precision/context and restore service.
- [x] Run full repository suite and commit adapter plus actual qualification evidence.

Expected: a correctness/capacity-qualified actual PD path, or reproducible adapted failure with efficacy explicitly unknown.

### Task 3: Matched comparative campaigns

**Files:** configs/pd-strategies.json; HDD pd-baselines/raw with unique pd-strategies run names; results/pd-strategies campaign summaries.

**Interfaces:** Uses existing `scripts.run_pd_baselines` case schema, per-group independent restart and `scripts.analyze_pd_baselines` strict raw export. Records frozen overlay source hashes in case metadata and raw scheduler event files/server logs.

- [ ] Verify active service identity/idle and frozen hashes; run baseline plus128/256 queue-only and128/64 responsive simple diagnostic groups with full code/tool gates.
- [ ] Complete three repetitions for matched baseline and best simple policy; if PD qualifies, probe at most three SM partitions and complete best configuration's three repetitions. Retain selection diagnostics and failed lanes.
- [ ] Verify long completion, strict-tool counts, actual capacity, raw arrival counts, restore original service and both manifests, and commit summaries.

Expected: no conclusions from failed/truncated/refused long work, no claims that small-sampleP95 is a productionP99 or that API cache counts are physical recomputation.

### Task 4: Analyze, review and preserve evidence

**Files:** Create docs/reports/2026-10-15-pd-strategies.md and results/pd-strategies final comparison/export; add only necessary analyzer tests/code if new fields are required.

**Interfaces:** Strict raw exports and source manifests produce per-repeat control/simple/PD comparisons, long completion cost, throughput and operational SLO judgments with explicit exclusions.

- [ ] Export deterministically twice and compare hashes; verify original raw/event/provenance hashes; report improvements and remaining unknowns with all repetitions retained.
- [ ] Run full suite and fresh whole-change review; one Important/Critical fix pass under RED→GREEN, then suite and final service verification.
- [ ] Archive execution ledger/artifacts, close plan, keep research branch in place and leave reviewable files; no merge/push.

Expected: direct measured answer, source-backed compatibility boundaries, and independently reviewed restoration/reproducibility.
