# Open-source PD Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Actually adapt MorphX and deploy open-source alternatives against the current 27B prefill admission blockage, with measured improvements and costs.

**Architecture:** Reuse the frozen concurrency workload and streaming observations. Isolate each engine/build on HDD, run an owned GPU1 campaign with guaranteed original-service restoration, and compare raw arrivals and completed requests rather than vendor speedup claims.

**Tech Stack:** Python/aiohttp/pytest, NInfer and llama.cpp C++/CUDA, MorphX LLVM/CUDA hooks, isolated vLLM environment if compatible.

**Spec:** docs/superpowers/specs/2026-10-15-pd-baselines-design.md

## Global Constraints

- Current workspace/research branch; GPU1 only, GPU0 preserved.
- All server downloads/dependencies/builds/raw results on HDD /home/guokc/mypro/gnn-mount-sda1/InferX, source env.sh, HF_ENDPOINT=https://hf-mirror.com.
- No system package/driver/permission/global CUDA changes; no saved credentials.
- PID UID/exe/start_ticks/argv and service-idle checks before any stop; finally restore 256K/MTP7 service and both manifests.
- Main model stays Qwen3.8-27B; no smaller-model substitution for PD conclusions.
- No unsupported MorphX fallback labeled as effective MorphX. Actual kernel correctness and coverage required.
- Core groups C4, MTP0: 48-second control2.5RPS/cold0.5RPS/cold2.5RPS/resume2.5RPS;128K long arrives16s;24K short history;3 independent repetitions.

## Review Focus

- Restoring from interruption during shutdown/startup; test existing RestorableService contract with new controller.
- Engine context refers to total vs per-slot; validate actual128K acceptance before main run.
- Complete tools inside truncated/error responses; reuse strict protocol+nontruncation semantics.
- Native prefix counters absent or ambiguous; never infer verified resume from timing alone.
- MorphX loads but never launches patched kernels; require runtime coverage in case manifest.

---

### Task 1: Engine adapters and owned campaign support

**Files:** Create inferx/pd_baselines.py, tests/test_pd_baselines.py, scripts/run_pd_baselines.py; reuse inferx/concurrency.py and inferx/service_ownership.py.

**Interfaces:** Consumes run_load(endpoint, requests, output, mode, workers), summarize_load(records, span_s), RestorableService(saved, restore, manifest_paths, pre_stop_check). Produces engine_command(engine:dict, root:Path, run_dir:Path)->list[str], parse_metrics(text:str)->dict[str,float], validate_case(case:dict)->None and a raw campaign schema with engine/provenance/group/repetition/arrival/data/terminal restoration fields.

- [x] Write failing tests for128K per-slot admission, prohibitedGPU0 selection, explicitquantization/morphxcoverage fields and multi-label Prometheus parsing.
- [x] Run targeted tests and inspect failures; implement only required adapter/controller changes, preserve existing protocol semantics.
- [x] Pass tests for owned stop/restore and real HTTP tool/error retention; commit.

Expected: every candidate command has explicit GPU/environment/context/budget/model path and rejects incomplete configuration; controllers preserve all errors and restore on exceptions.

### Task 2: Build and validate actual candidates

**Files:** Create scripts/prepare_pd_baselines.py, scripts/build_morphx.sh and configs/pd-baselines.json; HDDsource/pd-baselines, pd-baselines/builds and provenance.json.

**Interfaces:** Consumes Task1 engine case schema; produces pinned sources/model manifests, buildlogs, usable binaries/environments and actual compatibility/coverage proof for Task3.

- [x] Pin MorphX full source and compiler/runtime, inspect compile interfaces, build for SM89 using isolated toolchains. Add required adapter tests before persistent implementation changes.
- [x] Execute patched kernel correctness/coverage probe on owned GPU1; inspect failures and adapt missing components as feasible, preserve failure logs.
- [x] Verify existing llama.cpp 27B path and seek modern vLLM INT4 path through hf-mirror; validate versions/driver/API/context and functional gates.
- [x] Freeze runnable candidate config/source/weight hashes, commit adapters and compatibility evidence.

Expected: actually runnable same27B candidates or reproducible attempted compatibility failures, never untested paper claims.

### Task 3: Run comparative long-short PD campaign

**Files:** HDD pd-baselines/raw; results/pd-baselines campaign/workload/config/provenance summaries.

**Interfaces:** Consumes pinned Task2 cases and frozen prior workload files. Produces immutable raw SSE/native/metrics/arrival data and terminal restored-service manifest.

- [ ] Recheck baseline tests and current service identity/idle before replacingGPU1.
- [ ] Run protocol/code gates and128K admission probes before comparativegroups; retain actualtoken counts and cache state.
- [ ] Run four coregroups with3 independentrepetitions on baseline and validcandidates. Iterate chunk budgets if necessary and compare matched configurations; add MTP after plain PD groups.
- [ ] Revalidate MorphX actualpatchedcoverage with/withoutsameadmissionfix; capture fairness/longcompletion and rejectrates.
- [ ] Restore originalservice, validate bothmanifests/argv256K/OpenAI+Anthropictools/GPU0health, commit runsummary.

Expected: all attemptedrequests included; capacity/startupfailures separate; controllerterminal andoriginalservicehealthy. Do not stop atshort smoke orsinglewave.

### Task 4: Export comparison and independent review

**Files:** Create scripts/analyze_pd_baselines.py, tests/test_pd_baselines_analysis.py; results/pd-baselines/{summary.json,groups.csv,requests.csv,audit.json}; docs/reports/2026-10-15-pd-baselines.md.

**Interfaces:** Consumes Task3 immutable raw; produces measuredabsolute/relative improvementswith independent repetitions, exclusions/capacity/quality/coverage labels and reproduction instructions.

- [ ] Test raw-only deterministic export, rejectedrequestretention, missingprefixcounterlabels, repeat aggregation and longcompletionfairness; implement analyzer.
- [ ] Run completeappropriate tests, twice export and comparehashes; write report that directlyanswersMorphX/alternatives/remainingbottleneck.
- [ ] Freshwhole-changereview; fix important findings with reproducingtests and finalverification; archiveexecutionrecord.

Expected: source-evidence-backed, reproducible answer to full userobjective, no scope substitution and no claim ofcompletion with unverifiedMorphX efficacy.
