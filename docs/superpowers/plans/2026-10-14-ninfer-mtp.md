# NInfer and MTP Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy and measure the fastest stable tested Qwen3.8-27B configuration on a single RTX 4090.

**Architecture:** An HTTP benchmark records complete raw responses and derives comparable metrics. Independent launchers use GPU 1 and HDD-resident tools/models. Staged experiments choose a default after quality and long-context checks.

**Tech Stack:** Python 3.12, pytest, HTTP/SSE, llama.cpp CUDA 12.6, NInfer SM89 with isolated CUDA >=12.8, Bash.

**Spec:** `docs/superpowers/specs/2026-10-14-ninfer-mtp-design.md`

## Global Constraints

- Use GPU 1 on `vlm-server`; preserve the existing GPU 0 service.
- All remote models, downloads, dependencies, builds, caches and raw logs belong under `/home/guokc/mypro/gnn-mount-sda1/InferX` on HDD.
- Model downloads go through `https://hf-mirror.com`; pin revisions and verify hashes.
- Do not modify the system driver, global CUDA installations, existing conda environments or other users' processes.
- Bind deployment APIs to `127.0.0.1`; leave one tested winner available on GPU 1, with a reproducible launcher and rollback instructions.
- Keep local work on `research/ninfer-mtp-tuning` in the existing work directory; no push or merge.

## Review Focus

- A disconnect or API error cannot produce a successful performance record (Task 1 test).
- SSE chunks and missing token usage cannot be counted as tokens (Task 1 test).
- Warm-prefix reuse cannot silently enter cold-prompt comparisons (Task 1 test and Task 2 log audit).
- Another process on GPU 1 cannot be killed by a stale PID file (Task 2 manual ownership check).
- Long contexts and speculative decoding cannot trade broken outputs for reported speed (Task 3 runtime gates).

### Task 1: Comparable HTTP benchmark

**Files:** Create `inferx/tuning.py`, `scripts/bench_tuning.py`, `tests/test_tuning.py`.

**Interfaces:** Produces `summarize_response(events: list[dict], wall_s: float, ttft_s: float | None, cache_mode: str) -> dict` and a CLI accepting endpoint, model, corpus and output paths. Successful summaries require terminal completion, positive completion tokens and finite positive wall time. Missing usage remains unavailable, never inferred from chunk counts. Raw JSON retains errors, usage, engine timings and texts.

- [x] Write tests for valid throughput, API errors, missing usage, non-finite times and mislabeled cold cache reuse; run them and observe failures.
- [x] Implement metrics and the streaming CLI with deterministic prompts/sampling; preserve complete raw responses on HDD.
- [x] Run the whole pytest suite on the existing remote Python environment; commit the harness.

### Task 2: Engines and staged development sweep

**Files:** Create `scripts/start_tuned_server.sh`, `scripts/build_ninfer.sh`, model preparation instructions/script and `results/tuning/provenance.json`.

**Interfaces:** Consumes Task 1 CLI; produces pinned engine/model manifests, recorded launch configurations and development JSONL. No production library API changes.

- [x] Record hardware, GPU occupancy and source revisions; build NInfer dependencies in an isolated HDD prefix and obtain a verified model via the mirror.
- [x] Start single-slot llama.cpp baseline with 65,536 context and test native MTP. Run draft-length sweep 0/2/4/6/8, then promising KV/microbatch combinations, inspecting quality and errors after each stage.
- [x] Run NInfer baseline/MTP staged sweeps with comparable corpus; audit cache behavior and quantify acceptance when exposed.
- [x] Verify launchers with shell syntax checks and actual runtime requests. Record build limitations instead of inventing NInfer results. Commit launchers and provenance.

### Task 3: Held-out validation and deployment

**Files:** Create `results/tuning/summary.json`, `results/tuning/trials.jsonl`, `docs/reports/2026-10-14-ninfer-mtp.md`; update `README.md`.

**Interfaces:** Consumes Task 2 candidates and Task 1 metrics. Produces measured winner, reproducible invocation and report.

- [x] Repeat finalists three times on held-out code/reasoning/prose; retain individual observations and dispersion. Compare MTP/no-MTP output parity and runnable code checks.
- [x] Probe cold/warm 2K/8K, 32K and near-64K fit, concurrency 2/4 where supported, and Anthropic tool use. Report all failures and truncations separately.
- [x] Leave a quality-gated winner on GPU 1; inspect process ownership before replacing only our tuning processes. Verify health, real completion, free disk, and GPU 0 preservation.
- [ ] Write results/report, run full pytest plus artifact validation, commit. Run one fresh final review, fix important findings once, preserve branch without push/merge.
