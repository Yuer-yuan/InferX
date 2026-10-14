# Qwen3.8-27B: NInfer and MTP on one RTX 4090

The user authorized deployment and performance tuning with NInfer and MTP. Optimize interactive coding on one 24 GiB RTX 4090, targeting 65,536 total context tokens and one active conversation. Two and four simultaneous requests are secondary measurements, not the primary objective. This is an empirical search for the best verified configuration in the tested space, not a claim of global optimality.

## Constraints

- Use GPU 1 on `vlm-server`; preserve the existing GPU 0 service.
- All remote models, downloads, dependencies, builds, caches and raw logs belong under `/home/guokc/mypro/gnn-mount-sda1/InferX` on HDD.
- Model downloads go through `https://hf-mirror.com`; pin revisions and verify hashes.
- Do not modify the system driver, global CUDA installations, existing conda environments or other users' processes.
- Bind deployment APIs to `127.0.0.1`; leave one tested winner available on GPU 1, with a reproducible launcher and rollback instructions.
- Keep local work on `research/ninfer-mtp-tuning` in the existing work directory; no push or merge.

## Method

Establish a fresh single-slot llama.cpp Q4_K_M baseline. The prior eight-slot pilot is background evidence only. Test integrated MTP with maximum draft lengths 2, 4, 6 and 8; stage KV precision and microbatch tuning after finding a useful draft length. Keep model, prompt rendering, greedy sampling, output budget and clock/power settings constant for paired comparisons. Do not lock clocks or change power limits.

Build the SM89 NInfer fork on Linux using an isolated CUDA 12.8 toolchain if required. Inspect its source, runtime dependencies and model format before downloading another model. Prefer a pinned community artifact with the MTP head included; compare its quality separately because its quantization differs from GGUF. Use only native SM89 code on the installed driver; report exact build/runtime blockers if a compatible path fails.

The harness measures actual completion tokens, wall time, time to first output, engine timings when available, cache tokens and draft acceptance. Store complete responses and errors. Warm-prefix results must be labeled separately from fresh-prefix results. Novel prompts differ near the start; engine cache reset/disable is preferred for truly cold tests. SSE chunks are not tokens. Failed, empty, truncated-quality and non-finite measurements must not become successful throughput rows.

Use a small fixed development corpus spanning code generation, reasoning and explanatory prose, plus independently specified held-out prompts. Run a staged sweep rather than every combination. Repeat finalists at least three times, include ~2K/8K prompt tests and 32K/near-64K fit probes. Report context-fit probes separately from typical throughput. Include a multi-turn tool-call roundtrip and runnable coding checks. A speed winner must pass these smoke quality gates; this is not proof of equal benchmark accuracy. Compare greedy token/output parity for MTP within the same engine and quantization; flag any difference before promoting a configuration. Investigate differences rather than declaring speculation inherently lossless.

Select the configuration with the best median interactive decode throughput among stable candidates passing quality gates, with TTFT, memory and long-context limits visible. If NInfer wins only on cached/repetitive text, preserve that distinction. It is acceptable to retain a reliable llama.cpp default and a separately labeled experimental NInfer launcher if the latter fails quality or API checks.

## Deliverables

Versioned launch/build/download instructions, an auditable benchmark harness, compact result JSONL/summary and a Chinese report with provenance, measured speedups and remaining limitations. Raw server logs and full benchmark responses stay on HDD. Finish with a fresh independent code review and relevant tests, preserving the branch and the chosen running service.
