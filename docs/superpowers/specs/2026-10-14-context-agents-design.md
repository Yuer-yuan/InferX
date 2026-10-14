# Qwen3.8 larger context and local coding agents

The user explicitly chose increasing context first, then using OpenCode, Pi and Claude Code. Continue the authorized inline workflow in this work directory and research branch; no additional approval gate. This extends the completed NInfer/MTP deployment rather than restarting its engine build.

## Goal and constraints

Leave a tested NInfer context of at least128K if it fits and passes functional gates, and working local wrappers for all three agent clients. Preserve the64K rollback. Use GPU1 only and preserve GPU0PID2504785/port18080. All remote downloads, packages, caches, configurations, task checkouts and raw logs stay under `/home/guokc/mypro/gnn-mount-sda1/InferX` on HDD. Models continue to use hf-mirror; reuse verified weights and pinned NInfer source. Bind API to loopback18081. No system/global environment modifications, paid cloud inference, push or merge.

## Context choice (builder calls)

Start with128K(131072), INT8 KV, MTP7, prefill2048,C1. If memory permits compare256K(262144) with supported4-bit `rk4v4-e8` KV, MTP7. Select the largest stable candidate passing short-code gates and long-input lookup checks, keeping cold-prefill costs visible. Failure falls back to smaller capacity/draft or existing64K, with exact evidence. No claim that capacity proves general long-context accuracy. Use near-full inputs leaving>=8K output headroom, lookup questions at early/middle/late positions, and small code tests. Capture1-second GPU memory telemetry through requests and post-completion settle; require>=1024MiB observed headroom. Retain the existing short-prompt heldout to compare against64K. Cold requests use engine-level prefix disabling; warm deployment is separately tested.

## Agent integration

Install pinned isolated Node22.23.3 and OpenCode1.19.0 on HDD. Use current Pi package `@earendil-works/pi-coding-agent`1.2.0 (upstream renamed), not the older namespace by accident. Reuse installed Claude Code2.1.92; no account-login changes. Isolate client config/state/cache directories without repurposing HOME. ModelIDqwen38-27b and advertised context/output budgets match selected runtime. Native NInfer OpenAI and Anthropic routes are preferred; add a narrow compatibility adapter only if a real client failure proves necessary. Prevent cloud fallback by pinning local models for auxiliary calls too. No optional plugin/MCP installation. Thinkingoff/greedy remain the first integration baseline; no agent-quality parity claim from previous throughput.

## Tasks and validation

Use private copies of existing, pinned public ExecRepoBench repositories on HDD. Select three development tasks from different repositories deterministically before client runs, replace the target block with its syntax-valid failing ablation, then ask each client to inspect the repository, fix production code and run the original tests. All three receive the same task and fresh copies. External verification uses pristine tests, so modifying tests cannot count as success. Time limit600s/client/task; capture raw CLI events, elapsed time, server request IDs/token usage, tool activity, test outcome and code diff. One client request at a time to avoid confounding the C1 server. A failed benchmark task is an honest quality result; client connectivity is demonstrated by actual read/edit/execute tool activity, not model self-report. Distinguish protocol/infrastructure failures from wrong edits and timeouts.

Deliver reproducible launchers/configs, compact context/client results, Chinese report with commands and limits. Keep complete conversations and diffs on HDD. Run meaningful regression tests for helpers, full existing suite, one fresh final review. Existing128/256K tests may be long; communicate progress and never terminate another user's process.
