# Three-node optimization campaign

Coordinator thread: `01a143ef-ca3d-7382-af14-0d35ff68f29d`.

The user explicitly authorizes autonomous experiments through three existing Codex sessions, one per machine, performance first and quality afterward. Use the installed brainstorming/writing-plans skills where useful; do not install their other workflow dependencies or wait for an execution-method decision already supplied by the user.

## Ownership

| Machine | Existing session | Initial route |
|---|---|---|
| ucloud1 (`ssh ucloud`) | `01a1453d-cff3-7682-b649-9fa653728c3f` | NInfer scheduling and joint SLO |
| ucloud2 (`ssh ucloud2`) | `01a14532-a38c-7461-86ad-d994417b070e` | SGLang admission/backend, efficient quantized batching/MTP |
| ucloud3 (`ssh ucloud3`) | `01a14534-9c2f-7051-9962-2eb38d954cde` | NInfer INT8 prefill compute paths |

The previous baseline coordinator owns current ucloud1/ucloud2 jobs until explicit handoff. New workers may inspect and build using CPU while awaiting handoff, but must not launch GPU work or stop another session's process. ucloud3 is available after its current managed NInfer service identity is checked.

## Benchmark contract

- Model stays Qwen3.8-27B; record artifacts, precision, source and binary identity. A changed quantization or activation profile is a labeled configuration, not a scheduler-only comparison.
- Single GPU per run; original competition runner, exact cold 8192 input/512 output, greedy/no-thinking performance profile, ignore_eos, no hidden prefix reuse.
- TTFT P95 <= 10 s and TPOT P95 <= 40 ms must both pass. Record throughput and errors too. Do not optimize only one latency metric.
- Same-machine baseline for each causal comparison. Pilots can use smaller samples, clearly labeled; promote candidates to >=100 measured requests and independent seeds/repetitions. C3/C4/C6/C8 and then higher concurrency as evidence supports; do not stop merely at C4.
- Keep raw client and native service records, actual resolved arguments, input hashes, errors and failed cases. No fabricated SSE/token accounting or benchmark-specific cached answers.
- Performance candidates get short correctness smoke checks, not an expensive full quality run before useful performance screening. Expanded paired IFBench and uncertainty assessment follow selection.
- The current 100-prompt NInfer quality score is not proof of acceptable quality or a reason to silently change the task.

## Worker outputs

Write only `results/three-node-optimization/ucloudN/`, `docs/reports/2026-10-16-ucloudN-optimization.md`, and your own isolated engine source/build paths unless the coordinator explicitly assigns another file. Avoid shared README/submission edits and broad commits.

Start by writing `status.json` atomically with: `machine`, `thread_id`, `phase`, `updated_at`, `current_experiment`, `live_handles`, `last_result`, `next_action`, and `blocker`. A live handle includes process/session/job ID, host, start identity, log path and how to poll it; do not put credentials in artifacts. Update at each meaningful phase and result.

Keep `experiments.jsonl` with one record per completed or failed case: configuration identity, protocol/sample size/seed, TTFT/TPOT/throughput, error and cache evidence, raw paths and qualification status. Never overwrite an earlier run directory.

Communicate only with the coordinator through these artifacts and replies in your session. Do not message other workers or start additional agents/sessions. Continue actual experiment work after writing an update; report a real missing resource if blocked. On an observation timeout, re-poll the original live handle before deciding to restart.
