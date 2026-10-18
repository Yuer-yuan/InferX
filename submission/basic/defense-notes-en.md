# InferX — five-minute English talk

Matches the 31-slide [English presentation](presentation/InferX-EN.pdf). Total budget: 300 seconds including pointing and pauses. The live demonstration has a separate [three-minute script](demo.md). The Chinese counterpart is [here](defense-notes.md).

## 1. Cover — 5s

“We are Team Ni Shuo Sha Dou Dui. InferX optimizes single-GPU inference.”

## 2. Project overview — 10s

“The service supports 14 concurrent requests. Both tests meet SLOs at 306.04–309.95 tokens per second. The first 100 IFBench prompts yield 85 strict and 88 loose passes.”

Quality uses C13/KV532480; performance uses client14/native13/KV121856. These are separate protocols, not one combined run.

## 3. Problems and goals — 14s

“Two bottlenecks limit concurrency: prefill delays decode on the shared GPU, and matrix execution efficiency varies by shape. We target higher concurrency within TTFT P95 of 10 seconds and TPOT P95 of 40 milliseconds.”

The sketches are mechanisms, not measured durations or an attribution of bottleneck shares.

## 4. System design — 13s

“Two optimizations are supported by engineering integration: request-progress scheduling, INT4 computation, and concurrent MTP state handling. Actual progress feeds the scheduler. NInfer provides the base engine and MTP.”

## 5. Why ninfer-4090 fits this project — 12s

“With the model and GPU fixed, ninfer-4090 supplies matching low-bit and MTP paths. We can modify execution, scheduling and state together.”

Point to model–GPU specialization, low-bit path reuse, and execution/state control. These are project-specific integration advantages, not evidence of vLLM/SGLang performance or concurrency limits. Attribute the base engine, existing quantization and MTP to upstream; explain our extensions in the next seven slides.

## 6. Design I: chunking and scheduling — 12s

“Four prefill chunks create scheduling opportunities. The scheduler estimates remaining decode work and compares deadline slack while retaining the prefill deadline guard.”

Chunks are non-preemptible. The max-estimator patch preserves FCFS, deadline formulas and the batch cost model; the diagram is illustrative, not a measured GPU timeline.

## 7. Design I: commit history and estimator — 12s

“Actual committed tokens and decode rounds refine each request’s remaining-work estimate. The matched C16 comparison reduces TPOT by 6.23 percent, still above target.”

`g` includes the first prefill token, hence `g−1`. Increment `d` only after non-cancelled output-policy commits. EOS/stop and budget clipping use the actual committed count; new request objects start from zero even when reusing a slot. Remaining rounds times current batch round cost feeds completion slack. Fast requests cannot lower the existing batch estimate.

Sources: [exact two-header patch](../../results/three-node-optimization/ucloud3/request-yield-max-to079-handoff-001/minimal.patch), [executor commit order](../../results/three-node-optimization/ucloud3/request-yield-max-to079-handoff-001/delta/src/runtime/engine/concurrent_executor.h), [formal comparison](../../results/three-node-optimization/ucloud3/request-yield-max-C16-official128-recovery002/REPORT.md). 2362→837f, C16/K3/KV139264, cold8192/512, 128 measured+8 warmup, zero errors, one sequential comparison. No chunking ablation, fresh task-quality result or sustained-capacity claim.

## 8. Design I: bounded decode membership — 14s

“Serve nine urgent decode rows only when omitted requests retain enough slack. Then require two actual full-ready decode rounds before another reduced batch.”

Require B10–14, at least one negative slack, omitted-row slack≥three observed full-round costs, valid observations and stable row ordering; otherwise serve all ready. This redistributes service rather than establishing a pure host speedup or aggregate efficiency gain. The diagram is policy order, not measured time.

Sources: [selector patch](../../results/three-node-optimization/ucloud1/bounded-decode-C12-098/control/evidence/build100-selector.patch), [executor integration](../../results/three-node-optimization/ucloud1/bounded-decode-C12-098/control/evidence/build100-executor-integration.patch).

## 9. Design II: H512/G512 and INT4 — 12s

“Hadamard transforms and GPTQ prepare weights offline. Online INT4 matrix instructions accumulate within groups, then scale and restore BF16 outputs.”

H512 is a local 512-dimensional Hadamard transform; G512 groups 512 input channels. Before quantization, orthogonality preserves `W′x′=Wx`; quantization still introduces error. Preserve nonlinear, residual and BF16-rounding semantics. KV and recurrent state are not claimed to be INT4.

Sources: [GPTQ025](../../results/three-node-optimization/ucloud2/true-int4-all-layer-GPTQ-025/REPORT.md), [S4 arithmetic implementation](../../results/three-node-optimization/ucloud2/true-int4-community-implementation-032/private_gptq_s4_pipe4.cuh). Base GPTQ/Hadamard methods are upstream; our work covers caches, CUDA operators and model integration.

## 10. Design II: cache and route integration — 12s

“We integrate FFN, input and output projection caches, then verify real weight addresses, shapes, workspace and Graph dependencies. Whole-operation timing includes all stages.”

Calibration uses four fixed 8192-token WikiText training requests and two held out, not IFBench answers. Held-out linear error improves for64/64 gate and63/64 down; down layer3 worsens. These are operator diagnostics, not task quality.

Build6632 enables Verify T33–128 in47 GDN output layers; T1–32, L0 and16 full-attention outputs keep original routes. T counts token rows. Native checks inspect47 selected weight addresses, quantization→GEMM dependencies and disjoint input/residual/scratch bounds. See [076 integration report](../../results/three-node-optimization/ucloud1/output047-verify-integration-076/REPORT.md).

Historical reference, removed from the slide:305.91→363.58 tok/s is a separate historical whole-build C32 progression, with TPOT101.16ms still failing. It is neither build 6632 performance nor isolated kernel gain. Performance9270 and quality6632 remain separate.

## 11. Implementation: batch and resource expansion — 12s

“MTP concurrency is engineering adaptation: extend verification batches, row-to-slot maps and buffers; size KV and state; update CUDA Graph parameters.”

The native expansion changes18 files:9 reachable execution/ABI files,3 diagnostics and6 tests. Resources follow configured concurrency, not a blanket32 allocation. The completed Graph gate coversB16/B32, width4, mixed0–4 commits, high slots, permuted rows, a256-byte row table and336-byte arguments. Full-model capacity additionally depends on KV/state/workspace memory; the fixed8192/512 C16/C32 tests use KV139264/278528. Execution capacity is not SLO capacity.

Sources: [source rationale](../../results/three-node-optimization/ucloud3/native-batch32-source-proposal-001/REPORT.md), [completed build and Graph gate](../../results/three-node-optimization/ucloud3/native-batch32-delivery-001/REPORT.md). Pending work in the earlier rationale is superseded only by the actual completion report.

## 12. Implementation: resolution and state commit — 12s

“Resolve stop conditions and budgets before committing model state and progress, then publish output. Tests cover partial commits, cancellation, high slots and Graph replay.”

Executor order: `output.preview` fills per-row accepted/terminal/cancelled; `resolve_pending_batch` handles model state; non-cancelled rows update generated tokens, budget and committed decode rounds; `output.commit_preview` publishes. Final committed length is not merely the number of accepted MTP drafts.

The B16/B32 Graph gate covers mixed0–4 commits and high-slot replays. A separate later state-fusion gate has2013 matching records:1776 state hashes,168 execution/commit,68 prefill and1 snapshot;86 tests pass,4 skip for missing fixtures. It covers cancel/reset, slots15/31, both banks and resident checkpoint restore; disk restore is untested. See [lifecycle report](../../results/three-node-optimization/ucloud3/gdn-fusion-native-lifecycle-001/REPORT.md). These results retain their original build scope.

## 13. Results: six build 100 measurements — 14s

“These current measurements cover 4 to 14 concurrent requests, with separate node curves. Both tests at concurrency 14 meet the targets at 306.04 and 309.95 tokens per second.”

Same e4ed1bf4/c976692d, ordinaryK3, cold8192/512, original128+8+1. u1 supplies C12/13; u3 supplies C4/8/14. KV=max(65536,8704×clientC). These are not same-node concurrency-only effects; C4/8 cannot trigger the B10–14 selector. An invalid C4/KV34816 startup was rejected before loading or requests; the measured C4 run uses legal KV65536 and the rejected attempt is retained.

## 14. Results: native-capacity boundary — 12s

“Both tests at concurrency 14 meet the targets; concurrency 15 and 16 exceed TPOT. We admit at most 13 active requests, including excess-request waiting in TTFT.”

Different nodes prevent a one-slot causal comparison. Build 100 is independently rebuilt, without proven GPU-cubin equality to080. The [campaign data](presentation/selected-campaign.json) retain all six main measurements and complete failed boundaries. C15/native 13 and C16/native 16 fail TPOT at 41.881 and 51.221 ms, with 309.662 and 316.743 tok/s. C16/native 13 is a separate admission diagnostic: both SLOs fail (13.275 s / 42.328 ms).

## 15. Results: build 100 instruction following — 10s

“The original script selects the first 100 IFBench prompts, containing 111 instruction constraints. There are 85 strict and 88 loose passes. Strict scoring requires every constraint of a prompt to pass.”

Same e4ed/c976, flag1, ordinary K3; quality uses C13, KV532480 and max_tokens32768. Only max_tokens is overridden in each request; actual default temperature/top_k/top_p are 1/20/0.95, with server seed42. Original-verifier scores for all 100 rows match the raw reviews. Prompt-cluster bootstrap: 10,000 resamples, seed42. 95% intervals: prompt strict78–92%, loose81–94%; instruction strict80–92.381%, loose83.036–94.495%. This fixed first-100 observation is not a full-IFBench result; no paired OFF arm establishes lossless quantization or causal gain.

Historical build6632 scored84/87 at C16/KV532480 and remains historical evidence. [selected-quality.json](presentation/selected-quality.json) binds all raw sources.

## 16. Results: same-build pass and fail cases — 14s

“Key80 satisfies indentation, equal-length sentences and unique words. Key88 uses the wrong list marker, so strict and loose prompt scores both fail.”

Both displayed finals are complete original evaluation105 outputs. Key80 has0/4/8 leading spaces,28 trimmed characters per sentence including punctuation and internal spaces, and15 distinct words. Both instructions pass. Key88 passes one of two instructions (0.5); natural stop, no truncation. The example explains formatting, not mathematical correctness. Historical076 key83 is not reused: key83 actually passes in105.

## 17. Obtain and verify inputs — 6s

“Obtain complete source and pinned inputs; verify hashes in a fresh deployment directory.”

Verify the frozen v1 archive and its file manifest. The complete archive manifest covers 1212 files, including 1023 engine sources. Model weights, old quantized caches and installed environments are separate inputs. The 40 MB source kit does not contain all weights. See the [complete workflow](../track-a/reproduction-en.md) for downloads, dependencies and fixed IFBench cache preparation.

## 18. Regenerate quantized caches — 8s

“Rebuild FFN, input and output caches in dependency order, using fixed FP8 and four training plus two held-out inputs.”

Run 116 fits project caches from fixed FP8 logical weights: all 64 layers for 025 and 041, and the selected 47 layers for 047, totaling 478 serving files. Four training and two held-out inputs each contain 8192 raw token IDs; IFBench is not calibration data. Acquire the public .ninfer base separately. Each stage uses freshly generated predecessors and receives a new manifest/recipe. Consult the [delivery index](../DELIVERY.md) for actual stages, loading and evaluation receipts. New caches cannot inherit the following 664 old-cache scores.

## 19. Build from source — 8s

“Normally rebuild all 1,023 source files. Seven CPU tests pass; the new executable receives its own identity.”

The fresh directory normally compiles source aggregate c976692d… in Release/SM89 with CUDA 13.2.51 and GCC 13.4. All 1023 sources match before and after; seven CPU tests pass and 55 dynamic dependencies are recorded. Executable 664cfbb9… has a separate recipe. The actual run reused host dependencies and did not reinstall the OS or every environment. See the [build receipt](../../results/three-node-optimization/ucloud3/selected100-single-node-reproduction-20261018/build-delivery/build-job.json).

## 20. Start the performance profile — 6s

“Start the rebuilt performance profile: client 14, native 13, KV121856 and ordinary MTP K3.”

The same new recipe binds service and client. C14 denotes client requests in flight; native 13 is engine active capacity. KV121856, context65536, prefill2048 and ordinary MTP K3 are fixed. The 19 environment flags retain build 100 settings. launch.json records the real process, executable, arguments and environment. Verify identity and wait for model readiness.

## 21. Run the original performance benchmark — 4s

“Retain all 128 measurements, eight warmups and one probe.”

Retain all 137 payloads: 128 measured, eight warmup and one probe. Each uses cold input 8192 and output 512. Keep initial measured requests and admission waiting. Verify canonical payload identities; use linear P95 over per-request TTFT/TPOT and the original profiling window for throughput.

## 22. Reproduced performance — 4s

“The rebuilt executable passes both SLOs, delivering 304.90 tokens per second.”

The [original summary](../../results/three-node-optimization/ucloud3/selected100-single-node-reproduction-20261018/performance-delivery/performance-C14-001/summary.json) supplies this excerpt without renaming fields, changing nesting or rounding numbers. Latency is in milliseconds. There is no meets_slo field to copy. Values 9896.41735465 and 38.28584995440313 are below 10000/40; throughput is 65536/214.944959168=304.89665937584215 tok/s. This independent single run uses the new executable and old caches; historical build 100 curves remain separate.

## 23. Evaluate the original 100 prompts — 4s

“Stop performance, then run the separate 100-prompt quality evaluation.”

Stop the performance service normally and confirm the process and socket are gone. Use a separate quality service: client/native13, KV532480, K3, the original first 100 prompts and 111 instructions. Override only max_tokens32768; retain default thinking, sampling and natural EOS. Attempt 001 timed out before readiness, generating zero prompts. r2 only adds read-only /models waiting for the same verified process. Attempt 002 took three checks over 32.947901 seconds, with zero generation probes and no retried evaluated prompts.

## 24. Reproduced quality — 4s

“Strict 90, loose 96; one truncated empty answer remains counted.”

The [original score JSON](../../results/three-node-optimization/ucloud3/selected100-single-node-reproduction-20261018/quality-delivery/quality100-002/evaluation/summary.json) retains its accuracy nesting, Chinese status and score/num fields. Prompt strict 90/100, loose 96/100; instruction strict 100/111, loose 106/111. The raw summary rounds the latter ratios to 0.9009/0.955. Exact numerators come from all reviews. Key88 has both an empty final and a 32768-token length stop and remains in the 100-prompt denominator; request errors are zero. Historical 100/105 scores 85/88 remain separate, with no paired repetition attributing the difference to an optimization.

## 25. Exception handling — 8s

“Bound startup, memory and request failures. Preserve logs, configurations and every failed outcome.”

This table describes handling procedures, not automatic recovery from every failure.

## 26. Reliability — 10s

“Check API behavior, concurrent state and process ownership. A separate state-fusion experiment validates 2,013 matching records; long-term stability still needs repeated load.”

Record count is not test count, and the evidence applies to that experiment's build.

## 27. Observability — 10s

“Each row shows one measured request: engine admission, first token and completion. Lines show request lifetimes, not exclusive GPU execution.”

The source is scheduler.jsonl from the same 085/C16 diagnostic run. Clock anchors uniquely join all128 measured requests to client records. The figure shows the first8 by engine admission, request IDs11–18, each finishing with512 tokens. Admission is not HTTP arrival; lines are lifetimes, not exclusive GPU time. Original events, matching errors, source hashes and a Chrome Trace export are retained. Rebuild with`submission/tooling/inferx-submission/scripts/build_request_trace.py`.

## 28. Bottleneck: prefill interference — 8s main talk

“Competing prefill contributes 64.6 percent of generation tail latency in this diagnostic run. Execution cost and scheduling both matter.”

Discussion evidence: 085/C16/K5, traceON, original128, cold8192/512. All128 requests uniquely match1825 nonoverlapping synchronized executor spans. The original P95 contributors, weighted0.35/0.65, decompose49.398ms into31.934ms competing prefill,17.452ms shared decode, and0.012ms residual. Median own prefill is0.909s. The timeline is actual session_000052. These are host wall spans, not exclusive GPU kernel time or SM efficiency, and the diagnostic does not replace formal traceOFF results.

## 29. Resources: serving memory — 5s main talk

“Measured serving memory stays at 31.97 GiB; weights and caches occupy 21.53 GiB.”

Discussion evidence:173 samples at about1Hz. KV4.66GiB; non-KV recurrent/sequence state4.91GiB; workspace and observed Graph0.47GiB; unclassified residual0.40GiB. Fixed startup allocations explain components; device samples provide totals. This is not dynamic per-allocation tracing. Graph uses138MiB, not its5GiB budget. CUDA-free memory after startup is15.43GiB.

## 30. Concurrency tradeoff: request admission — 8s main talk

“Fourteen concurrent requests need not all enter the engine at once. Admission waiting spends TTFT headroom to limit generation contention; the selected configuration meets both SLOs twice.”

Discussion evidence: all three runs use build100, client concurrency14, K3/KV121856, cold8192/512,128 measured+8 warmup+1 probe, and identical full payloads. ucloud1/native14: TTFT9.724916s, TPOT41.022342ms,306.063528tok/s; TPOT fails. ucloud3/native13 run1:9.893643s/39.929242ms/306.038803tok/s; run2:9.821661s/39.141798ms/309.951742tok/s; both pass.

Thirteen slots limit active requests, not the exact membership of every GPU batch. The client keeps14 requests in flight; excess requests wait for admission. Client TTFT includes that waiting, prefill and first-token production; TPOT measures generation after the first token. Reducing active contention spends TTFT headroom. These observations come from different nodes and do not isolate the admission-control effect; do not turn them into a percentage gain. The first passing run has only0.071ms TPOT margin, and two passes do not establish sustained capacity.

The three pages retain the existing speaking budget; methods and exact numbers are for questions. See [build100 evidence](../track-a/data/selected-campaign.json), [diagnostic and memory evidence](../track-a/data/cost-evidence.json), and [analysis](../track-a/performance-analysis.md), which retains the historical085/086/088 same-node scheduling comparison.

## 31. Conclusions and limitations — 5s

“We integrated INT4 inference, progress scheduling and concurrent MTP. Concurrency 14 meets the targets, with 85 strict passes on the first 100 quality prompts. Next are better prefill and scheduling, repeated tests, and full-dataset evaluation.”

Cross-node/native-capacity differences do not establish causal gain; the historical 92.09 baseline is not a controlled speedup denominator.

## Evidence boundaries for questions

- Performance: build 100, client 14/native 13, two passing runs with small margins; native 14 failed. Historical 9270 is retained separately.
- Quality: build 100 evaluation 105, C13/K3/KV532480; strict 85/100 and loose 88/100, with one empty final retained. Historical 6632 remains separate.
- Baseline 100+8 and extended 128+8 are separate protocols; no unified speedup ratio.
- Original methods and engine have upstream attribution; local operator gains are not end-to-end gains.
- Independent rebuild 664 passed C14 using the old caches and scored 90/96 on the original 100. New-cache run 116 and fresh dependency installation have separate receipts and scopes.


## Historical results removed from the main talk

### Historical reference: Results: latency-constrained concurrency — 6s

“Historical C12 passes; C16 and C32 execute successfully but exceed latency targets.”

C12/16 share a node; C32 uses an equivalent node. KV104448/139264/278528; cold8192/512, K3, 128+8, zero errors. C16/C32 throughput307.15/363.58tok/s.


### Historical reference: Concurrency curves — 6s

“Throughput rises with concurrency, while C14 TPOT jumps to 55.55 milliseconds and fails.”

Same actual payload including salt; cold8192/512,128 measured+8 warmup,K3. Native slots and KV grow together:104448/113152/121856. One run each, not sustained capacity or a fixed-resource concurrency-only effect. TTFT9.60/9.69/9.83s; TPOT38.75/38.85/55.55ms; aggregate284.96/290.86/296.22tok/s.


Removing old slides 13–14 frees 12 seconds for transitions and pointing. MTP concurrency is engineering adaptation; intervals, full conditions and error counts are discussion material. Newly integrated reproduction pages are preserved, for 31 slides total.
