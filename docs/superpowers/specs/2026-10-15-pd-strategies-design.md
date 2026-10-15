# Simple admission and open-source PD mechanisms

## Intent and authorization

The user asks to actually verify simple scheduling strategies and whether an adapted open-source advanced PD mechanism solves the observed long/short interference. Their request accepts the preceding proposal: same-engine admission ablations first, then SGLang PD-Multiplexing. Continue directly in the existing research checkout; no additional approval or execution-method question is needed.

## Constraints

- Existing branch `research/ninfer-mtp-tuning`; preserve prior completed experiments and current uncommitted research notes.
- Remote root `/home/guokc/mypro/gnn-mount-sda1/InferX`; downloads, caches, overlays, builds and raw responses stay on its HDD. Source env.sh; use hf-mirror if weights are needed. Reuse existing 27B weights.
- GPU1 UUID `GPU-4d9bef9d-631c-bb36-3ff4-589d6a184189` only. Preserve GPU0 and its service. No global driver, permission or package changes.
- Use the existing owned-process controller and finally restoration of the original 256K/MTP7 service, actual process identity and both service manifests. Never stop an unverified or busy service.
- Main model Qwen3.8-27B, actual architecture Qwen3_5ForConditionalGeneration, same RedHat INT4 weights, C4, MTP0, FP8 full-attention KV and BF16 SSM. Do not replace with a smaller model or silently change precision.
- Existing frozen workloads: short histories about24K, long input about128K, arrival .5 shortRPS for48s and one long request at16s. Groups short-control/cold-low/resume-high. Long requests complete in full; retain errors/truncations/rejections.

## Mechanisms

Use the qualified SGLang0.5.19 CUDA12.9 environment as the common engine. Create independent copies of its Python package on HDD, preserving the existing singleton-sampler compatibility fix. Never edit installed wheel files or shared model configuration. Overlay manifests record original and changed file hashes and exact runtime arguments.

Simple policy A caps an ongoing chunked request at256 tokens when other requests are waiting, preserving the rest of the512-token round for new admission. Policy B uses128. Restore unused budget after invoking the original admission routine so the budget accounting and long-request progress stay valid. With no waiting requests, retain the original512. Never bypass capacity checks, lose an in-flight chunk or treat already cached input as new work. Use the original policy as the instrumented control. The cap guarantees some progress each eligible round; do not add an unbounded strict-short-priority queue.

Instrument original and modified PrefillAdder paths equally: request identity, cached-prefix length, remaining input, requested/consumed budget, queue length, admission result and wall time. Distinguish capacity rejection from exhausted round budget. Native cache reports are enabled in every new lane; API cached input and physical GPU recomputation remain distinct.

Adapt the actual open-source PD-Multiplexing scheduler and Green Context path, rather than implementing an unrelated scheduler under its name. Provide the Qwen text-only split-prefill entry point, preserving hidden and residual tensors between layer intervals and invoking final norm/logits exactly once. Initially support TP1/PP1/text-only/MTP0 only, fail closed outside that scope. Preserve per-request GDN state and the original model's layer execution conventions. Correct moved topology fields only when demonstrated by actual source/runtime failures. Validate SPLIT_PREFILL handling in hybrid/GDN metadata and backend dispatch before measuring.

Begin with the minimum useful PD-Multiplexing configuration (three stream groups, one P/D partition); sweep at most three valid splits if the mechanism qualifies. Record actual Green Context availability, layer intervals and P/D execution evidence; interface presence alone is not proof of GPU overlap. Include an unsplit/normal-forward correctness reference. Preserve source-backed compatibility failures separately from efficacy results.

## Adaptation findings carried into execution

Actual GPU tests showed the faithful PD path requires token chunking off and ordinary overlap off; lazy Mamba buffers therefore must use stock eager extra_buffer. Green Context96/32SM works, but whole-input prefill failed C4 warmup even at the minimum131328 pool (240MiB allocation,109.62MiBfree). Keep these failures separate. A separately labeled PD-Multiplexing+token-chunk variant retains the actual SM/layer scheduler while allowing only512-token chunks, stashing unfinished chunks and excluding them from decode using the normal scheduler's transition. Its post-merge metadata handoff waits on the prefill stream. A combined reserve128 variant is permitted, with an admission-only ablation at the same pool/buffer/overlap settings.

The initial simple implementation exposed a single-active-chunk invariant: original add_one_req can start another partial request when budget is reserved. The corrected policy only admits a candidate that fully fits the remaining round while another chunk is active; otherwise it returns the original OTHER result so the scheduler releases staged state. This is an adaptation correction, not evidence the simple strategy fails. Retain the failed first campaign.

## Capacity and fairness

First probe the original actual pool and PD-Multiplexing overhead. Main causal comparisons require the same explicit token-pool limit, actual concurrency, precision and chunk budget across control/simple/PD. If PD needs a smaller pool, qualify one common pool >=131328 tokens for all lanes and repeat their controls. Do not reuse old152639-token results as a matched control for a different pool. If no common valid pool exists, report that resource incompatibility and continue simple-policy comparisons at their common original pool. Do not lower context length to qualify PD.

## Gates and measurements

- Repository CPU tests first; adapter regressions must fail before implementation and pass afterward. Real installed SGLang CPU interface calls and GPU model gates supplement tests.
- Every GPU lane: startup resolved C4/context/token capacity, full code-quality check, four24K strict forced-tool warmups, and a complete128K strict-tool request before efficacy claims.
- One diagnostic repetition per simple cap; retain all. Complete three independent restarts per formal group for baseline and the selected simple policy. Do not treat requests within a wave as independent repetitions. Select using short latency together with long completion cost, not short latency alone.
- If PD qualifies, three repetitions of all three groups for the best validated partition; preliminary partition diagnostics are labeled separately. A failed correctness/capacity gate cannot be counted as PD performance.
- Primary: per-repeat short end-to-endP95, complete long latency, fixed48s completion counts, total completed-token throughput including drain. Also report cache/replay/retraction evidence, sampled memory and errors.
- Operational target inherited from the earlier study: shortP95 <=5s, all long requests correct and complete. Report whether long completion cost exceeds25% and throughput loses more than10% against matched control; these are study criteria, not measured enterprise requirements.
- An apparent win caused by rejecting/stalling long work, reduced context, changed pool/precision or incomplete responses is not a solution. If short-only control already violates5s, expose it.

## Deliverables and stopping conditions

Reproducible overlay preparation, tests, configuration, raw campaigns and analysis; a report directly answering which simple policies help and whether adapted PD solves the workload. Finish with verified restoration and one independent final review. No new KV-management project. HYPIC caching and unpublished DuetServe implementations are outside this direct cold-input PD experiment.

If a specialized backend cannot be made correct or fit after targeted source-backed attempts, retain the attempted adaptation and precise failure evidence, explicitly leave its efficacy unknown, and finish the runnable simple-policy study. Never replace a requested benchmark with a paper speedup.
