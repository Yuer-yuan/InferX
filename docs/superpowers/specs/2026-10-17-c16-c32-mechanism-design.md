# C16 and C32 mechanism experiments

> **Current instruction, 2026-10-17:** The user explicitly requests fundamental mechanism improvements and as much research and implementation as useful. The three existing sessions have autonomous control of their assigned nodes. This supersedes historical CPU-only, one-probe and Root-approval restrictions in earlier evidence. Continue private implementation, numerical validation, profiling, original performance evaluation and necessary separate accuracy evaluation without waiting for permission. No new sessions or commits.

The user sets C16 as an intermediate target and explicitly permits continued work toward C32. Success means measured single-GPU concurrency under the existing official joint SLO, with credible numerical fidelity. C16/C32 are targets, not current results. Use only the three existing sessions; Root coordinates and relays artifacts. No new sessions, agents, peer messages, or commits.

The latest final-defaults original100+8 result supports C7: TTFT P95 8.660524s, average TPOT P95 36.123044ms, 177.630995 output tok/s. C8 fails at 9.756216s/41.360977ms despite 178.730759 output tok/s. NInfer default-thinking/default-sampling first100 strict prompt accuracy is 84%, independently reproduced with the original verifiers. Finish and archive the already running vLLM thinking512 arm only. Cancel the other reference accuracy arms and SGLang collection, preserving partials and failures. Do not start further reference accuracy runs.

## Fixed contract

- One modified 49,140 MiB RTX4090, SM89, 128 SMs, original 450W limit per measured service; three machines run independent experiments.
- Official performance inputs: cold 8192 random nonspecial token IDs, exactly512 output tokens, temperature0, ignore_eos=true, original runner d80df390cf6d1d50129796b31fdd7f4d1e95d0043d3b942d47ac582118702fb0.
- TTFT P95 <=10s and request-average TPOT P95 <=40ms, TPOT=(E2E-TTFT)/511. Original100+8 includes all initial requests; diagnostics remain separate. Preserve fresh confirmation and sustained-stress gates for promising candidates.
- Canonical source9695ce1f5162bafba9d8538217c7693653e02a0e3e814e8e923d896696dfe21a and model eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e remain the controls.
- Numerical oracle, stream ordering, stable Graph addresses, sequence identity, causal attention, GDN state and MTP rollback remain mandatory. A changed quantization representation needs its own fidelity qualification; the existing84% score cannot qualify it.

## Evidence and capacity

[Source-derived planning receipt](../../../results/three-node-optimization/control/C16-C32-source-capacity-and-roofline-planning.json) rehashes12 relevant source files against the canonical993 manifest. Native concurrency currently caps at8 in public types, GDN operator validation, replay records, replay validation and a fixed row[8] kernel parameter array. HTTP concurrency and native GPU batch are different: both larger native batches and rotation of smaller batches are candidate designs.

The actual configuration is64 layers,16 full-attention layers,48 GDN layers,4 KV heads of256 dimensions, and48 GDN value heads with128x128 FP32 state. Per-request current/checkpoint state uses two slots. Extrapolating only selected terms at native B16/B32 gives model-file-plus-KV-plus-GDN-plus-replay planning subtotals26.34/35.72 GiB. These are neither measured resident allocations nor fit proofs. Actual page entitlements, transient MTP pages, arenas, Graphs, sampling and fragmentation remain to be accounted for.

Using30.44s as an assumed mean E2E planning point yields269/538 output tok/s at C16/C32. These are engineering references: two marginal P95 limits do not bound mean E2E. Existing profiling reports about1.588s prefill and1.293s amortized decode GPU service per request. Under that reference C32 budget0.95125s/request, even free decode requires substantially faster prefill. No hardware counter measurement of HBM traffic, SM utilization or achieved occupancy is available.

Actual artifact metadata contains24,350,556,160 decoder dense matrix elements and14,386,462,720 packed bytes. One8192 prompt requires about399T logical dense operations before attention/GDN/quantization/unpacking/LM-head/draft work. Stock4090 nominal peaks are660.6 dense INT8 TOPS,165.2 BF16 TFLOPS with FP32 accumulation, and1008 GB/s. They bound idealized arithmetic/traffic, not this board's achieved performance. [NVIDIA architecture source](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf).

## Three experiment boundaries

**ucloud1: layerwise mixed execution.** Read the actual Program/BatchFrame/Prefill/Verify interfaces. Propose the smallest mixed dense computation that reuses weights and reduces measured total service. Keep per-sequence attention/GDN semantics. Count actual tiles, padding and waves: concatenating token columns is not free decode. Q4 Prefill A8 versus Verify A16 is a numerical difference; Q5 Verify A8 currently applies only to17..32 columns. Evaluate C16/C32 and K=0/1/3/7. [SARATHI](https://arxiv.org/abs/2308.16369) and [Sarathi-Serve](https://arxiv.org/abs/2403.02310) motivate shared prefill/decode batches but do not predict gains for these groupwise kernels.

**ucloud2: full-K integer contraction / QoQ.** Reuse existing CPU row-W8/token-A8 evidence. Determine whether an INT8 full-K dot can remove expensive groupwise FP32 rescaling and register pressure. Distinguish current group64 FP16 scales from QoQ integer group scales. Count conversion, temporary buffers, packed weights, actual full operator movement, and peak model memory. Existing synthetic outlier activation error15.46% versus group64 1.64% prevents assuming fidelity. Propose actual activation calibration hooks before any approximate production route. [QServe primary source](https://arxiv.org/html/2405.04532v3), [implementation](https://github.com/mit-han-lab/omniserve).

**ucloud3: batch/MTP/state expansion.** Audit all native-eight limits, parameter ABI, lane mapping, Graph profiles, KV reservations and GDN rollback. Compare native B8/16/32 and K=0/1/3/7 using actual target/draft token shapes and kernel dispatch. Optimize committed tokens per measured service cost rather than acceptance alone. Cost-aware ragged verification is relevant, but DSpark's trained block drafter/confidence head are not supplied by this model. [SGLang DSpark mechanism](https://www.lmsys.org/blog/2026-07-06-dspark-sglang/).

## Execution and stopping decisions

Current authorization covers autonomous implementation and experiments in the three existing sessions. Root reviews evidence concurrently, relays source/results and prevents overlapping assignments. Workers select and recover experiments without per-step approval. Product promotion requires actual operator fidelity, whole-model correctness and official request metrics.

The completed Q5 C32 microkernel passes136 numerical cases and improves low-T operator latency6.7–8.9%, but slows Prefill T256 by73–75% and T2048 by122–135%. Preserve it as a low-T candidate; defer integration while pursuing the larger service-demand problem. Do not use its occupancy ceiling as a speed claim or extend it uniformly to Prefill.

Preserve all negative results, original failed controllers, canceled partials and exact process identities. Cancel only owned processes. New mechanism experiments use isolated directories, explicit source/binary identities and unchanged controls. Select the next GPU experiment by plausible whole-request savings, implementation size, precision risk and measured demand; unsupported routes can be rejected independently.

## Measured update and stronger mechanisms

Native B16/B32 now builds and runs. Root independently verified original100+8: C16/K1 gives17.830s TTFT P95,91.233ms TPOT P95 and186.334tok/s; C32/K0 gives49.485s,158.162ms and188.729tok/s. Neither meets SLO. Capacity expansion alone does not solve service demand.

Actual initial arrival spans are7.373ms/21.833ms. With100 observations, at most five TTFT values may exceed10s as a necessary condition for linear P95. Thus at least11/27 initial requests require first tokens by their own deadlines. At nominal INT8 peak, their decoder dense work alone costs6.643s/16.306s. C32 therefore requires a materially different arithmetic or algorithm under this conventional dense-compute model; decode-only or scheduling changes cannot remove the initial prefill requirement. This is not a universal impossibility theorem. See `control/C16-C32-initial-burst-observed-compute-bound.json`.

- **ucloud1:** Implement shared linear execution for prefill/decode or resumable layer execution that reduces measured nondecode interference. Use actual MTP yield and dispatch geometry. The rejected mixed-down tail kernel does not settle a different full shared-A8 mechanism. Preserve independent causal/state metadata and actual input precision.
- **ucloud2:** Finish the C128 full-K model experiment, then prioritize genuine INT4 arithmetic with explicit activation/outlier handling. Atom, APEX4 and QuaRot are source references; packed W4 expanded into INT8 does not reach the INT4 compute ceiling. Select quantization granularity by complete operator cost and actual activation fidelity. Independent calibration text and the original separate task evaluation determine whether an arithmetic change is usable.
- **ucloud3:** Reduce GDN state traffic and high-batch decode cost. Start with FP16 persistent state and FP32 updates if the observed ranges permit, then consider calibrated mixed FP16/INT8 state. DAMP distinguishes repeatedly quantized recurrent state from low-bit linear weights; its findings do not qualify our model. Validate long recurrence, prefill handoff, MTP rollback, lane reuse and actual model accuracy.

Root supplies pinned primary-source packets and exact raw audits. Workers may replace these proposals with stronger evidence-backed mechanisms; paper speedups are not predicted gains for this workload.
