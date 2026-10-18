# Current combined-version C32: complete, joint SLO FAIL

The single authorized original-runner run completed **128 measured + 8 warmup + 1 probe**, seed20261016, cold8192/512, client/native32, K3, KV278528, chunk2048, 450W, traceOFF. All startup requests remain included. Original runner d80 is unchanged. The legacy manifest label `PILOT` is a controller label; the actual measured count is128.

| Metric | Actual result | SLO |
|---|---:|---:|
| TTFT P95 | 23.040530955s | ≤10s, FAIL |
| TPOT P95 | 92.109024613ms | ≤40ms, FAIL |
| Phase output throughput | 374.046139510tok/s | — |
| Measured success/errors | 128/0 | — |

[Independent local summary](summary.json) and [all128 request indices](measured-request-index.jsonl) reproduce the original parser's percentiles and phase throughput. TPOT is `(E2E−TTFT)/511`. The throughput uses the original profiling-phase span, not the distinct `benchmark_duration` metric.

All137 native requests have prompt/computed8192, cache0 and full_reset;136 emit512 tokens and the sole probe emits1. All136 client records split into8 warmup and128 profiling. Actual137 payload SHA after removing only cache_salt is `62f94272362623a9d776cf99711ab67411957a1d7799b7987583b2570df4b021`; the measured and warmup cohorts individually match the preceding u3 dc99 C16 run. Different concurrency and hardware history prevent interpreting this as a GQA-only causal comparison.

The failure is not confined to startup. Initial32 contain21 TTFT and28 TPOT violations; later96 contain8 TTFT and84 TPOT violations. TTFT P95 interpolates arrival ranks25/27, weights.35/.65; TPOT P95 interpolates ranks5/75 with the same weights. No trace was collected in this formal run and no GPU stage attribution is inferred from these client values.

The binary is `8b20de4daa8c0556ab9e2b69f60f5ea6ae3b1598680ee6f04a33d193b7e3a1c7`; source1013 aggregate is `61a15f2f08d8efd9f185c4edabbad6370104584f522446537c765617b94a045e`. The dc99 parent environment, including all existing S4 caches and down053, was verified against the actual running process. Only the two-file GQA capacity/test patch was added to its source. This is the current combined-version C32 result; old u1/9270 C32 is historical reference, not a same-host dc99 control.

The isolated full portable CUDA13.2.51/GCC13.4/SM89 build completed without object/archive replacement. Actual CTest registration88 comprises **84 executed PASS and4 missing-variant skips**, zero failures. Original FP64 criteria remain unchanged. Added public Graph checks cover B8/15/16/32, mixed valid0..4, fragmented/permuted rows, real partial→reduce edges and repeat after state restoration. Original standalone long-context numeric gates remain available in [the operator report](../gqa-highbatch-split-probe-001/REPORT.md); its original performance SCREEN_FAIL remains unchanged. Root separately authorized engineering integration.

The native model-owned Graph gate completed32 health prefills,9 real rounds at B8/16/32,168 row observations and6 Graph definitions. It identifies114 actual GQA partials, including57 nodes in capture-only8K envelope graphs. It checks installed profiles, pointer/dataflow/workspace and state invariants. These health inputs and capture-only graphs are not official8192 workloads or task quality. The first external harness compile failure (missing private target include path) and its path-only recovery are preserved.

Formal startup reports KV payload10,004,594,688B, sequence arena20,224,094,208B, workspace capacity361,906,176B, observed Graph287,309,824B and free memory9,418,047,488B. These are actual allocation/startup facts, not occupancy or DRAM counters.

All15 recorded build/harness/formal handles were independently observed missing after recorded pidfd wait/reap receipts. The formal owner445557, driver445573, server445576 and telemetry445572 are terminal; GPU is idle1MiB at450W. No rerun, seed rescue, quality collection or additional performance case was started.

The [local delivery](delivery/local-delivery.json) verifies99 content-file hashes. Complete raw client/input/native, build commands/SDK/archives identity, CTest logs, native Graph records and first failure are under `delivery/content/`. Archive `delivery/complete-raw.tar.gz` is8,733,154B, SHA256 `fe33c6d0625d540a51ebad0f3f0f941db3354dedae5da5821a88a73bccfe6376`; the identical remote archive is `/workspace/InferX/runs/gqa-highbatch-C32-official128-001-delivery.tar.gz`. [The audit script](audit_delivery.py) checks raw records without model calls.
