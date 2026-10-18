# GQA high-batch split cap: frozen operator result

Original screening result: **FAIL** (both T4 shapes required >=1.2x). Root later explicitly approved low-cost engineering integration as a separate post-screen decision; the original threshold/result is unchanged.

| B | query width | old Graph wholeOp ms | capped Graph wholeOp ms | speedup |
|---|---:|---:|---:|---:|
| 16 | 1 | 0.399360001 | 0.380928010 | 1.048387x |
| 16 | 4 | 0.494592011 | 0.423936009 | 1.166667x |
| 32 | 1 | 0.736256003 | 0.694271982 | 1.060472x |
| 32 | 4 | 0.911872000 | 0.747008026 | 1.220699x |

All 12 paired cycles improved in each of the eight eager/Graph shape rows. All1536 fixed ABBA samples retained. WholeOp includes fused KV append+attention partial and reducer; full-cache reset and one256MiB flush are outside events. Histories8192+per-row0..511; each B row has its own physical KV pages, fragmented mapping, no shared-history alias. Same450W device.

Numerical gate: 5cases×2arms×(Graph twice+eager)=30 passed, full13,565,952 output-element checks (includes inactive zeros), full physical K/V codes and scale bits exact against independent CPU codec, immutable inputs/positions/table/valid, all allocation guards. Full FP64 attention and each live row use unchanged originalI8 criterion L2<=.00315 and maxerr<=.0011+.0022*maxabsreference. B16/B32,T1/T4 plus B32T4 mixed valid0..4,context8191/8197/8701,reverse table rows and fragmented pages. Changed split rounding is **not bitwise-equivalent between arms**; within each arm eager/Graph/reset replays are exact.

Saved-array CPU recompute passes; max full-output L2=0.00295906358, max per-row L2=0.00310484955.

Fresh-process cold capture proof is first B16/T4 candidate Graph before any attention eager launch. Later cases capture before their own eager calls but are not fresh processes; the original JSONfield is clarified in graph-cold-scope-clarification.json. No GPU rerun.

Implementation: unmodified current i8 kernel18b5b9… with both CTA barriers; only external launchcapacity changes64→8(B16)/4(B32), matching cap in reducer. Original conservative workspace retained. Real captured2kernel/1edge perop; T4regs128/staticSMEM43648/local80B, T1regs128/staticSMEM38464/local16B; identical code in both arms. These resource figures do not establish occupancy or DRAM saturation.

Parent local1013 files independently matched f592cb8d…; 27 compiled dependencies individually pinned. Standalone binary5d9a5f3e8ba9b2e596cf4c49cce02fc38c52c08715c4f27bdd1ed2ac7f5c4a4a; CUDA13.2.51/GCC13.4/SM89. Original dc99 model binary unchanged, no original archives rebuilt/swapped.

All12 child steps exit0/pidfd waited/reaped; owner and all12 handles independently missing after completion, GPU1MiB/no compute apps/450W. Full113-file delivery archive32,226,964B SHA98ecb6b29260fd66f7072d93388771fde8bb0c13222f7e187381a4f3d44781f7; all112 content-manifest hashes verified locally. Complete output arrays/oracles,1536samples and resource/command/terminal evidence in delivery/. Remote identical raw remains /workspace/InferX/runs/gqa-highbatch-split-probe-001.

Preserved failures: premature launcher before SCP completion (no owner/GPU), read-only observer syntax failure, local NumPy dependency query (stdlib verifier used instead). No numerical or runtime failure occurred. No model/SLO/task-quality inference follows from this operator result; historical~20%decode kernel share only supports a small potential increment, not an end-to-end22%gain.
