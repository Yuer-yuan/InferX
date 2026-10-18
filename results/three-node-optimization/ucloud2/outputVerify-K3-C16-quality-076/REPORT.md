# 6632 / outputVerify / K3 original100 quality

Completed all100 original IFBench prompts (111instructions) through the existing050 EvalScope1.13.0 collector and verifiers. Client/nativeC16, K3, fusionON, outputVerify1, KV532480/context65536; defaultthinking/model sampling, max32768, naturalEOS. Same frozen6632/source3a7 binary and quantization routes as candidate performance; qualityKV differs from performanceKV139264.

| Fixed original prefix | Prompt strict | Prompt loose | Instruction strict | Instruction loose |
|---|---:|---:|---:|---:|
|30|29/30 (96.6667%)|29/30 (96.6667%)|32/33 (96.9697%)|32/33 (96.9697%)|
|60|57/60 (95%)|57/60 (95%)|61/64 (95.3125%)|61/64 (95.3125%)|
|100|84/100 (84%)|87/100 (87%)|95/111 (85.5856%)|98/111 (88.2883%)|

All100 responses stop naturally:0length,0empty,0HTTP/generation failure,100attempts. All16 strict-prompt failures remain scored and preserved. Existing fullCPU scorer independently reran the original verifiers and matched original cached values; prompt-cluster10000seed42 CIs are in CPU-score-001/summary.json. No new evaluation/check was added after Root's final direction.

This is a fixed-first100 candidate observation, not the published full benchmark recipe reproduction or proof of an isolated quantization effect. Actual startupKV532480, free5,492,178,944B, Graph144,703,488B. Existing scheduler trace observed real B16/T64 execution; the separate u1 native qualification recorded94 actualGraph output nodes across47layers. No per-prompt physical kernel count claimed.

Original client naturally completed; the owned service was stopped and reaped and the owner finished. Service cleanup receipt and GPU-released.json are retained. The job's completed=99 is its last polling sample; final prefix100 and the full100 scorer establish completed100. Root directs u2 GPU to remain idle; no repeated076/baseline.

Compact delivery: delivery/compact-review.tgz, SHA25699dea0d84f12bde7eb3f58be4486dc03934dc855f8243d27f5610dc84b819d51,3,508,613B,68files. Full raw HTTP attempts, native predictions/reviews and scheduler trace remain under the same remote experiment directory.
