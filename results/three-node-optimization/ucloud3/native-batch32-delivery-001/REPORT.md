Native32 private build and separate Graph gate completed.

Source parent 9695/993 -> ec13896c/993, 18 changed and 975 unchanged files. Full portable CUDA13.2.51/GCC13.4/SM89 build succeeded; 389 compile commands, including145 CUDA commands, without object/archive replacement. Native32 ninfer-serve SHA256 is 3c1656c22584b91db09afaaec161bb2c5f909e379d59bb382fd7594a469afff0.

The original CTest run registered88:83 actually executed and passed,5 skipped,0 failed. The eec395 Qwen3.8 artifact was then supplied explicitly through NINFER_QWEN3_8_27B_WEIGHTS to the unchanged load-plan executable: exit0/reaped. The original skip log and LastTest.log are unchanged. This gives84 distinct tests executed successfully across the full run and supplement;4 missing-model cases remain skipped. The supplement binds the artifact; it does not perform model inference.

The separate native fold CUDA Graph test passed B16/B32, width4, nontrivial row permutation/high slots, mixed0..4 commits, actual256B rowtable/336B argument fields, two replays with same-stream state restoration, exact full-state repeat/guards and10 selected FP64 checks across layers0/47 and all48heads/128x128. This validates the fold ABI/Graph gate, not full-model B32 memory fit or SLO.

All three owners and all recorded children are actually missing after pidfd-waited/reaped child exits. GPU compute processes are empty at450W. The portable archive includes the exact service binary and48 resolved non-host shared libraries with per-file SHA; host glibc/driver dependencies remain explicit. Root can relay this local archive to ucloud1; no worker-to-worker contact occurred.

See DELIVERY-summary.json, CTest-summary.json, actual-terminal-receipt.json, original88-ctest.log, original88-LastTest.log, artifact-load-plan.log and the adjacent build/Graph directories for raw commands, SDK, link/dependency identities and receipts. No model performance/quality task was launched. The adjacent native32-C32-K0-performance-plan-001 contains review-only C32/B32/backendNone commands and memory/admission gates.
