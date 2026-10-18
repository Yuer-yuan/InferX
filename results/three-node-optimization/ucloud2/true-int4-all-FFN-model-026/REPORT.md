# All64 FFN actual-model bounded check

Same private binary and same corrected GQA object in both arms. Candidate uses FP8-source calibrated H512/G512 trueS4 gate/up+down for Prefill T256..2048 in all64 layers; Decode/Verify/MTP remain original. Baseline and candidate6long tasks each completed naturally with max256,seed42,greedy/noThinking,cache0,fullcoldprefill; no errors/empty/length. Scores4/6vs4/6;5texts identical. Ledger outputs differ, both wrong. This is bounded continuation evidence only, not originalmax32768 IFBench/task qualification.

Each candidate task invokes64layers×4T2048 gate/up and64layers×4T2048 down routes; all original request/response/native records retained. Complete8690073600B packedcode/scales cache H2D+D2H comparison passes. Load time21917.189ms. Actual candidate free-after-startup4.45GiB vs baseline12.54GiB, same KV278528 and graphs206.00MiB; this establishes this configuration's startup fit, not future capacity.

Both services reaped.029 full actualcandidateX/represented arithmetic/capturedY replay is underway before028r2samebinary untraced originalC16 performance. No formalaccuracy or SLO promotion yet. Sealed small archive2742841B/1023members/SHA256d74476e4f9a28077424b5bd52278e078373808ae2e02a3d427e482380314591e; large capture binaries stay under026baseline/candidate/activation with identity in subsequent029.
