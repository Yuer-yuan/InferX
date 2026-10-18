# GDN fold+record C32 result

**No promotion.** Fixed C32/native32/K3, original128 measured+8 warmup+1 probe, seed20261016, KV278528,450W,traceOFF. All128 successful, zero errors;137native cold8192,cache0/fullreset;136outputs512+probe1. Allstartup retained.

|Metric|Previous8b20|New2362|
|---|---:|---:|
|P95 TTFT(s)|23.040531|23.088049|
|P95 TPOT(ms)|92.109025|96.319341|
|Phase output tok/s|374.046140|375.714210|

Throughput+0.446%, TPOT tail worsens; bothjointFAIL. Initial32 TTFT21/TPOT29 violations, later96 TTFT8/TPOT79. No seed rescue or additional request. The entire137payload SHA, warm8 and measured128 cohort SHA exactly match previous8b20 after only cache_salt removal.

Native identity join:136/136 uniqueabsolute firsttoken associations within frozen10ms, max2.970ms, next-nearest minimum886.400ms. Finish metadata residual is separately retained. New same-contributor P95 TPOT96.319=participatingdecode25.902+nondecode70.418ms/token; old92.109=28.102+64.007. New P95 TTFT23.088=queue/control22.185+ownPF.903s. Each run uses its own actual P95 contributing requests and interpolation weights; stageP95s are never added.

Across128measured requests, native ownPF service sum115.937→116.141s. Mean participatingdecode per committedtoken22.576→21.271ms;109/128matched prompts decrease. Sumperrequest shareddecode1476.631→1391.291s repeats shared rounds across requests and is NOT unique GPU walltime. RowVerify invocations20194→20305 includingfallback36→40; yield3.239→3.221committed/rowround. Accepted/drafted74.902%→74.333%. Exact global Graph calls/B/flush counts are unavailable in this traceOFF run.

The new mechanism reduces measured per-request shareddecode time, while prefill service cost is unchanged and nondecode tail contribution grows. This supports a limited decode benefit, not overall SLO success or precise causal attribution of every tail difference. The formalrun has no physicaltrace, so remainder cannot all be labeled PF.

Source4fcb/1019,modelbinary2362fc; prebuild fullCTest86actualpass+4missingfixture skip. Native7realrounds/168rows perarm,2013paired recordszero differences,9actual partialcommits,ON17oldrow>=B/87positivepending merges,480deferredGraphnodes. Both banks andslot15/31 lifecycle verified. No taskquality claim. Previous8b20 source/binary remains immutable backup.

Raw: delivery/content/raw. All103files checked against manifest;9,012,066B archive SHA ba93e9573ec7191ef6bd6d8d6224c40978c20b8465603a0f9cf3b43a4b2eb378. All17 associated build/native/formalprocess identities missing after reap; GPU1MiB/noapps. Index files and raw remain available for Root independentaudit.
