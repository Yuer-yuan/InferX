### Task 3: Matched comparative campaigns

**Files:** configs/pd-strategies.json; HDD pd-baselines/raw with unique pd-strategies run names; results/pd-strategies campaign summaries.

**Interfaces:** Uses existing `scripts.run_pd_baselines` case schema, per-group independent restart and `scripts.analyze_pd_baselines` strict raw export. Records frozen overlay source hashes in case metadata and raw scheduler event files/server logs.

- [ ] Verify active service identity/idle and frozen hashes; run baseline plus128/256 queue-only and128/64 responsive simple diagnostic groups with full code/tool gates.
- [ ] Complete three repetitions for matched baseline and best simple policy; if PD qualifies, probe at most three SM partitions and complete best configuration's three repetitions. Retain selection diagnostics and failed lanes.
- [ ] Verify long completion, strict-tool counts, actual capacity, raw arrival counts, restore original service and both manifests, and commit summaries.

Expected: no conclusions from failed/truncated/refused long work, no claims that small-sampleP95 is a productionP99 or that API cache counts are physical recomputation.

