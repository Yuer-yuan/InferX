105 completed the original first100 natural-EOS evaluation on frozen100 e4ed/c976, bounded-membership flag1, native13/MTP K3, KV532480. W2 was off. Only generation max_tokens32768 was explicit; actual native sampling was temperature1/top_k20/top_p0.95, seed42, penalties0. All100 inputs matched the original messages and all native cache hits were0.

| Original metric | Count | Score | 95% prompt-cluster CI |
|---|---:|---:|---:|
| Prompt strict |85/100|85.00%|78.00–92.00%|
| Instruction strict |96/111|86.49%|80.00–92.38%|
| Prompt loose |88/100|88.00%|81.00–94.00%|
| Instruction loose |99/111|89.19%|83.04–94.50%|

All100 responses stopped naturally:0 HTTP errors,0 length truncations,1 empty final answer,100 nonempty reasoning fields,100 attempts with no retries. The empty answer remains scored. Original EvalScope1.13.0 verifiers reran all100; a separate CPU script rescored original HTTP content and independently reaggregated integer successes and all10000 seed42 bootstrap draws. Every per-sample value, aggregate and CI matched.

All583 archived raw/control files (59,291,703 bytes) were copied locally and SHA-verified. The original owner/waiter/service/collector PIDs are absent, port18505 closed, and the original owner recorded GPUempty. job.json retains its stale last-poll completed=99; complete raw and terminal files are authoritative. No new GPU task or service was launched.

The host scheduler recorded actual B>8 decode rounds and the first bounded subset13→9. It contains101 admissions versus100 HTTP requests; no direct scheduler/native-ID join or per-kernel coverage is claimed. Prompt lengths68..283 do not establish full cold8192 prefill task coverage. This is this candidate's first100 observation, not W2 quality, a pairedOFF comparison, or reproduction of the published79.5 recipe.

Existing startup evidence: KV payload19,119,919,104B, sequence23,326,205,952B, workspace361,906,176B, observed Graph117,440,512B, available6,486,228,992B. Subtracting W2 codes/scales2,941,255,680B leaves3,544,973,312B (3.302GiB), before changed Graph/workspace/loading peaks. No continuous free-memory sampler or actual W2 fit test is claimed.

Review paths: summary.json; delivery/CPU-score-001/{summary.json,per-sample.jsonl,failures-empty-length-strictfail.jsonl}; CPU-independent-review-002/; delivery/CPU-http-native-join-001/; CPU-local-coverage-003/; observations/memory-001/; observations/105-complete-raw.tgz with local-verification.json.
