# Client C16 / native cap12: FAIL, route closed

One fixed configuration, original128 measured +8warmup +1probe, all startup and pending queue time retained. Same2362/source4fcb, K3/fusionON/KV139264/chunk2048/450W/seed20261016. The only serving configuration change from the existing same-machine client16/native16 run is native cap16→12 (plus unique port/log paths). This is not clientC12.

| Original client concurrency | Native cap | TTFT P95(s) | TPOT P95(ms) | Phase output tok/s |
|---|---:|---:|---:|---:|
|16|16|9.855908|48.784071|311.768386|
|16|12|14.426440|43.719738|306.842133|

Cap12 improves TPOT10.38% but worsens TTFT46.37% and lowers throughput1.58%; both limits fail. No promotion/cap sweep/repeat. Among initial16,5TTFT and2TPOT violations; later112 still27TTFT and14TPOT violations. Nothing trimmed.

Actual137cold/8192/cache0/full_reset,136outputs512 plus original1-token probe,128client successes/zero errors. All137 salt-normalized actual input payload hashes and each phase cohort hash equal native16. All105 delivered files,128client rows and137native rows were checked directly; summary.json and measured-request-index.jsonl provide exact values and P95 interpolation weights.

Unique136client/native association uses the unchanged absolute-first-token10ms gate (max1.473ms, minimum next-nearest gap884.963ms). For the SAME P95 contributing requests: TTFT14.426440s = queue/prepare/control13.524018 + ownPF0.902422; TPOT43.719738ms = participating shared decode20.695577 + nondecode23.024161. Native shared-round elapsed is not unique GPU kernel time; nondecode contains competingPF and control/client accounting. No newtrace was collected. This supports admission queueing as the TTFT cost; it does not assert exact GPU causality.

Owner462372/47305022,driver462388/47305103,service462391/47305356 and all descendants checked missing after pidfd cleanup/reap; GPU empty1MiB/450W. Two temporary read-only SSH255 failures during the run caused no lifecycle action. Compact9,013,671B SHA4f2406e6bd533327424f5493fb7fae15ad02d60922aa50cdd80f673f1ebd5f1e; original raw retained remote and delivery/content/raw. No quality task, model change, compile, or additional HTTP smoke.
