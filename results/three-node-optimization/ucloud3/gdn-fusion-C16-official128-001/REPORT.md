# Current2362 C16 original128 result

TTFT P95 **9.855908s passes**, TPOT P95 **48.784071ms fails**; output phase throughput **311.768386tok/s**. All128 measured requests succeeded, plus8 warmup and1 load probe. All137 native requests are cold8192/cache0/full_reset;136 output512 and probe1. Every startup request is retained. No C12 or automatic repeat was launched.

Source4fcb/binary2362, GDNfusion1, native16/K3/KV139264/chunk2048/450W. Historical dc99 C16 differs in combined implementation; it is not a GDN-only causal control. Exact salt-stripped137-payload and phase cohort hashes match old C16. The actual argv/env are in delivery/content/owned/job.json; complete raw is delivery/content/raw.

The same P95 requests reconstruct TTFT as8.952183s queue/control +0.903726s own prefill; TPOT as19.255793ms participating shared decode +29.528278ms nondecode. These are native/client host times, not exclusive GPU time. Without a fresh trace the nondecode remainder cannot all be attributed to prefill.136 unique first-token associations pass the unchanged10ms gate (max1.283ms; next-best separation>=886.536ms); finish residuals remain in native-budget.json.

All17 build/native/formal identities are actually missing and GPU was idle1MiB/450W when collected. Driver and telemetry exited0 and were reaped.104 transferred files match their SHA manifest. Full128 raw, failures, and same-P95 interpolation contributors are retained. The first two CPU audit failures (historical summary schema, read-only re-extraction) are preserved; recovery003 only adapts field names and verifies already-extracted files without writing them. No performance or source change.

No quality claim or stable-capacity promotion follows.
