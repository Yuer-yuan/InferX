# QoServe native boundary qualification

Fourteen tests actually ran in the pinned vLLM0.31 environment and passed in
9.684 seconds. They use real Request, ModelRunnerOutput and native hybrid
allocators with a small logical attention/Mamba pool. They do not load27B
weights or establish GPU numerical correctness or performance.

The contracts cover joint full-input promises, query purity including internal
request maps, shared-prefix adoption and native CoW/checkpoint guards, C4,
abort/release, owned paused prefills, blocked-waiter progress, small128token
chunks, one-time relegation, arrival clocks, cumulative deadlines, malformed
caller metadata and actual runtime class/capacity/profile-mode evidence.

The complete final source capsule, native log/command, prepared reference
overlay, exact restored-service check and per-file hashes are in the evidence
archive. Its SHA256 is
`64362a0ef659b02d57ef88802f223c5c48e9ac47ddc8add09a8c685924479cb8`.
Earlier failing snapshots, trace-based diagnosis and RED→GREEN logs are retained
in `../native-boundary-v1` rather than overwritten.

Two native compatibility details required real regressions. A new capacity
query creates an empty Mamba defaultdict entry; the adapter removes only that
new empty lookup entry and never releases allocated blocks. Planning must also
begin after the previous synchronous step's Mamba flags are cleared. The
pinned clear is idempotent; the allocator's later clear and prohibition on
same-step checkpoint reuse remain active. No native source or GPU kernel is
modified.

The historical generated policy incorrectly said MIT, while its pinned LICENSE
has always contained Apache-2.0. The final snapshot corrects the header and
metadata. Its policy and native executable ASTs match the earlier14-test
snapshot exactly; the corrected snapshot also separately passes all14tests.
The original license bytes and old source snapshots remain available.

The prepared overlay is explicitly `author-reference` and refuses QoServe
deployment without supplied hardware coefficients and initial throughput.
Supplying these values alone does not qualify a profile. Real-model timing,
disjoint holdouts, tool/code checks and complete mixed/solo measurements remain
the next phases. The original256K/MTP7/C1 service, both runtime manifests,
exact argv and GPU0 stopped state were verified after these CPU tests.

The final full generic suite passed348tests, with28dedicated native tests
separately gated, on the HDD in33.59seconds. The same full suite passed locally
in40.28seconds. Two earlier remote full-suite attempts used incomplete fixture
envelopes: v8 omitted legacy config/docs/source data, and v9 still omitted C++
headers/fixtures. Their39/9failures and every test name remain in the failed
logs. Snapshotv10 adds those existing immutable fixtures; all129core Python
files are byte-identical to the actual native-testedv8. No test was deselected
or weakened to obtain the final generic pass.
