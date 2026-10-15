### Task 4: Normalize slowdown, report and preserve

**Files:** New docs/reports/2026-10-15-sota-validation.md and results/sota-validation comparison/export/reproduction artifacts.

**Interfaces:** Strict raw exports produce continuous short/long slowdown and throughput/headroom/coverage, retaining independent repeat identities and excluded lanes.

- [ ] Export qualified raw twice, verify determinism/source hashes, calculate lane-specific solo/mixed ratios and joint criteria; if no qualified new lane, deterministic compatibility summaries retain failures and existing-reference provenance instead.
- [ ] Write direct measured answer with candidate/code/hardware distinctions and all unknowns; verify reproduction commands and source/binary inventories.
- [ ] Full suite and commit; recordTask4 completion before final review.

Expected: evidence-based answer to whether tested SOTA solves this workload, without a universal novelty or optimality claim.

## Final review and finish

- [ ] One fresh whole-change read-only reviewer; re-grade findings/declined items, oneImportant/Critical RED→GREEN fix pass, fullsuite and fresh service verification.
- [ ] Archive ledger/helpers/review/tests, close plan, delete only this scratch, retain branch/no merge/push; preserve prior dirty/user files.
