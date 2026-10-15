### Task 4: Analyze, review and preserve evidence

**Files:** Create docs/reports/2026-10-15-pd-strategies.md and results/pd-strategies final comparison/export; add only necessary analyzer tests/code if new fields are required.

**Interfaces:** Strict raw exports and source manifests produce per-repeat control/simple/PD comparisons, long completion cost, throughput and operational SLO judgments with explicit exclusions.

- [ ] Export deterministically twice and compare hashes; verify original raw/event/provenance hashes; report improvements and remaining unknowns with all repetitions retained.
- [ ] Run full suite, commit the report and analysis evidence, and record Task4 completion before the separate final-review gate below.

Expected: direct measured answer, source-backed compatibility boundaries, and verified deterministic analysis ready for independent review.

## Final review and finish

- [ ] After the last task completion line, run one fresh whole-change review; one Important/Critical fix pass under RED→GREEN, then suite and final service verification.
- [ ] Archive execution ledger/artifacts, close plan, keep the research branch in place and leave reviewable files; no merge/push.

Expected: independently reviewed restoration/reproducibility and an archived execution record.
