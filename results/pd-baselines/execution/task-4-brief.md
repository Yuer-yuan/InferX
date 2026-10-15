### Task 4: Export comparison and independent review

**Files:** Create scripts/analyze_pd_baselines.py, tests/test_pd_baselines_analysis.py; results/pd-baselines/{summary.json,groups.csv,requests.csv,audit.json}; docs/reports/2026-10-15-pd-baselines.md.

**Interfaces:** Consumes Task3 immutable raw; produces measuredabsolute/relative improvementswith independent repetitions, exclusions/capacity/quality/coverage labels and reproduction instructions.

- [ ] Test raw-only deterministic export, rejectedrequestretention, missingprefixcounterlabels, repeat aggregation and longcompletionfairness; implement analyzer.
- [ ] Run completeappropriate tests, twice export and comparehashes; write report that directlyanswersMorphX/alternatives/remainingbottleneck.
- [ ] Freshwhole-changereview; fix important findings with reproducingtests and finalverification; archiveexecutionrecord.

Expected: source-evidence-backed, reproducible answer to full userobjective, no scope substitution and no claim ofcompletion with unverifiedMorphX efficacy.
