# Independent final review

Reviewer: /root/pd_strategies_final_review, gpt-6-astra/high, fresh context; read-only; no delegated reviewers or GPU experiments.
Range:19bfe21..3aa4d9cb9ba46ccb21ca40d1cb23e0291fc6a4ff. Previous-plan work was already reviewed at19bfe21. This is the single final review; the consolidated fixes below are verified by RED→GREEN and a full suite, not a second review.

## Strengths returned

- Admission reservation preserves stock consumption in finally, prevents a second unfinished chunk, and returns the stock rejection result so scheduler cleanup releases staged Mamba state.
- Split execution resets residual at interval zero, preserves batch-local intermediate state, and performs final normalization/logits once. Stream-context restoration and private Marlin workspaces address concrete source/runtime hazards.
- The report separates qualification failures, selection diagnostics, profiling, Graph probes, and the27selected formal waves. Latency, long-cost, and throughput match the canonical export.
- Reproduction artifacts preserve tested runtime bytes, patches, and complete inventories; local artifact/compressed-export hashes verified.
- Read-only inspection confirms GPU1 PID3604352 and matching manifests. GPU0's intentional absence is documented.

## Issues returned

Critical: none. Minor: none. Important: two; accepted at that severity.

1. docs/reports/2026-10-15-pd-strategies.md:101 / scripts/run_concurrency_campaign.py:64: documented GPU rerun invokes a controller whose final_shutdown_check still requires GPU0PID2504785/port18080, intentionally stopped by the user. Reproduction aborts beforeGPU1acquisition. Add explicit validated stopped-GPU0 path, retainGPU1ownership/idle checks and refuse unexpectedGPU0changes. Do not restartGPU0.
2. docs/reports/2026-10-15-pd-strategies.md:101 / scripts/run_pd_baselines.py:235: rerun instructions omit --groups short-control cold-low resume-high --short-rate .5. Frozen defaults contain cold-high and2.5RPS in some groups, giving4groups/120shorts rather than3groups/24shorts. Provide complete config/cases/groups/rate/repeat/fresh-run-id invocation.

Reviewer verdict: Ready to merge with fixes. Measured conclusions and implementation supported; fix reproduction path without newGPUmeasurements.

## Verification reported

Reviewed implementation/tests, material execution helpers, report, archived patches, relevant installed admission/split/scheduler source. Selected27waves/666strict/18full longs reconcile. Task4log205passed23.37s. Reviewer localPythonlackedpytest; no dependency installation or checkout/remote mutation performed.

## Declined to judge, returned verbatim

- General SOTA superiority, novelty, bandwidth limits, and efficacy on untested workloads: explicitly outside this bounded study and not claimed by the report.
- Full numerical/model-quality equivalence beyond the supplied code/tool gates: the report explicitly limits its correctness claims.
- Root causes of the original 8192-budget stall and Graph prefix-reuse loss, and Marlin safety across arbitrary shapes/partitions: retained as unresolved rather than used to support a causal claim.
- GPU0 endpoint availability or restoration: the user explicitly stopped it; restarting it would contradict the authorized final state.

The executor rules on every item in progress.md. Fix evidence: final-fixes-red.log (10failures before support/complete command), final-fixes-green.log (14targeted including prior recovery tests), final-suite.log. New CPU tests exercise the stopped-state guard, ownership changes, processing/deferred work, a real local TCP listener, unknown endpoint state, default refusal and the actual documented CLI parser before any side effects. A read-only live guard attempt refused before shutdown because a separate solo campaign had already acquired the original service. No live preflight success is claimed; ../service/final-external-service-state.json records the external owner and unavailable18081.
