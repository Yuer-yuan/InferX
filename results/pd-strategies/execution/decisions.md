# Rulings and deferred minors

All20rulings, in ledger order; each includes cost if wrong.

1. Task2: Ruling: faithful PD-Multiplexing requires token chunking disabled and non-overlap scheduling — follow upstream validation, compare at the same model/pool but label layer-vs-token splitting as mechanism differences — cost if wrong: cannot attribute differences to SM partitioning alone; full128K qualification may expose workspace pressure.

2. Task2/3: Ruling: serialize GPU probes and simple diagnostic campaigns while CPU adaptation proceeds — overlays are immutable and controllers snapshot their code/config — cost if wrong: overlapping controllers would invalidate measurements; require prior terminal restoration before launch.

3. Task2: Ruling: explicit matched pool must equal startup token_capacity, not merely exceed context — added regression RED→GREEN — cost if wrong: page-alignment differences would need explicit qualification rather than silent acceptance.

4. Task2: Ruling: add a separately labeled PDMux+token-chunk adaptation, retaining upstream SM/layer scheduler and actual Green Context — whole-input prefill failed minimum-pool warmup; reuse stock stash/filter transition for unfinished chunks, explicitly synchronize the stream handoff — cost if wrong: this is an adapted combination rather than a faithful paper reproduction, needs full GPU correctness and no SM-only causal claim.

5. Task2/3: Ruling: add responsive caps128 and64 triggered by either waiting requests OR running decodes — diagnose remaining execution interference with another ordinary policy before claiming complexity required — cost if wrong: adds two selection diagnostics; preserve all and do fresh independent formal repeats. Targeted active-decode regression RED→GREEN; base waiting-only cap remains unchanged.

6. Task2/3: Ruling: normal common-pool references must KEEP mixed batching enabled; source get_next_batch_to_run otherwise always selects a pending long-prefill batch ahead of decode — PD's separate-stream scheduler requires mixed batching off — cost if wrong: control would be artificially starved; label interleaving implementation as mechanism difference and never claim SM-only effect. No normal common-pool reference has run yet; queued PD-only case is unchanged.

7. Task2/3: Ruling: correct reserved admission's row accounting, rather than increase state memory — actual ReqToTokenPool.alloc reuses req.kv.holds_kv rows, while scheduler compares total can_run against only free rows; credit held rows but clamp hardC4/PPbudget and preserve beam behavior — cost if wrong: could oversubscribe if reused-row semantics differ; unit bounds and fullGPU gates required. Newresponsive128/64-credit-v4 overlays/cases, diagnosticPID3511178 afterresponsive-v1 terminal restored. No GPU1 job should launch concurrently.

8. Task3: Ruling: now that PD qualifies only at131328KV/16SSM/ws128/eagerextra_buffer/non-overlap, pivot formal comparison to common resources fornormalcontrol/normalresponsive128/PD+responsive128 — stop all3waiting-only state16formalqueues before anyGPUacquisition; unusedfrozenconfigskept — cost if wrong: resulting matched ordinary lane may be slower than best140000/lazy deployment; reportstate16selectionseparately, no cross-pool speedup. First matched formal r1predeclared before data; same best ordinary cap128, PD96/32partition, fullcode/C4warm/longgates everygroup. At most3partitionsperplan, onlyonechosenqualifiedpartition required; no paper-SOTA ceiling claim.

9. Task3: Ruling: calibrate PD layer budget to2048 before seeing anyPD8192 mixed-wave performance — stock forward_count=budget/extend_tokens, hence128token ongoingchunks at8192 runall64layers perinterval;2048 gives<=16layers whenP/Dcoexist, retainingtrue layerinterleaving — cost if wrong: smallerintervals mayhurtlongprogress; retain8192n1diagnostic andfresh3calibratedPDrepeats, no best/SOTA-ceiling claim. Waiting-only matched-formal-r2/r3 stopped verifiedbeforeacquisition; snapshotsretained. Normal r1 remainsformal predeclared configuration; calibratedr2/r3 repeatnormalcasesunchanged andnewPDl16, calibratedr4 isPD-only forits3rdfreshrepeat. Profile prior/selected must change to calibratedr4/l16.

10. Task3: Ruling: isolate compressed-tensors Marlin lockworkspace perCUDAstream/scheme forPD — actualsourceusesmutableglobalbarriers andsharedself.workspaceviolatesconcurrentcallcontract; alloldsourcepreserved, newflagdefaultoff — cost if wrongaboutstallcause: needsadditionalGPUqualificationandlocksconsumeextrascratch, no claimitfixesthatstall. Sourceauditexplicitlyrecordsthatcleanl16alreadycompletedwithoutisolation. Commita621c73, full204passed22.61s. Threepartitions96/32,112/16,104/24qualificationgridPID3559182aftercurrentnormalformalr2; nativewatchdog60boundedviaexplicitcaptureddebuglauncher. Onlycoldselectionwaves, nonecountsformal. Waiting-onlynormalr3PID3553188stoppedbeforeacquisition, finalnormalr3PID3559183queuedaftergrid, controlsimpleorder. No otherGPUjobsactive.

11. Task3: Ruling: trade static KV headroom for16SSMslots when genuineC4admission exposed12slotexhaustion — keepfull128K/precision, comparestate16againststate16control andlatercommon131328pool — cost if wrong: reducedKVheadroommayincreaseeviction, so nevercross-poolattributeeffect. (Decision made duringearlierstate16qualification, recordedexplicitlyhere.)

12. Task3: Ruling: addoneintermediatelayerbudget4096afterl16demonstratedhighlongcost — same96/32/privateMarlin sourceandresourcepool, budgetvaluesbounded2048/4096/8192,3SMsplitsmax — cost if wrong: extraqualificationwave; dofresh3afterselection, nevercountselectionasformal.

13. Task4: Ruling: move finalreview/finish out ofTask4 steps into an explicit post-task gate — executing-plans requires onefreshwholechange review AFTER lasttaskcompletionline; Task4stillownsdeterministicexports/report/tests, finalgateownsreview/fixes/finalrestore/archive — cost if wrong: lastarchive-onlymetadatawillfollowreview, whileallbehavior/data/report arecommittedandreviewedbeforeit. No review skipped or duplicated.

14. Task3: Ruling: add separate bounded decode-CUDAGraph qualification for normal128 andPD128 after frozen eager repetitions/profile — currentpool/workspaceoptimization leavesgraphheadroompossible, andactualSGDecodeCudaGraphRunner explicitlysupportspdmuxperstream; merelyfixinggraphsoffforcausalablationcannotestablishstrongpracticallimit — cost if wrong: twoextraqualificationlanes; preserveeagerdata/parameters, requireactualgraphmemory/replay plusfullgatesbeforeanyclaim, freshrepeatsforanyselectednewdeployment. No fullprefillGraphcompile orTorch/driverdowngrade.

15. Task3: Ruling: final restoration checks apply to GPU1; leave GPU0 stopped — user explicitly confirms stopping GPU0 intentionally because it has no active influence on this work — cost if wrong: no GPU0 endpoint is provided; do not misreport its absence as an experiment failure or restore against user intent.

16. Final: Ruling: general SOTA superiority, novelty, bandwidth limits and untested workloads remain unjudged — conclusions apply only to measured configs and frozen workload; do not claim all existing methods fail — cost if wrong: real workloads or stronger deployments may remove the apparent gap.

17. Final: Ruling: full numerical/model-quality equivalence remains unjudged — supplied code/tool gates establish narrow functionality only; preserve that qualification in the report — cost if wrong: unmeasured tasks may reveal quality differences despite passing gates.

18. Final: Ruling: original8192stall, Graphprefix-loss root causes and arbitrary-shape/partition Marlin safety remain unresolved — source-risk correction and measured scoped runs stand without a general causal or safety claim — cost if wrong: untested shapes may still stall or change cache behavior, requiring new qualification.

19. Final: Ruling: GPU0 endpoint availability is not restored — explicit user shutdown is the authorized final state; new stopped-state guard supports futureGPU1reproduction without restartingGPU0 — cost if wrong: port18080remains unavailable until the user deliberately starts another service.

20. Final: Ruling: preserve separately launched solo-baseline campaign and its restoration responsibility — read-only finalprobe found originalPID3604352absent, SGLang on18083, externalcontroller3622529 owns solo-baseline-20261015-1792072041; it captured originalPID3604352 and runs under the same owned controller. Do not stop or restart over another active task; preserve its untracked solo-comparison files — cost if wrong:18081remains temporarily unavailable until that controller restores; previous successful restoration is historical evidence only. No present-health claim.

Deferred minors: none.
