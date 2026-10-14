# Concurrent Qwen and Roofline Measurement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline. Steps use checkbox syntax.

**Goal:** Execute the three approved concurrency studies, test the bandwidth hypothesis at the available evidence strength, and restore the working256K agent service.
**Architecture:** A bounded HTTP load generator and owned-service controller reuse the pinned NInfer binary; independent CUDA-copy and Nsight traces do not contaminate performance runs. Frozen requests and native counters feed an idempotent summary.
**Tech Stack:** Python3.12/aiohttp/pytest/ctypes, installed CUDA12.8 and Nsight Systems, existing code verifier, Bash/SSH.
**Spec:** `docs/superpowers/specs/2026-10-14-concurrency-roofline-design.md`

## Global Constraints

- GPU1 only; preserve GPU0PID2504785/UUIDGPU-a66b08e8-8b65-0faf-22e2-b951a97b2b03/health18080.
- HDD root `/home/guokc/mypro/gnn-mount-sda1/InferX`; no new model/engine build, driver/global/account/profile-permission changes, push or merge.
- Original18081 service must be idle and identity-verified throughpidfd before stop; restore256K,rk4v4-e8,MTP7,prefill2048,C1,warm/no-thinking/greedy and refreshruntime/best-service manifests in finally.
- Experiments bind127.0.0.1:18083; max-context65536, pooledKV131072,rk4v4-e8,prefill2048,pending16/120s; primaryD0/4/7×C1/2/4, exploratoryD0/7×C8, startup headroom>=1024MiB.
- Retain errors/overload/timeouts/truncation and actual cache misses; profiler traces separate; no innovation, live-agent accuracy, losslessness or counter-based roofline proof without evidence.
- Cold/resume/control arrangement and counts/rates exactly as Spec; empiricalP95n>=20/P99n>=100, source counts and limitations explicit.

## Review Focus

- A stale manifest, foreign GPU process or busy service cannot be killed; restoration must update identity and preserve all client budgets (Task1/2).
- Slow completions cannot delay future open-loop arrivals, and dispatch lag/errors cannot disappear from latency or throughput denominators (Task1).
- Partial tool JSON and truncated code cannot become correctness passes; real HTTP error/stream events are retained (Task1/2).
- Configured concurrency/cache labels do not prove realized batch size or prefix reuse, and overlapping request timers are not GPU execution time (Task2/3).
- Repeat exports cannot read derived artifacts or duplicate identities; missing counters/trace failures cannot become roofline proof (Task3).

### Task 1: Load observations, process ownership and frozen campaign

**Files:** Create `inferx/concurrency.py`, `inferx/service_ownership.py`, `scripts/prepare_concurrency_workloads.py`, `tests/test_concurrency.py`, `tests/test_service_ownership.py`; freeze configs/workloads on HDD.
**Interfaces:** `reconstruct_tool_calls(events:list[dict])->list[dict]`; `summarize_load(records:list[dict],span_s:float)->dict`; `bandwidth_bytes_per_s(size:int,copies:int,elapsed_ms:float)->float`; `run_load(endpoint:str,requests:list[dict],output,mode:str,workers:int)->list[dict]`; `process_identity(pid:int,exe:Path)->dict`; `stop_matching_process(identity:dict,timeout:float)->None`; `refreshed_manifest(saved:dict,identity:dict)->dict`.

- [ ] RED tests for fragmented tool arguments, error retention/quantile thresholds and bandwidth units; real localHTTP delay proves arrivals overlap rather than serialize and closed-loopworker cap is honored. ExpectedmissingmoduleFAIL.
- [ ] RED realprocess test: mismatchedstartticks refusesstop, unrelatedprocesssurvives correctstop; metadatarefreshpreservesconfigandupdatesPID/startticks. Implementhelpers, targetedGREEN.
- [ ] Freeze code/prose/tool payloads, token-sizedshared12K history and matched16/24/32K cold/resume inputs; plannedcounts/rates/configs saved beforegeneration. Use readonlyGPU0tokenizer only. Verifybounds/hashes/tool schema invariants.
- [ ] FullremotepytestGREEN, commit andtask-done.

### Task 2: Owned campaign, bandwidth reference and three studies

**Files:** Create `scripts/run_concurrency_campaign.py`, `scripts/measure_device_bandwidth.py`, `scripts/profile_concurrency.py`; use Task1helpers and existingverifier.
**Interfaces:** Consumesfrozenworkloads/configs; writesper-group rawJSONL/nativeevents/telemetry/quality/status/ownership/restoration manifests and independentprofilerruns onHDD.

- [ ] Confirm88testbaseline andcurrentoriginalidentity/GPU0/idle; save recoverymetadata. Stoporiginalthroughpidfd; devicebandwidthuses2×256MiB,8warmups,64copies×5samples. ExpectedrealCUDAmeasurementorarchivederror, nevercounterproof.
- [ ] Run primary9 configurations plus2C8admissionprobes, sequentialownedservers. For each admittedcandidate run studies1/3 and declaredsubsetstudy2; eachgroup warmup/measurement/nativecursor bounded separately. Recordstartupfailures/headroombelow1024withoutdroppingthem. Expectedcompletegroup recordsorinvestigated explicitfailure.
- [ ] Run existingisolatedmerge-intervalqualitygate withfullbudget andnativeforcedtoolchecks; saveprofilerpermissionrecord andboundedNsightCUDAtraces onrepresentatives only. Expectedtrace/kernelartifactsor exactprofilererrors, neither countedthroughput.
- [ ] Finallyrestoreexactoriginalconfiguration at18081, refreshactualPID/startticks/exe/time inboth manifests, verifyOpenAI/Anthropictools,262144advertisement/GPU0health. FullsuiteGREEN, commitandtask-done.

### Task 3: Artifact analysis and research decision

**Files:** Create `scripts/analyze_concurrency.py`, `tests/test_concurrency_analysis.py`, `results/concurrency/`, `docs/reports/2026-10-14-concurrency-roofline.md`; updateREADME.
**Interfaces:** Consumesonlyrawcampaignmanifest/groupfiles/nativeevents/traceexports; writescompactcases/groups/requestmetrics/sourcehashes andbandwidth/profilinglimitations.

- [ ] RED realexporttests for repeat/idempotence/duplicateidentities andfailure-accounting; implementraw-input-onlyaggregation. ExpectedtestsFAILbeforeimplementationandGREENafter.
- [ ] Auditgrid/admission/exclusions/counts; compareD0/4/7andC1/2/4, cold/resume/control, uniform/mixedcontext/classes, kerneldecompositionandbandwidthreference. Expectednohiddenfailedrecords, nofalseP99/quality/rooflineclaims.
- [ ] FullsuiteGREEN, reportChineseevidence/limits/reproduction anddecision: ordinarybandwidthsharingversusavoidableinterference, orundetermined. Commitandtask-done.
- [ ] Freshwhole-branchreview perexecuting-plans; fixCritical/ImportantoncewithRED→GREEN, recordminors/rulings, archiveledger, preservebranch/liveGPU1service.
