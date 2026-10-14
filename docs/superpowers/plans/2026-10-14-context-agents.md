# Larger Context and Coding Agents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement inline. Steps use checkbox syntax.

**Goal:** Expand the NInfer context and leave working OpenCode, Pi and Claude Code local clients with measured repository runs.
**Architecture:** Reuse GPU1 engine and comparable HTTP harness; isolate client packages and task copies on HDD. Select context before serial client trials and verify edits against pristine tests.
**Tech Stack:** Python3.12/pytest/aiohttp, NInferSM89, Node22, Bash, three client CLIs.
**Spec:** `docs/superpowers/specs/2026-10-14-context-agents-design.md`

## Global Constraints

- GPU1 only; preserve GPU0PID2504785/port18080; API loopback18081.
- All remote downloads/configs/packages/taskcopies/cache/logs under `/home/guokc/mypro/gnn-mount-sda1/InferX` HDD; models throughhf-mirror, reuse verified weights.
- Keep64K rollback; no driver/global environment modifications, cloud fallback, push or merge.
- Context choices131072INT8 then262144rk4v4-e8,MTP7,C1,prefill2048; require1024MiB observed headroom and quality gates before promoting.
- Run all clients on fresh copies of identical3tasks,600s/task; verify pristine tests and archive errors.

## Review Focus

- Reused PID cannot kill a foreign process: identity/executable/start-ticks test and actual process check (Task1).
- Long-input probes cannot overflow output reserve or turn failed retrieval into a capacity pass (Task1).
- Client context declarations and auxiliary models cannot exceed runtime or use cloud defaults (Task2).
- CLI exit0/model claims/test modifications cannot substitute for pristine verification and actual tool events (Task3).
- Timeout must terminate only the owned client group and preserve failure logs (Task3).

### Task1: Context validation and safe deployment

**Files:** Create `inferx/context_agents.py`, `scripts/prepare_long_context.py`, `scripts/run_context_probe.py`, `tests/test_context_agents.py`; reuse `start_tuned_server.sh`.
**Interfaces:** `verify_process_identity(pid:int, start_ticks:str, executable:Path)->bool`; `long_context_messages(records:int)->list[dict]`; probe JSONL from existing `stream_request`; chosen runtime manifest for Task2.

- [ ] Write failing tests for stale/mismatched identity (real subprocess/temp paths), prompt early/middle/late expected lookups and output reserve; run targetedpytest, expectedFAIL before implementation.
- [ ] Implement helpers; generate131K/262K-fit corpora using GPU0 read-only tokenizer, keep>=8192 output headroom; preserve request text and hashes. Expectedthree expected values correctly generated and actual token counts belowcapacity-reserve.
- [ ] Stop only verified owned64KPID; run staged128KINT8 and256Krk4 probes, short heldoutcode and telemetry. Expectedquality/fit success or explicitfailedcandidate, no hidden exclusion. Leave selectedwarmserviceandmanifest,64Krollback.
- [ ] Run fullsuite, inspect evidence, commit; task-done usesremote `repo-venv/bin/python -m pytest -q`.

### Task2: Isolated client setup

**Files:** Create `scripts/install_agent_clients.sh`, `scripts/agent-env.sh`, `scripts/run_local_agent.sh`; configurations under `configs/agents/`; tests in `tests/test_agent_clients.py` if helpers enforce semantic invariants.
**Interfaces:** Consumeschosenruntimecontext; wrapperCLI `run_local_agent.sh {opencode|pi|claude} [CLIargs...]`, all state/cache/logpathsHDD, localmodelqwen38-27b. Producesversion/protocolsmoke manifests.

- [ ] Downloadandhash-checkNode22.23.3, installpinnedOpenCode1.19.0/Pi1.2.0 underHDDprefix; reuseClaude2.1.92. Expectedversions recorded, noexistingconfigmutation.
- [ ] Configure nativeOpenAI/Anthropic APIs and matchingcontext/outputbudget; isolateauxiliarymodelrouting. Runactualtoolsmoke (read/edit/test) perclient; fixonlyobservedcompatibilityerrors withRED→GREEN regressions ifanadapterisneeded.
- [ ] Verifylauncher syntax/statepaths/localserverrequestlogs and fullsuite, commit; savecommands.

### Task3: Repository agents, results and handoff

**Files:** Create `scripts/run_agent_trials.py`, `scripts/analyze_agent_trials.py`, `tests/test_agent_trials.py`, `results/context-agents/`, `docs/reports/2026-10-14-context-agents.md`; updateREADME.
**Interfaces:** Consumes3clientwrappers and frozen3tasks; producestrialmanifest withclient/task/version/wall/status/verification/toolcounts/requestusage/diffhash/rawpaths.

- [ ] Write failing tests for pristine-testverification, missingtoolactivity, nonzeroexit/APIerrors and ownedgroup timeout (realminimalprocesses), then implementboundedtrialrunner. Expectedfailures classified withoutsuccessclaim.
- [ ] Run3clients×3tasks freshcopies serially; archive transcripts/diffs and independentlyrunoriginaltests. Expected9complete result records (qualityfailuresallowed, infrastructure failures investigated).
- [ ] Validateartifacts, health/GPU0preservation, fullsuite; writereproductionandcontextlimitationsreport; commitandtask-done.
- [ ] Freshwhole-branch review perexecuting-plans, fixImportant/Critical oncewithRED→GREEN, recordminors; preservebranchandrunningservice.
