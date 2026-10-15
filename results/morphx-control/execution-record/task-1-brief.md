### Task 1: Source-qualified transport variants and actual model correctness

**Files:** Create inferx/morphx_control.py, scripts/morphx_control_ops.hpp, scripts/prepare_morphx_control.py, tests/test_morphx_control.py; new results/morphx-control/qualification and HDD immutable builds/helpers.

**Interfaces:** Consume exact v2 manifest/source; apply inferx.morphx_validation.upgrade_proc_address_queries first in the new tree, retain historical old binary, and requalify corrected ABI on GPU. `patch_control_runtime(source: str, *, device_fill: bool, event_completion: bool) -> str` rejects stale anchors; preparation verifies full source inventory before writing a fresh named tree/library. Small C++ control helper provides exact fill and recorded-event completion semantics; concrete native API table/layout is implementer-owned.

- [ ] Write source-mismatch/overwrite regressions plus compiled asynchronous-driver behavioral oracles: initialize agents128/768/1024 on exact stream after allocation; event not-ready leaves resources live, ready frees once, errors fail; unrecorded queries rejected and record precedes publish. Assert profiler retains original progress path and native launch blocks unchanged. Run targeted tests RED.
- [ ] Implement minimal checked source adapter/helper, preserving two synchronized progress/bound initializations and existing profiler; GREEN. No async-init publication race introduced. Event exists only on non-profile launches; cleanup on errors and record-after-success before pending publication.
- [ ] Build three fresh libraries on HDD (fill,event,both); rehash source/binary manifests. Actual GPU vector and five Marlin geometries each profiling on/off; concurrent two-stream completion/SM oracle. Use existing owned controller and preserve failed outputs.
- [ ] Full27B code/four24K/complete128K gates for selected plausible variants with actual runtime coverage; every measured variant must qualify. Fix demonstrated failures with RED→GREEN; unknown never becomes a performance point. Full CPU suite, exact service proof and commit.

Expected: actual mechanism-preserving, model-qualified transport variants or substantive unresolved boundary, no unmeasured efficacy claim.

