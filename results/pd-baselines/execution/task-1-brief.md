### Task 1: Engine adapters and owned campaign support

**Files:** Create inferx/pd_baselines.py, tests/test_pd_baselines.py, scripts/run_pd_baselines.py; reuse inferx/concurrency.py and inferx/service_ownership.py.

**Interfaces:** Consumes run_load(endpoint, requests, output, mode, workers), summarize_load(records, span_s), RestorableService(saved, restore, manifest_paths, pre_stop_check). Produces engine_command(engine:dict, root:Path, run_dir:Path)->list[str], parse_metrics(text:str)->dict[str,float], validate_case(case:dict)->None and a raw campaign schema with engine/provenance/group/repetition/arrival/data/terminal restoration fields.

- [ ] Write failing tests for128K per-slot admission, prohibitedGPU0 selection, explicitquantization/morphxcoverage fields and multi-label Prometheus parsing.
- [ ] Run targeted tests and inspect failures; implement only required adapter/controller changes, preserve existing protocol semantics.
- [ ] Pass tests for owned stop/restore and real HTTP tool/error retention; commit.

Expected: every candidate command has explicit GPU/environment/context/budget/model path and rejects incomplete configuration; controllers preserve all errors and restore on exceptions.

