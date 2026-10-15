### Task 1: Instrumented simple-policy overlays and controller configuration

**Files:** Create inferx/sglang_pd.py, scripts/prepare_pd_strategies.py, tests/test_sglang_pd.py, configs/pd-strategies.json; modify inferx/pd_baselines.py and tests/test_pd_baselines.py.

**Interfaces:** `cap_chunk_budget(adder, req, original, cap)` invokes original admission with a temporary per-request cap and restores remaining round budget; `prepare_overlay(source:Path, destination:Path, policy:str, cap:int|None)->dict` copies and patches source under a new owned directory and writes hashes. Engine adapter emits native cache-report/max-total-tokens/PD-config flags from explicit fields.

- [ ] Write failing tests for budget conservation, no-wait control, exhausted budget, exceptions, original capacity rejection, and valid128/256 caps; test explicit flags, HDD-only PD-config and no accidental GPU override.
- [ ] Run RED; implement smallest policy hooks and source-anchored atomic overlay preparation with equal instrumentation for control.
- [ ] GREEN targeted and full repository suite; CPU-validate real installed PrefillAdder accounting and preserve source hashes; commit.

Expected: instrumented independent overlays, verified unchanged no-wait behavior and no installed-package mutations.

