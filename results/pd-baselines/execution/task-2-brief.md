### Task 2: Build and validate actual candidates

**Files:** Create scripts/prepare_pd_baselines.py, scripts/build_morphx.sh and configs/pd-baselines.json; HDDsource/pd-baselines, pd-baselines/builds and provenance.json.

**Interfaces:** Consumes Task1 engine case schema; produces pinned sources/model manifests, buildlogs, usable binaries/environments and actual compatibility/coverage proof for Task3.

- [ ] Pin MorphX full source and compiler/runtime, inspect compile interfaces, build for SM89 using isolated toolchains. Add required adapter tests before persistent implementation changes.
- [ ] Execute patched kernel correctness/coverage probe on owned GPU1; inspect failures and adapt missing components as feasible, preserve failure logs.
- [ ] Verify existing llama.cpp 27B path and seek modern vLLM INT4 path through hf-mirror; validate versions/driver/API/context and functional gates.
- [ ] Freeze runnable candidate config/source/weight hashes, commit adapters and compatibility evidence.

Expected: actually runnable same27B candidates or reproducible attempted compatibility failures, never untested paper claims.

