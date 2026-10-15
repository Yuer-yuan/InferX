# MorphX-on-SGLang reproduction

The measured system is the explicitly scoped MorphX-on-SGLang adaptation, with
the original compiler/profiler/schedule_2 and primary-context P/D streams. It is
not the unavailable prepared author vLLM/GDRCopy environment. GPU1 only; all
source, dependencies, models, caches and raw runs use the existing server HDD.

Server root: `/home/guokc/mypro/gnn-mount-sda1/InferX` on `vlm-server`.
Source `env.sh` before commands. Run a single GPU controller at a time, after
the preceding campaign records terminal restoration. Every controller checks
fresh original-service identity/idle state, preserves intentionally stopped
GPU0, and restores the original256K/MTP7 service plus both manifests in finally.

The frozen configuration is `configs/sota-validation.json` in this repository,
copied on the server to `$INFERX_ROOT/sota-validation/configs/sota-validation.json`.
Its selected case is `morphx-copy96-no-launch-cap`; this disables launch-time CI
restriction only, retaining the actual agent scheduler. Keep its ablation label.

```sh
source /home/guokc/mypro/gnn-mount-sda1/InferX/env.sh
cd "$INFERX_ROOT/code-pd-baselines"
env PYTHONPATH="$INFERX_ROOT/code-pd-baselines" \
  "$INFERX_ROOT/repo-venv/bin/python" -u \
  "$INFERX_ROOT/sota-validation/run_sota_diagnostics.py" \
  --run-id reproduce-morphx-mixed-UNIQUE \
  --cases compiled-primary-control morphx-copy96-no-launch-cap \
  --groups cold-high resume-high --repeat 3
```

After that command completes and its campaign is restored, reproduce matched
isolated execution with the same server capacity, serializing only the client:

```sh
env PYTHONPATH="$INFERX_ROOT/code-pd-baselines" \
  "$INFERX_ROOT/repo-venv/bin/python" -u \
  "$INFERX_ROOT/sota-validation/run_sota_diagnostics.py" \
  --run-id reproduce-morphx-solo-UNIQUE \
  --cases compiled-primary-control morphx-copy96-no-launch-cap \
  --groups short-control cold-high resume-high --solo
```

Both commands reuse the exact frozen workload/template hashes. `*-high` names
are inherited; actual short rate is explicitly0.5RPS,48seconds,longat16seconds.
Every group starts a fresh engine, code check and four24K warmups. The main
comparison includes three fresh mixed waves per lane/group and one serial solo
diagnostic; request samples are not pooled as independent runs.

The diagnostic profile helper is `$INFERX_ROOT/sota-validation/run_sota_profile.py`.
Use `--case morphx-copy96-no-launch-cap --run-id reproduce-morphx-profile-UNIQUE`
after all timed runs. It uses a separate CUPTI stream-mapping overlay. Profiled
latencies are excluded from comparisons; kernel lifetime overlap does not
establish instantaneous SM occupancy or actual DRAM bandwidth.

Export only completed/restored campaigns:

```sh
env PYTHONPATH="$INFERX_ROOT/code-pd-baselines" \
  "$INFERX_ROOT/repo-venv/bin/python" -m scripts.analyze_sota_validation \
  --frozen "$INFERX_ROOT/concurrency-roofline" \
  --mixed "$INFERX_ROOT/sota-validation/raw/reproduce-morphx-mixed-UNIQUE" \
  --solo "$INFERX_ROOT/sota-validation/raw/reproduce-morphx-solo-UNIQUE" \
  --output "$INFERX_ROOT/sota-validation/reproduce-comparison.json"
```

The exporter reconstructs strict tools from SSE, checks exact original frozen
payloads and arrival schedule, complete128K inputs, actualC4/pool/precision/
eager/MTP0 state, four warmups, current-process transformation evidence and
archived harness/source hashes. Run twice to two output paths and compare their
SHA256; no wall-clock generation timestamps enter the normalized output.

`source-capsule-v1.tar.gz` contains246 exact custom source files, including the
runtime, compiler-pass source/build patch, Marlin headers/translation unit and
the11 overlay files that differ from the pinned installed wheel. Its manifest
hashes every payload. All246 entries were verified after downloading the archive.
Archive SHA256: `7f771fe8e41ec6c916324c1810f0ad32755d1b133e4968efc87841a3b9888551`.
Large unchanged wheels, models, source caches and original raw results remain
on HDD. Preserve before/after overlay hashes when reconstructing into a fresh
directory; a newly compiled binary needs numerical/model qualification again.

The actual qualified runtime is
`sota-validation/builds/libcuda_copy_v2.so`, SHA256
`64f3268499a920e30db33d48b8ecee9137216e3dd494e0a93d0e2119a542af3a`.
Marlin is `sota-validation/builds/marlin-morphx-v2/marlin_morphx.so`, SHA256
`22753219976c737e30598bcb60951309a63880446b513ee9f26c931557de24e7`.
The actual LLVM pass SHA256 is
`45d1d638a06ea178c03f6abc91d2f84f32c259f3915828ed07316c205ea91429`.
The runtime/source/overlay audit rehashed4605 files against immutable manifests;
see `../runtime-artifact-summary.json` and its compressed full inventory.

Run the full CPU suite outside timed performance windows:

```sh
taskset -c 40-43,96-99 "$INFERX_ROOT/repo-venv/bin/python" -m pytest -q
```

Missing GDRCopy, native-library kernel coverage, source-only Bullet/EB+ and
unqualified NInfer GQA are retained as explicit limits. They do not establish
negative efficacy of the corresponding author methods or universal novelty.
