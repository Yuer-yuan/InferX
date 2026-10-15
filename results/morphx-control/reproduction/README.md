# MorphX transport/NUMA validation reproduction

This is the explicitly scoped MorphX-on-SGLang port. It preserves the author's
LLVM pass, in-place progress profiler and schedule_2, with native-library partial
coverage and CUDA-copy instead of author GDRCopy. It does not establish an
original-author performance ceiling. All artifacts/weights/builds/raw are on the
existing server HDD; no global driver/package changes, GPU1 only, GPU0 stopped.

Root: `/home/guokc/mypro/gnn-mount-sda1/InferX` on `vlm-server`; source `env.sh`.
Reuse the frozen27B checkpoint and prior compiler/Marlin/overlay. The predecessor
source capsule/dependency details are in `../../sota-validation/reproduction/`.
The eight exact new source/build trees are in `source-capsule-v1.tar.gz`, plus
adapter source and upstream licenses. All185 payload hashes were verified after
downloading. Archive SHA256:
`a29c59f25d129901ea422565780096143ab4359b798bf48f6ba4a2e75c7a0b78`.
Libraries and large dependencies remain on HDD with hashes in the manifests.

`../qualification/host-placement.json` records actual GPU1NUMA1 CPU affinity.
Use agent36 through SMSCHED_AGENT_CPU (validated/logged), server28–35 and
client40–43/96–99, disjoint physical cores within this experiment. This does not
reserve the whole host or promise exclusive CPUs against unrelated workloads.
The placement override changes only the host binding helper. The device kernel,
progress-profiler, dynamic-SM and launch code remain unchanged. Current-stage
model qualification confirms actual `requested=36 observed=36` per backend.

Never overlap GPU controllers. Wait for the preceding campaign's terminal
`status=restored`, `restored=true`, complete expected cases and fresh identity/
idle/stopped-GPU0 guard. Every controller restores the exact original256K/MTP7
service plus both manifests in finally, using fresh PID/start-time ownership.
Do not stop foreign or busy services. Runtime/library/source/raw names are
immutable; use a unique run id.

The new predeclared configuration has13 potential cases (unrun cases have no
score): repository `configs/morphx-control.json`, server
`morphx-control/configs/morphx-local-cpu.json`. First compare four transports
at split96/mode16 and the compiler control. Mode16 disables the launch-time CI
cap only; actual dynamic agent scheduling remains. Keep the ablation label.

After the active campaign is terminal, reproduce diagnostics:

```sh
source /home/guokc/mypro/gnn-mount-sda1/InferX/env.sh
cd "$INFERX_ROOT/code-pd-baselines"
taskset -c 40-43,96-99 env PYTHONPATH="$INFERX_ROOT/code-pd-baselines" \
  "$INFERX_ROOT/repo-venv/bin/python" -u \
  "$INFERX_ROOT/morphx-control/run_local_diagnostics.py" \
  --run-id reproduce-transport-UNIQUE --groups cold-high \
  --cases compiled-primary-control \
  morphx-control-abi-only-96-ablation morphx-control-fill-96-ablation \
  morphx-control-event-96-ablation morphx-control-both-96-ablation
```

All cases get a fresh engine/code gate/four24K warmups and the full frozen25
request mixed wave (.5RPS48seconds,longat16; full131060cold tokens). Controls
and treatments have matched checkpoint/quantization/C4/131328KV/16BF16SSM/
FP8KV/eager/MTP0. Current-process transformation and actual agentCPU36 logging
are required. The completed7 diagnostic waves contain175 strict responses. Selection is
frozen in ../selection.json; formal3-repeat/serial-C4 solo evidence remains
in progress at this snapshot.

For formal runs use `run_local_formal.py` with
`--repeat 3 --groups cold-high resume-high --cases compiled-primary-control
morphx-control-both-96-ablation` after freezing selection. It retains the same
serving/workload harness and alternates treatment order across rounds (12 unique
wave identities verified). Use `--solo --groups short-control cold-high
resume-high` for each selected case's serial-client solo with the sameC4server.
The existing strict exporter (`scripts.analyze_sota_validation`) reconstructs
SSE, exact frozen inputs/counts, arrivals, runtime/configuration, independent
repeats and archive hashes; export twice and compare hashes. Profiled runs
never supply formal latency or bandwidth/SM occupancy claims.

Full CPU checks (currently272) belong outside timed performance campaigns:

```sh
taskset -c 40-43,96-99 "$INFERX_ROOT/repo-venv/bin/python" -m pytest -q
```
