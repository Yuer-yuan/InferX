# Portable v1 A8 prefill kernel candidate

This bundle applies to frozen `5c60b7c9b455231795c09da21a9fbb6aa53f08e5` **after** the existing v2/completions API patch (`d5534f0189f1e05c91dc32d5ca6d19541d52ca94f7f95209eb009193b2fe87d1`). It ports kernels from community branch `tensorninja`, pinned commit `5b99996bc7c1fe6ff27f271a205eb605302ec8bb`. The community runtime, artifact reader, scheduler, attention implementation, GDN recurrence and decode changes are not imported. The frozen v2 artifact loaded successfully and executed the 8K ladder and C4/C8 pilot.

`prefill-a8-v1.patch` contains 14 files: two modified frozen files (`src/CMakeLists.txt`, the 27B target's `variant.cpp`) and 12 added internal kernel/helper files. `files/` contains their exact resulting contents. Patch SHA256: `8feca94afe3bb222693fe2b3aca17b1e0f45252710ff0856a19f27d13d6492ca`. Portable source-file-set SHA256: `072d797468ad000d7d9c734ef81002ecf43dac1cd3cd68ccb8bf8e8f37d0f6b8`. `identity.json` gives every file SHA, source-tree identities and the tested binary SHA256 `13ab270d739111feaa53f5df9f34567269500014062004a73f3276abf8777281`.

The patch was applied with zero fuzz to an independent copy of the frozen patched tree. All 14 resulting file hashes match the actual compiled sources; see `apply-verification.json`. Build paths and toolchain details can change binary bytes, so record the receiving node's new binary hash. Combining scheduler changes is a new labeled configuration and needs same-machine comparison.

```bash
export INFERX_ROOT=/workspace/InferX
bash export-v1/build.sh \
  "$INFERX_ROOT/source/ninfer-4090" \
  "$INFERX_ROOT/experiments/a8-port/source" \
  "$INFERX_ROOT/experiments/a8-port/build"

# Only on an exclusively owned, idle GPU; event timing includes quantization.
source "$INFERX_ROOT/config/node-env.sh"
source "$INFERX_ROOT/config/ninfer-env.sh"
"$INFERX_ROOT/experiments/a8-port/build/bench/ninfer_a8_probe" 256 257 1024 2048 4097
ctest --test-dir "$INFERX_ROOT/experiments/a8-port/build" -R '^ucloud3_a8_numeric_smoke$' --output-on-failure
```

Activation profile is selected once per service process through `NINFER_PREFILL_A8`:

| Value | Changed Prefill routes |
|---|---|
| `none` or unset | none, frozen BF16 paths |
| `swiglu` | Q4 SwiGLU |
| `ffn` | SwiGLU plus Q5 FFN down |
| `all` | FFN plus attention/GDN input and Q5 output projections |

Every new route requires `TextPhase::Prefill` and T>=256. Decode, MTP verify and small-T calls follow the frozen call graph. This is group64 symmetric **INT8 activation**, separately labeled from unchanged INT8 KV storage. It changes model arithmetic and can change generated tokens and MTP acceptance. Expanded paired task quality is pending.

Example using the preserved baseline configuration (choose a free port and fresh native log):

```bash
NINFER_PREFILL_A8=all "$INFERX_ROOT/experiments/a8-port/build/apps/ninfer-serve" \
  "$INFERX_ROOT/models/qwen38-ninfer/qwen3_8_27b.ninfer" \
  --host 127.0.0.1 --port 8088 --model-id qwen38-27b \
  --max-context 65536 --kv-capacity 131072 --max-concurrency 8 \
  --max-pending-requests 32 --pending-timeout-ms 600000 --prefill-chunk 2048 \
  --kv-dtype int8 --no-thinking --greedy --presence-penalty 0 --frequency-penalty 0 \
  --seed 42 --no-ui --spec mtp --draft-tokens 4 --lm-head-draft \
  --no-prefix-reuse --no-disk-cache --request-log-jsonl /ABSOLUTE/NEW_RUN/requests.jsonl
```

Use `NINFER_PREFILL_A8=none` with the **same binary and same actual prompt-token cohorts** for causal A/B. Maintain the original 450W power limit. System `/usr/bin/python3` must own pidfd lifecycle management; AIPerf/EvalScope retain their existing environments. Do not signal an old numeric PID.

The activation staging is bounded to 4096 tokens. It uses the same allocator helper for capacity calculation and execution: `padded_K * min(T,4096)` INT8 bytes plus `(padded_K/64) * min(T,4096)` FP32 scales, aligned to arena boundaries. At T2048: K5120 = 11,141,120 bytes; K6144 = 13,369,344; K17408 = 37,879,808. These are shared call-scoped workspace slots, not persistent allocations per layer. Actual ucloud3 startup workspace remains 345.14 MiB because another frozen execution leaf dominates the total maximum; observed GPU use was 24,836 MiB. Recheck composed workspace when combining changes.

Evidence under the parent worker directory:

- `shape-probe-summary.json`, `jobs/a8-shape-probe-001/console.log`: 25/25 actual-shape seeded-synthetic cases, full-output finiteness and sampled independent FP64 oracles. Maximum sampled A8 NRMSE 0.008604; this is an operator smoke, not task-quality qualification. `probe/` includes source, summary and SM89 SASS evidence.
- `model8k-ladder-summary.json`: three cold8192/1 per mode, identical input hashes and cache0; same-binary none 2460.44 -> all 3693.09 tok/s, 1.501x at 450W.
- `cohort-audit-pilots.json`: actual measured and warmup prompt-token cohorts match baseline/A8 at C4/C8. `inputs.json` bytes differ only in request cache salt; salts are not model token IDs and reuse is disabled. `cohort-audit-c4-full-payload.json` and `cohort-audit-c8-full-payload.json` also confirm equality of complete generation payloads after removing only cache_salt.
- `baseline-pilot-001/`, `a8-pilot-001/`: original-runner 16 measured +4 warmup, exact8192/512, zero errors. C4 A8: 7.546s TTFT P95 / 32.577ms TPOT P95 / 127.179 tok/s, pilot SLO pass. C8: 17.509s /73.761ms /137.662 tok/s, fail. These small-sample tails must not be ranked against 100-request official tails.
- `frozen-profile-001/attribution.json`: actual 8K prefill kernel time is 91.54% dense GEMM, 2.87% existing chunkwise GDN recurrence, 3.36% attention. No FlashQLA-on-SM89 claim.

The first 100-request/8-warmup C4 seed42 confirmation finished: TTFT P95 4.177s, TPOT P95 36.934ms, 121.736 output tok/s, 100/100 exact8192/512, zero errors, all109 native requests cache0/full_reset, joint SLO pass. Independent seed43 is running; same-input none seed42 is queued. C6/C8 are queued at100/8 with the same seed42 dataset SHA lock. Current identities are in `status.json`. Expanded paired task quality remains pending.
