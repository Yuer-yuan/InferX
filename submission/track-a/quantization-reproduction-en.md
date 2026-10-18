# Regenerate the 025 / 041 / 047 caches from public FP8 inputs

[中文操作说明](quantization-reproduction.md) · [Source, dependencies and base-model preparation](reproduction-en.md) · [Delivery and actual receipts](../DELIVERY.md)

This workflow creates new quantized caches and runs their own performance and quality evaluations. First complete steps 1–2 of the main guide and the public-model download/tokenizer extraction in step 3. You do not need the old-cache attachment or its evaluations. FP8, the public .ninfer base and installed environments are explicit external inputs.

## Obtain the complete quantization workflow

Frozen archive 002 is 38,965,052 bytes. It contains four complete build-source trees, capture, fitting, binding and original evaluation code. The coordinator independently verified all 4277 archive files: 4276 payload files plus MANIFEST. There are no old serving caches, compiled objects or prebuilt services. Continue in the main guide's terminal and choose a destination that does not exist.

```bash
# Continue with INFERX_REPO and INFERX_DEPENDENCIES from the main guide.
export INFERX_Q="$INFERX_DEPENDENCIES/reproduction-inputs/portable116-002"
export INFERX_Q_ARCHIVE="$INFERX_REPO/results/three-node-optimization/ucloud2"
export INFERX_Q_ARCHIVE="$INFERX_Q_ARCHIVE/full-quantization-reproduction-116-portable-source-002.tar.gz"
mkdir -p "$INFERX_DEPENDENCIES/reproduction-inputs"
mkdir "$INFERX_Q"
printf '%s  %s\n' \
  7300c70a5857221537b878046d8c869ce640c788de4773d91d4e523b6bb41f9e \
  "$INFERX_Q_ARCHIVE" | sha256sum -c -
tar -xzf "$INFERX_Q_ARCHIVE" -C "$INFERX_Q" --strip-components=1
python3 - "$INFERX_Q" <<'PY'
from pathlib import Path
import hashlib, json, sys
root = Path(sys.argv[1])
files = json.loads((root / 'MANIFEST.json').read_text())['files']
assert len(files) == 4276
for name, spec in files.items():
    data = (root / name).read_bytes()
    assert len(data) == spec['bytes'], name
    assert hashlib.sha256(data).hexdigest() == spec['sha256'], name
print('Verified', len(files), 'payload files')
PY
```

## Quantization inputs and scope

Regenerate the project's 025 FFN, 041 input-projection and 047 output-projection caches. Download the fixed public `.ninfer` base container separately; it originates from a distinct pinned BF16 model and is not regenerated from the FP8 input in this workflow. New caches, loading checks and evaluations receive their own identities and do not inherit historical scores.

### Obtain the FP8 input

Run from the delivered repository root and select a new destination. The standard-library Python downloader pins each official file revision, size and SHA. Its 72 content files total 30,879,892,015 bytes. It excludes the old mount's `.mv/.msc` metadata and does not assert one shared snapshot revision.

```bash
export INFERX_FP8_FETCH="$INFERX_Q/fp8-acquisition"
export INFERX_FP8="$INFERX_DEPENDENCIES/inputs/Qwen3.8-27B-FP8"
mkdir -p "$INFERX_DEPENDENCIES/inputs"
python3 "$INFERX_FP8_FETCH/fetch_fp8.py" \
  --directory "$INFERX_FP8" \
  --receipt "$INFERX_DEPENDENCIES/inputs/fp8-acquisition-complete.json"
```

A complete receipt requires all 72 files to verify. Existing valid files are rehashed; mismatching files, symbolic links and existing partial files are rejected and preserved. The downloader's separate network smoke test fetched only the 51,350-byte `config.json`; this does not establish a new 30.9GB download. The actual regeneration run's full-input receipt identifies the FP8 directory it used.

The separate public base is `neroued/Qwen3.8-27B-NInfer@3526913004b1cf552cb57b88d6a5c6f5e4a89a70`, SHA `eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e`. Its pinned release documentation names BF16 input `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, converter `Neroued/ninfer@52320554b5e71a9da96bff809ddf67ac5773ed63` and recipe `qwen3_8_27b-v1`. This workflow acquires the fixed public container; it does not perform the 55.56GB BF16 download and conversion.

### Fixed calibration data and fitting arithmetic

Use four training and two held-out WikiText-2-raw-v1 inputs, each containing 8192 raw token IDs. Selection seeds are 20261017 for training and 20261018 for validation; the request seed is 42. The six-request file SHA is `f93b4d2b44c9ca642277f14c9260e64d8c95be2858b6c4e4b1db16874381481a`. Each request produces one token at temperature 0 with K0. Dataset revisions, offsets, exact token IDs and request salts are preserved. IFBench prompts are not calibration inputs.

Each input has four 2048-token chunks, with 512 captured rows per chunk. Per layer, 16 training chunks provide 8192 rows and eight held-out chunks provide 4096 rows. Freeze fitted outputs before computing held-out diagnostics. Preserve H512/G512, fixed per-row scales, FP64 GPTQ feedback, damping 0.01, block 128, no actorder and disabled TF32. Recover logical FP32 values directly from E4M3 weights times BF16 block scales, without another BF16 rounding step.

| Dependency order | Inputs used during capture | Fresh outputs |
|---|---|---|
| Capture 024 → fit 025 | Original Q4/Q5 forward from the fixed base; full S4 route disabled | 64 gate/down layers, 256 serving files |
| Capture 038 → fit 041 | Fixed base plus this run's new 025 | 64 input projections, 128 serving files |
| Output capture → fit 047 | Fixed base plus this run's new 025 and 041 | 47 selected output projections, 94 serving files |
| Select and load | Preserve all outputs; 047 fits the fixed 47-layer mask | 256+128+94=478 serving files and a separate asset manifest |

Every stage in the new directory reads its freshly produced predecessor. The portable entrypoint needs no old 020/021 arrays. Its 025 L31 check compares full new expanded/scales hashes with fixed expectations, preserving their provenance and the original zero-difference receipt. Core fitting functions and training remain unchanged. Per-matrix checks and an independent 256-row CPU feedback check remain; repeated large held-out diagnostics after freezing are omitted. These checks do not establish task accuracy.

### Install the fitting environment

The GPU fitter uses Python 3.12 and PyTorch 2.13.0+cu130, separately from the C++ service built with CUDA13.2. Its 44 dependencies have pinned official wheel URLs and hashes. Installed METADATA was checked; a full wheel download and new environment installation were not performed in this run. For a new host, first set `INFERX_REPO` and `INFERX_DEPENDENCIES` as in the main guide.

```bash
export INFERX_QUANT="$INFERX_Q"
export INFERX_FIT_ENV="$INFERX_DEPENDENCIES/envs/quant-fit"
"$INFERX_DEPENDENCIES/envs/build/bin/python" -m venv "$INFERX_FIT_ENV"
"$INFERX_FIT_ENV/bin/python" -m pip install --no-deps --no-index \
  --require-hashes -r "$INFERX_QUANT/fit-environment/requirements-wheels.lock"
"$INFERX_FIT_ENV/bin/python" -m pip check
export FIT_PY="$INFERX_FIT_ENV/bin/python"
"$FIT_PY" -c 'import torch; assert torch.__version__ == "2.13.0+cu130"; print(torch.__version__, torch.version.cuda)'
```


## Run the full workflow in a new directory

Set the path variables used on the slide. `INFERX_W` must not exist and must not point to the historical 116 run. Dependency and public-model paths continue from the main guide. The default system Python is `/usr/bin/python3` and must support pidfd. Default capture ports are 19048, 19050 and 19051; final performance and quality use 19052 and 19053.

```bash
export INFERX_W="$INFERX_DEPENDENCIES/reproductions/quantization-rerun01"
export INFERX_M="$INFERX_MODEL_DIR/qwen3_8_27b.ninfer"
export INFERX_T="$INFERX_TOKENIZER"
export INFERX_B="$INFERX_DEPENDENCIES/envs/build"
export INFERX_C=/usr/local/cuda-13.2
export INFERX_X="$INFERX_DEPENDENCIES/source/xgrammar_source-src"
export INFERX_E="$INFERX_DEPENDENCIES/eval-envs"
export INFERX_U="$INFERX_DEPENDENCIES/tools"
```
```bash
python3 "$INFERX_Q/reproduce.py" \
  --work-dir "$INFERX_W" --fp8 "$INFERX_FP8" \
  --model "$INFERX_M" --tokenizer "$INFERX_T" \
  --fit-python "$FIT_PY" --build-env "$INFERX_B" \
  --cuda-root "$INFERX_C" --xgrammar "$INFERX_X" \
  --eval-envs "$INFERX_E" --tools "$INFERX_U"
```

The entrypoint normally builds capture and service sources, runs 024→025→038→041→output→047, binds the new 478 serving files and executable, then runs original C14 performance and a separate original 100 quality evaluation. `--prepare-only` only materializes paths and performs CPU checks; it neither compiles nor runs inference. After using it, choose another new work directory for the actual run.

The entrypoint prepares the fixed IFBench data in its own work-directory cache: 300 prompts/344 instructions, with the original first 100/111 unchanged. Prepare separate AIPerf 0.13.0, EvalScope 1.13.0 and NLTK environments first. Performance begins only after all compilation and fitting have finished.

## Check outputs, failures and actual completion

| Output | What to verify |
|---|---|
| `invocation.json`, `generated-path-bindings.json` | Actual external inputs and paths; no historical owner invoked |
| `portable-job.json`, `normal-builds-job.json` | Real terminal states, exit codes and new build identities |
| Each 025 / 041 / 047 directory | Fresh packed/scales, per-matrix numerical checks and frozen manifests |
| `deployment116/` | New asset manifest, recipe/index, real service identity and loading records |
| Performance and quality directories below `deployment116/` | Raw requests, 128 measured records, 100 predictions/reviews and all failures |

Performance requires both TTFT P95≤10s and TPOT P95≤40ms; throughput retains the original profiling window. Quality retains all 100 prompts, four scores, errors, empty finals and length stops. A running process, an output directory or a verified archive does not establish completed evaluation.

The actual 116 run's first 024 capture used old objects before the additional full-source requirement arrived. The subsequent normal-source capture rehashed all 3072 activation files on both sides, with zero differences and all 128 fitting-input bindings verified. This permits reuse of this run’s newly fitted 025. The [actual receipt](../../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/024-normal-source-input-equivalence.json) and [independent audit](../../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/root-024-input-equivalence-audit.json) preserve the sequence; this 3072-file check establishes capture-input equality. Separate receipts now confirm all three fitting stages below; new-executable evaluation remains separate. Frozen 002 instead starts directly from new-source capture and does not require third parties to repeat that historical detour.

Consult the [delivery index](../DELIVERY.md) and final 116 attachment for actual new-cache stages, loading, performance and quality. Rebuild 664's 304.8967 tok/s and 90/96 scores used old caches and are not results for the regenerated assets. Base-container provenance, FP8 input and cache identities remain separate.

## Completed 116 quantization outputs

Run 116 actually refitted all 239 serving matrices: 128 for 025, 64 for 041 and the fixed 47 for 047, producing 478 packed/scales files. Comparing every fresh manifest entry with the original selected 478 contract gives identical SHA and byte count for every file, with zero differences. See the [three-cache comparison](../../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/fresh-vs-original478-SHA.json) and [actual fitting evidence](../../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/fresh239-fit-light-evidence-001/full-fresh-fit-evidence.json). These are regenerated and checked outputs; old caches were not copied in place of fitting.

| Cache | Fresh matrices | Serving files | Matching original SHA/size |
|---|---:|---:|---:|
|025 FFN |128 |256 |256 /256 |
|041 input projections |64 |128 |128 /128 |
|047 output projections |47 |94 |94 /94 |

The outputs are bound by a separate [116-fresh47 recipe](../../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/deployment116/entrypoint/candidate_recipes/116-fresh47.json). The normally rebuilt executable has SHA `e919f18084f3cc09718158cfae071cdf8c58d5660715648db375643016c878f5`, with source aggregate `c976692d9657408308193b50772c700382a6e243d9da353165205dc5252f76a8`. Its C14 performance has independently passed, as recorded below; original 100 quality also completed at 85/100 strict and 88/100 loose, with full terminal evidence below. Neither result inherits 664 or historical e4ed scores. Identical cache contents do not replace end-to-end evaluation of the new executable.

### Actual C14 performance of the new 116 executable

Executable e919f180… and this run's fresh 478 caches completed original C14: client 14/native 13, KV121856, ordinary MTP K3, cold 8192 input/512 output; 128 measured+8 warmup+1 probe. All 128 measurements succeeded with zero errors. TTFT P95 **9.862210240600s**, TPOT P95 **39.42860192427ms**, passing both 10s/40ms targets; throughput **305.532868219352 tok/s**. The denominator is the original 214.497380861-second profiling observation window, with 65536 output tokens. Initial measured requests and admission waiting are retained.

The [original summary](../../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/fresh-C14-raw-001/deployment116/performance-C14/summary.json) preserves exact values. The parser has no meets_slo field and the entrypoint records null; joint PASS is derived from both original P95 thresholds. The performance service was reaped normally before the independent original 100 quality run, whose own scores and complete failures are recorded below. These are e919's own measurements and do not replace 664 or historical 100/105 results.

See the [independent raw-record audit](../records/typst/clean-reproduction/fresh-performance-audit.json) and [C14 threshold/termination receipt](../../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/C14-threshold-evaluation.json).

Throughput denominator: profiling `end_ns=1792287366484028998` minus `start_ns=1792287151986648137` in `phase_manifest.json`, divided by 10⁹ after integer subtraction, gives 214.497380861s. Dividing 65536 by this duration gives 305.53286821935166 tok/s. AIPerf separately exports `benchmark_duration=214.49216998400001s`; this distinct field does not replace the throughput denominator.

### Original 100 quality of the new 116 executable

The same e919 executable and fresh 478 caches used a separate quality service: client/native concurrency 13, KV532480, ordinary MTP K3, default thinking and sampling (temperature 1, top_k 20, top_p 0.95, seed 42), with only max_tokens=32768 explicitly overridden and natural EOS. All original IFBench first 100 prompts/111 instructions completed, with zero generation errors. Prompt order, text, instruction IDs and kwargs match the historical frozen cohort.

| Scoring level | Strict | Loose |
|---|---:|---:|
| Prompt |85/100 |88/100 |
| Instruction |95/111 |99/111 |

There were 99 natural stops and one length stop. Ordinal/key 50 generated 32768 tokens and retained reasoning but had an empty final answer. This single sample counts as both empty and truncated and remains in every denominator. The original EvalScope 1.13.0 verifier independently rescored all 100 outputs and matched every native review. The original aggregate instruction scores remain rounded to 0.8559/0.8919; the table derives exact counts from per-prompt reviews. The service exited normally; owned processes were absent, ports closed and no GPU compute processes remained at observation time.

Sources: [original quality summary](../../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/fresh-quality100-raw-and-CPU-001/deployment116/quality100/evaluation/summary.json), [independent original-verifier receipt](../../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/fresh-quality100-raw-and-CPU-001/quality-independent-CPU-001/summary.json), [all failures and empty/length cases](../../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/fresh-quality100-raw-and-CPU-001/quality-independent-CPU-001/failures-empty-length.jsonl), [actual termination](../../results/three-node-optimization/ucloud2/full-quantization-reproduction-116/fresh-quality100-raw-and-CPU-001/actual-terminal-and-GPU-116.json), and [material-side archive/cohort/score audit](../records/typst/clean-reproduction/fresh-quality-audit.json). Prompt scores match historical 105 at 85/88, but strict instruction scores are 95/111 versus 96/111; this is not identical accuracy. Results cover only the original first 100 prompts, not the full 300 or general quality, and do not isolate quantization effects across builds.
