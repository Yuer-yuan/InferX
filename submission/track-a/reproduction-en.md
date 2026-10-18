# Frozen 100: rebuild, serve, benchmark and evaluate quality

This submission reconstructs quantized caches from source and does not bundle the large cache attachment. For a fresh reproduction, complete the package/environment setup and public base-model/tokenizer preparation here, then follow the [quantization guide](quantization-reproduction-en.md). It generates caches, builds and binds a new service, and runs performance and quality evaluation. The measured-cache workflow below is an optional way to revisit the historical 664 result when those exact assets are already available.

[中文操作说明](reproduction.md) · [Historical 100/105 results](selected-results.md) · [Actual scripts and commands](../../results/three-node-optimization/ucloud3/selected100-single-node-reproduction-20261018/COMMANDS.md)

This workflow rebuilds frozen 100 from its complete source and runs C14 performance followed by the original 100 quality evaluation on one 48 GB RTX 4090. Both evaluations use the new executable, with separate services, KV capacities and output directories.

The recorded ucloud3 root is `/workspace/InferX/reproductions/selected100-single-node-20261018`. Source, build and run directories are new; the model, quantized caches, tokenizer and installed environments are reused from the existing host. This is not a fresh operating-system installation or a new quantization run. Historical binary `e4ed1bf4…` and evaluation105 scores of 85%/88% remain separate from the new build's identity and measurements.

**The submission defaults to regenerating caches from source using the [complete 116 workflow](quantization-reproduction-en.md).** Use sections 1–2 and the first code block in section 3 below to prepare dependencies, the public base model and tokenizer, then continue with 116. The 12 GB old-cache archive is not included; the 664 historical branch below is optional for users who already possess that fixed asset.

## 1. Obtain and verify the frozen reproduction kit

Start in the delivered InferX repository root and choose a destination that does not exist. The frozen `reproduction-kit-v1.tar.gz` is 40,181,061 bytes. Its hash pins the source, entrypoints, dependency manifests and actual reproduction evidence. Large model/cache payloads, installed environments and old executables are external inputs.

```bash
export INFERX_REPO="$PWD"
export INFERX_BUNDLE="$INFERX_REPO/results/three-node-optimization/ucloud3"
export INFERX_BUNDLE="$INFERX_BUNDLE/selected100-single-node-reproduction-20261018"
export INFERX_REPRO=/workspace/InferX/reproductions/selected100-rerun01
mkdir -p /workspace/InferX/reproductions
mkdir "$INFERX_REPRO"
printf '%s  %s\n' \
  62fc432a278f2b7f8c60472c173929fc1f537ea2785908207e8b1ce7703ed545 \
  "$INFERX_BUNDLE/reproduction-kit-v1.tar.gz" | sha256sum -c -
tar -xzf "$INFERX_BUNDLE/reproduction-kit-v1.tar.gz" \
  -C "$INFERX_REPRO" --strip-components=1
cd "$INFERX_REPRO"
sha256sum -c SHA256SUMS
export INFERX_KIT="$INFERX_REPRO"
```

The manifest covers 1207 payload files, including all 1023 engine source files. The source aggregate is `c976692d9657408308193b50772c700382a6e243d9da353165205dc5252f76a8`: SHA256 of the sorted compact JSON mapping of relative paths to file hashes. It differs from the archive hash. The build controller checks every source file before and after compilation.

`evidence/` preserves the commands and build 664 receipts actually executed on ucloud3. The portable kit relocates the build lock to its own directory and maps external inputs through `asset-sources.json`. Its r2 entrypoints verify the same service and wait for read-only `/models` readiness before the unchanged evaluator. See `evidence/packaged-vs-actually-tested-code.json` for the exact relationship. A future build can have a different SHA and cannot inherit historical results.

## 2. Install dependencies on a new host

The recorded ucloud3 run reused installed environments. The following commands are first-install instructions, not a claim that those environments were reinstalled during this run. Prerequisites are Linux x86_64, a 48 GB RTX 4090, a driver compatible with CUDA 13.2, CUDA Toolkit 13.2.51 and a working micromamba executable (recorded version 2.9.0). This procedure does not modify the system driver.

Continue in the extracted kit directory, retaining the variables from step 1. The dependency prefix is `/workspace/InferX`, matching the recorded commands. Preserve an existing environment instead of overwriting it. Use `build-conda-explicit.lock`, which contains exact package URLs and checksums; the human-readable `.txt` listing is not an installation lock.

```bash
# Continue from the extracted kit root.
export INFERX_KIT="$INFERX_REPRO"
export INFERX_PROVENANCE="$INFERX_KIT/asset-origin"
export INFERX_LOCKS="$INFERX_PROVENANCE/remote-provenance/dependency-locks"
export INFERX_DEPENDENCIES=/workspace/InferX
export INFERX_MAMBA="$(command -v micromamba)"
export INFERX_CUDA_ROOT=/usr/local/cuda-13.2
mkdir -p "$INFERX_DEPENDENCIES"/{envs,eval-envs,config,cache,tools,source}

"$INFERX_MAMBA" --no-rc create --yes \
  --root-prefix "$INFERX_DEPENDENCIES/cache/mamba" \
  --prefix "$INFERX_DEPENDENCIES/envs/build" \
  --file "$INFERX_KIT/evidence/build-conda-explicit.lock"
```

```bash
# Preserve the original freeze. packaging comes from the explicit conda lock.
python3 - "$INFERX_LOCKS/build-pip-freeze.txt" \
  "$INFERX_DEPENDENCIES/config/build-pip-install.txt" <<'PY'
from pathlib import Path
import sys
lines = Path(sys.argv[1]).read_text().splitlines()
local = [line for line in lines if ' @ file:' in line]
assert len(local) == 1 and local[0].startswith('packaging @ file:///')
Path(sys.argv[2]).write_text('\n'.join(line for line in lines if line not in local) + '\n')
PY
export INFERX_BUILD_PY="$INFERX_DEPENDENCIES/envs/build/bin/python"
"$INFERX_BUILD_PY" -m pip install --no-deps \
  --index-url https://pypi.org/simple \
  -r "$INFERX_DEPENDENCIES/config/build-pip-install.txt"
"$INFERX_BUILD_PY" -c 'import packaging; assert packaging.__version__ == "26.3"'
"$INFERX_BUILD_PY" -m pip install --no-deps \
  --index-url https://pypi.org/simple uv==0.11.29
"$INFERX_DEPENDENCIES/envs/build/bin/uv" --version
"$INFERX_BUILD_PY" -m pip check

for inferx_eval_spec in aiperf:0.13.0 evalscope:1.13.0; do
  inferx_eval_name=${inferx_eval_spec%:*}
  inferx_eval_version=${inferx_eval_spec#*:}
  inferx_eval_dir="$INFERX_DEPENDENCIES/eval-envs/$inferx_eval_name-$inferx_eval_version"
  "$INFERX_BUILD_PY" -m venv "$inferx_eval_dir"
  "$inferx_eval_dir/bin/python" -m pip install \
    --index-url https://pypi.org/simple pip==26.2.1
  "$inferx_eval_dir/bin/python" -m pip install --no-deps \
    --index-url https://pypi.org/simple \
    -r "$INFERX_LOCKS/$inferx_eval_name-pip-freeze.txt"
  "$inferx_eval_dir/bin/python" -m pip check
done
```

```bash
# Download the three NLTK resources and verify their original content hashes.
python3 - \
  "$INFERX_PROVENANCE/original-support/results/ninfer-performance/preparation/nltk-resources.json" \
  "$INFERX_DEPENDENCIES/eval-envs/nltk_data" <<'PY'
from pathlib import Path
import hashlib, json, sys, urllib.request, zipfile
manifest, root = Path(sys.argv[1]), Path(sys.argv[2])
archives = root / '.archives'
archives.mkdir(parents=True, exist_ok=True)
for item in json.loads(manifest.read_text()):
    target = archives / item['file']
    if not target.exists():
        with urllib.request.urlopen(item['source'], timeout=120) as response:
            target.write_bytes(response.read())
    data = target.read_bytes()
    assert len(data) == item['bytes']
    assert hashlib.sha256(data).hexdigest() == item['sha256']
    kind = item['source'].split('/packages/', 1)[1].split('/', 1)[0]
    with zipfile.ZipFile(target) as archive:
        archive.extractall(root / kind)
PY
export NLTK_DATA="$INFERX_DEPENDENCIES/eval-envs/nltk_data"
"$INFERX_DEPENDENCIES/eval-envs/evalscope-1.13.0/bin/python" - <<'PY'
import nltk
for resource in ('tokenizers/punkt_tab', 'corpora/stopwords',
                 'taggers/averaged_perceptron_tagger_eng'):
    print(nltk.data.find(resource))
PY
```

```bash
# Obtain the exact grammar dependency; CUDA and its driver are prerequisites.
git clone https://github.com/mlc-ai/xgrammar.git \
  "$INFERX_DEPENDENCIES/source/xgrammar_source-src"
git -C "$INFERX_DEPENDENCIES/source/xgrammar_source-src" \
  checkout 4546988d1cf51670a10eafb8795d37e4a2f0d833
git -C "$INFERX_DEPENDENCIES/source/xgrammar_source-src" \
  submodule update --init --recursive
export PATH="$INFERX_DEPENDENCIES/envs/build/bin:$INFERX_CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$INFERX_DEPENDENCIES/envs/build/lib:$INFERX_CUDA_ROOT/lib64"
"$INFERX_CUDA_ROOT/bin/nvcc" --version
"$INFERX_DEPENDENCIES/envs/build/bin/x86_64-conda-linux-gnu-g++" --version
```

The `packaging @ file:///...` line in the original build freeze names a Conda build-machine path. The commands preserve that freeze and omit only this item from the pip installation list. The explicit Conda lock supplies `packaging==26.3`, which is checked separately. Each evaluator receives its own environment and original freeze, followed by `pip check`. NLTK archives must match the recorded full hashes; preserve a mismatching download and stop.

Quantization fitting uses a separate GPU PyTorch environment documented with that workflow. Do not substitute these API evaluation environments for the fitter.

### Actual independent installation in a fresh prefix

A separate prefix completed all 27 installation steps in 857.424 seconds, each with exit code 0. All three pip checks, the unchanged runner's check and composition with the original v1 prepare_assets passed. The [complete receipt](../../results/three-node-optimization/ucloud3/selected100-single-node-reproduction-20261018/environment-install-addendum/delivery/FINAL-SUMMARY.json) and [frozen addendum](../../results/three-node-optimization/ucloud3/selected100-single-node-reproduction-20261018/environment-install-addendum/environment-data-delivery.tgz) are separate from v1; archive SHA is `f82b9fc992e9cfac84c64bf8a1dfa87fe7c5d8c72da20fbbec3161a3bc1b2c36`. The original EvalScope offline loader read and matched all 300 fixed prompts, including the first 100 with 111 instructions. Six verifier-source hashes match. No inference requests or new scores were produced.

The 133-package Conda lock matches. All 138 performance and 123 quality dependency versions remain unchanged; each new environment additionally includes explicit pip 26.2.1. Of 55 runtime libraries, 42 have identical raw SHA. The remaining 13 differ only in installation-prefix strings and NUL padding; all other bytes match after strictly normalizing those positions. This is not a claim that all 55 raw hashes match.

Executable 664's DT_RPATH still resolves the original build environment. Its performance and quality were not rerun in the new prefix. The addendum validates installation, original data loading and asset preparation, with no new-environment engine build or GPU evaluation. The complete v1 build route uses this guide's default dependency prefix `/workspace/InferX`; build_owner still fixes that path. The installation test does not establish a complete build at arbitrary custom prefixes.


## 3. Prepare the fixed model, caches and original evaluation data

This section reproduces the reported ucloud3 configuration with existing frozen caches. The public model is18,210,531,328 bytes; extract the tokenizer from its embedded resources. The478 serving files are a separate `selected100-u3-47-caches.tar` attachment (12,082,360,320 bytes), excluded from the40.18MB source kit. Its [origin and checksums](../../results/three-node-optimization/ucloud1/selected100-reproduction-chain-CPU/README.md) identify the completed archive. Obtain that attachment and place it in the `inputs` directory below before checking and extracting it. This is cache acquisition, not new quantization.

To generate caches from public FP8 inputs, use the separate quantization workflow and bind a new asset manifest. The old recipe is valid only when the content hashes match exactly; regenerated caches cannot inherit the 664 results.

```bash
export INFERX_MODEL_DIR="$INFERX_DEPENDENCIES/models/qwen38-ninfer"
export INFERX_TOKENIZER="$INFERX_DEPENDENCIES/models/tokenizer"
export INFERX_FROZEN_ASSETS="$INFERX_DEPENDENCIES/inputs/selected100-assets"
export INFERX_DATA_CACHE="$INFERX_DEPENDENCIES/cache/modelscope-rerun01"
mkdir -p "$INFERX_DEPENDENCIES/inputs"
"$INFERX_DEPENDENCIES/envs/build/bin/hf" download \
  neroued/Qwen3.8-27B-NInfer qwen3_8_27b.ninfer \
  --revision 3526913004b1cf552cb57b88d6a5c6f5e4a89a70 \
  --local-dir "$INFERX_MODEL_DIR"
printf '%s  %s\n' \
  eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e \
  "$INFERX_MODEL_DIR/qwen3_8_27b.ninfer" | sha256sum -c -
python3 "$INFERX_KIT/asset-origin/extract_tokenizer.py" \
  --model "$INFERX_MODEL_DIR/qwen3_8_27b.ninfer" --out "$INFERX_TOKENIZER"
```

**The following branch is only for an existing old-cache archive. The default 116 route continues in the quantization guide after the model/tokenizer preparation above.**

```bash
# Optional historical branch only: use an existing verified old-cache archive.
export INFERX_CACHE_ARCHIVE="$INFERX_DEPENDENCIES/inputs/selected100-u3-47-caches.tar"
printf '%s  %s\n' \
  a8936aa7d39e9f971256a30b415f6f271f4f70418d0705233c0c84bfabfc3fb7 \
  "$INFERX_CACHE_ARCHIVE" | sha256sum -c -
mkdir "$INFERX_FROZEN_ASSETS"
tar -xf "$INFERX_CACHE_ARCHIVE" -C "$INFERX_FROZEN_ASSETS"
(cd "$INFERX_FROZEN_ASSETS" && sha256sum -c SHA256SUMS)
python3 "$INFERX_KIT/asset-origin/verify_assets.py" \
  --root "$INFERX_FROZEN_ASSETS"
```

```bash
export INFERX_IFBENCH_PACKAGE="$INFERX_REPO/results/three-node-optimization/ucloud1"
export INFERX_IFBENCH_PACKAGE="$INFERX_IFBENCH_PACKAGE/ifbench-fixed-dataset-acquisition-CPU"
export INFERX_IFBENCH_PACKAGE="$INFERX_IFBENCH_PACKAGE/ifbench-fixed-dataset-cache-r1.tar.gz"
printf '%s  %s\n' \
  c267da418e8c0a79b402e37237b14fde4a17233f60b8d5f866b85493fda757aa \
  "$INFERX_IFBENCH_PACKAGE" | sha256sum -c -
export INFERX_IFBENCH="$INFERX_REPRO/ifbench-data"
mkdir "$INFERX_IFBENCH"
tar -xzf "$INFERX_IFBENCH_PACKAGE" -C "$INFERX_IFBENCH" --strip-components=1
export MODELSCOPE_CACHE="$INFERX_DATA_CACHE"
python3 "$INFERX_IFBENCH/stage_ifbench.py" \
  --cache-root "$MODELSCOPE_CACHE" \
  --receipt "$INFERX_REPRO/ifbench-stage-receipt.json"
"$INFERX_DEPENDENCIES/eval-envs/evalscope-1.13.0/bin/python" \
  "$INFERX_IFBENCH/check_offline_cache.py" \
  --cache-root "$MODELSCOPE_CACHE" --evalscope \
  --receipt "$INFERX_REPRO/ifbench-original-loader-check.json"
```

```bash
python3 - "$INFERX_REPRO/asset-sources.json" \
  "$INFERX_DEPENDENCIES" "$INFERX_MODEL_DIR" "$INFERX_TOKENIZER" \
  "$INFERX_FROZEN_ASSETS" "$INFERX_DATA_CACHE" <<'PY'
from pathlib import Path
import json, sys
path, prefix, model, tokenizer, caches, data_cache = map(Path, sys.argv[1:])
mapping = {
    'assets/model/qwen3_8_27b.ninfer': str(model/'qwen3_8_27b.ninfer'),
    'assets/tokenizer': str(tokenizer),
    'envs/build': str(prefix/'envs/build'),
    'eval-envs': str(prefix/'eval-envs'),
    'tools': str(prefix/'tools'),
    'cache/modelscope': str(data_cache),
}
mapping.update({f'assets/{name}':str(caches/name)
                for name in ('weights025','weights041','weights047')})
path.write_text(json.dumps(mapping, indent=2)+'\n')
PY
cd "$INFERX_REPRO"
python3 prepare_assets.py
cat asset-verification.json
cat evaluation-dependency-check.json
```

Require `complete: true`, `content_sha_rehashed: true`,478 serving files and a zero evaluator dependency-check exit code. Cache025 contains256 files across64 layers,041 contains128 files across64 layers, and the047 serving subset contains94 files across47 layers. Preparation rehashes the entire model and caches. Routine service entrypoints mainly check sizes and directory membership, a narrower scope.

The IFBench addendum preserves the original300 prompts and344 instructions. The unchanged runner selects the first100 prompts and111 instructions. Parquet bytes from ModelScope revision `cf340e885a36d4b7b02692ddb7da242ea2643f4e` and Hugging Face revision `2e8a48de45ff3bf41242f927254ca81b59ca3ae2` match SHA `80037e4d99c39a55c1e2e7d5a863d8d9edeb5ebe136ba8c1e849f8c015027c6a`; all first100 input metadata also match runs105 and quality002. These are content-equivalent pinned acquisition revisions. The historical cache revision was not recorded and is not relabeled. Data staging sends no model requests and changes no evaluator code.

The actual ucloud3 run reused installed environments and read-only assets, rehashing all model/cache bytes. The first-install commands above have that separate scope. `asset-sources.json` explicitly maps every input; original node paths in the unmodified kit are replaced with those mappings.

## 4. Build normally and record the new executable's identity

The recorded toolchain is Linux x86_64, CUDA 13.2.51, GCC 13.4 and CMake/Ninja, with a Release build for CUDA architecture 89. Use an idle GPU and run one inference service at a time.

```bash
python3 "$INFERX_REPRO/build_owner.py"
cat "$INFERX_REPRO/build-job.json"
cat "$INFERX_REPRO/registered-CPU-CTest.log"
cat "$INFERX_REPRO/runtime-identity.json"
```

The controller configures `source/` into a new `build/` and normally compiles `ninfer-serve` plus the selected scheduler tests. It does not copy old objects or replace device archives. `build-job.json` preserves the full CMake configuration and executed arguments; `normal-build.log` contains compiler output. Require a `complete` build, seven passing registered CPU tests, matching source identities before and after compilation, and the new executable SHA plus 55 runtime-library identities.

These CPU tests cover selected scheduler contracts. GPU behavior, performance and quality require the subsequent independent runs; identical source does not transfer an old executable's scores.

## 5. Bind the portable entrypoints to the rebuilt executable

```bash
python3 "$INFERX_REPRO/bind_rebuilt_recipe.py"
cat "$INFERX_REPRO/rebuilt-recipe-binding.json"
cd "$INFERX_REPRO"
python3 -B -m unittest discover -s entrypoint/tests -p "test_*.py" -v
```

This copies the new executable to `artifacts/100-k3-rebuilt-20261018/ninfer-serve`, adds recipe `100-k3-u3-47-rebuilt-20261018.json` and its index entry, and records the actual executable and source hashes. The original `100-k3-u3-47.json` and historical executable identity are preserved. The new recipe starts with no inherited performance or quality results; use the new outputs to assess the run.

The server and both clients must select this new recipe. The official runner remains frozen at `d80df390…`, with unchanged input data, original request namespace, sampling and statistics. `entrypoint-runtime-freeze.json` records the bound package. Original recipes are preserved and the new recipe supplies its own qualification metadata. The r2 entrypoints have18 passing CPU checks, including identity and readiness; these do not replace GPU evaluation.

## 6. Start the service and run C14 performance

In both terminals, first run `cd "$INFERX_REPRO"`; set the same `INFERX_REPRO` in a new shell. Start the service in the foreground in terminal 1:

```bash
python3 -B entrypoint/serve_candidate.py \
  --recipe entrypoint/candidate_recipes/100-k3-u3-47-rebuilt-20261018.json \
  --profile perf-c14 --root "$PWD" --port 18540 \
  --log-dir "$PWD/service-perf-c14-001"
```

Run in terminal2; the r2 entrypoint verifies the same service and waits for actual `/models` readiness. A listening port alone does not establish that model loading has completed:

```bash
OPENAI_API_KEY=EMPTY python3 -B entrypoint/perf_candidate.py \
  --recipe entrypoint/candidate_recipes/100-k3-u3-47-rebuilt-20261018.json \
  --concurrency 14 --root "$PWD" \
  --service-launch "$PWD/service-perf-c14-001/launch.json" \
  --base-url http://127.0.0.1:18540/v1 \
  --output-dir "$PWD/performance-C14-001"

cat performance-C14-001/evaluation-identity.json
cat performance-C14-001/summary.json
```

The configuration is client concurrency 14, native capacity 13, KV121856, context65536, a 2048-token prefill chunk and ordinary MTP K3. Retain all 137 requests: 128 measured, 8 warmup and 1 probe; cold 8192-token inputs, 512-token outputs, temperature 0 and ignore_eos. Admission waiting remains part of TTFT, and the initial measured requests are retained.

The entrypoint verifies the local service PID, start time, executable, recipe, arguments, environment and listening socket. After evaluation, it verifies the canonical SHA of all 137 actual requests. The frozen parser does not emit `meets_slo` in `summary.json`, so the entrypoint receipt records `null`. Read `metrics.time_to_first_token.p95` and `metrics.inter_token_latency.p95`, both in milliseconds, and require ≤10000 and ≤40 respectively. Both conditions must pass. A valid SLO failure is retained; a zero process exit code does not imply an SLO pass.

## 7. Stop the performance service, then evaluate the original 100 quality set

After performance completes, use `Ctrl+C` in terminal 1 to stop that service normally. Confirm its original PID and listening socket are gone before starting the quality service. Run only one service on the GPU at a time.

Terminal 1:

```bash
python3 -B entrypoint/serve_candidate.py \
  --recipe entrypoint/candidate_recipes/100-k3-u3-47-rebuilt-20261018.json \
  --profile quality100 --root "$PWD" --port 18541 \
  --log-dir "$PWD/service-quality100-002"
```

Once ready, terminal 2:

```bash
OPENAI_API_KEY=EMPTY python3 -B entrypoint/accuracy_candidate.py \
  --recipe entrypoint/candidate_recipes/100-k3-u3-47-rebuilt-20261018.json \
  --root "$PWD" \
  --service-launch "$PWD/service-quality100-002/launch.json" \
  --base-url http://127.0.0.1:18541/v1 \
  --output-dir "$PWD/quality100-002"

cat quality100-002/evaluation-identity.json
cat quality100-002/evaluation/summary.json
```

Quality uses client/native concurrency 13, KV532480 and ordinary K3. The original first 100 IFBench prompts contain 111 instruction constraints. The generation configuration overrides only `max_tokens=32768`, preserving default thinking, model sampling and natural EOS. Score all 100 prompts, retaining empty final answers, formatting failures and other errors.

`evaluation-identity.json` records process completion; the original EvalScope report supplies prompt-level and instruction-level strict/loose scores. Stop the quality service normally when finished. These are separate evaluations with different KV capacities, even though they share the executable.

## 8. Locate and verify the outputs

All paths below are relative to `INFERX_REPRO`.

| Evidence | File or directory |
|---|---|
| Actual build commands, exit codes and compiler output | `build-job.json`, `configure.log`, `normal-build.log` |
| Source before/after, new executable and runtime libraries | `actual-source-before.json`, `actual-source-after.json`, `runtime-identity.json` |
| Full model/cache content checks | `model-verification.json`, `asset-verification.json`, `deployment-layout.json` |
| Live service identities and request logs | `service-perf-c14-001/`, `service-quality100-002/` |
| Performance summary and raw records | `performance-C14-001/summary.json`, `performance-C14-001/c14/` |
| Quality summary and configuration | `quality100-002/evaluation/summary.json`, `quality100-002/evaluation/accuracy/{task_config,resolved_config}.json` |
| Predictions, reviews and original scores | `quality100-002/evaluation/accuracy/native/{predictions,reviews,reports}/` |
| Portable package and recipe identity | `entrypoint-runtime-freeze.json`, `rebuilt-recipe-binding.json` |

Compute each request's `TPOT=(E2E−TTFT)/(output tokens−1)` before taking the linear P95. Throughput uses the original profiling phase boundaries, not the envelope of request timestamps. For quality, verify all 100 prompts, 111 constraints, the four scores, request errors, empty finals and truncation counts.

This guide uses two foreground terminals and does not depend on the original experiment owners. The kit’s evidence directory preserves the actual prior performance wrapper and r2 quality commands and receipts. The first quality attempt timed out before generating any questions and remains archived. The r2 change only waits on read-only `/models` while checking the same process identity; it adds no generation probe and does not retry evaluated prompts.

Report results separately when executable, assets, sampling or service settings differ. When comparing with historical 100/105, show both identities and the observed differences. Preserve a new SLO failure without altering historical passing curves.

## 9. Actual independent reproduction results

The completed ucloud3 rebuild has executable SHA `664cfbb989da8de5aaf58b755ecb68717a966c6992a3e377a8b04e2b2f1f7518`, with source `c976692d…`. The model and all 478 existing serving-cache files were fully rehashed. These results do not use regenerated caches.

| Evaluation | Measured result | Original evidence |
|---|---|---|
| C14 performance, client 14 / native 13, KV121856 | Original 128+8+1; 0 errors; TTFT P95 **9.89641735465 s**, TPOT P95 **38.28584995440313 ms**, both SLOs pass; throughput **304.89665937584215 tok/s** | [Raw JSON](../../results/three-node-optimization/ucloud3/selected100-single-node-reproduction-20261018/performance-delivery/performance-C14-001/summary.json), [independent audit](../../results/three-node-optimization/ucloud3/selected100-single-node-reproduction-20261018/root-C14-independent-audit.json) |
| Original 100 quality, client/native 13, KV532480 | Prompt strict **90/100**, loose **96/100**; instruction strict **100/111**, loose **106/111**; 0 errors, 1 empty final, 1 length stop | [Raw JSON](../../results/three-node-optimization/ucloud3/selected100-single-node-reproduction-20261018/quality-delivery/quality100-002/evaluation/summary.json), [independent audit](../../results/three-node-optimization/ucloud3/selected100-single-node-reproduction-20261018/root-quality-independent-audit.json) |

Throughput divides 65,536 output tokens by the 214.944959168-second profiling window. The original quality JSON rounds instruction scores to 0.9009 and 0.955. The integer numerators above come from all 100 per-prompt reviews, not from reversing rounded decimals. The empty final and length stop are the same prompt, key 88, which reaches the 32768-token output limit and remains in the 100-prompt denominator.

The first quality startup timed out before model loading completed and generated zero questions; its failure remains archived. The r2 readiness check took three attempts and 32.947901 seconds, with zero generation probes. Both performance and quality services were reaped normally. Historical e4ed scores 85/88 and this run’s 90/96 remain separate observations; no paired repeated experiment attributes their difference to performance optimization or quantization.

## 10. Regenerate quantized caches from public inputs

The [complete quantization workflow](quantization-reproduction-en.md) covers frozen archive 002, per-file FP8 verification, four training/two held-out inputs, fitting dependencies, three fresh cache stages, new-source service binding and original performance/quality evaluation. It does not require the old-cache attachment. Its results remain separate from this guide’s 664 old-cache reproduction; consult the [delivery index](../DELIVERY.md) for actual completion.

Run 116 update: all 239 matrices/478 serving files have been regenerated and match the original selected 478 SHA and byte counts, with zero differences. The normally rebuilt executable is `e919f180…878f5`, bound by its own 116-fresh47 recipe. See [completed quantization outputs](quantization-reproduction-en.md#completed-116-quantization-outputs). Its independent C14 run completed: TTFT P95 9.86221024060 s, TPOT P95 39.42860192427 ms, both SLOs pass, and throughput 305.53286821935 tok/s; all 128 measurements succeeded with zero errors. Independent original 100 quality completed: prompt strict 85/100 and loose 88/100, instruction 95/111 and 99/111, zero errors. Key 50 is both empty-final and length-limited at 32768 tokens and remains fully scored. The original verifier independently checked all 100; the service exited normally. These are e919’s own results, not inherited 664 scores. See [116 performance results](quantization-reproduction-en.md#actual-c14-performance-of-the-new-116-executable) and [116 quality terminal evidence](quantization-reproduction-en.md#original-100-quality-of-the-new-116-executable).
