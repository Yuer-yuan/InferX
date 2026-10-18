# Actual single-node reproduction commands

Deployment root: `/workspace/InferX/reproductions/selected100-single-node-20261018`.
Local evidence root: `results/three-node-optimization/ucloud3/selected100-single-node-reproduction-20261018/`.

This is a clean **engine-source build and deployment** on the existing ucloud3 OS. Model, already-quantized 025/041/047 caches, tokenizer, build toolchain and evaluation environments are explicitly reused through symlinks. It is not a clean OS installation or rerun of quantization. All model and 478 serving-weight contents are SHA-verified anew.

## Build

The complete frozen100 source is the 1023-file `source-identity.json` map, aggregate `c976692d9657408308193b50772c700382a6e243d9da353165205dc5252f76a8`. Extract `source-and-entrypoint-stage.tgz` into a new empty deployment root. Never copy old object files or an old serve executable.

```sh
cd /workspace/InferX/reproductions/selected100-single-node-20261018
export PATH=/workspace/InferX/envs/build/bin:/usr/local/cuda-13.2/bin:$PATH
export CC=/workspace/InferX/envs/build/bin/x86_64-conda-linux-gnu-gcc
export CXX=/workspace/InferX/envs/build/bin/x86_64-conda-linux-gnu-g++
export LD_LIBRARY_PATH=/workspace/InferX/envs/build/lib:/usr/local/cuda-13.2/lib64
export PKG_CONFIG_PATH=/workspace/InferX/envs/build/lib/pkgconfig
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cmake -S source -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER="$CXX" \
  -DCMAKE_PREFIX_PATH=/workspace/InferX/envs/build \
  -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=ON \
  -DNINFER_BUILD_BENCHMARKS=OFF -DNINFER_ENABLE_UI=OFF \
  -DFETCHCONTENT_SOURCE_DIR_XGRAMMAR_SOURCE=/workspace/InferX/source/xgrammar_source-src
cmake --build build --target ninfer-serve \
  ninfer_slo_bounded_decode_membership_test ninfer_slo_completion_policy_test \
  ninfer_slo_prefill_queue_policy_test ninfer_slo_budget_policy_test --parallel 4
ctest --test-dir build --output-on-failure \
  -R '^ninfer_slo_(bounded_decode_membership.*|completion_policy_test|prefill_queue_policy_test|budget_policy_test)$'
```

The actual owned execution uses `python3 build_owner.py`, which executes those commands, verifies source before/after, records compiler and full build logs, checks the seven registered tests and freezes the actual executable plus 55 dynamic dependency identities. Do not substitute original e4ed if the rebuilt SHA differs.

Run `python3 prepare_assets.py` once in the new deployment root to create the explicit model/tokenizer/cache/build/evaluation symlinks, verify model and all 478 cache hashes against the frozen inventory, inventory tokenizer/xgrammar and check both Python3.12 evaluation environments. These source assets are read-only reuse; no model/cache content is modified. `deployment-layout.json` lists every resolved source path.

After actual build and asset checks complete, run `python3 bind_rebuilt_recipe.py` once. It preserves original e4ed recipes and adds a new recipe/index entry binding this build's actual SHA. Its initial performance and quality statuses are unmeasured. `entrypoint-runtime-freeze.json` is the complete relocated-package freeze, including the one-line qualification-metadata fix; original runner bytes remain d80df390.

## Performance: real portable entrypoint

Terminal 1:

```sh
python3 -B entrypoint/serve_candidate.py \
  --recipe entrypoint/candidate_recipes/100-k3-u3-47-rebuilt-20261018.json \
  --profile perf-c14 --root "$PWD" --port 18540 \
  --log-dir "$PWD/service-perf-c14-001"
```

Once the process owns the listening socket, terminal 2:

```sh
OPENAI_API_KEY=EMPTY python3 -B entrypoint/perf_candidate.py \
  --recipe entrypoint/candidate_recipes/100-k3-u3-47-rebuilt-20261018.json \
  --concurrency 14 --root "$PWD" \
  --service-launch "$PWD/service-perf-c14-001/launch.json" \
  --base-url http://127.0.0.1:18540/v1 --output-dir "$PWD/performance-C14-001"
```

Client14/native13/KV121856, K3/PF2048, 19 fixed flags, original128+8warm+1probe, cold8192/512, temperature0/ignoreEOS and original9270C12 namespace are unchanged. TTFT includes admission queuing. No initial request is removed. Stop this service normally and confirm its exact PID/start has terminated before quality.

## Quality: independent original100 evaluation after readiness repair

The first attempt `quality100-001` failed before any generated question: the server listened while model/KV loading continued, and the frozen runner’s 15-second `/models` call timed out. The original attempt and original package are preserved. The repaired `entrypoint-r2` performs an identity-verified read-only `/models` wait (no generation probes) before invoking the unchanged runner; actual readiness took 3 attempts/32.947901 seconds. Engine binary and recipe SHA remain unchanged. This is the sole generated100-question evaluation, in a new `quality100-002` directory.

Terminal 1:

```sh
python3 -B entrypoint-r2/serve_candidate.py \
  --recipe entrypoint-r2/candidate_recipes/100-k3-u3-47-rebuilt-20261018.json \
  --profile quality100 --root "$PWD" --port 18541 \
  --log-dir "$PWD/service-quality100-002"
```

Terminal 2:

```sh
OPENAI_API_KEY=EMPTY python3 -B entrypoint-r2/accuracy_candidate.py \
  --recipe entrypoint-r2/candidate_recipes/100-k3-u3-47-rebuilt-20261018.json \
  --root "$PWD" --service-launch "$PWD/service-quality100-002/launch.json" \
  --base-url http://127.0.0.1:18541/v1 --output-dir "$PWD/quality100-002"
```

Original first100 IFBench, native/client13, KV532480, generation configuration only max_tokens32768, natural EOS/default thinking. All100 outcomes remain in scoring, including empty answers.

For the recorded execution, `python3 evaluation_owner.py` completed performance and preserved the pre-generation quality timeout. After repairing readiness, `python3 quality_r2_owner.py` runs the two corrected quality commands above; both wrappers validate live process identity and control only their own child pidfds. Neither changes the runner or scoring math. The original13 plus5 new readiness CPU tests passed. Actual argv/process identities are in `evaluation-job.json`, `quality-r2-job.json` and each output’s launch/evaluation identity. New results are independent from historical100 performance/quality. The shared `demo/selected100` source now includes the repair, so future fresh deployments use it for both profiles.
