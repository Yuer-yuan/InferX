# Fresh fitted-cache binding (CPU only)

This tool creates a **new**, unmeasured portable deployment. It never changes the old selected100 package, recipe, assets or scores. It hashes actual selected packed/scales contents and the executable; old cache hashes and old quality/performance evidence are not inherited.

Requirements: Python 3.10+, the repaired selected100 template package, new 025/041/047 fitter directories, an executable with recorded source identity, the fixed model, tokenizer and installed build/evaluation environments. `--fit-evidence` must identify the new fitter's completed serving manifest and frozen-fit checks; the tool records its hash but does **not** certify numerical equivalence/finite scales. The producing worker must pass those gates before service launch.

```sh
python3 bind_fresh_caches.py \
  --template-package /path/to/reproduction-kit/entrypoint \
  --weights025 /path/to/fresh/weights025 \
  --weights041 /path/to/fresh/weights041 \
  --weights047 /path/to/fresh/weights047 \
  --binary /path/to/new/ninfer-serve \
  --source-sha256 c976692d9657408308193b50772c700382a6e243d9da353165205dc5252f76a8 \
  --recipe-id 116-fresh47 \
  --model /path/to/qwen3_8_27b.ninfer \
  --tokenizer /path/to/tokenizer \
  --build-env /path/to/envs/build \
  --eval-envs /path/to/eval-envs \
  --tools /path/to/tools \
  --modelscope-cache /path/to/cache/modelscope \
  --fit-evidence /path/to/fresh-serving-manifest.json \
  --output /path/to/new-deployment
```

Use the actual source identity of the supplied executable. The command rehashes the fixed model (18.21 GB), the selected caches (12.08 GB, exactly 478 files), and the executable. Source caches must be finalized and remain immutable. Root output must not exist; a failed output is retained and must not be reused.

025 gate/down and 041 input use all original 64 layers. 047 output accepts either the original 47 layers or all 64 fitted layers, then selects the **same immutable 47-layer mask** as selected100. Missing/unknown layer directories, unknown serving-directory files, and wrong packed/scales sizes are rejected. Known fitter auxiliaries `expanded.i8`, `fit-equivalence.json`, `hessians.json`, `fit-frozen-before-heldout.json` are accepted but never copied or hashed as serving files. New deployment symlinks expose only the selected packed/scales files; no expanded tensor copy is made. Top-level fitter reports are ignored.

The new `entrypoint/candidate_recipes/assets-u3-47.json` contains actual fresh SHA256 values. Its recipe/index bind the fresh asset contract, supplied source identity and actual binary SHA, retaining the original 19 environment values, K3, profiles, model format and runner. `entrypoint/MANIFEST.json` freezes this new executable package; `evidence/fresh-binding.json` records the original template identity and fresh provenance. The generated package has no old measured report, old catalog or old numerical-equivalence receipt.

After independent numerical/native qualification, use the unchanged portable entrypoint:

```sh
python3 /path/to/new-deployment/entrypoint/serve_candidate.py \
  --root /path/to/new-deployment \
  --recipe /path/to/new-deployment/entrypoint/candidate_recipes/116-fresh47.json \
  --profile perf-c14 --port 18550 --log-dir /path/to/new-deployment/service-C14
# In another shell after launch (entrypoint waits for model readiness):
python3 /path/to/new-deployment/entrypoint/perf_candidate.py \
  --root /path/to/new-deployment \
  --recipe /path/to/new-deployment/entrypoint/candidate_recipes/116-fresh47.json \
  --service-launch /path/to/new-deployment/service-C14/launch.json \
  --base-url http://127.0.0.1:18550/v1 --concurrency 14 \
  --output-dir /path/to/new-deployment/performance-C14
```

No GPU/build/HTTP action is performed by the binder. The caller owns service lifecycle and is responsible for fresh performance and accuracy results; those are never inferred from this CPU check.

Validation: 14 CPU tests, including an isolated end-to-end fixture with all 478 file names, full64-to47 selection, the real portable contract loader, executable/asset SHA bindings, unknown/missing/size rejection, known auxiliaries and old-template immutability. Fixture bytes are deliberately tiny; this is contract validation, not real cache numerical qualification.
