# Selected100 portable reconstruction assets

This is u1's source/provenance handoff, not a new performance or quality run. Root coordinates actual clean build on u3 and fresh quantization on u2. No peer connection or GPU workload was made for this handoff.

## Obtain the exact measured478 cache

Fresh uncompressed archive on u1, created from read-only originals and SHA-checked **while writing every file**:

- Path `/workspace/InferX/experiments/ucloud1-selected100-reproduction-assets-r1/selected100-u3-47-caches.tar`
- Bytes `12082360320`; cache payload `12081971200`.
- SHA256 `a8936aa7d39e9f971256a30b415f6f271f4f70418d0705233c0c84bfabfc3fb7`.
- 025=256files;041=128;047=94. Exactly the u3-47 manifest, not the larger512-file set.
- `job.json`, `launch.json`, `terminal.json`, `files.jsonl` preserve owner588520/start50408265, exit0 and actual PID missing. No existing file was overwritten.

From a machine with access to the configured `ucloud` SSH host:

```bash
scp ucloud:/workspace/InferX/experiments/ucloud1-selected100-reproduction-assets-r1/selected100-u3-47-caches.tar .
printf '%s\n' 'a8936aa7d39e9f971256a30b415f6f271f4f70418d0705233c0c84bfabfc3fb7  selected100-u3-47-caches.tar' | sha256sum -c -
mkdir -p "$REPRO_ROOT/assets"
tar -xf selected100-u3-47-caches.tar -C "$REPRO_ROOT/assets"
(cd "$REPRO_ROOT/assets" && sha256sum -c SHA256SUMS)
python3 verify_assets.py --root "$REPRO_ROOT/assets"
```

Use a fresh asset directory. These commands acquire frozen measured assets; **they do not count as new quantization**. Fresh generation uses QUANTIZATION-HANDOFF.md and must retain its own source/inputs/output comparisons.

## Source, model and build

`selected100-source.tar.gz` contains all1023 original files, actual SHA verified against aggregate `c976692d9657408308193b50772c700382a6e243d9da353165205dc5252f76a8`. Archive SHA `9fae3ed7cbfc8e2dfbce46caa58da392db0b99409f0b4275aa61618bd3a1b0dc`;6,668,935B. This is the complete patched source, not merely upstream+API patch. Source origin and original model identities remain in `original-support/submission/model-manifest.json`.

```bash
mkdir -p "$REPRO_ROOT"
tar -xzf selected100-source.tar.gz -C "$REPRO_ROOT"
mkdir -p "$REPRO_ROOT/assets/model"
curl -fL 'https://huggingface.co/neroued/Qwen3.8-27B-NInfer/resolve/3526913004b1cf552cb57b88d6a5c6f5e4a89a70/qwen3_8_27b.ninfer' -o "$REPRO_ROOT/assets/model/qwen3_8_27b.ninfer"
printf '%s\n' "eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e  $REPRO_ROOT/assets/model/qwen3_8_27b.ninfer" | sha256sum -c -
python3 extract_tokenizer.py --model "$REPRO_ROOT/assets/model/qwen3_8_27b.ninfer" --out "$REPRO_ROOT/assets/tokenizer"
git clone https://github.com/mlc-ai/xgrammar.git "$REPRO_ROOT/xgrammar"
git -C "$REPRO_ROOT/xgrammar" checkout 4546988d1cf51670a10eafb8795d37e4a2f0d833
bash build_source100.sh "$REPRO_ROOT" "$TOOLCHAIN" "$CUDA_ROOT" "$REPRO_ROOT/xgrammar"
```

Actual toolchain is CUDA13.2.51, GCC13.4, SM89/Linux x86_64. Full conda explicit package URL lock and build/eval pip freezes are in `remote-provenance/dependency-locks/`; use the `.lock` explicit file, not the human formatted `.txt` export. `original-support/scripts/` contains original install scripts. CUDA/driver are external prerequisites; this handoff neither installs nor changes them. New normal build identity must be recorded and cannot be assumed equal to the old binary `e4ed1bf4...` solely because source matches.

The official clients remain AIPerf0.13.0 and EvalScope[ifbench]1.13.0 in separate Python3.12 environments; original scripts and exact dependency freezes are included. NLTK three-resource URL/SHA list is included. Use existing `demo/selected100` original runner/data/portable entrypoints for execution; no evaluator was rewritten. The local snapshot is `selected100-entrypoints.tar.gz`. Its recipe pins the original binary: a newly compiled binary needs an explicit newly identified recipe, not a silently changed SHA.

## Fresh quantization and remaining limits

`QUANTIZATION-HANDOFF.md` gives the actual024→025→038→041→outputcapture→047 chain and exact frozen token IDs. `OUTPUT-CAPTURE-REBUILD.md` plus full original9f5a source archive makes the047 observer parent rebuildable. All original fit/capture scripts, manifests, arithmetic dependencies, explicit old build commands and real output identities are retained.

`FP8-acquisition-manifest.json` contains72 content-file hashes/per-file revisions and2 original mount metadata entries. `fetch_fp8.py --directory NEW_FP8_DIR` downloads only content files at those revisions and requires their full expected hashes. Only config.json was re-fetched here: two official ModelScope fixed-revision endpoints returned the exact mounted SHA. No full fresh FP8 download is claimed; `.mv` says master, and there is no asserted single repository snapshot pin.

The public base NInfer artifact was originally downloaded. Its original author conversion invocation has not been recovered. The included source has a Qwen3.8 converter, but identical published-v2 reconstruction has not been established. The025 original fit also needs old020/021 L31 CPU-reference arrays available on u2; these large reference arrays and historical activation tensors are not copied into this small package. Root/u2 must record fresh captures/fit, preserve all numerical gates, and distinguish any necessary path/control adaptation from arithmetic changes.
