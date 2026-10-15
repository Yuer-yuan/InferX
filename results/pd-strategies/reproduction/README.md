# Exact source overlays

These artifacts preserve the actual runtime bytes, including intermediate adaptations that were never separate Git commits. The four packages were reconstructed on HDD and each complete set of4372files matched its original inventory. See reconstruction-check.json.

The base is the pinned SGLang0.5.19 installation with the separately documented singleton sampler compatibility correction; see ../compatibility/green-context-model-qualification.json and ../../pd-baselines/compatibility/sglang-singleton-guard/provenance.json. This archive does not replace the pinned wheel/dependency environment or model weights.

Each directory contains the original overlay-manifest.json, inferx_pd_runtime.py, a unified source.patch relative to that base, and source-hashes.json.gz. To reconstruct, copy the base package into a fresh owned HDD directory as sglang/, excluding __pycache__ and *.pyc; apply the patch with patch -p1 in that directory; copy the archived runtime alongside sglang/; decompress the original inventory and verify the complete file set and every SHA. Preserve the original manifest. Do not edit an existing experiment overlay or the installed package.

The actual capture and reconstruction helpers, their outputs and the execution ledger are preserved in the execution archive. Existing immutable packages remain on vlm-server under /home/guokc/mypro/gnn-mount-sda1/InferX/pd-strategies/overlays. Normal control uses control-v2; normal simple uses responsive128-credit-v4; the initial PD mixed attempt uses pdmux-responsive128-credit-bound-v6; stream-private Marlin candidates use pdmux-responsive128-private-marlin-v8. SM/layer YAML text and SHA are embedded in each case configuration.
