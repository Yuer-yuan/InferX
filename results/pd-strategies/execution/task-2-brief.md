### Task 2: Actual PD-Multiplexing Qwen adaptation and qualification

**Files:** Extend inferx/sglang_pd.py, scripts/prepare_pd_strategies.py, tests/test_sglang_pd.py, configs/pd-strategies.json; record results/pd-strategies/compatibility on HDD and locally.

**Interfaces:** `qwen_split_prefill(self,input_ids,positions,forward_batch,split_interval,input_embeds=None)` matches ModelRunner's real split entry and text-only wrapper hierarchy. Overlay contains narrowly scoped Qwen/backend/topology corrections as demonstrated necessary; manifests enumerate all deviations from upstream.

- [ ] Write failing interval/residual/interleaving/final-logits/unsupported-mode tests, run RED, implement adapter, run GREEN.
- [ ] Use real installed ModelRunner CPU entry to validate call compatibility, then owned GPU1 Green Context and full27B code/tool/full128K probes. Record native split/stream evidence and every failure.
- [ ] Resolve demonstrated hybrid dispatch/topology/capacity issues with targeted regressions; qualify matched actual pool or document why none fits. Preserve exact precision/context and restore service.
- [ ] Run full repository suite and commit adapter plus actual qualification evidence.

Expected: a correctness/capacity-qualified actual PD path, or reproducible adapted failure with efficacy explicitly unknown.

