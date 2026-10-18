# 050 original first100 complete

Original owner naturally completed at2026-10-17 11:58:08UTC. Original collector, service and checkpoint observer exited; owned service released GPU at11:58:06UTC. No replacement requests or repeated collection.

| Original EvalScope1.13.0 metric | Score | Prompt-cluster95% CI |
|---|---:|---:|
| Prompt strict | 86/100 =86% | 79–92% |
| Instruction strict | 97/111 =87.3874% | 80.7339–93.0435% |
| Prompt loose | 90/100 =90% | 84–95% |
| Instruction loose | 101/111 =90.9910% | 85.2174–95.6897% |

All100 predictions recomputed with original verifiers match original review scores.100 attempts,100 HTTP200,100 natural stop,0 request errors,0 length truncations;1 empty final answer is retained and scored.100 nonempty reasoning fields. Bootstrap uses10000 prompt-cluster samples,seed42,linear percentiles.

Configuration: same40b3 runtime/sourceed94 as the corresponding C32 candidate, all-layer FFN+input S4, MTPK3, original first100, batch2, max32768, default thinking and model sampling. Actual payloads contain only model/messages/max_tokens. Native sampling records seed42/temp1/top_k20/top_p≈.95, penalties0; all100 uniquely join HTTP/native completion and cache0. No native token-ID hash was captured. Exact per-layer dynamic route counts were not captured; frozen binary/source, actual environment and cache-readback evidence are retained.

This is the40b3 candidate observation, not a quality result for later9270 output-S4 changes, a paired OFF comparison, pure quantization attribution or reproduction of the published79.5 recipe.

Local compact results:

- `terminal-evidence-001/CPU-score-001/summary.json`
- `terminal-evidence-001/CPU-score-001/per-sample.jsonl`
- `terminal-evidence-001/CPU-score-001/failures-empty-length-strictfail.jsonl`
- `terminal-evidence-001/CPU-http-native-join-001/{summary.json,per-completion.jsonl}`

Raw preserved in `sealed-raw-001/raw050.tgz`:2,787,404 bytes,578 members,SHA256 `173a77eca2160599998773c52fd8e55d398b2e0b33f580b5c7d168b7c343070c`. Local readback verified every member SHA. Runtime binary is excluded from this small raw archive; its actual identity and complete original remote file remain retained. All original requests/responses/attempts/reasoning/content/predictions/reviews/native records and CPU results are included.

Original predictionSHA `9aa0a96e749fa3dfa44e082f73e35227ec61ca7d046403a1f9ed3b61f5cb9c42`; reviewSHA `fcb6eae68506a51379534e3ab6af0961b3f1a9dc03e1fe486764f76ff0b17256`.
