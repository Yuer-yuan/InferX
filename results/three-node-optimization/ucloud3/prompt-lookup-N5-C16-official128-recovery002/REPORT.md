# N5 priority: original C16/128 — FAIL

Same actual 137 payloads as parent2362 except cache salt; 128 measured + 8 warmup + 1 probe, all cold, 8192/512, no errors. Binary `4b6484e0...`, source `328fafc3...`.

|Metric|Parent2362|N5 priority|
|---|---:|---:|
|TTFT P95|9.855908 s|9.924643 s|
|TPOT P95|48.784071 ms|53.890755 ms|
|Phase output throughput|311.768386|319.119747|

Throughput +2.36%, TPOT worsens: no SLO promotion or independent-seed stability claim. Initial16 have 5 TTFT/6 TPOT violations; later112 have 0/28.

Same P95 TTFT interpolation: queue/prepare/control 9.021134 s + own prefill 0.903508 s. Same P95 TPOT: participating shared decode 18.978763 ms + remainder 34.911992 ms. No new trace: remainder cannot all be called competing PF or GPU time. Contributions and original128 request IDs are in `P95-contributor-index.json` / `measured-request-index.jsonl`.

136 long-request runtime lookup summaries plus one zero-count built-in startup warmup summary: attempts20614, overrides11680, drafted34941, target-licensed32823, full accepts10757. HTTP runner probe has no emitted runtime summary (serial2); the built-in service warmup is not official traffic. Runtime serial is not an exact HTTP request ID; no per-tail lookup match fabricated. These counts are before terminal clipping and not incremental acceptance over overwritten MTP.

Archive `33092970204aa13ffd7b14ae6f55a53ca5bb055ff6b86a7edf616075d94d73e0`, 8,373,732 bytes / 92 files fully verified locally. Owner/service/build/native handles actual missing; supervisor wait receipts retained; GPU empty (1 MiB), 450 W. Original zero-request preflight failure and cancelled early full build remain separate.
