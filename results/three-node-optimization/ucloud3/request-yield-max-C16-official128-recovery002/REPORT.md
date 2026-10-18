# Conservative request-yield max — original C16/128 FAIL

Default-OFF two-host-file scheduler candidate on parent4b6484; formal N5OFF/K3/fusionON/outputVerifyOFF. The estimate is max(original batch remaining rounds, own cumulative committed-token estimate). Fast rows never reduce the original estimate. FCFS/deadline formulas and model operators/Graph are unchanged.

|Metric|2362 historical same-machine reference|Current max|
|---|---:|---:|
|TTFT P95 seconds|9.855908|9.837810|
|TPOT P95 milliseconds|48.784071|45.744885|
|Original phase output tokens/s|311.768386|315.733530|

TPOT improves 6.230%, throughput +1.272%, but **SLO FAIL**. Single fixed-seed run; no stability claim. All128 measured +8warmup +1probe, allcold8192/512,0errors. Initial16:5TTFT/5TPOT violations; later112:0TTFT/18TPOT violations. Allstartup retained.

Actual137 normalized payload SHA `62f94272362623a9d776cf99711ab67411957a1d7799b7987583b2570df4b021` equals reference except salt, with separately matching warmup/profiling cohorts.

Same-P95-request decomposition (native per-request timings, not GPU critical path): TTFT8.934411s queue/prepare/control +.903399s ownPF; TPOT20.915960ms participating/shared decode +24.828925ms remainder. No capture: remainder cannot all be attributed to PF.136 unique client/native associations use original10ms absolute-first gate, max1.097656ms and minimum next-candidate gap885.154ms; finish residuals retained.

Binary `837f0f1f98e4d45901a2c521e145d00e01716c3880e0679b530854c827f2d925`; source `2bf64891c77d0c2e478d7f1648a76ae794b8b1c16852b3533a14ffc1134cdd01`/1020files. Only two host headers changed,1018 identical; actual normal incremental build6.93s, no object/archive swapping. Existing2policy CPU programs and4newflag/case runs pass; parent fullCTest reused by explicit Root instruction. No fresh model-numerics/quality claim.

Raw archive `4e810e4c4927e67dfb20cf40e7d1f9990371d9a296d990a9ff54c6a11821d872`:8,367,339bytes/99files verified locally. Every owned build/formal handle actualmissing; driver/telemetry exits0/reaped, GPUempty/1MiB/450W. Original WAN transfer failures and001 zero-request preflight failure remain.

Root chose079 as subsequent development baseline. `../request-yield-max-to079-handoff-001/` contains the exact2file patch/tests and actual079 preimage/zero-fuzz proof.2362 stays historical reference; no rollback or extra u3GPU run.
