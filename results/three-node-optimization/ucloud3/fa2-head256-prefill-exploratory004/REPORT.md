# FA2 HD256 complete-operator screen: rejected

Fixed selected official FA2 commit965c9212; current2362 public GQA baseline. Q2048/Hq24/Hkv4/D256, K2048/4096/6144/8192. Candidate includes append, paged INT8 KV dequantization, Q/output casts and FA2, not GEMM/kernel alone.

All1536 fixed ABBA samples retained. Graph gains 0.98664/1.01138/1.02128/1.02131; geometric mean1.01005 fails predeclared1.10. No integration or tile sweep. Raw20files and all384 paired positions locally verified; same frozen probe binaryf15696c2.

Original12-case numeric gate remains FAIL: one case exceeded unrounded FP64 max-error bound in both baseline and candidate, with21 sampled values having no representable BF16 within that bound. All276824064 conversion bits and structural guards/cache/repeat gates passed. Root authorized exploratory timing separately. The prepared BF16-aware CPUv2 contract was not executed after performance rejection; no numeric promotion/task-quality claim.

All owner/child handles terminal/reaped:462207/47244716 and462213/47244755; terminal receipt records empty GPU. All earlier namespace/zero-arena and oracle draft failures remain in recovery003 numeric delivery. Large numeric raw1.589GB stays remote with146file SHA manifest. Compact numeric is in ../fa2-head256-prefill-recovery003/delivery; complete timing/Graph/source receipts are in delivery/content.
