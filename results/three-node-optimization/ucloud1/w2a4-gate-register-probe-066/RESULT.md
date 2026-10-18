# Register W2 gate result

Private single-candidate probe complete; no product model changes. Six numeric cases and768 CPU group/FMA checks pass, every output bit matches the same-logical-W4 control, eager and three reset Graph replays included. All1200 samples and54 archived files were locally recomputed/hashed.

| T | S4 control ms | Register W2 ms | Speed ratio | Time change |
|---:|---:|---:|---:|---:|
| 64 | 0.245759994 | 0.173056006 | 1.420118x | -29.5833% |
| 128 | 0.244736001 | 0.178176001 | 1.373563x | -27.1967% |
| 2048 | 0.963584006 | 1.065984011 | 0.903939x | +10.6270% |

All10blocks agree in direction for each mode/shape. Candidate190registers, zero local/stack/spill,40960B shared, API resource bound1CTA/SM. Control212registers/45056B shared/API1. These are resource limits, not measured occupancy/roofline.

Reject large-prefill use. Retain smallT hardware potential only. 062 expanded-shared results are preserved and were not rerun. Comparing this run to062 is historical, not a simultaneous pair. Current valid pair is new register candidate versus unchanged S4 reference within066.

If every one of64gate calls had the measured same shape/work, simple savings sum to4.6531ms atT64 or4.2598ms atT128 per target traversal. This is a conditional microbenchmark budget, not per-token/request gain or an SLO prediction. Draft/down/attention/GDN and interfering prefill remain.

Synthetic weights use the same four-level logical values in both storage formats. This does not demonstrate task accuracy versus original W4, qualify real063 weights, all64 layers, partialT routes, or end-to-end improvement. No model integration or new service was started.

All owned processes actually missing, their direct children reaped exit0, GPU empty. See receipts/actual-terminal.json. Source identity9a4faba74386e806a8a72920255967bfc6e4b943fd4ebd0bd6ab64d43a8f8970; compact996027b8122e76347d1f2af4f546b2ece783e4989a4cb349155c8f1eb46b0eea.
