# Selected build 100 campaign

Campaign configurations on two nodes; C14 client14/native13, including admission waiting, two separate runs; KV=max(65536,8704*client_c). No same-node concurrency-only effect or stable-capacity guarantee.

| Node | Client / native | Run | KV | TTFT P95 s | TPOT P95 ms | Output tok/s | Phase denominator s |
|---|---:|---:|---:|---:|---:|---:|---:|
| ucloud3 | 4 / 4 | 1 | 65536 | 2.269116713 | 22.891142453 | 205.818228774 | 318.416888486 |
| ucloud3 | 8 / 8 | 1 | 69632 | 5.921628374 | 29.532079870 | 274.203847326 | 239.004669844 |
| ucloud1 | 12 / 12 | 1 | 104448 | 9.593670567 | 36.234647029 | 299.757157815 | 218.630308873 |
| ucloud1 | 13 / 13 | 1 | 113152 | 9.706068063 | 38.181167186 | 302.968101002 | 216.313201896 |
| ucloud3 | 14 / 13 | 1 | 121856 | 9.893642785 | 39.929241927 | 306.038802971 | 214.142779817 |
| ucloud3 | 14 / 13 | 2 | 121856 | 9.821661092 | 39.141798364 | 309.951742085 | 211.439366526 |

Separate failed boundary configurations (not added to the six-point main curve):

| Node | Client / native | TTFT P95 s | TPOT P95 ms | Output tok/s | SLO |
|---|---:|---:|---:|---:|---|
| ucloud1 | 14 / 14 | 9.724916388 | 41.022342138 | 306.063528093 | FAIL |
| ucloud1 | 15 / 13 | 9.960208729 | 41.881277425 | 309.661792244 | FAIL |
| ucloud3 | 16 / 16 | 9.967465982 | 51.220646405 | 316.742964197 | FAIL |
| ucloud3 | 16 / 13 | 13.275426042 | 42.327509392 | 310.629804028 | FAIL |

C16/native13 is a separate admission-capacity diagnostic, not a passing capacity point. All raw measurements remain retained.

Quality: MEASURED_SEPARATELY_IN_EVALUATION105_ORIGINAL100; see selected-quality.json for raw-audited quality scores and distinct configuration.

Pending: []

Excluded setup attempt: [{"path": "results/three-node-optimization/ucloud3/c12-c16-latency-20261018/C4-native4-bounded100", "reason": "KV capacity below max-context: rejected by CLI before model loading/requests; native_done=0. Not a measured performance failure. The legal run uses KV65536.", "evidence_status": "Audited original exclusion summary and server CLI error log; 0 measured requests.", "sources": {"summary": {"path": "results/three-node-optimization/ucloud3/c12-c16-latency-20261018/C4-native4-bounded100/summary.json", "sha256": "2948b6c17f203e5bc1d992fdd37236c35ce94b27b61ec73effe93ddc8303a4cc"}, "server_log": {"path": "results/three-node-optimization/ucloud3/c12-c16-latency-20261018/C4-native4-bounded100/delivery/raw/candidate/server.log", "sha256": "dd2445ce4dc69517f2e01e33ce3ad7518eff149d069a87c489cc75c1c90f2b25"}}}]

Rebuild: `uv run submission/tooling/inferx-submission/scripts/build_selected_campaign.py`

No edits to Typst, historical result-details data, published figures, or final PDFs. These are independent integration artifacts.
