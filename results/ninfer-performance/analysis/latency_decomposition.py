"""Recompute offline pilot timing evidence using only the Python standard library.

Run from any directory. Writes latency-decomposition.json beside this file.
Native/client records are selected by the profiling wall-clock interval, not
joined by request ID. This analysis neither changes nor replays a workload.
"""
from pathlib import Path
import hashlib
import json
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {}


def read(path):
    raw = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return raw.decode()


def analyze(group, row, expected):
    case = ROOT / group / "runs" / row["run"] / row["case"]
    tier = case / f"c{row['concurrency']}"
    clients = [json.loads(line) for line in read(tier / "profile_export.jsonl").splitlines()]
    clients = [r for r in clients if r["metadata"]["benchmark_phase"] == "profiling"]
    assert len(clients) == expected, (tier, len(clients))
    begin_ms = min(r["metadata"]["request_start_ns"] for r in clients) / 1e6
    end_ms = max(r["metadata"]["request_end_ns"] for r in clients) / 1e6
    native = [json.loads(line) for line in read(case / "requests.jsonl").splitlines()]
    native = [r for r in native if r.get("event") == "request_done"
              and r["result"]["completion_tokens"] == 512
              and begin_ms <= r["timestamp_unix_ms"] <= end_ms + 100]
    assert len(native) == expected, (tier, len(native))
    for r in native:
        assert r["result"]["prompt_tokens"] == 8192
        assert r["result"]["prefix_cache_hit_tokens"] == 0
    # The counter is per-request wall time spent in shared decode batches, plus
    # synchronized MTP replay tails. Summing counters would double-count batches.
    timings = [r["timings_seconds"] for r in native]
    residuals = [t["total"] - t["ttft"] - t["decode"] for t in timings]
    native_post = median(t["total"] - t["ttft"] for t in timings)
    for r in clients:
        for k in ("request_latency", "time_to_first_token"):
            assert r["metrics"][k]["unit"] == "ms"
    client_post = median((r["metrics"]["request_latency"]["value"]
                          - r["metrics"]["time_to_first_token"]["value"]) / 1000
                         for r in clients)
    result = {
        "group": group, "run": row["run"], "case": row["case"],
        "concurrency": row["concurrency"], "records": len(native),
        "median_native_prefill_s": median(t["prefill"] for t in timings),
        "median_native_decode_counter_s": median(t["decode"] for t in timings),
        "median_native_post_first_wall_s": native_post,
        "median_client_post_first_wall_s": client_post,
        "native_client_post_first_median_difference_ms": (native_post - client_post) * 1000,
        "median_residual_s": median(residuals),
        "profiling_window_unix_ms": [begin_ms, end_ms],
        "native_request_ids": sorted(r["request"]["request_id"] for r in native),
    }
    drafted = sum(r.get("speculative", {}).get("drafted_tokens", 0) for r in native)
    result["weighted_draft_acceptance"] = (
        sum(r["speculative"]["accepted_tokens"] for r in native) / drafted if drafted else None
    )
    # This short pilot admits exactly eight requests into eight slots with no
    # replacement admissions. The simple later-prefill sum is invalid in the
    # slots4 runs or a longer closed-loop workload; deliberately restrict it.
    if row.get("slots") == 8 and row["concurrency"] == expected == 8:
        native.sort(key=lambda r: r["timestamp_unix_ms"] / 1000
                    - r["timings_seconds"]["total"] + r["timings_seconds"]["ttft"])
        ranks = []
        for i, r in enumerate(native):
            t = r["timings_seconds"]
            residual = t["total"] - t["ttft"] - t["decode"]
            later_prefill = sum(n["timings_seconds"]["prefill"] for n in native[i + 1:])
            ranks.append({"first_token_rank": i + 1, "request_id": r["request"]["request_id"],
                          "residual_s": residual, "later_prefill_sum_s": later_prefill,
                          "difference_ms": (residual - later_prefill) * 1000})
        result["later_prefill_comparison"] = ranks
        result["median_abs_residual_minus_later_prefill_ms"] = median(
            abs(r["difference_ms"]) for r in ranks)
    return result


if __name__ == "__main__":
    rows = []
    for group in ("pilot-initial", "pilot-refinement"):
        comparison = json.loads(read(ROOT / group / "comparison.json"))
        assert comparison["label"] == "PILOT"
        for row in comparison["rows"]:
            assert row["status"] == "complete"
            rows.append(analyze(group, row, comparison["requests_per_tier"]))
    output = {
        "scope": "Offline diagnostic, 19 pilot tiers; excludes all warmups and compatibility probes.",
        "selection": "Native 512-token completions in each client profiling request interval; +100ms end-log tolerance.",
        "join_limit": "Interval selection and aggregate median cross-check, not an exact native/client request-ID join.",
        "counter_limit": "Native decode includes shared batch duration per request and MTP replay tails; never sum as exclusive GPU time.",
        "attribution_limit": "Later-prefill sum restricted to 8-request/8-slot pilot batches; no kernel timeline or bandwidth counters.",
        "rows": rows, "source_sha256": dict(sorted(SOURCES.items())),
    }
    target = Path(__file__).with_name("latency-decomposition.json")
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {len(rows)} tiers to {target}")
