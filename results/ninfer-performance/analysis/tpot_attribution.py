"""Recompute TPOT components from frozen official-r1 evidence, without inference.

Run from any directory. The P95 components use the SAME native request ranks
and interpolation weights; they are not independently computed percentiles.
Client/native requests are selected by phase window, not joined by request ID.
"""
from pathlib import Path
import gzip
import hashlib
import json
import math
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {}


def read(path):
    stored = path if path.exists() else path.with_name(path.name + ".gz")
    raw = stored.read_bytes()
    SOURCES[str(stored.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return (gzip.decompress(raw) if stored.suffix == ".gz" else raw).decode()


def rows(path):
    return [json.loads(line) for line in read(path).splitlines() if line.strip()]


def percentile(values, q):
    values = sorted(values)
    rank = (len(values) - 1) * q
    low, high = math.floor(rank), math.ceil(rank)
    weight = rank - low
    return values[low] * (1 - weight) + values[high] * weight


def main():
    official = ROOT / "official-r1"
    summary = json.loads(read(official / "analysis.json"))
    native = [r for r in rows(official / "runs/official-r1-service/requests.jsonl")
              if r.get("event") == "request_done"]
    output = {
        "scope": "Offline reanalysis of all 400 formal requests; no new GPU runs or service changes.",
        "selection": "Client profiling windows, 512-token native completions, +100ms native end-log tolerance; no exact client/native ID join.",
        "definitions": {
            "tpot_ms": "1000 * (total - ttft) / 511",
            "decode_component_ms": "1000 * decode_counter / 511",
            "residual_component_ms": "1000 * (total - ttft - decode_counter) / 511",
            "p95_components": "Both components interpolated at the same two requests ranked by native TPOT; NOT sum of component P95s.",
        },
        "limits": [
            "Decode counter includes full shared-batch duration for each participating request and MTP tails; not exclusive GPU kernel time.",
            "Residual includes other work and control/publication overhead; formal logs have no per-chunk intervals to assign every residual millisecond to prefill.",
            "Time before first token outside own prefill/prepare also includes intervening decode; it is NOT a pure admission-queue timer.",
            "A before-first-token queue delay contributes to TTFT and cancels from total-minus-TTFT, although its scheduling can affect other requests.",
        ],
        "rows": [],
    }
    for phase in summary["performance"]:
        concurrency = phase["concurrency"]
        clients = [r for r in rows(official / f"runs/official-r1/performance/c{concurrency}/profile_export.jsonl")
                   if r["metadata"]["benchmark_phase"] == "profiling"]
        assert len(clients) == 100
        begin = min(r["metadata"]["request_start_ns"] for r in clients) / 1e6
        end = max(r["metadata"]["request_end_ns"] for r in clients) / 1e6
        assert [begin, end] == phase["profiling_window_unix_ms"]
        selected = [r for r in native if r["result"]["completion_tokens"] == 512
                    and begin <= r["timestamp_unix_ms"] <= end + 100]
        assert len(selected) == 100
        assert len({r["request"]["request_id"] for r in selected}) == 100
        details = []
        for r in selected:
            assert r["result"]["prompt_tokens"] == 8192
            assert r["result"]["prefix_cache_hit_tokens"] == 0
            t = r["timings_seconds"]
            residual = t["total"] - t["ttft"] - t["decode"]
            assert residual >= 0
            details.append({
                "native_request_id": r["request"]["request_id"],
                "native_tpot_ms": 1000 * (t["total"] - t["ttft"]) / 511,
                "decode_component_ms": 1000 * t["decode"] / 511,
                "residual_component_ms": 1000 * residual / 511,
                "residual_s": residual,
                "residual_fraction": residual / (t["total"] - t["ttft"]),
                "ttft_s": t["ttft"], "prefill_s": t["prefill"],
                "prepare_s": t["prepare"],
                "pre_first_outside_own_prefill_prepare_s": t["ttft"] - t["prefill"] - t["prepare"],
            })
        details.sort(key=lambda r: r["native_tpot_ms"])
        rank = (len(details) - 1) * .95
        low, high = math.floor(rank), math.ceil(rank)
        weight = rank - low
        components = {key: details[low][key] * (1 - weight) + details[high][key] * weight
                      for key in ("native_tpot_ms", "decode_component_ms", "residual_component_ms")}
        assert abs(components["native_tpot_ms"] - components["decode_component_ms"] - components["residual_component_ms"]) < 1e-10
        client_values = []
        for r in clients:
            metrics = r["metrics"]
            assert all(metrics[key]["unit"] == "ms" for key in ("request_latency", "time_to_first_token"))
            client_values.append((metrics["request_latency"]["value"] - metrics["time_to_first_token"]["value"]) / 511)
        client_p95 = percentile(client_values, .95)
        assert abs(client_p95 - phase["tpot_p95_ms"]) < 1e-7
        difference = components["native_tpot_ms"] - client_p95
        assert abs(difference) < .01, difference
        output["rows"].append({
            "concurrency": concurrency, "requests": len(details),
            "client_tpot_p95_ms": client_p95,
            "native_p95_components_ms": components,
            "native_minus_client_p95_ms": difference,
            "p95_bracketing_native_requests": [details[low], details[high]],
            "p95_upper_rank_weight": weight,
            "median_residual_s": median(r["residual_s"] for r in details),
            "median_residual_fraction": median(r["residual_fraction"] for r in details),
            "median_prepare_s": median(r["prepare_s"] for r in details),
            "median_pre_first_outside_own_prefill_prepare_s": median(r["pre_first_outside_own_prefill_prepare_s"] for r in details),
            "weighted_draft_acceptance": phase["weighted_draft_acceptance"],
        })
    output["source_sha256"] = dict(sorted(SOURCES.items()))
    target = Path(__file__).with_name("tpot-attribution.json")
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["rows"], indent=2))


if __name__ == "__main__":
    main()
