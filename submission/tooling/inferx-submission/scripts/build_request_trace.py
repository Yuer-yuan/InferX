#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.7"]
# ///
"""Draw real request lifecycles from existing logs; CPU-only, no inference.

uv run submission/tooling/inferx-submission/scripts/build_request_trace.py

Outputs: bilingual vector PDFs, audited request-trace.json and Chrome Trace JSON.
The selected cohort is the first N measured requests by engine admission time.
Admission is an engine event, not HTTP arrival. Lines are lifetimes, not GPU spans.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TRACE = ROOT / "results/three-node-optimization/ucloud1/verify-w2-gate-085/trace-r1"
EVENTS = ("admission", "first_token", "request_finish")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def collect(trace_dir, count):
    sources = {}

    def read(relative):
        path = trace_dir / relative
        content = path.read_bytes()
        name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        sources[name] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        return content.decode()

    trace = [json.loads(line) for line in read("delivery/control/scheduler.jsonl").splitlines() if line.strip()]
    clients = [json.loads(line) for line in read("delivery/raw/candidate/c16/profile_export.jsonl").splitlines() if line.strip()]
    clients = [r for r in clients if r["metadata"]["benchmark_phase"] == "profiling"]
    if len(clients) != 128:
        raise ValueError(f"Expected 128 measured requests, found {len(clients)}")
    base_ns = min(r["metadata"]["request_start_ns"] for r in clients)
    anchors = []
    for name in ("start", "end"):
        samples = json.loads(read(f"delivery/control/clock-anchor-{name}.json"))["samples"]
        anchors.append(min(samples, key=lambda s: s["epoch_after_ns"] - s["epoch_before_ns"]))
    offsets = [(s["epoch_before_ns"] + s["epoch_after_ns"]) / 2 - s["monotonic_ns"] for s in anchors]
    drift_ms = abs(offsets[1] - offsets[0]) / 1e6
    if drift_ms >= 1:
        raise ValueError(f"Clock bridge drift exceeds 1 ms: {drift_ms}")

    def relative_epoch_ms(monotonic_ms):
        fraction = (monotonic_ms * 1e6 - anchors[0]["monotonic_ns"]) / (anchors[1]["monotonic_ns"] - anchors[0]["monotonic_ns"])
        return (monotonic_ms * 1e6 + offsets[0] + fraction * (offsets[1] - offsets[0]) - base_ns) / 1e6

    events = {name: {} for name in EVENTS}
    for row in trace:
        name = row["event"]
        if name in events:
            rid = row["request_id"]
            if rid in events[name]:
                raise ValueError(f"Duplicate {name} for request {rid}")
            events[name][rid] = row
    joined, seen = [], set()
    for client in clients:
        meta, metrics = client["metadata"], client["metrics"]
        start_ms = (meta["request_start_ns"] - base_ns) / 1e6
        first_ms = start_ms + metrics["time_to_first_token"]["value"]
        finish_ms = (meta["request_end_ns"] - base_ns) / 1e6
        candidates = []
        for rid, first in events["first_token"].items():
            finish = events["request_finish"].get(rid)
            if finish and finish["tokens"] == 512:
                delta_first = relative_epoch_ms(first["start_ms"]) - first_ms
                delta_finish = relative_epoch_ms(finish["start_ms"]) - finish_ms
                if abs(delta_first) <= 5 and abs(delta_finish) <= 5:
                    candidates.append((rid, delta_first, delta_finish))
        if len(candidates) != 1:
            raise ValueError(f"Non-unique trace match: {meta['conversation_id']}: {candidates}")
        rid, df, de = candidates[0]
        if rid in seen:
            raise ValueError(f"Trace request reused: {rid}")
        seen.add(rid)
        raw = [events[name][rid] for name in EVENTS]
        times = [r["start_ms"] for r in raw]
        if times != sorted(times) or metrics["output_sequence_length"]["value"] != 512:
            raise ValueError(f"Invalid event order or output length: {rid}")
        joined.append({"request_id": rid, "conversation_id": meta["conversation_id"],
                       "raw_events": raw, "client_match_error_ms": {"first_token": df, "request_finish": de}})
    joined.sort(key=lambda r: (r["raw_events"][0]["start_ms"], r["request_id"]))
    if not 1 <= count <= len(joined):
        raise ValueError("--requests must be between 1 and the measured cohort size")
    origin = joined[0]["raw_events"][0]["start_ms"]
    for request in joined:
        request["times_s"] = {row["event"]: (row["start_ms"] - origin) / 1000 for row in request["raw_events"]}
    return {"scope": "085 C16/K5 traceON diagnostic; measured cohort only; engine lifecycles, not exclusive GPU spans",
            "selection": f"First {count} measured requests sorted by engine admission time",
            "identity": json.loads(read("delivery/raw/candidate-identity.json")),
            "audit": {"measured": len(clients), "uniquely_joined": len(seen), "clock_bridge_drift_ms": drift_ms,
                      "match_tolerance_ms": 5, "max_first_token_error_ms": max(abs(r["client_match_error_ms"]["first_token"]) for r in joined),
                      "max_finish_error_ms": max(abs(r["client_match_error_ms"]["request_finish"]) for r in joined)},
            "origin_monotonic_ms": origin, "selected_ids": [r["request_id"] for r in joined[:count]],
            "requests": joined, "sources": sources}


def chrome_trace(data):
    out = []
    for request in data["requests"]:
        rid, times = request["request_id"], request["times_s"]
        out.append({"ph": "M", "name": "thread_name", "pid": 1, "tid": rid,
                    "args": {"name": f"Request {rid} / {request['conversation_id']}"}})
        out.append({"ph": "X", "name": "Engine request lifetime", "cat": "request", "pid": 1, "tid": rid,
                    "ts": times["admission"] * 1e6, "dur": (times["request_finish"] - times["admission"]) * 1e6,
                    "args": {"completion_tokens": 512, "conversation_id": request["conversation_id"]}})
        for name in EVENTS:
            out.append({"ph": "i", "s": "t", "name": name, "cat": "request", "pid": 1, "tid": rid,
                        "ts": times[name] * 1e6, "args": {"request_id": rid}})
    return {"traceEvents": out, "displayTimeUnit": "ms", "otherData": {"scope": data["scope"], "origin_monotonic_ms": data["origin_monotonic_ms"]}}


def draw(data, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"admission": "#657180", "first_token": "#7E2635", "request_finish": "#287657"}
    markers = {"admission": "o", "first_token": "D", "request_finish": "s"}
    selected = [r for r in data["requests"] if r["request_id"] in data["selected_ids"]]
    with plt.rc_context({"font.family": "Liberation Sans", "font.size": 15, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False}):
        for lang in ("zh", "en"):
            font = FontProperties(family=["Liberation Sans", "Droid Sans Fallback"], size=15)
            labels = ["开始处理", "首个Token", "请求完成"] if lang == "zh" else ["Engine admission", "First token", "Request finish"]
            fig, ax = plt.subplots(figsize=(276 / 25.4, 100 / 25.4))
            fig.subplots_adjust(left=.14, right=.97, bottom=.19, top=.82)
            for y, request in enumerate(selected):
                times = request["times_s"]
                ax.plot([times["admission"], times["request_finish"]], [y, y], color="#BCC7D0", lw=4, solid_capstyle="butt", zorder=2)
                for name in EVENTS:
                    ax.scatter(times[name], y, marker=markers[name], s=68, facecolor="white" if name == "admission" else colors[name],
                               edgecolor=colors[name], linewidth=1.3, zorder=3)
            ax.set_yticks(range(len(selected)), [f"{'请求' if lang == 'zh' else 'Request'} {r['request_id']}" for r in selected], fontproperties=font)
            ax.set_ylim(len(selected) - .35, -.65)
            right = max(r["times_s"]["request_finish"] for r in selected)
            ax.set_xlim(-.5, right + 1)
            ax.tick_params(axis="y", length=0, pad=9)
            ax.tick_params(axis="x", labelsize=14, colors="#657180")
            ax.spines["bottom"].set_color("#657180")
            ax.xaxis.grid(True, color="#E2E6EB", lw=.6)
            ax.set_axisbelow(True)
            ax.set_xlabel("相对时间（秒）" if lang == "zh" else "Time since first selected request was admitted (s)", fontproperties=font, labelpad=10)
            handles = [Line2D([], [], linestyle="none", marker=markers[name], markersize=8,
                              markerfacecolor="white" if name == "admission" else colors[name], markeredgecolor=colors[name], label=label)
                       for name, label in zip(EVENTS, labels)]
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, .99), ncol=3, frameon=False,
                       prop=font, handletextpad=.5, columnspacing=2)
            target = output_dir / f"request-trace-{lang}.pdf"
            temporary = target.with_suffix(f".{os.getpid()}.tmp.pdf")
            fig.savefig(temporary, metadata={"Title": f"Measured request lifecycle / {lang}",
                                             "Creator": "build_request_trace.py; original scheduler.jsonl", "CreationDate": None, "ModDate": None})
            temporary.replace(target)
            plt.close(fig)


def build_request_trace(trace_dir=DEFAULT_TRACE, output_dir=None, data_dir=None, requests=8):
    output_dir = output_dir or ROOT / "submission/basic/presentation/figures"
    data_dir = data_dir or ROOT / "submission/track-a/data"
    data = collect(Path(trace_dir).resolve(), requests)
    write_json(data_dir / "request-trace.json", data)
    write_json(data_dir / "request-trace.chrome.json", chrome_trace(data))
    draw(data, output_dir)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submission/basic/presentation/figures")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "submission/track-a/data")
    parser.add_argument("--requests", type=int, default=8)
    args = parser.parse_args()
    data = build_request_trace(args.trace_dir, args.output_dir, args.data_dir, args.requests)
    print(f"Verified {data['audit']['uniquely_joined']}/128 measured requests; plotted IDs {data['selected_ids']}; exported PDF and Chrome Trace.")


if __name__ == "__main__":
    main()
