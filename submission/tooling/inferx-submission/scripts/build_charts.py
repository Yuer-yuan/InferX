#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.7"]
# ///
"""Export measured slide charts as vector PDFs; authoritative input: results.json.
Run: uv run submission/tooling/inferx-submission/scripts/build_charts.py
"""
import json
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from cost_charts import build_cost_charts, build_admission_charts
from build_request_trace import build_request_trace
from build_result_details import build_result_details
from build_selected_campaign import audit as audit_selected_campaign, plots as plot_selected_campaign

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "submission/basic/presentation/figures"
RESULTS = ROOT / "submission/track-a/data/results.json"
ACCENT, GREEN, RED, MUTED = "#7E2635", "#287657", "#B44B45", "#657180"
plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.7,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": "#D5DBE1", "grid.linewidth": 0.55,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False,
})
CJK = FontProperties(family=["Liberation Sans", "Droid Sans Fallback"], size=13)


def export(name, lang, fig):
    target = OUT / f"{name}-{lang}.pdf"
    # Live compilation and an explicit export may overlap; keep their writes distinct.
    temporary = target.with_suffix(f'.{os.getpid()}.tmp.pdf')
    fig.savefig(temporary, metadata={
        "Title": f"InferX / {name} / {lang}",
        "Creator": "Matplotlib; audited evidence in submission/track-a/data/",
        "CreationDate": None, "ModDate": None,
    })
    temporary.replace(target)
    plt.close(fig)


def axes(width=140, height=74):
    fig, ax = plt.subplots(figsize=(width / 25.4, height / 25.4))
    fig.subplots_adjust(left=.115, right=.955, bottom=.23, top=.88)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    ax.tick_params(length=3, width=.6, pad=4)
    return fig, ax


def series(name, lang, rows, key, labels, ymax, ticks, xlabel, bars=False, target=None, digits=1):
    fig, ax = axes()
    if any('\n' in label for label in labels):
        fig.subplots_adjust(bottom=.31)
    vals = [r[key] for r in rows]
    xs = list(range(len(vals)))
    if bars:
        ax.bar(xs, vals, width=.43, color="#EDDDE1", edgecolor=ACCENT, linewidth=1.0)
    else:
        ax.plot(xs, vals, "o-", color=ACCENT, linewidth=1.5, markersize=5)
    for x, y in zip(xs, vals):
        ax.annotate(f"{y:.{digits}f}", (x, y), xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", color=ACCENT, fontsize=13)
    ax.set(xlim=(-.5, len(xs)-.5), ylim=(0, ymax), yticks=ticks, xticks=xs)
    ax.set_xticklabels(labels, fontproperties=CJK if lang == "zh" else None)
    ax.set_xlabel(xlabel, labelpad=7, fontproperties=CJK if lang == "zh" else None)
    if target is not None:
        ax.axhline(target, color=GREEN, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(.98, target / ymax + .025, f"SLO {target:g}", color=GREEN, fontsize=11,
                transform=ax.transAxes, ha="right", va="bottom", backgroundcolor="white")
    export(name, lang, fig)


def capacity(lang, rows):
    fig, ax = axes(174, 95)
    fig.subplots_adjust(left=.13, right=.95, bottom=.20, top=.87)
    ax.add_patch(plt.Rectangle((0, 0), 10, 40, color="#F0F7F3", zorder=0))
    ax.axvline(10, color=GREEN, linewidth=1, linestyle=(0, (4, 3)))
    ax.axhline(40, color=GREEN, linewidth=1, linestyle=(0, (4, 3)))
    for row in rows:
        ok = row["joint_slo"]
        x, y = row["ttft_s"], row["tpot_ms"]
        ax.scatter(x, y, marker="o" if ok else "x", color=GREEN if ok else RED, s=45, zorder=3)
        offset = (10, -18) if row["c"] == 12 else (10, 7)
        ax.annotate(f"C{row['c']}", (x, y), xytext=offset, textcoords="offset points",
                    color=GREEN if ok else RED, fontsize=14)
    ax.text(.7, 10, "两项均达标" if lang == "zh" else "Both targets met",
            color=GREEN, fontproperties=CJK if lang == "zh" else None)
    ax.set(xlim=(0, 28), ylim=(0, 125), xticks=[0, 5, 10, 15, 20, 25], yticks=[0, 40, 80, 120])
    ax.set_xlabel("首字延迟 P95（s）" if lang == "zh" else "TTFT P95 (s)", labelpad=8,
                  fontproperties=CJK if lang == "zh" else None)
    ax.text(0, 1.06, "单Token生成延迟 P95（ms）" if lang == "zh" else "TPOT P95 (ms/token)",
            transform=ax.transAxes, ha="left", fontproperties=CJK if lang == "zh" else None)
    export("capacity", lang, fig)


def concurrency_trends(lang, rows):
    fig, panels = plt.subplots(1, 3, figsize=(292 / 25.4, 78 / 25.4))
    fig.subplots_adjust(left=.055, right=.985, bottom=.23, top=.82, wspace=.35)
    specs = [
        ('ttft_s', 'TTFT P95 (s)', (0, 11.5), [0, 5, 10], 10),
        ('tpot_ms', 'TPOT P95 (ms/token)', (0, 66), [0, 20, 40, 60], 40),
        ('tps', '总吞吐量 (tok/s)' if lang == 'zh' else 'Aggregate output (tok/s)',
         (0, 350), [0, 100, 200, 300], None),
    ]
    for ax, (key, title, limits, ticks, target) in zip(panels, specs):
        xs, ys = [r['c'] for r in rows], [r[key] for r in rows]
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)
        ax.plot(xs, ys, 'o-', color=ACCENT, linewidth=1.6, markersize=5)
        ax.set(xlim=(min(xs)-.35, max(xs)+.35), xticks=xs, ylim=limits, yticks=ticks)
        ax.set_title(title, loc='left', pad=14, fontproperties=CJK if lang == 'zh' else None)
        ax.set_xlabel('并发请求数' if lang == 'zh' else 'Client concurrency', labelpad=8,
                      fontproperties=CJK if lang == 'zh' else None)
        for x, y in zip(xs, ys):
            below = key == 'ttft_s' or (key == 'tpot_ms' and y <= 40)
            ax.annotate(f'{y:.2f}', (x, y), xytext=(0, -17 if below else 9),
                        textcoords='offset points', ha='center', color=ACCENT, fontsize=12)
        if target:
            ax.axhline(target, color=GREEN, linewidth=1, linestyle=(0, (4, 3)))
            ax.text(.02, target / limits[1] + .03, f'SLO {target:g}', transform=ax.transAxes,
                    color=GREEN, fontsize=11, backgroundcolor='white')
    export('concurrency-trends', lang, fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(RESULTS.read_text())
    details = build_result_details()
    p = {r["id"]: r for r in data["performance"]}
    baseline = [p[f"zip-c{c}"] for c in [1, 2, 4, 8]]
    versions = [p[i] for i in ["s4-c32", "input-c32", "output-c32"]]
    tradeoff = [p[i] for i in ["fusion-c16", "fusion-client16-native12"]]
    scheduling = [p[i] for i in ["fusion-c16", "yield-c16"]]
    for lang in ["zh", "en"]:
        concurrency = "并发请求数" if lang == "zh" else "Client concurrency"
        series("baseline-throughput", lang, baseline, "tps", ["1", "2", "4", "8"], 130,
               [0, 40, 80, 120], concurrency)
        series("baseline-latency", lang, baseline, "tpot_ms", ["1", "2", "4", "8"], 110,
               [0, 40, 80], concurrency, target=40)
        labels = ["前馈层", "+ 输入投影", "+ 输出投影"] if lang == "zh" else ["FFN", "+ Input", "+ Output"]
        stages = "量化覆盖范围（历史版本）" if lang == "zh" else "Quantized matrices (historical builds)"
        series("versions-throughput", lang, versions, "tps", labels, 430,
               [0, 100, 200, 300, 400], stages, bars=True)
        series("versions-latency", lang, versions, "tpot_ms", labels, 150,
               [0, 40, 80, 120], stages, target=40)
        labels = ["16", "12"] if lang == "zh" else ["Active 16", "Active 12"]
        clients = "引擎并发上限（客户端16并发）" if lang == "zh" else "Client concurrency fixed at 16"
        series("tradeoff-ttft", lang, tradeoff, "ttft_s", labels, 18,
               [0, 5, 10, 15], clients, bars=True, target=10)
        series("tradeoff-tpot", lang, tradeoff, "tpot_ms", labels, 63,
               [0, 20, 40, 60], clients, bars=True, target=40)
        labels = ["批次估计", "单请求估计"] if lang == "zh" else ["Batch estimate", "Per-request estimate"]
        scope = "剩余生成轮次的估计方法" if lang == "zh" else "Progress estimate only (2362 → 837f)"
        series("schedule-ttft", lang, scheduling, "ttft_s", labels, 13,
               [0, 5, 10], scope, bars=True, target=10)
        series("schedule-tpot", lang, scheduling, "tpot_ms", labels, 63,
               [0, 20, 40, 60], scope, bars=True, target=40)
        capacity(lang, [p[f"output-c{c}"] for c in [12, 16, 32]])
        concurrency_trends(lang, details['performance'])
    build_cost_charts(export, series, CJK, ROOT)
    build_request_trace()
    campaign = audit_selected_campaign()
    plot_selected_campaign(campaign)
    build_admission_charts(export, CJK, campaign)
    print(f"Exported {len(list(OUT.glob('*.pdf')))} vector PDFs to {OUT}")


if __name__ == "__main__":
    main()
