"""Render the recorded pilot; requires Matplotlib (version in client freeze)."""
from pathlib import Path
import json
import matplotlib

matplotlib.use('Agg')
matplotlib.rcParams['svg.hashsalt'] = 'InferX-ninfer-pilot-20261016'
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
data = json.loads((root / 'comparison.json').read_text())
assert data['label'] == 'PILOT' and data['requests_per_tier'] == 8
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout='constrained')
panels = [
    ('output_tok_s', 'End-to-end output throughput', 'tokens/s', None),
    ('ttft_p95_s', 'TTFT P95', 'seconds', 10),
    ('tpot_p95_ms', 'TPOT P95', 'ms/token', 40),
]
for ax, (metric, title, unit, threshold) in zip(axes, panels):
    for case, label, color in [
        ('mtp0-slots4', 'MTP off', '#334155'),
        ('mtp4-slots4', 'MTP4', '#0284c7'),
        ('mtp7-slots4', 'MTP7', '#c2410c'),
    ]:
        rows = sorted((r for r in data['rows'] if r['case'] == case), key=lambda r: r['concurrency'])
        assert len(rows) == 3 and all(r['status'] == 'complete' for r in rows)
        ax.plot([r['concurrency'] for r in rows], [r[metric] for r in rows],
                '-o', label=label, color=color, linewidth=1.8, markersize=5)
    if threshold is not None:
        ax.axhline(threshold, linestyle='--', color='#64748b', linewidth=1)
        ax.text(7.9, threshold, ' reference SLO', ha='right', va='bottom', fontsize=8, color='#475569')
    ax.set(title=title, xlabel='Client concurrency', ylabel=unit, xticks=[1, 4, 8], ylim=(0, None))
    ax.grid(alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].legend(frameon=False, fontsize=9)
fig.suptitle('PILOT: 8192 input / 512 output tokens, 4 active slots, prefill chunk 2048', fontsize=11)
fig.supxlabel('One run, 8 measured requests per point; warmup excluded. Both latency SLOs must pass. No confidence intervals.', fontsize=8)
fig.savefig(root / 'pilot.png', dpi=180)
fig.savefig(root / 'pilot.svg', metadata={'Date': None})
