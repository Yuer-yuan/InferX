"""Plot the archived baseline; no benchmark execution or result selection.

Run with matplotlib==3.10.7. The SVG is public; the preview PNG is ignored.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
REFERENCE = json.loads((ROOT / 'reference-row-verification.json').read_text())
CAPACITY = json.loads((ROOT / 'capacity8-verification.json').read_text())
assert REFERENCE['all_twelve_rows_archived_and_verified']
assert CAPACITY['both_cases_wire_and_parser_verified']


def values(case):
    result = case['original_parser']
    return (result['output_token_throughput'],
            result['metrics']['time_to_first_token']['p95'] / 1000,
            result['metrics']['inter_token_latency']['p95'])


plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none'})
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
first_five = sorted(REFERENCE['rows'][:5], key=lambda row: row['reference']['concurrency'])
x = [row['reference']['concurrency'] for row in first_five]
styles = [('vllm', 'vLLM, 4 slots', '#2563eb', 'o'),
          ('sglang', 'SGLang, 4 slots', '#d97706', 's')]
for engine, label, color, marker in styles:
    series = [values(row['cases'][engine]) for row in first_five]
    repeat = REFERENCE['rows'][5]
    assert repeat['reference']['concurrency'] == 2
    for index, axis in enumerate(axes):
        axis.plot(x, [row[index] for row in series], color=color, marker=marker,
                  linewidth=1.7, label=label)
        axis.scatter([2], [values(repeat['cases'][engine])[index]], s=120,
                     facecolors='none', edgecolors=color, linewidths=1.5,
                     label='C2 repeat, 100 requests' if engine == 'vllm' else None,
                     zorder=4)

for index, axis in enumerate(axes):
    cases = CAPACITY['cases']
    axis.plot([case['concurrency'] for case in cases],
              [values(case)[index] for case in cases], color='#7c3aed',
              marker='D', linestyle='--', linewidth=1.5, label='vLLM, 8 slots')
    axis.set_xticks([1, 2, 3, 4, 8])
    axis.set_xlabel('Client concurrency')
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.2)
    axis.spines[['top', 'right']].set_visible(False)

axes[0].set(title='Total output throughput', ylabel='Output tokens / second')
axes[1].set(title='Time to first token, P95', ylabel='Seconds')
axes[2].set(title='Time per output token, P95', ylabel='Milliseconds')
for axis, threshold in [(axes[1], 10), (axes[2], 40)]:
    axis.axhline(threshold, color='#15803d', linestyle=':', linewidth=1.5)
    axis.text(1.15, threshold + 1, f'SLO limit: {threshold}', color='#15803d', fontsize=9)

fig.suptitle('8192 input / 512 output tokens; FP8, no MTP; one 48 GB RTX 4090 per framework',
             fontsize=12, y=0.99)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.03), ncol=4,
           frameon=False)
fig.text(0.5, 0.005,
         '32K service window. Main C1/2/3/4/8 counts: 12/24/24/32/24. Hollow rings: separate C2/100 row. '
         'One matrix; no confidence intervals.', ha='center', fontsize=8)
fig.tight_layout(rect=(0, 0.13, 1, 0.94))
fig.savefig(ROOT / 'performance.svg', metadata={'Date': None})
preview = ROOT.parents[1] / 'artifacts/framework-c4/performance-preview.png'
preview.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(preview, dpi=130)
print(ROOT / 'performance.svg')
print(preview)
