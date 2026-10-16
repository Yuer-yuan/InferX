"""Plot observed request latency components; these are not GPU execution times."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('case')
    parser.add_argument('output', type=Path)
    parser.add_argument('--repetition', type=int, default=0)
    args = parser.parse_args()
    groups = ['short-control', 'cold-low', 'resume-high']
    data, sources = [], {}
    for name in ['config.json', 'campaign.json']:
        raw = (args.run/name).read_bytes()
        sources[str(args.run/name)] = hashlib.sha256(raw).hexdigest()
    configuration = next(c for c in json.loads((args.run/'config.json').read_text())
                         if c['id'] == args.case)
    rates = {g['name']:g['rate'] for g in
             json.loads((args.run/'campaign.json').read_text())['workload_definitions']}
    for group in groups:
        path = args.run / f'{args.case}-r{args.repetition}-{group}' / group / 'requests.jsonl'
        raw = path.read_bytes()
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        rows.sort(key=lambda row: (row.get('prompt_id') != 'tool-24576', row['scheduled_s']))
        data.append(rows)
    limit = max(row['scheduled_s'] + row['latency_s'] for rows in data for row in rows) * 1.04
    fig, axes = plt.subplots(1, 3, figsize=(13, 6.6), sharex=True)
    for ax, group, rows in zip(axes, groups, data):
        labels = []
        for index, row in enumerate(rows):
            long = row.get('prompt_id') != 'tool-24576'
            labels.append('128K long' if long else f'S{len(labels):02d}')
            start, duration = row['scheduled_s'], row['latency_s']
            first = row.get('arrival_ttft_s')
            if first is not None:
                ax.barh(index, first, left=start, height=.65, color='#3c78a8')
                ax.barh(index, max(0, duration-first), left=start+first,
                        height=.65, color='#e29c45')
            else:
                ax.barh(index, duration, left=start, height=.65, color='#b8b8b8')
            if row.get('functional_ok') is not True:
                ax.plot(start+duration, index, 'x', color='#b22222', markersize=5)
        ax.axvline(16, color='#555555', linestyle='--', linewidth=.8)
        label = 'resume' if group == 'resume-high' else group
        ax.set_title(f'{label} ({rates[group]:g} RPS)', fontsize=11)
        ax.set_yticks(range(len(rows)), labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, limit)
        ax.set_xlabel('Seconds since wave start')
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle(f'{configuration["engine"]}: observed request timelines '
                 f'(C{configuration["concurrency"]}, one diagnostic repetition)', fontsize=13)
    fig.legend(handles=[Patch(color='#3c78a8', label='Arrival to first output'),
                        Patch(color='#e29c45', label='First to final output')],
               loc='lower center', ncol=2, frameon=False, fontsize=9)
    fig.text(.5, .065, '24K short history; 128K long arrives at 16 s. '
             'Bars show API latency, including waiting, not GPU execution time.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, .95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ['.png', '.pdf']:
        fig.savefig(args.output.with_suffix(suffix), dpi=180)
    args.output.with_suffix('.json').write_text(json.dumps(dict(
        run=str(args.run), case=args.case, repetition=args.repetition,
        source_sha256=sources, limitation=__doc__), indent=2)+'\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
