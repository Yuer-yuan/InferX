"""Standalone publication-style screening plots (English labels for portability)."""
import argparse
import json
from pathlib import Path


def main(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result = json.loads(args.summary.read_text())
    policies = ['fixed1', 'fixed2', 'fixed4', 'feedback-fair']
    colors = ['#8a98ac', '#65809f', '#385a80', '#25877a']
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), constrained_layout=True)
    for col, mode in enumerate(('burst', 'stagger')):
        groups = [result['groups'][f'eval/{mode}/{p}/all'] for p in policies]
        for row, (metric, label) in enumerate([
                ('mean_capped_solution_s', 'Mean capped solution time (s; failures = 120 s)'),
                ('observed_tokens_per_task', 'Observed output tokens / task (lower bound)')]):
            ax = axes[row, col]
            values = [g[metric] for g in groups]
            bars = ax.bar(policies, values, color=colors)
            ax.set_ylabel(label, fontsize=9); ax.set_title(mode.capitalize())
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
            ax.set_ylim(0, max(values)*1.22)
            ax.tick_params(axis='x', labelsize=9)
            for bar, value, group in zip(bars, values, groups):
                annotation = f'{value:.1f}'
                if row == 0:
                    annotation += f'\n{group["solved"]}/{group["trials"]} solved'
                ax.text(bar.get_x()+bar.get_width()/2, value, annotation,
                        ha='center', va='bottom', fontsize=8)
    fig.suptitle('Qwen3.8-27B · single RTX 4090 · repository completion\n'
                 + result['decision'].replace('_', ' ') + f' | dev-selected: {result["selected_baseline"]}', fontsize=11)
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(args.output.with_suffix('.'+suffix), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('summary', type=Path)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
