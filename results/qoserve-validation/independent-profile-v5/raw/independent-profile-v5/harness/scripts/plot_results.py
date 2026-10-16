"""Plot repeated evaluation metrics from an exported summary (Matplotlib)."""
import argparse
import json
from pathlib import Path


def main(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = json.loads(Path(args.summary).read_text())
    widths = [1, 2, 4, 8]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for round_id, color in [(0, '#2463ad'), (1, '#d26b32')]:
        rows = [summary['groups'][f'eval/width{w}/round{round_id}'] for w in widths]
        axes[0].plot(widths, [r['mean_capped_solution_s'] for r in rows], 'o-', color=color,
                     label=f'Round {round_id} / GPU {round_id}')
        axes[1].plot(widths, [100*r['feedback_fraction_median'] if r['feedback_fraction_median'] is not None else float('nan') for r in rows], 'o-', color=color)
        axes[1].plot(widths, [100*r['feedback_fraction_p95'] if r['feedback_fraction_p95'] is not None else float('nan') for r in rows], 'o:', color=color, alpha=.7)
        axes[2].plot(widths, [r['observed_tokens']/r['trials'] for r in rows], 'o-', color=color)
    axes[0].set_ylabel('Mean capped solution time (s)')
    axes[0].set_title('Failures contribute the full 120 s')
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Winner feedback / solution time (%)')
    axes[1].set_title('Solid: median; dotted: p95')
    axes[1].axhline(5, color='gray', linestyle='--', linewidth=.8)
    axes[2].set_ylabel('Observed generated tokens / task')
    axes[2].set_title('Includes unsuccessful/cancelled work')
    for ax in axes:
        ax.set_xscale('log', base=2)
        ax.set_xticks(widths, [str(w) for w in widths])
        ax.set_xlabel('Active generation limit')
        ax.grid(axis='y', alpha=.2)
        ax.axvline(summary['selected_width'], color='#389476', alpha=.25, linewidth=5)
    fig.suptitle('Qwen3.8-27B UD-Q4_K_M / one RTX 4090 per run / HumanEval pilot', fontsize=11)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('summary')
    parser.add_argument('output')
    main(parser.parse_args())
