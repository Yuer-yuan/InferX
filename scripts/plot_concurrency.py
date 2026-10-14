"""Standalone observational plots; each point is one measured group, no inferred CI."""
import argparse
import json
from pathlib import Path


def main(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=json.loads(Path(summary).read_text())
    lookup={(g['case'],g['group']):g for g in data['groups']}
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    colors={0:'#64748b',4:'#0891b2',7:'#7c3aed'}
    for ax,group,title in zip(axes[0],['scaling-code','uniform-long-code'],['Short code (~0.5K input)','Long code (~48K input)']):
        for draft in [0,4,7]:
            observed=[(c,lookup[(f'd{draft}-c{c}',group)]) for c in [1,2,4] if (f'd{draft}-c{c}',group) in lookup]
            ax.plot([x[0] for x in observed],[x[1]['summary']['aggregate_tokens_per_s'] for x in observed],
                    marker='o',color=colors[draft],label=f'Draft {draft}')
        ax.set(title=title,xlabel='Configured active slots',ylabel='Committed output tokens / s',xticks=[1,2,4])
        ax.legend(frameon=False)
    names=['short-control','cold-low','cold-high','resume-high']
    for c,style in [(1,'--'),(4,'-')]:
        for draft in [0,7]:
            observed=[]
            for i,name in enumerate(names):
                group=lookup.get((f'd{draft}-c{c}','contention-'+name))
                if group and 'tool' in group['classes']:
                    observed.append((i,group['classes']['tool']))
            axes[1,0].plot([x[0] for x in observed],[x[1]['slo']['5']['tool_goodput_per_s'] for x in observed],
                marker='o',linestyle=style,color=colors[draft],label=f'D{draft} C{c}')
            quantiles=[(i,g['latency_s']['p95']) for i,g in observed if g['latency_s']['p95'] is not None]
            axes[1,1].plot([x[0] for x in quantiles],[x[1] for x in quantiles],marker='o',linestyle=style,
                color=colors[draft],label=f'D{draft} C{c}')
    axes[1,0].set(title='Short-tool valid goodput (5 s SLO)',ylabel='Valid short requests / s')
    axes[1,1].set(title='Successful short-tool tail latency',ylabel='Empirical P95 latency (s)')
    for ax in axes[1]:
        ax.set_xticks(range(4),names,rotation=15);ax.legend(frameon=False,fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Qwen3.8-27B / one RTX 4090 / NInfer / synthetic serving snapshots\n'
                 'One group per point; errors and drain included in throughput; counter roofline undetermined',fontsize=11)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    for extension in ['png','svg','pdf']:
        fig.savefig(output.with_suffix('.'+extension),dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('summary',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();main(a.summary,a.output)
