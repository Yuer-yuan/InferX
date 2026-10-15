"""Plot completed mixed-wave medians; no paper-only or failed method points."""
import argparse,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--comparison',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
loaded=json.loads(a.comparison.read_text())
current=loaded['comparison'] if 'comparison' in loaded else [dict(row,**row['medians']) for row in loaded['rows']]
previous=json.loads(a.previous.read_text())['rows']
methods=[('sglang-ctx-responsive128','Simple (responsive128)','#2a9d8f','o'),
         ('sglang-pdmux-private-best','SGLang PD','#6a4c93','s'),
         ('compiled-primary-control','Compiler control','#0077bb','D'),
         ('morphx-copy96-no-launch-cap','MorphX port (ablation)','#d55e00','^')]
fig,axes=plt.subplots(1,2,figsize=(10.5,4.1),layout='constrained')
points=[]
for ax,group,title in zip(axes,['cold-high','resume-high'],['Cold 128K input','Resumed 128K input']):
    for case,label,color,marker in methods:
        if case.startswith('sglang'):
            row=next(r for r in previous if r['case']==case and r['group']==('cold-low' if group=='cold-high' else group))
            x,y=row['long_mixed_latency_median_s'],row['short_mixed_p95_median_s']
        else:
            row=next(r for r in current if r['case']==case and r['group']==group)
            assert row['independent_mixed_repeats']==3
            x,y=row['long_s'],row['short_p95_s']
        ax.scatter(x,y,color=color,marker=marker,s=85,label=label,edgecolor='white',linewidth=.7,zorder=3)
        points.append(dict(case=case,group=group,long_s=x,short_p95_s=y))
    ax.set(title=title,xlabel='Complete long-request latency (s)',ylabel='Short-request P95 (s)')
    ax.grid(alpha=.2);ax.margins(x=.12,y=.2)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False,fontsize=9)
fig.suptitle('Residual tradeoff after the admission fix — lower is better on both axes',fontsize=12)
a.output.mkdir(parents=True,exist_ok=True)
fig.savefig(a.output/'mixed-pareto.png',dpi=180)
fig.savefig(a.output/'mixed-pareto.svg',metadata={'Date':None})
fig.savefig(a.output/'mixed-pareto.pdf',metadata={'CreationDate':None,'ModDate':None})
metadata=dict(scope='Medians of three fresh mixed waves; no confidence intervals. Previous Simple/PD and current compiler/MorphX are distinct deployment cohorts; compiler-vs-MorphX is the same-compiler causal pair. Excludes original starvation control for readability and all unqualified methods.',
    points=points,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.comparison,a.previous]})
(a.output/'mixed-pareto-data.json').write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')
print('plotted',len(points),'audited median points')
