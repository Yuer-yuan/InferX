"""Generate the report figure from the strict final export, keeping every wave."""
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(os.environ['INFERX_ROOT'])/'pd-baselines'
source=root/'derived-final-v2-a/summary.json'
data=json.loads(source.read_text())
choices=[
 ('NInfer\nchunk2048','ninfer-d0-c4-chunk2048',{'ninfer-lowrate-main-v2'}),
 ('llama.cpp\nchunk128','llama-d0-c4-chunk128',{'llama-ch128-lowrate-main-v2'}),
 ('vLLM 0.24\nlegacy','vllm024-int4-c4-b4096-cap2048-eager-mem95',
  {'vllm-mem95-lowrate-diagnostic-v1','vllm-mem95-lowrate-repeat-v3'}),
 ('vLLM 0.31','vllm031-int4-c4-b4096-cap2048-eager-mem95',
  {'vllm031-codecfix-lowrate-diagnostic-v1','vllm031-codecfix-lowrate-repeat-v1'}),
 ('SGLang 0.5.19\nsingleton guard / BF16 SSM',
  'sglang0519-int4-c4-ch512-eager-mem97-hftext-bf16-lazy12-skiplock-singleton-guard',
  {'sglang-singleton-lowrate-main-v5'}),
]
conditions=[('short-control','Short only','#307aa3',-.22),
            ('cold-low','With cold 128K','#d28c26',0),
            ('resume-high','With resume-template 128K','#ba4e4e',.22)]
fig,ax=plt.subplots(figsize=(11,5.8))
observed=[]
for index,(label,case,runs) in enumerate(choices):
    item=dict(label=label,case=case,groups={})
    for group,legend,color,offset in conditions:
        waves=[g for g in data['groups'] if g['case']==case and g['run'] in runs
               and g['group']==group and g['rate']==.5 and g['comparison_eligible']]
        assert len(waves)==3,(case,group,len(waves))
        values=[w['short']['latency_s']['p95'] for w in waves]
        assert all(v is not None and v>0 for v in values)
        assert all(w['summary']['tool_valid']==w['summary']['offered'] for w in waves)
        ax.scatter([index+offset+v for v in [-.035,0,.035]],values,
                   s=33,color=color,edgecolors='white',linewidths=.5,
                   label=legend if index==0 else None,zorder=3)
        ax.plot([index+offset-.065,index+offset+.065],
                [sorted(values)[1]]*2,color=color,linewidth=1.4,zorder=2)
        item['groups'][group]=dict(p95_s=values,wave_ids=[w['identity'] for w in waves])
    observed.append(item)
ax.set_yscale('log')
ax.set_ylim(.8,210)
ax.set_xlim(-.55,4.6)
ax.axhline(5,color='#555555',linestyle='--',linewidth=.9)
ax.text(4.53,5.35,'5 s target',ha='right',va='bottom',fontsize=9,color='#555555')
ax.set_yticks([1,2,5,10,20,50,100,200],[str(v) for v in [1,2,5,10,20,50,100,200]])
ax.set_xticks(range(len(choices)),[c[0] for c in choices],fontsize=9)
ax.set_ylabel('Short-request P95 (seconds, logarithmic scale)')
ax.set_title('Observed tail latency: all three independent repetitions',fontsize=13)
ax.grid(axis='y',alpha=.2)
ax.set_axisbelow(True)
ax.legend(loc='upper left',frameon=False,fontsize=9)
fig.text(.5,.045,'27B / one RTX4090 / C4 / 0.5 short RPS / 24K short history / 128K long.\n'
         'Different quantization and cache capacities; these are deployment observations, not isolated scheduler rankings.',
         ha='center',fontsize=8)
fig.tight_layout(rect=(0,.105,1,1))
out=root/'final-comparison-v2'
out.mkdir(exist_ok=False)
for suffix in ['png','pdf']:
    fig.savefig(out/f'comparison.{suffix}',dpi=180)
manifest=dict(summary_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              matplotlib=matplotlib.__version__,observed=observed)
(out/'comparison.json').write_text(json.dumps(manifest,indent=2)+'\n')
plt.close(fig)
print('generated',str(out),flush=True)
