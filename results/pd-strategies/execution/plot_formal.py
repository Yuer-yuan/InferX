"""Plot independent wave statistics, not request-level confidence intervals."""
import hashlib,json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
source=Path(sys.argv[1]);output=Path(sys.argv[2]);d=json.loads(source.read_text());cases=list(dict.fromkeys(r['case'] for r in d['rows']));groups=['short-control','cold-low','resume-high'];labels={'sglang-ctx-control':'Original','sglang-ctx-responsive128':'Simple128'};colors=['#626c7a','#24789e','#bd6b2c'];fig,axes=plt.subplots(2,3,figsize=(11,6.5))
for col,group in enumerate(groups):
 for index,case in enumerate(cases):
  row=next(r for r in d['rows'] if r['case']==case and r['group']==group)
  for ax,key in [(axes[0,col],'short_p95_s'),(axes[1,col],'long_latency_s')]:
   vals=[x for x in row[key] if x is not None]
   if vals:
    xx=[index+(n-1)*.07 for n in range(len(vals))];ax.scatter(xx,vals,color=colors[index],s=32,zorder=3);ax.plot([index-.15,index+.15],[sorted(vals)[len(vals)//2]]*2,color=colors[index],lw=2)
  axes[0,col].axhline(5,color='#8a2525',ls='--',lw=.8) if index==0 else None
 for ax in axes[:,col]:
  ax.set_xticks(range(len(cases)),[labels.get(x,'PD adapted') for x in cases],fontsize=9);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 axes[0,col].set_title(['Short only','+ cold128K','+ resume128K'][col]);axes[0,col].set_yscale('log');axes[0,col].set_ylim(2,140)
 if group=='short-control':axes[1,col].text(.5,.5,'No long request',ha='center',va='center',transform=axes[1,col].transAxes)
axes[0,0].set_ylabel('Short request P95 (s, log)');axes[1,0].set_ylabel('Complete long latency (s)');fig.suptitle('27B / single RTX4090 / matched resources / 3 independent waves',fontsize=12);fig.text(.5,.025,'Dots: independent restarted waves. Bars: median, not confidence intervals. Dashed: study5s target. All long inputs complete.',ha='center',fontsize=8);fig.tight_layout(rect=(0,.05,1,.95));output.parent.mkdir(parents=True,exist_ok=True)
for ext in ['.png','.pdf']:fig.savefig(output.with_suffix(ext),dpi=170)
output.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scope=__doc__),indent=2)+'\n');plt.close(fig)
