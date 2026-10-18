"""Measured diagnostic and resource charts; data audited by build_cost_evidence.py."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

PREFILL = '#7E2635'
DECODE = '#718CA5'
GREEN = '#287657'
MUTED = '#657180'
MEMORY_COLORS = ['#C4CDD6','#90ABC0','#D3A3AE','#D8C5A1','#657180']


def build_admission_charts(export, cjk, campaign):
    """C14 configuration observations, explicitly separated by node and run."""
    comparison = next(r for r in campaign['comparison_runs']
                      if r['client_c'] == r['native_c'] == 14)
    selected = sorted((r for r in campaign['performance'] if r['client_c'] == 14),
                      key=lambda r: r['repeat'])
    assert len(selected) == 2 and {r['repeat'] for r in selected} == {1, 2}
    assert all(r['native_c'] == 13 and r['joint_slo'] for r in selected)
    assert not comparison['joint_slo']
    rows = [comparison] + selected
    assert len({r['payload_sha256'] for r in rows}) == 1
    assert len({r['kv'] for r in rows}) == 1
    for lang in ['zh', 'en']:
        def L(zh, en): return zh if lang == 'zh' else en
        fp = cjk if lang == 'zh' else None
        labels = [L(f"{r['native_c']}个活动槽", f"{r['native_c']} active slots")
                  + f"\n{r['node']} · r{r['repeat']}" for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(292/25.4, 84/25.4))
        fig.subplots_adjust(left=.065, right=.985, top=.85, bottom=.29, wspace=.22)
        panels = [('ttft_s', L('首字延迟 P95（s）', 'TTFT P95 (s)'), 12.5, [0, 5, 10], 10),
                  ('tpot_ms', L('逐Token生成延迟 P95（ms）', 'TPOT P95 (ms/token)'), 52, [0, 20, 40], 40)]
        for ax, (key, title, ymax, ticks, target) in zip(axes, panels):
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, color='#D5DBE1', linewidth=.55)
            ax.bar(range(3), [r[key] for r in rows], width=.45,
                   color=['#E3E9EF', '#EDDDE1', '#EDDDE1'],
                   edgecolor=[DECODE, PREFILL, PREFILL], linewidth=1)
            for i, r in enumerate(rows):
                ax.annotate(f'{r[key]:.3f}', (i, r[key]), xytext=(0, 9),
                            textcoords='offset points', ha='center', fontsize=13,
                            color=DECODE if i == 0 else PREFILL)
            ax.set(xlim=(-.65, 2.65), ylim=(0, ymax), yticks=ticks, xticks=range(3))
            ax.set_xticklabels(labels, fontproperties=fp, fontsize=12)
            ax.tick_params(axis='x', length=0, pad=6)
            ax.set_xlabel(L('准入配置', 'Admission configuration'), fontproperties=fp, labelpad=6)
            ax.set_title(title, fontproperties=fp, loc='left', pad=13)
            ax.axhline(target, color=GREEN, linewidth=1, linestyle=(0, (4, 3)))
            ax.text(.99, .98, f'SLO ≤ {target:g}', transform=ax.transAxes,
                    color=GREEN, ha='right', va='top', fontsize=11)
        export('admission-c14', lang, fig)


def build_cost_charts(export, series, cjk, root):
    d=json.loads((root/'submission/track-a/data/cost-evidence.json').read_text())
    for lang in ['zh','en']:
        fp=cjk if lang=='zh' else None
        def L(zh,en):return zh if lang=='zh' else en
        diag=d['diagnostic'];part=diag['parts_ms_per_token'];rep=diag['representative_request']
        fig,axs=plt.subplots(2,1,figsize=(174/25.4,94/25.4),gridspec_kw={'height_ratios':[1,1.2]})
        fig.subplots_adjust(left=.14,right=.97,top=.90,bottom=.15,hspace=.95)
        ax=axs[0];left=0
        for value,color,label in [(part['competing_prefill_ms'],PREFILL,L('预填充','Prefill')),(part['shared_decode_with_request_ms'],DECODE,L('解码','Decode'))]:
            ax.barh(0,value,left=left,height=.7,color=color)
            ax.text(left+value/2,0,f'{label}\n{value:.2f}',color='white',ha='center',va='center',fontproperties=fp,fontsize=12)
            left+=value
        ax.barh(0,part['host_residual_ms'],left=left,height=.7,color=MUTED)
        ax.set(xlim=(0,55),ylim=(-.6,.7),xticks=[0,20,40,50],yticks=[])
        ax.spines['left'].set_visible(False)
        ax.axvline(40,color=GREEN,linestyle='--',linewidth=1)
        ax.text(40,.57,'SLO 40',ha='right',color=GREEN,fontsize=10)
        ax.text(0,1.16,L('TPOT P95耗时分解：49.40 ms','TPOT P95 decomposition: 49.40 ms/token'),transform=ax.transAxes,fontproperties=fp,fontsize=12)
        ax.set_xlabel(L('单Token生成延迟（ms）','Time per output token (ms)'),fontproperties=fp,fontsize=11,labelpad=2)
        ax=axs[1]
        for kind,y,color in [('competing_prefill_ms',1,PREFILL),('shared_decode_with_request_ms',0,DECODE)]:
            spans=[(s['start_s'],s['duration_s']) for s in rep['segments'] if s['kind']==kind]
            ax.broken_barh(spans,(y-.3,.6),facecolors=color,linewidth=0)
        ax.set(xlim=(0,26),ylim=(-.6,1.6),xticks=[0,5,10,15,20,25],yticks=[0,1])
        ax.set_yticklabels([L('解码','Decode'),L('预填充','Prefill')],fontproperties=fp)
        ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
        ax.text(0,1.12,L('尾延迟样本：请求52的生成阶段','Actual tail request 52: execution after first token'),transform=ax.transAxes,fontproperties=fp,fontsize=12)
        ax.set_xlabel(L('首个Token输出后的时间（秒）','Time since first token (s)'),fontproperties=fp,fontsize=11,labelpad=3)
        export('cost-prefill-attribution',lang,fig)

        memory=d['memory'];a=memory['allocations_bytes'];samples=memory['samples'];gib=2**30
        keys=['weights_and_quantization_caches','kv','non_kv_sequence_state','workspace_and_graph']
        x=[s['time_s'] for s in samples]
        levels=[[a[k]/gib for _ in x] for k in keys]+[[s['unclassified_bytes']/gib for s in samples]]
        labels=[L('权重与量化缓存','Weights + quant. caches'),'KV',L('递推与序列状态','Recurrent / sequence state'),L('工作区与CUDA Graph','Workspace + Graph'),L('其他（未细分）','Unclassified residual')]
        fig,ax=plt.subplots(figsize=(174/25.4,94/25.4));fig.subplots_adjust(left=.12,right=.97,top=.88,bottom=.37)
        ax.stackplot(x,*levels,colors=MEMORY_COLORS,labels=labels,linewidth=0)
        ax.plot(x,[s['total_bytes']/gib for s in samples],color=MUTED,linewidth=1)
        ax.set(xlim=(0,memory['window_seconds']),ylim=(0,40),xticks=[0,40,80,120,160],yticks=[0,10,20,30,40])
        ax.text(0,1.1,L('GPU显存（GiB）','GPU memory (GiB)'),transform=ax.transAxes,fontproperties=fp)
        ax.text(memory['window_seconds']*.96,33.6,'31.97 GiB',ha='right',color=MUTED,fontsize=13)
        ax.set_xlabel(L('压测时间（秒）','Time since measured serving began (s)'),fontproperties=fp,fontsize=12,labelpad=5)
        handles,legend_labels=ax.get_legend_handles_labels()
        legend=fig.legend(handles,legend_labels,loc='lower center',bbox_to_anchor=(.5,.01),ncol=2,frameon=False,prop=cjk.copy() if fp else {'size':10},columnspacing=1.0,handlelength=1.1)
        # CJK legend size uses the same family fallback but a compact chart-label size.
        if fp:
            for t in legend.get_texts():t.set_fontsize(10)
        export('cost-serving-memory',lang,fig)

        labels=[L('原始策略\n085','Baseline\n085'),L('放宽调度限制\n086','Relax bound\n086'),L('计入待处理负载\n088','PF debt\n088')]
        xlabel=L('调度策略','Scheduling policy')
        series('cost-policy-ttft',lang,d['scheduling'],'ttft_s',labels,17,[0,5,10,15],xlabel,bars=True,target=10,digits=2)
        series('cost-policy-tpot',lang,d['scheduling'],'tpot_ms',labels,65,[0,20,40,60],xlabel,bars=True,target=40,digits=2)
