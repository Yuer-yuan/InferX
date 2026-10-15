"""Audit frozen inputs and actual capacity, summarize independent formal waves."""
import hashlib,json,re,statistics,sys
from pathlib import Path
from inferx.pd_baselines import load_frozen_workloads,arrival_requests
root=Path('/home/guokc/mypro/gnn-mount-sda1/InferX');summary=Path(sys.argv[1]);output=Path(sys.argv[2])
data=json.loads(summary.read_text());cases=['sglang-ctx-control','sglang-ctx-responsive128']+([sys.argv[3]] if len(sys.argv)>3 and sys.argv[3]!='none' else []);names=['short-control','cold-low','resume-high'];sources={str(summary):hashlib.sha256(summary.read_bytes()).hexdigest()};audits=[];table=[]
templates,grid=load_frozen_workloads(root/'concurrency-roofline')
def read(p):
 b=p.read_bytes();sources[str(p.relative_to(root))]=hashlib.sha256(b).hexdigest();return json.loads(b)
for case in cases:
 for name in names:
  waves=sorted([g for g in data['groups'] if g['case']==case and g['group']==name],key=lambda g:g['run']);assert len(waves)==3,(case,name,len(waves));assert len({g['configuration_sha256'] for g in waves})==1
  for g in waves:
   assert g['comparison_eligible'] and g['state']=='measured' and g['summary']['errors']==0 and g['summary']['truncated']==0
   assert g['summary']['tool_valid']==g['planned']==g['observed_responses'] and g['full_long_completed']==(0 if name=='short-control' else 1)
   path=root/'pd-baselines/raw'/g['run']/f"{case}-r{g['repetition']}-{name}"
   startup=read(path/'startup.json');state=startup['server_info']['internal_states'][0]
   assert startup['server_info']['context_length']==131328 and state['memory_usage']['token_capacity']==131328 and state['effective_max_running_requests_per_dp']==4
   assert state['mamba_ssm_dtype']=='bfloat16' and state['max_mamba_cache_size']==16 and state['mamba_radix_cache_strategy']=='extra_buffer' and state['disable_overlap_schedule'] and state['kv_cache_dtype']=='fp8_e4m3'
   graph_memory=state['memory_usage']['graph']['decode']
   if g['configuration'].get('cuda_graph') is False:assert graph_memory==0
   elif g['configuration'].get('cuda_graph') is True:assert graph_memory>0 and not state['disable_decode_cuda_graph']
   log=(path/'server.log').read_bytes();sources[str((path/'server.log').relative_to(root))]=hashlib.sha256(log).hexdigest();assert b'Mamba Cache is allocated. max_mamba_cache_size: 16' in log
   campaign=read(path.parent/'campaign.json');definition=next(x for x in campaign['workload_definitions'] if x['name']==name)
   specs=read(path/name/'specifications.json');assert specs==arrival_requests(definition,templates,grid,path.name)
   audits.append(dict(identity=g['identity'],actual_context=131328,actual_kv=131328,actual_concurrency=4,actual_mamba_slots=16,actual_decode_graph_gib=graph_memory,strict_requests=g['planned'],full_long=g['full_long_completed'],native_mamba_max_usage=max([float(x) for x in re.findall(rb'mamba usage: ([0-9.]+)',log)] or [0])))
  def values(key):return [g['summary'][key] for g in waves]
  table.append(dict(case=case,group=name,independent_repetitions=3,waves=[g['identity'] for g in waves],short_p95_s=[g['short']['latency_s']['p95'] for g in waves],long_latency_s=[g['long']['latency_s']['median'] for g in waves],output_tokens_per_s=values('aggregate_tokens_per_s'),fixed_48s_strict=[g['fixed_window_completion']['strict_tool'] for g in waves],short_5s_tool_goodput=[g['short']['slo']['5']['tool_goodput_per_s'] for g in waves],errors=sum(values('errors')),strict_requests=sum(values('tool_valid')),sampled_peak_mib=[g['gpu_memory']['sampled_peak_mib'] for g in waves],sampled_min_free_mib=[g['gpu_memory']['sampled_min_free_mib'] for g in waves],prefix=[g['prefix'] for g in waves]))
for row in table:
 baseline=next(x for x in table if x['case']==cases[0] and x['group']==row['group'])
 row['short_5s_pass_all_repeats']=all(v<=5 for v in row['short_p95_s'])
 row['output_throughput_change_median']=statistics.median(row['output_tokens_per_s'])/statistics.median(baseline['output_tokens_per_s'])-1
 row['long_cost_change_median']=(statistics.median(row['long_latency_s'])/statistics.median(baseline['long_latency_s'])-1) if row['group']!='short-control' else None
 row['study_criteria_pass']=row['short_5s_pass_all_repeats'] and (row['long_cost_change_median'] is None or row['long_cost_change_median']<=.25) and row['output_throughput_change_median']>=-.1
output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(dict(scope='Every selected case uses three independent fresh repetitions. Normalr1 complete groups survive an interrupted PD lane; default PD8192 and all qualification waves are separate diagnostics. Each wave independent restart; no request-level independence or production-P99 claim. Long/throughput cost uses ratio of 3-wave medians, not artificially paired waves.',rows=table,runtime_input_audit=audits,source_sha256=sources),indent=2)+'\n');print('audited',len(audits),'waves;',sum(x['strict_requests'] for x in table),'strict responses')
