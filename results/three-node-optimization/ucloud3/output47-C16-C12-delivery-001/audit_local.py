import collections, datetime, hashlib, json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P/'raw'
read=lambda p:json.loads(p.read_text())
rows=lambda p:[json.loads(l)for l in p.read_text().splitlines()if l.strip()]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def quant(a,q):
 a=sorted(a);i=(len(a)-1)*q;lo=int(i);w=i-lo;return a[lo]*(1-w)+a[min(lo+1,len(a)-1)]*w
results={};cohorts={};index=[]
for c in [16,12]:
 d=R/f'C{c}';p=d/f'candidate/c{c}';mf=read(d/'manifest.json');tier=mf['cases'][0]['tiers'][0]
 native=rows(d/'candidate/requests.jsonl');done=[x for x in native if x['event']=='request_done'];starts=[x for x in native if x['event']=='server_start']
 assert len(starts)==1 and len(done)==137 and not any(x['event']=='request_error'for x in native)
 assert sorted(x['result']['completion_tokens']for x in done)==[1]+[512]*136
 assert all(x['result']['prompt_tokens']==x['result']['computed_prefill_tokens']==8192 and x['result']['prefix_cache_hit_tokens']==0 and x['result']['prefix_reuse_path']=='full_reset'and x['result']['finish_reason']=='output_limit'for x in done)
 assert all(x['request']['protocol']=='openai_completions'and x['request']['message_count']==0 and not x['request']['enable_thinking']and x['request']['sampling']['temperature']==0 for x in done)
 client=rows(p/'profile_export.jsonl');assert collections.Counter(x['metadata']['benchmark_phase']for x in client)=={'warmup':8,'profiling':128}
 inputs=read(p/'inputs.json')['data'];assert len(inputs)==137;norm={}
 for row in inputs:
  assert len(row['payloads'])==1;q=dict(row['payloads'][0]);q.pop('cache_salt')
  assert len(q['prompt'])==8192 and q['max_tokens']==512 and q['ignore_eos']and q['temperature']==0 and 'messages'not in q
  norm[row['session_id']]=q
 cohorts[c]=norm;phasehash={};phaseids={}
 for phase in ['warmup','profiling']:
  ids=sorted(x['metadata']['conversation_id']for x in client if x['metadata']['benchmark_phase']==phase);assert len(ids)==len(set(ids));phaseids[phase]=ids;phasehash[phase]=digest({k:norm[k]for k in ids})
 a=[]
 for rank,row in enumerate(sorted([x for x in client if x['metadata']['benchmark_phase']=='profiling'],key=lambda x:x['metadata']['request_start_ns']),1):
  mm=row['metrics'];md=row['metadata'];assert mm['input_sequence_length']['value']==8192 and mm['output_sequence_length']['value']==512
  tt=mm['time_to_first_token']['value'];tp=(mm['request_latency']['value']-tt)/511;assert abs(tp-mm['inter_token_latency']['value'])<1e-8
  a.append(dict(C=c,arrival_rank=rank,conversation_id=md['conversation_id'],request_start_ns=md['request_start_ns'],TTFT_ms=tt,TPOT_ms=tp,prompt_SHA256=digest(norm[md['conversation_id']]['prompt'])))
 index+=a;tt=quant([x['TTFT_ms']for x in a],.95);tp=quant([x['TPOT_ms']for x in a],.95)
 assert abs(tt-tier['metrics']['time_to_first_token']['p95'])<1e-6 and abs(tp-tier['metrics']['inter_token_latency']['p95'])<1e-6
 contributors={}
 for metric in ['TTFT_ms','TPOT_ms']:
  ordered=sorted(a,key=lambda x:x[metric]);ix=.95*(len(a)-1);lo=int(ix);contributors[metric]=[dict(weight=1-(ix-lo),**ordered[lo]),dict(weight=ix-lo,**ordered[lo+1])]
 agg=read(p/'phases/profiling/profile_export_aiperf.json');assert agg['total_output_tokens']['avg']==65536
 tps=tier['output_token_throughput'];assert tps==agg['output_token_throughput']['avg']
 # Exported ISO endpoints have microsecond precision; their duration differs
 # from the distinct benchmark_duration metric. Preserve the original metric.
 delta=(datetime.datetime.fromisoformat(agg['end_time'])-datetime.datetime.fromisoformat(agg['start_time'])).total_seconds()
 implied=65536/tps;assert abs(implied-delta)<=1e-6
 results[c]=dict(C=c,measured=128,warmup=8,probe=1,native_cold=137,zero_errors=True,TTFT_p95_ms=tt,TPOT_p95_ms=tp,output_tokens_per_second=tps,joint_PASS=tt<=10000 and tp<=40,TTFT_violation_ranks=[x['arrival_rank']for x in a if x['TTFT_ms']>10000],TPOT_violation_ranks=[x['arrival_rank']for x in a if x['TPOT_ms']>40],P95_same_request_contributors=contributors,actual_phase_payload_SHA256=phasehash,actual_phase_session_ids=phaseids,engine=starts[0]['engine'],memory=starts[0]['memory'],GPU=starts[0]['environment'],seed=read(d/'config.json')['seed'],throughput_audit=dict(original_phase_value=tps,ISO_phase_duration_seconds=delta,implied_high_precision_phase_duration=implied,ISO_endpoint_precision_seconds=1e-6,distinct_benchmark_duration=tier['benchmark_duration']),source_files={str(q.relative_to(R)):sha(q)for q in [d/'manifest.json',p/'inputs.json',p/'profile_export.jsonl',d/'candidate/requests.jsonl']})
assert cohorts[16]==cohorts[12]
assert results[16]['actual_phase_payload_SHA256']==results[12]['actual_phase_payload_SHA256']
j16=read(R/'owned-C16-C12/job.json');j12=read(R/'owned-C12-recovery/job.json')
assert j16['runs'][0]['joint_PASS']==results[16]['joint_PASS']and j12['runs'][0]['joint_PASS']==results[12]['joint_PASS']
assert j16['runs'][0]['actual_env']==j12['runs'][0]['actual_env']
audit=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),binary_SHA256='9270f73e92409bc94e4b8223adb755c8b1527e611eec21e1ef8778482d0b74c2',source_SHA256='91621a3c46a4c901a51863442704ea7c398ebdfb73ade9f3b1c4afac4ab235cc',same_actual_payloads_except_cache_salt=True,full137_actual_payload_SHA256=digest(cohorts[16]),same_warmup_and_measured_cohorts=True,same_quantization_environment=True,all_startup_retained=True,C16=results[16],C12=results[12],limitations=['Single seed20261016; no stable capacity claim','Concurrency and KV capacity differ; not a kernel causal A/B','TraceOFF; no measured GPU critical-path attribution','OldprotectedT4 parity failure preserved; selected47 actual newroute PASS only','Original100 quality user-cancelled,5responses/7attempts; no current full quality qualification'])
(P/'independent-local-raw-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
(P/'measured-request-index.jsonl').write_text(''.join(json.dumps(x)+'\n'for x in index))
print(json.dumps({c:{k:results[c][k]for k in ['joint_PASS','TTFT_p95_ms','TPOT_p95_ms','output_tokens_per_second']}for c in [16,12]}))
