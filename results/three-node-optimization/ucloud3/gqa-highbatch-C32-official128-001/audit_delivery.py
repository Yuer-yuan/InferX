"""Independent local audit of original client/input/native files; no inference."""
import collections, datetime, hashlib, json, math, tarfile
from pathlib import Path

P=Path(__file__).resolve().parent
D=P/'delivery'
ARCHIVE_SHA='fe33c6d0625d540a51ebad0f3f0f941db3354dedae5da5821a88a73bccfe6376'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
rows=lambda p:[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def percentile(a,q):
    a=sorted(a);x=(len(a)-1)*q;i=int(x);w=x-i
    return a[i]*(1-w)+a[min(i+1,len(a)-1)]*w

assert sha(D/'complete-raw.tar.gz')==ARCHIVE_SHA
with tarfile.open(D/'complete-raw.tar.gz') as tf:
    for m in tf.getmembers():
        assert not m.issym() and not m.islnk() and not m.name.startswith('/') and '..' not in Path(m.name).parts
    tf.extractall(D/'content')
R=D/'content';files=read(R/'delivery-file-hashes.json')
for name,v in files.items():
    p=R/name
    assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],name
job=read(R/'owned/job.json');run=job['runs'][0]
assert job['status']=='complete' and run['status']=='complete'
assert run['driver_cleanup']['reaped'] and run['telemetry_cleanup']['reaped']
assert run['driver_cleanup']['exit_code']==run['telemetry_cleanup']['exit_code']==0
raw=R/'raw';profile=raw/'candidate/c32';manifest=read(raw/'manifest.json');tier=manifest['cases'][0]['tiers'][0]
assert manifest['status']=='complete' and tier['status']=='complete'
client=rows(profile/'profile_export.jsonl')
assert collections.Counter(x['metadata']['benchmark_phase'] for x in client)=={'warmup':8,'profiling':128}
assert tier['records']==tier['valid_records']==tier['successful_requests']==128 and tier['error_requests']==0
native=rows(raw/'candidate/requests.jsonl');done=[x for x in native if x['event']=='request_done'];starts=[x for x in native if x['event']=='server_start']
assert len(starts)==1 and len(done)==137 and not any(x['event']=='request_error' for x in native)
assert sorted(x['result']['completion_tokens'] for x in done)==[1]+[512]*136
assert all(x['result']['prompt_tokens']==x['result']['computed_prefill_tokens']==8192 and x['result']['prefix_cache_hit_tokens']==0 and x['result']['prefix_reuse_path']=='full_reset' and x['result']['finish_reason']=='output_limit' for x in done)
assert all(x['request']['protocol']=='openai_completions' and x['request']['message_count']==0 and not x['request']['enable_thinking'] and x['request']['sampling']['temperature']==0 for x in done)
inputs=read(profile/'inputs.json')['data'];assert len(inputs)==137
norm={}
for row in inputs:
    assert len(row['payloads'])==1
    payload=dict(row['payloads'][0]);payload.pop('cache_salt')
    assert len(payload['prompt'])==8192 and payload['max_tokens']==512 and payload['ignore_eos'] and payload['temperature']==0 and 'messages' not in payload
    assert row['session_id'] not in norm
    norm[row['session_id']]=payload
phasehash={}
for phase in ['warmup','profiling']:
    ids=sorted(x['metadata']['conversation_id'] for x in client if x['metadata']['benchmark_phase']==phase)
    assert len(ids)==len(set(ids))
    phasehash[phase]=digest({k:norm[k] for k in ids})
index=[]
for rank,row in enumerate(sorted([x for x in client if x['metadata']['benchmark_phase']=='profiling'],key=lambda x:x['metadata']['request_start_ns']),1):
    mm=row['metrics'];md=row['metadata']
    assert mm['input_sequence_length']['value']==8192 and mm['output_sequence_length']['value']==512
    tt=mm['time_to_first_token']['value'];lat=mm['request_latency']['value'];tp=(lat-tt)/511
    assert abs(tp-mm['inter_token_latency']['value'])<1e-8
    index.append(dict(arrival_rank=rank,conversation_id=md['conversation_id'],request_start_ns=md['request_start_ns'],TTFT_ms=tt,TPOT_ms=tp,E2E_ms=lat,prompt_SHA256=digest(norm[md['conversation_id']]['prompt'])))
tt=percentile([x['TTFT_ms'] for x in index],.95);tp=percentile([x['TPOT_ms'] for x in index],.95)
assert abs(tt-tier['metrics']['time_to_first_token']['p95'])<1e-6 and abs(tp-tier['metrics']['inter_token_latency']['p95'])<1e-6
contributors={}
for metric in ['TTFT_ms','TPOT_ms']:
    ordered=sorted(index,key=lambda x:x[metric]);ix=.95*127;lo=int(ix)
    contributors[metric]=[dict(weight=1-(ix-lo),**ordered[lo]),dict(weight=ix-lo,**ordered[lo+1])]
agg=read(profile/'phases/profiling/profile_export_aiperf.json')
assert agg['total_output_tokens']['avg']==65536
tps=tier['output_token_throughput'];assert tps==agg['output_token_throughput']['avg']
duration=(datetime.datetime.fromisoformat(agg['end_time'])-datetime.datetime.fromisoformat(agg['start_time'])).total_seconds()
assert abs(65536/tps-duration)<=1e-6
terminal=read(R/'owned/terminal-readback.json');assert terminal['all_actual_missing'] and not terminal['GPUapps'].strip()
assert run['actual_env']==read(P/'environment.json')
cfg=read(raw/'config.json');assert cfg['seed']==20261016 and cfg['requests']==128 and cfg['warmup']==8
assert starts[0]['engine']['max_concurrency']==32 and starts[0]['engine']['speculative_draft_window']==3 and starts[0]['engine']['kv_capacity']==278528
build=read(R/'build/state.json');seam=read(R/'native/state.json')
old=read(P.parent/'down053-C16-official128-001/summary.json')
cohort=dict(full137_SHA256=digest(norm),phase_SHA256=phasehash,same_actual_payloads_as_previous_u3_dc99_C16_except_salt=digest(norm)==old['payload_SHA256'],same_actual_phases_as_previous_u3_dc99_C16=phasehash==old['phase_payload_SHA256'])
assert cohort['same_actual_payloads_as_previous_u3_dc99_C16_except_salt'] and cohort['same_actual_phases_as_previous_u3_dc99_C16']
summary=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),binary_SHA256=job['runtime']['binary_sha256'],source_SHA256=job['runtime']['source_sha256'],C=32,native_cap=32,K=3,KV=278528,chunk=2048,measured=128,warmup=8,probe=1,seed=20261016,powerW=450,traceOFF=True,all137cold=True,zero_errors=True,all_startup_retained=True,TTFT_p95_ms=tt,TPOT_p95_ms=tp,output_tokens_per_second=tps,joint_PASS=tt<=10000 and tp<=40,TTFT_violation_ranks=[x['arrival_rank'] for x in index if x['TTFT_ms']>10000],TPOT_violation_ranks=[x['arrival_rank'] for x in index if x['TPOT_ms']>40],P95_same_request_contributors=contributors,cohort=cohort,throughput_scope=dict(original_phase_output_tps=tps,phase_export_seconds=duration,ISO_endpoint_resolution_seconds=1e-6,distinct_benchmark_duration=tier['benchmark_duration']),engine=starts[0]['engine'],memory=starts[0]['memory'],GPU=starts[0]['environment'],actual_environment=run['actual_env'],terminal=terminal,delivery=dict(archive_SHA256=ARCHIVE_SHA,bytes=(D/'complete-raw.tar.gz').stat().st_size,files_verified=len(files)),limitations=['Single fixed-seed combined-version C32 run; fails joint SLO','Historical u1 C32 and u3 dc99 C16 are not same-machine same-C GQA causal controls','No task-quality qualification or new profiler capture','Original operator fixed SCREEN_FAIL retained; integration separately authorized by Root'])
summary['initial32_later96']={name:dict(count=len(part),TTFT_violations=sum(x['TTFT_ms']>10000 for x in part),TPOT_violations=sum(x['TPOT_ms']>40 for x in part),max_TTFT_ms=max(x['TTFT_ms'] for x in part),max_TPOT_ms=max(x['TPOT_ms'] for x in part)) for name,part in [('initial32',index[:32]),('later96',index[32:])]}
(P/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(P/'measured-request-index.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in index))
(D/'local-delivery.json').write_text(json.dumps(dict(archive_SHA256=ARCHIVE_SHA,files_verified=len(files),raw=str(R),audit='PASS'),indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ['TTFT_p95_ms','TPOT_p95_ms','output_tokens_per_second','joint_PASS','cohort','initial32_later96','delivery']},indent=2))
