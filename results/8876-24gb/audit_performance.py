"""Independently recompute one completed point from all original raw records."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('point',type=Path)
p.add_argument('--owner-status',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--expected-binary-sha',default='cabb42bb0ec13dbc0f2634a429c2872d4dd58d33c55337e1869d4bc6b7d425b4')
p.add_argument('--qualification',type=Path)
p.add_argument('--expected-kv-dtype',choices=['int8','rk4v4-e8'],default='rk4v4-e8')
p.add_argument('--expected-prefill-chunk',type=int,default=2048)
p.add_argument('--expect-env',action='append',default=[])
p.add_argument('--input-contract',type=Path)
a=p.parse_args();base=a.point.resolve();sources={}
def read(path,lines=False):
    blob=path.read_bytes();sources[str(path)]=dict(sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob))
    return [json.loads(x) for x in blob.splitlines() if x] if lines else json.loads(blob)
def q95(values):
    values=sorted(values);r=.95*(len(values)-1);lo=math.floor(r);hi=math.ceil(r)
    return values[lo]+(r-lo)*(values[hi]-values[lo])
def same(x,y):assert math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-8),(x,y)

summary=read(base/'summary.json');launch=read(base/'launch.json');recipe=read(base/'recipe.json')
owner=read(a.owner_status)
assert owner['status']=='complete' and owner['service_reaped']
assert launch['pid']==owner['service_pid'] and launch['start_ticks']==owner['service_start_ticks']
assert launch['binary_sha256']==recipe['binary_sha256']==a.expected_binary_sha
if a.qualification:
    qualification=read(a.qualification)
    assert qualification['status']=='complete' and not qualification['failed']
    assert qualification['binary_sha256']==recipe['binary_sha256'] and qualification['source_sha256']==recipe['source_sha256']
assert launch['process_environment']['CUDA_VISIBLE_DEVICES']=='GPU-a66b08e8-8b65-0faf-22e2-b951a97b2b03'
assert launch['environment']['NINFER_INPUT_S4_OMIT_LEGACY']=='1'
assert launch['environment']['NINFER_COLD_REQUEST_STATE']=='1'
assert launch['environment']['NINFER_CUDA_GRAPH_ALLOWANCE_MB']=='256'
for setting in a.expect_env:
    key,value=setting.split('=',1)
    assert launch['environment'][key]==launch['process_environment'][key]==value,(key,value)
rows=read(base/'c16/profile_export.jsonl',True)
assert Counter(r['metadata']['benchmark_phase'] for r in rows)=={'profiling':128,'warmup':8}
formal=[r for r in rows if r['metadata']['benchmark_phase']=='profiling']
assert len({r['metadata']['x_request_id'] for r in formal})==128
for r in rows:
    m=r['metrics'];meta=r['metadata']
    assert m['input_sequence_length']['value']==8192 and m['output_sequence_length']['value']==512
    assert m['usage_prompt_cache_read_tokens']['value']==0
    assert not meta['was_cancelled'] and not meta['context_overflow_skip']
    assert all(m[n]['unit']=='ms' for n in ['request_latency','time_to_first_token','inter_token_latency'])
    same((m['request_latency']['value']-m['time_to_first_token']['value'])/511,m['inter_token_latency']['value'])
ttft=q95([r['metrics']['time_to_first_token']['value'] for r in formal])
tpot=q95([(r['metrics']['request_latency']['value']-r['metrics']['time_to_first_token']['value'])/511 for r in formal])
same(ttft,summary['metrics']['time_to_first_token']['p95']);same(tpot,summary['metrics']['inter_token_latency']['p95'])
assert summary['records']==summary['valid_records']==summary['successful_requests']==128
assert summary['error_requests']==0 and not summary['issues'] and summary['lengths_verified']
native=read(base/'requests.jsonl',True)
starts=[r for r in native if r.get('event')=='server_start']
done=[r for r in native if r.get('event')=='request_done']
assert len(starts)==1 and len(done)==137
engine=starts[0]['engine']
for key,value in dict(max_concurrency=16,kv_capacity=139264,max_context=65536,
                      speculative_draft_window=3,speculative_backend='mtp',prefill_chunk=a.expected_prefill_chunk,prefix_reuse=False).items():
    assert engine[key]==value,(key,engine[key])
assert engine['kv_cache'].startswith(a.expected_kv_dtype),engine['kv_cache']
assert launch['argv'][launch['argv'].index('--kv-dtype')+1]==a.expected_kv_dtype
assert int(launch['argv'][launch['argv'].index('--prefill-chunk')+1])==a.expected_prefill_chunk
assert Counter(r['result']['completion_tokens'] for r in done)=={512:136,1:1}
assert all(r['result']['prompt_tokens']==r['result']['computed_prefill_tokens']==8192 and
           r['result']['prefix_cache_hit_tokens']==0 and r['result']['prefix_reuse_path']=='full_reset' for r in done)
inputs=read(base/'c16/inputs.json');assert len(inputs['data'])==137
canonical=hashlib.sha256(json.dumps(inputs,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
if a.input_contract:
    contract=read(a.input_contract)
    assert canonical==contract['expected_payload_sha256']
    assert contract['generator_sha256']=='d80df390cf6d1d50129796b31fdd7f4d1e95d0043d3b942d47ac582118702fb0'
    dataset_blob=(base/'dataset.jsonl').read_bytes()
    assert hashlib.sha256(dataset_blob).hexdigest()==contract['dataset_sha256']
    dataset=[json.loads(x) for x in dataset_blob.splitlines()];assert len(dataset)==137
    for actual,expected in zip(inputs['data'],dataset):
        assert actual['payloads'][0]['prompt']==expected['extra']['prompt'] and expected['output_length']==512
else:
    assert canonical=='6a22077fc53f81d60176e844adc01dd2dfeb7af19adebe9069f1d9e52969c7d1'
manifest=read(base/'c16/phase_manifest.json')
phase=next(x for x in manifest['phases'] if x['phase_kind']=='profiling')
assert phase['total_request_count']==phase['successful_request_count']==128 and phase['error_request_count']==0 and not phase['was_cancelled']
duration=(phase['end_ns']-phase['start_ns'])/1e9
tokens=sum(r['metrics']['output_token_count']['value'] for r in formal);assert tokens==65536
throughput=tokens/duration;same(throughput,summary['output_token_throughput'])
violations={name:[r['metadata']['x_request_id'] for r in formal if r['metrics'][name]['value']>limit]
    for name,limit in [('time_to_first_token',10000),('inter_token_latency',40)]}
result=dict(integrity='PASS',joint_slo=ttft<=10000 and tpot<=40,ttft_p95_ms=ttft,tpot_p95_ms=tpot,
    output_token_throughput=throughput,profiling_start_ns=phase['start_ns'],profiling_end_ns=phase['end_ns'],
    observation_seconds=duration,client_concurrency=16,native_concurrency=16,measured=128,warmup=8,probe=1,
    all137cold=True,errors=0,service_reaped=True,kv_cache=engine['kv_cache'],binary_sha256=recipe['binary_sha256'],
    source_sha256=recipe['source_sha256'],payload_canonical_sha256=canonical,prefill_chunk=a.expected_prefill_chunk,violations=violations,
    quality='unmeasured; no inherited score',sources=sources,auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['integrity','joint_slo','ttft_p95_ms','tpot_p95_ms','output_token_throughput']},indent=2))
