#!/usr/bin/env python3
"""Audit existing trace/VRAM/scheduling evidence. CPU only; never rewrites raw data."""
import csv
import hashlib
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / 'submission/track-a/data/cost-evidence.json'
BASE = ROOT / 'results/three-node-optimization/ucloud1'
TRACE = BASE / 'verify-w2-gate-085/trace-r1'
SOURCES = {}


def read(path):
    content = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = {'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
    return content.decode()


def obj(path): return json.loads(read(path))
def rows(path): return [json.loads(x) for x in read(path).splitlines() if x.strip()]
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def p95(values):
    a = sorted(values); index = .95 * (len(a)-1); lo = int(index)
    return a[lo] * (1-index+lo) + a[min(lo+1,len(a)-1)] * (index-lo)


def formal_run(name):
    base = BASE/name/'C16-r1'; raw = base/'delivery/raw/candidate'
    summary = obj(raw/'c16/summary.json')
    clients = rows(raw/'c16/profile_export.jsonl')
    measured = [r for r in clients if r['metadata']['benchmark_phase']=='profiling']
    assert len(measured)==128 and summary['successful_requests']==128 and summary['error_requests']==0
    ttft = []; tpot = []
    for r in measured:
        m = r['metrics']; assert m['output_sequence_length']['value']==512
        ttft.append(m['time_to_first_token']['value'])
        tpot.append((m['request_latency']['value']-m['time_to_first_token']['value'])/511)
    assert abs(p95(ttft)-summary['metrics']['time_to_first_token']['p95'])<1e-7
    assert abs(p95(tpot)-summary['metrics']['inter_token_latency']['p95'])<1e-7
    native=rows(raw/'requests.jsonl'); startup=next(r for r in native if r.get('event')=='server_start')
    done=[r for r in native if r.get('event')=='request_done']; assert len(done)==137
    assert all(r['result']['computed_prefill_tokens']==8192 and r['result']['prefix_cache_hit_tokens']==0 for r in done)
    assert startup['engine']['max_concurrency']==16 and startup['engine']['kv_capacity']==139264 and startup['engine']['speculative_draft_window']==5
    inputs=obj(raw/'c16/inputs.json')['data']; assert len(inputs)==137
    identity=obj(base/'delivery/raw/candidate-identity.json')
    reported=obj(base/'summary.json')
    return {'id':name[-3:],'ttft_s':p95(ttft)/1000,'tpot_ms':p95(tpot),'tps':summary['output_token_throughput'],
            'n':128,'warmup':8,'probe':1,'cold_requests':137,'errors':0,'identity':identity,
            'gpu_uuid':startup['environment']['gpu_uuid'],'engine':startup['engine'],
            'payload_sha256':digest(inputs),'reported_checks':{k:v for k,v in reported.items() if k.startswith('payload_same')}},inputs


def main():
    trace = rows(TRACE/'delivery/control/scheduler.jsonl')
    clients = [r for r in rows(TRACE/'delivery/raw/candidate/c16/profile_export.jsonl') if r['metadata']['benchmark_phase']=='profiling']
    clients.sort(key=lambda r:r['metadata']['request_start_ns']); assert len(clients)==128
    base_ns=clients[0]['metadata']['request_start_ns']
    anchors=[min(obj(TRACE/f'delivery/control/clock-anchor-{name}.json')['samples'], key=lambda s:s['epoch_after_ns']-s['epoch_before_ns']) for name in ['start','end']]
    offsets=[(s['epoch_before_ns']+s['epoch_after_ns'])/2-s['monotonic_ns'] for s in anchors]
    assert abs(offsets[1]-offsets[0])<1e6
    def epoch(t):
        fraction=(t*1e6-anchors[0]['monotonic_ns'])/(anchors[1]['monotonic_ns']-anchors[0]['monotonic_ns'])
        return (t*1e6+offsets[0]+fraction*(offsets[1]-offsets[0])-base_ns)/1e6
    first={};finish={};members=defaultdict(list); units=[]
    for r in trace:
        if r['event']=='first_token': first[r['request_id']]=epoch(r['start_ms'])
        if r['event']=='request_finish': finish[r['request_id']]=(epoch(r['start_ms']),r['tokens'])
        if r['event']=='decode_row': members[r['start_ms']].append(r['request_id'])
    for r in trace:
        if r['event'] not in ['prefill_chunk','decode_round']:continue
        u={'phase':'prefill' if r['event']=='prefill_chunk' else 'decode','start_ms':epoch(r['start_ms']),'end_ms':epoch(r['end_ms']),
           'request_id':r['request_id'],'members':members[r['start_ms']] if r['event']=='decode_round' else []}
        if u['phase']=='decode':assert len(u['members'])==r['rows']
        units.append(u)
    units.sort(key=lambda u:u['start_ms'])
    assert all(a['end_ms']<=b['start_ms']+1e-5 for a,b in zip(units,units[1:]))
    joined=[]; seen=set()
    for c in clients:
        m=c['metrics'];meta=c['metadata'];assert m['output_sequence_length']['value']==512
        start=(meta['request_start_ns']-base_ns)/1e6; ft=start+m['time_to_first_token']['value']; end=(meta['request_end_ns']-base_ns)/1e6
        candidates=[rid for rid,t in first.items() if rid in finish and finish[rid][1]==512 and abs(t-ft)<=5 and abs(finish[rid][0]-end)<=5]
        assert len(candidates)==1;rid=candidates[0];assert rid not in seen;seen.add(rid)
        parts={'competing_prefill_ms':0.,'shared_decode_with_request_ms':0.,'decode_without_request_ms':0.};own_pf=0.;segments=[]
        for u in units:
            if u['phase']=='prefill' and u['request_id']==rid:own_pf+=max(0,min(ft,u['end_ms'])-max(start,u['start_ms']))
            duration=max(0,min(end,u['end_ms'])-max(ft,u['start_ms']))
            if duration==0:continue
            if u['phase']=='prefill': assert u['request_id']!=rid; key='competing_prefill_ms'
            else:key='shared_decode_with_request_ms' if rid in u['members'] else 'decode_without_request_ms'
            parts[key]+=duration
            segments.append({'kind':key,'start_s':(max(ft,u['start_ms'])-ft)/1000,'duration_s':duration/1000})
        elapsed=m['request_latency']['value']-m['time_to_first_token']['value']
        parts['host_residual_ms']=elapsed-sum(parts.values());assert parts['host_residual_ms']>=-1e-5
        joined.append({'request':meta['conversation_id'],'engine_id':rid,'tpot_ms':elapsed/511,'own_prefill_ms':own_pf,
                       'first_ms':ft,'end_ms':end,'parts_ms':parts,'segments':segments})
    ordered=sorted(joined,key=lambda r:r['tpot_ms']);ix=.95*127;lo=int(ix);weights=[1-(ix-lo),ix-lo];tail=ordered[lo:lo+2]
    parts={k:sum(w*r['parts_ms'][k]/511 for w,r in zip(weights,tail)) for k in tail[0]['parts_ms']}
    total=sum(parts.values());previous=obj(TRACE/'attribution.json')
    assert abs(total-previous['metrics']['TPOT_ms'])<1e-7
    for k in ['competing_prefill_ms','shared_decode_with_request_ms','decode_without_request_ms']:
        assert abs(parts[k]-previous['original128_weighted_stage_ms']['TPOT_ms'][k])<1e-5
    representative=tail[1]
    diagnostic={'build':'085','identity':obj(TRACE/'delivery/raw/candidate-identity.json'),'n':128,'joined_requests':len(seen),
        'prefill_units':sum(u['phase']=='prefill' for u in units),'decode_units':sum(u['phase']=='decode' for u in units),'overlaps':0,
        'tpot_p95_ms':total,'parts_ms_per_token':parts,'prefill_fraction':parts['competing_prefill_ms']/total,
        'own_prefill_median_ms':statistics.median(r['own_prefill_ms'] for r in joined),
        'p95_contributors':[{'request':r['request'],'weight':w} for w,r in zip(weights,tail)],
        'representative_request':representative,
        'scope':'085 C16/K5 traceON, cold8192/512. Synchronized executor host spans, not exclusive GPU kernel time or formal traceOFF score.'}

    native=rows(TRACE/'delivery/raw/candidate/requests.jsonl'); startup=next(r for r in native if r.get('event')=='server_start');m=startup['memory']
    log=read(TRACE/'delivery/raw/candidate/server.log')
    def logged(pattern):return int(re.search(pattern,log).group(1))
    cache={'ffn_s4':logged(r'allFFN_total_bytes=(\d+)'),'input_s4':logged(r'INPUT_S4_cache_load layers=64 appended_bytes=(\d+)'),
           'output_s4':logged(r'OUTPUT_S4_cache_load layers=47 appended_bytes=(\d+)'),'verify_w2':logged(r'VERIFY_W2_READY layers=64 bytes=(\d+)')}
    allocations={'weights_and_quantization_caches':m['weights']['capacity_bytes']+sum(cache.values()),'kv':m['kv_payload_bytes'],
                 'non_kv_sequence_state':m['sequence']['capacity_bytes']-m['kv_payload_bytes'],
                 'workspace_and_graph':m['workspace']['capacity_bytes']+m['cuda_graph_observed_bytes']}
    # GPU CSV uses the same UTC wall-clock display as the executor's timestamped startup log.
    log_start=re.search(r'\[([^]]+)\] \[info\] ninfer-serve: model loaded',log).group(1)
    log_epoch=datetime.strptime(log_start,'%Y-%m-%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp()
    assert abs(log_epoch-startup['timestamp_unix_ms']/1000)<.01
    start_s=base_ns/1e9;end_s=max(c['metadata']['request_end_ns'] for c in clients)/1e9;samples=[]
    for r in csv.DictReader(read(TRACE/'delivery/control/C16-gpu.csv').splitlines()):
        t=datetime.strptime(r['timestamp'],'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp()
        if not start_s<=t<=end_s:continue
        assert r[' uuid'].strip()==startup['environment']['gpu_uuid']
        used=int(r[' memory.used [MiB]'].split()[0])*2**20
        residual=used-sum(allocations.values());assert residual>=0
        samples.append({'time_s':t-start_s,'total_bytes':used,'unclassified_bytes':residual})
    assert samples and max(b['time_s']-a['time_s'] for a,b in zip(samples,samples[1:]))<1.1
    memory={'build':'085 traceON','sample_count':len(samples),'window_seconds':end_s-start_s,'sampling':'approximately 1 Hz',
        'window_start_utc':datetime.fromtimestamp(start_s,timezone.utc).isoformat(),'window_end_utc':datetime.fromtimestamp(end_s,timezone.utc).isoformat(),
        'min_total_bytes':min(s['total_bytes'] for s in samples),'max_total_bytes':max(s['total_bytes'] for s in samples),
        'allocations_bytes':allocations,'base_weights_bytes':m['weights']['capacity_bytes'],'cache_bytes':cache,
        'workspace_bytes':m['workspace']['capacity_bytes'],'graph_observed_bytes':m['cuda_graph_observed_bytes'],
        'graph_budget_bytes':m['cuda_graph_allowance_bytes'],'cuda_available_after_startup_bytes':m['available_after_startup_bytes'],
        'samples':samples,'scope':'Fixed startup allocations carried over the serving window; sampled device total minus accounted allocations is explicitly unclassified. Non-KV state includes GDN/replay/sequence buffers. Not a dynamic per-allocation trace.'}
    policies=[];reference=None
    for name in ['verify-w2-gate-085','reachable-prefill-queue-086','known-prefill-debt-088']:
        policy,inputs=formal_run(name)
        if reference is None:reference=inputs
        else:assert inputs==reference,'Payload mismatch'
        policies.append(policy)
    assert len({p['gpu_uuid'] for p in policies})==1
    output={'diagnostic':diagnostic,'memory':memory,'scheduling':policies,
        'scheduling_scope':'Same GPU, original137 payloads identical including salt, C16/K5/KV139264, one sequential traceOFF run per policy. 086 and088 are separate descendants of085, not cumulative policies.',
        'sources':SOURCES}
    DATA.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(f'Verified {len(seen)}/128 joins, {len(units)} nonoverlapping executor spans; Prefill {100*diagnostic["prefill_fraction"]:.2f}% of tail decomposition.')
    print(f'Serving memory: {len(samples)} samples over {end_s-start_s:.2f}s; {memory["max_total_bytes"]/2**30:.4f} GiB; {len(SOURCES)} source files hashed.')

if __name__=='__main__':main()
