"""Analyze completed diagnostic trace, never use its latencies as formal data."""
import collections,gzip,hashlib,json,os,re
from pathlib import Path
from inferx.sota_trace import analyze_trace
from inferx.sota_results import audit_wave
r=Path(os.environ['INFERX_ROOT']);p=r/'sota-validation/raw/profile-copy-v2';out=r/'sota-validation/profile-evidence.json'
metadata=json.loads((p/'campaign.json').read_text());assert metadata['status']=='restored' and metadata['restored'] and not metadata.get('profile_error')
traces=list((p/'trace').rglob('*.trace.json.gz'));maps=list((p/'stream-ids').glob('*.json'));assert len(traces)==len(maps)==1
mapping=json.loads(maps[0].read_text());streams={k:v['cupti_stream_id'] for k,v in mapping['streams'].items()}
trace=json.load(gzip.open(traces[0],'rt'));events=trace['traceEvents'];result=analyze_trace(events,streams)
folder=p/'engine/profiled-cold';specs=json.loads((folder/'specifications.json').read_text());rows=[json.loads(x) for x in (folder/'requests.jsonl').read_text().splitlines()]
audited=audit_wave(specs,rows,metadata['result']['summary']['measured_span_s'],kind='mixed');assert audited['strict_requests']==25
copies=collections.Counter((e['args'].get('stream'),e['args'].get('bytes'),e['name']) for e in events if e.get('cat')=='gpu_memcpy')
costs={}
for e in events:
    if e.get('cat') not in {'cuda_runtime','cuda_driver'} or e.get('ph')!='X':continue
    k=(e['cat'],e['name']);entry=costs.setdefault(k,dict(category=k[0],name=k[1],count=0,sum_us=0));entry['count']+=1;entry['sum_us']+=e['dur']
log=(p/'engine/server.log').read_text();coefficients=[]
for name,ci,waves in re.findall(r'\[MorphX-profile\] name=(\S+) CI=(\S+) waves=(\d+)',log):coefficients.append(dict(name=name,ci=float(ci),waves=int(waves)))
control_writes=re.findall(r'\[MorphX-control\] stream=(\S+) name=(\S+) old=(\d+):(\d+) new=(\d+):(\d+)',log)
overlay=r/'sota-validation/overlays/morphx-pd-profile-v3'
files=[traces[0],maps[0],p/'campaign.json',p/'controller.py',p/'engine/startup.json',p/'engine/server.log',folder/'requests.jsonl',folder/'specifications.json',r/'code-pd-baselines/inferx/sota_trace.py',Path(__file__)]
files += [overlay/n for n in ['overlay-manifest.json','source-hashes.json','inferx_morphx_runtime.py','inferx_pd_runtime.py','sitecustomize.py']]
flow_warning=re.search(r'Warning: Profiler observed (\d+) duplicate flow start ID',log)
report=dict(scope='Separate profiled8-second mixed diagnostic, not normal latency/throughput or whole-wave coverage. Kernel gaps are not proof of GPU idle. CPU API duration sums overlap across threads/nesting and are not a critical-path decomposition.',
    trace=result,stream_mapping=mapping,strict_profile_requests=25,profile_latency_excluded=True,
    trace_window=dict(kernel_envelope_us=result['kernel_envelope_us'],base_time_nanoseconds=trace.get('baseTimeNanoseconds')),
    category_counts=dict(collections.Counter(str(e.get('cat')) for e in events)),
    profiler_warnings=dict(duplicate_flow_start_ids=int(flow_warning.group(1)) if flow_warning else 0,
        scope='Profiler reports runtime-correlation fallback for flow-ID collisions. No CPU/GPU causal flow reconstruction is used; lifetime intervals still carry instrumentation and capture uncertainty.'),
    copy_patterns=[dict(stream=k[0],bytes=k[1],name=k[2],count=n) for k,n in copies.most_common(30)],
    host_api_costs=sorted(costs.values(),key=lambda e:-e['sum_us'])[:25],
    observed_profile_coefficients=coefficients,
    sm_bounds=dict(changed_writes_logged=len(control_writes),agent_nonzero_min_writes=sum(int(x[4])>0 for x in control_writes),
                   scope='Actual model-run bound writes, including warmups; plain runtime messages do not locate all writes within the timed trace window or measure SM occupancy.'),
    source_sha256={str(f.relative_to(r)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(trace={k:result[k] for k in ['status','kernel_count','kernel_envelope_us','kernel_lifetime_union_us','pd_kernel_lifetime_overlap_us','transformed_kernel_time_fraction','memcpy_event_count','memcpy_sum_us']},roles=result['roles'],sm_bounds=report['sm_bounds'],top_copies=report['copy_patterns'][:6],top_host_api_costs=report['host_api_costs'][:5]),indent=2))
