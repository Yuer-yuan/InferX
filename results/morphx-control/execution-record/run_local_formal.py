"""Owned frozen mixed diagnostics/solos; each group starts a fresh engine."""
import argparse,copy,hashlib,json,os,time
from pathlib import Path
from inferx.pd_baselines import load_frozen_workloads,arrival_requests,snapshot_harness,record_morphx_coverage
from inferx.service_ownership import process_identity
from scripts.run_pd_baselines import run_with_restoration,OwnedEngine,observe
from scripts import run_concurrency_campaign as previous
parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--cases',nargs='+');parser.add_argument('--groups',nargs='+',default=['cold-high']);parser.add_argument('--repeat',type=int,default=1);parser.add_argument('--solo',action='store_true');args=parser.parse_args()
assert Path(args.run_id).name==args.run_id and args.run_id not in {'.','..'} and args.repeat>0
r=Path(os.environ['INFERX_ROOT']);out=r/'morphx-control/raw'/args.run_id;out.mkdir(parents=True,exist_ok=False);previous.DIRECTORY=out
saved=json.loads((r/'tuning/best-service.json').read_text());assert saved==json.loads((r/'context-agents/runtime.json').read_text());identity=process_identity(saved['pid'],previous.BINARY);assert all(saved.get(k)==v for k,v in identity.items());saved['original_argv']=Path(f'/proc/{saved["pid"]}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
(out/'original-service.json').write_text(json.dumps(saved,indent=2)+'\n');(out/'controller.py').write_bytes(Path(__file__).read_bytes())
config=json.loads((r/'morphx-control/configs/morphx-local-cpu.json').read_text());cases=[c for c in config['cases'] if args.cases is None or c['id'] in args.cases];assert cases and (args.cases is None or set(args.cases)=={c['id'] for c in cases})
(out/'config.json').write_text(json.dumps(cases,indent=2)+'\n');templates,grid=load_frozen_workloads(r/'concurrency-roofline');definitions=[d for d in grid['mixed_cases'] if d['name'] in args.groups];assert {d['name'] for d in definitions}==set(args.groups)
for d in definitions:d['rate']=.5  # frozen formal PD-study arrival rate, not originalroofline2.5RPS
metadata=dict(status='running',controller_pid=os.getpid(),started_unix=time.time(),kind='isolated-serial-C4' if args.solo else 'mixed',repeat=args.repeat,cases=[],selection=config['selection'],workload_definitions=definitions,arrival_duration_s=grid['arrival_duration_s'],long_offsets_s=grid['long_offsets_s'],long_targets=grid['long_targets'],harness_sha256=snapshot_harness(r/'code-pd-baselines',out/'harness',[str(p.relative_to(r/'code-pd-baselines')) for folder in ['scripts','inferx'] for p in sorted((r/'code-pd-baselines'/folder).rglob('*.py'))]))
metadata['client_cpu_affinity']=sorted(os.sched_getaffinity(0))
assert metadata['client_cpu_affinity']==[40,41,42,43,96,97,98,99]
def flush():(out/'campaign.json').write_text(json.dumps(metadata,indent=2)+'\n')
def measure():
 jobs=[(c,repeat,d) for repeat in range(args.repeat) for d in definitions for c in (cases if repeat%2==0 else list(reversed(cases)))]
 metadata["execution_order"]=[dict(case=c["id"],repeat=repeat,group=d["name"]) for c,repeat,d in jobs];flush()
 for c,repeat,d in jobs:
  label=f'{c["id"]}-r{repeat}-{d["name"]}';entry=dict(case=c['id'],repeat=repeat,group=d['name'],state='starting');metadata['cases'].append(entry);flush()
  try:
   previous.gpu_free();case=copy.deepcopy(c);case['environment']['INFERX_MORPHX_IR']=str(out/(label+'-ir'));server=OwnedEngine(case,r,out/label);server.args+=['--watchdog-timeout','600']
   with server:
    previous.quality_gate(server,templates);entry['quality']=json.loads((server.directory/'quality-result.json').read_text());flush()
    if entry['quality']['status']!='passed':raise RuntimeError('code gate failed')
    if case['morphx']['enabled']:
     assert '[MorphX-affinity] requested=36 observed=36' in (server.directory/'server.log').read_text()
    short=templates['tool-24576'];seeds=[dict(id=f'seed-{i}',prompt_id='tool-24576',payload=short['payload'],expected_tool=short['expected_tool']) for i in range(4)]
    specs=arrival_requests(d,templates,grid,label)
    if args.solo:
     # Same C4 server and payload/cache warmup; serialize only the producer.
     if d['long_mode']:
      name=f"{d['long_mode']}-{grid['long_targets'][0]}";template=templates[name];specs=[dict(id='solo-long',prompt_id=name,payload=template['payload'],expected_tool=template['expected_tool'])]
     else:specs=[{k:v for k,v in s.items() if k!='offset_s'} for s in specs]
     # observe uses case concurrency as client workers; a per-request producer
     # dependency serializes without changing the server's C4 capacity.
     for s in specs:s['producer']=0
    result=observe(server,d['name'],specs,seeds)
    if result['summary']['errors'] or result['summary']['tool_valid']!=result['summary']['offered']:raise RuntimeError('strict complete request gate failed')
    if case['morphx']['enabled']:
     coverage_dir=server.directory/'final-coverage';coverage_dir.mkdir();entry['coverage']=record_morphx_coverage(server.directory/'server.log',coverage_dir)
     if entry['coverage']['status']!='observed':raise RuntimeError('runtime coverage absent')
    entry.update(state='measured',result=result)
   print(json.dumps(dict(case=c['id'],group=d['name'],repeat=repeat,state=entry['state'],short_p95=result['short']['latency_s']['p95'],long=result['long']['latency_s']['median'],span=result['summary']['measured_span_s'])),flush=True)
  except Exception as exc:
   entry.update(state='failed',error=f'{type(exc).__name__}: {exc}');print(json.dumps(entry),flush=True)
   # Preserve first failure; no silent reruns or success by request omission.
   flush();return
  finally:flush()
flush()
try:
 run_with_restoration(saved,previous.restore,[r/'context-agents/runtime.json',r/'tuning/best-service.json'],lambda:previous.final_shutdown_check(saved,gpu0_state='stopped'),measure);metadata['status']='restored'
except BaseException as exc:metadata.update(status='interrupted_or_failed',error=f'{type(exc).__name__}: {exc}');raise
finally:metadata.update(finished_unix=time.time(),restored=(out/'restored-service.json').exists());flush()
