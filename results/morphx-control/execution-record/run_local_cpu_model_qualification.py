"""Require terminal kernel gates, then fresh owned full27B/128K qualification."""
import copy,hashlib,json,os,time
from pathlib import Path
from inferx.pd_baselines import load_frozen_workloads,snapshot_harness
from inferx.service_ownership import process_identity
from inferx.morphx_control import load_qualified_base_case
from scripts.run_pd_baselines import run_with_restoration,OwnedEngine,observe
from scripts import run_concurrency_campaign as previous
r=Path(os.environ['INFERX_ROOT']);kernel=r/'morphx-control/raw/kernels-v1/campaign.json';gates=json.loads(kernel.read_text())
assert gates['status']=='restored' and gates['restored'] and len(gates['cases'])==20 and all(x['status']=='passed' for x in gates['cases'])
base=load_qualified_base_case(r)
out=r/'morphx-control/raw/models-local-v3';out.mkdir(parents=True,exist_ok=False);previous.DIRECTORY=out
saved=json.loads((r/'tuning/best-service.json').read_text());assert saved==json.loads((r/'context-agents/runtime.json').read_text());identity=process_identity(saved['pid'],previous.BINARY);assert all(saved.get(k)==v for k,v in identity.items())
saved['original_argv']=Path(f'/proc/{saved["pid"]}/cmdline').read_bytes().decode().rstrip('\0').split('\0');(out/'original-service.json').write_text(json.dumps(saved,indent=2)+'\n');(out/'controller.py').write_bytes(Path(__file__).read_bytes())
metadata=dict(status='running',scope='Full model/complete input and dynamic runtime qualification only; no efficacy comparison.',controller_pid=os.getpid(),started_unix=time.time(),cases=[],kernel_campaign_sha256=hashlib.sha256(kernel.read_bytes()).hexdigest(),harness_sha256=snapshot_harness(r/'code-pd-baselines',out/'harness',[str(p.relative_to(r/'code-pd-baselines')) for folder in ['scripts','inferx'] for p in sorted((r/'code-pd-baselines'/folder).rglob('*.py'))]))
libraries={name:r/('morphx-control/builds/libcuda_'+name.replace('-','_')+'_local_v2.so') for name in ['abi-only','fill','event','both']}
def flush():(out/'campaign.json').write_text(json.dumps(metadata,indent=2)+'\n')
def measure():
 templates,_=load_frozen_workloads(r/'concurrency-roofline')
 for variant,library in libraries.items():
  actual_hash=hashlib.sha256(library.read_bytes()).hexdigest();build=json.loads((r/('morphx-control/runtime-'+variant+'-local-v2/manifest.json')).read_text());assert build['library_sha256']==actual_hash;assert {x['library_sha256'] for x in gates['cases'] if x['variant']==variant}=={build['base_library_sha256']}
  case=copy.deepcopy(base);case['id']='morphx-control-'+variant;case['morphx'].update(patched_launches=0,scope='Qualification bootstrap only, no inherited old-process coverage.',library_sha256=actual_hash)
  case['environment'].update(LD_PRELOAD=str(library)+':'+str(r/'pd-baselines/builds/libpreload.so'),NEUTRINO_HOOK_DRIVER=str(library),SMSCHED_CONTROL_TRACE='1',SMSCHED_AGENT_CPU='36',SMSCHED_LLM_TRACE_LIMIT='4096',INFERX_MORPHX_IR=str(out/(variant+'-ir')))
  entry=dict(variant=variant,case=case,library_sha256=actual_hash,status='starting');metadata['cases'].append(entry);flush()
  try:
   previous.gpu_free();server=OwnedEngine(case,r,out/variant,qualify_morphx=True);server.args+=['--watchdog-timeout','600']
   with server:
    previous.quality_gate(server,templates);entry['quality']=json.loads((server.directory/'quality-result.json').read_text());flush()
    assert entry['quality']['status']=='passed'
    short=templates['tool-24576'];long=templates['cold-131072'];seeds=[dict(id=f'seed-{i}',prompt_id='tool-24576',payload=short['payload'],expected_tool=short['expected_tool']) for i in range(4)]
    result=observe(server,'full128k',[dict(id='full-long',prompt_id='cold-131072',payload=long['payload'],expected_tool=long['expected_tool'])],seeds)
    assert result['summary']['errors']==0 and result['summary']['tool_valid']==1 and result['summary']['truncated']==0
    folder=server.directory/'full128k';rows=[json.loads(x) for x in (folder/'requests.jsonl').read_text().splitlines()]
    warm=[json.loads(x) for x in (folder/'warmup.jsonl').read_text().splitlines()]
    assert len(rows)==1 and rows[0]['summary']['prompt_tokens']==131060
    assert len(warm)==4 and all(x['summary']['prompt_tokens']==24568 and x['functional_ok'] and x['summary']['ok'] and not x['summary']['truncated'] for x in warm)
    assert '[MorphX-affinity] requested=36 observed=36' in (server.directory/'server.log').read_text()
    entry.update(status='passed',result=result)
  except Exception as exc:entry.update(status='failed',error=repr(exc))
  flush();print(json.dumps(dict(variant=variant,status=entry['status'],error=entry.get('error'),long_s=entry.get('result',{}).get('long',{}).get('latency_s',{}).get('median'))),flush=True)
flush()
try:
 run_with_restoration(saved,previous.restore,[r/'context-agents/runtime.json',r/'tuning/best-service.json'],lambda:previous.final_shutdown_check(saved,gpu0_state='stopped'),measure);metadata['status']='restored'
except BaseException as exc:metadata.update(status='interrupted_or_failed',error=repr(exc));raise
finally:metadata.update(finished_unix=time.time(),restored=(out/'restored-service.json').exists());flush()
