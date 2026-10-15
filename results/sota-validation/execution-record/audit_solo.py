"""Rehash and inspect existing solo data without rerunning or overwriting it."""
import hashlib,json,os,time
from pathlib import Path
from inferx.pd_baselines import validate_sglang_runtime
from inferx.service_ownership import process_identity
from scripts import run_concurrency_campaign as previous
r=Path(os.environ['INFERX_ROOT']);run='pd-strategies-solo-baseline-20261015-1792072041';raw=r/'pd-baselines/raw'/run;reference=r/'pd-strategies'/run;out=r/'sota-validation';out.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();c=read(raw/'campaign.json');a=read(reference/'audit.json');scope=read(reference/'scope.json')
assert c['restored'] and c['status']=='restored' and len(c['cases'])==9
sources={**a['source_sha256'],**scope['source_sha256']}
for name,expected in sources.items():assert sha(Path(name))==expected,name
cases={x['id']:x for x in json.loads((raw/'config.json').read_text())};groups=[]
for state in c['cases']:
 assert state['state']=='measured' and state['quality']['status']=='passed'
 case,group=state['case'],state['group'];p=raw/f'{case}-r0-{group}';startup=read(p/'startup.json');config=cases[case];assert startup['case']==config
 validate_sglang_runtime(config,startup['server_info']);runtime=startup['server_info']['internal_states'][0]
 assert runtime['effective_max_running_requests_per_dp']==4 and runtime['memory_usage']['token_capacity']==131328 and runtime['mamba_ssm_dtype']=='bfloat16' and runtime['max_mamba_cache_size']==16 and runtime['kv_cache_dtype']=='fp8_e4m3' and runtime['disable_overlap_schedule'] and runtime['memory_usage']['graph']['decode']==0
 specs=read(p/group/'specifications.json');rows=[json.loads(line) for line in (p/group/'requests.jsonl').read_text().splitlines()];assert len(rows)==len(specs)==(24 if group=='short-control' else 1)
 assert all(b['dispatched_s']>=aa['finished_s'] for aa,b in zip(rows,rows[1:]));assert all('offset_s' not in s and s['producer']==0 for s in specs)
 assert all(x['summary']['ok'] and x['functional_ok'] and not x['summary'].get('truncated') for x in rows)
 origin='pd-strategies-pdmux-private-main-v1' if case=='sglang-pdmux-private-best' else 'pd-strategies-matched-formal-r1';frozen=read(r/'pd-baselines/raw'/origin/f'{case}-r0-{group}'/group/'specifications.json');frozen=frozen if group=='short-control' else [s for s in frozen if s['prompt_id']!='tool-24576'];assert len(frozen)==len(specs) and all(x['payload']==y['payload'] and x['expected_tool']==y['expected_tool'] for x,y in zip(specs,frozen))
 prompt=[x['summary']['prompt_tokens'] for x in rows];cached=[x['summary'].get('cached_tokens') for x in rows]
 if group!='short-control':assert min(prompt)>=130000
 if group=='resume-high':assert cached==[24576]
 result=read(p/group/'result.json');begin=result['started_unix_ms']/1000;end=result['ended_unix_ms']/1000;native=[]
 for line in (p/'server.log').read_text().splitlines():
  if 'PD_ADMISSION ' in line:
   e=json.loads(line.split('PD_ADMISSION ',1)[1])
   if begin<=e['unix']<=end and e['method']=='new' and e['admitted']:native.append({k:e[k] for k in ['rid','cached_tokens','remaining_input','chunk_before','result']})
 groups.append(dict(case=case,group=group,n=len(rows),serial_dispatch=True,full_correct=True,payload_exact=True,actual_c4=4,actual_kv=131328,actual_ssm_slots=16,prompt_tokens=prompt,api_cached_tokens=cached,native_admissions=native,independent_solo_repetitions=1))
assert sum(g['n'] for g in groups)==78
manifest=read(r/'tuning/best-service.json');assert manifest==read(r/'context-agents/runtime.json');identity=process_identity(manifest['pid'],previous.BINARY);assert all(manifest.get(k)==v for k,v in identity.items());previous.final_shutdown_check(manifest,gpu0_state='stopped')
report=dict(verified_unix=time.time(),scope='Existing78request isolated diagnostic rehashed and exact frozen payload/runtime/serial dispatch verified; one solo wave per case/group, not three.',source_files_rehashed=len(sources),source_sha256=sources,groups=groups,service=dict(identity=identity,manifests_equal=True,stopped_gpu0_owned_idle_gpu1_guard=True),limitations=['API cached tokens are not physical GPU work. Missing counters remain unknown.','Mixed comparisons contain three prior fresh waves; solo baseline one fresh wave.','Equal input and preparation do not prove identical cache residency during contention; inspect recorded admissions/counters.'])
(out/'isolated-reference-audit.json').write_text(json.dumps(report,indent=2)+'\n');print('verified',len(sources),'source hashes,78strict solo requests;GPU1idle/GPU0stopped',manifest['pid'])
