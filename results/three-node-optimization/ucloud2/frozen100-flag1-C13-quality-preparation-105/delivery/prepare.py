from pathlib import Path
import json,hashlib,shutil,difflib,py_compile
R=Path(__file__).resolve().parent;U=R.parent;P=U/'outputVerify-K3-C16-quality-076'
REMOTE='/workspace/InferX/three-node-optimization/ucloud2/'+R.name
BINARY='/workspace/InferX/three-node-optimization/ucloud2/bounded-decode-membership-host-100/candidate/ninfer-serve'
SHA='e4ed1bf4baebf7da3a81e4aed01307c7e3e7ae97faef7672c507ed21bdaeb38c';SOURCE='c976692d9657408308193b50772c700382a6e243d9da353165205dc5252f76a8'
def write(n,v):(R/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for n in ['recording_entry.py','checkpoints.py','client-environment.json','first100-frozen-cohort.jsonl','audit_http_native_join.py']:
 shutil.copyfile(P/n,R/n)
s=(P/'preflight.py').read_text().replace("raw['eval_batch_size']==16","raw['eval_batch_size']==13").replace("resolved['generation_config']['batch_size']==16","resolved['generation_config']['batch_size']==13");(R/'preflight.py').write_text(s)
s=(P/'score_first100_cpu.py').read_text().replace('same6632/3a7','frozen100 e4ed/c976 flag1').replace('batch16/max32768','batch13/max32768').replace('Same frozen6632/3a7 binary','Same frozen100 e4ed/c976 flag1 binary').replace('corresponding C16 performance candidate (qualityKV532480,performanceKV139264)','corresponding native13 performance candidate (qualityKV532480,performanceKV113152; client14/native13 performance uses121856)')
(R/'score_first100_cpu.py').write_text(s)
cfg=json.loads((P/'server-config.json').read_text());cfg['name']='frozen100_e4ed_flag1_native13_K3_original100_defaultsampling_qualityKV532480';cmd=cfg['command'];cmd[0]=BINARY;cmd[cmd.index('--port')+1]='18505';cmd[cmd.index('--max-concurrency')+1]='13';cmd[cmd.index('--request-log-jsonl')+1]=REMOTE+'/service/native.jsonl'
cfg['environment']=json.loads((U/'frozen100-flag1-lowC-preparation-102/environment.json').read_text());cfg['environment'].update(LD_LIBRARY_PATH='/workspace/InferX/envs/build/lib:/usr/local/cuda-13.2/lib64',NINFER_SCHED_TRACE=REMOTE+'/scheduler.jsonl');write('server-config.json',cfg)
task=json.loads((P/'task_config.json').read_text());task.update(api_url='http://127.0.0.1:18505/v1',eval_batch_size=13,work_dir=REMOTE+'/accuracy/native');write('task_config.json',task)
write('expected-runtime.json',dict(binary=BINARY,binary_bytes=243219440,binary_sha256=SHA,source_aggregate_sha256=SOURCE))
shutil.copyfile(U/'bounded-decode-membership-host-100/delivery/handoff-manifest.json',R/'qualification-evidence.json')
s=(P/'owner.py').read_text();original=s
s=s.replace('Single6632/source3a7 original100 C16 quality after actual native qualification.','Single frozen100 e4ed/c976 flag1 original100 C13 quality; explicit Root execution release required.')
s=s.replace('PORT=18476','PORT=18505').replace("scope='6632/3a7 outputVerify originalfirst100 C16 quality'","scope='frozen100 e4ed/c976 flag1 originalfirst100 C13 quality'")
s=s.replace("release['native_qualification_pass'] is True", "release['execute_quality_after_lowC_complete'] is True")
s=s.replace("qualification=json.loads((R/'native-qualification.json').read_text());assert qualification['status']=='PASS_native_health_only' and qualification['binary']==expected['binary_sha256'] and qualification['source']==expected['source_aggregate_sha256'];assert sha(R/'native-qualification.json')==release['native_receipt_sha256']", "qualification=json.loads((R/'qualification-evidence.json').read_text());assert qualification['binary']['sha256']==expected['binary_sha256'] and qualification['source_aggregate']==expected['source_aggregate_sha256'];assert sha(R/'qualification-evidence.json')==release['qualification_evidence_sha256']\n # Prior source/native-boundary evidence does not claim new task accuracy or actual KV532480 fit.\n lock=(C/'.ucloud2-performance-owner.lock').open('a');__import__('fcntl').flock(lock,__import__('fcntl').LOCK_EX|__import__('fcntl').LOCK_NB)")
s=s.replace("('--port','18476'),('--max-concurrency','16')","('--port','18505'),('--max-concurrency','13')").replace("engine['max_concurrency']==16","engine['max_concurrency']==13")
s=s.replace("assert not any(any(w in k", "assert actual['NINFER_SCHED_BOUNDED_DECODE_MEMBERSHIP']=='1' and actual['NINFER_PROMPT_LOOKUP_N5_PRIORITY']==actual['NINFER_PROMPT_LOOKUP_N5_HYBRID']=='0'\n assert not any(any(w in k")
s=s.replace("same_generic_input_route_as_performance=True,quality_only_switch=False", "same_generic_input_route_as_performance=True,quality_only_switch=False,bounded_decode_membership_enabled=True")
s=s.replace("plus u1 actualGraph qualification", "plus frozen100 source and prior native boundary evidence")
# Construct parent job dir only after explicit release, no job/waiter is started by preparation.
s=s.replace("save('explicit_native_qualification_release_verified')", "J.parent.mkdir(parents=True,exist_ok=True);save('explicit_Root_quality_release_verified')")
(R/'owner.py').write_text(s);(R/'owner-from076.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='076/owner.py',tofile='105/owner.py')))
cohort=[json.loads(l)for l in (R/'first100-frozen-cohort.jsonl').read_text().splitlines()];assert len(cohort)==100;assert sum(len(v['corpus']['instruction_id_list'])for v in cohort)==111
assert task['generation_config']=={'max_tokens':32768}
assert all(x not in cmd for x in ['--greedy','--no-thinking','--temperature','--top-p','--top-k'])
reuse={n:{'parent_SHA256':sha(P/n),'candidate_SHA256':sha(R/n),'identical':sha(P/n)==sha(R/n)}for n in ['recording_entry.py','checkpoints.py','client-environment.json','first100-frozen-cohort.jsonl','audit_http_native_join.py']}
assert all(x['identical']for x in reuse.values());write('source-reuse.json',reuse)
write('PLAN.json',dict(scope='CPU preparation only; no service/collector/scorer/waiter launched',binary_sha256=SHA,source_aggregate=SOURCE,native=13,client=13,KV=532480,K=3,max_context=65536,generation_explicit={'max_tokens':32768},sampling='original076 default thinking/model sampling; serverseed42, no request sampling/thinking override',samples=100,instructions=111,flag=1,quality_only_route_switch=False,score='Original EvalScope1.13.0 all four verifiers and10000seed42 prompt-cluster CIs; retain errors/empty/length/all attempts and raw reasoning/content',request_payload='byte-identical original076 messages/model/max_tokens; only client concurrency and endpoint change',start_condition='Root explicit execute_quality_after_lowC_complete; do not create release from elapsed time',unknowns=['KV532480 actual fit/startup remains unmeasured for this binary/native13; preserve OOM/unsupported without grid','No076 score inheritance','No published79.5 reproduction or first100 population representativeness claim'],qualification_evidence_sha256=sha(R/'qualification-evidence.json')))
for p in R.glob('*.py'):py_compile.compile(str(p),doraise=True)
write('CPU-preflight.json',dict(syntax='PASS',fixed_cohort_samples=100,instructions=111,recording_entry_byte_identical=True,original_collector_generation_unchanged=True,new_GPU_requests=0,new_processes=0,actual_EvalScope_resolved_preflight='will execute original CPU check at explicit released owner startup; no claim run now'))
files={str(p.relative_to(R)):{'sha256':sha(p),'bytes':p.stat().st_size}for p in R.rglob('*')if p.is_file()and'__pycache__'not in p.parts and p.name!='source-freeze.json'};write('source-freeze.json',{'files':files,'CPU_only':True})
print('CPU frozen',len(files),'files',sha(R/'source-freeze.json'))
