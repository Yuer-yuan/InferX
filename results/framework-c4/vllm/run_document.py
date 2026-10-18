"""Reproduce root benchmark MD rows with untouched organizer helpers and raw export."""
import argparse, hashlib, importlib.util, json, math, socket, subprocess, time, urllib.request, uuid
from pathlib import Path
ROOT=Path('/workspace/InferX'); BASE=ROOT/'framework-c4/vllm'; DATA=BASE/'document-datasets'
assert socket.gethostname()=='cpod-1wjh4iq1ibw7'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()=='GPU-e7536e1c-450e-152d-e5e2-92c032a1c353'
rp=ROOT/'code/submission/organizer/runner.py'
assert hashlib.sha256(rp.read_bytes()).hexdigest()=='d80df390cf6d1d50129796b31fdd7f4d1e95d0043d3b942d47ac582118702fb0'
sp=importlib.util.spec_from_file_location('untouched_runner',rp); r=importlib.util.module_from_spec(sp); sp.loader.exec_module(r)
MATRIX=[{'name':f'row{i+1:02d}-c{c}-seed{s}','concurrency':c,'requests':n,'warmup':w,'seed':s} for i,(c,n,w,s) in enumerate([(1,12,4,42),(2,24,4,43),(4,32,4,44),(8,24,8,45),(3,24,4,46),(2,100,4,47)])]
def write(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def prompt_hash(p): return hashlib.sha256(json.dumps(p,separators=(',',':')).encode()).hexdigest()
def snapshot(out,name):
    try: data=urllib.request.urlopen('http://127.0.0.1:8088/metrics',timeout=10).read().decode()
    except Exception as e: data=str(e)
    (out/name).write_text(data); return data
def probe(args,dataset,out):
    entry=r.read_jsonl(dataset)[-1]
    payload={'model':args.model,'prompt':entry['extra']['prompt'],'max_tokens':1,'ignore_eos':True,'temperature':0,'stream':True,'stream_options':{'include_usage':True}}
    write(out/'probe-request.json',payload)
    req=urllib.request.Request(args.base_url+'/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    started=time.time_ns();usage=None;done=False
    with urllib.request.urlopen(req,timeout=args.request_timeout) as response,(out/'probe-response.sse').open('wb') as raw:
        for line in response:
            raw.write(line)
            if not line.startswith(b'data:'):continue
            body=line[5:].strip()
            if body==b'[DONE]':done=True;continue
            j=json.loads(body);assert not j.get('error'),j;usage=j.get('usage') or usage
    result={'request_start_ns':started,'response_end_ns':time.time_ns(),'usage':usage,'done':done,'prompt_hash':prompt_hash(payload['prompt'])}
    write(out/'probe.json',result)
    assert done and usage and usage['prompt_tokens']==8192 and usage['completion_tokens']==1,result
    return result
def find_prompts(obj):
    found=[]
    if isinstance(obj,dict):
        for k,v in obj.items():
            if k=='prompt' and isinstance(v,list) and len(v)==8192 and all(type(x)==int for x in v): found.append(prompt_hash(v))
            else: found.extend(find_prompts(v))
    elif isinstance(obj,list):
        for v in obj: found.extend(find_prompts(v))
    elif isinstance(obj,str) and obj.startswith('{') and '"prompt"' in obj:
        try: found.extend(find_prompts(json.loads(obj)))
        except ValueError: pass
    return found
def audit(dest,dataset,row):
    expected={prompt_hash(j['extra']['prompt']) for j in r.read_jsonl(dataset)}
    spare=prompt_hash(r.read_jsonl(dataset)[-1]['extra']['prompt'])
    groups={'warmup':[],'profiling':[]}; times={'warmup':[],'profiling':[]};issues=[];response_checks=[]
    metric_rows=r.read_jsonl(dest/'profile_export.jsonl')
    def key(j):
        m=j['metadata'];return (m.get('benchmark_phase'),m.get('request_start_ns'),m.get('conversation_id'))
    raw_paths=[dest/'profile_export_raw.jsonl'] if (dest/'profile_export_raw.jsonl').exists() else sorted((dest/'raw_records').glob('raw_records*.jsonl'))
    assert raw_paths,'No raw request/response exports found'
    raw_rows=[j for path in raw_paths for j in r.read_jsonl(path)]
    if {key(j) for j in metric_rows}!={key(j) for j in raw_rows}:issues.append('raw/metric phase correlation mismatch')
    for j in raw_rows:
        phase=j.get('metadata',{}).get('benchmark_phase')
        if phase not in groups: continue
        md=j['metadata'];times[phase].append([md.get('request_start_ns'),md.get('request_end_ns')])
        payload=j.get('payload') or {};hashes=set(find_prompts(payload))
        if len(hashes)!=1: issues.append(f'{phase}: expected one actual prompt hash; found {len(hashes)}')
        else: groups[phase].append(next(iter(hashes)))
        if payload.get('max_tokens')!=512 or payload.get('ignore_eos') is not True or payload.get('temperature')!=0 or 'cache_salt' in payload:issues.append('actual request generation parameters differ from MD')
        usage=None;done=False;reasons=[]
        for response in j.get('responses',[]):
            for packet in response.get('packets',[]):
                if packet.get('name')!='data':continue
                value=packet.get('value','').strip()
                if value=='[DONE]':done=True;continue
                data=json.loads(value);usage=data.get('usage') or usage
                reasons.extend(c['finish_reason'] for c in data.get('choices',[]) if c.get('finish_reason'))
        valid=bool(j.get('status')==200 and not j.get('error') and done and usage and usage.get('prompt_tokens')==8192 and usage.get('completion_tokens')==512 and reasons==['length'])
        response_checks.append({'key':list(key(j)),'valid':valid,'done':done,'usage':usage,'finish_reasons':reasons})
        if not valid:issues.append('raw SSE/usage/finish validation failed')
    if len(groups['warmup'])!=row['warmup'] or len(groups['profiling'])!=row['requests']: issues.append('raw request counts do not match row')
    combined=groups['warmup']+groups['profiling']
    if len(set(combined))!=len(combined): issues.append('warmup/profiling prompts reused')
    if set(combined)-expected: issues.append('sent prompt not in frozen row dataset')
    if spare in combined:issues.append('probe input reused by timed/warmup requests')
    barrier=False
    if times['warmup'] and times['profiling'] and all(type(v)==int for group in times.values() for pair in group for v in pair):
        barrier=max(x[1] for x in times['warmup'])<=min(x[0] for x in times['profiling'])
    if not barrier:issues.append('warmup completion before profiling start not proven')
    probe_info=json.loads((dest.parent/'probe.json').read_text())
    probe_barrier=bool(times['warmup'] and probe_info['response_end_ns']<=min(x[0] for x in times['warmup']))
    if probe_info['prompt_hash']!=spare or not probe_barrier:issues.append('probe identity/completion barrier not proven')
    counters={}
    for label in ['before','after']:
        text=(dest.parent/f'metrics-{label}.txt').read_text()
        values=[float(line.rsplit(' ',1)[1]) for line in text.splitlines() if not line.startswith('#') and line.split('{',1)[0].endswith('prefix_cache_hits_total')]
        counters[label]=sum(values) if values else None
    if None in counters.values():issues.append('prefix cache hit counters not available')
    elif counters['after']!=counters['before']:issues.append('unexpected prefix cache hits in independent workload')
    result={'status':'pass' if not issues else 'fail','issues':issues,'expected_unique_dataset_entries':len(expected),'actual_warmup':len(groups['warmup']),'actual_profiling':len(groups['profiling']),'actual_unique':len(set(combined)),'warmup_profile_intersection':sorted(set(groups['warmup'])&set(groups['profiling'])),'prompt_hashes':groups,'probe_prompt_hash':spare,'probe_finished_before_warmup':probe_barrier,'warmup_finished_before_profiling':barrier,'request_times_ns':times,'raw_response_checks':response_checks,'prefix_cache_hit_counters':counters,'raw_record_files':[str(path) for path in raw_paths]}
    result['cohort_sha256']={name:hashlib.sha256(json.dumps(sorted(values),separators=(',',':')).encode()).hexdigest() for name,values in {**groups,'all':combined}.items()}
    write(dest.parent/'input-audit.json',result); return result
def validate_summary(out,checked):
    path=out/'summary.json';summary=json.loads(path.read_text());summary['raw_audit_status']=checked['status'];summary['baseline_valid']=summary.get('status')=='完整' and not summary.get('command_error') and checked['status']=='pass';summary['validated_slo_met']=summary['baseline_valid'] and summary['both_slo'];write(path,summary)
p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','audit']);p.add_argument('--row',type=int);p.add_argument('--start-row',type=int,default=1);p.add_argument('--campaign',choices=['document-performance','capacity8-performance'],default='document-performance');opts=p.parse_args()
if opts.action=='prepare':
    DATA.mkdir(exist_ok=True); manifest=[]
    for row in MATRIX:
        f=DATA/(row['name']+'.jsonl')
        if not f.exists(): r.dataset_worker(str(ROOT/'models/tokenizer'),None,f,row['requests']+row['warmup']+1,row['seed'])
        lines=r.read_jsonl(f); assert len(lines)==row['requests']+row['warmup']+1
        assert all(len(x['extra']['prompt'])==8192 and x['output_length']==512 for x in lines)
        f.chmod(0o444); manifest.append({**row,'dataset':str(f),'entries':len(lines),'sha256':digest(f)})
    write(DATA/'manifest.json',manifest); print(json.dumps(manifest,indent=2)); raise SystemExit
rows=json.loads((DATA/'manifest.json').read_text()); campaign=BASE/opts.campaign;campaign.mkdir(exist_ok=True)
if opts.action=='audit':
    assert opts.row is not None
    row=rows[opts.row-1];out=campaign/row['name'];checked=audit(out/'aiperf',Path(row['dataset']),row);validate_summary(out,checked);print(json.dumps(checked,ensure_ascii=False),flush=True);raise SystemExit(0 if checked['status']=='pass' else 1)
for index,row in enumerate(rows,1):
    if index<opts.start_row or (opts.row and opts.row!=index): continue
    out=campaign/row['name'];out.mkdir(exist_ok=False);dest=out/'aiperf';dataset=Path(row['dataset']);assert digest(dataset)==row['sha256']
    service=json.loads((BASE/'candidate-service.json').read_text());expected='document-literal' if opts.campaign=='document-performance' else 'capacity8-context32k';assert service['name']==expected,service['name'];write(out/'server-service.json',service)
    args=argparse.Namespace(base_url='http://127.0.0.1:8088/v1',model='qwen3.8-27b-fp8',tokenizer=str(ROOT/'models/tokenizer'),tokenizer_revision=None,env_dir=ROOT/'eval-envs',seed=row['seed'],perf_requests=row['requests'],request_timeout=180)
    salt=str(uuid.uuid4()); before=snapshot(out,'metrics-before.txt');r.check_model(args.base_url,args.model,'EMPTY');probe(args,dataset,out)
    cmd=r.aiperf_command(args,row['concurrency'],dataset,dest,salt+':row','EMPTY')
    cmd[cmd.index('--warmup-request-count')+1]=str(row['warmup']);cmd[cmd.index('--export-level')+1]='raw';write(out/'command.json',cmd)
    extra=json.loads(cmd[cmd.index('--extra-inputs')+1]);extra.pop('cache_salt');cmd[cmd.index('--extra-inputs')+1]=json.dumps(extra);write(out/'command.json',cmd)
    log=(out/'gpu.csv').open('w');sampler=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm','--format=csv','-l','1'],stdout=log,stderr=subprocess.STDOUT)
    started=time.time();error=None
    try:r.execute(cmd,out/'console.log',env=r.child_environment(args.env_dir/'nltk_data'))
    except Exception as e:error=str(e)
    finally:sampler.terminate();sampler.wait(timeout=10);log.close()
    after=snapshot(out,'metrics-after.txt')
    try:summary=r.parse_performance(dest,row['requests'],row['concurrency'],warmup_expected=row['warmup'],strict_isl=True)
    except Exception as e:summary={'status':'异常','issues':[str(e)]}
    summary.update(label='Official MD protocol reproduction; versions/seeds not supplied by MD' if opts.campaign=='document-performance' else 'Capacity8 supplement using frozen MD workload',row=row,started_unix=started,finished_unix=time.time(),command_error=error,runner_sha256=digest(rp),dataset_sha256=digest(dataset))
    m=summary.get('metrics',{});summary['ttft_slo']=m.get('time_to_first_token',{}).get('p95',math.inf)<=10000;summary['tpot_slo']=m.get('inter_token_latency',{}).get('p95',math.inf)<=40;summary['both_slo']=summary.get('status')=='完整' and not error and summary['ttft_slo'] and summary['tpot_slo']
    write(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
    if summary.get('status')!='完整' or error:raise RuntimeError(f'{row["name"]}: incomplete; preserve and investigate')
    checked=audit(dest,dataset,row)
    validate_summary(out,checked)
    if checked['status']!='pass':raise RuntimeError(f'{row["name"]}: raw input audit failed; preserve and investigate')
