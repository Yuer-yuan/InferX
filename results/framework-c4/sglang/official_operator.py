#!/usr/bin/env python3
"""Explicit MD protocol wrapper around unchanged organizer runner helpers."""
import argparse,hashlib,importlib.util,json,os,pathlib,socket,subprocess,sys,time,urllib.request,uuid
P=pathlib.Path;ROOT=P('/workspace/InferX/framework-c4/sglang');RUNNER=P('/workspace/InferX/code/submission/organizer/runner.py')
ROWS=[(1,12,4,42),(2,24,4,43),(4,32,4,44),(8,24,8,45),(3,24,4,46),(2,100,4,47)]
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def guard():
 assert socket.gethostname()=='cpod-1wjk44m2ji94'
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()=='GPU-c1c59c83-fc0a-2baf-58a5-84bd4020924c'
def runner():
 assert sha(RUNNER)=='d80df390cf6d1d50129796b31fdd7f4d1e95d0043d3b942d47ac582118702fb0'
 spec=importlib.util.spec_from_file_location('original_runner',RUNNER);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);return r
def row_name(index,row):return f'{index+1:02d}-c{row[0]}-n{row[1]}-s{row[3]}'
def prepare():
 guard();r=runner();folder=ROOT/'official-datasets';folder.mkdir(exist_ok=False);manifest=[]
 env=r.child_environment(P('/workspace/InferX/eval-envs/nltk_data'))
 for i,row in enumerate(ROWS):
  c,n,w,seed=row;path=folder/(row_name(i,row)+'.jsonl');command=['/workspace/InferX/eval-envs/aiperf-0.13.0/bin/python',str(RUNNER),'_dataset','/workspace/InferX/models/tokenizer','',str(path),str(n+w+1),str(seed)]
  write(path.with_suffix('.command.json'),command);r.execute(command,path.with_suffix('.log'),env=env)
  data=[json.loads(s)['extra']['prompt'] for s in path.read_text().splitlines()];hashes=[hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest() for x in data]
  assert len(set(hashes))==n+w+1 and all(len(x)==8192 for x in data)
  manifest.append(dict(name=path.stem,concurrency=c,requests=n,warmup=w,seed=seed,entries=len(data),sha256=sha(path),prompt_hashes=hashes))
 write(folder/'manifest.json',manifest);print(json.dumps(manifest,ensure_ascii=False),flush=True)
def measure(index):
 guard();r=runner();row=ROWS[index];c,n,w,seed=row;name=row_name(index,row);folder=ROOT/'official-performance'/name;folder.mkdir(parents=True,exist_ok=False)
 dataset=ROOT/'official-datasets'/(name+'.jsonl');manifest=json.loads((ROOT/'official-datasets/manifest.json').read_text())[index];assert sha(dataset)==manifest['sha256']
 args=argparse.Namespace(base_url='http://127.0.0.1:8089/v1',model='qwen3.8-27b-fp8',tokenizer='/workspace/InferX/models/tokenizer',tokenizer_revision=None,seed=seed,request_timeout=180,perf_requests=n,env_dir=P('/workspace/InferX/eval-envs'))
 r.check_model(args.base_url,args.model,'EMPTY');command=r.aiperf_command(args,c,dataset,folder,uuid.uuid4().hex+':'+name,'EMPTY')
 extra=json.loads(command[command.index('--extra-inputs')+1]);extra.pop('cache_salt',None);command[command.index('--extra-inputs')+1]=json.dumps(extra)
 command[command.index('--warmup-request-count')+1]=str(w);command[command.index('--export-level')+1]='raw';write(folder/'command.json',command)
 write(folder/'metadata.json',dict(protocol='official MD corresponding SGLang baseline',row=index+1,dataset=manifest,runner_sha256=sha(RUNNER),started=r.utc_now(),explicit_helper_deltas=['row concurrency/count/warmup/seed','timeout180','export-level raw','remove helper cache_salt absent from MD'],note='Native server defaults; original MD does not publish exact original seeds or framework versions.'))
 failure=None
 current=json.loads((ROOT/'current-sglang.json').read_text());serverlog=ROOT/current['name']/'server.log';offset=serverlog.stat().st_size
 with urllib.request.urlopen('http://127.0.0.1:8089/server_info',timeout=10) as response:(folder/'server-info-before.json').write_bytes(response.read())
 with (folder/'gpu.csv').open('w') as f:
  sampler=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm,clocks.mem,temperature.gpu','--format=csv','--loop=1'],stdout=f,stderr=subprocess.STDOUT)
  try:
   try:r.execute(command,folder/'console.log',env=r.child_environment(args.env_dir/'nltk_data'))
   except Exception as e:failure=str(e);write(folder/'execution-error.json',dict(error=failure))
  finally:sampler.terminate();sampler.wait(timeout=10)
 with serverlog.open('rb') as f:f.seek(offset);(folder/'server-interval.log').write_bytes(f.read())
 with urllib.request.urlopen('http://127.0.0.1:8089/server_info',timeout=10) as response:(folder/'server-info-after.json').write_bytes(response.read())
 result=r.parse_performance(folder,n,c,warmup_expected=w,strict_isl=True)
 if failure:result['status']='异常';result['issues'].append('command execution failed: '+failure)
 result.update(ttft_slo_met=result.get('metrics',{}).get('time_to_first_token',{}).get('p95',float('inf'))<=10000,tpot_slo_met=result.get('metrics',{}).get('inter_token_latency',{}).get('p95',float('inf'))<=40)
 result['both_slo_met']=result['status']=='完整' and result['ttft_slo_met'] and result['tpot_slo_met'];write(folder/'summary.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare()
 elif sys.argv[1]=='measure':measure(int(sys.argv[2])-1)
