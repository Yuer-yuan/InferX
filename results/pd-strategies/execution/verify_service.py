"""Read ownership/argv and probe both public tool protocols after restoration."""
import json,subprocess,time,urllib.request
from pathlib import Path
from inferx.service_ownership import process_identity
from scripts import run_concurrency_campaign as previous
from scripts.run_pd_baselines import get
r=Path('/home/guokc/mypro/gnn-mount-sda1/InferX');out=r/'pd-strategies';a=json.loads((r/'tuning/best-service.json').read_text());assert a==json.loads((r/'context-agents/runtime.json').read_text());identity=process_identity(a['pid'],previous.BINARY);assert all(a.get(k)==v for k,v in identity.items())
argv=Path(f'/proc/{a["pid"]}/cmdline').read_bytes().decode().rstrip('\0').split('\0');expected=json.loads((r/'pd-baselines/raw/pd-strategies-pdmux-profile-v1/original-service.json').read_text())['original_argv'];assert argv==expected
pairs=previous.gpu_processes();assert (a['pid'],previous.GPU) in pairs
gpu0=[pair for pair in pairs if pair[1]=='GPU-a66b08e8-8b65-0faf-22e2-b951a97b2b03']
assert not gpu0, f'GPU0 changed again: {gpu0}'
assert sum(uuid==previous.GPU for _,uuid in pairs)==1
assert get('/health','http://127.0.0.1:18081')['status']=='ok'
models=get('/v1/models','http://127.0.0.1:18081');assert models['data'][0]['context_window']==262144
endpoint='http://127.0.0.1:18081';messages=[dict(role='user',content='Use add_numbers to calculate 2 + 3. After the tool result, answer with only the number.')];function=dict(name='add_numbers',description='Add two integers.',parameters=dict(type='object',properties={'a':{'type':'integer'},'b':{'type':'integer'}},required=['a','b'],additionalProperties=False));payload=dict(model='qwen38-27b',max_tokens=256,temperature=0,messages=messages,tools=[dict(type='function',function=function)],tool_choice=dict(type='function',function=dict(name='add_numbers')))
def post(body):
 req=urllib.request.Request(endpoint+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=120) as response:return json.load(response)
first=post(payload);message=first['choices'][0]['message'];calls=message.get('tool_calls',[]);assert len(calls)==1 and calls[0]['function']['name']=='add_numbers' and json.loads(calls[0]['function']['arguments'])=={'a':2,'b':3}
messages.extend([message,dict(role='tool',tool_call_id=calls[0]['id'],content='5')]);payload['tool_choice']='auto';second=post(payload);assert second['choices'][0]['message']['content'].strip().strip('`* .\n')=='5';(out/'final-openai-tools.json').write_text(json.dumps(dict(passed=True,first=first,second=second),indent=2)+'\n')
command=[str(r/'repo-venv/bin/python'),'-m','scripts.check_tuning_tools','--output',str(out/'final-anthropic-tools.json')];done=subprocess.run(command,cwd=r/'code-pd-baselines',capture_output=True,text=True,timeout=240);assert done.returncode==0,done.stdout+done.stderr
report=dict(verified_unix=time.time(),manifest_equal=True,identity=identity,argv_matches_preprofile=True,argv=argv,gpu0_state='user_intentionally_stopped',gpu0_restarted_by_experiment=False,gpu_processes=sorted(pairs),models=models,openai_tool_roundtrip=True,anthropic_tool_roundtrip=True,anthropic_probe_stdout=done.stdout)
(out/'final-service-verification.json').write_text(json.dumps(report,indent=2)+'\n');print('restored ownership/argv/262144context/OpenAI+Anthropic tool roundtrips verified',a['pid'])
