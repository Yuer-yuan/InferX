"""Wait for owned PD probe restoration, then replace this process with controller."""
import hashlib,json,os,time
from pathlib import Path
root=Path(os.environ['INFERX_ROOT'])
source=root/'code-pd-baselines/configs/pd-strategies.json'
d=json.loads(source.read_text());byid={c['id']:c for c in d['cases']}
ids=['sglang-pd-reserve128','sglang-pd-control','sglang-pd-reserve256']
config=root/'pd-strategies/diagnostic-config-v2.json'
with config.open('x') as out:out.write(json.dumps({'cases':[byid[x] for x in ids]},indent=2)+'\n')
sha=hashlib.sha256(config.read_bytes()).hexdigest()
prior=root/'pd-baselines/raw/pd-strategies-reserve128-128k-v2/campaign.json'
deadline=time.monotonic()+900
while True:
 d=json.loads(prior.read_text())
 if d['status']=='restored':
  assert d.get('restored') and all(x['state']=='measured' for x in d['cases'])
  paths=list(prior.parent.glob('*/probe-128k/requests.jsonl'))
  assert len(paths)==1
  rows=[json.loads(x) for x in paths[0].read_text().splitlines()]
  assert len(rows)==1 and rows[0]['functional_ok'] is True and rows[0]['summary']['ok']
  assert not rows[0]['summary'].get('truncated') and rows[0]['summary']['prompt_tokens']>=130000
  break
 if d['status']!='running':raise RuntimeError('prior restoration failed')
 if time.monotonic()>deadline:raise TimeoutError('prior probe did not restore in15min')
 time.sleep(1)
assert hashlib.sha256(config.read_bytes()).hexdigest()==sha
os.chdir(root/'code-pd-baselines')
command=['taskset','-c','0-3,56-59',str(root/'repo-venv/bin/python'),'-m','scripts.run_pd_baselines',
 '--config',str(config),'--cases',*ids,'--run-id','pd-strategies-simple-diagnostic-v2','--repeat','1',
 '--groups','short-control','cold-low','resume-high','--short-rate','.5']
print(json.dumps({'event':'exec_simple_diagnostic','config_sha256':sha,'argv':command}),flush=True)
os.execvpe(command[0],command,dict(os.environ))
