"""Serialize owned campaigns; exec avoids an intermediate parent/cleanup tree."""
import argparse,hashlib,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--prior',required=True);p.add_argument('--run-id',required=True)
p.add_argument('--cases',nargs='+',required=True);p.add_argument('--probe-only',action='store_true')
p.add_argument('--repeat',type=int,default=1);a=p.parse_args()
root=Path(os.environ['INFERX_ROOT']);assert root==Path('/home/guokc/mypro/gnn-mount-sda1/InferX')
for value in [a.prior,a.run_id]:assert Path(value).name==value and value not in {'.','..'}
raw=root/'pd-baselines/raw';source=root/'code-pd-baselines/configs/pd-strategies.json'
byid={x['id']:x for x in json.loads(source.read_text())['cases']}
selected=[byid[x] for x in a.cases]
config=root/'pd-strategies'/f'{a.run_id}-config.json'
with config.open('x') as f:f.write(json.dumps({'cases':selected},indent=2)+'\n')
config_sha=hashlib.sha256(config.read_bytes()).hexdigest();deadline=time.monotonic()+14400
while True:
 try:
  previous=json.loads((raw/a.prior/'campaign.json').read_text())
 except (FileNotFoundError,json.JSONDecodeError):previous={'status':'running'}
 if previous['status']!='running':
  assert previous['status']=='restored' and previous.get('restored'),previous
  break
 if time.monotonic()>deadline:raise TimeoutError('prior controller did not restore')
 time.sleep(1)
assert hashlib.sha256(config.read_bytes()).hexdigest()==config_sha
for case in selected:
 overlay=Path(case['environment']['PYTHONPATH']);assert overlay.resolve().is_relative_to(root)
 manifest=json.loads((overlay/'overlay-manifest.json').read_text());assert manifest==case['overlay_manifest']
 inventory=overlay/'source-hashes.json';assert hashlib.sha256(inventory.read_bytes()).hexdigest()==manifest['source_hashes_sha256']
 for relative,digest in json.loads(inventory.read_text()).items():
  f=overlay/relative;assert f.resolve().is_relative_to(overlay)
  assert hashlib.sha256(f.read_bytes()).hexdigest()==digest,str(f)
 if case.get('pdmux_config'):
  f=root/case['pdmux_config'];assert f.resolve().is_relative_to(root)
  assert hashlib.sha256(f.read_bytes()).hexdigest()==case['pdmux_config_sha256']
os.chdir(root/'code-pd-baselines')
command=['taskset','-c','0-3,56-59',str(root/'repo-venv/bin/python'),'-m','scripts.run_pd_baselines',
 '--config',str(config),'--cases',*a.cases,'--run-id',a.run_id,'--repeat',str(a.repeat)]
command+=['--probe-only'] if a.probe_only else ['--groups','short-control','cold-low','resume-high','--short-rate','.5']
print(json.dumps({'event':'verified_exec','pid':os.getpid(),'config_sha256':config_sha,'argv':command}),flush=True)
os.execvpe(command[0],command,dict(os.environ))
