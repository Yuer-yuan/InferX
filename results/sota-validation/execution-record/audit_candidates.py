"""Read-only source downloads to HDD; no code executed or packages installed."""
import concurrent.futures,hashlib,json,os,subprocess,time,urllib.request
from pathlib import Path
root=Path(os.environ['INFERX_ROOT']);out=root/'sota-validation';out.mkdir(exist_ok=True)
cache=out/'sources';cache.mkdir(exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
repos=[('thustorage/MorphX','4404aed4c180ad83c31179e1ce08ae2234390853'),('zejia-lin/BulletServe','c4aa493b737c861128e6a5b6b0b3413cb26aed63')]
result=[]
def fetch(url):
 req=urllib.request.Request(url,headers={'User-Agent':'InferX-readonly-source-audit'})
 with urllib.request.urlopen(req,timeout=60) as response:return response.read()
for repo,commit in repos:
 path=cache/repo.split('/')[-1];path.mkdir(exist_ok=True)
 source=(root/'source/pd-baselines/MorphX' if repo.endswith('/MorphX') else cache/'BulletServe-repo')
 if not source.exists():subprocess.run(['git','clone','--depth','1','--no-checkout','https://github.com/'+repo+'.git',str(source)],check=True)
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==commit
 files=subprocess.check_output(['git','ls-tree','-r','--name-only',commit],cwd=source,text=True).splitlines()
 tree_url=f'https://github.com/{repo}/tree/{commit}';tree_data=json.dumps(dict(source='git ls-tree -r --name-only at verified exact HEAD',commit=commit,paths=files),indent=2).encode();(path/'git-tree.json').write_bytes(tree_data)
 if repo.endswith('/MorphX'):
  selected=['README.md','requirements.txt','docs/environment-packages.txt','scripts/ae/run_pd_colocate_part2.py','scripts/ae/run_pd_colocate_part2.sh','runtime/src/cuda-llm.cpp','include/common.h','scripts/sweep_pd.py','scripts/run.py','smsched-pass/src/SMResourceLimitPass.cpp']
  selected += [p for p in files if ('pd' in p.lower() and p.endswith(('.py','.sh'))) or ('smsched-pass/' in p and p.endswith(('.cpp','.h')))]
 else:
  selected=['README.md','python/sglang/srt/models/registry.py','python/sglang/srt/models/transformers.py','docs/supported_models/transformers_fallback.md','python/sglang/srt/models/qwen3_5.py','python/sglang/srt/models/qwen3.py','python/sglang/srt/model_executor/model_runner.py','python/sglang/srt/managers/scheduler.py','python/sglang/srt/server_args.py','python/pyproject.toml','scripts/start_mps.sh','csrc/README.md']
  selected += [p for p in files if ('bullet' in p.lower() or 'weight' in p.lower() and 'ipc' in p.lower()) and p.endswith(('.py','.cpp','.h','.cc','.c'))]
 selected=sorted(set(selected));entries=[]
 def download(name):
  url=f'https://raw.githubusercontent.com/{repo}/{commit}/{name}'
  if name not in files:return dict(path=name,url=url,status='absent_in_complete_git_tree')
  b=subprocess.check_output(['git','show',commit+':'+name],cwd=source);destination=path/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(b)
  return dict(path=name,url=url,status='downloaded',bytes=len(b),sha256=sha(b))
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for entry in pool.map(download,selected):entries.append(entry)
 result.append(dict(repo=repo,commit=commit,tree_url=tree_url,tree_sha256=sha(tree_data),complete_tree_files=len(files),files=entries,model_files=[p for p in files if p.startswith('python/sglang/srt/models/') and p.endswith('.py')]))
 print(repo,'audited',len(entries),'source files',flush=True)
(out/'source-audit.json').write_text(json.dumps(dict(captured_unix=time.time(),scope='Pinned public source existence and bytes only; no GPU correctness or efficacy claim.',repos=result),indent=2)+'\n')
