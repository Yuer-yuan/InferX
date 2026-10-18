from pathlib import Path
import sys,os,json,hashlib,time,subprocess,importlib.util
R=Path(__file__).resolve().parent
assert not (R/'asset-verification.json').exists()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def save(name,data):(R/name).write_text(json.dumps(data,indent=2)+'\n')
def link(rel,target):
 p=R/rel;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists() and not p.is_symlink();p.symlink_to(target)
links={'assets/model/qwen3_8_27b.ninfer':'/workspace/InferX/models/qwen38-ninfer/qwen3_8_27b.ninfer','assets/tokenizer':'/workspace/InferX/models/tokenizer','envs/build':'/workspace/InferX/envs/build','eval-envs':'/workspace/InferX/eval-envs','tools':'/workspace/InferX/tools','cache/modelscope':'/workspace/InferX/cache/modelscope'}
for n in ('weights025','weights041','weights047'):links['assets/'+n]='/workspace/InferX/experiments/ucloud3-c32-tonight/'+n
for rel,target in links.items():link(rel,target)
save('deployment-layout.json',dict(root=str(R),links=links,read_only_reuse=True,scope='New source/build/deployment on existing host. Reuses model, quantized caches, tokenizer and installed tool/evaluation environments; not clean-OS or fresh model quantization.'))
contract=json.loads((R/'entrypoint/candidate_recipes/assets-u3-47.json').read_text());records={};started=time.time_ns()
p=R/'assets/model/qwen3_8_27b.ninfer';model=dict(path=str(p),resolved=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p));assert {k:model[k]for k in ('bytes','sha256')}==contract['model'];save('model-verification.json',dict(**model,source='https://huggingface.co/neroued/Qwen3.8-27B-NInfer/resolve/3526913004b1cf552cb57b88d6a5c6f5e4a89a70/qwen3_8_27b.ninfer',content_sha_rehashed=True))
for cache,spec in contract['caches'].items():
 root=R/'assets'/cache;actual={str(p.relative_to(root))for p in root.glob('weights-L*/*')if p.name in ('packed.bin','scales.f32') and p.is_file()};assert actual==set(spec['files'])
 for rel,item in spec['files'].items():
  p=root/rel;entry=dict(bytes=p.stat().st_size,sha256=sha(p));assert entry==item,(cache,rel);records[cache+'/'+rel]=entry
save('asset-verification.json',dict(complete=True,content_sha_rehashed=True,model=model,cache_files=records,cache_file_count=len(records),started_ns=started,finished_ns=time.time_ns(),contract_sha256=sha(R/'entrypoint/candidate_recipes/assets-u3-47.json')))
for key,root in [('tokenizer',R/'assets/tokenizer'),('xgrammar',Path('/workspace/InferX/source/xgrammar_source-src'))]:
 files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in sorted(root.rglob('*'))if p.is_file() and '.git'not in p.parts and '__pycache__'not in p.parts};save(key+'-manifest.json',dict(root=str(root),resolved=str(root.resolve()),files=files,aggregate_sha256=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()))
E=os.environ.copy();E['PATH']=str(R/'tools')+':'+str(R/'envs/build/bin')+':'+E.get('PATH','')
p=subprocess.run(['python3',str(R/'entrypoint/organizer/runner.py'),'--mode','check','--env-dir',str(R/'eval-envs')],text=True,capture_output=True,env=E);(R/'evaluation-dependency-check.log').write_text(p.stdout+p.stderr);save('evaluation-dependency-check.json',dict(exit_code=p.returncode,status=json.loads(p.stdout)));assert p.returncode==0
save('asset-preparation-complete.json',dict(status='complete',at_ns=time.time_ns()))
