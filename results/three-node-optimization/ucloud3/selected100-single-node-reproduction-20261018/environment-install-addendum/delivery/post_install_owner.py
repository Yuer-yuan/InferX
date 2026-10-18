from pathlib import Path
import os,json,time,subprocess,hashlib,traceback
from owned_process import Owned,identity,write_json
R=Path(__file__).resolve().parent;J=R/'post-install-job.json';assert not J.exists();assert json.loads((R/'job.json').read_text())['status']=='complete'
E={k:v for k,v in os.environ.items()if not k.startswith(('NINFER_','CUDA_MPS_','PYTHON','CONDA_','VIRTUAL_ENV'))};E.update(PATH=str(R/'tools')+':'+str(R/'envs/build/bin')+':/usr/local/cuda-13.2/bin:'+E.get('PATH',''),LD_LIBRARY_PATH=str(R/'envs/build/lib')+':/usr/local/cuda-13.2/lib64',NLTK_DATA=str(R/'eval-envs/nltk_data'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
j=dict(status='running',stage='preflight',owner=identity(os.getpid()),started_ns=time.time_ns(),handles=[],steps=[]);write_json(J,j);active=None

def run(name,argv):
 global active
 active=Owned([str(x)for x in argv],R/(name+'.log'),env=E,role=name);j['handles'].append(active.record);j['stage']=name;write_json(J,j)
 while active.child.poll()is None:j['heartbeat_ns']=time.time_ns();write_json(J,j);time.sleep(2)
 rc=active.child.returncode;cleanup=active.stop();active=None;j['steps'].append(dict(name=name,argv=[str(x)for x in argv],exit_code=rc,cleanup=cleanup));write_json(J,j);assert rc==0,name
try:
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 run('actual-v1-prepare-assets',[R/'envs/build/bin/python',R/'cpu-linkage-run/prepare_assets.py'])
 link=R/'cpu-linkage-run/cache/modelscope';assert link.is_symlink()and link.resolve()==R/'cache/modelscope'
 run('original-EvalScope-fixed-data-load',[R/'eval-envs/evalscope-1.13.0/bin/python',R/'ifbench-fixed-dataset-cache-r1/check_offline_cache.py','--cache-root',link,'--receipt',R/'ifbench-original-EvalScope-loader-check.json','--evalscope'])
 code='from pathlib import Path; import json,hashlib; from evalscope.benchmarks.ifbench import evaluation_lib; actual={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(evaluation_lib.__file__).parent.glob("*.py")}; expected=json.loads(Path("'+str(R/'expected-verifier-files.json')+'").read_text()); assert actual==expected; print(json.dumps(actual,indent=2))'
 run('new-verifier-source-identity',[R/'eval-envs/evalscope-1.13.0/bin/python','-c',code])
 run('new-libraries-resolve-measured-binary',['ldd','/workspace/InferX/reproductions/selected100-single-node-20261018/artifacts/100-k3-rebuilt-20261018/ninfer-serve'])
 assert 'not found'not in(R/'new-libraries-resolve-measured-binary.log').read_text()
 run('old-vs-new-environment-comparison',[R/'envs/build/bin/python',R/'compare_environments.py'])
 expected={line.split('#')[0].rsplit('/',1)[-1]:line.split('#')[1]for line in(R/'build-conda-explicit.lock').read_text().splitlines()if line.startswith('https://')}
 conda=[json.loads(p.read_text())for p in(R/'envs/build/conda-meta').glob('*.json')];actual={x['url'].rsplit('/',1)[-1]:x['md5']for x in conda};assert actual==expected and len(actual)==133
 write_json(R/'new-environment-identity.json',dict(conda_packages=conda,conda_exact133_lock_match=True,prefix=str(R),engine_rebuilt_in_new_environment=False,GPU_runs=0,original_runner_unchanged=True,verifier6_source_exact_match=True,actual_symlink=dict(path=str(link),resolved=str(link.resolve())),dataset_loader_receipt=json.loads((R/'ifbench-original-EvalScope-loader-check.json').read_text())))
 j.update(status='complete',stage='fresh_env_and_original_v1_data_linkage_verified',finished_ns=time.time_ns());write_json(J,j)
except BaseException as e:j.update(status='failed',error=repr(e),traceback=traceback.format_exc(),finished_ns=time.time_ns());write_json(J,j);raise
finally:
 if active:j['cleanup']=active.stop(timeout=15);write_json(J,j)
