from pathlib import Path
import os,json,time,subprocess,hashlib,traceback,shutil
from owned_process import Owned,identity,write_json
R=Path(__file__).resolve().parent;J=R/'job.json';assert not J.exists();assert not(R/'envs').exists()and not(R/'eval-envs').exists()
for d in ('config','cache','tools','source'): (R/d).mkdir(exist_ok=True)
j=dict(status='running',stage='preflight',owner=identity(os.getpid()),started_ns=time.time_ns(),handles=[],steps=[],fresh_prefix=str(R),old_environment_mutated=False,download_cache_reused='/workspace/InferX/pkgs');write_json(J,j)
E={k:v for k,v in os.environ.items()if not k.startswith(('NINFER_','CUDA_MPS_','CONDA_','MAMBA_','PYTHON','PIP_','VIRTUAL_ENV'))};E.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CONDA_PKGS_DIRS='/workspace/InferX/pkgs',PIP_DEFAULT_TIMEOUT='45',PIP_DISABLE_PIP_VERSION_CHECK='1',PYTHONDONTWRITEBYTECODE='1')
active=None;deadline=time.monotonic()+1200
def run(name,args):
 global active
 assert time.monotonic()<deadline,'20 minute installation limit reached'
 active=Owned([str(x)for x in args],R/(name+'.log'),env=E,role=name);j['handles'].append(active.record);j.update(stage=name);write_json(J,j)
 while active.child.poll()is None:
  if time.monotonic()>deadline:raise TimeoutError('20 minute installation limit reached during '+name)
  j['heartbeat_ns']=time.time_ns();write_json(J,j);time.sleep(2)
 rc=active.child.returncode;cleanup=active.stop();active=None;j['steps'].append(dict(name=name,argv=[str(x)for x in args],exit_code=rc,cleanup=cleanup));write_json(J,j)
 assert rc==0,name+' failed; preserved original install output'
try:
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True);assert not gpu.strip();write_json(R/'GPU-before.json',dict(compute_apps=gpu))
 run('mamba-version',['/workspace/InferX/tools/micromamba','--version'])
 run('conda-build-install',['/workspace/InferX/tools/micromamba','--no-rc','create','--yes','--root-prefix',R/'cache/mamba','--prefix',R/'envs/build','--file',R/'build-conda-explicit.lock'])
 lines=(R/'build-pip-freeze.txt').read_text().splitlines();local=[x for x in lines if ' @ file:'in x];assert len(local)==1 and local[0].startswith('packaging @ file:///');(R/'config/build-pip-install.txt').write_text('\n'.join(x for x in lines if x not in local)+'\n')
 py=R/'envs/build/bin/python';E.update(PATH=str(R/'tools')+':'+str(R/'envs/build/bin')+':/usr/local/cuda-13.2/bin:'+E.get('PATH',''),LD_LIBRARY_PATH=str(R/'envs/build/lib')+':/usr/local/cuda-13.2/lib64',NLTK_DATA=str(R/'eval-envs/nltk_data'))
 run('build-pip-install',[py,'-m','pip','install','--no-deps','--index-url','https://pypi.org/simple','-r',R/'config/build-pip-install.txt'])
 run('build-packaging',[py,'-c','import packaging; assert packaging.__version__ == "26.3"; print(packaging.__version__)'])
 run('uv-install',[py,'-m','pip','install','--no-deps','--index-url','https://pypi.org/simple','uv==0.11.29'])
 run('uv-version',[R/'envs/build/bin/uv','--version']);(R/'tools/uv').symlink_to(R/'envs/build/bin/uv')
 run('build-pip-check',[py,'-m','pip','check'])
 for package,version in [('aiperf','0.13.0'),('evalscope','1.13.0')]:
  prefix=R/'eval-envs'/(package+'-'+version);ep=prefix/'bin/python'
  run(package+'-venv',[py,'-m','venv',prefix])
  run(package+'-pip-version',[ep,'-m','pip','install','--index-url','https://pypi.org/simple','pip==26.2.1'])
  run(package+'-install',[ep,'-m','pip','install','--no-deps','--index-url','https://pypi.org/simple','-r',R/(package+'-pip-freeze.txt')])
  run(package+'-pip-check',[ep,'-m','pip','check'])
  run(package+'-freeze',[ep,'-m','pip','freeze'])
  run(package+'-identity',[ep,'-c','import sys,importlib.metadata,json; print(json.dumps(dict(prefix=sys.prefix,base_prefix=sys.base_prefix,executable=sys.executable,python=sys.version,version=importlib.metadata.version("'+package+'"))))'])
 run('nltk-download-and-verify',[py,R/'prepare_nltk.py'])
 run('nltk-locate',[R/'eval-envs/evalscope-1.13.0/bin/python','-c','import nltk; [print(nltk.data.find(x)) for x in ("tokenizers/punkt_tab", "corpora/stopwords", "taggers/averaged_perceptron_tagger_eng")]'])
 run('xgrammar-clone',['git','clone','https://github.com/mlc-ai/xgrammar.git',R/'source/xgrammar_source-src'])
 run('xgrammar-checkout',['git','-C',R/'source/xgrammar_source-src','checkout','4546988d1cf51670a10eafb8795d37e4a2f0d833'])
 run('xgrammar-submodules',['git','-C',R/'source/xgrammar_source-src','submodule','update','--init','--recursive'])
 run('compiler-version',[R/'envs/build/bin/x86_64-conda-linux-gnu-g++','--version'])
 run('CUDA-version',['/usr/local/cuda-13.2/bin/nvcc','--version'])
 run('original-runner-check',[py,R/'runner.py','--mode','check','--env-dir',R/'eval-envs'])
 for prefix in (R/'envs/build',R/'eval-envs/aiperf-0.13.0',R/'eval-envs/evalscope-1.13.0'):
  assert prefix.is_dir()and not prefix.is_symlink();assert(prefix/'bin/python').resolve().is_relative_to(R)
 j.update(status='complete',stage='fresh_dependency_install_and_original_runner_check',finished_ns=time.time_ns());write_json(J,j)
except BaseException as e:
 j.update(status='failed',error=repr(e),traceback=traceback.format_exc(),finished_ns=time.time_ns());write_json(J,j);raise
finally:
 if active:j['cleanup']=active.stop(timeout=15);write_json(J,j)
