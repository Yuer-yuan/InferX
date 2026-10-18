import os,json,time,hashlib,subprocess,traceback,fcntl,re
from pathlib import Path
from owned_process import Owned,identity,write_json
R=Path(__file__).resolve().parent;J=R/'build-job.json';assert not J.exists()
lock=(Path('/workspace/InferX/experiments/u3-c12-c16-latency-20261018/.ucloud3-performance-owner.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
j=dict(status='running',stage='preflight',owner=identity(os.getpid()),handles=[],steps=[],started_ns=time.time_ns());write_json(J,j);active=None
E={k:v for k,v in os.environ.items() if not k.startswith(('NINFER_','CUDA_MPS_'))};E.update(PATH='/workspace/InferX/envs/build/bin:/usr/local/cuda-13.2/bin:'+E.get('PATH',''),LD_LIBRARY_PATH='/workspace/InferX/envs/build/lib:/usr/local/cuda-13.2/lib64',PKG_CONFIG_PATH='/workspace/InferX/envs/build/lib/pkgconfig',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def source_identity(name):
 expected=json.loads((R/'source-identity.json').read_text());files={str(p.relative_to(R/'source')):sha(p)for p in sorted((R/'source').rglob('*'))if p.is_file()};assert files==expected['files'];aggregate=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest();assert aggregate==expected['aggregate_sha256']=='c976692d9657408308193b50772c700382a6e243d9da353165205dc5252f76a8';write_json(R/name,dict(files=files,aggregate_sha256=aggregate,at_ns=time.time_ns()));return aggregate
def run(name,args):
 global active
 active=Owned(args,R/(name+'.log'),env=E,role=name);j.update(stage=name);j['handles'].append(active.record);write_json(J,j)
 while active.child.poll()is None:j['heartbeat_ns']=time.time_ns();write_json(J,j);time.sleep(2)
 rc=active.child.returncode;rec=active.stop();active=None;j['steps'].append(dict(name=name,argv=args,exitcode=rc,cleanup=rec));write_json(J,j);assert rc==0,name+' failed'
try:
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True);assert not gpu.strip(),gpu;write_json(R/'GPU-empty-before.json',dict(compute_processes=gpu,at_ns=time.time_ns()))
 for p in Path('/proc').glob('[0-9]*/comm'):
  try:assert p.read_text().strip()not in ['ninfer-serve','nvcc','ptxas','cicc','cc1plus','ninja','nsys','ncu'],'contending workload'
  except(FileNotFoundError,ProcessLookupError):pass
 for n,h in json.loads((R/'controller-freeze.json').read_text())['files'].items():assert sha(R/n)==h,n
 assert len(json.loads((R/'source-identity.json').read_text())['files'])==1023;j['source_before']=source_identity('actual-source-before.json');j['preflight_GPU_empty']=True;write_json(J,j)
 E['CC']='/workspace/InferX/envs/build/bin/x86_64-conda-linux-gnu-gcc';E['CXX']='/workspace/InferX/envs/build/bin/x86_64-conda-linux-gnu-g++'
 run('SDK',['/usr/local/cuda-13.2/bin/nvcc','--version']);run('host-SDK',[E['CXX'],'--version'])
 assert 'V13.2.51' in (R/'SDK.log').read_text() and '13.4.0' in (R/'host-SDK.log').read_text()
 run('configure',['/workspace/InferX/envs/build/bin/cmake','-S',str(R/'source'),'-B',str(R/'build'),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=89','-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc','-DCMAKE_CUDA_HOST_COMPILER='+E['CXX'],'-DCMAKE_PREFIX_PATH=/workspace/InferX/envs/build','-DNINFER_BUILD_APPS=ON','-DBUILD_TESTING=ON','-DNINFER_BUILD_BENCHMARKS=OFF','-DNINFER_ENABLE_UI=OFF','-DFETCHCONTENT_SOURCE_DIR_XGRAMMAR_SOURCE=/workspace/InferX/source/xgrammar_source-src'])
 run('normal-build',['/workspace/InferX/envs/build/bin/cmake','--build',str(R/'build'),'--target','ninfer-serve','ninfer_slo_bounded_decode_membership_test','ninfer_slo_completion_policy_test','ninfer_slo_prefill_queue_policy_test','ninfer_slo_budget_policy_test','--parallel','4'])
 run('registered-CPU-CTest',['/workspace/InferX/envs/build/bin/ctest','--test-dir',str(R/'build'),'--output-on-failure','-R','^ninfer_slo_(bounded_decode_membership.*|completion_policy_test|prefill_queue_policy_test|budget_policy_test)$'])
 checks=(R/'registered-CPU-CTest.log').read_text();assert '100% tests passed, 0 tests failed out of 7'in checks
 binary=R/'build/apps/ninfer-serve';assert binary.is_file();j['binary']=dict(path=str(binary),bytes=binary.stat().st_size,sha256=sha(binary));write_json(J,j)
 run('runtime-dependencies',['ldd',str(binary)])
 ldd=(R/'runtime-dependencies.log').read_text();assert 'not found'not in ldd
 deps={os.path.normpath(p)for p in re.findall(r'(?:=>\s+|^\s*)(/[^\s]+)',ldd,re.M)};libraries={p:dict(bytes=Path(p).stat().st_size,sha256=sha(p),realpath=str(Path(p).resolve()))for p in sorted(deps)}
 assert len(libraries)==55,len(libraries)
 j['source_after']=source_identity('actual-source-after.json');write_json(R/'runtime-identity.json',dict(binary=str(binary),binary_sha256=j['binary']['sha256'],source_sha256=j['source_after'],parent_original_binary_sha256='e4ed1bf4baebf7da3a81e4aed01307c7e3e7ae97faef7672c507ed21bdaeb38c',frozen_source_files=1023,normal_full_rebuild=True,GPU_cubin_identity_not_claimed=True,runtime_dependencies=libraries))
 for n,h in json.loads((R/'controller-freeze.json').read_text())['files'].items():assert sha(R/n)==h,n
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 j.update(status='complete',stage='normal_serve_and_registered7_CPU_terminal',finished_ns=time.time_ns());write_json(J,j)
except BaseException as e:j.update(status='failed',error=repr(e),traceback=traceback.format_exc(),finished_ns=time.time_ns());write_json(J,j);raise
finally:
 if active:j['exception_cleanup']=active.stop(timeout=60);write_json(J,j)
