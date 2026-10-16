"""Experimental campaign driver; each server is an owned child, never a PID-file kill."""
import json,os,subprocess,time,urllib.request
from pathlib import Path
root=Path(os.environ['INFERX_ROOT']); directory=root/'tuning'
configs=json.loads(Path(os.environ['INFERX_GRID']).read_text())
for cfg in configs:
 name=cfg['name']; log=(directory/(name+'.log')).open('w')
 args=['bash',str(root/'code-tuning/scripts/start_tuned_server.sh')]+list(map(str,cfg['args']))
 (directory/(name+'-launch.json')).write_text(json.dumps(dict(args=args,started=time.time()),indent=2))
 env={**os.environ,'INFERX_REQUEST_LOG':str(directory/(name+'-requests.jsonl'))}
 proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env)
 try:
  for _ in range(240):
   if proc.poll() is not None: raise RuntimeError(f'{name} startup exited {proc.returncode}')
   try:
    urllib.request.urlopen('http://127.0.0.1:18081/health',timeout=1).read(); break
   except Exception: time.sleep(.5)
  else: raise RuntimeError(f'{name} startup timeout')
  time.sleep(1)
  for job in cfg.get('jobs',[dict(corpus='dev',cache='cold',repeat=1,concurrency=1)]):
   suffix=job.get('suffix',job['corpus'])
   cmd=[str(root/'repo-venv/bin/python'),'-m','scripts.bench_tuning','--corpus',str(directory/(job['corpus']+'.json')),'--output',str(directory/(name+'-'+suffix+'.jsonl')),'--config',name,'--cache',job['cache'],'--repeat',str(job['repeat']),'--concurrency',str(job['concurrency'])]
   print('BENCH',name,suffix,flush=True)
   subprocess.run(cmd,cwd=root/'code-tuning',check=True)
  subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,power.draw,temperature.gpu','--format=csv'],stdout=(directory/(name+'-gpu.txt')).open('w'),check=True)
 finally:
  proc.terminate()
  try: proc.wait(timeout=30)
  except subprocess.TimeoutExpired: proc.kill();proc.wait()
  log.close()
  time.sleep(2)
