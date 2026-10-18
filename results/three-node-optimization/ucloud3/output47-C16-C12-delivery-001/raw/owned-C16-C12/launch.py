#!/usr/bin/python3
import json,os,pathlib,subprocess,time
P=pathlib.Path(__file__).resolve().parent
assert not(P/'job.json').exists() and not(P/'launch.json').exists()
with(P/'supervisor.log').open('x')as log:
 c=subprocess.Popen(['/usr/bin/python3',str(P/'owner.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 p=pathlib.Path('/proc')/str(c.pid);d=dict(pid=c.pid,start_ticks=p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19],uid=p.stat().st_uid,exe=str(p.joinpath('exe').resolve()),log=str(P/'supervisor.log'))
 (P/'launch.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
