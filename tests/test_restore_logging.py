"""Losing the experiment client must not invalidate a healthy restoration."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_closed_output_pipe_does_not_retry_or_fail_a_healthy_restoration(tmp_path):
    (tmp_path/'context-agents').mkdir()
    script=r'''
import json,os
from pathlib import Path
from types import SimpleNamespace
from scripts import run_concurrency_campaign as runner
root=Path(os.environ['INFERX_ROOT'])
runner.DIRECTORY=root
calls=[]
def start(*args,**kwargs):
    calls.append(args)
    return SimpleNamespace(pid=os.getpid(),poll=lambda:None)
runner.subprocess.Popen=start
runner.gpu_free=lambda:None
runner.get=lambda path,*args:({'status':'ok'} if path=='/health' else {'data':[{'context_window':262144}]})
runner.process_identity=lambda pid,exe:dict(pid=pid,proc_start_ticks='fixture',uid=os.getuid(),executable=str(exe))
runner.gpu_processes=lambda:{(os.getpid(),runner.GPU)}
runner.time.sleep=lambda seconds:None
restored=runner.restore(dict(original_argv=['fixture-server'],max_context=262144,draft_tokens=7))
(root/'returned.json').write_text(json.dumps(dict(restored=restored,starts=len(calls))))
'''
    env={**os.environ,'INFERX_ROOT':str(tmp_path),'PYTHONPATH':str(Path(__file__).parents[1])}
    child=subprocess.Popen([sys.executable,'-c',script],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    child.stdout.close()  # A real pipe with no reader, rather than a print mock.
    stderr=child.stderr.read().decode();code=child.wait(timeout=10)
    assert code==0,stderr
    data=json.loads((tmp_path/'returned.json').read_text())
    assert data['starts']==1
    assert data['restored']['max_context']==262144 and data['restored']['draft_tokens']==7
