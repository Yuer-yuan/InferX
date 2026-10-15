"""Sequential follow-ups: no GPU child before predecessor terminal restoration."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
r=Path(os.environ['INFERX_ROOT']);state_path=r/'sota-validation/followups-copy-v2.json'
assert not state_path.exists()
state=dict(status='waiting_for_formal',pid=os.getpid(),started_unix=time.time(),actions=[])
child=None;cancelled=False
def flush():state_path.write_text(json.dumps(state,indent=2)+'\n')
def stop(number,frame):
    global cancelled
    cancelled=True
    if child is not None and child.returncode is None:
        # This unreaped Popen child belongs to this queue. Signal its controller
        # only, so its own signal-safe finally can finish service restoration.
        child.send_signal(signal.SIGTERM)
for number in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]:signal.signal(number,stop)
flush();deadline=time.monotonic()+7200
try:
    previous=r/'sota-validation/raw/formal-copy-v2/campaign.json'
    while True:
        if cancelled:raise RuntimeError('queue cancelled before GPU follow-up')
        metadata=json.loads(previous.read_text())
        if metadata['status']!='running':break
        if time.monotonic()>deadline:raise RuntimeError('predecessor terminal state timed out')
        time.sleep(2)
    assert metadata['status']=='restored' and metadata['restored']
    assert len(metadata['cases'])==12 and all(c['state']=='measured' for c in metadata['cases'])
    jobs=[
        ('solo-copy-v2','run_sota_diagnostics.py',['--run-id','solo-copy-v2','--cases','compiled-primary-control','morphx-copy96-no-launch-cap','--groups','short-control','cold-high','resume-high','--solo']),
        ('profile-copy-v2','run_sota_profile.py',['--case','morphx-copy96-no-launch-cap','--run-id','profile-copy-v2']),
    ]
    for run,script,args in jobs:
        if cancelled:raise RuntimeError('queue cancelled; no next GPU child')
        state['status']='starting_'+run;entry=dict(run=run,state='running',started_unix=time.time());state['actions'].append(entry);flush()
        command=[sys.executable,'-u',str(r/'sota-validation'/script),*args]
        child=subprocess.Popen(command,cwd=r/'code-pd-baselines',stdin=subprocess.DEVNULL,start_new_session=True)
        entry['controller_pid']=child.pid;flush();print(json.dumps(entry),flush=True)
        while child.poll() is None:time.sleep(.5)
        entry['exit_code']=child.returncode;child=None
        metadata=json.loads((r/'sota-validation/raw'/run/'campaign.json').read_text())
        assert metadata['status']=='restored' and metadata['restored']
        if run.startswith('solo'):
            assert len(metadata['cases'])==6 and all(c['state']=='measured' for c in metadata['cases'])
        else:
            assert not metadata.get('profile_error'),metadata.get('profile_error')
            assert metadata['result']['summary']['errors']==0 and metadata['result']['summary']['tool_valid']==25
        assert entry['exit_code']==0
        entry.update(state='restored',finished_unix=time.time());flush()
    state['status']='complete'
except BaseException as exc:
    state.update(status='failed_or_cancelled',error=repr(exc));raise
finally:
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGTERM)
        while child.poll() is None:time.sleep(.5)
    state['finished_unix']=time.time();flush()
