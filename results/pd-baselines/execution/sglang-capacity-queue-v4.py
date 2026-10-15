import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root=Path(os.environ['INFERX_ROOT'])
config=root/'pd-baselines/sglang-capacity-config-v4.json'
output=root/'pd-baselines/sglang-capacity-queue-v4.json'
state=dict(status='waiting',pid=os.getpid(),stages=[],
           config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())

def save():
    temporary=output.with_suffix('.json.new')
    temporary.write_text(json.dumps(state,indent=2))
    temporary.replace(output)

def run(name,cases,extra):
    command=[str(root/'repo-venv/bin/python'),'-m','scripts.run_pd_baselines',
             '--config',str(config),'--cases',*cases,'--run-id',name,*extra]
    with (root/'pd-baselines'/f'{name}-controller.log').open('x') as log:
        process=subprocess.Popen(command,cwd=root/'code-pd-baselines',stdout=log,
                                 stderr=subprocess.STDOUT)
        state.update(status='running',current=name,child_pid=process.pid)
        save()
        code=process.wait()
    result=json.loads((root/'pd-baselines/raw'/name/'campaign.json').read_text())
    state['stages'].append(dict(run=name,returncode=code,status=result['status'],
        restored=result.get('restored'),cases=result['cases']))
    save()
    assert code==0 and result['status']=='restored' and result.get('restored')
    return result

save()
while True:
    prior=json.loads((root/'pd-baselines/sglang-capacity-queue-v3.json').read_text())
    if prior['status']=='terminal':
        break
    time.sleep(1)
assert all(stage.get('restored') for stage in prior['stages'])
ids=['sglang0519-int4-c4-ch512-eager-mem97-hftext-bf16-lazy12-skiplock',
     'sglang0519-int4-c4-ch512-eager-mem96-hftext-bf16-lazy12-skiplock',
     'sglang0519-int4-c4-ch512-eager-mem95-hftext-bf16-lazy12-skiplock']
name='sglang-capacity-128k-probe-v4'
result=run(name,ids,['--repeat','1','--probe-only'])
valid=[]
for item in result['cases']:
    if item['state']!='measured':
        continue
    path=root/'pd-baselines/raw'/name/(item['case']+'-r0-probe-128k')/'probe-128k/requests.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    q=rows[0]
    if (len(rows)==1 and q['summary']['ok'] and not q['summary'].get('truncated')
            and q.get('functional_ok') and q['summary']['prompt_tokens']>=130000):
        valid.append(item['case'])
state['qualified']=valid
save()
if valid:
    # Largest tested cache budget that passed the full workload gates.
    selected=valid[0]
    state['selected']=selected
    save()
    run('sglang-capacity-lowrate-main-v4',[selected],['--repeat','3','--groups',
        'short-control','cold-low','resume-high','--short-rate','.5'])
state.update(status='terminal',finished=time.time())
save()
print(json.dumps(state),flush=True)
