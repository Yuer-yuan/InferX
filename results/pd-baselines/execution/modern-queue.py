"""One owned GPU coordinator; installers are CPU-only and keep separate logs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(os.environ['INFERX_ROOT'])
work = root/'code-pd-baselines'
state_path = root/'pd-baselines/modern-queue.json'
config = root/'pd-baselines/modern-queue-config.json'
state = dict(status='running', stages=[], started=time.time(),
             config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())


def save():
    state_path.write_text(json.dumps(state, indent=2)+'\n')


def launch(command, log, env=None):
    with (root/'pd-baselines'/log).open('x') as output:
        proc=subprocess.Popen(command, cwd=work, env=env, stdout=output,
                              stderr=subprocess.STDOUT)
        state.update(child_pid=proc.pid, command=command)
        save()
        return proc.wait()


def installer_finished(name):
    path=root/'pd-baselines'/name
    while True:
        item=json.loads(path.read_text())
        if item['status']=='terminal':
            return item.get('returncode')==0
        time.sleep(1)


def candidate(case, environment, installer, tag, module, extension):
    state.update(current=tag, phase='waiting_install')
    save()
    if not installer_finished(installer):
        state['stages'].append(dict(candidate=tag, status='install_failed'))
        save()
        return
    python=str(root/f'pd-baselines/{environment}/bin/python')
    env=dict(os.environ, CUDA_VISIBLE_DEVICES='')
    check=('import importlib.metadata as m;import torch;'
           f'import {extension};'
           f'print({{"torch":torch.__version__,"cuda":torch.version.cuda,"engine":m.version("{module}")}})')
    state['phase']='cpu_import'
    save()
    if launch(['taskset','-c','0-3,56-59',python,'-c',check],f'{tag}-import.log',env):
        state['stages'].append(dict(candidate=tag,status='import_failed'))
        save()
        return
    if launch(['taskset','-c','0-3,56-59',python,'-m',
               'vllm.entrypoints.openai.api_server' if module=='vllm' else 'sglang.launch_server',
               '--help'],f'{tag}-help.log',env):
        state['stages'].append(dict(candidate=tag,status='cli_failed'))
        save()
        return
    for suffix, extra in [('128k-probe-v1',['--probe-only']),
                          ('lowrate-diagnostic-v1',['--repeat','1','--groups',
                            'short-control','cold-low','resume-high','--short-rate','.5'])]:
        run=f'{tag}-{suffix}'
        state.update(phase='gpu_campaign',run=run)
        save()
        command=[str(root/'repo-venv/bin/python'),'-m','scripts.run_pd_baselines',
                 '--config',str(config),'--cases',case,'--run-id',run,*extra]
        code=launch(command,f'{run}-controller.log')
        result=json.loads((root/'pd-baselines/raw'/run/'campaign.json').read_text())
        item=dict(candidate=tag,run=run,returncode=code,status=result['status'],
                  restored=result.get('restored'),cases=result['cases'])
        state['stages'].append(item)
        save()
        if code or result['status']!='restored' or not result.get('restored'):
            raise RuntimeError(f'{run}: restoration failed')
        if any(c['state']!='measured' for c in result['cases']):
            return


save()
try:
    prior=json.loads((root/'pd-baselines/queued-mtp.json').read_text())
    assert prior['status']=='terminal' and all(s['restored'] for s in prior['stages'])
    candidate('vllm031-int4-c4-b4096-cap2048-eager-mem95','vllm031-venv',
              'vllm031-install.json','vllm031','vllm','vllm._C_stable_libtorch')
    candidate('sglang0519-int4-c4-ch512-eager-mem95','sglang0519-venv',
              'sglang0519-install.json','sglang0519','sglang','sgl_kernel')
    state.update(status='terminal',finished=time.time())
except BaseException as exc:
    state.update(status='failed',error=f'{type(exc).__name__}: {exc}',finished=time.time())
    raise
finally:
    save()
    print(json.dumps(state),flush=True)
