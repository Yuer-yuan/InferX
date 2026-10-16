"""Run three real clients serially on frozen private repository tasks."""
import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

from inferx.agent_trials import (assess_agent_run, parse_agent_events,
                                run_owned_process, verify_agent_edit)
from inferx.repository import ablated_source, safe_relative, reference_matches

parser = argparse.ArgumentParser()
parser.add_argument('--client', choices=['opencode', 'pi', 'claude', 'all'], default='all')
parser.add_argument('--timeout', type=float, default=600)
args = parser.parse_args()
root = Path(os.environ['INFERX_ROOT'])
directory = root / 'context-agents'
fixture_root = root / 'data/exec-repo-v1/fixtures'
rows = [json.loads(l) for l in (root / 'data/exec-repo-v1/calibration-v5/tasks.jsonl').read_text().splitlines()]
tasks, repos = [], set()
for row in rows:
    if row['split'] == 'dev' and row['repo_name'] not in repos:
        tasks.append(row)
        repos.add(row['repo_name'])
    if len(tasks) == 3:
        break
assert len(tasks) == 3
manifest = []
for task in tasks:
    assert reference_matches(task, fixture_root)
    relative = safe_relative(task['repo_name'], task['file_name'])
    manifest.append(dict(task_id=task['task_id'], repo=task['repo_name'], target=str(relative),
        line=task['prefix_code'].count('\n')+1, expected_counts=task['expected_counts'],
        source_sha256=hashlib.sha256((fixture_root/task['repo_name']/relative).read_bytes()).hexdigest()))
manifest_path = directory / 'tasks-manifest.json'
if manifest_path.exists():
    assert json.loads(manifest_path.read_text()) == manifest, 'frozen task selection changed'
else:
    manifest_path.write_text(json.dumps(manifest, indent=2))

clients = ['opencode', 'pi', 'claude'] if args.client == 'all' else [args.client]
python = str(root / 'repo-venv/bin/python')
wrapper = root / 'code-tuning/scripts/run_local_agent.sh'
request_log = directory / 'agent-server-requests.jsonl'
output = directory / f'trials-{args.client}.jsonl'
# Existing user/account variables are excluded; wrappers add only local dummy credentials.
env = {k:v for k,v in os.environ.items() if k in {'PATH','LANG','HOME','USER','LOGNAME','INFERX_ROOT','TMPDIR'}}
env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONDONTWRITEBYTECODE='1',
           INFERX_AGENT_RESTRICT='1')


def native_requests():
    return [json.loads(l) for l in request_log.read_text().splitlines()
            if l and json.loads(l).get('event') in ['request_start', 'request_done']]


def wait_idle():
    for _ in range(120):
        with urllib.request.urlopen('http://127.0.0.1:18081/metrics', timeout=5) as response:
            metrics = dict(line.split() for line in response.read().decode().splitlines())
        if float(metrics['llamacpp:requests_processing']) == 0 and float(metrics['llamacpp:requests_deferred']) == 0:
            return
        time.sleep(.5)
    raise RuntimeError('server did not drain; refusing to confound next trial')


with output.open('x') as out:
    for task, specification in zip(tasks, manifest):
        for client in clients:
            wait_idle()
            case = directory/'runs'/f"{client}-{task['task_id'].split('/')[-1]}"
            case.mkdir(parents=True, exist_ok=False)
            repo = case/'repo'
            shutil.copytree(fixture_root/task['repo_name'], repo,
                ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache','.git'))
            relative = safe_relative(task['repo_name'], task['file_name'])
            target = repo/relative
            target.write_text(ablated_source(task))
            initial_hash = hashlib.sha256(target.read_bytes()).hexdigest()
            # Independent original tests must actually fail on this ablation.
            baseline = asyncio.run(verify_agent_edit(repo, task, fixture_root=fixture_root,
                python=python, work_root=directory/'verification', cpu=36))
            if baseline['status'] != 'failed':
                raise RuntimeError('ablation failed to produce a covered failing task: '+str(baseline))
            subprocess.run(['git','-c','init.templateDir=','-c','core.hooksPath=/dev/null','init','-q'], cwd=repo, check=True)
            subprocess.run(['git','add','.'],cwd=repo,check=True)
            subprocess.run(['git','-c','user.name=InferX','-c','user.email=inferx@localhost',
                '-c','core.hooksPath=/dev/null','commit','-qm','frozen failing fixture'],cwd=repo,check=True)
            test_command = ('PYTHONPATH=src:. python3 -m unittest discover -s tests -v'
                            if task['test_mode']=='unittest' else
                            'PYTHONPATH=src:. python3 -m pytest -q -p no:cacheprovider tests')
            prompt = (f'The repository has a production code block replaced by pass near line '
                f"{specification['line']} in {relative}. Inspect the relevant source and tests, "
                'restore the intended behavior, and run the complete original tests. '
                f'Use this command: {test_command}. Modify only the named production file, '
                'do not edit tests or configuration. Keep working until tests pass. '
                'Stay in the current repository; no network, package installation, subagents, or commits. '
                'Finish with a concise description of the fix and the test result.')
            (case/'prompt.txt').write_text(prompt)
            command = ['bash', str(wrapper), client]
            if client=='opencode':
                command += ['run','--format','json','--port','18082','--model','inferx/qwen38-27b','--auto','--',prompt]
            elif client=='pi':
                command += ['--print','--mode','json','--no-session','--',prompt]
            else:
                command += ['-p','--output-format','stream-json','--verbose','--no-session-persistence',
                    '--permission-mode','dontAsk','--allowedTools','Read Edit Write Bash',
                    '--tools','Read,Edit,Write,Bash','--',prompt]
            before = max((x['request']['request_id'] for x in native_requests()),default=0)
            trial = dict(client=client, task_id=task['task_id'], repo=task['repo_name'],
                target=str(relative), started_unix=time.time(), baseline=baseline,
                stdout=str(case/'events.jsonl'), stderr=str(case/'stderr.txt'),
                command=command[:-1]+['<prompt.txt>'])
            print('START',client,task['task_id'],flush=True)
            try:
                result=run_owned_process(command,cwd=repo,stdout_path=case/'events.jsonl',
                    stderr_path=case/'stderr.txt',timeout=args.timeout,env=env)
                parsed=parse_agent_events(client,(case/'events.jsonl').read_text())
                verification=asyncio.run(verify_agent_edit(repo,task,fixture_root=fixture_root,
                    python=python,work_root=directory/'verification',cpu=36))
                changed=hashlib.sha256(target.read_bytes()).hexdigest()!=initial_hash
                trial.update(**result,events=parsed,verification=verification,source_changed=changed)
                trial['status']=assess_agent_run(result['returncode'],result['timed_out'],parsed,
                                                verification['status'],changed)
            except Exception as e:
                trial.update(status='infrastructure_error',error=f'{type(e).__name__}: {e}')
            wait_idle()
            observed=[x for x in native_requests() if x['request']['request_id']>before]
            trial['server_requests']=observed
            done=[x for x in observed if x['event']=='request_done']
            trial['server_completion_tokens']=sum(x['result']['completion_tokens'] for x in done)
            trial['server_prompt_tokens']=sum(x['result']['prompt_tokens'] for x in done)
            trial['server_cached_tokens']=sum(x['result']['prefix_cache_hit_tokens'] for x in done)
            trial['server_max_prompt_tokens']=max((x['result']['prompt_tokens'] for x in done),default=0)
            trial['server_request_count']=len(done)
            diff=subprocess.check_output(['git','diff','--no-ext-diff','--binary'],cwd=repo)
            (case/'changes.diff').write_bytes(diff)
            trial['diff_sha256']=hashlib.sha256(diff).hexdigest()
            trial['finished_unix']=time.time()
            (case/'result.json').write_text(json.dumps(trial,indent=2))
            out.write(json.dumps(trial)+'\n');out.flush()
            print('END',client,task['task_id'],trial['status'],round(trial.get('wall_s',0),2),
                  trial.get('events',{}).get('tool_counts'),flush=True)
