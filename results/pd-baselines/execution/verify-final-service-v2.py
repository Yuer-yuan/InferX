"""One-off terminal-data verification; never stops or starts a serving process."""
from pathlib import Path
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
import urllib.request

from inferx.context_agents import verify_process_identity

root = Path(os.environ['INFERX_ROOT'])
base = root/'pd-baselines'
state_path = base/'verify-final-service-v2.json'
state = dict(status='waiting', pid=os.getpid())

def save():
    tmp = state_path.with_suffix('.new')
    tmp.write_text(json.dumps(state, indent=2)+'\n')
    tmp.replace(state_path)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def tree(path):
    result = {}
    for p in sorted(path.rglob('*')):
        assert p.lstat().st_uid == os.getuid(), p
        assert not p.is_symlink(), p
        if p.is_file():
            result[str(p.relative_to(path))] = dict(bytes=p.stat().st_size, sha256=digest(p))
    return result

def get(endpoint, route):
    with urllib.request.urlopen(endpoint+route, timeout=10) as response:
        data = response.read().decode()
        return data if route == '/metrics' else json.loads(data)

def post(route, payload):
    request = urllib.request.Request('http://127.0.0.1:18081'+route,
        json.dumps(payload).encode(), {'Content-Type':'application/json'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)

a = json.loads((root/'tuning/best-service.json').read_text())
b = json.loads((root/'context-agents/runtime.json').read_text())
assert a == b
assert verify_process_identity(a['pid'],a['proc_start_ticks'],Path(a['executable']))
argv = Path(f"/proc/{a['pid']}/cmdline").read_bytes().decode().rstrip('\0').split('\0')
assert argv == shlex.split(a['command']), (argv,a['command'])
for flag, value in [('--max-context','262144'),('--draft-tokens','7'),
                    ('--max-concurrency','1'),('--kv-dtype','rk4v4-e8')]:
    assert argv[argv.index(flag)+1] == value
assert '--lm-head-draft' in argv
gpu = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                              '--format=csv,noheader'], text=True)
observed = {(line.split(',')[0].strip(),int(line.split(',')[1]))
            for line in gpu.splitlines() if line.strip()}
expected = {('GPU-a66b08e8-8b65-0faf-22e2-b951a97b2b03',2504785),
            ('GPU-4d9bef9d-631c-bb36-3ff4-589d6a184189',a['pid'])}
assert observed == expected, observed
for p in Path('/proc').glob('[0-9]*'):
    try:
        if p.stat().st_uid == os.getuid():
            executable = str((p/'exe').resolve(strict=True))
            assert '/pd-baselines/sglang0519-venv/' not in executable
            assert '/pd-baselines/vllm031-venv/' not in executable
    except (FileNotFoundError,PermissionError,ProcessLookupError):
        pass


# Strict OpenAI forced function call plus existing Anthropic tool-result roundtrip.
endpoint = 'http://127.0.0.1:18081'
deadline = time.monotonic()+120
while True:
    metrics = dict(line.split() for line in get(endpoint,'/metrics').splitlines())
    if all(float(metrics[k]) == 0 for k in ['llamacpp:requests_processing','llamacpp:requests_deferred']):
        break
    assert time.monotonic() < deadline, 'original service remains busy'
    time.sleep(.2)
schema = dict(type='object',properties={'a':{'type':'integer'},'b':{'type':'integer'}},
              required=['a','b'],additionalProperties=False)
response = post('/v1/chat/completions',dict(model='qwen38-27b',temperature=0,max_tokens=128,
    messages=[dict(role='user',content='Call add_numbers with a=2 and b=3.')],
    tools=[dict(type='function',function=dict(name='add_numbers',description='Add integers.',parameters=schema))],
    tool_choice=dict(type='function',function=dict(name='add_numbers'))))
choice = response['choices'][0]
calls = choice['message']['tool_calls']
assert choice['finish_reason'] == 'tool_calls' and len(calls) == 1
assert calls[0]['function']['name'] == 'add_numbers'
assert json.loads(calls[0]['function']['arguments']) == {'a':2,'b':3}
(base/'final-openai-tool.json').write_text(json.dumps(response,indent=2)+'\n')
anthropic = base/'final-anthropic-tools.json'
subprocess.run([str(root/'repo-venv/bin/python'),'scripts/check_tuning_tools.py',
                '--endpoint',endpoint,'--output',str(anthropic)],check=True,cwd=root/'code-pd-baselines')
assert json.loads(anthropic.read_text())['passed']
assert a == json.loads((root/'tuning/best-service.json').read_text())
assert a == json.loads((root/'context-agents/runtime.json').read_text())
assert verify_process_identity(a['pid'],a['proc_start_ticks'],Path(a['executable']))
health = {str(port):get(f'http://127.0.0.1:{port}','/health') for port in [18080,18081]}
assert all(v['status'] == 'ok' for v in health.values())
verification = dict(captured_unix=time.time(),passed=True,manifest=a,actual_argv=argv,
    manifests_equal=True,gpu_processes=gpu,health=health,
    openai_forced_tool=True,anthropic_roundtrip=True,
    evidence_sha256={p.name:digest(p) for p in [base/'final-openai-tool.json',anthropic,
                                               base/'sglang-cache-migration.json']})
(base/'final-service-verification.json').write_text(json.dumps(verification,indent=2)+'\n')
state.update(status='terminal',finished_unix=time.time(),service_verified=True)
save()
print(json.dumps(state),flush=True)
