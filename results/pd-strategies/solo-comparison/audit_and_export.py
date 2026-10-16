"""Verify the completed isolated replay and export its raw-backed observations."""
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

from inferx.pd_baselines import validate_sglang_runtime
from inferx.service_ownership import process_identity
from scripts.analyze_pd_baselines import export
from scripts import run_pd_baselines as controller
from scripts import run_concurrency_campaign as previous


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


root = Path(os.environ['INFERX_ROOT'])
run = sys.argv[1]
raw = root / 'pd-baselines/raw' / run
destination = root / 'pd-strategies' / run
campaign = read(raw / 'campaign.json')
assert campaign['status'] == 'restored' and campaign['restored']
assert len(campaign['cases']) == 9 and all(s['state'] == 'measured' for s in campaign['cases'])
scope = read(destination / 'scope.json')
assert all(sha(Path(p)) == expected for p, expected in scope['source_sha256'].items())
cases = {c['id']: c for c in read(raw / 'config.json')}
checks = []
source_hashes = {}
for state in campaign['cases']:
    case, group = state['case'], state['group']
    directory = raw / f'{case}-r0-{group}'
    startup = read(directory / 'startup.json')
    assert startup['case'] == cases[case]
    validate_sglang_runtime(cases[case], startup['server_info'])
    specifications = read(directory / group / 'specifications.json')
    records = [json.loads(line) for line in (directory / group / 'requests.jsonl').read_text().splitlines()]
    assert len(records) == len(specifications) == (24 if group == 'short-control' else 1)
    assert all(b['dispatched_s'] >= a['finished_s'] for a, b in zip(records, records[1:]))
    assert all(s.get('producer') == 0 and 'offset_s' not in s for s in specifications)
    assert all(r['summary']['ok'] and r.get('functional_ok') is True and
               not r['summary'].get('truncated') for r in records)
    source_run = ('pd-strategies-pdmux-private-main-v1' if case == 'sglang-pdmux-private-best'
                  else 'pd-strategies-matched-formal-r1')
    source_path = root / 'pd-baselines/raw' / source_run / f'{case}-r0-{group}' / group / 'specifications.json'
    frozen = read(source_path)
    if group != 'short-control':
        frozen = [r for r in frozen if r['prompt_id'] != 'tool-24576']
    assert len(frozen) == len(specifications)
    assert all(s['payload'] == f['payload'] and s['expected_tool'] == f['expected_tool']
               for s, f in zip(specifications, frozen))
    if group == 'resume-high':
        assert records[0]['summary']['cached_tokens'] == 24576
        assert records[0]['summary']['prompt_tokens'] == 131049
    first_admissions = []
    for line in (directory / 'server.log').read_text(errors='replace').splitlines():
        if 'PD_ADMISSION ' in line:
            event = json.loads(line.split('PD_ADMISSION ', 1)[1])
            if (event.get('method') == 'new' and event.get('admitted') and
                    event.get('remaining_input', 0) > 100000):
                first_admissions.append(event)
    checks.append(dict(case=case, group=group, n=len(records), no_client_overlap=True,
                       exact_frozen_payloads=True, runtime_validation_passed=True,
                       cached_tokens=[r['summary'].get('cached_tokens') for r in records],
                       native_long_admissions=first_admissions))
    for path in [directory / 'startup.json', directory / 'server.log',
                 directory / group / 'specifications.json', directory / group / 'requests.jsonl']:
        source_hashes[str(path)] = sha(path)

saved = read(raw / 'original-service.json')
restored = read(root / 'tuning/best-service.json')
assert restored == read(root / 'context-agents/runtime.json')
identity = process_identity(restored['pid'], previous.BINARY)
assert all(restored.get(k) == v for k, v in identity.items())
argv = Path(f'/proc/{restored["pid"]}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
assert argv == saved['original_argv']
processes = previous.gpu_processes()
gpu0 = 'GPU-a66b08e8-8b65-0faf-22e2-b951a97b2b03'
assert not any(uuid == gpu0 for _, uuid in processes)
assert {pid for pid, uuid in processes if uuid == previous.GPU} == {restored['pid']}
health = controller.get('/health', 'http://127.0.0.1:18081')
assert health['status'] == 'ok'
models = controller.get('/v1/models', 'http://127.0.0.1:18081')
assert models['data'][0]['context_window'] == 262144
metrics = controller.get('/metrics', 'http://127.0.0.1:18081')
assert metrics['llamacpp:requests_processing'] == metrics['llamacpp:requests_deferred'] == 0
service = dict(identity=identity, exact_original_argv=True, manifests_equal=True,
               health=health, context_window=262144, gpu0_compute_processes=[],
               processing=0, deferred=0)
(destination / 'audit.json').write_text(json.dumps(dict(groups=checks,
    source_sha256=source_hashes, service=service), indent=2) + '\n')
index = destination / 'raw-index'
index.mkdir(exist_ok=False)
(index / run).symlink_to(raw, target_is_directory=True)
export(index, destination / 'derived')
summary = (destination / 'derived/summary.json').read_bytes()
(destination / 'solo-summary.json.gz').write_bytes(gzip.compress(summary, mtime=0))
print(json.dumps(dict(groups=len(checks), requests=sum(c['n'] for c in checks),
                      service=service, destination=str(destination))))
