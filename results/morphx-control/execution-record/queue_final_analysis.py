"""After all timing, strictly export twice; no serving changes or GPU work."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

r = Path(os.environ['INFERX_ROOT'])
root = r / 'morphx-control'
path = root / 'analysis-followup-v1.json'
assert not path.exists()
metadata = dict(status='waiting_for_timed_campaigns', pid=os.getpid(), started_unix=time.time())
stopping = False
child = None


def flush():
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(metadata, indent=2) + '\n')
    tmp.replace(path)


def stop(number, frame):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGTERM)


for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(number, stop)


def run(command, log):
    global child
    if stopping:
        raise InterruptedError('analysis stopped before action')
    with log.open('x') as handle:
        child = subprocess.Popen(command, env={**os.environ, 'PYTHONPATH': str(r / 'code-pd-baselines')},
                                 cwd=r / 'code-pd-baselines', stdin=subprocess.DEVNULL,
                                 stdout=handle, stderr=subprocess.STDOUT)
        while True:
            try:
                code = child.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                pass
    child = None
    if code != 0:
        raise RuntimeError('analysis failed: ' + str(log))
    if stopping:
        raise InterruptedError('analysis stopped after own child cleanup')


flush()
try:
    while True:
        if stopping:
            raise InterruptedError('stopped without signalling prerequisites')
        try:
            queue = json.loads((root / 'formal-followups-v1.json').read_text())
            cpu = json.loads((r / 'ebplus-validation/cpu-followup-v1.json').read_text())
        except json.JSONDecodeError:
            time.sleep(5)
            continue
        if queue['status'] == 'complete' and cpu['status'] in ('complete', 'failed', 'stopped'):
            break
        if queue['status'] not in ('running', 'complete'):
            raise RuntimeError('timed prerequisite failed')
        for prerequisite, expected in [(queue, b'queue_policy_and_formal.py'),
                                       (cpu, b'queue_cpu_gate.py')]:
            if prerequisite['status'] in ('running', 'waiting_for_timed_campaigns', 'running_cpu_gates'):
                cmdline = Path(f"/proc/{prerequisite['pid']}/cmdline").read_bytes()
                if expected not in cmdline:
                    raise RuntimeError('prerequisite process identity changed')
        time.sleep(5)
    metadata.update(status='exporting', cpu_gate_status=cpu['status'])
    flush()
    for name, count in [('formal-local-v1', 12), ('solo-local-v1', 6)]:
        campaign = json.loads((root / 'raw' / name / 'campaign.json').read_text())
        assert campaign['status'] == 'restored' and campaign['restored']
        assert len(campaign['cases']) == count and all(x['state'] == 'measured' for x in campaign['cases'])
        assert campaign['client_cpu_affinity'] == [40, 41, 42, 43, 96, 97, 98, 99]
    for number in (1, 2):
        target = root / f'comparison-{number}.json'
        assert not target.exists()
        run([str(r / 'repo-venv/bin/python'), '-m', 'scripts.analyze_sota_validation',
             '--frozen', str(r / 'concurrency-roofline'),
             '--mixed', str(root / 'raw/formal-local-v1'),
             '--solo', str(root / 'raw/solo-local-v1'), '--output', str(target)],
            root / f'comparison-{number}.log')
    hashes = [hashlib.sha256((root / f'comparison-{n}.json').read_bytes()).hexdigest() for n in (1, 2)]
    assert hashes[0] == hashes[1]
    result = json.loads((root / 'comparison-1.json').read_text())
    placement = []
    for key, run_id, count in [('mixed', 'formal-local-v1', 12), ('solo', 'solo-local-v1', 6)]:
        export = result[key]
        assert len(export['groups']) == count
        run_root = root / 'raw' / run_id
        configurations = {c['id']: c for c in json.loads((run_root / 'config.json').read_text())}
        for wave in export['groups']:
            folder = run_root / wave['identity']
            startup = json.loads((folder / 'startup.json').read_text())
            assert startup['args'][:3] == ['taskset', '-c', '28-35']
            case = configurations[wave['case']]
            if not case['morphx']['enabled']:
                continue
            observed = re.findall(r'\[MorphX-affinity\] requested=(\d+) observed=(\d+)',
                                  (folder / 'server.log').read_text())
            assert observed and set(observed) == {('36', '36')}
            assert case['environment']['SMSCHED_AGENT_CPU'] == '36'
            library = Path(case['environment']['NEUTRINO_HOOK_DRIVER'])
            assert hashlib.sha256(library.read_bytes()).hexdigest() == case['morphx']['library_sha256']
            placement.append(dict(kind=key, identity=wave['identity'], observed_cpu=36,
                                  library_sha256=case['morphx']['library_sha256']))
    metadata.update(status='complete', export_sha256=hashes[0], placement=placement,
                    strict_mixed=sum(w['strict_requests'] for w in result['mixed']['groups']),
                    strict_solo=sum(w['strict_requests'] for w in result['solo']['groups']),
                    comparison=result['comparison'])
except BaseException as exc:
    metadata.update(status='stopped' if stopping else 'failed', error=repr(exc))
    raise
finally:
    metadata['finished_unix'] = time.time()
    flush()
