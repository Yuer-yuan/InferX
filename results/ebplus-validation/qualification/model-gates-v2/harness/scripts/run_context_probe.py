"""Run a bounded owned server with GPU sampling and independent retrieval checks."""
import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--context', type=int, required=True)
parser.add_argument('--kv', required=True)
args = parser.parse_args()
root = Path(os.environ['INFERX_ROOT'])
directory = root / 'context-agents'
name = f'context-{args.context}-{args.kv}'
log = (directory / (name + '.log')).open('x')
telemetry = (directory / (name + '-gpu.csv')).open('x')
launch = ['bash', str(root / 'code-tuning/scripts/start_tuned_server.sh'),
          'ninfer', '7', args.kv, '2048', str(args.context), '1', 'cold']
manifest = dict(args=launch, started_unix=time.time(), context=args.context, kv=args.kv)
path = directory / (name + '-result.json')
proc = subprocess.Popen(launch, stdout=log, stderr=subprocess.STDOUT,
    env={**os.environ, 'INFERX_REQUEST_LOG': str(directory / (name + '-requests.jsonl'))})
monitor = subprocess.Popen(['nvidia-smi', '-i', '1', '--query-gpu=memory.used,memory.free,power.draw,temperature.gpu',
    '--format=csv,noheader,nounits', '-l', '1'], stdout=telemetry)
try:
    for _ in range(180):
        if proc.poll() is not None:
            raise RuntimeError(f'server startup exited {proc.returncode}')
        try:
            urllib.request.urlopen('http://127.0.0.1:18081/health', timeout=1).read()
            break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError('server startup timeout')
    python = str(root / 'repo-venv/bin/python')
    for corpus, suffix, repeat in [(directory / f'lookup-{args.context}.json', 'lookup', 1),
                                  (root / 'tuning/heldout.json', 'heldout', 3)]:
        output = directory / f'{name}-{suffix}.jsonl'
        subprocess.run([python, '-m', 'scripts.bench_tuning', '--corpus', str(corpus),
            '--output', str(output), '--config', name, '--cache', 'cold', '--repeat', str(repeat)],
            cwd=root / 'code-tuning', check=True)
    lookup = json.loads((directory / f'lookup-{args.context}.json').read_text())[0]
    row = json.loads((directory / f'{name}-lookup.jsonl').read_text().splitlines()[0])
    text = row['summary']['text'].strip()
    if text.startswith('```'):
        text = '\n'.join(text.splitlines()[1:-1])
    manifest['lookup_expected'] = lookup['expected']
    manifest['lookup_answer'] = json.loads(text)
    manifest['lookup_passed'] = manifest['lookup_answer'] == lookup['expected']
    if not manifest['lookup_passed']:
        raise RuntimeError('early/middle/late retrieval failed')
    time.sleep(10)
    telemetry.flush()
    samples = [line.split(',') for line in (directory / (name + '-gpu.csv')).read_text().splitlines()]
    manifest['max_observed_memory_mib'] = max(int(s[0].strip()) for s in samples)
    manifest['min_observed_free_mib'] = min(int(s[1].strip()) for s in samples)
    manifest['headroom_passed'] = manifest['min_observed_free_mib'] >= 1024
    manifest['ok'] = manifest['headroom_passed']
    if not manifest['ok']:
        raise RuntimeError('observed GPU headroom below 1024MiB')
except Exception as e:
    manifest.update(ok=False, error=f'{type(e).__name__}: {e}')
    raise
finally:
    manifest['finished_unix'] = time.time()
    path.write_text(json.dumps(manifest, indent=2))
    for child in [proc, monitor]:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
    log.close()
    telemetry.close()
