"""One bounded validation of coefficients frozen before any new GPU holdout."""
import hashlib
import json
import os
import time
from pathlib import Path
from inferx.ebplus_calibration import require_terminal_restoration
from inferx.pd_baselines import run_owned_command, snapshot_harness
from inferx.service_ownership import process_identity
from scripts import run_concurrency_campaign as previous
from scripts.run_pd_baselines import run_with_restoration

root = Path(os.environ['INFERX_ROOT'])
q = root / 'qoserve-validation'
code = q / 'code-task3-v9'
driver = Path(__file__).resolve().parent
prior = q / 'raw/private128-gate-v1'
overlay = q / 'overlays/adapter-profile-v9-reference'
plan = json.loads((code / 'configs/qoserve-validation.json').read_text())
definition = json.loads((driver / 'run-definition.json').read_text())
frozen = driver / 'frozen-profile.json'
assert hashlib.sha256(frozen.read_bytes()).hexdigest() == definition['frozen_profile_sha256']
assert hashlib.sha256((driver/'fitter_source.py').read_bytes()).hexdigest() == definition['fitter_sha256']
previous_campaign = json.loads((prior / 'campaign.json').read_text())
assert all(case['state'] == 'passed' for case in previous_campaign['cases'])
saved = json.loads((root / 'tuning/best-service.json').read_text())
assert saved == json.loads((root / 'context-agents/runtime.json').read_text())
identity = process_identity(saved['pid'], previous.BINARY)
assert all(saved.get(k) == v for k,v in identity.items())
require_terminal_restoration(previous_campaign, json.loads((prior / 'restored-service.json').read_text()), saved)
saved['original_argv'] = Path(f"/proc/{saved['pid']}/cmdline").read_bytes().decode().rstrip('\0').split('\0')
assert saved['original_argv'] == json.loads((prior / 'original-service.json').read_text())['original_argv']
out = q / 'raw/envelope-holdouts-v1'
out.mkdir(parents=True, exist_ok=False)
previous.DIRECTORY = out
for name in ('controller.py', 'worker.py', 'frozen-profile.json', 'fitter_source.py', 'run-definition.json'):
    (out / name).write_bytes((driver / name).read_bytes())
(out / 'original-service.json').write_text(json.dumps(saved, indent=2) + '\n')
(out / 'plan.json').write_bytes((code / 'configs/qoserve-validation.json').read_bytes())
(out / 'reference-repair-definition.json').write_bytes((q / 'raw/independent-profile-repair-v9/repair-definition.json').read_bytes())
metadata = dict(status='running', restored=False, controller_pid=os.getpid(),
    started_unix=time.time(), cases=[], frozen_profile_sha256=definition['frozen_profile_sha256'],
    scope='Fresh cost-profile qualification only; no policy efficacy.',
    harness_sha256=snapshot_harness(code, out/'harness',
        [str(p.relative_to(code)) for d in ('inferx', 'scripts') for p in sorted((code/d).glob('*.py'))]))


def flush():
    (out/'campaign.json').write_text(json.dumps(metadata, indent=2) + '\n')


def measure():
    previous.gpu_free()
    command = ['taskset', '-c', '28-35', str(root/plan['base_case']['binary']), '-u', str(driver/'worker.py'),
               '--plan', str(out/'plan.json'), '--output', str(out/'profile-result.json')]
    environment = {**os.environ, **plan['base_case']['environment'],
        'PYTHONPATH': f'{overlay}:{code}', 'CUDA_VISIBLE_DEVICES': plan['identity']['gpu_uuid'],
        'HF_ENDPOINT': 'https://hf-mirror.com', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'TMPDIR': str(root/'tmp'), 'INFERX_VLLM_RUNTIME': str(out/'runtime.json'),
        'INFERX_QOSERVE_EVIDENCE': str(out/'model-observer.json'),
        'INFERX_FROZEN_COST_PROFILE': str(out/'frozen-profile.json'),
        'INFERX_FROZEN_COST_PROFILE_SHA256': definition['frozen_profile_sha256'],
        'INFERX_REFERENCE_REPAIR_DEFINITION': str(out/'reference-repair-definition.json')}
    entry = dict(case='independent-envelope-holdouts', state='starting', command=command)
    metadata['cases'].append(entry)
    flush()
    with (out/'worker.log').open('x') as log:
        outcome = run_owned_command(command, environment, log, timeout=1800)
    if outcome:
        entry.update(state='failed', exit_code=outcome)
        flush()
        raise RuntimeError('fresh holdout collection failed; preserve raw')
    state = json.loads((out/'profile-result.json').read_text())
    assert state['status'] == 'complete' and len(state['rollouts']) == 29
    assert state['frozen_profile_sha256'] == definition['frozen_profile_sha256']
    entry['state'] = 'collected'
    flush()


flush()
try:
    run_with_restoration(saved, previous.restore,
        [root/'context-agents/runtime.json', root/'tuning/best-service.json'],
        lambda: previous.final_shutdown_check(saved, gpu0_state='stopped'), measure)
    metadata['status'] = 'restored'
except BaseException as exc:
    metadata.update(status='failed', error=repr(exc))
    raise
finally:
    metadata.update(finished_unix=time.time(), restored=(out/'restored-service.json').exists())
    flush()
