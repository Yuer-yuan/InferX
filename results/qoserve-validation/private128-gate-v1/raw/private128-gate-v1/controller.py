"""One full-model private-small-chunk correctness gate, protected restoration."""
import asyncio
import copy
import hashlib
import json
import os
import time
from pathlib import Path

from inferx.concurrency import run_load
from inferx.ebplus_calibration import require_terminal_restoration, qualify_tool_rows
from inferx.pd_baselines import load_frozen_workloads, snapshot_harness
from inferx.service_ownership import process_identity
from inferx.sota_results import verify_prompt_count_contract
from scripts import run_concurrency_campaign as previous
from scripts.run_pd_baselines import OwnedEngine, ENDPOINT, wait_idle, run_with_restoration

root = Path(os.environ['INFERX_ROOT'])
q = root / 'qoserve-validation'
code = q / 'code-task3-v9'
overlay = q / 'overlays/adapter-profile-v9-reference'
prior = q / 'raw/shape-transition-cprofile-v1'
driver = Path(__file__).resolve().parent
plan = json.loads((code / 'configs/qoserve-validation.json').read_text())
saved = json.loads((root / 'tuning/best-service.json').read_text())
assert saved == json.loads((root / 'context-agents/runtime.json').read_text())
identity = process_identity(saved['pid'], previous.BINARY)
assert all(saved.get(k) == v for k, v in identity.items())
require_terminal_restoration(json.loads((prior / 'campaign.json').read_text()),
    json.loads((prior / 'restored-service.json').read_text()), saved)
saved['original_argv'] = Path(f"/proc/{saved['pid']}/cmdline").read_bytes().decode().rstrip('\0').split('\0')
assert saved['original_argv'] == json.loads((prior / 'original-service.json').read_text())['original_argv']
manifest = json.loads((overlay / 'manifest.json').read_text())
for name, digest in manifest['files'].items():
    assert hashlib.sha256((overlay / name).read_bytes()).hexdigest() == digest
verify_prompt_count_contract(root / plan['base_case']['model'], backend='vllm031')
templates, _ = load_frozen_workloads(root / 'concurrency-roofline')
out = q / 'raw/private128-gate-v1'
out.mkdir(parents=True, exist_ok=False)
previous.DIRECTORY = out
for name in ('controller.py', 'validation_scheduler.py'):
    (out / name).write_bytes((driver / name).read_bytes())
(out / 'plan.json').write_bytes((code / 'configs/qoserve-validation.json').read_bytes())
(out / 'overlay-manifest.json').write_bytes((overlay / 'manifest.json').read_bytes())
(out / 'original-service.json').write_text(json.dumps(saved, indent=2) + '\n')
report = dict(status='running', restored=False, controller_pid=os.getpid(),
    started_unix=time.time(), cases=[],
    scope='Unscored correctness gate: no priming or performance conclusion.',
    harness_sha256=snapshot_harness(code, out / 'harness',
        [str(p.relative_to(code)) for d in ('inferx', 'scripts') for p in sorted((code/d).glob('*.py'))]))


def write(name, value):
    (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def flush():
    write('campaign.json', report)


def execute(server, name, prompt, count):
    wait_idle()
    template = templates[prompt]
    specs = [dict(id=f'private128/{name}/{i}', prompt_id=prompt,
                  payload=copy.deepcopy(template['payload']), expected_tool=template['expected_tool'])
             for i in range(count)]
    with (server.directory / (name + '.jsonl')).open('x') as raw:
        rows = asyncio.run(run_load(ENDPOINT, specs, raw, 'closed', 4))
    result = qualify_tool_rows(rows, templates, [prompt] * count)
    if name in ('cold128k', 'resume128k'):
        expected_cached = 0 if name == 'cold128k' else 23520
        assert rows[0]['summary']['cached_tokens'] == expected_cached
    write(name + '-strict.json', result)
    return result


def measure():
    previous.gpu_free()
    directory = out / 'engine'
    case = copy.deepcopy(plan['base_case'])
    case.update(id='private128-functional', async_scheduling=False,
        shutdown_timeout=30, num_gpu_blocks_override=98,
        scheduler_reserve_full_isl=False, long_prefill_token_threshold=0,
        scheduler_class='validation_scheduler.Private128Scheduler')
    case['environment'].update(PYTHONPATH=f'{driver}:{overlay}:{code}',
        INFERX_VLLM_RUNTIME=str(directory / 'native-runtime.json'),
        INFERX_QOSERVE_EVIDENCE=str(directory / 'model-observer.json'),
        INFERX_PRIVATE128_GEOMETRY=str(directory / 'geometry.json'))
    entry = dict(case=case, state='starting')
    report['cases'].append(entry)
    flush()
    with OwnedEngine(case, root, directory) as server:
        runtime = json.loads((directory / 'native-runtime.json').read_text())
        expected = dict(concurrency=4, max_active=4, chunk=4096, long_prefill_token_threshold=0,
            scheduler_reserve_full_isl=False, async_scheduling=False, enforce_eager=True,
            speculative=False, kv_dtype='fp8', num_gpu_blocks=98,
            scheduler_class='Private128Scheduler', qoserve_policy_enabled=False, profiling_mode='off')
        assert all(runtime.get(k) == v for k, v in expected.items()), runtime
        previous.quality_gate(server, templates)
        assert json.loads((directory / 'quality-result.json').read_text())['status'] == 'passed'
        entry['code_gate'] = 'passed'
        flush()
        entry['warm'] = execute(server, 'warm24k', 'tool-24576', 4)
        flush()
        entry['cold'] = execute(server, 'cold128k', 'cold-131072', 1)
        flush()
        entry['resume_prime'] = execute(server, 'resume-prime', 'tool-24576', 1)
        flush()
        entry['resume'] = execute(server, 'resume128k', 'resume-131072', 1)
        flush()
    geometry = json.loads((directory / 'geometry.json').read_text())
    observer = json.loads((directory / 'model-observer.json').read_text())
    assert geometry['worker_budget'] == 4096 and geometry['runtime_step_budget'] == 128
    assert geometry['rows'] and max(r['features']['batch_num_tokens'] for r in geometry['rows']) == 128
    assert len(observer['requests']) == 8
    assert all(r['finish_status'] == 'FINISHED_STOPPED' for r in observer['requests'].values())
    for length in (131049, 131038):
        assert any(req['prefill_end'] == length and req['stage'] == 'prefill'
            and req['computed_before'] + req['scheduled_tokens'] == length
            for row in geometry['rows'] for req in row['requests'])
    entry.update(state='passed', completed_requests=8, actual_steps=len(geometry['rows']))
    flush()


flush()
try:
    run_with_restoration(saved, previous.restore,
        [root / 'context-agents/runtime.json', root / 'tuning/best-service.json'],
        lambda: previous.final_shutdown_check(saved, gpu0_state='stopped'), measure)
    report['status'] = 'restored'
except BaseException as exc:
    report.update(status='failed', error=repr(exc))
    raise
finally:
    report.update(finished_unix=time.time(), restored=(out / 'restored-service.json').exists())
    flush()
