"""Full real-model policy qualification, with automatic original-service restore."""
import asyncio
import copy
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from inferx.concurrency import run_load
from inferx.ebplus_calibration import require_terminal_restoration, qualify_tool_rows
from inferx.pd_baselines import load_frozen_workloads, snapshot_harness
from inferx.qoserve_profile import attach_common_contract
from inferx.service_ownership import process_identity
from inferx.sota_results import verify_prompt_count_contract
from scripts import run_concurrency_campaign as previous
from scripts.run_pd_baselines import OwnedEngine, ENDPOINT, wait_idle, run_with_restoration

root = Path(os.environ['INFERX_ROOT'])
q = root/'qoserve-validation'
code = q/'code-task3-v9'
driver = Path(__file__).resolve().parent
prior = q/'raw/envelope-holdouts-v1'
overlay = q/'overlays/adapter-envelope-v1'
plan = json.loads((code/'configs/qoserve-validation.json').read_text())
profile_path = q/'diagnostic-code-envelope-v1/policy-qualified-profile.json'
profile = json.loads(profile_path.read_text())
assert profile['qualified'] and profile['identity'] == plan['identity']
audit_path = profile_path.with_name('policy-query-audit.json')
assert hashlib.sha256(audit_path.read_bytes()).hexdigest() == profile['policy_query_audit_sha256']
manifest = json.loads((overlay/'manifest.json').read_text())
assert manifest['policy']['coefficients'] == profile['coefficients']
for name, digest in manifest['files'].items():
    assert hashlib.sha256((overlay/name).read_bytes()).hexdigest() == digest
budgets_path = q/'raw/independent-native-budgets-v6/common-budgets.json'
budgets = json.loads(budgets_path.read_text())
assert budgets['identity'] == plan['identity']
saved = json.loads((root/'tuning/best-service.json').read_text())
assert saved == json.loads((root/'context-agents/runtime.json').read_text())
identity = process_identity(saved['pid'], previous.BINARY)
assert all(saved.get(k) == v for k,v in identity.items())
require_terminal_restoration(json.loads((prior/'campaign.json').read_text()),
    json.loads((prior/'restored-service.json').read_text()), saved)
saved['original_argv'] = Path(f"/proc/{saved['pid']}/cmdline").read_bytes().decode().rstrip('\0').split('\0')
assert saved['original_argv'] == json.loads((prior/'original-service.json').read_text())['original_argv']
verify_prompt_count_contract(root/plan['base_case']['model'], backend='vllm031')
templates, _ = load_frozen_workloads(root/'concurrency-roofline')
out = q/'raw/full-policy-gate-v1'
out.mkdir(parents=True, exist_ok=False)
previous.DIRECTORY = out
for name in ('controller.py', 'validation_scheduler.py'):
    (out/name).write_bytes((driver/name).read_bytes())
for source, name in ((profile_path,'qualified-profile.json'), (audit_path,'policy-query-audit.json'),
    (budgets_path,'common-budgets.json'), (overlay/'manifest.json','overlay-manifest.json'),
    (code/'configs/qoserve-validation.json','plan.json')):
    (out/name).write_bytes(source.read_bytes())
(out/'original-service.json').write_text(json.dumps(saved, indent=2)+'\n')
report = dict(status='running', restored=False, controller_pid=os.getpid(), started_unix=time.time(), cases=[],
    scope='Full-policy correctness/order/chunk/relegation qualification only, no efficacy score.',
    harness_sha256=snapshot_harness(code,out/'harness',
        [str(p.relative_to(code)) for d in ('inferx','scripts') for p in sorted((code/d).glob('*.py'))]))


def flush():
    (out/'campaign.json').write_text(json.dumps(report, indent=2)+'\n')


def execute(server, name, prompt, count, *, kind='cold', late=False):
    template = templates[prompt]
    specs = []
    for i in range(count):
        trace = f'policy-qualification/{name}/{i}'
        payload = attach_common_contract(template['payload'],budgets,
            tier=0 if prompt=='tool-24576' else 1, kind=kind, trace_id=trace)
        if late:
            payload['vllm_xargs']['qoserve_deadline_s'] = 1e-9
        specs.append(dict(id=trace, prompt_id=prompt, payload=payload, expected_tool=template['expected_tool']))
    wait_idle()
    with (server.directory/(name+'.jsonl')).open('x') as raw:
        rows = asyncio.run(run_load(ENDPOINT, specs, raw, 'closed',4))
    checked = copy.deepcopy(rows)
    for row in checked:
        extra = row['request']['vllm_xargs']
        assert extra['qoserve_trace_id'] == row['id']
        for tag in ('qoserve_tier','qoserve_deadline_s','qoserve_tbt_s','qoserve_trace_id'):
            extra.pop(tag)
        if 'vllm_xargs' not in template['payload']:
            assert not extra
            row['request'].pop('vllm_xargs')
    proof = qualify_tool_rows(checked,templates,[prompt]*count)
    if name in ('cold128k','resume128k'):
        assert rows[0]['summary']['cached_tokens'] == (0 if name=='cold128k' else 23520)
    (out/(name+'-strict.json')).write_text(json.dumps(proof,indent=2)+'\n')
    return proof


def measure():
    previous.gpu_free()
    directory = out/'engine'
    case = copy.deepcopy(plan['base_case'])
    case.update(id='qoserve-full-policy-functional', async_scheduling=False, shutdown_timeout=30,
        num_gpu_blocks_override=98, scheduler_reserve_full_isl=True,
        scheduler_class='validation_scheduler.PolicyProbeScheduler')
    case['environment'].update(PYTHONPATH=f'{driver}:{overlay}:{code}',
        INFERX_QOSERVE_INITIAL_PPS=str(profile['input_evidence']['fresh_pps']),
        INFERX_VLLM_RUNTIME=str(directory/'native-runtime.json'),
        INFERX_QOSERVE_EVIDENCE=str(directory/'model-observer.json'),
        INFERX_POLICY_VALIDATION=str(directory/'policy-validation.json'))
    entry = dict(case=case,state='starting')
    report['cases'].append(entry)
    flush()
    with OwnedEngine(case,root,directory) as server:
        runtime = json.loads((directory/'native-runtime.json').read_text())
        expected = dict(concurrency=4,max_active=4,chunk=4096,long_prefill_token_threshold=0,
            scheduler_reserve_full_isl=True,async_scheduling=False,enforce_eager=True,
            speculative=False,kv_dtype='fp8',num_gpu_blocks=98,
            scheduler_class='PolicyProbeScheduler',qoserve_policy_enabled=True,
            full_input_commitments=True,profiling_mode='off')
        assert all(runtime.get(k)==v for k,v in expected.items()),runtime
        entry['runtime'] = runtime
        previous.quality_gate(server,templates)
        assert json.loads((directory/'quality-result.json').read_text())['status']=='passed'
        entry['code_gate'] = 'passed'
        flush()
        for name,prompt,count,kind,late in (
            ('warm24k','tool-24576',4,'cold',False),
            ('policy-prime','tool-24576',100,'cold',False),
            ('late-requeue','tool-24576',1,'cold',True),
            ('cold128k','cold-131072',1,'cold',False),
            ('resume-prime','tool-24576',1,'resume',False),
            ('resume128k','resume-131072',1,'resume',False)):
            entry[name] = execute(server,name,prompt,count,kind=kind,late=late)
            flush()
    observer = json.loads((directory/'model-observer.json').read_text())
    validation = json.loads((directory/'policy-validation.json').read_text())
    assert len(observer['requests'])==109
    assert all(r['finish_status']=='FINISHED_STOPPED' for r in observer['requests'].values())
    assert not observer['remaining_views']
    counts = Counter(r['request_id'] for r in validation['requeues'] if r['requeued'])
    assert counts and max(counts.values())==1
    assert all(r['before']==0 and r['after']==1 and r['requeued'] for r in validation['requeues'])
    assert sum(r['trace_id']=='policy-qualification/late-requeue/0' for r in validation['requeues'])==1
    mixed = []
    for row in validation['geometry']:
        stages = [r['stage'] for r in row['requests']]
        assert stages==sorted(stages,key=lambda s:s=='prefill')
        assert not row['preempted_request_ids']
        if row['features']['batch_num_decode_tokens'] and row['features']['batch_num_prefill_tokens']:
            mixed.append(row['step'])
    assert mixed
    choices = sorted({d['allowance'] for d in observer['decisions'] if d['allowance']>0})
    assert len(choices)>1
    entry.update(state='passed',complete_requests=109,mixed_decode_prefill_steps=mixed,
        actual_prefill_allowances=choices,once_requeued_requests=len(counts))
    flush()


flush()
try:
    run_with_restoration(saved,previous.restore,
        [root/'context-agents/runtime.json',root/'tuning/best-service.json'],
        lambda:previous.final_shutdown_check(saved,gpu0_state='stopped'),measure)
    report['status']='restored'
except BaseException as exc:
    report.update(status='failed',error=repr(exc))
    raise
finally:
    report.update(finished_unix=time.time(),restored=(out/'restored-service.json').exists())
    flush()
