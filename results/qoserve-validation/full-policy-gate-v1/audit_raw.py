"""Independently reconstruct tool completion, token joins and policy execution."""
import copy
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

here = Path(__file__).resolve().parent
run = here/'raw/full-policy-gate-v1'
sys.path.insert(0,str(run/'harness'))
from inferx.ebplus_calibration import qualify_tool_rows
from inferx.pd_baselines import load_frozen_workloads
from inferx.qoserve_profile import attach_common_contract
from scripts.calibrate_qoserve_budgets import join_solo_records

inventory = json.loads((here/'full-policy-gate-v1-inventory.json').read_text())
for name,evidence in inventory['files'].items():
    data = (here/name).read_bytes()
    assert len(data)==evidence['size_bytes'] and hashlib.sha256(data).hexdigest()==evidence['sha256']
campaign = json.loads((run/'campaign.json').read_text())
assert campaign['status']=='restored' and campaign['restored'] and campaign['cases'][0]['state']=='passed'
templates,_ = load_frozen_workloads(here.parent/'independent-native-budgets-v6/frozen-workloads')
budgets = json.loads((run/'common-budgets.json').read_text())
engine = run/'engine'
runtime = json.loads((engine/'native-runtime.json').read_text())
assert runtime['scheduler_class']=='PolicyProbeScheduler' and runtime['full_input_commitments']
assert runtime['qoserve_policy_enabled'] and runtime['num_gpu_blocks']==98
assert runtime['chunk']==4096 and runtime['concurrency']==runtime['max_active']==4
assert runtime['kv_dtype']=='fp8' and runtime['enforce_eager'] and not runtime['speculative']
assert not runtime['async_scheduling'] and runtime['long_prefill_token_threshold']==0
observer = json.loads((engine/'model-observer.json').read_text())
assert len(observer['requests'])==109 and not observer['remaining_views']
assert all(r['finish_status']=='FINISHED_STOPPED' for r in observer['requests'].values())
all_rows = []
strict = {}
for name,prompt,count,kind,late in (
    ('warm24k','tool-24576',4,'cold',False), ('policy-prime','tool-24576',100,'cold',False),
    ('late-requeue','tool-24576',1,'cold',True), ('cold128k','cold-131072',1,'cold',False),
    ('resume-prime','tool-24576',1,'resume',False), ('resume128k','resume-131072',1,'resume',False)):
    rows = [json.loads(line) for line in (engine/(name+'.jsonl')).read_text().splitlines()]
    assert len(rows)==count
    checked = copy.deepcopy(rows)
    for row,clean in zip(rows,checked):
        expected = attach_common_contract(templates[prompt]['payload'],budgets,
            tier=0 if prompt=='tool-24576' else 1,kind=kind,trace_id=row['id'])
        if late:
            expected['vllm_xargs']['qoserve_deadline_s']=1e-9
        assert row['request']==expected
        for key in ('qoserve_tier','qoserve_deadline_s','qoserve_tbt_s','qoserve_trace_id'):
            clean['request']['vllm_xargs'].pop(key)
        if 'vllm_xargs' not in templates[prompt]['payload']:
            assert not clean['request']['vllm_xargs']
            clean['request'].pop('vllm_xargs')
        matches=[r for r in observer['requests'].values() if r['trace_id']==row['id']]
        assert len(matches)==1
        assert len(matches[0]['model_token_times_s'])==row['summary']['completion_tokens']
    strict[name] = qualify_tool_rows(checked,templates,[prompt]*count)
    assert strict[name]==json.loads((run/(name+'-strict.json')).read_text())
    if name in ('cold128k','resume128k'):
        assert rows[0]['summary']['cached_tokens']==(0 if name=='cold128k' else 23520)
    all_rows.extend(rows)
assert len(all_rows)==108 and len({r['id'] for r in all_rows})==108
joined = join_solo_records(all_rows,observer)
assert len(joined)==108
quality = [json.loads(line) for line in (engine/'quality.jsonl').read_text().splitlines()]
expected = copy.deepcopy(templates['code-512']['payload'])
expected['max_tokens']=1024
assert len(quality)==1 and quality[0]['request']==expected
assert quality[0]['summary']['ok'] and not quality[0]['summary']['truncated']
assert json.loads((engine/'quality-result.json').read_text())['status']=='passed'
validation = json.loads((engine/'policy-validation.json').read_text())
counts = Counter(r['request_id'] for r in validation['requeues'])
assert counts and max(counts.values())==1
assert all(r['before']==0 and r['after']==1 and r['requeued'] for r in validation['requeues'])
assert sum(r['trace_id']=='policy-qualification/late-requeue/0' for r in validation['requeues'])==1
decisions = {d['step']:d for d in observer['decisions']}
assert len(decisions)==len(observer['decisions'])==len(validation['geometry'])
mixed = 0
choices = set()
for row in validation['geometry']:
    decision = decisions[row['step']]
    assert decision['actual']=={r['request_id']:r['scheduled_tokens'] for r in row['requests']}
    assert sum(decision['actual'].values())<=decision['selected_total']<=4096
    assert not row['preempted_request_ids']
    stages = [r['stage'] for r in row['requests']]
    assert stages==sorted(stages,key=lambda s:s=='prefill')
    f = row['features']
    if f['batch_num_prefill_tokens']:
        assert math.isfinite(decision['predicted_s']) and decision['predicted_s']>0
        choices.add(decision['allowance']+f['batch_num_decode_tokens'])
    mixed += bool(f['batch_num_prefill_tokens'] and f['batch_num_decode_tokens'])
assert mixed==112 and len(choices)>1
assert choices<=set(json.loads((run/'plan.json').read_text())['profile']['candidates'])
for length in (131049,131038):
    requests = [r for row in validation['geometry'] for r in row['requests']
                if r['stage']=='prefill' and r['prefill_end']==length]
    assert requests and len({r['request_id'] for r in requests})==1
    assert min(r['computed_before'] for r in requests)==(0 if length==131049 else 23520)
    assert max(r['computed_before']+r['scheduled_tokens'] for r in requests)==length
audit = dict(status='passed',scope='Full-policy qualification only, no efficacy score.',
    archive_sha256=inventory['archive_sha256'],verified_files=len(inventory['files']),
    complete_requests=109,strict_tools=108,actual_http_model_token_joins=108,
    mixed_decode_prefill_steps=mixed,actual_nominal_total_choices=sorted(choices),
    once_requeued_requests=len(counts),late_probe_once_requeued=True,
    native_preemption_steps=0,code_gate='passed',strict_groups=strict)
(here/'raw-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit))
