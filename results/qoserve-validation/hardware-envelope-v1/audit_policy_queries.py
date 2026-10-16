"""Evaluate the literal source's nominal queries; never fit or relabel GPU work."""
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
repo = here.parents[2]
sys.path.insert(0, str(repo))
from inferx.qoserve_policy import load_author_policy
from inferx.qoserve_profile import CONTEXT_BANDS, validate_step_row

run = here / 'raw/envelope-holdouts-v1'
frozen = json.loads((here / 'frozen-profile.json').read_text())
realized = json.loads((here / 'qualified-profile.json').read_text())
assert realized['coefficients'] == frozen['coefficients']
assert realized['input_evidence']['frozen_profile_sha256'] == hashlib.sha256((here/'frozen-profile.json').read_bytes()).hexdigest()
plan = json.loads((run / 'plan.json').read_text())
assert realized['identity'] == frozen['identity'] == plan['identity']
inventory = json.loads((here / 'envelope-holdouts-v1-inventory.json').read_text())
for name, evidence in inventory['files'].items():
    assert hashlib.sha256((here/name).read_bytes()).hexdigest() == evidence['sha256']
author = here.parent
policy_type = load_author_policy(author/'author-source-v1.tar.gz',
    json.loads((author/'author-source-v1.json').read_text()), frozen['coefficients'])
policy = policy_type()
assert policy.pred_thres == 1.2
state = json.loads((run / 'profile-result.json').read_text())
rollouts = {entry['spec']['rollout_id']: entry['spec'] for entry in state['rollouts']}
records = []
excluded = []
seen = set()
for line in (run / 'steps.jsonl').read_text().splitlines():
    row = json.loads(line)
    validate_step_row(row, plan['identity'])
    assert row['row_id'] not in seen
    seen.add(row['row_id'])
    spec = rollouts[row['rollout_id']]
    if spec['kind'] != 'holdout' or row['warmup']:
        excluded.append(dict(row_id=row['row_id'], reason='PPS or declared warmup'))
        continue
    assert row['phase'] == 'holdout' and row['calibration_kind'] == 'holdout'
    if row['preempted_request_ids']:
        excluded.append(dict(row_id=row['row_id'], reason='Explicit native-preemption calibration stop; full-commitment query unavailable',
            proposed_total=row['proposed_total_tokens'], preempted_request_ids=row['preempted_request_ids']))
        continue
    proposed = row['proposed_total_tokens']
    if proposed not in plan['profile']['candidates']:
        assert proposed in (32, 64, 96)
        excluded.append(dict(row_id=row['row_id'], reason='Calibration-only proposal, absent from deployed author search'))
        continue
    prefill = [r for r in row['requests'] if r['stage'] == 'prefill']
    assert len(prefill) == 1 and prefill[0]['request_id'] == spec['rollout_id'] + '/long/0'
    actual = row['features']
    decoders = actual['batch_num_decode_tokens']
    assert decoders == spec['decoders']
    query = dict(actual, batch_num_tokens=proposed, batch_num_prefill_tokens=proposed-decoders)
    predicted = policy._predict_batch_time(query['batch_decode_context'], query['batch_prefill_context'],
        query['batch_num_prefill_tokens'], query['batch_num_decode_tokens'])
    actual_shape_prediction = policy._predict_batch_time(actual['batch_decode_context'], actual['batch_prefill_context'],
        actual['batch_num_prefill_tokens'], actual['batch_num_decode_tokens'])
    duration = row['timing']['batch_execution_time']
    context = prefill[0]['computed_before']
    band = CONTEXT_BANDS[sum(context >= edge for edge in (4096, 24576, 65536))]
    records.append(dict(row_id=row['row_id'], protocol=row['holdout_protocol'], proposed_total=proposed,
        bin='tokens_0_512' if proposed <= 512 else 'tokens_513_plus', context_band=band,
        actual_features=actual, query_features=query, actual_s=duration, predicted_s=predicted,
        protected=math.isfinite(predicted) and predicted>0 and predicted*1.2>=duration,
        realized_shape_prediction_s=actual_shape_prediction,
        realized_shape_protected=actual_shape_prediction>0 and actual_shape_prediction*1.2>=duration))


def metrics(rows):
    n = len(rows)
    return dict(count=n, coverage=sum(r['protected'] for r in rows)/n if n else 0.,
        context_bands=sorted({r['context_band'] for r in rows}),
        overprediction_above_2x_fraction=sum(r['predicted_s']>2*r['actual_s'] for r in rows)/n if n else 0.,
        failed_row_ids=[r['row_id'] for r in rows if not r['protected']])


reasons = []
for name, info in realized['bins'].items():
    if info['eligible_training_rows'] < 100 or set(info['training_context_bands']) != set(CONTEXT_BANDS):
        reasons.append(name + ' training support incomplete')
if realized['prediction_failures'] or realized['deployment_prediction_failures']:
    reasons.append('nonpositive/nonfinite observed or deployment predictions')
protocols = {}
for protocol in ('original-cycle-v1', 'remainder-cycle-v1'):
    selected = [r for r in records if r['protocol'] == protocol]
    group = dict(bins={}, candidates={})
    protocols[protocol] = group
    for name in ('tokens_0_512', 'tokens_513_plus'):
        info = metrics([r for r in selected if r['bin'] == name])
        group['bins'][name] = info
        if info['count'] < 30 or info['coverage'] < .95:
            reasons.append(protocol + ' bin ' + name + ' count/guard failed')
    for candidate in plan['profile']['candidates']:
        info = metrics([r for r in selected if r['proposed_total'] == candidate])
        group['candidates'][str(candidate)] = info
        if info['count'] < 5 or info['coverage'] < .95:
            reasons.append(protocol + ' candidate ' + str(candidate) + ' count/guard failed')
for name in ('tokens_0_512', 'tokens_513_plus'):
    if set(metrics([r for r in records if r['bin'] == name])['context_bands']) != set(CONTEXT_BANDS):
        reasons.append(name + ' query holdout context support incomplete')
audit = dict(qualified=not reasons, qualification_reasons=reasons,
    scope='Exact extracted author nominal-candidate queries versus real completed steps; no refitting.',
    guard=1.2, minimum_guard_coverage=.95, protocols=protocols, records=records, excluded=excluded,
    coefficients_unchanged_since_freeze=True, realized_shape_qualified=realized['qualified'],
    realized_shape_qualification_reasons=realized['qualification_reasons'],
    source_realized_misses=[r for r in records if not r['realized_shape_protected']],
    driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(here/'policy-query-audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
profile = copy.deepcopy(realized)
profile.update(qualified=audit['qualified'], qualification_reasons=reasons,
    qualification_scope='author nominal candidate queries; realized-shape verdict retained separately',
    realized_shape_qualified=realized['qualified'], realized_shape_qualification_reasons=realized['qualification_reasons'],
    policy_query_audit_sha256=hashlib.sha256((here/'policy-query-audit.json').read_bytes()).hexdigest())
(here/'policy-qualified-profile.json').write_text(json.dumps(profile, indent=2, allow_nan=False)+'\n')
print(json.dumps(dict(qualified=audit['qualified'], reasons=reasons, evaluated=len(records),
    excluded=len(excluded), realized_shape_qualified=realized['qualified'],
    protocol_bins={p:{b:dict(count=m['count'],coverage=m['coverage']) for b,m in g['bins'].items()} for p,g in protocols.items()},
    realized_shape_misses=len(audit['source_realized_misses']))))
