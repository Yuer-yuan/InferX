"""Qualify the pre-frozen table using new, independently collected holdouts."""
import hashlib
import importlib.util
import json
import statistics
from pathlib import Path

here = Path(__file__).resolve().parent
run = here / 'raw/envelope-holdouts-v1'
frozen_bytes = (here / 'frozen-profile.json').read_bytes()
frozen_hash = hashlib.sha256(frozen_bytes).hexdigest()
frozen = json.loads(frozen_bytes)
source = here / 'fitter_source.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == frozen['freeze_provenance']['fitter_sha256']
spec = importlib.util.spec_from_file_location('frozen_qoserve_fitter', source)
fitter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fitter)
inventory = json.loads((here / 'envelope-holdouts-v1-inventory.json').read_text())
for name, entry in inventory['files'].items():
    data = (here / name).read_bytes()
    assert len(data) == entry['size_bytes'] and hashlib.sha256(data).hexdigest() == entry['sha256']
campaign = json.loads((run / 'campaign.json').read_text())
state = json.loads((run / 'profile-result.json').read_text())
assert campaign['status'] == 'restored' and campaign['restored']
assert state['status'] == 'complete' and len(state['rollouts']) == 29
assert state['matrix_kind'] == 'frozen-envelope-independent-holdouts-v1' and not state['training_allowed']
assert state['frozen_profile_sha256'] == campaign['frozen_profile_sha256'] == frozen_hash
assert (run / 'frozen-profile.json').read_bytes() == frozen_bytes
plan = json.loads((run / 'plan.json').read_text())
assert state['identity'] == frozen['identity'] == plan['identity']
training = []
for run_id in ('independent-profile-v5', 'independent-profile-repair-v9'):
    previous = here.parent / run_id / 'raw' / run_id
    old_state = json.loads((previous / 'profile-result.json').read_text())
    kinds = {entry['spec']['rollout_id']: entry['spec']['kind'] for entry in old_state['rollouts']}
    training.extend(json.loads(line) for line in (previous / 'steps.jsonl').read_text().splitlines()
                    if kinds[json.loads(line)['rollout_id']] == 'train')
assert len(training) == 5336
rollouts = {entry['spec']['rollout_id']: entry for entry in state['rollouts']}
rows = [json.loads(line) for line in (run / 'steps.jsonl').read_text().splitlines()]
counts = {rid: 0 for rid in rollouts}
held = []
pps = []
for row in rows:
    fitter.validate_step_row(row, plan['identity'])
    entry = rollouts[row['rollout_id']]
    declared = entry['spec']
    counts[row['rollout_id']] += 1
    assert row['cache_salt'] == declared['long_salt'] and row['engine_id'] == state['engine_id']
    assert row['warmup'] == (counts[row['rollout_id']] <= declared['warmup_steps'])
    assert row['calibration_kind'] == declared['kind']
    if declared['kind'] == 'holdout':
        assert row['phase'] == 'holdout' and row['holdout_protocol'] == declared['holdout_protocol']
        held.append(row)
    else:
        assert declared['kind'] == 'initial-pps' and row['features']['batch_num_prefill_tokens'] == 512
        if not row['warmup']:
            pps.append(512 / row['timing']['batch_execution_time'])
assert all(counts[rid] == entry['rows'] and entry['status'] == 'collected' for rid, entry in rollouts.items())
assert len(pps) == 20 and statistics.median(pps) == state['initial_pps']
plan['required_holdout_protocols'] = ['original-cycle-v1', 'remainder-cycle-v1']
result = fitter.fit_profile(training, held, plan, calibration_envelope=True)
assert result['coefficients'] == frozen['coefficients']
assert result['mean_ols_coefficients'] == frozen['mean_ols_coefficients']
for name in result['bins']:
    assert result['bins'][name]['calibration_envelope'] == frozen['bins'][name]['calibration_envelope']
result['input_evidence'] = dict(frozen_profile_sha256=frozen_hash,
    fresh_holdout_archive_sha256=inventory['archive_sha256'],
    original_archives_sha256=frozen['freeze_provenance']['archive_sha256'],
    fitter_sha256=frozen['freeze_provenance']['fitter_sha256'],
    fresh_pps=state['initial_pps'], training_rows=5336, fresh_holdout_rows=len(held),
    holdout_used_for_fit=False,
    compatibility_scope='Exact QoServe policy with disclosed internal-residual cost envelope; not literal meanOLS.')
(here / 'qualified-profile.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
summary = dict(qualified=result['qualified'], reasons=result['qualification_reasons'],
    coefficients_unchanged_since_freeze=True, fresh_pps=state['initial_pps'],
    fresh_holdout_rows=len(held),
    protocols={p: dict(bins={k: dict(count=v['count'], coverage=v['guard_coverage'],
        overprediction_above_2x_fraction=v['overprediction_above_2x_fraction']) for k,v in group['bins'].items()},
        candidates={k: dict(count=v['count'], coverage=v['guard_coverage']) for k,v in group['candidates'].items()})
        for p,group in result['holdout_protocols'].items()})
(here / 'qualification-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary))
