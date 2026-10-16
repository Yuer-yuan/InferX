"""Freeze internal-only calibration before collecting new external holdouts."""
import hashlib
import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
repo = here.parents[2]
sys.path.insert(0, str(repo))
import inferx.qoserve_profile as fitter

results = here.parent
training = []
archives = {}
for run_id, expected in (('independent-profile-v5', 4129), ('independent-profile-repair-v9', 1207)):
    directory = results / run_id
    run = directory / 'raw' / run_id
    inventory = json.loads((directory / (run_id + '-inventory.json')).read_text())
    for name, entry in inventory['files'].items():
        data = (directory / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry['sha256']
    rows = [json.loads(line) for line in (run / 'steps.jsonl').read_text().splitlines()]
    state = json.loads((run / 'profile-result.json').read_text())
    kinds = {entry['spec']['rollout_id']: entry['spec']['kind'] for entry in state['rollouts']}
    selected = [r for r in rows if kinds[r['rollout_id']] == 'train']
    assert all(r['phase'] == 'train' and r.get('calibration_kind', 'train') == 'train' for r in selected)
    assert len(selected) == expected
    training.extend(selected)
    archives[run_id] = inventory['archive_sha256']
plan = json.loads((results / 'independent-profile-repair-v9/raw/independent-profile-repair-v9/plan.json').read_text())
plan['required_holdout_protocols'] = ['original-cycle-v1', 'remainder-cycle-v1']
profile = fitter.fit_profile(training, [], plan, calibration_envelope=True)
assert not profile['qualified']
assert not profile['prediction_failures'] and not profile['deployment_prediction_failures']
source = Path(fitter.__file__).read_bytes()
profile['freeze_provenance'] = dict(stage='frozen-before-new-external-holdout',
    scope='Compatibility cost profile; not literal meanOLS or SOTA efficacy.',
    archive_sha256=archives, training_rows=len(training), external_holdout_rows=0,
    fitter_sha256=hashlib.sha256(source).hexdigest(),
    freeze_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    training_row_ids_sha256=hashlib.sha256(json.dumps([r['row_id'] for r in training]).encode()).hexdigest())
path = here / 'frozen-profile.json'
with path.open('x') as stream:
    json.dump(profile, stream, indent=2, allow_nan=False)
    stream.write('\n')
(here / 'fitter_source.py').write_bytes(source)
print(json.dumps(dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    corrections={k: v['calibration_envelope']['intercept_correction_s'] for k,v in profile['bins'].items()},
    training_rows=len(training), status='frozen-awaiting-independent-holdout')))
