"""Fresh independent holdouts only; frozen coefficients never train here."""
import hashlib
import json
import os
import sys
from pathlib import Path
from scripts import calibrate_qoserve_model as calibration

profile_path = Path(os.environ['INFERX_FROZEN_COST_PROFILE'])
profile_hash = hashlib.sha256(profile_path.read_bytes()).hexdigest()
assert profile_hash == os.environ['INFERX_FROZEN_COST_PROFILE_SHA256']
frozen = json.loads(profile_path.read_text())
assert frozen['freeze_provenance']['external_holdout_rows'] == 0
assert frozen['fit_protocol']['compatibility_variant'] == 'internal-max-residual-intercept-envelope-v1'
definition = json.loads(Path(os.environ['INFERX_REFERENCE_REPAIR_DEFINITION']).read_text())
original_specs = calibration.rollout_specs


def holdout_specs(plan, engine_id):
    calibration.rollout_specs = original_specs
    try:
        declared = calibration.repair_rollout_specs(plan, engine_id, definition)
    finally:
        calibration.rollout_specs = holdout_specs
    result = [s for s in declared if s['kind'] in ('initial-pps', 'holdout')]
    for spec in result:
        spec['rollout_id'] = 'envelope-validation-' + spec['rollout_id']
        spec['long_salt'] = engine_id + '/' + spec['rollout_id'] + '/long'
        spec['decoder_salt'] = engine_id + '/' + spec['rollout_id'] + '/decoder'
        spec['token_seed'] += 1000
    assert len(result) == 29 and sum(s['kind'] == 'holdout' for s in result) == 4
    assert not any(s['kind'] in ('train', 'diagnostic') for s in result)
    return result


output = Path(sys.argv[sys.argv.index('--output') + 1])
calibration.rollout_specs = holdout_specs
try:
    calibration.main()
finally:
    calibration.rollout_specs = original_specs
    if output.exists():
        state = json.loads(output.read_text())
        state.update(matrix_kind='frozen-envelope-independent-holdouts-v1',
            frozen_profile_sha256=profile_hash, training_allowed=False,
            validation_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        output.write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
