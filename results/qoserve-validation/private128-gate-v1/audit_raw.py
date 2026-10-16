"""Reconstruct the unscored correctness gate from archived native/SSE records."""
import copy
import hashlib
import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
run = here / 'raw/private128-gate-v1'
sys.path.insert(0, str(run / 'harness'))
from inferx.ebplus_calibration import qualify_tool_rows
from inferx.pd_baselines import load_frozen_workloads

inventory = json.loads((here / 'private128-gate-v1-inventory.json').read_text())
for name, entry in inventory['files'].items():
    data = (here / name).read_bytes()
    assert len(data) == entry['size_bytes'] and hashlib.sha256(data).hexdigest() == entry['sha256']
campaign = json.loads((run / 'campaign.json').read_text())
assert campaign['status'] == 'restored' and campaign['restored']
assert campaign['cases'][0]['state'] == 'passed'
templates, _ = load_frozen_workloads(here.parent / 'independent-native-budgets-v6/frozen-workloads')
engine = run / 'engine'
proofs = {}
for name, prompt, count in (('warm24k', 'tool-24576', 4), ('cold128k', 'cold-131072', 1),
                           ('resume-prime', 'tool-24576', 1), ('resume128k', 'resume-131072', 1)):
    rows = [json.loads(line) for line in (engine / (name + '.jsonl')).read_text().splitlines()]
    proofs[name] = qualify_tool_rows(rows, templates, [prompt] * count)
    assert proofs[name] == json.loads((run / (name + '-strict.json')).read_text())
    if name in ('cold128k', 'resume128k'):
        assert rows[0]['summary']['cached_tokens'] == (0 if name == 'cold128k' else 23520)
quality = [json.loads(line) for line in (engine / 'quality.jsonl').read_text().splitlines()]
expected = copy.deepcopy(templates['code-512']['payload'])
expected['max_tokens'] = 1024
assert len(quality) == 1 and quality[0]['request'] == expected
assert quality[0]['summary']['ok'] and not quality[0]['summary']['truncated']
assert json.loads((engine / 'quality-result.json').read_text())['status'] == 'passed'
geometry = json.loads((engine / 'geometry.json').read_text())
observer = json.loads((engine / 'model-observer.json').read_text())
assert geometry['worker_budget'] == 4096 and geometry['runtime_step_budget'] == 128
assert len(geometry['rows']) == 3269 and len(observer['requests']) == 8
request_ids = set()
for row in geometry['rows']:
    features = row['features']
    assert sum(r['scheduled_tokens'] for r in row['requests']) == features['batch_num_tokens'] <= 128
    request_ids.update(r['request_id'] for r in row['requests'])
assert request_ids == set(observer['requests'])
for request in observer['requests'].values():
    assert request['finish_status'] == 'FINISHED_STOPPED'
    times = request['model_token_times_s']
    assert times and all(b > a for a, b in zip(times, times[1:]))
longs = {}
for length in (131049, 131038):
    ids = {r['request_id'] for row in geometry['rows'] for r in row['requests']
           if r['stage'] == 'prefill' and r['prefill_end'] == length}
    assert len(ids) == 1
    rid = ids.pop()
    rows = [r for row in geometry['rows'] for r in row['requests'] if r['request_id'] == rid and r['stage'] == 'prefill']
    assert max(r['computed_before'] + r['scheduled_tokens'] for r in rows) == length
    start = min(r['computed_before'] for r in rows)
    assert start == (0 if length == 131049 else 23520)
    longs[str(length)] = dict(request_id=rid, first_computed=start,
        final_computed=length, prefill_steps=len(rows))
audit = dict(status='passed', scope='Correctness gate only, no efficacy conclusion.',
    archive_sha256=inventory['archive_sha256'], verified_files=len(inventory['files']),
    code_gate='passed', strict_tools=proofs, full_long_geometry=longs,
    actual_steps=3269, complete_requests=8, maximum_actual_batch_tokens=128)
(here / 'raw-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(audit))
