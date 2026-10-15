import hashlib,json,os
from pathlib import Path
base=Path(os.environ['INFERX_ROOT'])/'pd-baselines'
s=json.loads((base/'derived-final-v1-a/summary.json').read_text())
run='sglang-singleton-lowrate-main-v5'
groups=[g for g in s['groups'] if g['run']==run]
assert len(groups)==9 and all(g['comparison_eligible'] for g in groups)
assert sum(g['summary']['tool_valid'] for g in groups)==222
assert sum(g['full_long_completed'] for g in groups)==6
record=dict(scope='Completed SGLang three-repetition measurement, not an isolated cross-engine scheduler ranking.',campaigns=[c for c in s['campaigns'] if c['run']==run],groups=groups,repeats=[r for r in s['repeats'] if r['case']==groups[0]['case']],source_hashes={k:v for k,v in s['source_hashes'].items() if k.startswith(run+'/')})
(base/'sglang-main-v5-summary.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
sg=[g for g in s['groups'] if g['run'].startswith('sglang')]
record=dict(scope='All attempted SGLang serving groups, including startup/capacity/quality failures; CPU PDMux interface probe is a separate artifact.',groups=[{k:g[k] for k in ['identity','case','group','state','error','complete','comparison_eligible','quality_status','observed_responses']} for g in sg],source_hashes={k:v for k,v in s['source_hashes'].items() if k.startswith('sglang')})
(base/'sglang-attempts.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
for group in ['short-control','cold-low','resume-high']:
 gs=[g for g in groups if g['group']==group]
 print(group,[g['short']['latency_s']['p95'] for g in gs], 'longs',[g['long']['latency_s']['maximum'] for g in gs])
print('audit',json.dumps(json.loads((base/'derived-final-v1-a/audit.json').read_text())))
