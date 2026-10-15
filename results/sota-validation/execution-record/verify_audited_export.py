"""Compare two strengthened exports and retain the exact historical comparison."""
import hashlib,json,os
from pathlib import Path
r=Path(os.environ['INFERX_ROOT'])/'sota-validation';a=(r/'comparison-audited-1.json').read_bytes();b=(r/'comparison-audited-2.json').read_bytes();assert a==b
current=json.loads(a);old=json.loads((r/'comparison-1.json').read_text())
assert current['comparison']==old['comparison']
assert all(current[k]==old[k] for k in ['mixed','solo','diagnostic','frozen_source_sha256'])
proof=dict(sha256=hashlib.sha256(a).hexdigest(),identical=True,outputs=[str(r/f'comparison-audited-{i}.json') for i in [1,2]],
 strict_requests=sum(g['strict_requests'] for k in ['mixed','solo','diagnostic'] for g in current[k]['groups']),
 analysis_source_sha256=current['analysis_source_sha256'],prompt_count_contract=current['prompt_count_contract'],
 historical_sha256=hashlib.sha256((r/'comparison-1.json').read_bytes()).hexdigest(),
 historical_metrics_and_all_raw_campaign_audits_unchanged=True)
assert proof['strict_requests']==452
(r/'comparison-audited-determinism.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
