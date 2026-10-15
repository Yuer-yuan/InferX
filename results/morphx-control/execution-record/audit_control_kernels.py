"""Strictly retain actual completed kernel gates and immutable source evidence."""
import hashlib,json,os,re
from pathlib import Path
r=Path(os.environ['INFERX_ROOT']);p=r/'morphx-control/raw/kernels-v1';metadata=json.loads((p/'campaign.json').read_text())
assert metadata['status']=='restored' and metadata['restored'] and len(metadata['cases'])==20
sources={};rows=[]
def data(path):
 raw=path.read_bytes();sources[str(path.relative_to(r))]=hashlib.sha256(raw).hexdigest();return raw
for case in metadata['cases']:
 assert case['status']=='passed' and case['returncode']==0
 name=case['variant']+'-'+case['test'];result=json.loads(data(p/(name+'-artifacts/result.json')));log=data(p/(name+'.log')).decode()
 assert result['status']=='passed' and 'phase=patched' in log
 if case['test'].startswith('vector'):
  assert result['n']==1048613 and result['entry']+'_pk' in result['ptx_entries'];checks=4;max_abs=0
 elif case['test'].startswith('marlin'):
  assert len(result['cases'])==5 and all(x['status']=='passed' for x in result['cases']);checks=5;max_abs=max(x['max_abs'] for x in result['cases'])
 else:
  assert len(result['streams'])==2 and len(set(result['streams']))==2 and len(result['checks'])==48;checks=48;max_abs=max(x['max_abs'] for x in result['checks'])
 controls=re.findall(r'\[MorphX-control\].*new=([1-9][0-9]*):([0-9]+)',log)
 rows.append(dict(variant=case['variant'],test=case['test'],checks=checks,max_abs=max_abs,nonzero_min_sm_bound_writes=len(controls),library_sha256=case['library_sha256']))
data(p/'campaign.json');data(p/'controller.py');data(p/'restored-service.json')
for name in ['concurrent_marlin_probe.py','run_control_kernel_qualification.py']:data(r/'morphx-control'/name)
report=dict(scope='Actual20GPUjobs,228numericalchecks vs independent vector formula/originalNVCC including4x48two-stream checks. Bounds reflect real runtime writes, not measured SM occupancy or serving efficacy.',
 restored=True,rows=rows,source_sha256=sources)
assert sum(x['checks'] for x in rows)==264
report['scope']=report['scope'].replace('228numericalchecks','264numericalchecks')
(r/'morphx-control/kernel-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(jobs=len(rows),checks=sum(x['checks'] for x in rows),max_abs=max(x['max_abs'] for x in rows))))
