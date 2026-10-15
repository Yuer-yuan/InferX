import os,json,ast,hashlib,copy,difflib
from pathlib import Path
r=Path(os.environ['INFERX_ROOT'])/'ebplus-validation'
a=r/'overlays/adapter-v3';b=r/'overlays/adapter-v4'
m=[json.loads((p/'manifest.json').read_text()) for p in [a,b]]
assert m[0]['extracted_method_sha256']==m[1]['extracted_method_sha256']
assert m[0]['native_sha256']==m[1]['native_sha256']
assert (a/'inferx_ebplus_policy.py').read_bytes()==(b/'inferx_ebplus_policy.py').read_bytes()
old=(a/'inferx_ebplus_scheduler.py').read_text();new=(b/'inferx_ebplus_scheduler.py').read_text()
mods=[ast.parse(s) for s in [old,new]]
classes=[next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='EBPlusScheduler') for t in mods]
methods=[{n.name:ast.dump(n,include_attributes=False) for n in c.body if isinstance(n,ast.FunctionDef)} for c in classes]
assert set(methods[0])==set(methods[1])
changed=[k for k in methods[0] if methods[0][k]!=methods[1][k]]
assert set(changed)=={'__init__','_evaluate_mode_switch','_ebplus_dump_evidence'}
for c in classes:
 c.body=[n for n in c.body if not isinstance(n,ast.FunctionDef) or n.name not in changed]
assert ast.dump(mods[0],include_attributes=False)==ast.dump(mods[1],include_attributes=False)
result=dict(status='unchanged_numerical_and_policy_bodies',changed_observation_methods=changed,
 unchanged_bridge_methods=[k for k in methods[0] if k not in changed],
 original_author_methods=m[0]['extracted_method_sha256'],
 policy_sha256=m[0]['files']['inferx_ebplus_policy.py'],native_sha256=m[0]['native_sha256'],
 before_scheduler_sha256=m[0]['files']['inferx_ebplus_scheduler.py'],after_scheduler_sha256=m[1]['files']['inferx_ebplus_scheduler.py'],
 scope='Exact author policy bytes and all allocation/state/phase-dispatch/native-scan bodies unchanged. Three observation methods differ; their read-only additions are separately CPU-regressed and reviewed in the archived diff. New diagnostics still require full-input gates before formal use.')
out=r/'artifacts/observer-comparison-v3-v4.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
(out.with_suffix('.diff')).write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='adapter-v3',tofile='adapter-v4')))
print(json.dumps(dict(status=result['status'],changed=changed,unchanged=len(result['unchanged_bridge_methods']),author_methods=len(result['original_author_methods']))))
