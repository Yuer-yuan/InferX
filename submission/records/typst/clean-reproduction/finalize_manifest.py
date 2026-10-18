#!/usr/bin/env python3
"""Refresh only explicitly audited material pins; retain historical measurements."""
import datetime,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
STAGE=Path(__file__).resolve().parent

def dump(v):return json.dumps(v,ensure_ascii=False,indent=2)+'\n'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def spec(p):return dict(sha256=digest(p),bytes=p.stat().st_size)
def main():
    path=ROOT/'submission/track-a/evidence-manifest.json';old=json.loads(path.read_text());manifest=json.loads(path.read_text())
    before=STAGE/'evidence-manifest-before-r2-finalization.json'
    if not before.exists():before.write_text(dump(old))
    authorized=[];base=ROOT/'results/three-node-optimization/ucloud3/selected100-single-node-reproduction-20261018'
    for name in ['perf_candidate.py','accuracy_candidate.py','candidate_common.py']:
        rel='demo/selected100/'+name;p=ROOT/rel
        actual=base/'readiness-repair-delivery/entrypoint-r2'/name
        frozen=base/'reproduction-kit/entrypoint'/name
        assert p.read_bytes()==actual.read_bytes()==frozen.read_bytes(),name
        now=spec(p)
        if manifest['inputs'][rel]!=now:
            authorized.append(dict(path=rel,previous=manifest['inputs'][rel],current=now,reason='Authorized r2 read-only model-readiness repair. Byte-identical to actual tested entrypoint-r2 and immutable v1; original runner unchanged. Original failure/old wrappers retained.',actual_source=str(actual.relative_to(ROOT)),frozen_source=str(frozen.relative_to(ROOT))))
        manifest['inputs'][rel]=now
    handoff=json.loads((STAGE/'manifest-handoff.json').read_text())
    for rel,v in handoff['inputs'].items():
        assert spec(ROOT/rel)==v,rel
        if rel in manifest['inputs']:assert manifest['inputs'][rel]==v,('Unapproved existing source drift',rel)
        manifest['inputs'][rel]=v
    reports=['material-content-audit.json','environment-install-audit.json','audit_materials.py','finalize_manifest.py']
    for name in reports:
        q=STAGE/name;manifest['inputs'][str(q.relative_to(ROOT))]=spec(q)
    for rel in handoff['outputs']:
        manifest['outputs'][rel]=spec(ROOT/rel)
    manifest['outputs']['submission/basic/presentation/clean-reproduction.json']=spec(ROOT/'submission/basic/presentation/clean-reproduction.json')
    changes=STAGE/'authorized-entrypoint-pin-update.json'
    if authorized:
        changes.write_text(dump(dict(status='PASS',updates=authorized,actual_repair_audit='submission/records/typst/clean-reproduction/readiness-audit.json',original_runner_sha256='d80df390cf6d1d50129796b31fdd7f4d1e95d0043d3b942d47ac582118702fb0',historical_measurements_modified=False)))
    if changes.exists():manifest['inputs'][str(changes.relative_to(ROOT))]=spec(changes)
    for key in ['inputs','outputs']:
        for rel,v in manifest[key].items():assert spec(ROOT/rel)==v,(key,rel)
    manifest['scope']='Audited bilingual materials, historical100/105 measurements, independently rebuilt664 with old fixed caches, frozen v1 and portable116 source workflows, and fresh-prefix CPU environment/data-loader validation.116 new-cache original C14 and first100 quality independently completed and audited, including empty/length cases; final submission/archive separately recorded in DELIVERY. No inherited scores or new-environment GPU claim.'
    manifest['updated_at_shanghai']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M +08:00')
    manifest['input_fingerprint']=hashlib.sha256(dump(dict(inputs=manifest['inputs'],tools=manifest['tools'])).encode()).hexdigest()
    path.write_text(dump(manifest))
    print('Manifest refreshed:',len(manifest['inputs']),'inputs,',len(manifest['outputs']),'outputs;',len(authorized),'documented r2 source-pin changes; historical metrics unchanged.')
if __name__=='__main__':main()
