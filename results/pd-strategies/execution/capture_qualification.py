import hashlib,json
from pathlib import Path
root=Path('/home/guokc/mypro/gnn-mount-sda1/InferX');raw=root/'pd-baselines/raw';source={};items=[]
def read(p):
 b=p.read_bytes();source[str(p.relative_to(root))]=hashlib.sha256(b).hexdigest();return json.loads(b)
run=raw/'pd-strategies-pdmux-context-probe-v2';campaign=read(run/'campaign.json');assert campaign['status']=='restored' and campaign['restored'];configs={x['id']:x for x in read(run/'config.json')}
for state in campaign['cases']:
 p=run/f"{state['case']}-r0-probe-128k";start=read(p/'startup.json');quality=read(p/'quality-result.json');log=p/'server.log';source[str(log.relative_to(root))]=hashlib.sha256(log.read_bytes()).hexdigest()
 warm=p/'probe-128k/warmup.jsonl';rows=[json.loads(l) for l in warm.read_text().splitlines()];source[str(warm.relative_to(root))]=hashlib.sha256(warm.read_bytes()).hexdigest()
 rp=p/'probe-128k/result.json';result=read(rp) if rp.exists() else None
 items.append(dict(state=state,configuration=configs[state['case']],internal_states=start['server_info']['internal_states'],context_length=start['server_info']['context_length'],quality=quality,warmup_requests=len(rows),warmup_strict=sum(x['summary']['ok'] and x.get('functional_ok') is True and not x['summary'].get('truncated') for x in rows),result=result))
probe=read(raw/'pd-strategies-green-context-numerical-v2/probe-result.json')
out=root/'pd-strategies/compatibility/green-context-model-qualification.json';out.write_text(json.dumps(dict(scope='Full model code, four24K warmup and one complete128K forced tool; not general model accuracy or PD interference efficacy. Exact resources fixed across nobind/bound A/B.',cases=items,numerical_probe=probe,source_sha256=source),indent=2)+'\n');print(out)
