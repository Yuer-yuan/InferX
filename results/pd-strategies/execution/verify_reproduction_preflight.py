"""Invoke the final shutdown guard read-only; do not signal or replace anything."""
import hashlib,json,time
from pathlib import Path
from inferx.service_ownership import process_identity
from scripts import run_concurrency_campaign as controller
r=Path('/home/guokc/mypro/gnn-mount-sda1/InferX')
a=json.loads((r/'tuning/best-service.json').read_text())
assert a==json.loads((r/'context-agents/runtime.json').read_text())
identity=process_identity(a['pid'],controller.BINARY)
assert all(a.get(k)==v for k,v in identity.items())
controller.final_shutdown_check(a,gpu0_state='stopped')
report=dict(verified_unix=time.time(),read_only=True,no_signal_sent=True,gpu0_state='stopped',gpu1_identity=identity,shutdown_guard='passed',source_sha256={name:hashlib.sha256((r/'code-pd-baselines'/name).read_bytes()).hexdigest() for name in ['scripts/run_concurrency_campaign.py','scripts/run_pd_baselines.py']})
(r/'pd-strategies/final-reproduction-preflight.json').write_text(json.dumps(report,indent=2)+'\n')
print('actual stopped-GPU0/owned-idle-GPU1 guard passed without sending signals')
