"""Read-only current identity/service/foreign-work verification; no GPU work."""
import json,os,time
from pathlib import Path
from inferx.service_ownership import process_identity
from scripts import run_concurrency_campaign as previous
r=Path(os.environ['INFERX_ROOT']);saved=json.loads((r/'tuning/best-service.json').read_text())
assert saved==json.loads((r/'context-agents/runtime.json').read_text())
identity=process_identity(saved['pid'],previous.BINARY)
assert all(saved.get(k)==v for k,v in identity.items())
argv=Path(f'/proc/{saved["pid"]}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
original=json.loads((r/'sota-validation/raw/profile-copy-v2/original-service.json').read_text())
assert argv==original['original_argv']
previous.final_shutdown_check(saved,gpu0_state='stopped')
followups=json.loads((r/'sota-validation/followups-copy-v2.json').read_text());assert followups['status']=='complete'
report=dict(verified_unix=time.time(),identity=identity,manifests_equal=True,original_argv_equal=True,
 health=previous.get('/health','http://127.0.0.1:18081'),
 restored_configuration=dict(max_context=262144,max_concurrency=1,draft_tokens=7,kv_dtype='rk4v4-e8'),
 gpu0_stopped_and_original_gpu1_idle=True,followups=followups)
assert report['health']['status']=='ok'
(r/'sota-validation/final-service-verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
