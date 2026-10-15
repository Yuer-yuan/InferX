"""Keep the real preflight failure separate from GPU/model outcomes."""
import hashlib,json,os,time
from pathlib import Path
r=Path(os.environ['INFERX_ROOT']);p=r/'morphx-control/raw/models-v1';original=json.loads((p/'original-service.json').read_text())
report=dict(status='preflight_failed',recorded_unix=time.time(),failure='FileNotFoundError',
 missing_path=str(r/'code-pd-baselines/configs/sota-validation.json'),
 phase='Before metadata flush and RestorableService invocation; no model engine or performance sample started.',
 original_snapshot_pid=original['pid'],controller_sha256=hashlib.sha256((p/'controller.py').read_bytes()).hexdigest(),
 correction='models-v2 reads the verified frozen HDD sota-validation/configs file before creating run or stopping service; kernel20/20 prerequisites unchanged.')
with (p/'preflight-failure.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report))
