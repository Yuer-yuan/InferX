"""Exercise the corrected guard on actual historical inputs; never build/load GPU."""
import hashlib,json,os
from pathlib import Path
from inferx.morphx_control import prepare_control_runtime
r=Path(os.environ['INFERX_ROOT']);root=r/'morphx-control';dest=root/'preparation-review-guard-v1'
result=prepare_control_runtime(r/'sota-validation/runtime-cuda-copy-v2',dest,device_fill=True,event_completion=True)
old=root/'runtime-both-v1';old_manifest=json.loads((old/'manifest.json').read_text())
for name,h in result['source_sha256'].items():
 assert hashlib.sha256((old/name).read_bytes()).hexdigest()==h,name
report=dict(source='sota-validation/runtime-cuda-copy-v2',new_preparation=str(dest.relative_to(r)),unchanged_comparison_tree='morphx-control/runtime-both-v1',source_files_equal_to_measured_build=len(result['source_sha256']),prepared_source_sha256=result['source_sha256'],scope='Corrected complete-inventory guard accepts actual pinned inputs and produces byte-identical source files. No existing source/binary/raw/capsule changed; no GPU/build work.')
(root/'review-preparation-verification.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='prepared_source_sha256'}))
