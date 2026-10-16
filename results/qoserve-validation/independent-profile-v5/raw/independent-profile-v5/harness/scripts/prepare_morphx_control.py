"""Build a fresh source-checked transport variant on the existing server HDD."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from inferx.morphx_control import BASE_LIBRARY_SHA256,prepare_control_runtime


def main():
    parser=argparse.ArgumentParser()
    for name in ['source','destination','library']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device-fill',action='store_true');parser.add_argument('--event-completion',action='store_true')
    args=parser.parse_args();source=args.source.resolve();destination=args.destination.resolve();library=args.library.resolve()
    if library.exists():raise FileExistsError(library)
    original=json.loads((source/'manifest.json').read_text());old_command=original['command']
    old_library=Path(old_command[old_command.index('-o')+1])
    if hashlib.sha256(old_library.read_bytes()).hexdigest()!=BASE_LIBRARY_SHA256:
        raise ValueError('actual base library hash mismatch')
    report=prepare_control_runtime(source,destination,device_fill=args.device_fill,event_completion=args.event_completion)
    library.parent.mkdir(parents=True,exist_ok=True)
    command=[x.replace(str(source),str(destination)) for x in old_command]
    command[command.index('-o')+1]=str(library)
    if command[:2]==['taskset','-c']:command[2]='40-43,96-99'
    report.update(command=command,library=str(library),status='building')
    try:
        with (destination/'build.log').open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        report.update(status='built',library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
    finally:
        report['source_sha256']={str(p.relative_to(destination)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(destination.rglob('*')) if p.is_file()}
        code=Path(__file__).resolve().parents[1]
        report['adapter_source_sha256']={name:hashlib.sha256((code/name).read_bytes()).hexdigest() for name in ['inferx/morphx_validation.py','inferx/morphx_control.py','scripts/morphx_control_ops.hpp','scripts/prepare_morphx_control.py']}
        (destination/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],library=str(library),sha256=report['library_sha256'])))


if __name__=='__main__':main()
