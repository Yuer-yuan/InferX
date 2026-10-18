#!/usr/bin/env python3
"""Freeze explicit submission materials, source bundles and linked evidence."""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[4]
LINK = re.compile(r'\]\(([^)]+)\)')
BINARY = {'.bin', '.ninfer', '.safetensors', '.pt', '.pth', '.npy', '.npz', '.o', '.a', '.so', '.pyc', '.partial'}
SCAN_SKIP = {'source', 'src', 'build', '.git', '__pycache__', '.venv', 'node_modules', 'weights025', 'weights041', 'weights047'}
TEXT_EXT = {'.md', '.json', '.jsonl', '.log', '.csv', '.tsv', '.txt', '.py', '.sh', '.patch', '.diff', '.yaml', '.yml', '.toml'}
SOURCE = 'c976692d9657408308193b50772c700382a6e243d9da353165205dc5252f76a8'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()

def discover():
    selected, pending, directories, missing = set(), [], set(), []
    def add(path):
        path = path.resolve()
        if not path.is_relative_to(ROOT): raise ValueError(f'Outside repository: {path}')
        if path == ROOT / 'README.md': return  # The archive has its own reviewer-facing README.
        if not path.exists():
            missing.append(str(path.relative_to(ROOT))); return
        if path.is_dir():
            if path in directories: return
            directories.add(path)
            for child in sorted(path.rglob('*')):
                rel = child.relative_to(path)
                if any(x in SCAN_SKIP for x in rel.parts[:-1]): continue
                if child.is_file() and child.suffix in TEXT_EXT:
                    if child.stat().st_size > 64 * 1024 * 1024: raise ValueError(f'Large linked record: {child}')
                    add(child)
            return
        raw_http = path.name.endswith('.raw.bin') and 'http-attempts' in path.parts
        if (path.suffix in BINARY and not raw_http) or path.name.endswith(('.tar.partial', 'caches.tar')):
            raise ValueError(f'Large asset/build output must not be packaged: {path}')
        if path not in selected:
            selected.add(path)
            if path.suffix == '.md': pending.append(path)
    for directory in ['submission/basic', 'submission/track-a', 'submission/organizer', 'submission/tooling/inferx-submission', 'demo/selected100']:
        for path in sorted((ROOT / directory).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and not path.name.startswith('.') and path.suffix not in BINARY:
                add(path)
    for rel in ['submission/README.md', 'submission/DELIVERY.md', 'submission/SNAPSHOT.md', 'submission/records/typst-build.json']:
        add(ROOT / rel)
    manifest = json.loads((ROOT / 'submission/track-a/evidence-manifest.json').read_text())
    for section in ['inputs', 'outputs']:
        for rel in manifest[section]: add(ROOT / rel)
    build = json.loads((ROOT / 'submission/records/typst-build.json').read_text())
    for rel in build['inputs']: add(ROOT / rel)
    for edition in build['editions'].values():
        for rel in edition['previews']: add(ROOT / rel)
    while pending:
        path = pending.pop()
        for target in LINK.findall(path.read_text()):
            target = target.strip('<>')
            if target.startswith(('http:', 'https:', 'mailto:', '#')): continue
            target = unquote(target.split('#', 1)[0])
            if target: add(path.parent / target)
    # Links in historical reports may refer to paths only present on the original host.
    # They are reported during inventory rather than silently rewritten.
    return selected, directories, sorted(set(missing))

VERIFY = '''#!/usr/bin/env python3
import hashlib, json
from pathlib import Path
root = Path(__file__).resolve().parent
manifest = json.loads((root / 'MANIFEST.json').read_text())['files']
actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
assert actual == set(manifest) | {'MANIFEST.json', 'SHA256SUMS'}, 'File set differs from snapshot'
for name, spec in manifest.items():
    p = root / name
    assert not p.is_symlink(), name
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    assert p.stat().st_size == spec['bytes'] and h.hexdigest() == spec['sha256'], name
print(f'PASS: {len(manifest)} payload files match the snapshot')
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default='InferX-TrackA-20261018-r1')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/submission')
    parser.add_argument('--build', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.id): raise ValueError('Invalid archive ID')
    files, directories, missing = discover()
    total = sum(p.stat().st_size for p in files)
    print(json.dumps({'files':len(files),'uncompressed_bytes':total,'missing_local_links':missing,'largest':[(str(p.relative_to(ROOT)),p.stat().st_size) for p in sorted(files,key=lambda p:p.stat().st_size,reverse=True)[:8]]},ensure_ascii=False))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory = {'files':[str(p.relative_to(ROOT)) for p in sorted(files)],'linked_directories':[str(p.relative_to(ROOT)) for p in sorted(directories)],'missing_local_links':missing}
    (args.output_dir / (args.id + '-inventory.json')).write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n')
    if not args.build: return
    if missing: raise ValueError('Resolve missing linked files before packaging')
    stage = args.output_dir / args.id
    archive = args.output_dir / (args.id + '.tar.gz')
    if stage.exists() or archive.exists(): raise FileExistsError('Snapshot already exists; choose a new ID')
    stage.mkdir()
    identities = {}
    for src in sorted(files):
        rel = str(src.relative_to(ROOT)); expected = sha(src)
        dest = stage / rel; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,dest)
        dest.chmod(src.stat().st_mode & 0o777)
        if sha(dest) != expected or sha(src) != expected: raise RuntimeError(f'File changed while freezing: {rel}')
        identities[rel] = {'sha256':expected,'bytes':dest.stat().st_size}
    for directory in directories:
        target=stage/directory.relative_to(ROOT);target.mkdir(parents=True,exist_ok=True)
    readme = '''# InferX · 赛道A提交快照

队名：你说啥都队；成员：陈文欣、张志雄、郭坤昌。

请先运行 `python3 VERIFY.py` 校验归档，再按以下入口阅读：

- [材料总览](submission/README.md)、[中文答辩PDF](submission/basic/presentation/InferX-ZH.pdf)、[英文PDF](submission/basic/presentation/InferX-EN.pdf)。
- [技术报告](submission/track-a/README.md)、[Benchmark](submission/track-a/benchmark.md)及其原始结果。
- [从源码重建量化缓存与评测](submission/track-a/quantization-reproduction.md)：完整源码、固定输入、环境与执行步骤；不要求旧缓存附件。
- [现场C14性能评测](submission/basic/demo.md)。
- [快照和版本说明](submission/SNAPSHOT.md)、[源码包与依赖入口](submission/DELIVERY.md)。

本包保留仓库相对目录，包括必要的results与demo。大型公开模型和FP8输入按固定来源取得，量化缓存从源码重建。各次测量的版本、配置和原始结果分开保存。

代码仓库：https://github.com/Yuer-yuan/InferX 。本次实际提交内容以MANIFEST.json及两个完整源码包为准，Git HEAD不是未提交工作区内容的替代。
'''
    (stage/'README.md').write_text(readme)
    (stage/'VERIFY.py').write_text(VERIFY)
    snapshot={'id':args.id,'created_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'git_head_for_reference':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'snapshot_includes_uncommitted_files':True,'engine_source_aggregate_sha256':SOURCE,'large_weights_and_caches_included':False,'reproduction_entry':'submission/track-a/quantization-reproduction.md','selected_file_count':len(files),'linked_directory_policy':'Referenced raw text records are included; historical build objects and unreferenced source trees are excluded. Complete selected sources are in the immutable source bundles.'}
    (stage/'SNAPSHOT.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
    payload={str(p.relative_to(stage)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(stage.rglob('*')) if p.is_file()}
    (stage/'MANIFEST.json').write_text(json.dumps({'snapshot':args.id,'files':payload},ensure_ascii=False,indent=2)+'\n')
    checks={**payload,'MANIFEST.json':{'sha256':sha(stage/'MANIFEST.json')}}
    (stage/'SHA256SUMS').write_text(''.join(f"{item['sha256']}  {rel}\n" for rel,item in sorted(checks.items())))
    subprocess.run(['python3',str(stage/'VERIFY.py')],check=True)
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        tar.add(stage,arcname=args.id)
    digest=sha(archive)
    archive.with_name(archive.name+'.sha256').write_text(f'{digest}  {archive.name}\n')
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':digest,'payload_files':len(payload)},ensure_ascii=False))

if __name__=='__main__': main()
