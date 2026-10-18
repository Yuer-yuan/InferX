#!/usr/bin/env python3
"""Build one attachment under 50 MB from a committed, reproducible snapshot."""
import argparse
import hashlib
import json
import re
import subprocess
import tarfile
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[4]
SOURCE_BUNDLE = 'results/three-node-optimization/ucloud2/full-quantization-reproduction-116-portable-source-002.tar.gz'
SOURCE_SHA = '7300c70a5857221537b878046d8c869ce640c788de4773d91d4e523b6bb41f9e'
LINK = re.compile(r'\]\(([^)]+)\)')
LIMIT = 50_000_000


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', required=True)
    parser.add_argument('--id', default='InferX-TrackA-20261018-r2')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/submission')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.id):
        raise ValueError('Invalid attachment ID')
    commit = git('rev-parse', args.commit + '^{commit}').decode().strip()
    entries = {}
    for entry in git('ls-tree', '-r', '-l', '-z', commit).split(b'\0'):
        if not entry:
            continue
        spec, name = entry.split(b'\t', 1)
        mode, kind, oid, size = spec.decode().split()
        if kind == 'blob':
            entries[name.decode()] = {'git_blob': oid, 'bytes': int(size), 'mode': mode}
    prefixes = ['submission/basic/', 'submission/track-a/', 'submission/organizer/',
                'submission/tooling/inferx-submission/', 'demo/selected100/']
    selected = {p for p in entries if any(p.startswith(prefix) for prefix in prefixes)
                and not p.startswith('submission/track-a/data/ledger-snapshots/')}
    selected.update(['README.md', 'submission/README.md', 'submission/DELIVERY.md',
                     'submission/SNAPSHOT.md', SOURCE_BUNDLE,
                     'results/three-node-optimization/ucloud1/ifbench-fixed-dataset-acquisition-CPU/ifbench-fixed-dataset-cache-r1.tar.gz',
                     'results/three-node-optimization/ucloud1/fp8-pinned-acquisition-r2/fp8-pinned-acquisition-r2.tar.gz',
                     'results/three-node-optimization/ucloud2/full-quantization-reproduction-116/fresh239-fit-light-evidence-001.tar.gz',
                     'results/three-node-optimization/ucloud2/full-quantization-reproduction-116/fresh-quality100-raw-and-CPU-001.tar.gz',
                     'results/three-node-optimization/ucloud2/full-quantization-reproduction-116/fresh-C14-raw-001/deployment116/performance-C14/c14/profile_export.jsonl',
                     'results/three-node-optimization/ucloud2/full-quantization-reproduction-116/fresh-C14-raw-001/deployment116/service-perf-c14/requests.jsonl'])
    # Keep the original small configuration/summary records; large raw records stay
    # available in the exact public Git commit, with all Markdown links pinned below.
    evidence = json.loads(git('show', commit + ':submission/track-a/evidence-manifest.json'))
    selected.update(p for p in evidence['inputs'] if p.endswith('.json') and entries[p]['bytes'] <= 256_000
                    and not p.startswith('submission/track-a/data/ledger-snapshots/'))
    missing = selected - entries.keys()
    if missing:
        raise ValueError(f'Not committed: {sorted(missing)}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage = args.output_dir / args.id
    archive = args.output_dir / (args.id + '.tar.gz')
    if stage.exists() or archive.exists():
        raise FileExistsError('Attachment already exists; choose a new ID')
    stage.mkdir()
    source_hashes, remote = {}, {}
    repository = 'https://github.com/Yuer-yuan/InferX'

    def rewrite(name, text):
        def replace(match):
            target = match.group(1).strip('<>')
            if urlsplit(target).scheme or target.startswith('#'):
                return match.group(0)
            rel, sep, anchor = target.partition('#')
            dest = (ROOT / name).parent.joinpath(unquote(rel)).resolve()
            if not dest.is_relative_to(ROOT):
                raise ValueError(f'Link outside repository: {name}: {target}')
            path = str(dest.relative_to(ROOT))
            if path in selected:
                return match.group(0)
            if path in entries:
                url = f'{repository}/blob/{commit}/{quote(path, safe="/")}'
                remote[path] = {'url': url, **entries[path]}
            elif any(p.startswith(path.rstrip('/') + '/') for p in entries):
                url = f'{repository}/tree/{commit}/{quote(path, safe="/")}'
                remote[path] = {'url': url, 'type': 'directory'}
            else:
                raise ValueError(f'Uncommitted link target: {name}: {target}')
            return '](' + url + (sep + anchor if sep else '') + ')'
        return LINK.sub(replace, text)

    for name in sorted(selected):
        data = git('show', commit + ':' + name)
        source_hashes[name] = sha(data)
        if name == SOURCE_BUNDLE and sha(data) != SOURCE_SHA:
            raise ValueError('Source bundle differs from its fixed identity')
        if name.endswith('.md'):
            data = rewrite(name, data.decode()).encode()
        dest = stage / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        dest.chmod(0o755 if entries[name]['mode'] == '100755' else 0o644)

    readme = f'''# InferX · 赛道A提交材料

队名：你说啥都队；成员：陈文欣、张志雄、郭坤昌。

本附件小于50 MB，包含中英文答辩PDF、技术文档、完整002量化重建源码包、固定输入获取工具、现场性能评测命令、116原100题完整记录与独立评分、C14逐请求记录及必要配置。模型权重与大缓存不随包提供，按指南取得公开输入并重建缓存。

完整原始测试记录和历史源码包保存在[固定仓库版本]({repository}/tree/{commit})：`{commit}`。附件中文档指向未附文件的链接已转成该提交的永久链接；它们是明确的网络获取项。附件只调整Markdown链接，不改PDF、源码包或原始数值。仓库中的evidence-manifest用于核验完整仓库，附件自身用根目录MANIFEST.json核验。

## 阅读顺序

1. [中文答辩PDF](submission/basic/presentation/InferX-ZH.pdf) / [English PDF](submission/basic/presentation/InferX-EN.pdf)。
2. [技术文档](submission/track-a/README.md)、[Benchmark](submission/track-a/benchmark.md)。
3. [从源码重建缓存](submission/track-a/quantization-reproduction.md)、[交付物与实测范围](submission/DELIVERY.md)。
4. [现场C14性能评测](submission/basic/demo.md)。

## 校验与复验

解压后先执行`python3 VERIFY.py`。完整构建/评测需要公开模型、FP8输入及环境依赖；依赖的来源、版本和获取命令均在复验指南中。若要逐条使用指南的仓库相对路径、查看全部原始证据或运行材料检查，取得完整固定版本后操作：

```bash
git clone https://github.com/Yuer-yuan/InferX.git InferX-reproduction
cd InferX-reproduction
git checkout --detach {commit}
```

附件内的002源码包位于`{SOURCE_BUNDLE}`，SHA256为`{SOURCE_SHA}`；可以独立解压并从`portable116/README.md`检查完整源码与执行入口。基础模型和FP8按固定来源下载；量化缓存从源码生成。历史664复验包在固定仓库中另行提供，不是默认重建路线的缓存附件。

先校验原始附件，再在其他目录运行测试。现场版本如有改变，主动说明代码、模型或配置的差异。
'''
    (stage / 'README.md').write_text(readme)
    snapshot = {'id': args.id, 'git_commit': commit, 'repository': repository,
                'maximum_attachment_bytes': LIMIT, 'source_bundle': SOURCE_BUNDLE,
                'source_bundle_sha256': SOURCE_SHA, 'weights_and_caches_included': False,
                'markdown_links_pinned_to_commit': True,
                'original_committed_sha256': source_hashes}
    (stage / 'SNAPSHOT.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n')
    (stage / 'REMOTE_FILES.json').write_text(json.dumps(remote, ensure_ascii=False, indent=2) + '\n')
    verifier = '''import hashlib, json
from pathlib import Path
root = Path(__file__).resolve().parent
files = json.loads((root / 'MANIFEST.json').read_text())['files']
assert {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()} == set(files) | {'MANIFEST.json', 'SHA256SUMS'}
assert not any(p.is_symlink() for p in root.rglob('*'))
for name, spec in files.items():
    data = (root / name).read_bytes()
    assert len(data) == spec['bytes'] and hashlib.sha256(data).hexdigest() == spec['sha256'], name
print(f'PASS: {len(files)} attachment files match the snapshot')
'''
    (stage / 'VERIFY.py').write_text(verifier)
    payload = {str(p.relative_to(stage)): {'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())}
               for p in sorted(stage.rglob('*')) if p.is_file()}
    (stage / 'MANIFEST.json').write_text(json.dumps({'git_commit': commit, 'files': payload}, ensure_ascii=False, indent=2) + '\n')
    sums = {name: item['sha256'] for name, item in payload.items()}
    sums['MANIFEST.json'] = sha((stage / 'MANIFEST.json').read_bytes())
    (stage / 'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name, digest in sorted(sums.items())))
    subprocess.run(['python3', str(stage / 'VERIFY.py')], check=True)
    with tarfile.open(archive, 'w:gz', compresslevel=9) as tar:
        tar.add(stage, arcname=args.id)
    if archive.stat().st_size >= LIMIT:
        raise ValueError(f'Attachment exceeds limit: {archive.stat().st_size} bytes')
    digest = sha(archive.read_bytes())
    archive.with_name(archive.name + '.sha256').write_text(f'{digest}  {archive.name}\n')
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size,
                      'sha256': digest, 'git_commit': commit, 'files': len(payload),
                      'pinned_remote_targets': len(remote)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
