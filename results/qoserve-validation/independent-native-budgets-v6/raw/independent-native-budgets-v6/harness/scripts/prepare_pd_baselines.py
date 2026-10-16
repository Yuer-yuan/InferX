"""Download a pinned INT4 checkpoint exclusively through hf-mirror to HDD."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests

REPO = 'RedHatAI/Qwen3.8-27B-INT4'
REVISION = '91bd022d5b49442a868bc35008f6c21e1860edfa'


def main():
    root = Path(os.environ['INFERX_ROOT'])
    assert root == Path('/home/guokc/mypro/gnn-mount-sda1/InferX')
    assert os.environ['HF_ENDPOINT'] == 'https://hf-mirror.com'
    sys.path.insert(0, str(root/'download-deps'))
    import hf_transfer
    destination = root/'models/qwen38-int4-redhat'
    destination.mkdir(exist_ok=True)
    response = requests.get(f'https://hf-mirror.com/api/models/{REPO}/revision/{REVISION}',
                            params={'blobs':'true'}, timeout=60)
    response.raise_for_status()
    metadata = response.json()
    assert metadata['sha'] == REVISION
    (destination/'source-api.json').write_text(json.dumps(metadata, indent=2))
    manifest = dict(repo=REPO, revision=REVISION, endpoint='https://hf-mirror.com', files=[])
    for entry in metadata['siblings']:
        name = entry['rfilename']
        if name.startswith('.') or name in {'README.md', 'recipe.yaml'}:
            continue
        path = destination/name
        assert path.resolve().is_relative_to(destination.resolve())
        path.parent.mkdir(parents=True, exist_ok=True)
        lfs = entry.get('lfs', {})
        expected = lfs.get('sha256') or lfs.get('oid')
        size = lfs.get('size') or entry.get('size')
        if not path.exists() or (size is not None and path.stat().st_size != size):
            part = path.with_name(path.name+'.part')
            if part.exists():
                part.rename(part.with_name(part.name+f'.previous-{int(time.time())}'))
            # A new query avoids hf-mirror returning an expired cached CDN URL.
            url = f'https://hf-mirror.com/{REPO}/resolve/{REVISION}/{name}?download=true&inferx={time.time_ns()}'
            print(json.dumps(dict(event='downloading', file=name, bytes=size)), flush=True)
            if lfs:
                hf_transfer.download(url=url, filename=str(part), max_files=8,
                    chunk_size=16*1024*1024, parallel_failures=3, max_retries=5)
            else:
                downloaded = requests.get(url, timeout=120)
                downloaded.raise_for_status()
                part.write_bytes(downloaded.content)
            if size is not None and part.stat().st_size != size:
                raise RuntimeError(f'download size mismatch: {name}')
            part.rename(path)
        digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
        if expected and digest != expected:
            raise RuntimeError(f'download hash mismatch: {name}')
        manifest['files'].append(dict(file=name, bytes=path.stat().st_size,
                                     sha256=digest, expected_sha256=expected))
        (destination/'manifest.json').write_text(json.dumps(manifest, indent=2))
        print(json.dumps(dict(event='verified', file=name, bytes=path.stat().st_size)), flush=True)
    manifest['verified'] = True
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
