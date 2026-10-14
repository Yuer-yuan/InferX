"""Pinned v2 artifact for the SM89 fork; current upstream main uses incompatible v3."""
import hashlib
import json
import os
from pathlib import Path
import time

import requests
import hf_transfer

root = Path(os.environ['INFERX_ROOT'])
assert root.resolve() == Path('/home/guokc/mypro/gnn-mount-sda1/InferX')
assert os.environ['HF_ENDPOINT'] == 'https://hf-mirror.com'
repo = 'neroued/Qwen3.8-27B-NInfer'
revision = '3526913004b1cf552cb57b88d6a5c6f5e4a89a70'
filename = 'qwen3_8_27b.ninfer'
expected = 'eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e'
size = 18210531328
directory = root / 'models/qwen38-ninfer'
directory.mkdir(parents=True, exist_ok=True)
path = directory / filename
manifest = dict(repo=repo, revision=revision, filename=filename, size=size,
                expected_sha256=expected, endpoint=os.environ['HF_ENDPOINT'], verified=False)
(directory / 'manifest.json').write_text(json.dumps(manifest, indent=2))
state = dict(bytes=0, start=time.monotonic(), last=0)


def progress(count):
    state['bytes'] += count
    elapsed = time.monotonic() - state['start']
    if elapsed - state['last'] >= 30:
        state['last'] = elapsed
        print(json.dumps(dict(bytes=state['bytes'], total=size,
                              MBps=round(state['bytes'] / elapsed / 1e6, 2))), flush=True)


if not path.exists() or path.stat().st_size != size:
    url = f'https://hf-mirror.com/{repo}/resolve/{revision}/{filename}'
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        redirected_url = response.url
    hf_transfer.download(redirected_url, str(path), max_files=64, chunk_size=8*1024**2,
                         parallel_failures=8, max_retries=10, callback=progress)
with path.open('rb') as f:
    actual = hashlib.file_digest(f, 'sha256').hexdigest()
if path.stat().st_size != size or actual != expected:
    raise RuntimeError('NInfer artifact size or SHA256 mismatch; remove invalid artifact before retry')
manifest.update(verified=True, actual_sha256=actual)
(directory / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print('NInfer artifact VERIFIED', actual, flush=True)
