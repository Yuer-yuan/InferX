"""Fetch the selected 27B GGUF through hf-mirror and verify its LFS digest."""
import hashlib
import json
import os
from pathlib import Path
import time
import requests
import hf_transfer
from huggingface_hub import HfApi

root = Path(os.environ['INFERX_ROOT'])
assert root.resolve() == Path('/home/guokc/mypro/gnn-mount-sda1/InferX')
assert os.environ['HF_ENDPOINT'] == 'https://hf-mirror.com'
repo = 'unsloth/Qwen3.8-27B-GGUF'
revision = '4ca720788d1e01f1bff70c033e0d0028fd02e502'
filename = 'Qwen3.8-27B-UD-Q4_K_M.gguf'
info = HfApi(endpoint=os.environ['HF_ENDPOINT']).model_info(repo, revision=revision, files_metadata=True)
entry = next(x for x in info.siblings if x.rfilename == filename)
manifest = dict(repo=repo, revision=revision, filename=filename, size=entry.size,
                expected_sha256=entry.lfs.sha256, endpoint=os.environ['HF_ENDPOINT'], verified=False)
directory = root / 'models/qwen38-27b'
directory.mkdir(parents=True, exist_ok=True)
path = directory / filename
(directory / 'manifest.json').write_text(json.dumps(manifest, indent=2))
url = f"https://hf-mirror.com/{repo}/resolve/{revision}/{filename}"
state = dict(bytes=0, start=time.monotonic(), last=0)
def progress(count):
    state['bytes'] += count
    elapsed = time.monotonic() - state['start']
    if elapsed - state['last'] >= 30:
        state['last'] = elapsed
        print(json.dumps(dict(bytes=state['bytes'], total=entry.size,
                              MBps=round(state['bytes']/elapsed/1e6, 2))), flush=True)

# Obtain the standard CDN redirect through the required mirror; do not log
# its temporary signed query string. hf_transfer issues bounded parallel ranges.
with requests.get(url, stream=True, timeout=120) as response:
    response.raise_for_status()
    redirected_url = response.url
print('Downloading pinned Qwen3.8-27B GGUF via hf-mirror', flush=True)
hf_transfer.download(redirected_url, str(path), max_files=64, chunk_size=8*1024**2,
                     parallel_failures=8, max_retries=10, callback=progress)
with path.open('rb') as f:
    actual = hashlib.file_digest(f, 'sha256').hexdigest()
if path.stat().st_size != entry.size or actual != entry.lfs.sha256:
    raise RuntimeError('GGUF size or SHA256 mismatch')
manifest.update(verified=True, actual_sha256=actual)
(directory / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print('GGUF VERIFIED', actual, flush=True)
