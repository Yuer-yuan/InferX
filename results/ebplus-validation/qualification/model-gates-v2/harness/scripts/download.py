"""Prepare the pinned HumanEval split; use prepare_27b.py for model weights."""
import hashlib
import json
import os
from pathlib import Path
import random

from huggingface_hub import snapshot_download


def main():
    root = Path(os.environ['INFERX_ROOT']).resolve()
    if str(root) != '/home/guokc/mypro/gnn-mount-sda1/InferX' or os.environ.get('HF_ENDPOINT') != 'https://hf-mirror.com':
        raise RuntimeError('source remote-env.sh first')
    data_id = 'openai/openai_humaneval'
    data_rev = '7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544'
    snapshot_download(data_id, repo_type='dataset', revision=data_rev,
        local_dir=root / 'data/humaneval', allow_patterns=['*.parquet', 'README.md'], max_workers=2)
    import pyarrow.parquet as pq
    rows = []
    for path in sorted((root / 'data/humaneval').rglob('*.parquet')):
        rows.extend(pq.read_table(path).to_pylist())
    rows.sort(key=lambda x: int(x['task_id'].split('/')[-1]))
    if len(rows) != 164 or len({x['task_id'] for x in rows}) != 164:
        raise RuntimeError(f'unexpected dataset size: {len(rows)}')
    random.Random(20261014).shuffle(rows)
    for i, row in enumerate(rows):
        row['split'] = 'dev' if i < 16 else 'eval' if i < 64 else 'reserve'
    taskfile = root / 'data/tasks.jsonl'
    content = ''.join(json.dumps(x) + '\n' for x in rows)
    digest = hashlib.sha256(content.encode()).hexdigest()
    expected = 'e1a6f59bb1b9b1877e3763e122a63a6839020f1c73f53473a08e98763a89df8a'
    if digest != expected:
        raise RuntimeError('Task serialization or pinned dataset changed')
    taskfile.write_text(content)
    metadata = dict(dataset_id=data_id, dataset_revision=data_rev,
        endpoint=os.environ['HF_ENDPOINT'], split_seed=20261014,
        split_sizes={'dev': 16, 'eval': 48, 'reserve': 100}, tasks_sha256=digest)
    (root / 'data/manifest.json').write_text(json.dumps(metadata, indent=2))
    print('Verified dataset:', digest, flush=True)


if __name__ == '__main__':
    main()
