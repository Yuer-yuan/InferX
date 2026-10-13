"""Run after sourcing remote-env.sh; all artifacts and caches stay on HDD."""
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess

from huggingface_hub import HfApi, snapshot_download


def main():
    root = Path(os.environ["INFERX_ROOT"]).resolve()
    if str(root) != "/home/guokc/mypro/gnn-mount-sda1/InferX" or os.environ.get("HF_ENDPOINT") != "https://hf-mirror.com":
        raise RuntimeError("source remote-env.sh first")
    api = HfApi(endpoint=os.environ["HF_ENDPOINT"])
    model_id = "Qwen/Qwen2.5-Coder-3B-Instruct"
    model_rev = "488639f1ff808d1d3d0ba301aef8c11461451ec5"
    data_id = "openai/openai_humaneval"
    data_rev = api.dataset_info(data_id).sha
    metadata = dict(model_id=model_id, model_revision=model_rev,
                    dataset_id=data_id, dataset_revision=data_rev,
                    endpoint=os.environ["HF_ENDPOINT"])
    (root / "data/manifest.json").write_text(json.dumps(metadata, indent=2))
    snapshot_download(data_id, repo_type="dataset", revision=data_rev,
                      local_dir=root / "data/humaneval", allow_patterns=["*.parquet", "README.md"], max_workers=2)
    import pyarrow.parquet as pq
    rows = []
    for path in sorted((root / "data/humaneval").rglob("*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    rows.sort(key=lambda x: int(x["task_id"].split("/")[-1]))
    if len(rows) != 164 or len({x['task_id'] for x in rows}) != 164:
        raise RuntimeError(f"unexpected dataset size: {len(rows)}")
    random.Random(20261014).shuffle(rows)
    for i, row in enumerate(rows):
        row["split"] = "dev" if i < 16 else "eval" if i < 64 else "reserve"
    taskfile = root / "data/tasks.jsonl"
    taskfile.write_text("".join(json.dumps(x) + "\n" for x in rows))
    metadata["tasks_sha256"] = hashlib.sha256(taskfile.read_bytes()).hexdigest()
    print("Dataset ready:", len(rows), "tasks", flush=True)
    model_dir = root / "models/qwen-coder-3b"
    snapshot_download(model_id, revision=model_rev, local_dir=model_dir,
                      allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken", "README.md"], max_workers=4)
    # Range requests avoid a single slow mirror connection dominating a 4 GB
    # shard. Published LFS SHA256 is checked before any model is loaded.
    info = api.model_info(model_id, revision=model_rev, files_metadata=True)
    for item in info.siblings:
        if not item.rfilename.endswith('.safetensors'):
            continue
        target = model_dir / item.rfilename
        expected = item.lfs.sha256
        if target.exists() and target.stat().st_size == item.size:
            with target.open('rb') as handle:
                if hashlib.file_digest(handle, 'sha256').hexdigest() == expected:
                    print('Verified existing shard:', item.rfilename, flush=True)
                    continue
        if not target.exists():
            partials = list((model_dir / '.cache/huggingface/download').glob('*.' + expected + '.incomplete'))
            if partials:
                partials[0].rename(target)
        url = f"{os.environ['HF_ENDPOINT']}/{model_id}/resolve/{model_rev}/{item.rfilename}"
        subprocess.run(['aria2c', '--continue=true', '--file-allocation=none', '--split=16',
                        '--max-connection-per-server=16', '--min-split-size=4M',
                        '--max-tries=10', '--retry-wait=3', '--summary-interval=30',
                        '--console-log-level=warn', '--download-result=hide',
                        '--dir=' + str(model_dir), '--out=' + item.rfilename, url], check=True)
        with target.open('rb') as handle:
            actual = hashlib.file_digest(handle, 'sha256').hexdigest()
        if actual != expected:
            raise RuntimeError('LFS SHA256 mismatch: ' + item.rfilename)
        print('Verified shard:', item.rfilename, flush=True)
    metadata["files"] = {}
    for path in sorted(model_dir.glob("*")):
        if path.is_file():
            with path.open('rb') as handle:
                digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            metadata["files"][path.name] = {"size": path.stat().st_size, "sha256": digest}
    (root / "data/manifest.json").write_text(json.dumps(metadata, indent=2))
    print("Download and hash verification complete", flush=True)


if __name__ == "__main__":
    main()
