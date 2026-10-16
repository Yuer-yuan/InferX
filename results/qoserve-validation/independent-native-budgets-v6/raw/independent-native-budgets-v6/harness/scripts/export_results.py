"""Export a completed pilot with source hashes and without generated code text."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze import analyze, paired_intervals, validate_coverage


def main(args):
    root = Path(os.environ['INFERX_ROOT'])
    rows, origins, configs = [], {}, {}
    for name in args.runs:
        directory = root / 'runs' / name
        path = directory / 'results.jsonl'
        content = path.read_bytes()
        current = [json.loads(x) for x in content.splitlines()]
        if int((directory / 'COMPLETE').read_text()) != len(current):
            raise RuntimeError('Run completion count mismatch: ' + name)
        configs[name] = json.loads((directory / 'config.json').read_text())
        origins[name] = dict(path=str(path), rows=len(current),
                             raw_sha256=hashlib.sha256(content).hexdigest())
        revision = directory / 'code-revision.txt'
        origins[name]['code_revision'] = revision.read_text().strip() if revision.exists() else 'unrecorded'
        hashes = directory / 'runtime-source-hashes.json'
        if hashes.exists():
            origins[name]['runtime_sources'] = json.loads(hashes.read_text())
        for row in current:
            for event in row['candidates']:
                event.pop('raw_output', None)
                event.pop('code', None)
            row['source_run'] = name
        rows.extend(current)
    validate_coverage(rows, {'dev': 16, 'eval': 48}, [1, 2, 4, 8], [0, 1])
    result = analyze(rows)
    result['selected_vs_width1'] = paired_intervals(rows, result['selected_width'])
    output = root / 'exports' / args.name
    output.mkdir(parents=True, exist_ok=False)
    (output / 'trials.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
    (output / 'summary.json').write_text(json.dumps(result, indent=2))
    (output / 'provenance.json').write_text(json.dumps(dict(source_runs=origins, configs=configs), indent=2))
    with (output / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, ['group'] + list(next(iter(result['groups'].values()))))
        writer.writeheader()
        writer.writerows(dict(group=k, **v) for k, v in result['groups'].items())
    for name in ['hardware.json', 'manifest.json', 'canonical-check.json', 'verifier-capacity.json']:
        shutil.copyfile(root / 'data' / name, output / name)
    shutil.copyfile(root / 'models/qwen38-27b/manifest.json', output / 'model-manifest.json')
    shutil.copyfile(root / 'source/llama-source.sha256', output / 'llama-source.sha256')
    shutil.copyfile(root / 'source/llama-source.commit', output / 'llama-source.commit')
    print(json.dumps(dict(output=str(output), trials=len(rows), selected_width=result['selected_width'],
                          decision=result['decision'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--name', default='pilot')
    main(parser.parse_args())
