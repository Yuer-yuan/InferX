"""Export lightweight evidence; candidate text and full logs remain on HDD."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from analyze_repository import validate_grid, select_baseline


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_episode(episode):
    for task in episode['tasks']:
        for candidate in task['candidates']:
            candidate.pop('code', None)
            candidate.pop('raw_output', None)
            if 'verification' in candidate:
                candidate['verification'].pop('output_tail', None)
    return episode


def main(args):
    rows = []; sources = {}
    for phase, count in [('dev', 48), ('eval', 64)]:
        run = getattr(args, phase)
        if int((run/'COMPLETE').read_text()) != count:
            raise ValueError('unexpected COMPLETE count')
        phase_rows = [json.loads(line) for line in (run/'episodes.jsonl').read_text().splitlines()]
        if len(phase_rows) != count:
            raise ValueError('raw episode count differs from COMPLETE')
        rows.extend(phase_rows)
        for name in ('config.json', 'episodes.jsonl', 'prompts.json', 'warmup.json', 'telemetry.csv', 'COMPLETE'):
            sources[str(run/name)] = digest(run/name)
    validate_grid(rows)
    selected = json.loads((args.dev/'selection.json').read_text())['selected_baseline']
    if selected != select_baseline(rows):
        raise ValueError('saved pre-evaluation development selection differs')
    args.output.mkdir(parents=True, exist_ok=False)
    for phase in ('dev', 'eval'):
        run = getattr(args, phase)
        for name in ('config.json', 'telemetry.csv', 'COMPLETE'):
            shutil.copyfile(run/name, args.output/f'{phase}-{name}')
        subset = [compact_episode(e) for e in rows if e['split'] == phase]
        (args.output/f'{phase}-episodes.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in subset))
    shutil.copyfile(args.dev/'selection.json', args.output/'development-selection.json')
    for name in ('manifest.json', 'dependencies.txt'):
        shutil.copyfile(args.prepared/name, args.output/f'dataset-{name}')
    calibration = json.loads((args.prepared/'calibration.json').read_text())
    for record in calibration:
        for key in ('canonical', 'ablation'):
            if key in record:
                record[key].pop('output_tail', None)
    (args.output/'calibration.json').write_text(json.dumps(calibration, indent=2) + '\n')
    (args.output/'raw-sources.json').write_text(json.dumps(sources, indent=2) + '\n')
    hashes = {p.name: digest(p) for p in args.output.iterdir() if p.is_file()}
    (args.output/'SHA256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(json.dumps(dict(episodes=len(rows), tasks=sum(len(e['tasks']) for e in rows), selected_baseline=selected)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('dev', 'eval', 'prepared', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    main(p.parse_args())
