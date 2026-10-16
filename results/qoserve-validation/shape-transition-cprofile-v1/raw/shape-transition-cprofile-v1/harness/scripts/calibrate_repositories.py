#!/usr/bin/env python3
"""Freeze eligibility and split before any model generation."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inferx.repository import ablated_source, reference_matches
from inferx.repo_verifier import verify_repo
from prepare_repositories import REPOSITORIES


async def main(args):
    output = args.data/args.output
    output.mkdir(exist_ok=False)
    pools = {repo: [] for repo in REPOSITORIES}
    exclusions = []
    for index, line in enumerate((args.data/'exec_repo_bench.jsonl').read_text().splitlines()):
        row = json.loads(line)
        row['task_id'] = f'ExecRepoBench/{index}'
        if row['repo_name'] not in pools:
            exclusions.append(dict(task_id=row['task_id'], reason='repository outside declared pool'))
            continue
        try:
            if row['fill_type'] != 'grammar-based: block':
                raise ValueError('not grammar-based block')
            if not reference_matches(row, args.data/'fixtures'):
                raise ValueError('reference differs from fixture')
            ablated_source(row)
        except ValueError as exc:
            exclusions.append(dict(task_id=row['task_id'], reason=str(exc)))
            continue
        row['test_mode'] = 'unittest' if row['repo_name'] in ('more-itertools', 'compressed_rtf', 'PTable') else 'pytest'
        pools[row['repo_name']].append(row)
    selected = []
    records = []
    chosen_repos = []
    for repo, pool in pools.items():
        random.Random(2026101402).shuffle(pool)
        print(json.dumps(dict(repo=repo, structurally_eligible=len(pool))), flush=True)
        if len(chosen_repos) == 4:
            exclusions.extend(dict(task_id=x['task_id'], reason='fallback not needed') for x in pool)
            continue
        eligible = []
        for row in pool:
            if len(eligible) == 6:
                exclusions.append(dict(task_id=row['task_id'], reason='after first six eligible'))
                continue
            canonical = await verify_repo(row['middle_code'], row, fixture_root=args.data/'fixtures',
                python=args.python, work_root=args.data/'verification-work', cpu=args.cpu)
            record = dict(task_id=row['task_id'], repo=repo, canonical=canonical)
            if canonical['status'] != 'passed':
                record['reason'] = 'canonical suite failed'
            else:
                row['expected_counts'] = canonical['counts']
                ablation = await verify_repo('pass', row, fixture_root=args.data/'fixtures',
                    python=args.python, work_root=args.data/'verification-work', cpu=args.cpu)
                record['ablation'] = ablation
                # An actual failed assertion/test is needed; timeout or collection
                # failure alone is not evidence the masked block was exercised.
                if (ablation['status'] == 'failed' and ablation.get('counts')
                        and ablation['counts']['failed'] > 0
                        and ablation['counts']['collected'] == canonical['counts']['collected']):
                    eligible.append(row)
                    record['reason'] = 'eligible'
                else:
                    record['reason'] = 'no demonstrated test failure under valid pass ablation'
            records.append(record)
            (output/'calibration.json').write_text(json.dumps(records, indent=2))
            print(json.dumps(dict(task=row['task_id'], reason=record['reason'],
                canonical=canonical['status'], seconds=round(canonical['elapsed'], 3))), flush=True)
            if record['reason'] != 'eligible':
                exclusions.append(dict(task_id=row['task_id'], reason=record['reason']))
            # Same canonical source tree for every row in a repository. If it
            # cannot run, retain evidence and reject that repository as a whole.
            if canonical['status'] != 'passed':
                exclusions.extend(dict(task_id=x['task_id'], reason='repository canonical suite unavailable')
                                  for x in pool if x is not row)
                break
        if len(eligible) >= 6:
            chosen_repos.append(repo)
            for i, row in enumerate(eligible):
                row['split'] = 'dev' if i < 2 else 'eval'
                row['repository_index'] = len(chosen_repos) - 1
                row['split_index'] = i if i < 2 else i - 2
                selected.append(row)
        else:
            exclusions.extend(dict(task_id=x['task_id'], reason='repository has fewer than six eligible')
                              for x in eligible)
    taskfile = output/'tasks.jsonl'
    taskfile.write_text(''.join(json.dumps(row) + '\n' for row in selected))
    freeze = subprocess.check_output([args.python, '-m', 'pip', 'freeze'], text=True)
    (output/'dependencies.txt').write_text(freeze)
    manifest = dict(seed=2026101402, repositories=chosen_repos, tasks=len(selected),
        tasks_sha256=hashlib.sha256(taskfile.read_bytes()).hexdigest(),
        selected_ids={split: [x['task_id'] for x in selected if x['split'] == split] for split in ('dev', 'eval')},
        exclusions=exclusions, complete=len(selected) == 24)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != 'exclusions'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--python', required=True)
    p.add_argument('--cpu', type=int, default=36)
    p.add_argument('--output', default='calibration-v1')
    asyncio.run(main(p.parse_args()))
