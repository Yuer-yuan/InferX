"""Replay frozen formal payloads alone, using the existing owned GPU1 controller.

One diagnostic repetition per case, fresh server for each workload. The server
keeps its formal C4/resource configuration; producer=0 serializes client work.
GPU0 was explicitly released by the user and must remain free.
"""
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

from inferx.pd_baselines import validate_case
from scripts import run_pd_baselines as controller
from scripts import run_concurrency_campaign as previous


ROOT = Path(os.environ['INFERX_ROOT'])
RAW = ROOT / 'pd-baselines/raw'
CASES = ['sglang-ctx-control', 'sglang-ctx-responsive128', 'sglang-pdmux-private-best']
SOURCE_RUNS = {
    CASES[0]: 'pd-strategies-matched-formal-r1',
    CASES[1]: 'pd-strategies-matched-formal-r1',
    CASES[2]: 'pd-strategies-pdmux-private-main-v1',
}
GPU0 = 'GPU-a66b08e8-8b65-0faf-22e2-b951a97b2b03'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shutdown_check(saved):
    processes = previous.gpu_processes()
    if (saved['pid'], previous.GPU) not in processes:
        raise RuntimeError('Original GPU1 process changed')
    if any(uuid == GPU0 for _, uuid in processes):
        raise RuntimeError('GPU0 no longer free; refusing to change services')
    metrics = previous.get('/metrics', 'http://127.0.0.1:18081')
    if any(float(metrics[key]) != 0 for key in
           ['llamacpp:requests_processing', 'llamacpp:requests_deferred']):
        raise RuntimeError('Original GPU1 service is busy')


def main():
    run_id = sys.argv[1]
    destination = ROOT / 'pd-strategies' / run_id
    destination.mkdir(exist_ok=False)
    original_script = Path(__file__)
    script_copy = destination / 'run_solo.py'
    script_copy.write_bytes(original_script.read_bytes())
    cases, specifications, sources = [], {}, {}
    for case_id in CASES:
        source = RAW / SOURCE_RUNS[case_id]
        config_path = source / 'config.json'
        sources[str(config_path)] = digest(config_path)
        case = next(c for c in json.loads(config_path.read_text()) if c['id'] == case_id)
        validate_case(case)
        cases.append(case)
        for group in ['short-control', 'cold-low', 'resume-high']:
            path = source / f'{case_id}-r0-{group}' / group / 'specifications.json'
            sources[str(path)] = digest(path)
            frozen = json.loads(path.read_text())
            selected = (frozen if group == 'short-control' else
                        [r for r in frozen if r['prompt_id'] != 'tool-24576'])
            expected_count = 24 if group == 'short-control' else 1
            assert len(selected) == expected_count
            requests = copy.deepcopy(selected)
            for request in requests:
                request.pop('offset_s')
                request['producer'] = 0
            assert all(r['payload'] == f['payload'] and r['expected_tool'] == f['expected_tool']
                       for r, f in zip(requests, selected))
            specifications[(case_id, group)] = requests
    config = destination / 'config.json'
    config.write_text(json.dumps({'cases': cases}, indent=2) + '\n')
    scope = dict(
        scope='Supplemental isolated execution diagnostic; one fresh-server repetition per case/group.',
        server_configuration='Exactly frozen formal C4 configurations, MTP0/eager/overlap-off.',
        short='24 exact formal short payloads, sent serially; no client request overlap.',
        long='One exact cold/resume payload alone after the same four concurrent short warmups.',
        cache='Same warmup protocol; observed cache counters must be checked, not assumed.',
        compared_with='Existing three independent formal mixed repetitions; not three solo repetitions.',
        gpu0='User explicitly stopped its service; preserve empty compute-process set.',
        source_sha256=sources,
        controller_sha256=digest(script_copy),
    )
    (destination / 'scope.json').write_text(json.dumps(scope, indent=2) + '\n')

    def isolated_requests(definition, templates, grid, label):
        group = definition['name']
        case_id = next(c for c in CASES if label == f'{c}-r0-{group}')
        return copy.deepcopy(specifications[(case_id, group)])

    previous.final_shutdown_check = shutdown_check
    controller.arrival_requests = isolated_requests
    sys.argv = [str(script_copy), '--config', str(config), '--cases', *CASES,
                '--repeat', '1', '--groups', 'short-control', 'cold-low', 'resume-high',
                '--short-rate', '.5', '--run-id', run_id]
    controller.main()


if __name__ == '__main__':
    main()
