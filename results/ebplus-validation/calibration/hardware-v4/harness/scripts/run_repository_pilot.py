#!/usr/bin/env python3
"""Single GPU, frozen repository workload and complete episode records."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inferx.episodes import POLICIES, run_episode
from inferx.llama_client import LlamaClient, create_session
from inferx.repo_prompt import build_prompt
from inferx.repo_verifier import verify_repo
from inferx.repository import parse_completion, line_start_task


def job_grid(tasks, phase):
    dev = [x for x in tasks if x['split'] == 'dev']
    if phase == 'dev':
        jobs = [dict(tasks=[task], policy=policy, round_id=r, arrivals=[0], mode='single',
                     episode_id=task['task_id'])
                for r in (0, 1) for task in dev for policy in ('fixed1', 'fixed2', 'fixed4')]
    elif phase == 'smoke':
        sample = [next(t for t in dev if t['repository_index'] == i) for i in range(4)]
        jobs = [dict(tasks=sample, policy=p, round_id=999, arrivals=[0]*4,
                     mode='burst', episode_id='smoke') for p in ('fixed4', 'feedback-fair')]
    else:
        rng = random.Random(2026101402)
        groups = []
        for i in range(4):
            group = [x for x in tasks if x['split'] == 'eval' and x['repository_index'] == i]
            rng.shuffle(group)
            groups.append(group)
        episodes = [[g[i] for g in groups] for i in range(4)]
        jobs = [dict(tasks=episode, policy=policy, round_id=r,
                     arrivals=[0]*4 if mode == 'burst' else [0, 5, 10, 15],
                     mode=mode, episode_id=f'episode/{i}')
                for r in (0, 1) for i, episode in enumerate(episodes)
                for mode in ('burst', 'stagger') for policy in POLICIES]
    random.Random(2026101402).shuffle(jobs)
    return jobs


async def cancel_smoke(client, prompt):
    events = [dict(tokens=0, _origin=time.monotonic()) for _ in range(8)]
    jobs = [asyncio.create_task(client.generate(prompt, 98700+i, events[i], 1024)) for i in range(8)]
    peak = 0
    try:
        async with asyncio.timeout(25):
            while peak < 8:
                slots = await client.request_json('GET', '/slots')
                peak = max(peak, sum(s['is_processing'] for s in slots))
                for job in jobs:
                    if job.done():
                        job.result()
                await asyncio.sleep(.01)
    finally:
        for job in jobs:
            job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
    drain = await client.wait_idle()
    return dict(peak_active=peak, drain_s=drain, tokens=sum(e['tokens'] for e in events))


async def main(args):
    root = Path(os.environ['INFERX_ROOT'])
    data = root/'data/exec-repo-v1'
    prepared = data/'calibration-v5'
    manifest = json.loads((prepared/'manifest.json').read_text())
    taskfile = prepared/'tasks.jsonl'
    assert manifest['complete'] and hashlib.sha256(taskfile.read_bytes()).hexdigest() == manifest['tasks_sha256']
    tasks = [line_start_task(json.loads(line)) for line in taskfile.read_text().splitlines()]
    jobs = job_grid(tasks, args.phase)
    selected = {t['task_id']: t for j in jobs for t in j['tasks']}
    warmup_task = next(t for t in tasks if t['split'] == 'dev')
    selected[warmup_task['task_id']] = warmup_task
    output = root/'runs'/args.name
    output.mkdir(parents=True, exist_ok=False)
    cpus = asyncio.Queue()
    for cpu in (36, 37):
        cpus.put_nowait(cpu)
    code_root = Path(__file__).resolve().parents[1]
    runtime_files = list((code_root/'inferx').glob('*.py')) + [Path(__file__), code_root/'scripts/start_repo_server.sh']
    hashes = {str(p.relative_to(code_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime_files}
    async with create_session() as session:
        client = LlamaClient(session, 'http://127.0.0.1:18081')
        await client.wait_idle()
        slots = await client.request_json('GET', '/slots')
        if len(slots) != 8 or any(s['n_ctx'] != 8192 for s in slots):
            raise RuntimeError('Expected eight slots, each with 8192 context')
        prompts = {}; metadata = {}
        for task_id, task in selected.items():
            prompts[task_id], metadata[task_id] = await build_prompt(client, task)
        config = dict(phase=args.phase, runtime_revision=args.revision, runtime_hashes=hashes,
            dataset_manifest=manifest, insertion_boundary='line start; prefix indentation moved into gap before generation',
            dependency_freeze=(prepared/'dependencies.txt').read_text(),
            model_manifest=json.loads((root/'models/qwen38-27b/manifest.json').read_text()),
            server_props=await client.request_json('GET', '/props'), server_slots=slots,
            prompts=metadata, client_affinity=sorted(os.sched_getaffinity(0)), gpu=1,
            verify_cpus=[36, 37], max_tokens=1024, max_prompt_tokens=6000, budget=4, deadline=120,
            verify_timeout=30, temperature=.8, top_p=.95, top_k=0, min_p=0, repeat_penalty=1,
            planned_jobs=[{k: ([t['task_id'] for t in v] if k == 'tasks' else v)
                           for k, v in j.items()} for j in jobs],
            utc_start=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        (output/'config.json').write_text(json.dumps(config, indent=2))
        (output/'prompts.json').write_text(json.dumps(prompts, indent=2))

        async def generate(task, i, seed, event):
            await client.generate(prompts[task['task_id']], seed, event, 1024)
            try:
                code = parse_completion(event['raw_output'])
            except ValueError as exc:
                event['format_error'] = str(exc)
                return ''
            event['code_sha256'] = hashlib.sha256(code.encode()).hexdigest()
            return code

        async def check(task, code, event):
            if 'format_error' in event:
                return dict(status='failed', error='invalid JSON completion', counts=None,
                            elapsed=0, test_elapsed=0)
            cpu = await cpus.get()
            try:
                event['verify_start'] = time.monotonic() - event['_origin']
                return await verify_repo(code, task, fixture_root=data/'fixtures',
                    python=str(root/'repo-venv/bin/python'), work_root=data/'verification-work', cpu=cpu)
            finally:
                cpus.put_nowait(cpu)

        async def trial(job):
            try:
                result = await run_episode(job['tasks'], generate, check, policy=job['policy'],
                    arrivals=job['arrivals'], round_id=job['round_id'])
            finally:
                drain = await client.wait_idle()
            result.update(episode_id=job['episode_id'], mode=job['mode'], split='eval' if args.phase == 'eval' else 'dev',
                server_drain_s=drain, makespan_with_drain=result['makespan']+drain,
                wall_timestamp=time.time(), host_loadavg=list(os.getloadavg()))
            return result

        if args.phase == 'smoke':
            (output/'cancel-smoke.json').write_text(json.dumps(await cancel_smoke(client, next(iter(prompts.values())))))
            canonical_task = next(iter(selected.values()))
            canonical = await verify_repo(canonical_task['middle_code'], canonical_task,
                fixture_root=data/'fixtures', python=str(root/'repo-venv/bin/python'),
                work_root=data/'verification-work', cpu=36)
            if canonical['status'] != 'passed':
                raise RuntimeError('live canonical repository integration failed: ' + str(canonical))
            (output/'canonical-smoke.json').write_text(json.dumps(canonical))
        warmup_job = dict(jobs[0], tasks=[warmup_task], arrivals=[0], policy='fixed1', round_id=998)
        (output/'warmup.json').write_text(json.dumps(await trial(warmup_job)))
        with (output/'telemetry.csv').open('w') as telemetry:
            monitor = await asyncio.create_subprocess_exec('nvidia-smi', '-i', '1',
                '--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,temperature.gpu',
                '--format=csv', '-l', '1', stdout=telemetry, stderr=telemetry)
            try:
                with (output/'episodes.jsonl').open('w', buffering=1) as handle:
                    for n, job in enumerate(jobs):
                        result = await trial(job)
                        handle.write(json.dumps(result) + '\n')
                        print(json.dumps(dict(progress=f'{n+1}/{len(jobs)}', episode=job['episode_id'],
                            policy=job['policy'], mode=job['mode'], round=job['round_id'],
                            solved=sum(t['solved'] for t in result['tasks']),
                            seconds=round(result['makespan_with_drain'], 3))), flush=True)
            finally:
                if monitor.returncode is None:
                    monitor.terminate()
                await monitor.wait()
        (output/'COMPLETE').write_text(str(len(jobs)) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--phase', choices=['smoke', 'dev', 'eval'], required=True)
    p.add_argument('--revision', required=True)
    asyncio.run(main(p.parse_args()))
