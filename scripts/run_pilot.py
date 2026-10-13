"""Live, single-GPU llama.cpp pilot; source remote-env.sh before running."""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inferx.llama_client import LlamaClient
from inferx.scheduler import run_task
from inferx.verifier import verify


async def main(args):
    import aiohttp
    root = Path(os.environ['INFERX_ROOT'])
    model_manifest = json.loads((root / 'models/qwen38-27b/manifest.json').read_text())
    if not model_manifest.get('verified'):
        raise RuntimeError('Model download has not passed SHA256 verification')
    taskfile = root / 'data/tasks.jsonl'
    tasks = [json.loads(x) for x in taskfile.read_text().splitlines()]
    selected = [x for x in tasks if x['split'] in args.splits.split(',')]
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        raise ValueError('No tasks selected')
    cpus = list(map(int, args.verify_cpus.split(',')))
    if len(set(cpus)) != len(cpus):
        raise ValueError('Verifier CPU assignments must be distinct')
    slots = asyncio.Semaphore(len(cpus))
    free_cpus = asyncio.Queue()
    for cpu in cpus:
        free_cpus.put_nowait(cpu)
    output = root / 'runs' / args.name
    output.mkdir(parents=True, exist_ok=False)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        client = LlamaClient(session, args.url)
        await client.wait_idle()
        props = await client.request_json('GET', '/props')
        server_slots = await client.request_json('GET', '/slots')
        if len(server_slots) < max(args.widths):
            raise RuntimeError('Server has fewer slots than requested generation width')
        prompts = {}
        prompt_metadata = {}
        for task in selected:
            prompt = await client.prompt(task['prompt'])
            tokens = await client.request_json('POST', '/tokenize', {'content': prompt})
            n_prompt = len(tokens['tokens'])
            if n_prompt + args.max_tokens > args.context_per_slot:
                raise RuntimeError(f"Context budget exceeded for {task['task_id']}: {n_prompt}")
            prompts[task['task_id']] = prompt
            prompt_metadata[task['task_id']] = dict(tokens=n_prompt,
                sha256=hashlib.sha256(prompt.encode()).hexdigest(), tail=prompt[-120:])
        config = vars(args) | dict(model_manifest=model_manifest, server_props=props,
            server_slots=server_slots, engine='llama.cpp v0.6.0 / native streaming HTTP',
            client_affinity=sorted(os.sched_getaffinity(0)), verifier_workers=len(cpus),
            python=sys.version, aiohttp=importlib.metadata.version('aiohttp'),
            tasks_sha256=hashlib.sha256(taskfile.read_bytes()).hexdigest(),
            prompts=prompt_metadata,
            utc_start=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        (output / 'config.json').write_text(json.dumps(config, indent=2))

        async def trial(task, width, round_id):
            async def generate(i, seed, event):
                return await client.generate(prompts[task['task_id']], seed, event, args.max_tokens)

            async def check(code, event):
                async with slots:
                    cpu = await free_cpus.get()
                    try:
                        event['verify_start'] = time.monotonic() - event['_origin']
                        return await verify(code, task['test'], task['entry_point'], args.verify_timeout, cpu)
                    finally:
                        free_cpus.put_nowait(cpu)

            seed = 100000 * round_id + int(task['task_id'].split('/')[-1]) * 32 + 1000
            try:
                result = await run_task(task, generate, check, width, args.budget, args.deadline, seed)
            finally:
                # Wait for server-side cancellation before the next trial.
                drain = await client.wait_idle()
            result.update(split=task['split'], round=round_id, policy='fixed',
                          wall_timestamp=time.time(), server_drain_s=drain,
                          verifier_workers=len(cpus), gpu=args.gpu)
            result['total_with_drain'] = result['total'] + drain
            return result

        warmup = await trial(selected[0], min(2, max(args.widths)), 999)
        (output / 'warmup.json').write_text(json.dumps(warmup))
        jobs = [(t, w, r) for r in range(args.round_offset, args.round_offset + args.rounds)
                for t in selected for w in args.widths]
        random.Random(20261014 + args.round_offset).shuffle(jobs)
        with (output / 'results.jsonl').open('w', buffering=1) as handle:
            for n, (task, width, round_id) in enumerate(jobs):
                result = await trial(task, width, round_id)
                handle.write(json.dumps(result) + '\n')
                print(json.dumps(dict(progress=f'{n + 1}/{len(jobs)}', task=task['task_id'],
                    width=width, round=round_id, solved=result['solved'],
                    seconds=round(result['total_with_drain'], 3))), flush=True)
        (output / 'COMPLETE').write_text(str(len(jobs)) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18080')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--context-per-slot', type=int, default=2048)
    parser.add_argument('--splits', default='dev,eval')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--widths', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--round-offset', type=int, default=0)
    parser.add_argument('--budget', type=int, default=8)
    parser.add_argument('--deadline', type=float, default=120)
    parser.add_argument('--max-tokens', type=int, default=768)
    parser.add_argument('--verify-timeout', type=float, default=3)
    parser.add_argument('--verify-cpus', default='8,9')
    asyncio.run(main(parser.parse_args()))
