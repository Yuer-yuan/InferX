"""Single-GPU live pilot. Launch with remote-env.sh and CUDA_VISIBLE_DEVICES."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inferx.scheduler import run_task
from inferx.verifier import verify


def extract_code(text):
    block = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (block.group(1) if block else text).strip()


async def main(args):
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    from transformers import AutoTokenizer
    root = Path(os.environ['INFERX_ROOT'])
    model = str(root / 'models/qwen-coder-3b')
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=model, dtype='bfloat16', tensor_parallel_size=1, max_model_len=2048,
        gpu_memory_utilization=.9, enable_prefix_caching=True,
        enable_chunked_prefill=True, max_num_batched_tokens=2048,
        max_num_seqs=16, disable_log_requests=True,
    ))
    tasks = [json.loads(x) for x in (root / 'data/tasks.jsonl').read_text().splitlines()]
    selected = [x for x in tasks if x['split'] in args.splits.split(',')]
    if args.limit:
        selected = selected[:args.limit]
    cpus = list(map(int, args.verify_cpus.split(',')))
    slots = asyncio.Semaphore(len(cpus))
    free_cpus = asyncio.Queue()
    for cpu in cpus:
        free_cpus.put_nowait(cpu)
    output = root / 'runs' / args.name
    output.mkdir(parents=True, exist_ok=False)
    config = vars(args) | dict(model=model, engine='vLLM AsyncLLMEngine V0',
                              visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                              engine_affinity=sorted(os.sched_getaffinity(0)),
                              utc_start=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    (output / 'config.json').write_text(json.dumps(config, indent=2))

    async def trial(task, width, round_id):
        prompt = tokenizer.apply_chat_template([
            {'role': 'system', 'content': 'You are a helpful coding assistant.'},
            {'role': 'user', 'content': 'Implement the following Python function according to its specification. Use only the Python standard library. Return a complete Python module including any imports, in a single Python code block. Do not include tests or explanations.\n\n' + task['prompt']}
        ], tokenize=False, add_generation_prompt=True)

        async def generate(i, seed, event):
            rid = uuid.uuid4().hex
            completed = False
            latest = None
            try:
                async for result in engine.generate(prompt, SamplingParams(
                    temperature=.8, top_p=.95, max_tokens=args.max_tokens, seed=seed), rid):
                    latest = result.outputs[0]
                    event['tokens'] = len(latest.token_ids)
                    if latest.token_ids and 'first_token' not in event:
                        event['first_token'] = time.monotonic() - event['_origin']
                    completed = result.finished
                event['finish_reason'] = latest.finish_reason
                event['raw_output'] = latest.text
                code = extract_code(latest.text)
                event['code_sha256'] = hashlib.sha256(code.encode()).hexdigest()
                return code
            finally:
                if not completed:
                    await engine.abort(rid)

        async def check(code, event):
            async with slots:
                cpu = await free_cpus.get()
                try:
                    event['verify_start'] = time.monotonic() - event['_origin']
                    return await verify(code, task['test'], task['entry_point'], args.verify_timeout, cpu)
                finally:
                    free_cpus.put_nowait(cpu)

        seed = 100000 * round_id + int(task['task_id'].split('/')[-1]) * 32 + 1000
        result = await run_task(task, generate, check, width, args.budget, args.deadline, seed)
        result.update(split=task['split'], round=round_id, policy='fixed',
                      wall_timestamp=time.time())
        return result

    # Separate warmup, never included in the measured file.
    warmup = await trial(selected[0], 2, 999)
    (output / 'warmup.json').write_text(json.dumps(warmup))
    jobs = [(t, w, r) for r in range(args.round_offset, args.round_offset + args.rounds)
            for t in selected for w in args.widths]
    random.Random(20261014).shuffle(jobs)
    with (output / 'results.jsonl').open('w', buffering=1) as handle:
        for n, (task, width, round_id) in enumerate(jobs):
            result = await trial(task, width, round_id)
            handle.write(json.dumps(result) + '\n')
            print(json.dumps(dict(progress=f'{n + 1}/{len(jobs)}', task=task['task_id'],
                                  width=width, round=round_id, solved=result['solved'],
                                  seconds=round(result['total'], 3))), flush=True)
    (output / 'COMPLETE').write_text(str(len(jobs)) + '\n')
    engine.shutdown_background_loop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
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
