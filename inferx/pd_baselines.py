"""Explicit engine commands and measurements for the owned single-GPU PD study."""
import json
import hashlib
import copy
import math
import re
from pathlib import Path

GPU1 = 'GPU-4d9bef9d-631c-bb36-3ff4-589d6a184189'


def load_frozen_workloads(directory: Path) -> tuple[dict, dict]:
    manifest = json.loads((directory/'workload-manifest.json').read_text())
    for entry in manifest['files']:
        path = directory/entry['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'frozen workload hash mismatch: {path.name}')
    return (json.loads((directory/'workloads.json').read_text()),
            json.loads((directory/'grid.json').read_text()))


def arrival_requests(definition: dict, templates: dict, grid: dict, label: str) -> list[dict]:
    requests = []
    def append(name, identity, offset):
        template = templates[name]
        requests.append(dict(id=identity, prompt_id=name, offset_s=offset,
                             payload=copy.deepcopy(template['payload']),
                             expected_tool=template.get('expected_tool')))
    for i in range(int(definition['rate']*grid['arrival_duration_s'])):
        append('tool-24576', f'{label}-short-{i}', i/definition['rate'])
    if definition['long_mode']:
        for i, (offset, target) in enumerate(zip(grid['long_offsets_s'], grid['long_targets'])):
            append(f"{definition['long_mode']}-{target}", f'{label}-long-{i}', offset)
    return requests


def validate_case(case: dict) -> None:
    for key in ('id', 'engine', 'binary', 'model', 'quantization', 'gpu_uuid',
                'context_window', 'kv_capacity', 'concurrency', 'chunk', 'kv_dtype',
                'draft', 'morphx'):
        if key not in case:
            raise ValueError(f'missing {key}')
    if case['gpu_uuid'] != GPU1:
        raise ValueError('only owned GPU1 is permitted')
    if case['engine'] not in {'llama', 'ninfer', 'vllm'}:
        raise ValueError('unknown engine')
    for key in ('context_window', 'kv_capacity', 'concurrency', 'chunk'):
        if type(case[key]) is not int or case[key] <= 0:
            raise ValueError(f'positive integer {key} required')
    if type(case['draft']) is not int or case['draft'] < 0:
        raise ValueError('nonnegative draft required')
    if min(case['context_window'], case['kv_capacity']) < 131072 + 256:
        raise ValueError('128K input plus output must fit')
    if (case['engine'] == 'llama' and not case.get('unified_kv')
            and case['kv_capacity']//case['concurrency'] < 131072+256):
        raise ValueError('128K cannot fit the divided per-slot context')
    if case['morphx'].get('enabled') and case['morphx'].get('patched_launches', 0) <= 0:
        raise ValueError('MorphX needs observed patched kernel launches')


def _owned_path(root: Path, relative: str) -> str:
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('model and executable must remain on the HDD root')
    return str(path)


def engine_command(case: dict, root: Path, run_dir: Path) -> list[str]:
    validate_case(case)
    binary = _owned_path(root, case['binary'])
    model = _owned_path(root, case['model'])
    common = ['taskset', '-c', '28-35', binary]
    if case['engine'] == 'ninfer':
        args = common + [model, '--host', '127.0.0.1', '--port', '18083',
            '--model-id', 'qwen38-27b', '--max-context', str(case['context_window']),
            '--kv-capacity', str(case['kv_capacity']), '--kv-dtype', case['kv_dtype'],
            '--prefill-chunk', str(case['chunk']), '--default-max-tokens', '8192',
            '--max-concurrency', str(case['concurrency']), '--max-pending-requests', '16',
            '--pending-timeout-ms', '600000', '--log-stats-interval-ms', '1000',
            '--no-thinking', '--greedy', '--presence-penalty', '0', '--frequency-penalty', '0',
            '--request-log-jsonl', str(run_dir/'native.jsonl')]
        if case['draft']:
            args += ['--spec', 'mtp', '--draft-tokens', str(case['draft']), '--lm-head-draft']
        return args
    if case['engine'] == 'llama':
        args = common + ['--model', model, '--alias', 'qwen38-27b', '--host', '127.0.0.1',
            '--port', '18083', '--parallel', str(case['concurrency']),
            '--ctx-size', str(case['kv_capacity']), '--gpu-layers', 'all',
            '--split-mode', 'none', '--fit', 'off', '--flash-attn', 'on',
            '--cache-type-k', case['kv_dtype'], '--cache-type-v', case['kv_dtype'],
            '--threads', '8', '--threads-batch', '8', '--batch-size', str(case['chunk']),
            '--ubatch-size', str(min(case['chunk'], 512)), '--metrics', '--slots',
            '--no-webui', '--threads-http', '8', '--reasoning', 'off',
            '--chat-template-kwargs', '{"enable_thinking":false}', '--cont-batching',
            '--no-context-shift']
        if case.get('unified_kv'):
            args += ['--kv-unified', '--kv-unified-per-slot', str(case['context_window'])]
        if case['draft']:
            args += ['--spec-type', 'draft-mtp', '--spec-draft-n-max', str(case['draft'])]
            if case.get('draft_kv_dtype'):
                args += ['--spec-draft-type-k', case['draft_kv_dtype'],
                         '--spec-draft-type-v', case['draft_kv_dtype']]
        return args
    args = common + ['-m', 'vllm.entrypoints.openai.api_server', '--model', model,
        '--served-model-name', 'qwen38-27b', '--host', '127.0.0.1', '--port', '18083',
        '--tensor-parallel-size', '1', '--max-model-len', str(case['context_window']),
        '--max-num-seqs', str(case['concurrency']), '--max-num-batched-tokens',
        str(case['chunk']), '--enable-chunked-prefill', '--enable-prefix-caching',
        '--kv-cache-dtype', case['kv_dtype'], '--gpu-memory-utilization',
        str(case.get('gpu_memory_utilization', .95)), '--enable-auto-tool-choice',
        '--tool-call-parser', 'qwen3_xml', '--reasoning-parser', 'qwen3',
        '--limit-mm-per-prompt', '{"image":0,"video":0}']
    if case['draft']:
        args += ['--speculative-config', json.dumps(
            {'method':'mtp', 'num_speculative_tokens':case['draft']})]
    return args


def parse_metrics(text: str) -> dict[str, float]:
    metrics = {}
    sample = re.compile(r'^([\w:]+)(?:\{[^\n]*\})?\s+([^\s]+)(?:\s+[^\s]+)?$')
    for line in text.splitlines():
        match = sample.fullmatch(line.strip())
        if not match:
            continue
        try:
            value = float(match.group(2))
        except ValueError:
            continue
        if math.isfinite(value):
            metrics[match.group(1)] = metrics.get(match.group(1), 0) + value
    return metrics
