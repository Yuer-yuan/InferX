"""Freeze synthetic serving snapshots and the grid before any GPU1 measurement."""
import copy
import hashlib
import json
import os
import urllib.request
from pathlib import Path

root = Path(os.environ['INFERX_ROOT'])
directory = root / 'concurrency-roofline'
directory.mkdir(exist_ok=True)
system = 'You are a precise local coding assistant. Answer directly without hidden reasoning.'
tool = dict(type='function', function=dict(name='record_status',
    description='Record the explicitly requested status, task identifier and attempt number.',
    parameters=dict(type='object', properties=dict(status=dict(type='string'),
        task=dict(type='string'), attempt=dict(type='integer')),
        required=['status', 'task', 'attempt'], additionalProperties=False)))
expected = dict(name='record_status', arguments=dict(status='ready', task='status-probe', attempt=3))
questions = {
    'code': 'Implement Python merge_intervals(intervals) that merges overlapping closed integer intervals. '
            'Handle empty input, reversed endpoints and unsorted input without mutating it. '
            'Return only a Python code block with type hints and a docstring explaining the algorithm.',
    'prose': 'Explain how prefix caching, continuous batching, and multi-token prediction interact in a local '
             'code assistant. Discuss latency and memory, give one case where each optimization hurts, '
             'and provide a compact deployment checklist. Use about 250 words.',
    'tool': 'Call record_status exactly once with status="ready", task="status-probe", attempt=3. '
            'Do not answer with prose.'}


def registry(records, prefix='R'):
    return 'Configuration registry:\n' + '\n'.join(
        f'{prefix}{i:05d}: value={hashlib.sha256(f"registry:{prefix}:{i}".encode()).hexdigest()[:16]}; '
        f'timeout={100+i%1000}; owner=team_{i%37}.' for i in range(records))


def payload(messages, kind, output=None):
    p = dict(model='qwen38-27b', messages=messages, stream=True,
             stream_options=dict(include_usage=True), temperature=0, top_p=1, seed=42,
             presence_penalty=0, frequency_penalty=0,
             max_tokens=output or (128 if kind == 'tool' else 192))
    if kind == 'tool':
        p.update(tools=[tool], tool_choice=dict(type='function', function=dict(name='record_status')))
    return p


def token_count(p):
    body = {k: p[k] for k in ['messages', 'tools'] if k in p}
    body['chat_template_kwargs'] = dict(enable_thinking=False)
    def post(route, value):
        request = urllib.request.Request('http://127.0.0.1:18080' + route,
            json.dumps(value).encode(), {'Content-Type': 'application/json'})
        return json.load(urllib.request.urlopen(request, timeout=120))
    rendered = post('/apply-template', body)['prompt']
    return len(post('/tokenize', dict(content=rendered, add_special=False))['tokens'])


def fit(builder, target):
    low, high = 0, max(32, target//20)
    while token_count(builder(high)) <= target:
        high *= 2
    while low < high:
        mid = (low+high+1)//2
        if token_count(builder(mid)) <= target:
            low = mid
        else:
            high = mid-1
    p = builder(low)
    actual = token_count(p)
    assert actual <= target and actual >= target-100
    return p, low, actual


templates = {}
for length in [512, 12288]:
    for kind in ['code', 'prose', 'tool']:
        def builder(records):
            return payload([dict(role='system', content=system),
                dict(role='user', content=registry(records)),
                dict(role='assistant', content='Registry loaded.'),
                dict(role='user', content=questions[kind])], kind)
        p, records, actual = fit(builder, length)
        name = f'{kind}-{length}'
        templates[name] = dict(id=name, payload=p, reference_prompt_tokens=actual,
            registry_records=records, expected_tool=expected if kind == 'tool' else None)

base = templates['tool-12288']['payload']['messages'][:-1]
for target in [16384, 24576, 32768]:
    prefix = f'D{target}_'
    answer = hashlib.sha256(f'registry:{prefix}:0'.encode()).hexdigest()[:16]
    wanted = dict(name='record_status', arguments=dict(status='ready', task=answer, attempt=1))
    def builder(records):
        return payload(copy.deepcopy(base) + [dict(role='user', content=registry(records, prefix) +
            f'\nLook up the value of {prefix}00000. Call record_status with status="ready", '
            'task equal to that exact value, attempt=1. No prose.')], 'tool')
    p, records, actual = fit(builder, target)
    templates[f'resume-{target}'] = dict(id=f'resume-{target}', payload=p,
        reference_prompt_tokens=actual, registry_records=records, expected_tool=wanted)
    cold = copy.deepcopy(p)
    cold['messages'][0]['content'] = f'Independent cold snapshot {target}. ' + system
    cold_actual = token_count(cold)
    assert abs(cold_actual-actual) < 32
    templates[f'cold-{target}'] = dict(id=f'cold-{target}', payload=cold,
        reference_prompt_tokens=cold_actual, registry_records=records, expected_tool=wanted)

configs = [dict(id=f'd{draft}-c{concurrency}', draft=draft, concurrency=concurrency,
               exploratory=False) for concurrency in [1, 2, 4] for draft in [0, 4, 7]]
configs += [dict(id=f'd{draft}-c8', draft=draft, concurrency=8, exploratory=True) for draft in [0, 7]]
grid = dict(version=1, endpoint='http://127.0.0.1:18083', max_context=65536,
            kv_capacity=131072, kv_dtype='rk4v4-e8', prefill_chunk=2048,
            configs=configs, pure_requests_per_class=12, uniform_long_requests=8,
            mixed_requests=16, arrival_duration_s=48,
            mixed_cases=[dict(name='short-control', rate=2.5, long_mode=None),
                         dict(name='cold-low', rate=.5, long_mode='cold'),
                         dict(name='cold-high', rate=2.5, long_mode='cold'),
                         dict(name='resume-high', rate=2.5, long_mode='resume')],
            long_offsets_s=[0,16,32], long_targets=[16384,24576,32768])
for name, value in [('workloads.json', templates), ('grid.json', grid)]:
    with (directory/name).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
manifest = dict(scope='Synthetic serving snapshots; no live agent outcome claim.',
    tokenizer_reference='GPU0 read-only llama.cpp apply-template/tokenize; NInfer usage authoritative',
    files=[dict(file=name, sha256=hashlib.sha256((directory/name).read_bytes()).hexdigest(),
                bytes=(directory/name).stat().st_size) for name in ['workloads.json', 'grid.json']],
    templates={k:{key:value for key,value in v.items() if key!='payload'} for k,v in templates.items()})
(directory/'workload-manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest['templates'], indent=2))
