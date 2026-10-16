"""Explicit engine commands and measurements for the owned single-GPU PD study."""
import json
import hashlib
import copy
import math
import re
import os
import signal
import subprocess
from contextlib import contextmanager
from pathlib import Path

GPU1 = 'GPU-4d9bef9d-631c-bb36-3ff4-589d6a184189'


def record_morphx_coverage(server_log: Path, directory: Path) -> dict:
    """Freeze model-process evidence; a prior standalone probe is insufficient."""
    raw = Path(server_log).read_bytes()
    names = re.findall(r'\[MorphX-llm\] phase=patched\b[^\r\n]*\bname=(\S+)',
                       raw.decode(errors='replace'))
    evidence = dict(status='observed' if names else 'unobserved',
        patched_launches_observed=len(names), patched_kernel_names=sorted(set(names)),
        snapshot_sha256=hashlib.sha256(raw).hexdigest(),
        scope='Model-process warmup/quality before load; partial kernel coverage '
              'does not prove PD overlap or a change to request admission.')
    with (Path(directory)/'morphx-coverage.log').open('xb') as out:
        out.write(raw)
    with (Path(directory)/'morphx-coverage.json').open('x') as out:
        json.dump(evidence, out, indent=2)
    return evidence


def snapshot_harness(source, destination, files):
    """Archive source bytes, so later workspace edits cannot erase provenance."""
    relatives=[Path(name) for name in files]
    if any(p.is_absolute() or '..' in p.parts for p in relatives):
        raise ValueError('harness source names must be relative without traversal')
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    hashes={}
    for relative in relatives:
        data=Path(_owned_path(Path(source),str(relative))).read_bytes()
        target=destination/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as out:out.write(data)
        hashes[str(relative)]=hashlib.sha256(data).hexdigest()
    return hashes


@contextmanager
def defer_interruptions():
    """Deliver a controller signal after an indivisible spawn/cleanup operation."""
    numbers=(signal.SIGINT,signal.SIGTERM,signal.SIGHUP)
    previous={number:signal.getsignal(number) for number in numbers}
    pending=[]
    def defer(number,frame):
        if previous[number]!=signal.SIG_IGN and not pending:
            pending.append(number)
    try:
        for number in numbers:signal.signal(number,defer)
        yield
    finally:
        for number,handler in previous.items():signal.signal(number,handler)
        if pending:signal.raise_signal(pending[0])


@defer_interruptions()
def stop_owned_group(process, timeout=10):
    """Clean a private group while its unreaped leader still anchors its ID."""
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    import time
    deadline=time.monotonic()+timeout
    while (os.waitid(os.P_PID,process.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT) is None
           and time.monotonic()<deadline):
        time.sleep(.02)
    # Leader exit does not establish worker exit. Kill remaining members BEFORE
    # wait() releases the leader and permits its numeric process group ID reuse.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def run_owned_command(command, env, output, timeout=None):
    """Run an external suite and clean its owned descendants on every exit."""
    import time
    process=None
    try:
        with defer_interruptions():
            process=subprocess.Popen(command,env=env,stdout=output,stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL,start_new_session=True)
        deadline=None if timeout is None else time.monotonic()+timeout
        while True:
            exited=os.waitid(os.P_PID,process.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT)
            if exited is not None:
                return exited.si_status if exited.si_code==os.CLD_EXITED else -exited.si_status
            if deadline is not None and time.monotonic()>=deadline:
                raise subprocess.TimeoutExpired(command,timeout)
            time.sleep(.02)
    finally:
        if process is not None:stop_owned_group(process)


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


def validate_case(case: dict, *, qualify_morphx=False) -> None:
    for key in ('id', 'engine', 'binary', 'model', 'quantization', 'gpu_uuid',
                'context_window', 'kv_capacity', 'concurrency', 'chunk', 'kv_dtype',
                'draft', 'morphx'):
        if key not in case:
            raise ValueError(f'missing {key}')
    if case['gpu_uuid'] != GPU1:
        raise ValueError('only owned GPU1 is permitted')
    if case['engine'] not in {'llama', 'ninfer', 'vllm', 'sglang'}:
        raise ValueError('unknown engine')
    for key in ('context_window', 'kv_capacity', 'concurrency', 'chunk'):
        if (key=='chunk' and case['engine']=='sglang' and case.get('pdmux_config')
                and case[key]==-1):
            continue
        if type(case[key]) is not int or case[key] <= 0:
            raise ValueError(f'positive integer {key} required')
    if type(case['draft']) is not int or case['draft'] < 0:
        raise ValueError('nonnegative draft required')
    if case['engine']=='sglang' and case['draft']:
        raise ValueError('SGLang speculative configuration is not yet qualified')
    if min(case['context_window'], case['kv_capacity']) < 131072 + 256:
        raise ValueError('128K input plus output must fit')
    if case['engine']=='ninfer' and case['context_window']>case['kv_capacity']:
        raise ValueError('NInfer context cannot exceed explicit KV capacity')
    if (case['engine'] == 'llama' and not case.get('unified_kv')
            and case['kv_capacity']//case['concurrency'] < 131072+256):
        raise ValueError('128K cannot fit the divided per-slot context')
    if case['morphx'].get('enabled') and case['morphx'].get('patched_launches', 0) <= 0 and not qualify_morphx:
        raise ValueError('MorphX needs observed patched kernel launches')
    if (case.get('cuda_graph') is False and
            case.get('compilation_config', {}).get('cudagraph_mode') not in (None, 0, 'NONE')):
        raise ValueError('CUDA graph policy conflicts with enforce-eager')


def _owned_path(root: Path, relative: str) -> str:
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('model and executable must remain on the HDD root')
    return str(path)


def validate_sglang_runtime(case: dict, info: dict) -> None:
    """Hybrid state reservations can clamp concurrency despite the CLI request."""
    workers=info.get('internal_states',[])
    context=info.get('context_length')
    if (type(context) is not int or context < case['context_window'] or len(workers)!=1
            or workers[0].get('effective_max_running_requests_per_dp')!=case['concurrency']
            or workers[0].get('memory_usage',{}).get('token_capacity',0)<case['context_window']):
        raise RuntimeError('SGLang resolved capacity does not preserve requested context/concurrency')
    if ('max_total_tokens' in case and
            workers[0]['memory_usage']['token_capacity']!=case['max_total_tokens']):
        raise RuntimeError('SGLang actual token pool differs from matched pool limit')


def engine_command(case: dict, root: Path, run_dir: Path, *, qualify_morphx=False) -> list[str]:
    validate_case(case, qualify_morphx=qualify_morphx)
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
        if case.get('cuda_graph') is False:
            args += ['--no-cuda-graph']
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
    if case['engine'] == 'sglang':
        args = common + ['-m', 'sglang.launch_server', '--model-path', model,
            '--served-model-name', 'qwen38-27b', '--host', '127.0.0.1', '--port', '18083',
            '--tp-size', '1', '--context-length', str(case['context_window']),
            '--max-running-requests', str(case['concurrency']),
            '--chunked-prefill-size', str(case['chunk']), '--kv-cache-dtype', case['kv_dtype'],
            '--mem-fraction-static', str(case.get('gpu_memory_utilization', .95)),
            '--enable-metrics', '--tool-call-parser', 'qwen3_coder',
            '--reasoning-parser', 'qwen3',
            '--default-chat-template-kwargs', '{"enable_thinking":false}']
        if case.get('text_only_via_hf_config'):
            args += ['--json-model-override-args', '{"language_model_only":true}']
        else:
            args += ['--language-model-only']
        if case.get('unified_memory'):
            args += ['--enable-unified-memory']
        if case.get('mixed_chunk', True):
            args += ['--enable-mixed-chunk']
        if case.get('cuda_graph') is False:
            args += ['--disable-prefill-cuda-graph', '--disable-decode-cuda-graph']
        if case.get('enable_cache_report'):
            args += ['--enable-cache-report']
        if case.get('disable_overlap_schedule'):
            args += ['--disable-overlap-schedule']
        if 'max_total_tokens' in case:
            if type(case['max_total_tokens']) is not int or case['max_total_tokens'] < case['context_window']:
                raise ValueError('actual token-pool limit must preserve full context')
            args += ['--max-total-tokens', str(case['max_total_tokens'])]
        if case.get('pdmux_config'):
            args += ['--enable-pdmux','--pdmux-config-path',_owned_path(root,case['pdmux_config'])]
        for field, flag in [('max_mamba_cache_size', '--max-mamba-cache-size'),
                            ('mamba_full_memory_ratio', '--mamba-full-memory-ratio'),
                            ('mamba_radix_cache_strategy', '--mamba-radix-cache-strategy'),
                            ('mamba_ssm_dtype', '--mamba-ssm-dtype'),
                            ('sm_group_num', '--sm-group-num'),
                            ('attention_backend', '--attention-backend'),
                            ('cuda_graph_max_bs_decode', '--cuda-graph-max-bs-decode')]:
            if field in case:
                args += [flag, str(case[field])]
        return args
    args = common + ['-m', 'vllm.entrypoints.openai.api_server', '--model', model,
        '--served-model-name', 'qwen38-27b', '--host', '127.0.0.1', '--port', '18083',
        '--tensor-parallel-size', '1', '--max-model-len', str(case['context_window']),
        '--max-num-seqs', str(case['concurrency']), '--max-num-batched-tokens',
        str(case['chunk']), '--enable-chunked-prefill', '--enable-prefix-caching',
        '--kv-cache-dtype', case['kv_dtype'], '--gpu-memory-utilization',
        str(case.get('gpu_memory_utilization', .95)), '--enable-auto-tool-choice',
        '--enable-prompt-tokens-details',
        '--tool-call-parser', 'qwen3_coder', '--reasoning-parser', 'qwen3',
        '--language-model-only', '--default-chat-template-kwargs',
        '{"enable_thinking":false}']
    if case.get('cuda_graph') is False:
        args += ['--enforce-eager']
    if case.get('async_scheduling') is False:
        args += ['--no-async-scheduling']
    if 'scheduler_reserve_full_isl' in case:
        if type(case['scheduler_reserve_full_isl']) is not bool:
            raise ValueError('scheduler_reserve_full_isl must be a boolean')
        args += ['--scheduler-reserve-full-isl' if case['scheduler_reserve_full_isl']
                 else '--no-scheduler-reserve-full-isl']
    if case.get('scheduler_class'):
        args += ['--scheduler-cls',case['scheduler_class']]
    if 'shutdown_timeout' in case:
        args += ['--shutdown-timeout',str(case['shutdown_timeout'])]
    if 'compilation_config' in case:
        args += ['--compilation-config', json.dumps(case['compilation_config'])]
    for field, flag in [('long_prefill_token_threshold','--long-prefill-token-threshold'),
                        ('num_gpu_blocks_override','--num-gpu-blocks-override'),
                        ('mamba_cache_mode','--mamba-cache-mode'),
                        ('mamba_block_size','--mamba-block-size'),
                        ('cpu_offload_gb','--cpu-offload-gb')]:
        if field in case:
            args += [flag,str(case[field])]
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
