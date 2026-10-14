"""Owned GPU1 experiment controller with finally restoration of the user's service."""
import argparse
import asyncio
import copy
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from inferx.concurrency import run_load, summarize_load, summarize_native
from inferx.context_agents import verify_process_identity
from inferx.llama_client import extract_code
from inferx.service_ownership import RestorableService, process_identity, refreshed_manifest, stop_matching_process
from inferx.verifier import verify

ROOT = Path(os.environ['INFERX_ROOT'])
DIRECTORY = ROOT/'concurrency-roofline'
BINARY = ROOT/'source/ninfer-build-sm89/apps/ninfer-serve'
MODEL = ROOT/'models/qwen38-ninfer/qwen3_8_27b.ninfer'
GPU = 'GPU-4d9bef9d-631c-bb36-3ff4-589d6a184189'
ENDPOINT = 'http://127.0.0.1:18083'
ENV = {**os.environ, 'CUDA_VISIBLE_DEVICES': GPU,
    'LD_LIBRARY_PATH': str(ROOT/'toolchains/cuda-12.8.1/lib')+':'+
        str(ROOT/'toolchains/ninfer-deps/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH', '')}


def get(path, endpoint=ENDPOINT):
    with urllib.request.urlopen(endpoint+path, timeout=5) as response:
        data = response.read().decode()
        return json.loads(data) if path != '/metrics' else dict(line.split() for line in data.splitlines())


def wait_idle(endpoint=ENDPOINT, timeout=900):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        metrics = get('/metrics', endpoint)
        if float(metrics['llamacpp:requests_processing']) == 0 and float(metrics['llamacpp:requests_deferred']) == 0:
            time.sleep(.1)
            return
        time.sleep(.2)
    raise RuntimeError('server did not drain; refusing next measurement')


def gpu_memory():
    return [int(x.strip()) for x in subprocess.check_output(['nvidia-smi', '-i', '1',
        '--query-gpu=memory.used,memory.free', '--format=csv,noheader,nounits'], text=True).split(',')]


def gpu_processes():
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                   '--format=csv,noheader'], text=True)
    return {(int(row.split(',')[0]), row.split(',')[1].strip())
            for row in text.splitlines() if row.strip()}


def final_shutdown_check(saved):
    processes = gpu_processes()
    if ((saved['pid'], GPU) not in processes or
            (2504785, 'GPU-a66b08e8-8b65-0faf-22e2-b951a97b2b03') not in processes):
        raise RuntimeError('GPU process ownership changed; refusing shutdown')
    if get('/health', 'http://127.0.0.1:18080')['status'] != 'ok':
        raise RuntimeError('GPU0 service health changed; refusing shutdown')
    metrics = get('/metrics', 'http://127.0.0.1:18081')
    if (float(metrics['llamacpp:requests_processing']) != 0 or
            float(metrics['llamacpp:requests_deferred']) != 0):
        raise RuntimeError('original service became busy; refusing shutdown')


def gpu_free(timeout=45):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if not any(uuid == GPU for _, uuid in gpu_processes()) and gpu_memory()[0] <= 512:
            return
        time.sleep(.2)
    raise RuntimeError('GPU1 still occupied; refusing to replace a foreign process')


def command(configuration, request_log, grid):
    args = ['taskset', '-c', '28-35', str(BINARY), str(MODEL), '--host', '127.0.0.1',
        '--port', '18083', '--model-id', 'qwen38-27b', '--max-context',
        str(min(grid['max_context'], configuration.get('kv_capacity', grid['kv_capacity']))),
        '--kv-capacity', str(configuration.get('kv_capacity', grid['kv_capacity'])),
        '--kv-dtype', grid['kv_dtype'], '--prefill-chunk', str(grid['prefill_chunk']),
        '--default-max-tokens', '8192', '--max-concurrency', str(configuration['concurrency']),
        '--max-pending-requests', '16', '--pending-timeout-ms', '600000',
        '--log-stats-interval-ms', '1000', '--no-thinking', '--greedy', '--presence-penalty', '0',
        '--frequency-penalty', '0', '--request-log-jsonl', str(request_log)]
    if configuration['draft']:
        args += ['--spec', 'mtp', '--draft-tokens', str(configuration['draft']), '--lm-head-draft']
    return args


class Server:
    def __init__(self, configuration, grid, prefix=None, directory=None):
        self.configuration, self.grid = configuration, grid
        self.directory = directory or DIRECTORY/configuration['id']
        self.directory.mkdir(exist_ok=False)
        self.requests = self.directory/'native.jsonl'
        self.args = (prefix or []) + command(configuration, self.requests, grid)
        self.log = (self.directory/'server.log').open('x')
        self.proc = self.telemetry = self.identity = None
        self.launched_at = time.monotonic()

    def start(self):
        gpu_free()
        self.proc = subprocess.Popen(self.args, stdin=subprocess.DEVNULL,
            stdout=self.log, stderr=subprocess.STDOUT, env=ENV, start_new_session=True)
        deadline = time.monotonic()+240
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f'server startup exit {self.proc.returncode}')
            try:
                if get('/health')['status'] == 'ok':
                    break
            except Exception:
                pass
            time.sleep(.5)
        else:
            raise RuntimeError('server startup timeout')
        # A profiler wrapper may be the parent. Select only the process in our owned group.
        candidates = []
        for pid, uuid in gpu_processes():
            if uuid != GPU:
                continue
            try:
                if os.getpgid(pid) == self.proc.pid:
                    candidates.append(process_identity(pid, BINARY))
            except (OSError, RuntimeError):
                pass
        if len(candidates) != 1:
            raise RuntimeError('GPU1 process ownership ambiguous')
        self.identity = candidates[0]
        memory = gpu_memory()
        metadata = dict(configuration=self.configuration, args=self.args, identity=self.identity,
                        memory_used_mib=memory[0], memory_free_mib=memory[1], model=get('/v1/models'))
        (self.directory/'startup.json').write_text(json.dumps(metadata, indent=2))
        if memory[1] < 1024:
            raise RuntimeError(f'headroom below1024MiB: {memory[1]}MiB')
        telemetry = (self.directory/'gpu.csv').open('x')
        self.telemetry = subprocess.Popen(['nvidia-smi', '-i', '1',
            '--query-gpu=timestamp,memory.used,memory.free,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,power.draw,temperature.gpu',
            '--format=csv,noheader,nounits', '-lms', '200'], stdout=telemetry,
            stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        telemetry.close()
        return metadata

    def stop(self):
        if self.telemetry:
            self.telemetry.terminate()
            self.telemetry.wait(timeout=5)
        if self.identity and verify_process_identity(self.identity['pid'], self.identity['proc_start_ticks'], BINARY):
            stop_matching_process(self.identity)
        if self.proc:
            try:
                self.proc.wait(timeout=40)
            except subprocess.TimeoutExpired:
                # An unreaped live parent anchors this owned process group.
                os.killpg(self.proc.pid, signal.SIGTERM)
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                    self.proc.wait()
        self.log.close()


def native_since(path, offset, started_ms, ended_ms):
    with path.open() as f:
        f.seek(offset)
        events = [json.loads(line) for line in f if line.strip()]
    return [e for e in events if e.get('event') != 'throughput' or
            (e['timestamp_unix_ms']-1000*e['interval_seconds'] >= started_ms
             and e['timestamp_unix_ms'] <= ended_ms)]


def snapshots(names, number, templates, prefix):
    return [dict(id=f'{prefix}-{i}', prompt_id=names[i%len(names)],
                 payload=templates[names[i%len(names)]]['payload'],
                 expected_tool=templates[names[i%len(names)]].get('expected_tool')) for i in range(number)]


def group(server, name, requests, warm_names, templates, mode='closed', duration=0, write_index=True):
    directory = server.directory/name
    directory.mkdir(exist_ok=False)
    c = server.configuration['concurrency']
    warm = snapshots(warm_names, c*len(warm_names), templates, name+'-warm')
    with (directory/'warmup.jsonl').open('x') as out:
        warm_rows = asyncio.run(run_load(ENDPOINT, warm, out, mode='closed', workers=c))
    if not all(r['summary']['ok'] for r in warm_rows):
        raise RuntimeError('warmup protocol failure: '+str([r['summary']['errors'] for r in warm_rows]))
    wait_idle()
    offset = server.requests.stat().st_size
    started_ms, started = time.time()*1000, time.perf_counter()
    with (directory/'requests.jsonl').open('x') as out:
        rows = asyncio.run(run_load(ENDPOINT, requests, out, mode=mode, workers=c))
    if time.perf_counter()-started < duration:
        time.sleep(duration-(time.perf_counter()-started))
    wait_idle()
    span, ended_ms = time.perf_counter()-started, time.time()*1000
    native = native_since(server.requests, offset, started_ms, ended_ms)
    (directory/'native.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in native))
    record = dict(case=server.configuration['id'], group=name, mode=mode, workers=c,
        closed_ordering='producer-affinity' if any('producer' in r for r in requests) else 'shared-queue',
        requested=len(requests), duration_s=duration, start_unix_ms=started_ms, end_unix_ms=ended_ms,
        workload_ids=sorted({r['prompt_id'] for r in requests}),
        relative_path=str(directory.relative_to(DIRECTORY)), summary=summarize_load(rows, span),
        native=summarize_native(native), profiler=not write_index)
    (directory/'result.json').write_text(json.dumps(record, indent=2))
    if write_index:
        with (DIRECTORY/'groups.jsonl').open('a') as out:
            out.write(json.dumps(record)+'\n')
    print(json.dumps(dict(event='group_done', case=record['case'], group=name,
        offered=record['summary']['offered'], errors=record['summary']['errors'],
        tool_valid=record['summary']['tool_valid'],
        aggregate_tokens_per_s=record['summary']['aggregate_tokens_per_s'], span_s=span)), flush=True)
    return record


def quality_gate(server, templates):
    spec = copy.deepcopy(templates['code-512'])
    spec['payload']['max_tokens'] = 1024
    with (server.directory/'quality.jsonl').open('x') as out:
        rows = asyncio.run(run_load(ENDPOINT, [dict(id='quality-code', payload=spec['payload'])],
                                    out, mode='closed', workers=1))
    tests = '''def check(f):
    assert f([])==[]
    data=[(6,3),(2,1),(2,3),(10,10)]
    assert [tuple(x) for x in f(data)]==[(1,6),(10,10)]
    assert data==[(6,3),(2,1),(2,3),(10,10)]
    assert [tuple(x) for x in f([(1,1),(1,1)])]==[(1,1)]
'''
    row = rows[0]
    if not row['summary']['ok'] or row['summary']['truncated']:
        result = dict(status='invalid_or_truncated', errors=row['summary']['errors'])
    else:
        result = asyncio.run(verify(extract_code(row['summary']['text']), tests, 'merge_intervals', 3, None))
    (server.directory/'quality-result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(event='quality', case=server.configuration['id'], result=result)), flush=True)


def measure_case(server, templates, grid):
    c, draft = server.configuration['concurrency'], server.configuration['draft']
    if server.configuration.get('near256_only'):
        for kind in ['code', 'prose']:
            name = kind+'-253952'
            group(server, 'near256-'+kind, snapshots([name], 4, templates, name), [name], templates)
        return
    for kind in ['code', 'prose', 'tool']:
        name = kind+'-512'
        group(server, 'scaling-'+kind, snapshots([name], 12, templates, name), [name], templates)
    capacity = grid['capacity_targets'][str(c)]
    common = 15360 if c == 8 else grid['common_long_target']
    for kind in ['code', 'prose']:
        name = f'{kind}-{common}'
        group(server, 'uniform-long-'+kind, snapshots([name], 8, templates, name), [name], templates)
    names = [f'code-{common}', f'prose-{common}']
    requests = snapshots(names, 16, templates, 'mixed-long')
    if c > 1:
        for i, request in enumerate(requests):
            request['producer'] = i % c
    else:
        requests.sort(key=lambda request: request['prompt_id'])
    group(server, 'mixed-long', requests, names, templates)
    names = ['code-512', f'prose-{capacity}']
    requests = snapshots(names, 16, templates, 'mixed-context')
    if c > 1:
        for i, request in enumerate(requests):
            request['producer'] = i % c
    else:
        # C1 is a serial weighted control. Avoid alternating 192K and short histories.
        requests.sort(key=lambda request: request['prompt_id'])
    group(server, 'mixed-context', requests, names, templates)
    if c in [1,2]:
        for kind in ['code','prose']:
            name = f'{kind}-{capacity}'
            group(server, 'uniform-capacity-'+kind, snapshots([name], 4, templates, name), [name], templates)
    if draft in [0,7] and c in [1,4]:
        for definition in grid['mixed_cases']:
            name = definition['name']
            requests = snapshots(['tool-24576'], int(definition['rate']*grid['arrival_duration_s']),
                                 templates, name+'-short')
            for i, request in enumerate(requests):
                request['offset_s'] = i/definition['rate']
            if definition['long_mode']:
                for offset, target in zip(grid['long_offsets_s'], grid['long_targets']):
                    request = snapshots([f"{definition['long_mode']}-{target}"], 1, templates, name+'-long')[0]
                    request['offset_s'] = offset
                    requests.append(request)
            group(server, 'contention-'+name, requests, ['tool-24576'], templates,
                  mode='open', duration=grid['arrival_duration_s'])
    quality_gate(server, templates)


def restore(saved):
    gpu_free()
    log = (ROOT/'context-agents/agent-server.log').open('a')
    proc = subprocess.Popen(['taskset','-c','28-35',*saved['original_argv']], env=ENV,
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    log.close()
    for _ in range(480):
        if proc.poll() is not None:
            raise RuntimeError(f'restored server startup exited {proc.returncode}')
        try:
            if get('/health','http://127.0.0.1:18081')['status']=='ok':
                assert get('/v1/models','http://127.0.0.1:18081')['data'][0]['context_window']==262144
                identity = process_identity(proc.pid, BINARY)
                assert (proc.pid, GPU) in gpu_processes()
                manifest = refreshed_manifest({k:v for k,v in saved.items() if k!='original_argv'}, identity)
                (DIRECTORY/'restored-service.json').write_text(json.dumps(manifest, indent=2))
                print(json.dumps(dict(event='service_restored', pid=proc.pid, context=262144)), flush=True)
                return manifest
        except (OSError, RuntimeError):
            pass
        time.sleep(.5)
    raise RuntimeError('restored service failed health validation')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore-only', action='store_true')
    args = parser.parse_args()
    if args.restore_only:
        saved = json.loads((DIRECTORY/'original-service.json').read_text())
        manifest = restore(saved)
        for path in [ROOT/'context-agents/runtime.json', ROOT/'tuning/best-service.json']:
            path.write_text(json.dumps(manifest, indent=2))
        return
    assert not (DIRECTORY/'campaign.json').exists(), 'campaign exists; do not overwrite measurements'
    grid = json.loads((DIRECTORY/'grid.json').read_text())
    templates = json.loads((DIRECTORY/'workloads.json').read_text())
    for entry in json.loads((DIRECTORY/'workload-manifest.json').read_text())['files']:
        assert hashlib.sha256((DIRECTORY/entry['file']).read_bytes()).hexdigest()==entry['sha256']
    saved = json.loads((ROOT/'context-agents/runtime.json').read_text())
    assert verify_process_identity(saved['pid'], saved['proc_start_ticks'], BINARY), 'stale original manifest'
    saved.update(process_identity(saved['pid'], BINARY))
    saved['original_argv'] = [x.decode() for x in Path(f"/proc/{saved['pid']}/cmdline").read_bytes().split(b'\0') if x]
    assert (saved['pid'], GPU) in gpu_processes()
    assert (2504785,'GPU-a66b08e8-8b65-0faf-22e2-b951a97b2b03') in gpu_processes()
    assert get('/health','http://127.0.0.1:18080')['status']=='ok'
    wait_idle('http://127.0.0.1:18081', timeout=180)
    (DIRECTORY/'original-service.json').write_text(json.dumps(saved, indent=2))
    metadata = dict(status='running', started_unix=time.time(), controller_pid=os.getpid(),
        harness_sha256={name:hashlib.sha256((ROOT/'code-tuning'/name).read_bytes()).hexdigest()
            for name in ['scripts/run_concurrency_campaign.py','scripts/measure_device_bandwidth.py',
                'scripts/profile_concurrency.py','scripts/prepare_concurrency_workloads.py',
                'inferx/concurrency.py','inferx/service_ownership.py']},
        binary_sha256=hashlib.sha256(BINARY.read_bytes()).hexdigest(),
        engine_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'source/ninfer-4090',text=True).strip(),
        grid_sha256=hashlib.sha256((DIRECTORY/'grid.json').read_bytes()).hexdigest(),
        hardware_counter_permission=[s for s in Path('/proc/driver/nvidia/params').read_text().splitlines() if 'Profil' in s],
        model=str(MODEL), binary=str(BINARY), cases=[])
    (DIRECTORY/'campaign.json').write_text(json.dumps(metadata, indent=2))
    def interrupted(number, frame):
        raise KeyboardInterrupt(f'controller received signal{number}')
    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
        signal.signal(sig, interrupted)
    try:
        with RestorableService(saved, restore, [ROOT/'context-agents/runtime.json', ROOT/'tuning/best-service.json'],
                pre_stop_check=lambda: final_shutdown_check(saved)):
            gpu_free()
            measurement = subprocess.run([sys.executable,'-m','scripts.measure_device_bandwidth'],
                cwd=ROOT/'code-tuning', env=ENV, stdin=subprocess.DEVNULL, capture_output=True, text=True)
            (DIRECTORY/'bandwidth.log').write_text(measurement.stdout+measurement.stderr)
            print(json.dumps(dict(event='bandwidth_done', returncode=measurement.returncode)), flush=True)
            for configuration in grid['configs']:
                print(json.dumps(dict(event='case_start', **configuration)), flush=True)
                server = Server(configuration, grid)
                result = dict(configuration=configuration, started_unix=time.time())
                try:
                    result['startup'] = server.start()
                    measure_case(server, templates, grid)
                    result['status'] = 'completed'
                except BaseException as exc:
                    result.update(status='interrupted' if not isinstance(exc,Exception) else 'infrastructure_error',
                                  error=f'{type(exc).__name__}: {exc}')
                    print(json.dumps(dict(event='case_error', **result)), flush=True)
                    if not isinstance(exc,Exception):
                        raise
                finally:
                    server.stop()
                    result['finished_unix'] = time.time()
                    (server.directory/'case.json').write_text(json.dumps(result, indent=2))
                    metadata['cases'].append(result)
                    (DIRECTORY/'campaign.json').write_text(json.dumps(metadata, indent=2))
            from scripts.profile_concurrency import run_profiles
            metadata['profiles'] = run_profiles(grid, templates)
            metadata['status'] = 'measured'
        # The finally restoration is complete before testing the original client routes.
        with (DIRECTORY/'restored-openai.jsonl').open('x') as out:
            rows = asyncio.run(run_load('http://127.0.0.1:18081',
                snapshots(['tool-512'], 1, templates, 'restored'), out, mode='closed', workers=1))
        assert rows[0]['summary']['ok'] and rows[0]['functional_ok']
        tool = templates['tool-512']['payload']['tools'][0]['function']
        body = dict(model='qwen38-27b', max_tokens=128, messages=[dict(role='user',
            content='Call record_status with status="ready", task="status-probe", attempt=3.')],
            tools=[dict(name=tool['name'], description=tool['description'], input_schema=tool['parameters'])],
            tool_choice=dict(type='tool', name=tool['name']))
        request = urllib.request.Request('http://127.0.0.1:18081/v1/messages', json.dumps(body).encode(),
            {'Content-Type':'application/json','anthropic-version':'2023-06-01','x-api-key':'inferx-local'})
        response = json.load(urllib.request.urlopen(request, timeout=60))
        (DIRECTORY/'restored-anthropic.json').write_text(json.dumps(response, indent=2))
        calls = [x for x in response['content'] if x.get('type')=='tool_use']
        assert len(calls)==1 and calls[0]['input']==templates['tool-512']['expected_tool']['arguments']
    except BaseException as exc:
        metadata.update(status='interrupted_or_failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        metadata['finished_unix'] = time.time()
        metadata['restored'] = (DIRECTORY/'restored-service.json').exists()
        metadata['gpu0_healthy'] = get('/health','http://127.0.0.1:18080')['status']=='ok'
        (DIRECTORY/'campaign.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
