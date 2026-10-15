"""Owned GPU1 cross-engine experiments; raw outputs never overwrite earlier runs."""
import argparse
import asyncio
import copy
import hashlib
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from inferx.concurrency import run_load, summarize_load
from inferx.pd_baselines import (GPU1, engine_command, parse_metrics, validate_case,
                                load_frozen_workloads, arrival_requests)
from inferx.service_ownership import RestorableService, process_identity, stop_matching_process

ENDPOINT = 'http://127.0.0.1:18083'


def get(path, endpoint=ENDPOINT):
    with urllib.request.urlopen(endpoint+path, timeout=5) as response:
        value = response.read().decode()
    if path == '/health' and not value.strip():
        return {'status':'ok'}
    return parse_metrics(value) if path == '/metrics' else json.loads(value)


def wait_idle(timeout=900):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        m = get('/metrics')
        keys = ['llamacpp:requests_processing', 'llamacpp:requests_deferred',
                'vllm:num_requests_running', 'vllm:num_requests_waiting']
        observed = [m[k] for k in keys if k in m]
        if observed and all(v == 0 for v in observed):
            return
        time.sleep(.2)
    raise RuntimeError('server did not drain or lacks observable queue metrics')


class OwnedEngine:
    def __init__(self, case, root, directory):
        self.case, self.root, self.directory = case, Path(root), Path(directory)
        self.configuration = case  # Existing independently validated code-quality gate interface.
        self.directory.mkdir(parents=True, exist_ok=False)
        self.args = engine_command(case, self.root, self.directory)
        self.proc = self.identity = self.log = self.telemetry = None

    def __enter__(self):
        try:
            self.start()
            return self
        except BaseException:
            self.stop()
            raise

    def start(self):
        # The parent controller independently verifies the GPU is free first.
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU1)
        if self.case['engine'] == 'ninfer':
            env['LD_LIBRARY_PATH'] = ':'.join([
                str(self.root/'toolchains/cuda-12.8.1/lib'),
                str(self.root/'toolchains/ninfer-deps/usr/lib/x86_64-linux-gnu'),
                env.get('LD_LIBRARY_PATH', '')])
        elif self.case['engine'] == 'llama':
            env['LD_LIBRARY_PATH'] = '/usr/local/cuda-12.6/lib64:'+env.get('LD_LIBRARY_PATH', '')
        env.update(self.case.get('environment', {}))
        if env.get('CUDA_VISIBLE_DEVICES') != GPU1:
            raise RuntimeError('case environment cannot replace owned GPU1')
        self.log = (self.directory/'server.log').open('x')
        self.proc = subprocess.Popen(self.args, env=env, stdin=subprocess.DEVNULL,
            stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic()+self.case.get('startup_timeout_s', 600)
        while time.monotonic() < deadline:
            code = self.proc.poll()
            if code is not None:
                raise RuntimeError(f'server startup exit {code}')
            try:
                if get('/health').get('status', 'ok') == 'ok':
                    self.identity = process_identity(self.proc.pid, Path(self.args[3]))
                    startup = dict(case=self.case, args=self.args, identity=self.identity,
                                   models=get('/v1/models'))
                    (self.directory/'startup.json').write_text(json.dumps(startup, indent=2))
                    telemetry = (self.directory/'gpu.csv').open('x')
                    self.telemetry = subprocess.Popen(['nvidia-smi', '-i', '1',
                        '--query-gpu=timestamp,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw',
                        '--format=csv,noheader,nounits', '-lms', '200'], stdout=telemetry,
                        stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
                    telemetry.close()
                    return
            except (OSError, ValueError):
                pass
            time.sleep(.5)
        raise RuntimeError('server startup timeout')

    def stop(self):
        if self.telemetry is not None:
            self.telemetry.terminate()
            self.telemetry.wait(timeout=5)
        if self.proc is not None:
            if self.identity is not None:
                stop_matching_process(self.identity, timeout=30)
            elif self.proc.poll() is None:
                # An unreaped live Popen anchors our own process group even during startup.
                os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=10)
        if self.log is not None:
            self.log.close()

    def __exit__(self, *_):
        self.stop()


def run_with_restoration(saved, restore, paths, pre_stop_check, measure):
    with RestorableService(saved, restore, paths, pre_stop_check=pre_stop_check):
        return measure()


def observe(server, name, specifications, warmups=()):
    directory = server.directory/name
    directory.mkdir()
    wait_idle()
    with (directory/'warmup.jsonl').open('x') as out:
        warm = asyncio.run(run_load(ENDPOINT, list(warmups), out, 'closed', server.case['concurrency']))
    if any(not r['summary']['ok'] or r.get('functional_ok') is False for r in warm):
        raise RuntimeError('warmup protocol/tool gate failed; raw output retained')
    wait_idle()
    metrics_before = get('/metrics')
    with (directory/'specifications.json').open('x') as f:
        json.dump(specifications, f)
    start = time.monotonic()
    with (directory/'requests.jsonl').open('x') as out:
        records = asyncio.run(run_load(ENDPOINT, specifications, out,
            'open' if all('offset_s' in r for r in specifications) else 'closed',
            server.case['concurrency']))
    if specifications and all('offset_s' in r for r in specifications):
        time.sleep(max(0, 48-(time.monotonic()-start)))
    wait_idle()
    span = time.monotonic()-start
    result = dict(group=name, case=server.case['id'], summary=summarize_load(records, span),
        metrics_before=metrics_before, metrics_after=get('/metrics'),
        raw_records='requests.jsonl')
    for kind, subset in [('short', [r for r in records if r.get('prompt_id') == 'tool-24576']),
                         ('long', [r for r in records if r.get('prompt_id') != 'tool-24576'])]:
        result[kind] = summarize_load(subset, span)
    result['completed_within_48s'] = sum(r['summary']['ok'] and r['finished_s'] <= 48 for r in records)
    (directory/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(event='group_done', group=name, case=server.case['id'],
        errors=result['summary']['errors'], long_ok=result['long']['protocol_ok'],
        short_p95=result['short']['latency_s']['p95'], span_s=span)), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--kernel-probe', action='store_true')
    parser.add_argument('--atomic-probe', action='store_true')
    parser.add_argument('--groups', nargs='+')
    parser.add_argument('--short-rate', type=float)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    root = Path(os.environ['INFERX_ROOT'])
    if '/' in args.run_id or args.repeat < 1:
        parser.error('simple run-id and positive repeat required')
    output = root/'pd-baselines/raw'/args.run_id
    output.mkdir(parents=True, exist_ok=False)
    cases = json.loads(Path(args.config).read_text())['cases']
    cases = [c for c in cases if args.cases is None or c['id'] in args.cases]
    if not cases:
        raise RuntimeError('no selected cases')
    for case in cases:
        validate_case(case)
    (output/'config.json').write_text(json.dumps(cases, indent=2))
    # Existing frozen helpers provide GPU identity checks and exact original argv restoration.
    from scripts import run_concurrency_campaign as previous
    previous.DIRECTORY = output
    saved = json.loads((root/'tuning/best-service.json').read_text())
    assert saved == json.loads((root/'context-agents/runtime.json').read_text())
    identity = process_identity(saved['pid'], previous.BINARY)
    assert all(saved.get(k) == v for k, v in identity.items()), 'stale original identity'
    saved['original_argv'] = Path(f'/proc/{saved["pid"]}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
    (output/'original-service.json').write_text(json.dumps(saved, indent=2))
    templates, grid = load_frozen_workloads(root/'concurrency-roofline')
    definitions = copy.deepcopy(grid['mixed_cases'])
    if args.groups is not None:
        if set(args.groups)-{d['name'] for d in definitions}:
            parser.error('unknown workload group')
        definitions = [d for d in definitions if d['name'] in args.groups]
    if args.short_rate is not None:
        if not 0 < args.short_rate <= 20:
            parser.error('short-rate must be positive and at most20RPS')
        for definition in definitions:
            definition['rate'] = args.short_rate
    metadata = dict(status='running', started_unix=time.time(), cases=[],
        repeat=args.repeat, probe_only=args.probe_only, controller_pid=os.getpid(),
        workload_definitions=definitions,
        harness_sha256={str(p):hashlib.sha256((root/'code-pd-baselines'/p).read_bytes()).hexdigest()
            for p in ['scripts/run_pd_baselines.py', 'inferx/pd_baselines.py',
                      'inferx/concurrency.py', 'inferx/service_ownership.py']})
    status = output/'campaign.json'
    def flush():
        status.write_text(json.dumps(metadata, indent=2))
    def measurement():
        if args.kernel_probe or args.atomic_probe:
            previous.gpu_free()
            env = dict(previous.ENV, SMSCHED_LLM_TRACE_LIMIT='32', SMSCHED_PROFILE='0')
            env['LD_LIBRARY_PATH'] = (str(root/'source/pd-baselines/gdrcopy/src')+':'
                                      +env['LD_LIBRARY_PATH'])
            probes=([('unhooked',None),('mapped-host','libcuda_hook_mapped.so'),
                     ('mapped-full','libcuda_mapped_full.so')] if args.atomic_probe
                else [('unhooked',None),('upstream-gdr','libcuda_hook.so'),
                      ('mapped-host','libcuda_hook_mapped.so')])
            for mode, hook in probes:
                if hook and not (root/'pd-baselines/builds'/hook).exists():
                    continue
                probe_env = dict(env)
                if hook:
                    probe_env['LD_PRELOAD'] = str(root/'pd-baselines/builds'/hook)
                if mode=='mapped-full':
                    probe_env['LD_PRELOAD'] += ':'+str(root/'pd-baselines/builds/libpreload.so')
                    probe_env['NEUTRINO_HOOK_DRIVER'] = str(root/'pd-baselines/builds'/hook)
                command=([str(root/'pd-baselines/builds'/('cuda18_bf16_atomic_morphx' if hook
                    else 'cuda18_bf16_atomic'))] if args.atomic_probe else
                    [str(root/'pd-baselines/builds/morphx_probe'),str(root/'pd-baselines/builds/morphx_probe.ptx')])
                with (output/f'{mode}.log').open('x') as log:
                    result = subprocess.run(command, env=probe_env,
                        stdout=log, stderr=subprocess.STDOUT, timeout=60)
                trace=(output/f'{mode}.log').read_text()
                patched=[line.split('name=',1)[-1] for line in trace.splitlines()
                         if '[MorphX-llm] phase=patched' in line]
                metadata['cases'].append(dict(case=mode, returncode=result.returncode,
                    patched_launches_observed=len(patched),patched_kernel_names=patched))
                flush()
            return
        for case in cases:
            for repetition in range(args.repeat):
                selected = [dict(name='probe-128k')] if args.probe_only else definitions
                for definition in selected:
                    label = f'{case["id"]}-r{repetition}-{definition["name"]}'
                    state = dict(case=case['id'], repetition=repetition,
                                 group=definition['name'], state='starting')
                    metadata['cases'].append(state)
                    flush()
                    try:
                        previous.gpu_free()
                        with OwnedEngine(case, root, output/label) as server:
                            previous.quality_gate(server, templates)
                            quality = json.loads((server.directory/'quality-result.json').read_text())
                            state['quality'] = quality
                            if quality.get('status') != 'passed':
                                raise RuntimeError('complete code quality gate failed')
                            short = templates['tool-24576']
                            seeds = [dict(id=f'seed-short-{i}', prompt_id='tool-24576',
                                payload=short['payload'], expected_tool=short['expected_tool'])
                                for i in range(case['concurrency'])]
                            if args.probe_only:
                                long = templates['cold-131072']
                                requests = [dict(id='probe-long', prompt_id='cold-131072',
                                    payload=long['payload'], expected_tool=long['expected_tool'])]
                            else:
                                requests = arrival_requests(definition, templates, grid, label)
                            observe(server, definition['name'], requests, seeds)
                            state['state'] = 'measured'
                    except Exception as exc:
                        state.update(state='failed', error=f'{type(exc).__name__}: {exc}')
                        print(json.dumps(state), flush=True)
                    finally:
                        flush()
    flush()
    try:
        run_with_restoration(saved, previous.restore,
            [root/'context-agents/runtime.json', root/'tuning/best-service.json'],
            lambda: previous.final_shutdown_check(saved), measurement)
        metadata['status'] = 'restored'
    except BaseException as exc:
        metadata.update(status='interrupted_or_failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        metadata['finished_unix'] = time.time()
        metadata['restored'] = (output/'restored-service.json').exists()
        flush()


if __name__ == '__main__':
    main()
