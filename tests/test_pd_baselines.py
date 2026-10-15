"""Contracts that prevent silently measuring a different PD workload."""
import copy
from pathlib import Path

import pytest

from inferx import pd_baselines as pd


def case(engine='llama'):
    return dict(id='llama-c4', engine=engine, binary='source/llama/bin/llama-server',
                model='models/qwen38/model.gguf', quantization='UD-Q4_K_M',
                gpu_uuid='GPU-4d9bef9d-631c-bb36-3ff4-589d6a184189',
                context_window=212992 if engine=='ninfer' else 262144,
                kv_capacity=212992, concurrency=4,
                chunk=2048, kv_dtype='q4_0', draft=0, unified_kv=True,
                morphx={'enabled': False, 'patched_launches': 0})


def test_llama_preserves_long_window_with_shared_capacity(tmp_path):
    c = case()
    pd.validate_case(c)
    args = pd.engine_command(c, tmp_path, tmp_path/'run')
    assert args[args.index('--ctx-size')+1] == '212992'
    assert args[args.index('--kv-unified-per-slot')+1] == '262144'
    assert '--kv-unified' in args and '--no-context-shift' in args
    assert '--cont-batching' in args
    c['unified_kv'] = False
    with pytest.raises(ValueError, match='128K'):
        pd.validate_case(c)


def test_case_rejects_foreign_gpu_and_unobserved_morphx():
    c = case()
    c['gpu_uuid'] = 'GPU-a66b08e8-8b65-0faf-22e2-b951a97b2b03'
    with pytest.raises(ValueError, match='GPU1'):
        pd.validate_case(c)
    c = case()
    c['morphx'] = {'enabled': True, 'patched_launches': 0}
    with pytest.raises(ValueError, match='patched'):
        pd.validate_case(c)
    c['morphx']['patched_launches'] = 5
    pd.validate_case(c)
    del c['quantization']
    with pytest.raises(ValueError, match='quantization'):
        pd.validate_case(c)


def test_serving_measurement_rejects_declared_morphx_without_model_runtime_trace(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from scripts import run_pd_baselines as runner
    c=case();c['morphx']={'enabled':True,'patched_launches':5}
    (tmp_path/'server.log').write_text('[MorphX-llm] phase=capture-bypass name=native_kernel\n')
    server=SimpleNamespace(case=c,directory=tmp_path)
    monkeypatch.setattr(runner,'wait_idle',lambda:pytest.fail('measurement reached without model coverage'))
    with pytest.raises(RuntimeError,match='runtime coverage'):
        runner.observe(server,'cold-low',[])
    assert __import__('json').loads((tmp_path/'cold-low/morphx-coverage.json').read_text())['status']=='unobserved'


def test_morphx_model_coverage_snapshot_is_independent_of_later_log_writes(tmp_path):
    source=tmp_path/'server.log'
    raw=(b'[MorphX-llm] phase=patched grid=24 name=gdn_kernel\n'
         b'[MorphX-llm] phase=capture-bypass name=native_kernel\n'
         b'[MorphX-llm] phase=patched grid=64 name=gemm_kernel\n')
    source.write_bytes(raw)
    out=tmp_path/'coverage';out.mkdir()
    evidence=pd.record_morphx_coverage(source,out)
    assert evidence['status']=='observed' and evidence['patched_launches_observed']==2
    assert evidence['patched_kernel_names']==['gdn_kernel','gemm_kernel']
    source.write_bytes(raw+b'later runtime log\n')
    import hashlib
    assert evidence['snapshot_sha256']==hashlib.sha256((out/'morphx-coverage.log').read_bytes()).hexdigest()


def test_models_and_commands_are_owned_and_context_is_real(tmp_path):
    c = case('ninfer')
    c['kv_dtype'] = 'rk4v4-e8'
    c['binary'] = 'source/ninfer/apps/ninfer-serve'
    args = pd.engine_command(c, tmp_path, tmp_path/'run')
    assert args[0:3] == ['taskset', '-c', '28-35']
    assert '--no-thinking' in args
    assert args[args.index('--kv-capacity')+1] == '212992'
    assert '--spec' not in args
    c['model'] = '../foreign-model'
    with pytest.raises(ValueError, match='HDD'):
        pd.engine_command(c, tmp_path, tmp_path/'run')
    c = case()
    c['context_window'] = 65536
    with pytest.raises(ValueError, match='128K'):
        pd.validate_case(c)


def test_morphx_ninfer_uses_eager_launches_for_actual_kernel_interception(tmp_path):
    c=case('ninfer');c.update(cuda_graph=False,kv_dtype='rk4v4-e8')
    assert '--no-cuda-graph' in pd.engine_command(c,tmp_path,tmp_path/'run')


def test_ninfer_context_cannot_exceed_its_explicit_kv_pool():
    c=case('ninfer')
    c['context_window']=262144
    with pytest.raises(ValueError,match='NInfer.*context'):
        pd.validate_case(c)


def test_prometheus_labels_are_not_split_as_plain_two_column_text():
    metrics = pd.parse_metrics('''# HELP requests requests in flight
vllm:num_requests_running{model_name="qwen38-27b",engine="0"} 2
llamacpp:requests_processing 1
vllm:request_success_total{model_name="a b",finished_reason="stop"} 5 123456
vllm:num_requests_waiting{model_name="qwen38-27b"} 0
''')
    assert metrics['vllm:num_requests_running'] == 2
    assert metrics['llamacpp:requests_processing'] == 1
    assert metrics['vllm:request_success_total'] == 5
    assert metrics['vllm:num_requests_waiting'] == 0


def test_vllm_command_retains_solo_gpu_and_disables_image_budget(tmp_path):
    c = case('vllm')
    c.update(binary='pd-baselines/vllm-venv/bin/python', kv_dtype='fp8',
             quantization='compressed-tensors-int4', gpu_memory_utilization=.95)
    args = pd.engine_command(c, tmp_path, tmp_path/'run')
    assert '--enable-chunked-prefill' in args
    assert '--enable-prompt-tokens-details' in args
    assert args[args.index('--tensor-parallel-size')+1] == '1'
    assert args[args.index('--max-num-seqs')+1] == '4'
    assert '--speculative-config' not in args


def test_vllm_preserves_decode_graph_policy_without_inductor(tmp_path):
    import json
    c=case('vllm')
    c.update(cuda_graph=True, compilation_config={
        'mode':0, 'cudagraph_mode':'FULL_DECODE_ONLY',
        'cudagraph_capture_sizes':[1,2,4]})
    args=pd.engine_command(c,tmp_path,tmp_path/'run')
    assert '--enforce-eager' not in args
    assert json.loads(args[args.index('--compilation-config')+1]) == c['compilation_config']
    c['cuda_graph']=False
    with pytest.raises(ValueError,match='graph.*conflict'):
        pd.engine_command(c,tmp_path,tmp_path/'run')


def test_sglang_keeps_full_context_and_explicit_mixed_schedule(tmp_path):
    c=case('sglang')
    c.update(binary='pd-baselines/sglang0519-venv/bin/python',
             kv_dtype='fp8_e4m3',context_window=131328,chunk=512,
             cuda_graph=False,max_mamba_cache_size=16)
    args=pd.engine_command(c,tmp_path,tmp_path/'run')
    assert args[args.index('--context-length')+1]=='131328'
    assert args[args.index('--tp-size')+1]=='1'
    assert args[args.index('--max-running-requests')+1]=='4'
    assert '--enable-mixed-chunk' in args and '--enable-metrics' in args
    assert '--language-model-only' in args
    assert '--disable-prefill-cuda-graph' in args
    assert '--disable-decode-cuda-graph' in args
    assert args[args.index('--max-mamba-cache-size')+1]=='16'
    assert '--max-total-tokens' not in args  # Observe the allocator's actual pool.
    c['draft']=1
    with pytest.raises(ValueError,match='SGLang.*speculative'):
        pd.engine_command(c,tmp_path,tmp_path/'run')


def test_sglang_idle_wait_observes_running_and_queued_work(monkeypatch):
    from scripts import run_pd_baselines as runner
    samples=iter([{'sglang:num_running_reqs':0,'sglang:num_queue_reqs':1},
                  {'sglang:num_running_reqs':1,'sglang:num_queue_reqs':0},
                  {'sglang:num_running_reqs':0,'sglang:num_queue_reqs':0}])
    sleeps=[]
    monkeypatch.setattr(runner,'get',lambda _:next(samples))
    monkeypatch.setattr(runner.time,'sleep',lambda seconds:sleeps.append(seconds))
    runner.wait_idle(timeout=1)
    assert len(sleeps)==2


def test_sglang_runtime_cannot_silently_shrink_context_or_concurrency():
    c=case('sglang');c['context_window']=131328
    info={'context_length':131328,'internal_states':[
        {'effective_max_running_requests_per_dp':4,
         'memory_usage':{'token_capacity':140000}}]}
    pd.validate_sglang_runtime(c,info)
    for changed in [dict(info,context_length=65536),
                    dict(info,internal_states=[]),
                    dict(info,internal_states=[{'effective_max_running_requests_per_dp':3,
                                               'memory_usage':{'token_capacity':140000}}]),
                    dict(info,internal_states=[{'effective_max_running_requests_per_dp':4,
                                               'memory_usage':{'token_capacity':100000}}])]:
        with pytest.raises(RuntimeError,match='SGLang.*capacity'):
            pd.validate_sglang_runtime(c,changed)


def test_llama_mtp_has_explicit_draft_cache_precision(tmp_path):
    c=case();c.update(draft=4,draft_kv_dtype='q4_0')
    args=pd.engine_command(c,tmp_path,tmp_path/'run')
    assert args[args.index('--spec-draft-type-k')+1]=='q4_0'
    assert args[args.index('--spec-draft-type-v')+1]=='q4_0'


def test_owned_engine_start_failure_stops_child_and_leaves_other_process(tmp_path,monkeypatch):
    import subprocess
    import sys
    from scripts.run_pd_baselines import OwnedEngine
    def unavailable(*_):
        raise ConnectionError('fixture has no HTTP server')
    monkeypatch.setattr('scripts.run_pd_baselines.get',unavailable)
    c = case()
    other = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
    engine = OwnedEngine(c, tmp_path, tmp_path/'run')
    engine.args = [sys.executable, '-c', 'raise SystemExit(17)']
    try:
        with pytest.raises(RuntimeError, match='startup exit 17'):
            with engine:
                pytest.fail('failed server cannot enter measurements')
        assert engine.proc.poll() == 17
        assert other.poll() is None
    finally:
        other.terminate()
        other.wait(timeout=3)


def test_owned_measurement_failure_is_restored_by_existing_service_contract(tmp_path):
    from scripts.run_pd_baselines import run_with_restoration
    from inferx.service_ownership import process_identity
    import json
    import subprocess
    import sys
    child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
    saved = process_identity(child.pid, Path(sys.executable))
    saved.update(max_context=262144, draft_tokens=7)
    paths = [tmp_path/'runtime.json', tmp_path/'best-service.json']
    restored = []
    def restore(original):
        restored.append(original)
        return dict(original, pid=999, proc_start_ticks='restored')
    def measurement():
        raise RuntimeError('experiment failed')
    try:
        with pytest.raises(RuntimeError, match='experiment failed'):
            run_with_restoration(saved, restore, paths, None, measurement)
        child.wait(timeout=3)
        assert len(restored) == 1
        assert json.loads(paths[0].read_text()) == json.loads(paths[1].read_text())
        assert json.loads(paths[0].read_text())['max_context'] == 262144
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=3)


def test_frozen_workloads_verify_actual_manifest_and_reject_changed_payload(tmp_path):
    import hashlib
    import json
    workloads = {'cold-131072': {'payload': {'messages': []}}}
    (tmp_path/'workloads.json').write_text(json.dumps(workloads))
    (tmp_path/'grid.json').write_text('{"arrival_duration_s":48}')
    files = [dict(file=name, sha256=hashlib.sha256((tmp_path/name).read_bytes()).hexdigest())
             for name in ('workloads.json', 'grid.json')]
    (tmp_path/'workload-manifest.json').write_text(json.dumps({'files': files}))
    templates, grid = pd.load_frozen_workloads(tmp_path)
    assert templates == workloads and grid['arrival_duration_s'] == 48
    (tmp_path/'workloads.json').write_text('{}')
    with pytest.raises(ValueError, match='hash'):
        pd.load_frozen_workloads(tmp_path)


def test_arrival_stream_contains_full_long_request_and_all_short_attempts():
    short = {'payload': {'messages': ['24K']}, 'expected_tool': {'name':'record_status'}}
    long = {'payload': {'messages': ['128K']}, 'expected_tool': {'name':'record_status'}}
    templates = {'tool-24576':short, 'cold-131072':long}
    grid = dict(arrival_duration_s=48, long_offsets_s=[16], long_targets=[131072])
    rows = pd.arrival_requests({'rate':2.5,'long_mode':'cold'}, templates, grid, 'wave')
    assert len(rows) == 121
    assert rows[-1]['prompt_id'] == 'cold-131072' and rows[-1]['offset_s'] == 16
    assert rows[-1]['payload']['messages'] == ['128K']
    assert rows[119]['offset_s'] == 47.6
    assert len({r['id'] for r in rows}) == 121


def test_vllm_empty_health_response_is_healthy():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    from scripts.run_pd_baselines import get
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
        def log_message(self, *_):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert get('/health', f'http://127.0.0.1:{server.server_port}') == {'status':'ok'}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_timed_out_external_suite_cleans_its_group_and_preserves_unrelated_process(tmp_path):
    import os,subprocess,sys,time
    from inferx.pd_baselines import run_owned_command
    other=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
    script=tmp_path/'parent.py'
    script.write_text('import subprocess,sys,time\nfrom pathlib import Path\n'
        f'child=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"])\n'
        f'Path({str(tmp_path/"child.pid")!r}).write_text(str(child.pid))\n'
        'time.sleep(30)\n')
    try:
        with (tmp_path/'output').open('w') as output:
            with pytest.raises(subprocess.TimeoutExpired):
                run_owned_command([sys.executable,str(script)],os.environ.copy(),output,timeout=1)
        child=int((tmp_path/'child.pid').read_text())
        for _ in range(100):
            path=Path(f'/proc/{child}/stat')
            try:alive=path.read_text().split(') ',1)[1][0]!='Z'
            except FileNotFoundError:alive=False
            if not alive:break
            time.sleep(.01)
        assert not alive and other.poll() is None
    finally:
        other.terminate();other.wait(timeout=3)


@pytest.mark.parametrize('crashed',[False,True])
def test_healthy_engine_shutdown_also_stops_owned_worker(tmp_path,crashed):
    import os,signal,subprocess,sys,time
    from scripts.run_pd_baselines import OwnedEngine
    from inferx.service_ownership import process_identity
    engine=OwnedEngine(case(),tmp_path,tmp_path/'run')
    program=('import subprocess,sys,time;from pathlib import Path;'
        'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]);'
        f'Path({str(tmp_path/"worker.pid")!r}).write_text(str(p.pid))\n')
    program+=(f'while not Path({str(tmp_path/"exit")!r}).exists(): time.sleep(.01)\n'
              if crashed else 'time.sleep(30)\n')
    engine.proc=subprocess.Popen([sys.executable,'-c',program],start_new_session=True)
    other=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
    worker_fd=None
    try:
        for _ in range(100):
            if (tmp_path/'worker.pid').exists():break
            time.sleep(.01)
        worker=int((tmp_path/'worker.pid').read_text());worker_fd=os.pidfd_open(worker)
        engine.identity=process_identity(engine.proc.pid,Path(sys.executable))
        if crashed:
            (tmp_path/'exit').touch()
            for _ in range(100):
                if os.waitid(os.P_PID,engine.proc.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT):break
                time.sleep(.01)
            else:pytest.fail('fixture frontend did not exit')
        engine.stop()
        for _ in range(100):
            try:alive=Path(f'/proc/{worker}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
            except (FileNotFoundError,ProcessLookupError):alive=False
            if not alive:break
            time.sleep(.01)
        assert not alive and other.poll() is None
    finally:
        if worker_fd is not None:
            try:signal.pidfd_send_signal(worker_fd,signal.SIGKILL)
            except ProcessLookupError:pass
            os.close(worker_fd)
        if engine.proc.poll() is None:
            engine.proc.terminate();engine.proc.wait(timeout=3)
        other.terminate();other.wait(timeout=3)


def test_campaign_preserves_harness_bytes_when_workspace_changes(tmp_path):
    import hashlib
    source=tmp_path/'code';(source/'inferx').mkdir(parents=True)
    module=source/'inferx'/'example.py';module.write_bytes(b'original = True\n')
    snapshot=tmp_path/'raw'/'harness'
    hashes=pd.snapshot_harness(source,snapshot,['inferx/example.py'])
    module.write_bytes(b'original = False\n')
    assert (snapshot/'inferx/example.py').read_bytes()==b'original = True\n'
    assert hashes['inferx/example.py']==hashlib.sha256(b'original = True\n').hexdigest()
    with pytest.raises(ValueError,match='relative'):
        pd.snapshot_harness(source, tmp_path/'invalid', ['../private.py'])
