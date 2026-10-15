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


def test_sglang_hf_text_mode_preserves_model_and_explicit_pool_policy(tmp_path):
    import json
    c=case('sglang');c.update(context_window=131328,text_only_via_hf_config=True,
        unified_memory=True,max_mamba_cache_size=20,
        mamba_radix_cache_strategy='extra_buffer_lazy')
    args=pd.engine_command(c,tmp_path,tmp_path/'run')
    assert '--language-model-only' not in args and '--language-only' not in args
    assert json.loads(args[args.index('--json-model-override-args')+1])=={'language_model_only':True}
    assert '--enable-unified-memory' in args
    assert args[args.index('--context-length')+1]=='131328'
    assert args[args.index('--max-running-requests')+1]=='4'
    assert args[args.index('--max-mamba-cache-size')+1]=='20'
    assert args[args.index('--mamba-radix-cache-strategy')+1]=='extra_buffer_lazy'


def test_sglang_state_precision_change_is_explicit_and_optional(tmp_path):
    c=case('sglang');c.update(mamba_ssm_dtype='bfloat16',attention_backend='triton')
    args=pd.engine_command(c,tmp_path,tmp_path/'run')
    assert args[args.index('--mamba-ssm-dtype')+1]=='bfloat16'
    assert args[args.index('--attention-backend')+1]=='triton'
    del c['mamba_ssm_dtype']
    assert '--mamba-ssm-dtype' not in pd.engine_command(c,tmp_path,tmp_path/'run')


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


@pytest.mark.parametrize('crashed',[False,True,'during_stop','live_mismatch'])
def test_healthy_engine_shutdown_also_stops_owned_worker(tmp_path,crashed,monkeypatch):
    import os,signal,subprocess,sys,time
    from scripts.run_pd_baselines import OwnedEngine
    from inferx.service_ownership import process_identity
    engine=OwnedEngine(case(),tmp_path,tmp_path/'run')
    program=('import subprocess,sys,time;from pathlib import Path;'
        'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]);'
        f'Path({str(tmp_path/"worker.pid")!r}).write_text(str(p.pid))\n')
    program+=(f'while not Path({str(tmp_path/"exit")!r}).exists(): time.sleep(.01)\n'
              'time.sleep(.1)\n'
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
        if crashed is True:
            (tmp_path/'exit').touch()
            for _ in range(100):
                if os.waitid(os.P_PID,engine.proc.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT):break
                time.sleep(.01)
            else:pytest.fail('fixture frontend did not exit')
        elif crashed in ('during_stop','live_mismatch'):
            def mismatch(*_,**__):
                if crashed=='during_stop':(tmp_path/'exit').touch()
                raise RuntimeError('process identity mismatch; refusing signal')
            monkeypatch.setattr('scripts.run_pd_baselines.stop_matching_process',mismatch)
        if crashed=='live_mismatch':
            with pytest.raises(RuntimeError,match='refusing signal'):engine.stop()
            assert engine.proc.poll() is None and other.poll() is None
            assert Path(f'/proc/{worker}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
            return
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


def stubborn_frontend(tmp_path, exits):
    """Worker publishes identity only after installing its TERM refusal."""
    worker=tmp_path/'worker.py'
    worker.write_text('import os,signal,time,json\nfrom pathlib import Path\n'
        'signal.signal(signal.SIGTERM,signal.SIG_IGN)\n'
        f'Path({str(tmp_path/"worker.json")!r}).write_text(json.dumps(dict(pid=os.getpid(),'
        'ticks=Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")",1)[1].split()[19])))\n'
        'time.sleep(60)\n')
    front=tmp_path/'frontend.py'
    front.write_text('import subprocess,sys,time\nfrom pathlib import Path\n'
        f'subprocess.Popen([sys.executable,{str(worker)!r}])\n'
        f'while not Path({str(tmp_path/"worker.json")!r}).exists():time.sleep(.01)\n'
        + ('' if exits else 'time.sleep(60)\n'))
    return front


def fixture_worker(tmp_path):
    import json,time
    path=tmp_path/'worker.json'
    deadline=time.monotonic()+5
    while not path.exists():
        assert time.monotonic()<deadline,'worker fixture did not become ready'
        time.sleep(.01)
    return json.loads(path.read_text())


def fixture_alive(identity):
    try:
        fields=Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(')',1)[1].split()
        return fields[19]==identity['ticks'] and fields[0]!='Z'
    except (FileNotFoundError,ProcessLookupError):
        return False


def kill_fixture(identity):
    import os,signal
    try:
        fd=os.pidfd_open(identity['pid'])
    except ProcessLookupError:
        return
    try:
        if fixture_alive(identity):signal.pidfd_send_signal(fd,signal.SIGKILL)
    finally:
        os.close(fd)


@pytest.mark.parametrize('exits',[False,True])
def test_owned_engine_kills_stubborn_worker_before_reaping_frontend(tmp_path,exits):
    import os,subprocess,sys,time
    from scripts.run_pd_baselines import OwnedEngine
    from inferx.service_ownership import process_identity
    script=stubborn_frontend(tmp_path,exits)
    engine=OwnedEngine(case(),tmp_path,tmp_path/'run')
    engine.proc=subprocess.Popen([sys.executable,str(script)],start_new_session=True)
    other=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
    identity=None
    try:
        identity=fixture_worker(tmp_path)
        if exits:
            deadline=time.monotonic()+5
            while os.waitid(os.P_PID,engine.proc.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT) is None:
                assert time.monotonic()<deadline
                time.sleep(.01)
        else:
            engine.identity=process_identity(engine.proc.pid,Path(sys.executable))
        engine.stop()
        deadline=time.monotonic()+2
        while fixture_alive(identity) and time.monotonic()<deadline:time.sleep(.01)
        assert not fixture_alive(identity),'TERM-ignoring owned worker survived stop()'
        assert other.poll() is None
    finally:
        if identity:kill_fixture(identity)
        if engine.proc.poll() is None:engine.proc.kill()
        engine.proc.wait(timeout=3)
        other.terminate();other.wait(timeout=3)


@pytest.mark.parametrize('exits',[False,True])
def test_external_suite_cleans_stubborn_worker_on_timeout_and_success(tmp_path,exits):
    import os,subprocess,sys,time
    script=stubborn_frontend(tmp_path,exits)
    other=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
    identity=None
    try:
        with (tmp_path/'suite.log').open('w') as output:
            command=[sys.executable,str(script)]
            if exits:
                assert pd.run_owned_command(command,os.environ.copy(),output,timeout=3)==0
            else:
                with pytest.raises(subprocess.TimeoutExpired):
                    pd.run_owned_command(command,os.environ.copy(),output,timeout=1)
        identity=fixture_worker(tmp_path)
        deadline=time.monotonic()+2
        while fixture_alive(identity) and time.monotonic()<deadline:time.sleep(.01)
        assert not fixture_alive(identity),'external command left its TERM-ignoring worker'
        assert other.poll() is None
    finally:
        if (tmp_path/'worker.json').exists():kill_fixture(fixture_worker(tmp_path))
        other.terminate();other.wait(timeout=3)


@pytest.mark.parametrize('signal_name',['SIGTERM','SIGHUP'])
@pytest.mark.parametrize('stage',['entry','measurement','restore','shutdown','spawn'])
def test_real_controller_signal_restores_service_and_both_manifests(tmp_path,signal_name,stage):
    import json,os,signal,subprocess,sys,time
    stubborn_frontend(tmp_path,False)
    controller=tmp_path/'controller.py'
    controller.write_text('''import json,os,signal,subprocess,sys,time
from pathlib import Path
from inferx import service_ownership as ownership
from inferx.context_agents import verify_process_identity
from scripts import run_pd_baselines as runner
from scripts.run_pd_baselines import run_with_restoration
root=Path(sys.argv[1]);stage=sys.argv[2]
original=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
saved=ownership.process_identity(original.pid,Path(sys.executable))
(root/'original.json').write_text(json.dumps(dict(pid=original.pid,ticks=saved['proc_start_ticks'])))
paths=[root/'first.json',root/'second.json']
for p in paths:p.write_text(json.dumps(saved))
restored=[];fronts=[]
def restore(_):
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
    restored.append(child)
    if stage=='restore':
        (root/'ready').touch();time.sleep(.3)
    # Repeated termination while restoring must not abort this callback.
    os.kill(os.getpid(),getattr(signal,sys.argv[3]))
    return ownership.process_identity(child.pid,Path(sys.executable))
real_stop=ownership.stop_matching_process
def stop(identity,**kwargs):
    real_stop(identity,**kwargs)
    if stage=='entry' and not (root/'ready').exists():
        (root/'ready').touch();time.sleep(60)
ownership.stop_matching_process=stop
def measure():
    if stage=='restore':return
    if stage in ('spawn','shutdown'):
        engine=object.__new__(runner.OwnedEngine)
        engine.root=root;engine.directory=root/'engine';engine.directory.mkdir()
        engine.case={'engine':'ninfer','environment':{}}
        engine.args=[sys.executable,str(root/'frontend.py')]
        engine.proc=engine.identity=engine.log=engine.telemetry=None
        real_popen=subprocess.Popen
        def launch(*args,**kwargs):
            child=real_popen(*args,**kwargs)
            if kwargs.get('start_new_session'):
                fronts.append(child)
                while not (root/'worker.json').exists():time.sleep(.01)
                if stage=='spawn':
                    (root/'ready').touch();time.sleep(.3)
            return child
        if stage=='spawn':
            subprocess.Popen=launch
            with engine:pass
        else:
            engine.proc=launch(engine.args,start_new_session=True)
            engine.identity=ownership.process_identity(engine.proc.pid,Path(sys.executable))
            def stop_front(identity,**kwargs):
                real_stop(identity,**kwargs)
                (root/'ready').touch();time.sleep(.3)
            runner.stop_matching_process=stop_front
            engine.stop()
        return
    (root/'ready').touch();time.sleep(60)
try:
    try:run_with_restoration(saved,restore,paths,None,measure)
    except KeyboardInterrupt:pass
    a,b=[json.loads(p.read_text()) for p in paths]
    clean=True
    if (root/'worker.json').exists():
        worker=json.loads((root/'worker.json').read_text())
        deadline=time.monotonic()+1
        while True:
            try:
                stat=Path(f"/proc/{worker['pid']}/stat").read_text().rsplit(')',1)[1].split()
                clean=stat[19]!=worker['ticks'] or stat[0]=='Z'
            except FileNotFoundError:clean=True
            if clean or time.monotonic()>=deadline:break
            time.sleep(.01)
    (root/'verified.json').write_text(json.dumps(dict(equal=a==b,
        live=verify_process_identity(a['pid'],a['proc_start_ticks'],Path(a['executable'])),
        changed=a['pid']!=original.pid,handlers_reset=signal.getsignal(signal.SIGTERM)==signal.SIG_DFL
        and signal.getsignal(signal.SIGHUP)==signal.SIG_DFL,workers_clean=clean)))
finally:
    if (root/'worker.json').exists():
        worker=json.loads((root/'worker.json').read_text())
        try:
            fd=os.pidfd_open(worker['pid'])
            stat=Path(f"/proc/{worker['pid']}/stat").read_text().rsplit(')',1)[1].split()
            if stat[19]==worker['ticks']:signal.pidfd_send_signal(fd,signal.SIGKILL)
            os.close(fd)
        except ProcessLookupError:pass
    for p in [original,*restored,*fronts]:
        if p.poll() is None:p.kill()
        p.wait()
''')
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    with (tmp_path/'controller.log').open('w') as output:
        process=subprocess.Popen([sys.executable,str(controller),str(tmp_path),stage,signal_name],
                                 env=env,stdout=output,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+8
            while not (tmp_path/'ready').exists():
                assert process.poll() is None,(tmp_path/'controller.log').read_text()
                assert time.monotonic()<deadline
                time.sleep(.01)
            os.kill(process.pid,getattr(signal,signal_name))
            assert process.wait(timeout=8)==0,(tmp_path/'controller.log').read_text()
            assert json.loads((tmp_path/'verified.json').read_text())==dict(
                equal=True,live=True,changed=True,handlers_reset=True,workers_clean=True)
        finally:
            if process.poll() is None:process.kill()
            process.wait(timeout=3)
            if (tmp_path/'original.json').exists():
                kill_fixture(json.loads((tmp_path/'original.json').read_text()))
