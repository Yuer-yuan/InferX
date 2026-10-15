"""Contracts that prevent silently measuring a different PD workload."""
import copy
from pathlib import Path

import pytest

from inferx import pd_baselines as pd


def case(engine='llama'):
    return dict(id='llama-c4', engine=engine, binary='source/llama/bin/llama-server',
                model='models/qwen38/model.gguf', quantization='UD-Q4_K_M',
                gpu_uuid='GPU-4d9bef9d-631c-bb36-3ff4-589d6a184189',
                context_window=262144, kv_capacity=212992, concurrency=4,
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
    assert args[args.index('--tensor-parallel-size')+1] == '1'
    assert args[args.index('--max-num-seqs')+1] == '4'
    assert '--speculative-config' not in args


def test_llama_mtp_has_explicit_draft_cache_precision(tmp_path):
    c=case();c.update(draft=4,draft_kv_dtype='q4_0')
    args=pd.engine_command(c,tmp_path,tmp_path/'run')
    assert args[args.index('--spec-draft-type-k')+1]=='q4_0'
    assert args[args.index('--spec-draft-type-v')+1]=='q4_0'


def test_owned_engine_start_failure_stops_child_and_leaves_other_process(tmp_path):
    import subprocess
    import sys
    from scripts.run_pd_baselines import OwnedEngine
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
