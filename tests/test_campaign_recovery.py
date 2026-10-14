import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from inferx.service_ownership import RestorableService, process_identity, refreshed_manifest


def test_bandwidth_script_refuses_unselected_gpu_before_loading_cuda(tmp_path):
    env = {**os.environ, 'INFERX_ROOT': str(tmp_path)}
    env.pop('CUDA_VISIBLE_DEVICES', None)
    result = subprocess.run([sys.executable, '-m', 'scripts.measure_device_bandwidth'],
                            env=env, text=True, capture_output=True)
    assert result.returncode != 0 and 'GPU1 UUID selection required' in result.stderr


def test_campaign_exception_restores_service_and_refreshes_both_manifests(tmp_path):
    original = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
    restored = []
    saved = dict(max_context=262144, draft_tokens=7, max_concurrency=1,
                 **process_identity(original.pid, Path(sys.executable)))
    manifests = [tmp_path/'runtime.json', tmp_path/'best-service.json']

    def restart(old):
        child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
        restored.append(child)
        return refreshed_manifest(old, process_identity(child.pid, Path(sys.executable)))

    try:
        with pytest.raises(RuntimeError, match='benchmark failed'):
            with RestorableService(saved, restart, manifests):
                original.wait(timeout=2)
                raise RuntimeError('benchmark failed')
        assert len(restored) == 1 and restored[0].poll() is None
        assert unrelated.poll() is None
        for path in manifests:
            manifest = json.loads(path.read_text())
            assert manifest['pid'] == restored[0].pid
            assert manifest['proc_start_ticks'] != saved['proc_start_ticks'] or manifest['pid'] != saved['pid']
            assert manifest['max_context'] == 262144 and manifest['draft_tokens'] == 7
    finally:
        for child in [original, unrelated, *restored]:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=2)


def test_interrupt_after_shutdown_signal_still_restores_service(tmp_path, monkeypatch):
    import signal
    import inferx.service_ownership as ownership
    original = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
    saved = process_identity(original.pid, Path(sys.executable))
    restored = []
    paths = [tmp_path/'runtime.json', tmp_path/'best.json']
    send = signal.pidfd_send_signal
    interrupted = False

    def interrupt_once(fd, number):
        nonlocal interrupted
        send(fd, number)
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt('interrupted after SIGTERM')

    def restart(old):
        original.wait(timeout=2)
        child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
        restored.append(child)
        return refreshed_manifest(old, process_identity(child.pid, Path(sys.executable)))

    monkeypatch.setattr(ownership.signal, 'pidfd_send_signal', interrupt_once)
    try:
        with pytest.raises(KeyboardInterrupt, match='after SIGTERM'):
            with RestorableService(saved, restart, paths):
                pytest.fail('interrupted entry must not execute body')
        assert len(restored) == 1 and restored[0].poll() is None
        assert unrelated.poll() is None
        assert all(json.loads(p.read_text())['pid'] == restored[0].pid for p in paths)
    finally:
        for child in [original, unrelated, *restored]:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=2)


def test_final_idle_check_refuses_newly_busy_service_before_signal(tmp_path, monkeypatch):
    import importlib
    import signal
    import inferx.service_ownership as ownership
    monkeypatch.setenv('INFERX_ROOT', str(tmp_path))
    campaign = importlib.import_module('scripts.run_concurrency_campaign')
    original = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
    saved = process_identity(original.pid, Path(sys.executable))
    busy = False
    signals, restored = [], []
    monkeypatch.setattr(campaign, 'gpu_processes', lambda: {
        (original.pid, campaign.GPU), (2504785, 'GPU-a66b08e8-8b65-0faf-22e2-b951a97b2b03')})

    def metrics(path, endpoint=None):
        if path == '/health':
            return dict(status='ok')
        return {'llamacpp:requests_processing': int(busy), 'llamacpp:requests_deferred': 0}

    monkeypatch.setattr(campaign, 'get', metrics)
    monkeypatch.setattr(ownership.signal, 'pidfd_send_signal', lambda fd, number: signals.append(number))
    try:
        campaign.final_shutdown_check(saved)
        busy = True  # A request arrives while the controller prepares metadata.
        with pytest.raises(RuntimeError, match='busy'):
            with RestorableService(saved, lambda old: restored.append(old), [],
                    pre_stop_check=lambda: campaign.final_shutdown_check(saved)):
                pytest.fail('busy service must not enter experiment')
        assert not signals and not restored and original.poll() is None
    finally:
        original.terminate()
        original.wait(timeout=2)
