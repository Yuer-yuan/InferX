"""Reproduction must work with the user-authorized service state and load."""
import argparse
import importlib
from pathlib import Path
import re
import shlex
import socket

import pytest

GPU0 = 'GPU-a66b08e8-8b65-0faf-22e2-b951a97b2b03'


@pytest.fixture
def shutdown_guard(tmp_path, monkeypatch):
    monkeypatch.setenv('INFERX_ROOT', str(tmp_path))
    campaign = importlib.import_module('scripts.run_concurrency_campaign')
    state = dict(processes={(42, campaign.GPU)}, processing=0, deferred=0,
                 port_open=False, requests=[])
    monkeypatch.setattr(campaign, 'gpu_processes', lambda: state['processes'])

    def get(path, endpoint):
        state['requests'].append((path, endpoint))
        assert endpoint == 'http://127.0.0.1:18081'
        return {'llamacpp:requests_processing': state['processing'],
                'llamacpp:requests_deferred': state['deferred']}

    def connect(address, timeout):
        assert address == ('127.0.0.1', 18080)
        raise ConnectionRefusedError()

    monkeypatch.setattr(campaign, 'get', get)
    monkeypatch.setattr(socket, 'create_connection', connect)
    return campaign, state


def test_stopped_gpu0_requires_explicit_mode_and_keeps_gpu1_idle_check(shutdown_guard):
    campaign, state = shutdown_guard
    with pytest.raises(RuntimeError, match='ownership'):
        campaign.final_shutdown_check({'pid': 42})
    campaign.final_shutdown_check({'pid': 42}, gpu0_state='stopped')
    assert state['requests'] == [('/metrics', 'http://127.0.0.1:18081')]


@pytest.mark.parametrize('change', ['missing-gpu1', 'foreign-gpu1', 'gpu0-restarted',
                                   'gpu0-replaced', 'processing', 'deferred'])
def test_stopped_mode_refuses_changed_ownership_or_new_work(shutdown_guard, change):
    campaign, state = shutdown_guard
    if change == 'missing-gpu1':
        state['processes'].clear()
    elif change == 'foreign-gpu1':
        state['processes'].add((43, campaign.GPU))
    elif change == 'gpu0-restarted':
        state['processes'].add((2504785, GPU0))
    elif change == 'gpu0-replaced':
        state['processes'].add((99, GPU0))
    else:
        state[change] = 1
    with pytest.raises(RuntimeError):
        campaign.final_shutdown_check({'pid': 42}, gpu0_state='stopped')


def test_stopped_mode_refuses_a_listening_gpu0_endpoint(shutdown_guard, monkeypatch):
    campaign, _ = shutdown_guard
    real_connect = socket.socket.connect
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)

        def connect(address, timeout):
            connection = socket.socket()
            connection.settimeout(timeout)
            real_connect(connection, listener.getsockname())
            return connection

        monkeypatch.setattr(socket, 'create_connection', connect)
        with pytest.raises(RuntimeError, match='GPU0'):
            campaign.final_shutdown_check({'pid': 42}, gpu0_state='stopped')


def test_stopped_mode_does_not_treat_unknown_port_state_as_absent(shutdown_guard, monkeypatch):
    campaign, _ = shutdown_guard

    def timeout(*args, **kwargs):
        raise TimeoutError('unobservable endpoint')

    monkeypatch.setattr(socket, 'create_connection', timeout)
    with pytest.raises(RuntimeError, match='GPU0'):
        campaign.final_shutdown_check({'pid': 42}, gpu0_state='stopped')


def test_report_gpu_invocation_parses_exact_formal_workload_before_side_effects(monkeypatch):
    from scripts import run_pd_baselines as runner
    report = (Path(__file__).parents[1]/'docs/reports/2026-10-15-pd-strategies.md').read_text()
    command = re.search(r'^"\$INFERX_ROOT/repo-venv/bin/python" -m scripts\.run_pd_baselines .*?(?=\n[^ ]|\Z)',
                        report, re.M | re.S)
    assert command is not None, 'report needs a complete executable GPU rerun command'
    argv = shlex.split(command.group(0).replace('\\\n', ' '))[3:]
    parse = argparse.ArgumentParser.parse_args
    captured = []

    class Parsed(Exception):
        pass

    def capture(parser):
        captured.append(parse(parser, argv))
        raise Parsed()

    monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', capture)
    with pytest.raises(Parsed):
        runner.main()
    args = captured[0]
    assert args.groups == ['short-control', 'cold-low', 'resume-high']
    assert args.short_rate == .5 and args.repeat == 3
    assert args.gpu0_state == 'stopped'
    assert args.cases == ['sglang-ctx-control', 'sglang-ctx-responsive128',
                          'sglang-pdmux-private-best']
    assert args.config == '$INFERX_ROOT/code-pd-baselines/configs/pd-strategies.json'
    assert args.run_id == 'pd-strategies-reproduce-NEW-ID'
