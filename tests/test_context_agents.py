import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from inferx.context_agents import (long_context_messages, lookup_answers,
                                   max_input_tokens, process_start_ticks,
                                   verify_process_identity)


def test_identity_rejects_stale_pid_and_wrong_executable(tmp_path):
    proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    try:
        ticks = process_start_ticks(proc.pid)
        executable = Path(sys.executable).resolve()
        assert verify_process_identity(proc.pid, ticks, executable)
        assert not verify_process_identity(proc.pid, str(int(ticks) + 1), executable)
        assert not verify_process_identity(proc.pid, ticks, tmp_path / 'foreign')
        assert not verify_process_identity(-1, ticks, executable)
    finally:
        proc.terminate()
        proc.wait(timeout=5)
    assert not verify_process_identity(proc.pid, ticks, executable)


def test_lookup_checks_use_distinct_early_middle_late_records():
    messages = long_context_messages(100)
    answers = lookup_answers(100)
    expected = {label: hashlib.sha256(f'inferx-registry-v1:{i}'.encode()).hexdigest()[:16]
                for label, i in [('early', 0), ('middle', 50), ('late', 99)]}
    assert answers == expected
    body = messages[-1]['content']
    positions = [body.index(value) for value in expected.values()]
    assert positions == sorted(positions)
    assert body.count('Record R') == 100
    assert 'R00000' in body and 'R00050' in body and 'R00099' in body
    assert 'Return only a JSON object' in body


def test_context_reserves_output_and_rejects_impossible_budget():
    assert max_input_tokens(131072) == 122880
    assert max_input_tokens(262144) == 253952
    with pytest.raises(ValueError):
        max_input_tokens(4096)
    with pytest.raises(ValueError):
        long_context_messages(2)
