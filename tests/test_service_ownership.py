import os
import subprocess
import sys
from pathlib import Path

import pytest

from inferx.service_ownership import process_identity, refreshed_manifest, stop_matching_process


def test_stale_identity_cannot_kill_process_and_other_process_survives():
    owned = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
    other = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
    try:
        identity = process_identity(owned.pid, Path(sys.executable))
        with pytest.raises(RuntimeError, match='identity'):
            stop_matching_process({**identity, 'proc_start_ticks': '0'}, timeout=1)
        assert owned.poll() is None
        stop_matching_process(identity, timeout=2)
        owned.wait(timeout=2)
        assert owned.returncode is not None and other.poll() is None
    finally:
        for child in [owned, other]:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=2)


def test_refreshed_manifest_keeps_context_and_only_updates_live_identity():
    saved = dict(pid=1, proc_start_ticks='old', max_context=262144, draft_tokens=7,
                 max_concurrency=1, kv_dtype='rk4v4-e8', command='unchanged')
    result = refreshed_manifest(saved, dict(pid=2, proc_start_ticks='new',
                                           uid=os.getuid(), executable='/expected'))
    assert result['pid'] == 2 and result['proc_start_ticks'] == 'new'
    assert result['max_context'] == 262144 and result['command'] == 'unchanged'
    assert saved['pid'] == 1
