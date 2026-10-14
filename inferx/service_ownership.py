"""Linux pidfd ownership checks; never signal a process from a stale PID file."""
import os
import signal
import time
import json
from pathlib import Path

from inferx.context_agents import process_start_ticks, verify_process_identity


def process_identity(pid: int, exe: Path) -> dict:
    ticks = process_start_ticks(pid)
    if not verify_process_identity(pid, ticks, exe):
        raise RuntimeError('process identity/ownership mismatch')
    return dict(pid=pid, proc_start_ticks=ticks, uid=os.getuid(), executable=str(exe.resolve()))


def stop_matching_process(identity: dict, timeout: float = 20) -> None:
    fd = os.pidfd_open(identity['pid'])
    try:
        if identity.get('uid', os.getuid()) != os.getuid() or not verify_process_identity(
                identity['pid'], identity['proc_start_ticks'], Path(identity['executable'])):
            raise RuntimeError('process identity mismatch; refusing signal')
        signal.pidfd_send_signal(fd, signal.SIGTERM)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                stat = Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(')', 1)[1].split()
                if stat[0] == 'Z' or stat[19] != identity['proc_start_ticks']:
                    return
            except FileNotFoundError:
                return
            time.sleep(.02)
        # The fd still refers to the checked process even if its numeric PID changes later.
        signal.pidfd_send_signal(fd, signal.SIGKILL)
    finally:
        os.close(fd)


def refreshed_manifest(saved: dict, identity: dict) -> dict:
    return {**saved, **identity, 'created_unix': time.time()}


class RestorableService:
    """Stop a checked service and restore its configuration even on experiment failure."""
    def __init__(self, saved, restore, manifest_paths):
        self.saved, self.restore = saved, restore
        self.manifest_paths = manifest_paths
        self.restored = None

    def __enter__(self):
        stop_matching_process(self.saved)
        return self

    def __exit__(self, kind, value, traceback):
        self.restored = self.restore(self.saved)
        for path in self.manifest_paths:
            temporary = path.with_name(path.name + f'.refresh-{os.getpid()}')
            temporary.write_text(json.dumps(self.restored, indent=2))
            os.replace(temporary, path)
        return False
