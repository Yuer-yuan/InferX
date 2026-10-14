import asyncio
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time
import uuid

from inferx.repository import safe_relative
from inferx.verifier import InfrastructureError


async def verify_repo(code: str, task: dict, *, fixture_root: Path, python: str,
                      work_root: Path, cpu: int, timeout: float = 30,
                      worker: Path | None = None) -> dict:
    if timeout <= 0:
        raise ValueError('timeout must be positive')
    relative = safe_relative(task['repo_name'], task['file_name'])
    original = fixture_root/task['repo_name']
    # Refuse links rather than following one outside the fixture boundary.
    if original.is_symlink() or any(p.is_symlink() for p in original.rglob('*')):
        raise InfrastructureError('symlink in repository fixture')
    worker = worker or Path(__file__).with_name('repo_worker.py')
    work_root.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='candidate-', dir=work_root) as directory:
        private = Path(directory)
        repo, scratch = private/'repo', private/'scratch'
        shutil.copytree(original, repo, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
        scratch.mkdir()
        target = repo/relative
        if not target.is_file():
            raise InfrastructureError('target source is absent: ' + str(relative))
        target.write_text(task['prefix_code'] + code + task['suffix_code'])
        marker = 'INFERX_' + uuid.uuid4().hex + ':'
        payload = dict(repo=str(repo.resolve()), scratch=str(scratch.resolve()), cpu=cpu,
                       timeout=timeout, marker=marker, test_mode=task['test_mode'],
                       expected_counts=task.get('expected_counts'))
        process = await asyncio.create_subprocess_exec(
            python, '-I', '-B', '-u', str(worker.resolve()),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT, start_new_session=True,
            env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8',
                 'TMPDIR': str(scratch.resolve()), 'HOME': str(scratch.resolve()),
                 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1', 'OMP_NUM_THREADS': '1',
                 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}, limit=65536)
        output = bytearray()
        status = None
        try:
            async with asyncio.timeout(timeout):
                process.stdin.write(json.dumps(payload).encode())
                await process.stdin.drain()
                process.stdin.close()
                while chunk := await process.stdout.read(4096):
                    output.extend(chunk)
                    if len(output) > 256 * 1024:
                        status = 'output_limit'
                        break
                if status is None:
                    await process.wait()
        except TimeoutError:
            status = 'timeout'
        finally:
            if process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            # Drain after termination: asyncio.wait() alone can deadlock when
            # the pipe transport paused reading an output flood.
            while await process.stdout.read(65536):
                pass
            await process.wait()
        records = []
        for line in output.decode(errors='replace').splitlines():
            if line.startswith(marker):
                try:
                    records.append(json.loads(line[len(marker):]))
                except json.JSONDecodeError:
                    pass
        if any(x.get('status') == 'infrastructure_error' for x in records):
            raise InfrastructureError(str(records))
        if not any(x.get('status') == 'ready' for x in records):
            raise InfrastructureError('worker never ready: ' + output[:4000].decode(errors='replace'))
        result = next((x for x in reversed(records) if x.get('status') in ('passed', 'failed')), None)
        if status:
            result = {'status': status, 'test_elapsed': None, 'counts': None, 'error': status}
        elif result is None:
            result = {'status': 'failed', 'test_elapsed': None, 'counts': None,
                      'error': 'exit without test result'}
        result.update(elapsed=time.monotonic() - start,
                      output_tail=output[-8000:].decode(errors='replace'))
        return result
