import asyncio
import json
import os
from pathlib import Path
import signal
import time
import uuid


class InfrastructureError(RuntimeError):
    """Environment/transport failures must not be scored as model failures."""


async def verify(code: str, tests: str, entry_point: str, timeout: float,
                 cpu: int | None, *, worker: Path | None = None) -> dict:
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    worker = worker or Path(__file__).with_name("sandbox_worker.py")
    marker = "INFERX_" + uuid.uuid4().hex + ":"
    payload = json.dumps(dict(code=code, tests=tests, entry_point=entry_point,
                              timeout=timeout, cpu=cpu, marker=marker)).encode()
    start = time.monotonic()
    process = await asyncio.create_subprocess_exec(
        "/usr/bin/python3", "-I", "-u", str(worker),
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT, start_new_session=True,
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "OPENBLAS_NUM_THREADS": "1"},
        limit=65536,
    )
    output = bytearray()
    status = None
    try:
        async with asyncio.timeout(timeout):
            process.stdin.write(payload)
            await process.stdin.drain()
            process.stdin.close()
            while chunk := await process.stdout.read(4096):
                output.extend(chunk)
                if len(output) > 65536:
                    status = "output_limit"
                    break
            if status is None:
                await process.wait()
    except TimeoutError:
        status = "timeout"
    finally:
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
    elapsed = time.monotonic() - start
    records = []
    for line in output.decode(errors="replace").splitlines():
        if line.startswith(marker):
            try:
                records.append(json.loads(line[len(marker):]))
            except json.JSONDecodeError:
                pass
    if any(x["status"] == "infrastructure_error" for x in records):
        raise InfrastructureError(str(records))
    if not any(x["status"] == "ready" for x in records):
        raise InfrastructureError("sandbox never became ready: " + output[:1000].decode(errors="replace"))
    result = next((x for x in reversed(records) if x["status"] in ("passed", "failed")), None)
    if status:
        result = {"status": status, "test_elapsed": None, "error": status}
    elif result is None:
        result = {"status": "failed", "test_elapsed": None, "error": "exit without test result"}
    result["elapsed"] = elapsed
    return result
