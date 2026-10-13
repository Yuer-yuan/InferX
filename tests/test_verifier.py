import asyncio
import os
from pathlib import Path
import tempfile
import time
import unittest

try:
    from inferx.verifier import verify, InfrastructureError
except ImportError:
    verify = None
    InfrastructureError = RuntimeError


class VerifierTests(unittest.IsolatedAsyncioTestCase):
    async def check(self, code, expected="passed", timeout=2):
        self.assertIsNotNone(verify, "restricted verifier is not implemented")
        result = await verify(code, "def check(f):\n    assert f() == 1", "f", timeout, None)
        self.assertEqual(result["status"], expected, result)
        return result

    async def test_correct_and_incorrect(self):
        result = await self.check("def f(): return 1")
        self.assertGreater(result["elapsed"], result["test_elapsed"])
        await self.check("def f(): return 2", "failed")

    async def test_benchmark_standard_library_imports(self):
        await self.check("import hashlib, copy\ndef f():\n    assert len(hashlib.md5(b'abc').hexdigest()) == 32\n    return copy.deepcopy([1])[0]")

    async def test_deadline_kills_loop(self):
        before = time.monotonic()
        await self.check("def f():\n    while True: pass", "timeout", .3)
        self.assertLess(time.monotonic() - before, 2)

    async def test_file_and_network_are_blocked(self):
        for action in ["open('/etc/passwd').read()", "__import__('socket').socket()", "__import__('os').fork()"]:
            code = "def f():\n    try: " + action + "\n    except PermissionError: return 1\n    return 0"
            await self.check(code)

    async def test_cancel_reaps_process(self):
        self.assertIsNotNone(verify, "restricted verifier is not implemented")
        task = asyncio.create_task(verify("def f():\n    while True: pass", "def check(f): f()", "f", 30, None))
        await asyncio.sleep(.1)
        before = time.monotonic()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertLess(time.monotonic() - before, 2)

    async def test_missing_worker_is_infrastructure_failure(self):
        self.assertIsNotNone(verify, "restricted verifier is not implemented")
        with self.assertRaises(InfrastructureError):
            await verify("def f(): return 1", "def check(f): pass", "f", 2, None, worker=Path('/does-not-exist'))

    async def test_excess_output_is_bounded(self):
        await self.check("def f():\n    while True: print('x' * 4096)", "output_limit")

    async def test_abrupt_exit_is_not_success(self):
        await self.check("import os\nos._exit(0)\ndef f(): return 1", "failed")


if __name__ == '__main__':
    unittest.main()
