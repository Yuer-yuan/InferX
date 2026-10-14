import asyncio
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

from inferx.verifier import InfrastructureError
try:
    from inferx.repo_verifier import verify_repo
except ImportError:
    verify_repo = None


class RepoVerifierTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.assertIsNotNone(verify_repo, 'restricted repository verifier missing')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root/'fixtures/demo'
        (self.repo/'tests').mkdir(parents=True)
        (self.repo/'mod.py').write_text('def f(x):\n    return x + 1\n')
        (self.repo/'tests/test_mod.py').write_text(
            'import unittest\nfrom mod import f\nclass T(unittest.TestCase):\n'
            '    def test_a(self): self.assertEqual(f(2), 3)\n'
            '    def test_b(self): self.assertEqual(f(5), 6)\n')
        self.task = dict(task_id='fixture/0', repo_name='demo', file_name='/demo/mod.py',
                         prefix_code='def f(x):\n    ', suffix_code='\n', test_mode='unittest',
                         expected_counts={'collected': 2, 'skipped': 0})
        self.cpu = min(os.sched_getaffinity(0))

    async def check(self, code, **kwargs):
        return await verify_repo(code, self.task, fixture_root=self.root/'fixtures',
            python=sys.executable, work_root=self.root/'work', cpu=self.cpu, **kwargs)

    async def test_correct_and_wrong_patch_execute_private_source(self):
        good = await self.check('return x + 1')
        bad = await self.check('return x - 1')
        self.assertEqual(good['status'], 'passed', good)
        self.assertEqual(good['counts']['collected'], 2)
        self.assertEqual(bad['status'], 'failed', bad)
        self.assertEqual(bad['counts']['failed'], 2)
        self.assertIn('return x + 1', (self.repo/'mod.py').read_text())

    async def test_outside_read_write_and_repo_write_denied(self):
        secret = self.root/'private.txt'; secret.write_text('sentinel')
        new = self.root/'outside-created'
        code = ('import os\n    from pathlib import Path\n'
                f'    for path, mode in [({str(secret)!r}, "r"), ({str(new)!r}, "w"), (__file__, "w")]:\n'
                '        try:\n            open(path, mode)\n'
                '        except PermissionError:\n            pass\n'
                '        else:\n            raise AssertionError("filesystem escaped")\n'
                '    scratch = Path(os.environ["TMPDIR"]) / "allowed"\n'
                '    scratch.write_text("ok")\n    assert scratch.read_text() == "ok"\n'
                '    return x + 1')
        result = await self.check(code)
        self.assertEqual(result['status'], 'passed', result)
        self.assertEqual(secret.read_text(), 'sentinel')
        self.assertFalse(new.exists())

    async def test_relative_test_outputs_are_in_scratch(self):
        result = await self.check('from pathlib import Path\n    Path("test-output").write_text("ok")\n    return x + 1')
        self.assertEqual(result['status'], 'passed', result)
        self.assertFalse((self.repo/'test-output').exists())

    async def test_network_fork_and_exec_are_denied(self):
        code = ('import os, socket\n'
                '    for op in [lambda: socket.socket(), lambda: os.execv("/bin/true", ["true"])]:\n'
                '        try:\n            op()\n'
                '        except PermissionError:\n            pass\n'
                '        else:\n            raise AssertionError("syscall escaped")\n'
                '    try:\n        child = os.fork()\n'
                '    except PermissionError:\n        return x + 1\n'
                '    if child == 0:\n        os._exit(0)\n'
                '    os.waitpid(child, 0)\n    raise AssertionError("fork escaped")')
        result = await self.check(code)
        self.assertEqual(result['status'], 'passed', result)

    async def test_zero_and_reduced_tests_do_not_pass(self):
        (self.repo/'tests/test_mod.py').write_text('import unittest\n')
        self.assertEqual((await self.check('return x + 1'))['status'], 'failed')
        (self.repo/'tests/test_mod.py').write_text(
            'import unittest\nfrom mod import f\nclass T(unittest.TestCase):\n'
            '    def test_a(self): self.assertEqual(f(2), 3)\n')
        self.assertEqual((await self.check('return x + 1'))['status'], 'failed')

    async def test_wall_timeout_reaps_worker(self):
        result = await self.check('while True:\n        pass', timeout=.5)
        self.assertEqual(result['status'], 'timeout', result)
        self.assertLess(result['elapsed'], 3)

    async def test_output_flood_is_bounded_and_reaped(self):
        async with asyncio.timeout(3):
            result = await self.check('while True:\n        print("x" * 65536)', timeout=2)
        self.assertEqual(result['status'], 'output_limit', result)
        self.assertLess(len(result['output_tail']), 10000)

    async def test_cancellation_reaps_started_worker(self):
        code = ('import os\n    from pathlib import Path\n'
                '    (Path(os.environ["TMPDIR"]) / "pid").write_text(str(os.getpid()))\n'
                '    while True:\n        pass')
        future = asyncio.create_task(self.check(code, timeout=10))
        pid = None
        try:
            for _ in range(200):
                found = list((self.root/'work').glob('*/scratch/pid'))
                if found:
                    pid = int(found[0].read_text()); break
                if future.done():
                    self.fail(str(future.result()))
                await asyncio.sleep(.01)
            self.assertIsNotNone(pid)
        finally:
            future.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await future
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    async def test_worker_closing_input_before_setup_is_infrastructure_failure(self):
        worker = self.root/'closes-input.py'
        worker.write_text('import os\nos.close(0)\n')
        self.task['expected_counts'] = {'diagnostic': 'x'*(1024*1024)}
        with self.assertRaises(InfrastructureError):
            await self.check('return x + 1', worker=worker)

    async def test_missing_worker_is_infrastructure_failure(self):
        with self.assertRaises(InfrastructureError):
            await self.check('return x + 1', worker=self.root/'missing.py')

    @unittest.skipUnless(importlib.util.find_spec('pytest'), 'pytest is installed in HDD verifier venv')
    async def test_pytest_full_suite_uses_scratch_for_logs(self):
        self.task['test_mode'] = 'pytest'
        result = await self.check('return x + 1')
        self.assertEqual(result['status'], 'passed', result)


if __name__ == '__main__':
    unittest.main()
