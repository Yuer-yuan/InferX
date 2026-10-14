"""One copied repository, one candidate, one restricted process."""
import json
import math
import os
from pathlib import Path
import resource
import sys
import sysconfig
import time
import traceback
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repo_restrictions import restrict_repository


def main():
    payload = json.loads(sys.stdin.buffer.read())
    marker = payload['marker']
    emit = lambda value: print('\n' + marker + json.dumps(value), flush=True)
    repo = Path(payload['repo'])
    scratch = Path(payload['scratch'])
    try:
        os.sched_setaffinity(0, {payload['cpu']})
        sys.dont_write_bytecode = True
        # Import the runner before restriction, but collect/import repository
        # tests only after both boundaries are irrevocably installed.
        if payload['test_mode'] == 'pytest':
            import pytest
        read_paths = {repo, Path(__file__).resolve().parent}
        for key in ('stdlib', 'platstdlib', 'purelib', 'platlib'):
            read_paths.add(Path(sysconfig.get_path(key)))
        for value in ('/usr/lib/x86_64-linux-gnu', '/lib/x86_64-linux-gnu',
                      '/dev/null', '/dev/urandom', '/etc/localtime'):
            if Path(value).exists():
                read_paths.add(Path(value))
        os.chdir(scratch)
        sys.path[:0] = [str(repo), str(repo/'src')]
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (max(1, math.ceil(payload['timeout'])),) * 2)
        resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**2,) * 2)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
        os.closerange(3, 1024)
        restrict_repository(list(read_paths), scratch)
    except BaseException as exc:
        emit({'status': 'infrastructure_error', 'error': repr(exc)})
        return
    emit({'status': 'ready'})
    start = time.monotonic()
    counts = {'collected': 0, 'skipped': 0, 'failed': 0}
    try:
        if payload['test_mode'] == 'unittest':
            suite = unittest.defaultTestLoader.discover(str(repo/'tests'))
            outcome = unittest.TextTestRunner(verbosity=0).run(suite)
            counts = {'collected': outcome.testsRun, 'skipped': len(outcome.skipped),
                      'failed': len(outcome.failures) + len(outcome.errors)}
            passed = outcome.wasSuccessful()
        else:
            class Counter:
                def pytest_collection_finish(self, session):
                    counts['collected'] = len(session.items)
                def pytest_runtest_logreport(self, report):
                    if report.skipped:
                        counts['skipped'] += 1
                    if report.failed:
                        counts['failed'] += 1
            passed = pytest.main(['-q', '-p', 'no:cacheprovider',
                                  '-o', 'log_file=' + str(scratch/'pytest.log'), str(repo/'tests')],
                                 plugins=[Counter()]) == 0
        expected = payload.get('expected_counts')
        passed = passed and counts['collected'] > 0
        if expected:
            passed = (passed and counts['collected'] == expected['collected']
                      and counts['skipped'] <= expected['skipped'])
        result = {'status': 'passed' if passed else 'failed', 'error': None}
    except BaseException as exc:
        traceback.print_exc()
        result = {'status': 'failed', 'error': repr(exc)[:1000]}
    result.update(counts=counts, test_elapsed=time.monotonic() - start)
    emit(result)


if __name__ == '__main__':
    main()
