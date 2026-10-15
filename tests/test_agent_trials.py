import json
import os
import subprocess
import sys
import time
from pathlib import Path

from inferx.agent_trials import assess_agent_run, run_owned_process


def test_exit_zero_claim_cannot_replace_pristine_tests_or_tool_activity():
    good = dict(tool_counts={'read': 1, 'edit': 1, 'bash': 1}, errors=[])
    assert assess_agent_run(0, False, good, 'passed', True) == 'passed'
    assert assess_agent_run(0, False, good, 'failed', True) == 'quality_failed'
    assert assess_agent_run(0, False, {'tool_counts': {}, 'errors': []}, 'passed', True) == 'no_tool_activity'
    assert assess_agent_run(0, False, good, 'passed', False) == 'no_source_change'
    assert assess_agent_run(1, False, good, 'passed', True) == 'client_error'
    assert assess_agent_run(0, False, {'tool_counts': good['tool_counts'], 'errors': ['HTTP400']}, 'passed', True) == 'client_error'
    assert assess_agent_run(0, True, good, 'passed', True) == 'timeout'


def test_timeout_kills_owned_descendants_and_preserves_logs(tmp_path):
    other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    program = ("import subprocess,sys,time; "
               "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
               "print(p.pid,flush=True); time.sleep(60)")
    try:
        result = run_owned_process([sys.executable, '-c', program], cwd=tmp_path,
            stdout_path=tmp_path/'stdout', stderr_path=tmp_path/'stderr', timeout=.3,
            env=dict(os.environ))
        assert result['timed_out'] and result['returncode'] != 0
        child_pid = int((tmp_path/'stdout').read_text().strip())
        path = Path(f'/proc/{child_pid}/stat')
        # Container init may reap the killed grandchild asynchronously.
        for _ in range(50):
            try:
                state = path.read_text().rsplit(')',1)[1].split()[0]
            except (FileNotFoundError, ProcessLookupError):
                break
            if state == 'Z':
                break
            time.sleep(.02)
        else:
            raise AssertionError('owned descendant survived timeout')
        assert other.poll() is None
        assert (tmp_path/'stderr').exists()
    finally:
        other.terminate()
        other.wait(timeout=5)


def test_changed_agent_tests_cannot_make_wrong_source_pass(tmp_path):
    import asyncio
    import shutil
    from inferx.agent_trials import verify_agent_edit
    fixture = tmp_path/'fixtures'/'demo'
    (fixture/'tests').mkdir(parents=True)
    (fixture/'pkg.py').write_text('def value():\n    return 1\n')
    (fixture/'tests'/'test_pkg.py').write_text('from pkg import value\ndef test_value():\n    assert value() == 1\n')
    workspace = tmp_path/'workspace'
    shutil.copytree(fixture, workspace)
    (workspace/'pkg.py').write_text('def value():\n    return 2\n')
    (workspace/'tests'/'test_pkg.py').write_text('from pkg import value\ndef test_value():\n    assert value() == 2\n')
    task = dict(repo_name='demo', file_name='/demo/pkg.py', test_mode='pytest',
                expected_counts={'collected': 1, 'skipped': 0, 'failed': 0})
    result = asyncio.run(verify_agent_edit(workspace, task, fixture_root=fixture.parent,
        python=sys.executable, work_root=tmp_path/'verify', cpu=min(os.sched_getaffinity(0))))
    assert result['status'] == 'failed'


def test_client_cannot_consume_controller_stdin(tmp_path):
    program = '''
import os,sys
from pathlib import Path
from inferx.agent_trials import run_owned_process
p=Path(sys.argv[1])
r=run_owned_process([sys.executable,'-c','import sys; print(repr(sys.stdin.read()))'],
    cwd=p, stdout_path=p/'out', stderr_path=p/'err', timeout=2, env=dict(os.environ))
assert not r['timed_out']
'''
    subprocess.run([sys.executable, '-c', program, str(tmp_path)],
                   input='controller commands must stay private\n', text=True, check=True)
    assert (tmp_path/'out').read_text().strip() == "''"


def test_native_event_parsers_count_completed_tools_and_terminal():
    from inferx.agent_trials import parse_agent_events
    examples = {
        'opencode': [dict(type='tool_use',part=dict(tool='read',callID='a',state=dict(status='completed'))),
                     dict(type='step_finish',part=dict(reason='stop'))],
        'pi': [dict(type='tool_execution_end',toolName='read',toolCallId='a',isError=False),
               dict(type='agent_settled',aborted=False)],
        'claude': [dict(type='assistant',message=dict(content=[dict(type='tool_use',id='a',name='Read')])),
                   dict(type='user',message=dict(content=[dict(type='tool_result',tool_use_id='a',content='file')])),
                   dict(type='result',is_error=False,subtype='success')],
    }
    for client, events in examples.items():
        parsed = parse_agent_events(client, '\n'.join(json.dumps(e) for e in events))
        assert parsed['tool_counts'] == {'read': 1}
        assert parsed['terminal'] and not parsed['errors']
        duplicate = parse_agent_events(client, '\n'.join(json.dumps(e) for e in events + events))
        assert duplicate['tool_counts'] == {'read': 1}
    assert not parse_agent_events('pi','')['terminal']
    parsed = parse_agent_events('opencode',json.dumps(dict(type='error',error={'message':'HTTP400'})))
    assert parsed['errors']
    assert assess_agent_run(0,False,dict(tool_counts={'read':1},errors=[],terminal=False),'passed',True)=='client_error'
