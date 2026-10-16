"""Bounded client execution and independent verification against pristine tests."""
import os
import collections
import json
import signal
import subprocess
import time
from pathlib import Path

from inferx.repository import safe_relative
from inferx.repo_verifier import verify_repo


def run_owned_process(command: list[str], *, cwd: Path, stdout_path: Path,
                      stderr_path: Path, timeout: float, env: dict) -> dict:
    if timeout <= 0:
        raise ValueError('timeout must be positive')
    started = time.monotonic()
    with stdout_path.open('x') as out, stderr_path.open('x') as err:
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                stdout=out, stderr=err,
                                start_new_session=True)
        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            # proc is still an unreaped child: this session/group ID cannot be reused.
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    return dict(returncode=proc.returncode, timed_out=timed_out,
                wall_s=time.monotonic()-started, pid=proc.pid)


def assess_agent_run(returncode: int, timed_out: bool, events: dict,
                     verification_status: str, changed_source: bool) -> str:
    if timed_out:
        return 'timeout'
    if returncode or events['errors'] or not events.get('terminal', True):
        return 'client_error'
    if not any(events['tool_counts'].values()):
        return 'no_tool_activity'
    if not changed_source:
        return 'no_source_change'
    if verification_status == 'infrastructure_error':
        return 'infrastructure_error'
    return 'passed' if verification_status == 'passed' else 'quality_failed'


async def verify_agent_edit(workspace: Path, task: dict, *, fixture_root: Path,
                            python: str, work_root: Path, cpu: int) -> dict:
    target = workspace / safe_relative(task['repo_name'], task['file_name'])
    if target.is_symlink() or not target.is_file():
        return dict(status='infrastructure_error', error='target missing or symlink')
    full_source_task = dict(task, prefix_code='', suffix_code='')
    return await verify_repo(target.read_text(), full_source_task, fixture_root=fixture_root,
                             python=python, work_root=work_root, cpu=cpu, timeout=30)


def parse_agent_events(client: str, text: str) -> dict:
    if client not in {'opencode', 'pi', 'claude'}:
        raise ValueError('unknown client')
    counts = collections.Counter()
    event_counts = collections.Counter()
    errors, seen, calls = [], set(), {}
    terminal, tool_errors = False, 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            errors.append('malformed JSON event')
            continue
        kind = row.get('type')
        event_counts[kind] += 1
        tool_id, tool_name = None, None
        if kind == 'error':
            errors.append(str(row.get('error', row)))
        if client == 'opencode':
            part = row.get('part', {})
            if kind == 'tool_use':
                state = part.get('state', {}).get('status')
                if state == 'completed':
                    tool_id, tool_name = part.get('callID', part.get('id')), part.get('tool')
                elif state == 'error':
                    tool_errors += 1
            if kind == 'step_finish' and part.get('reason') == 'stop':
                terminal = True
        elif client == 'pi':
            if kind == 'tool_execution_end':
                if row.get('isError'):
                    tool_errors += 1
                else:
                    tool_id, tool_name = row.get('toolCallId'), row.get('toolName')
            if kind == 'message_end' and row.get('message', {}).get('stopReason') == 'error':
                errors.append(row['message'].get('errorMessage', 'model error'))
            if kind == 'agent_settled':
                terminal = not row.get('aborted', False)
        else:
            content = row.get('message', {}).get('content', [])
            if not isinstance(content, list):
                content = []
            for block in content:
                if kind == 'assistant' and block.get('type') == 'tool_use':
                    calls[block['id']] = block['name']
                if kind == 'user' and block.get('type') == 'tool_result':
                    identity = block.get('tool_use_id')
                    if block.get('is_error'):
                        tool_errors += 1
                    elif identity in calls and identity not in seen:
                        seen.add(identity)
                        counts[calls[identity].lower()] += 1
            if kind == 'result':
                terminal = not row.get('is_error', False)
                if not terminal:
                    errors.append(str(row.get('result', row.get('subtype'))))
        if tool_id and tool_name and tool_id not in seen:
            seen.add(tool_id)
            counts[tool_name.lower()] += 1
    return dict(tool_counts=dict(counts), event_counts=dict(event_counts), errors=errors,
                terminal=terminal, tool_errors=tool_errors)
