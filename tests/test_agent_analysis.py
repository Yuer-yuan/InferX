import json
import os
import subprocess
import sys
from pathlib import Path


def prepare_export(tmp_path):
    directory = tmp_path / 'context-agents'
    directory.mkdir()
    rows = [dict(client=client, task_id=f'task-{i}', status='passed', wall_s=10,
                 server_completion_tokens=20, server_prompt_tokens=100,
                 server_cached_tokens=50, server_max_prompt_tokens=100)
            for client in ['opencode', 'pi', 'claude'] for i in range(3)]
    (directory / 'trials-all.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in rows))
    for capacity, kv in [(131072, 'int8'), (262144, 'rk4v4-e8')]:
        name = f'context-{capacity}-{kv}'
        (directory / f'{name}-result.json').write_text('{}')
        (directory / f'{name}-lookup.jsonl').write_text(json.dumps(dict(
            summary=dict(prompt_tokens=10, ttft_s=1, completion_tokens=5))) + '\n')
        (directory / f'{name}-heldout.jsonl').write_text(''.join(
            json.dumps(dict(prompt_id=task, summary=dict(observed_decode_tps=20,
                                                       truncated=False))) + '\n'
            for task in ['code_lru', 'code_toposort', 'reason_probability', 'prose_migration']))
    for name in ['runtime.json', 'clients.json', 'tasks-manifest.json']:
        (directory / name).write_text('{}')
    return directory, rows


def export(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts/analyze_agent_trials.py'
    return subprocess.run([sys.executable, str(script)],
                          env={**os.environ, 'INFERX_ROOT': str(tmp_path)},
                          text=True, capture_output=True)


def test_repeated_export_preserves_nine_trials_and_aggregates(tmp_path):
    directory, _ = prepare_export(tmp_path)
    first = export(tmp_path)
    assert first.returncode == 0, first.stderr
    before = {name: (directory / name).read_bytes()
              for name in ['trials-compact.jsonl', 'summary.json', 'raw-manifest.json']}
    second = export(tmp_path)
    assert second.returncode == 0, second.stderr
    after = {name: (directory / name).read_bytes() for name in before}
    assert after == before
    assert json.loads(after['summary.json'])['trial_count'] == 9


def test_export_rejects_duplicate_raw_client_task_identity(tmp_path):
    directory, rows = prepare_export(tmp_path)
    (directory / 'trials-pi.jsonl').write_text(json.dumps(rows[3]) + '\n')
    result = export(tmp_path)
    assert result.returncode != 0
    assert 'duplicate client/task identity' in result.stderr
    assert not (directory / 'summary.json').exists()
