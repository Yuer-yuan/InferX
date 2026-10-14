import json
import os
import subprocess
import sys


def test_http_failure_is_not_a_failed_generated_program(tmp_path):
    directory = tmp_path / 'tuning'
    directory.mkdir()
    row = {'kind': 'trial', 'config': 'rejected', 'prompt_id': 'code_merge', 'repetition': 0,
           'summary': {'ok': False, 'errors': ['HTTP 400: invalid request'],
                       'text': '', 'truncated': False}}
    (directory / 'bad-dev.jsonl').write_text(json.dumps(row) + '\n')
    result = subprocess.run([sys.executable, '-m', 'scripts.check_tuning_quality', '*-dev.jsonl'],
                            env={**os.environ, 'INFERX_ROOT': str(tmp_path)},
                            text=True, capture_output=True, check=True)
    rows = json.loads(result.stdout)
    assert rows[0]['status'] == 'infrastructure_error'
    assert rows[0]['error'] == ['HTTP 400: invalid request']
