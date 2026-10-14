import json
import os
import subprocess
import sys


def test_quality_and_performance_budgets_are_not_pooled(tmp_path):
    directory = tmp_path / 'tuning'
    directory.mkdir()
    for suite, budget, speed in [('heldout', 1024, 100.), ('quality', 2048, 10.)]:
        row = dict(kind='trial', config='same-engine', prompt_id='same-prompt',
                   repetition=0, concurrency=1, index=0, wave_s=2., request_sha256='fixture',
                   request={'max_tokens': budget},
                   summary=dict(ok=True, text='answer', reasoning='', tool_calls=[], errors=[],
                                truncated=False, cache_mode='cold', observed_decode_tps=speed,
                                e2e_tps=speed, wall_s=2., ttft_s=.1, cached_tokens=0))
        (directory / f'same-engine-{suite}.jsonl').write_text(json.dumps(row) + '\n')
    subprocess.run([sys.executable, '-m', 'scripts.export_tuning_results'], check=True,
                   env={**os.environ, 'INFERX_ROOT': str(tmp_path)}, capture_output=True)
    groups = json.loads((directory / 'summary-current.json').read_text())
    assert len(groups) == 2
    assert {g['output_budget'] for g in groups} == {1024, 2048}
    assert all(g['n'] == 1 for g in groups)
