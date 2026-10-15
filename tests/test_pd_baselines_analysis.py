"""Keep load errors, partial experiments and request fairness visible in exports."""
import hashlib
import json
from pathlib import Path

import pytest


def write(path, value, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r)+'\n' for r in value) if lines else json.dumps(value))


def make_group(root, repetition=0, rate=.5, with_long=True, cached=None,run_name='main'):
    run=root/run_name
    case=dict(id='engine',engine='ninfer',draft=0,concurrency=4,quantization='q4',
              kv_capacity=212992,kv_dtype='rk4v4-e8',morphx={'enabled':False})
    group='cold-low'
    statuses=[]
    if (run/'campaign.json').exists():
        statuses=json.loads((run/'campaign.json').read_text())['cases']
    statuses.append(dict(case='engine',repetition=repetition,group=group,state='measured'))
    write(run/'config.json',[case])
    write(run/'campaign.json',dict(status='restored',restored=True,cases=statuses,
        workload_definitions=[dict(name=group,rate=rate,long_mode='cold')]))
    directory=run/f'engine-r{repetition}-{group}'
    write(directory/'quality-result.json',{'status':'passed'})
    raw=directory/group
    expected=dict(name='record_status',arguments={'status':'ready'})
    good=dict(id='short',prompt_id='tool-24576',summary=dict(ok=True,completion_tokens=5,
        wall_s=1,ttft_s=.1,truncated=False,prompt_tokens=24557,cached_tokens=cached,errors=[]),
        latency_s=1,arrival_ttft_s=.1,functional_ok=True,expected_tool=expected,
        events=[dict(choices=[dict(delta=dict(tool_calls=[dict(index=0,
            function=dict(name='record_status',arguments='{"status":"ready"}'))]))])],finished_s=1)
    bad=dict(id='reject',prompt_id='tool-24576',summary=dict(ok=False,completion_tokens=0,
        wall_s=2,ttft_s=None,truncated=False,prompt_tokens=None,cached_tokens=None,errors=['HTTP429']),
        latency_s=2,arrival_ttft_s=None,functional_ok=False,finished_s=2)
    rows=[good,bad]
    if with_long:
        long=dict(good,id='long',prompt_id='cold-131072',summary=dict(good['summary'],
            prompt_tokens=131046,cached_tokens=cached,wall_s=8),latency_s=8,finished_s=24)
        rows.append(long)
    write(raw/'specifications.json',[dict(id=r['id'],prompt_id=r['prompt_id']) for r in rows])
    write(raw/'requests.jsonl',rows,True)
    write(raw/'result.json',dict(summary={'measured_span_s':48}))
    return run,raw


def test_export_is_raw_only_idempotent_and_retains_429_and_long_completion(tmp_path):
    from scripts.analyze_pd_baselines import export
    make_group(tmp_path)
    destination=tmp_path/'derived'
    first=export(tmp_path,destination)
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.glob('*') if p.is_file()}
    write(destination/'fake/campaign.json',dict(status='restored',cases=[]))
    second=export(tmp_path,destination)
    after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.glob('*') if p.is_file()}
    assert first==second and before==after
    group=second['groups'][0]
    assert group['summary']['offered']==3 and group['summary']['errors']==1
    assert group['short']['tool_valid']==1 and group['long']['protocol_ok']==1
    assert group['short']['slo']['5']['tool_goodput_per_s']==1/48
    assert group['prefix']['long']['cached_tokens'] is None
    assert group['prefix']['long']['status']=='unobserved'
    assert len(second['requests'])==3 and second['requests'][1]['errors']==['HTTP429']


def test_partial_and_running_data_cannot_become_final_or_offered_counts(tmp_path):
    from scripts.analyze_pd_baselines import export
    run,group=make_group(tmp_path)
    (group/'result.json').unlink()
    with (group/'requests.jsonl').open('a') as out:out.write('{"unfinished":')
    campaign=json.loads((run/'campaign.json').read_text());campaign['status']='running'
    write(run/'campaign.json',campaign)
    with pytest.raises(ValueError,match='running'):
        export(tmp_path,tmp_path/'out')
    result=export(tmp_path,tmp_path/'out',allow_running=True)
    measured=result['groups'][0]
    assert measured['complete'] is False and measured['summary'] is None
    assert measured['observed_responses']==3 and measured['planned']==3
    assert measured['parse_errors'] and measured['offered'] is None


def test_duplicate_request_identity_is_rejected(tmp_path):
    from scripts.analyze_pd_baselines import export
    _,group=make_group(tmp_path)
    row=(group/'requests.jsonl').read_text().splitlines()[0]
    with (group/'requests.jsonl').open('a') as out:out.write(row+'\n')
    with pytest.raises(ValueError,match='duplicate request'):
        export(tmp_path,tmp_path/'out')


def test_repeat_summary_counts_independent_groups_and_does_not_merge_rates(tmp_path):
    from scripts.analyze_pd_baselines import export
    make_group(tmp_path,repetition=0)
    make_group(tmp_path,repetition=1)
    make_group(tmp_path,repetition=2)
    make_group(tmp_path,rate=2.5,run_name='high-rate')
    result=export(tmp_path,tmp_path/'out')
    assert len(result['repeats'])==2
    aggregate=next(r for r in result['repeats'] if r['rate']==.5)
    assert aggregate['independent_repetitions']==3
    assert aggregate['long_completed']==3
    assert aggregate['short_errors']==3
    assert aggregate['short_p95_s']['n']==0  # Too few requests in each wave for P95.


def test_truncated_tool_and_rejected_long_cannot_be_counted_as_success(tmp_path):
    from scripts.analyze_pd_baselines import export
    _,group=make_group(tmp_path)
    rows=[json.loads(line) for line in (group/'requests.jsonl').read_text().splitlines()]
    rows[0]['summary']['truncated']=True
    rows[-1]['summary'].update(ok=False,completion_tokens=0,errors=['HTTP429'])
    write(group/'requests.jsonl',rows,True)
    result=export(tmp_path,tmp_path/'out')
    assert result['groups'][0]['short']['tool_valid']==0
    assert result['repeats'][0]['long_completed']==0
    assert result['repeats'][0]['long_offered']==1


def test_gpu_memory_samples_keep_sampling_scope_and_invalid_values_visible(tmp_path):
    from scripts.analyze_pd_baselines import export
    _,group=make_group(tmp_path)
    path=group.parent/'gpu.csv'
    raw=('2026/10/15 12:00:00.000, 22000, 2108, 90, 40, 250\n'
         '2026/10/15 12:00:00.200, 23500, 608, 99, 60, 300\n'
         '2026/10/15 12:00:00.400, nan, N/A, 99, 60, 300\n')
    path.write_text(raw)
    result=export(tmp_path,tmp_path/'out')
    observed=result['groups'][0]['gpu_memory']
    assert observed['status']=='partial' and observed['valid_samples']==2
    assert observed['invalid_rows']==1 and observed['sampled_peak_mib']==23500
    assert observed['sampled_min_free_mib']==608
    assert observed['sampling_interval_ms']==200
    assert 'startup excluded' in observed['scope']
    assert result['source_hashes']['main/engine-r0-cold-low/gpu.csv']['sha256']==hashlib.sha256(raw.encode()).hexdigest()
    path.unlink()
    absent=export(tmp_path,tmp_path/'out')['groups'][0]['gpu_memory']
    assert absent['status']=='unobserved' and absent['sampled_peak_mib'] is None
