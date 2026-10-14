import hashlib
import json
from pathlib import Path

import pytest

from scripts.analyze_concurrency import export


def write(path, value, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(v)+'\n' for v in value) if lines else json.dumps(value))


def fixture(root):
    config=dict(id='d0-c1',draft=0,concurrency=1)
    failed=dict(id='d7-c8',draft=7,concurrency=8,exploratory=True)
    write(root/'grid.json', dict(configs=[config,failed]))
    write(root/'workloads.json', {})
    write(root/'workload-manifest.json',dict(files=[dict(file=f,sha256=hashlib.sha256((root/f).read_bytes()).hexdigest()) for f in ['grid.json','workloads.json']]))
    write(root/'campaign.json',dict(status='measured',cases=[dict(configuration=config,status='completed'),dict(configuration=failed,status='infrastructure_error',error='headroom')],restored=True,gpu0_healthy=True))
    write(root/'d0-c1/case.json',dict(configuration=config,status='completed'))
    write(root/'d7-c8/case.json',dict(configuration=failed,status='infrastructure_error',error='headroom'))
    directory=root/'d0-c1/scaling-tool'
    write(directory/'result.json',dict(group='scaling-tool',case='d0-c1',requested=2,mode='closed',summary=dict(measured_span_s=4)))
    good=dict(id='ok',prompt_id='tool-512',summary=dict(ok=True,completion_tokens=8,wall_s=1,ttft_s=.2,truncated=False),latency_s=1,arrival_ttft_s=.2,functional_ok=True)
    bad=dict(id='error',prompt_id='tool-512',summary=dict(ok=False,completion_tokens=0,wall_s=3,ttft_s=None,errors=['http503']),latency_s=3,functional_ok=False)
    write(directory/'requests.jsonl',[good,bad],True)
    write(directory/'native.jsonl',[],True)
    return directory


def test_export_is_idempotent_reads_only_raw_and_retains_failures(tmp_path):
    group=fixture(tmp_path)
    destination=tmp_path/'derived'
    first=export(tmp_path,destination)
    write(destination/'fake/scaling-tool/requests.jsonl',[dict(id='irrelevant')],True)
    second=export(tmp_path,destination)
    assert first==second
    assert second['cases'][1]['status']=='infrastructure_error'
    measured=second['groups'][0]
    assert measured['summary']['offered']==2 and measured['summary']['errors']==1
    assert measured['summary']['aggregate_tokens_per_s']==2
    assert measured['summary']['slo']['1']['tool_goodput_per_s']==.25
    assert measured['summary']['latency_s']['p99'] is None
    assert len(second['requests'])==2
    assert 'http503' in second['requests'][1]['errors']
    assert (destination/'groups.csv').exists() and (destination/'requests.csv').exists()
    assert second['roofline']['classification']=='undetermined'


def test_export_rejects_duplicate_requests_and_cases(tmp_path):
    group=fixture(tmp_path)
    rows=[json.loads(line) for line in (group/'requests.jsonl').read_text().splitlines()]
    write(group/'requests.jsonl',rows+[rows[0]],True)
    with pytest.raises(ValueError,match='duplicate request'):
        export(tmp_path,tmp_path/'out')
    write(group/'requests.jsonl',rows,True)
    campaign=json.loads((tmp_path/'campaign.json').read_text())
    campaign['cases'].append(campaign['cases'][0])
    write(tmp_path/'campaign.json',campaign)
    with pytest.raises(ValueError,match='duplicate case'):
        export(tmp_path,tmp_path/'out')


def test_partial_group_and_running_campaign_cannot_become_completed(tmp_path):
    fixture(tmp_path)
    partial=tmp_path/'d0-c1/failed-warmup'
    write(partial/'warmup.jsonl',[dict(id='warm',summary=dict(ok=False,errors=['timeout']))],True)
    result=export(tmp_path,tmp_path/'out')
    assert result['incomplete_groups']==[dict(case='d0-c1',group='failed-warmup',warmup_attempts=1,measurement_attempts=0)]
    campaign=json.loads((tmp_path/'campaign.json').read_text());campaign['status']='running'
    write(tmp_path/'campaign.json',campaign)
    with pytest.raises(ValueError,match='running'):
        export(tmp_path,tmp_path/'out')
    assert export(tmp_path,tmp_path/'out',allow_running=True)['campaign']['status']=='running'


def test_native_class_counters_require_unique_usage_sizes_and_matching_completions(tmp_path):
    group=fixture(tmp_path)
    rows=[json.loads(line) for line in (group/'requests.jsonl').read_text().splitlines()]
    rows[0]['summary']['prompt_tokens']=512
    rows[0]['summary']['cached_tokens']=508
    write(group/'requests.jsonl',rows,True)
    event=dict(event='request_done',request=dict(request_id=1),
        result=dict(completion_tokens=8,prompt_tokens=512,prefix_cache_hit_tokens=508,
                    computed_prefill_tokens=4,prefix_reuse_path='restore_turn_checkpoint'),
        speculative=dict(accepted_tokens=5,drafted_tokens=7,rounds=1),
        timings_seconds=dict(prepare=.01,prefill=.02,decode=.9,ttft=.03,total=1))
    write(group/'native.jsonl',[event],True)
    result=export(tmp_path,tmp_path/'out')
    native=result['groups'][0]['class_native']['tool']
    assert native['matching']=='disjoint_prompt_usage_sizes'
    assert native['observations']['cached_tokens']==508
    assert native['observations']['computed_prefill_tokens']==4
    rows[1].update(prompt_id='cold-512')
    rows[1]['summary'].update(ok=True,prompt_tokens=512,completion_tokens=8)
    write(group/'requests.jsonl',rows,True)
    result=export(tmp_path,tmp_path/'out')
    assert result['groups'][0]['class_native']['tool']['observations'] is None
    assert result['groups'][0]['class_native']['cold']['observations'] is None


def test_telemetry_is_clipped_to_measurement_and_memory_activity_is_not_bandwidth(tmp_path):
    from datetime import datetime,timezone
    group=fixture(tmp_path)
    start=datetime(2026,10,14,13,0,0,tzinfo=timezone.utc).timestamp()*1000
    result=json.loads((group/'result.json').read_text());result.update(start_unix_ms=start,end_unix_ms=start+1000)
    write(group/'result.json',result)
    (group.parent/'gpu.csv').write_text(
        '2026/10/14 20:59:59.000, 20000, 4000, 0, 0, 2500, 10251, 100, 50\n'
        '2026/10/14 21:00:00.000, 20000, 4000, 100, 80, 2700, 10251, 350, 65\n'
        '2026/10/14 21:00:00.500, 20000, 4000, 100, 90, 2700, 10251, 370, 66\n'
        '2026/10/14 21:00:02.000, 20000, 4000, 0, 0, 2500, 10251, 100, 50\n')
    result=export(tmp_path,tmp_path/'out')
    telemetry=result['groups'][0]['gpu_telemetry']
    assert telemetry['samples']==2
    assert telemetry['memory_activity_percent']['median']==85
    assert telemetry['gpu_util_percent']['median']==100
    assert 'GB/s' in telemetry['limitation']
    assert 'd0-c1/gpu.csv' in result['source_hashes']


def test_export_rechecks_tool_arguments_and_rejects_truncation(tmp_path):
    group=fixture(tmp_path)
    rows=[json.loads(line) for line in (group/'requests.jsonl').read_text().splitlines()]
    expected=dict(name='record_status',arguments=dict(status='ready'))
    rows[0]['expected_tool']=expected
    rows[0]['events']=[dict(choices=[dict(delta=dict(tool_calls=[dict(index=0,function=dict(name='record_status',arguments='{"status":"ready"}'))]))])]
    rows[0]['summary']['truncated']=True
    write(group/'requests.jsonl',rows,True)
    result=export(tmp_path,tmp_path/'out')
    assert result['groups'][0]['summary']['tool_valid']==0
    assert result['requests'][0]['functional_ok'] is False
    rows[0]['summary']['truncated']=False
    rows[0]['events'][0]['choices'][0]['delta']['tool_calls'][0]['function']['arguments']='{"status":"wrong"}'
    write(group/'requests.jsonl',rows,True)
    assert export(tmp_path,tmp_path/'out')['groups'][0]['summary']['tool_valid']==0
    rows[0]['events'][0]['choices'][0]['delta']['tool_calls'][0]['function']['arguments']='{"status":"ready"}'
    write(group/'requests.jsonl',rows,True)
    assert export(tmp_path,tmp_path/'out')['groups'][0]['summary']['tool_valid']==1
