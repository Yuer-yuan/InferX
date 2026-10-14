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
