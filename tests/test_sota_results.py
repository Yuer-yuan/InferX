"""Result audits must reject attractive but invalid performance evidence."""
import copy
import hashlib
import json

import pytest

from inferx import sota_results as results


def fixture():
    expected={'name':'record_status','arguments':{'ok':True}}
    payload={'messages':[{'role':'user','content':'frozen'}]}
    specs=[];rows=[]
    for i in range(25):
        prompt='tool-24576' if i<24 else 'cold-131072'
        scheduled=i*2 if i<24 else 16
        specification=dict(id=str(i),prompt_id=prompt,payload=payload,expected_tool=expected,offset_s=scheduled)
        specs.append(specification)
        events=[{'choices':[{'delta':{'tool_calls':[{'index':0,'id':'call',
            'function':{'name':'record_status','arguments':'{"ok":true}'}}]},'finish_reason':'tool_calls'}],
            'usage':{'prompt_tokens':24568 if i<24 else 131060,'completion_tokens':64}}]
        summary=dict(ok=True,errors=[],truncated=False,wall_s=2.,ttft_s=.5,
                     prompt_tokens=24568 if i<24 else 131060,completion_tokens=64,
                     cached_tokens=None,cache_mode='natural',finish_reason='tool_calls')
        rows.append(dict(id=str(i),prompt_id=prompt,request=copy.deepcopy(payload),
            request_sha256=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),
            expected_tool=expected,functional_ok=True,events=events,summary=summary,
            scheduled_s=scheduled,dispatched_s=scheduled,finished_s=scheduled+2,dispatch_lag_s=0.,
            latency_s=2.,arrival_ttft_s=.5))
    return specs,rows


def test_reconstructs_tools_instead_of_trusting_functional_flag():
    specs,rows=fixture()
    rows[-1]['events'][0]['choices'][0]['delta']['tool_calls'][0]['function']['arguments']='{"ok":false}'
    with pytest.raises(ValueError,match='tool'):
        results.audit_wave(specs,rows,48.,kind='mixed')


def test_rejects_rehashed_payload_that_differs_from_frozen_specification():
    specs,rows=fixture();rows[-1]['request']['messages'][0]['content']='shortened'
    rows[-1]['request_sha256']=hashlib.sha256(json.dumps(rows[-1]['request'],sort_keys=True).encode()).hexdigest()
    with pytest.raises(ValueError,match='payload'):
        results.audit_wave(specs,rows,48.,kind='mixed')


@pytest.mark.parametrize('failure',['truncated','short_long','missing','duplicate'])
def test_incomplete_long_never_becomes_a_performance_win(failure):
    specs,rows=fixture()
    if failure=='truncated':rows[-1]['summary']['truncated']=True
    elif failure=='short_long':
        rows[-1]['summary']['prompt_tokens']=4096
        rows[-1]['events'][0]['usage']['prompt_tokens']=4096
    elif failure=='missing':rows.pop()
    else:rows[-1]['id']=rows[0]['id']
    with pytest.raises(ValueError):results.audit_wave(specs,rows,48.,kind='mixed')


def test_valid_full_wave_has_continuous_metrics_and_unknown_cache():
    specs,rows=fixture();out=results.audit_wave(specs,rows,48.,kind='mixed')
    assert out['short_p95_s']==2 and out['long_s']==2
    assert out['strict_requests']==25 and out['completed_within_48s']==25
    assert out['completion_tokens']==1600 and out['output_tokens_per_s']==pytest.approx(1600/48)
    assert out['api_cached_tokens']==[None]*25
    assert out['physical_prefill_tokens'] is None


def test_serial_solo_rejects_concurrent_dispatch_with_same_c4_server():
    specs,rows=fixture();specs=specs[:24];rows=rows[:24]
    for spec,row in zip(specs,rows):spec.pop('offset_s');spec['producer']=0;row['producer']=0
    rows[1]['dispatched_s']=1
    with pytest.raises(ValueError,match='serial'):results.audit_wave(specs,rows,48.,kind='isolated-serial-C4')


@pytest.mark.parametrize('index,offset',[(24,0),(3,5.)])
def test_rehashed_wave_cannot_change_long_arrival_or_short_rate(index,offset):
    specs,rows=fixture();specs[index]['offset_s']=offset
    rows[index].update(scheduled_s=offset,dispatched_s=offset,finished_s=offset+2)
    with pytest.raises(ValueError,match='arrival'):
        results.audit_wave(specs,rows,48.,kind='mixed')


def test_repeat_median_uses_wave_identity_and_lane_specific_solo():
    waves=[dict(case='method',group='cold-high',repeat=i,short_p95_s=x,long_s=100+x,
                output_tokens_per_s=10.,completed_within_48s=20,strict_requests=25) for i,x in enumerate([2.,8.,3.])]
    solos=[dict(case='method',group='short-control',short_p95_s=2.),
           dict(case='method',group='cold-high',long_s=100.)]
    out=results.compare_lanes(waves,solos)[0]
    assert out['short_p95_s']==3. and out['short_slowdown']==1.5
    assert out['long_s']==103. and out['long_slowdown']==1.03
    assert out['independent_mixed_repeats']==3 and out['independent_solo_repeats']==1
    with pytest.raises(ValueError,match='repeat'):results.compare_lanes(waves+[waves[0]],solos)
    with pytest.raises(ValueError,match='repeat'):results.compare_lanes(waves[:2],solos)


def test_campaign_export_audits_frozen_input_runtime_and_source_hashes(tmp_path):
    from scripts.analyze_sota_validation import export_campaign
    specs,rows=fixture();case=dict(id='method',concurrency=4,context_window=131328,
        max_total_tokens=131328,morphx={'enabled':False},environment={})
    def write(path,data):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))
    write(tmp_path/'campaign.json',dict(status='restored',restored=True,kind='mixed',repeat=1,
        harness_sha256={},cases=[dict(case='method',repeat=0,group='cold-high',state='measured')]))
    write(tmp_path/'config.json',[case]);write(tmp_path/'restored-service.json',{'pid':123})
    folder=tmp_path/'method-r0-cold-high';wave=folder/'cold-high'
    write(folder/'startup.json',dict(case=case,server_info=dict(context_length=131328,internal_states=[dict(
        effective_max_running_requests_per_dp=4,memory_usage={'token_capacity':131328,'graph':{'decode':0,'prefill':0}},
        mamba_ssm_dtype='bfloat16',max_mamba_cache_size=16,kv_cache_dtype='fp8_e4m3',
        disable_overlap_schedule=True,memory_placeholder=None)])))
    write(folder/'quality-result.json',{'status':'passed'})
    write(wave/'specifications.json',specs);write(wave/'result.json',{'summary':{'measured_span_s':48.}})
    (wave/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (wave/'warmup.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows[:4]))
    (folder/'server.log').write_text('native log\n');(folder/'gpu.csv').write_text('time,23000,1100,99,99,300\n')
    templates={s['prompt_id']:{'payload':s['payload'],'expected_tool':s['expected_tool']} for s in specs}
    output=export_campaign(tmp_path,templates)
    assert output['groups'][0]['strict_requests']==25
    assert output['groups'][0]['gpu_memory']['sampled_min_free_mib']==1100
    assert output['source_sha256']['method-r0-cold-high/cold-high/requests.jsonl']==hashlib.sha256((wave/'requests.jsonl').read_bytes()).hexdigest()
    assert export_campaign(tmp_path,templates)==output
    startup=json.loads((folder/'startup.json').read_text())
    startup['server_info']['internal_states'][0]['memory_usage']['graph']['decode']=.1
    write(folder/'startup.json',startup)
    with pytest.raises(ValueError,match='eager'):export_campaign(tmp_path,templates)
    startup['server_info']['internal_states'][0]['memory_usage']['graph']['decode']=0
    startup['server_info']['internal_states'][0]['speculative_algorithm']='NEXTN'
    write(folder/'startup.json',startup)
    with pytest.raises(ValueError,match='MTP0'):export_campaign(tmp_path,templates)
    startup['server_info']['internal_states'][0]['speculative_algorithm']=None
    write(folder/'startup.json',startup)
    templates['cold-131072']['payload']={'messages':[]}
    with pytest.raises(ValueError,match='frozen'):export_campaign(tmp_path,templates)
