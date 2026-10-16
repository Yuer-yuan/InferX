"""A vLLM export must audit actual runtime, full inputs, conditioning and source."""
import copy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from test_sota_results import fixture
from inferx.ebplus_comparison import comparison_cases,comparison_specs
from inferx.pd_baselines import engine_command


def comparison_fixture(root,lane=0):
    repo=Path(__file__).parents[1];artifact=Path('/pinned')
    plan=json.loads((repo/'configs/ebplus-validation.json').read_text())
    hardware=json.loads((repo/'results/ebplus-validation/calibration/hardware-v4/hardware-result.json').read_text())
    case=comparison_cases(plan)[lane]
    mb=json.loads((repo/'results/ebplus-validation/calibration/mb-grid-v1/mb-grid-result.json').read_text())
    if lane:case['environment'].update(mb['fit']['env_vars'])
    def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    write(root/'plan.json',plan)
    write(root/'comparison-definition.json',dict(phase='diagnostic',kind='mixed',repeats=1,case_ids=[case['id']]))
    write(root/'config.json',[case])
    write(root/'campaign.json',dict(status='restored',restored=True,harness_sha256={'inferx/measured.py':hashlib.sha256(b'actual measured source').hexdigest()},cases=[]))
    (root/'harness/inferx').mkdir(parents=True);(root/'harness/inferx/measured.py').write_bytes(b'actual measured source')
    write(root/'restored-service.json',dict(pid=123))
    for source,target in [(repo/'results/ebplus-validation/calibration/hardware-v4',root/'hardware'),
                          (repo/'results/ebplus-validation/calibration/mb-grid-v1',root/'mb')]:
        target.mkdir()
        for name in (['hardware-result.json','pd-hardware-parameters.json'] if target.name=='hardware' else ['mb-grid-result.json']):
            shutil.copy2(source/name,target/name)
    shutil.copytree(repo/'results/ebplus-validation/reproduction/adapter-v4',root/'overlay')
    _,original=fixture();templates={}
    for name in ['tool-24576','cold-131072','resume-131072']:
        templates[name]=dict(payload=copy.deepcopy(original[0]['request']),expected_tool=original[0]['expected_tool'])
    campaign=json.loads((root/'campaign.json').read_text())
    for group in ['cold-low','resume-high']:
        identity=f'{case["id"]}-r0-{group}';directory=root/identity;wave=directory/group
        native=copy.deepcopy(case);native['environment'].update(PYTHONPATH='/pinned/ebplus-validation/overlays/adapter-v4:/pinned/ebplus-validation/code-task3-v3',
            INFERX_VLLM_RUNTIME=str(artifact/'ebplus-validation/raw'/root.name/identity/'native-runtime.json'))
        if lane:
            native['environment'].update(VLLM_PD_CALIBRATION_FILE='/pinned/ebplus-validation/raw/hardware-v4/pd-hardware-parameters.json',
                INFERX_EBPLUS_EVIDENCE=str(artifact/'ebplus-validation/raw'/root.name/identity/'policy-evidence.json'),VLLM_COLLECT_SCHEDULE_STATS='0')
            inputs=dict(hardware['parameters'],**{'mb_cost_'+k:v for k,v in mb['fit']['coefficients'].items()},
                mu_L=24557.,mu_o=64.,N_obs=2.,theta0=.5,alpha_mb=0.,delta=.0001,cooldown_before=0,active_before='mb')
            evidence=dict(mode='auto',active_scheduler='mb',evaluations=1,parameters_initialized=True,mb_cost_profiled=True,
                execution_counts=dict(mb=100,prefill=0,decode=0),completed=100,max_num_running_reqs=4,num_gpu_blocks=98,last_evaluation_inputs=inputs)
            if lane==3:
                # Actual diagnostic inputs yield margin6.69e-6: above the author's1e-6 knob.
                inputs.update(mu_L=24399.259061176508,mu_o=56.95,N_obs=3.9699999961217456,theta0=.01,delta=.000001)
                evidence['active_scheduler']='eb'
                evidence['execution_counts'].update(prefill=200,decode=100)
            write(directory/'policy-before.json',evidence);write(directory/'policy-evidence.json',evidence)
        write(directory/'startup.json',dict(case=native,args=engine_command(native,artifact,artifact/'ebplus-validation/raw'/root.name/identity)))
        runtime=dict(model='/pinned/models/qwen38-int4-redhat',context_window=131328,concurrency=4,max_active=4,
            chunk=4096,long_prefill_token_threshold=2048,async_scheduling=False,enforce_eager=True,cudagraph_mode='NONE',
            speculative=False,quantization='compressed-tensors',kv_dtype='fp8',mamba_cache_mode='align',
            **{k:hardware['native_runtime'][k] for k in ['block_size','num_gpu_blocks','kv_groups']})
        write(directory/'native-runtime.json',runtime);write(directory/'quality-result.json',{'status':'passed'})
        (directory/'server.log').write_text('actual server log\n');(directory/'gpu.csv').write_text('time,23000,1100,99,99,300\n')
        specs=comparison_specs(templates,group,identity,kind='mixed');rows=[]
        for i,spec in enumerate(specs):
            r=copy.deepcopy(original[i]);r.update(id=spec['id'],prompt_id=spec['prompt_id'])
            count=24557 if i<24 else (131049 if group=='cold-low' else 131038)
            r['summary']['prompt_tokens']=r['events'][0]['usage']['prompt_tokens']=count
            rows.append(r)
        write(wave/'specifications.json',specs);write(wave/'result.json',{'span_s':48.,'started_unix_ms':1000.,'ended_unix_ms':49000.})
        (wave/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        for label,n in [('warmup',4),('policy-prime',100)]:
            warm=[dict(copy.deepcopy(rows[0]),id=f'{label}-{i}') for i in range(n)]
            (directory/(label+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in warm))
        campaign['cases'].append(dict(case=case['id'],group=group,repeat=0,identity=identity,state='measured'))
    write(root/'campaign.json',campaign)
    return templates


def test_vllm_export_reconstructs_full_counts_and_rejects_unmatched_runtime(tmp_path):
    from scripts import analyze_ebplus_comparison as api
    templates=comparison_fixture(tmp_path)
    out=api.export_campaign(tmp_path,templates)
    assert len(out['groups'])==2
    assert out['groups'][0]['strict_requests']==25
    assert out['groups'][0]['short_p95_s']==2.
    p=tmp_path/'native-r0-cold-low/native-runtime.json';runtime=json.loads(p.read_text())
    runtime['speculative']=True;p.write_text(json.dumps(runtime))
    with pytest.raises(ValueError,match='runtime'):api.export_campaign(tmp_path,templates)


@pytest.mark.parametrize('change',['priming','source','frozen','input-count','missing-wave'])
def test_vllm_export_rejects_incomplete_history_or_rehashed_bad_inputs(tmp_path,change):
    from scripts import analyze_ebplus_comparison as api
    templates=comparison_fixture(tmp_path);folder=tmp_path/'native-r0-cold-low'
    if change=='priming':
        p=folder/'policy-prime.jsonl';p.write_text('\n'.join(p.read_text().splitlines()[:-1])+'\n')
    elif change=='source':(tmp_path/'harness/inferx/measured.py').write_text('different')
    elif change=='frozen':templates['cold-131072']['payload']={'messages':[]}
    elif change=='input-count':
        p=folder/'cold-low/requests.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()]
        rows[-1]['summary']['prompt_tokens']=rows[-1]['events'][0]['usage']['prompt_tokens']=131048
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    else:
        p=tmp_path/'campaign.json';d=json.loads(p.read_text());d['cases'].pop();p.write_text(json.dumps(d))
    with pytest.raises(ValueError):api.export_campaign(tmp_path,templates)


def test_automatic_export_binds_policy_to_measured_coefficients(tmp_path):
    from scripts import analyze_ebplus_comparison as api
    templates=comparison_fixture(tmp_path,lane=1)
    out=api.export_campaign(tmp_path,templates)
    assert out['groups'][0]['policy']['decision']['expected_active']=='mb'
    p=tmp_path/'auto-default-r0-cold-low/policy-evidence.json';d=json.loads(p.read_text())
    d['last_evaluation_inputs']['beta_p']=0.;p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='calibration'):api.export_campaign(tmp_path,templates)


def test_author_tuned_automatic_export_uses_its_actual_hysteresis_and_phase_evidence(tmp_path):
    from scripts import analyze_ebplus_comparison as api
    templates=comparison_fixture(tmp_path,lane=3)
    out=api.export_campaign(tmp_path,templates)
    assert out['groups'][0]['policy']['decision']['expected_active']=='eb'
    p=tmp_path/'auto-tuned-r0-cold-low/policy-evidence.json';d=json.loads(p.read_text())
    d['execution_counts'].update(prefill=0,decode=0);p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='exclusive'):api.export_campaign(tmp_path,templates)
