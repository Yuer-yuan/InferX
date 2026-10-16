"""Strict exports for calibrated vLLM EB+; legacy SGLang audit stays unchanged."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from inferx.ebplus_calibration import validate_hardware_result,fit_mb_cost,qualify_tool_rows
from inferx.ebplus_comparison import comparison_cases,comparison_specs,audit_native_runtime,audit_policy,crossover_decision
from inferx.ebplus_validation import AUTHOR_SHA256,NATIVE_SHA256,CALIBRATION_SHA256
from inferx.pd_baselines import engine_command,load_frozen_workloads
from inferx.sota_results import audit_wave,compare_lanes,verify_prompt_count_contract
from scripts.analyze_pd_baselines import sampled_gpu_memory


def export_campaign(root,templates):
    root=Path(root);sources={}
    def data(relative):
        relative=Path(relative)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('invalid source path')
        path=root/relative
        if path.is_symlink():raise ValueError('symlink source')
        raw=path.read_bytes();sources[str(relative)]=hashlib.sha256(raw).hexdigest();return raw
    def read(relative):return json.loads(data(relative))
    def rows(relative):return [json.loads(line) for line in data(relative).splitlines() if line.strip()]
    campaign=read('campaign.json');definition=read('comparison-definition.json');plan=read('plan.json')
    if campaign['status']!='restored' or campaign.get('restored') is not True:raise ValueError('campaign not restored/terminal')
    read('restored-service.json')
    actual={str(p.relative_to(root/'harness')) for p in (root/'harness').rglob('*.py')}
    if actual!=set(campaign['harness_sha256']):raise ValueError('source inventory mismatch')
    for name,digest in campaign['harness_sha256'].items():
        if hashlib.sha256(data(Path('harness')/name)).hexdigest()!=digest:raise ValueError('source hash mismatch')
    hardware=read('hardware/hardware-result.json');validate_hardware_result(hardware,plan)
    params=read('hardware/pd-hardware-parameters.json')
    if hardware['status']!='complete' or any(params[k]!=v for k,v in hardware['parameters'].items()):
        raise ValueError('hardware calibration mismatch')
    mb=read('mb/mb-grid-result.json')
    if mb['identity']!=plan['identity'] or mb['status']!='complete' or fit_mb_cost(mb['rows'],plan)!=mb['fit']:
        raise ValueError('MB calibration mismatch')
    manifest=read('overlay/manifest.json')
    overlay_name=manifest.get('overlay_directory_name','adapter-v4')
    if not isinstance(overlay_name,str) or Path(overlay_name).name!=overlay_name or overlay_name in {'.','..',''}:
        raise ValueError('invalid overlay source directory')
    if (manifest['author_sha256']!=AUTHOR_SHA256 or manifest['native_sha256']!=NATIVE_SHA256 or
        manifest['calibration_sha256']!=CALIBRATION_SHA256 or
        set(manifest['files'])!={'inferx_ebplus_policy.py','inferx_ebplus_scheduler.py','LICENSE'}):
        raise ValueError('overlay source inventory mismatch')
    for name,digest in manifest['files'].items():
        if hashlib.sha256(data(Path('overlay')/name)).hexdigest()!=digest:raise ValueError('overlay source hash mismatch')
    expected={c['id']:c for c in comparison_cases(plan)};cases=read('config.json')
    if len({c['id'] for c in cases})!=len(cases) or [c['id'] for c in cases]!=definition['case_ids']:
        raise ValueError('comparison configuration identity mismatch')
    for case in cases:
        target=copy.deepcopy(expected[case['id']])
        if case['id']!='native':target['environment'].update(mb['fit']['env_vars'])
        if target!=case:raise ValueError('comparison configuration differs from frozen calibrated case')
    kind=definition['kind'];phase=definition['phase'];repeat_count=definition['repeats']
    if phase not in {'diagnostic','formal','solo'} or kind!=('isolated-serial-C4' if phase=='solo' else 'mixed') or repeat_count!=(3 if phase=='formal' else 1):
        raise ValueError('unfrozen phase/kind/repeats')
    group_names=['short-control','cold-low','resume-high'] if kind=='isolated-serial-C4' else ['cold-low','resume-high']
    required={(c['id'],g,r) for c in cases for g in group_names for r in range(repeat_count)}
    observed=[(s['case'],s['group'],s['repeat']) for s in campaign['cases']]
    if len(set(observed))!=len(observed) or set(observed)!=required:raise ValueError('missing or duplicate independent wave')
    case_map={c['id']:c for c in cases};groups=[]
    for state in campaign['cases']:
        if state['state']!='measured':raise ValueError('unqualified/failed wave')
        case=case_map[state['case']];group=state['group'];repeat=state['repeat'];identity=f'{case["id"]}-r{repeat}-{group}'
        if state['identity']!=identity:raise ValueError('wave identity mismatch')
        folder=Path(identity);wave=folder/group;startup=read(folder/'startup.json');runtime=read(folder/'native-runtime.json')
        model_path=Path(runtime['model']);artifact=model_path.parents[1]
        audit_native_runtime(runtime,plan,hardware,root=artifact)
        actual_case=startup['case'];wanted=copy.deepcopy(case);environment=wanted['environment']
        run_dir=artifact/'ebplus-validation/raw'/root.name/identity
        pythonpath=actual_case['environment'].get('PYTHONPATH','').split(':')
        if len(pythonpath)!=2 or pythonpath[0]!=str(artifact/'ebplus-validation/overlays'/overlay_name) or not pythonpath[1].startswith(str(artifact/'ebplus-validation/code-task3-v')):
            raise ValueError('runtime source path mismatch')
        environment.update(PYTHONPATH=':'.join(pythonpath),INFERX_VLLM_RUNTIME=str(run_dir/'native-runtime.json'))
        if case['id']!='native':
            environment.update(VLLM_PD_CALIBRATION_FILE=str(artifact/'ebplus-validation/raw/hardware-v4/pd-hardware-parameters.json'),
                INFERX_EBPLUS_EVIDENCE=str(run_dir/'policy-evidence.json'),VLLM_COLLECT_SCHEDULE_STATS='0')
        if actual_case!=wanted or startup['args']!=engine_command(wanted,artifact,run_dir):
            raise ValueError('actual startup configuration/arguments mismatch')
        if read(folder/'quality-result.json')['status']!='passed':raise ValueError('code quality gate failed')
        qualify_tool_rows(rows(folder/'warmup.jsonl'),templates,['tool-24576']*4)
        qualify_tool_rows(rows(folder/'policy-prime.jsonl'),templates,['tool-24576']*100)
        specs=read(wave/'specifications.json')
        if specs!=comparison_specs(templates,group,identity,kind=kind):raise ValueError('frozen workload specification mismatch')
        records=rows(wave/'requests.jsonl');result=read(wave/'result.json')
        metrics=audit_wave(specs,records,result['span_s'],kind=kind,backend='vllm031')
        policy=None;before=None;sensitivity=None
        if case['id']!='native':
            evidence=read(folder/'policy-evidence.json');policy=audit_policy(evidence,hardware,mb,delta=float(case['environment'].get('VLLM_PD_MODE_SWITCH_DELTA','.0001')),require_exclusive_execution=True)
            if case['id'].startswith('auto-'):
                before_evidence=read(folder/'policy-before.json');before=audit_policy(before_evidence,hardware,mb,delta=float(case['environment'].get('VLLM_PD_MODE_SWITCH_DELTA','.0001')))
                x=copy.deepcopy(evidence['last_evaluation_inputs']);x['delta']=0.;sensitivity=crossover_decision(x)
            policy=dict(decision=policy,evidence=evidence,before=before,zero_hysteresis_counterfactual=sensitivity,
                scope='Counters include unscored conditioning; they are not exact measured-wave physical work.')
        data(folder/'server.log');gpu=sampled_gpu_memory(data(folder/'gpu.csv'))
        groups.append(dict(case=case['id'],group=group,repeat=repeat,identity=identity,
            configuration_sha256=hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest(),
            runtime=runtime,gpu_memory=gpu,policy=policy,**metrics))
    return dict(run=root.name,kind=kind,phase=phase,groups=groups,source_sha256=sources,
        scope='Fresh-engine waves with matched code+4warm+100actual prime; strict full inputs/tools. API cache is not physical GPU work.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--frozen',required=True,type=Path)
    p.add_argument('--mixed',required=True,type=Path);p.add_argument('--solo',type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    templates,_=load_frozen_workloads(a.frozen)
    contract=verify_prompt_count_contract(a.frozen.parent/'models/qwen38-int4-redhat',backend='vllm031')
    mixed=export_campaign(a.mixed,templates);result=dict(mixed=mixed,prompt_count_contract=contract)
    if a.solo:
        solo=export_campaign(a.solo,templates)
        configs={}
        for run in [mixed,solo]:
            for row in run['groups']:
                prior=configs.setdefault(row['case'],row['configuration_sha256'])
                if prior!=row['configuration_sha256']:raise ValueError('mixed/solo configuration mismatch')
        result.update(solo=solo,comparison=compare_lanes(mixed['groups'],solo['groups']))
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')


if __name__=='__main__':main()
