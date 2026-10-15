"""Audit exact raw SOTA campaigns and export repeat-level, reproducible metrics."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from inferx.pd_baselines import load_frozen_workloads, validate_sglang_runtime
from inferx.sota_results import audit_wave, compare_lanes
from inferx.concurrency import reconstruct_tool_calls
from scripts.analyze_pd_baselines import sampled_gpu_memory


def export_campaign(root,templates):
    root=Path(root);sources={}
    def data(relative):
        path=root/relative;raw=path.read_bytes()
        sources[str(relative)]=hashlib.sha256(raw).hexdigest()
        return raw
    def read(relative):return json.loads(data(relative))
    def rows(relative):return [json.loads(line) for line in data(relative).splitlines() if line.strip()]
    campaign=read('campaign.json');configuration=read('config.json')
    if campaign['status']!='restored' or not campaign['restored']:
        raise ValueError('campaign not terminal/restored')
    read('restored-service.json')
    for name,expected in campaign['harness_sha256'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('invalid archived harness path')
        if hashlib.sha256(data(Path('harness')/name)).hexdigest()!=expected:
            raise ValueError('archived harness hash mismatch')
    cases={c['id']:c for c in configuration}
    if len(cases)!=len(configuration):raise ValueError('duplicate configuration identity')
    groups=[];identities=set()
    for state in campaign['cases']:
        case=cases[state['case']];group=state['group'];repeat=state['repeat']
        if type(repeat) is not int or repeat<0 or any(Path(n).name!=n or n in {'.','..'} for n in [case['id'],group]):
            raise ValueError('invalid wave identity')
        identity=f'{case["id"]}-r{repeat}-{group}'
        if identity in identities:raise ValueError('duplicate repeat identity')
        identities.add(identity)
        if state['state']!='measured':raise ValueError('unqualified/failed wave has no efficacy result')
        folder=Path(identity);wave=folder/group;startup=read(folder/'startup.json')
        expected=copy.deepcopy(case)
        # Immutable runtime IR output directory varies only with the wave name.
        if 'INFERX_MORPHX_IR' in expected['environment']:
            expected['environment']['INFERX_MORPHX_IR']=startup['case']['environment']['INFERX_MORPHX_IR']
        if expected!=startup['case']:raise ValueError('startup configuration mismatch')
        validate_sglang_runtime(case,startup['server_info']);runtime=startup['server_info']['internal_states'][0]
        graph=runtime.get('memory_usage',{}).get('graph')
        if not isinstance(graph,dict) or not graph or any(value!=0 for value in graph.values()):
            raise ValueError('actual eager graph allocation must be zero')
        if runtime.get('speculative_algorithm') is not None:
            raise ValueError('actual MTP0 configuration required')
        if (runtime.get('mamba_ssm_dtype')!='bfloat16' or runtime.get('max_mamba_cache_size')!=16 or
            runtime.get('kv_cache_dtype')!='fp8_e4m3' or not runtime.get('disable_overlap_schedule')):
            raise ValueError('unmatched hybrid-state/precision/scheduling configuration')
        if read(folder/'quality-result.json')['status']!='passed':raise ValueError('code quality gate failed')
        specs=read(wave/'specifications.json');records=rows(wave/'requests.jsonl')
        for spec in specs:
            template=templates[spec['prompt_id']]
            if spec['payload']!=template['payload'] or spec['expected_tool']!=template['expected_tool']:
                raise ValueError('specification differs from original frozen template')
        warm=rows(wave/'warmup.jsonl')
        if len(warm)!=4:raise ValueError('four independently prepared strict warmups required')
        for row in warm:
            template=templates['tool-24576'];calls=reconstruct_tool_calls(row['events'])
            if (row['request']!=template['payload'] or not row['summary']['ok'] or row['summary'].get('truncated') or
                len(calls)!=1 or calls[0]['name']!=template['expected_tool']['name'] or
                calls[0]['arguments']!=template['expected_tool']['arguments']):
                raise ValueError('strict warmup/frozen input mismatch')
        result=read(wave/'result.json');metrics=audit_wave(specs,records,result['summary']['measured_span_s'],kind=campaign['kind'])
        log=data(folder/'server.log');coverage=None
        if case['morphx']['enabled']:
            coverage=read(wave/'morphx-coverage.json')
            snapshot=data(wave/'morphx-coverage.log')
            if coverage['status']!='observed' or coverage['snapshot_sha256']!=hashlib.sha256(snapshot).hexdigest():
                raise ValueError('model-process transformation coverage unobserved or modified')
        gpu=sampled_gpu_memory(data(folder/'gpu.csv'))
        admissions=[]
        for line in log.decode(errors='replace').splitlines():
            if 'PD_ADMISSION ' not in line:continue
            entry=json.loads(line.split('PD_ADMISSION ',1)[1])
            if (result.get('started_unix_ms',float('inf'))/1000<=entry['unix']<=result.get('ended_unix_ms',0)/1000
                and entry['method']=='new' and entry['admitted']):
                admissions.append({k:entry[k] for k in ['rid','cached_tokens','remaining_input','chunk_before','result']})
        groups.append(dict(case=case['id'],group=group,repeat=repeat,identity=identity,
            configuration_sha256=hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest(),
            runtime=dict(kv_capacity=runtime['memory_usage']['token_capacity'],concurrency=runtime['effective_max_running_requests_per_dp'],
                         ssm_slots=16,ssm_dtype='bfloat16',kv_dtype='fp8_e4m3'),
            gpu_memory=gpu,morphx_coverage=coverage,native_new_admissions=admissions,**metrics))
    return dict(run=root.name,kind=campaign['kind'],groups=groups,source_sha256=sources,
                scope='Reconstructed strict tool output and complete input counts; native admissions are not a physical GPU work counter.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--frozen',required=True,type=Path)
    parser.add_argument('--mixed',required=True,type=Path);parser.add_argument('--solo',required=True,type=Path)
    parser.add_argument('--diagnostic',type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();templates,_=load_frozen_workloads(args.frozen)
    mixed=export_campaign(args.mixed,templates);solo=export_campaign(args.solo,templates)
    if mixed['kind']!='mixed' or solo['kind']!='isolated-serial-C4':raise ValueError('mixed and serial-C4 campaigns required')
    configs={}
    for run in [mixed,solo]:
        for group in run['groups']:
            previous=configs.setdefault(group['case'],group['configuration_sha256'])
            if previous!=group['configuration_sha256']:raise ValueError('solo/mixed configuration mismatch')
    output=dict(comparison=compare_lanes(mixed['groups'],solo['groups']),mixed=mixed,solo=solo,
                diagnostic=export_campaign(args.diagnostic,templates) if args.diagnostic else None,
                frozen_source_sha256={name:hashlib.sha256((args.frozen/name).read_bytes()).hexdigest()
                                      for name in ['workloads.json','grid.json','workload-manifest.json']})
    code=Path(__file__).resolve().parents[1]
    output['analysis_source_sha256']={name:hashlib.sha256((code/name).read_bytes()).hexdigest() for name in
        ['scripts/analyze_sota_validation.py','inferx/sota_results.py','inferx/concurrency.py',
         'inferx/tuning.py','inferx/pd_baselines.py','scripts/analyze_pd_baselines.py']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(output['comparison'],indent=2))


if __name__=='__main__':main()
