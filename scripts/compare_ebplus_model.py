"""Fresh-engine full-input EB+ waves; invoked only by owned restoration controller."""
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import time

from inferx.concurrency import run_load
from inferx.ebplus_calibration import qualify_tool_rows
from inferx.ebplus_comparison import comparison_cases,comparison_specs,audit_policy,audit_native_runtime
from inferx.pd_baselines import load_frozen_workloads
from inferx.sota_results import audit_wave,verify_prompt_count_contract
from scripts.qualify_ebplus_model import verify_inputs,configuration_snapshot
from scripts.run_pd_baselines import OwnedEngine,wait_idle,get,ENDPOINT
from scripts import run_concurrency_campaign as previous


def run_comparison(plan,root,out,progress,hardware_run,mb_run,overlay,definition):
    hardware,mb=verify_inputs(plan,hardware_run,mb_run,overlay)
    phase=definition['phase'];kind=definition['kind'];repeats=definition['repeats'];selected=definition['case_ids']
    if (phase not in {'diagnostic','formal','solo'} or
        kind!=('isolated-serial-C4' if phase=='solo' else 'mixed') or
        repeats!=(3 if phase=='formal' else 1) or len(set(selected))!=len(selected) or
        not selected or not set(selected)<={'native','auto-default','forced-eb'}):
        raise ValueError('unfrozen comparison definition')
    templates,_=load_frozen_workloads(root/'concurrency-roofline')
    verify_prompt_count_contract(root/plan['base_case']['model'],backend='vllm031')
    code=Path(__file__).resolve().parents[1]
    cases=[c for c in comparison_cases(plan) if c['id'] in selected]
    if hardware['native_runtime']['num_gpu_blocks']!=98:raise ValueError('qualified pool changed')
    for c in cases:
        if c['id']!='native':c['environment'].update(mb['fit']['env_vars'])
    def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
    write(out/'comparison-definition.json',definition);write(out/'config.json',cases)
    for label,source in [('hardware',hardware_run),('mb',mb_run),('overlay',overlay)]:
        target=out/label;target.mkdir()
        if label=='overlay':files=['manifest.json',*json.loads((overlay/'manifest.json').read_text())['files']]
        else:files=['hardware-result.json','pd-hardware-parameters.json'] if label=='hardware' else ['mb-grid-result.json']
        for name in files:(target/name).write_bytes((source/name).read_bytes())
    groups=['short-control','cold-low','resume-high'] if phase=='solo' else ['cold-low','resume-high']
    def tools(directory,name,count):
        template=templates['tool-24576']
        specs=[dict(id=f'{name}-{i}',prompt_id='tool-24576',payload=template['payload'],
                    expected_tool=template['expected_tool']) for i in range(count)]
        wait_idle()
        with (directory/(name+'.jsonl')).open('x') as raw:
            records=asyncio.run(run_load(ENDPOINT,specs,raw,'closed',4))
        return qualify_tool_rows(records,templates,['tool-24576']*count)
    for repeat in range(repeats):
        # Rotate lane ordering across repeats; each wave still uses a fresh engine.
        ordered=cases[repeat%len(cases):]+cases[:repeat%len(cases)]
        for static in ordered:
            for group in groups:
                previous.gpu_free()
                identity=f'{static["id"]}-r{repeat}-{group}';directory=out/identity
                case=copy.deepcopy(static)
                case['environment'].update(PYTHONPATH=str(overlay)+':'+str(code),
                    INFERX_VLLM_RUNTIME=str(directory/'native-runtime.json'))
                if case['id']!='native':
                    case['environment'].update(VLLM_PD_CALIBRATION_FILE=str(hardware_run/'pd-hardware-parameters.json'),
                        INFERX_EBPLUS_EVIDENCE=str(directory/'policy-evidence.json'),
                        VLLM_COLLECT_SCHEDULE_STATS='0')
                entry=dict(case=static['id'],group=group,repeat=repeat,identity=identity,state='starting')
                progress(entry)
                with OwnedEngine(case,root,directory) as server:
                    runtime=json.loads((directory/'native-runtime.json').read_text())
                    audit_native_runtime(runtime,plan,hardware,root=root)
                    write(directory/'configuration-evidence.json',configuration_snapshot(runtime,get))
                    previous.quality_gate(server,templates)
                    if json.loads((directory/'quality-result.json').read_text())['status']!='passed':
                        raise ValueError('code gate failed')
                    condition=dict(order=['code','warmup4','policy-prime100','measured-wave'],
                        warmup=tools(directory,'warmup',4),priming=tools(directory,'policy-prime',100))
                    write(directory/'conditioning.json',condition)
                    wait_idle()
                    policy=directory/'policy-evidence.json'
                    if case['id']=='auto-default':
                        state=json.loads(policy.read_text())
                        audit_policy(state,hardware,mb,delta=.0001)
                        (directory/'policy-before.json').write_bytes(policy.read_bytes())
                    specs=comparison_specs(templates,group,identity,kind=kind)
                    wave=directory/group;wave.mkdir();write(wave/'specifications.json',specs)
                    metrics_before=get('/metrics');started=time.time()*1000;start=time.monotonic()
                    with (wave/'requests.jsonl').open('x') as raw:
                        records=asyncio.run(run_load(ENDPOINT,specs,raw,
                            'open' if kind=='mixed' else 'closed',4))
                    if kind=='mixed':time.sleep(max(0,48-(time.monotonic()-start)))
                    wait_idle();span=time.monotonic()-start
                    result=audit_wave(specs,records,span,kind=kind,backend='vllm031')
                    result.update(started_unix_ms=started,ended_unix_ms=time.time()*1000,
                                  metrics_before=metrics_before,metrics_after=get('/metrics'))
                    write(wave/'result.json',result)
                if case['id']!='native':
                    state=json.loads(policy.read_text())
                    decision=audit_policy(state,hardware,mb,delta=.0001)
                    write(directory/'policy-audit.json',decision)
                entry.update(state='measured',metrics=result);progress(None)
                print(json.dumps(dict(event='ebplus_wave_done',**entry)),flush=True)
    return cases
