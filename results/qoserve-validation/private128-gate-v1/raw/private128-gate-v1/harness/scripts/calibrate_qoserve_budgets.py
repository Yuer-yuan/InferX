"""Matched independent native solos and actual HTTP-to-engine QoS tag joins."""
import copy
import json
from pathlib import Path


def join_solo_records(clients,observer):
    result=[];seen=set()
    for client in clients:
        trace=client['id']
        if trace in seen:raise ValueError('duplicate solo client trace')
        seen.add(trace)
        matches=[r for r in observer['requests'].values() if r.get('trace_id')==trace]
        if len(matches)!=1:raise ValueError('solo client and actual engine records do not join uniquely')
        engine=matches[0]
        if (client.get('functional_ok') is not True or client['summary'].get('ok') is not True
                or engine.get('finish_status')!='FINISHED_STOPPED'):
            raise ValueError('solo client or engine strict completion failed')
        result.append(dict(request_id=trace,strict_ok=True,ttft_s=client['arrival_ttft_s'],
            latency_s=client['latency_s'],model_token_times_s=engine['model_token_times_s']))
    return result


def run_budget_calibration(plan,root,out,progress,overlay):
    import asyncio
    from inferx.concurrency import run_load
    from inferx.ebplus_calibration import qualify_tool_rows
    from inferx.pd_baselines import load_frozen_workloads
    from inferx.qoserve_profile import freeze_common_budgets
    from inferx.qoserve_validation import parse_contract
    from inferx.sota_results import verify_prompt_count_contract
    from scripts.run_pd_baselines import OwnedEngine,wait_idle,ENDPOINT
    from scripts import run_concurrency_campaign as previous
    templates,_=load_frozen_workloads(root/'concurrency-roofline')
    verify_prompt_count_contract(root/plan['base_case']['model'],backend='vllm031')
    code=Path(__file__).resolve().parents[1]
    calibration=dict(identity=plan['identity'],phase='independent-native-solo',strategy='plain-native',
                     short=[],cold=[],resume=[])
    report=dict(status='running',identity=plan['identity'],cases=[],calibration=calibration)
    def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    def flush():write(out/'budget-calibration.json',report)
    groups=[('short',0,24)]+[(kind,repeat,1) for kind in ('cold','resume') for repeat in range(3)]
    def execute(server,name,prompt,count,*,serial=False,tagged=False):
        template=templates[prompt];specs=[]
        for i in range(count):
            trace=f'{server.case["id"]}/{name}/{i}';payload=copy.deepcopy(template['payload'])
            if tagged:
                extra=payload.setdefault('vllm_xargs',{})
                assert not any(k.startswith('qoserve_') for k in extra)
                # Plain native policy is off; these loose observer-only tags
                # establish transport/trace joins before common budgets exist.
                extra.update(qoserve_tier=0 if prompt=='tool-24576' else 1,
                    qoserve_deadline_s=3600.,qoserve_tbt_s=1.,qoserve_trace_id=trace)
            spec=dict(id=trace,prompt_id=prompt,payload=payload,expected_tool=template['expected_tool'])
            if serial:spec['producer']=0
            specs.append(spec)
        wait_idle()
        with (server.directory/(name+'.jsonl')).open('x') as raw:
            rows=asyncio.run(run_load(ENDPOINT,specs,raw,'closed',4))
        checked=copy.deepcopy(rows)
        if tagged:
            for row in checked:
                extra=row['request']['vllm_xargs']
                tags={k:extra.pop(k) for k in ('qoserve_tier','qoserve_deadline_s','qoserve_tbt_s','qoserve_trace_id')}
                assert tags['qoserve_trace_id']==row['id'];parse_contract(tags)
                if 'vllm_xargs' not in template['payload']:
                    assert not extra;row['request'].pop('vllm_xargs')
        proof=qualify_tool_rows(checked,templates,[prompt]*count)
        write(server.directory/(name+'-strict.json'),proof)
        return rows
    flush()
    try:
        for kind,repeat,count in groups:
            previous.gpu_free();name=f'native-budget-{kind}-r{repeat}';directory=out/name
            case=copy.deepcopy(plan['base_case'])
            case.update(id=name,async_scheduling=False,shutdown_timeout=30,num_gpu_blocks_override=98,
                scheduler_reserve_full_isl=True,long_prefill_token_threshold=0,
                scheduler_class='inferx_qoserve_scheduler.ObservedNativeScheduler')
            case['environment'].update(PYTHONPATH=str(overlay)+':'+str(code),
                INFERX_VLLM_RUNTIME=str(directory/'native-runtime.json'),
                INFERX_QOSERVE_EVIDENCE=str(directory/'model-observer.json'))
            entry=dict(case=name,state='starting',group=kind,repeat=repeat);progress(entry)
            item=dict(group=kind,repeat=repeat,case=case);report['cases'].append(item);flush()
            with OwnedEngine(case,root,directory) as server:
                runtime=json.loads((directory/'native-runtime.json').read_text())
                expected=dict(concurrency=4,max_active=4,chunk=4096,long_prefill_token_threshold=0,
                    scheduler_reserve_full_isl=True,async_scheduling=False,enforce_eager=True,
                    speculative=False,kv_dtype='fp8',num_gpu_blocks=98,
                    scheduler_class='ObservedNativeScheduler',qoserve_policy_enabled=False,profiling_mode='off')
                if any(runtime.get(k)!=v for k,v in expected.items()):raise ValueError('native budget runtime drift')
                item['runtime']=runtime;flush()
                previous.quality_gate(server,templates)
                if json.loads((directory/'quality-result.json').read_text())['status']!='passed':
                    raise ValueError('budget calibration code quality gate failed')
                execute(server,'warmup','tool-24576',4)
                execute(server,'policy-prime','tool-24576',100)
                prompt='tool-24576' if kind=='short' else kind+'-131072'
                rows=execute(server,'measured',prompt,count,serial=True,tagged=True)
            observer=json.loads((directory/'model-observer.json').read_text())
            joined=join_solo_records(rows,observer);calibration[kind].extend(joined)
            item.update(status='collected',joined=joined);entry['state']='collected';progress(None);flush()
        budgets=freeze_common_budgets(calibration,plan['identity'])
        write(out/'common-budgets.json',budgets);report.update(status='complete',budgets=budgets)
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));raise
    finally:flush()
    return report
