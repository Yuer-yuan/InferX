"""Full27B/native-hybrid qualification, separate from efficacy measurements."""
import asyncio
import copy
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError

from inferx.concurrency import run_load
from inferx.ebplus_calibration import (validate_hardware_result,fit_mb_cost,
                                      qualify_tool_rows,validate_policy_evidence)
from inferx.ebplus_validation import AUTHOR_SHA256,NATIVE_SHA256,CALIBRATION_SHA256
from inferx.pd_baselines import load_frozen_workloads
from inferx.sota_results import verify_prompt_count_contract
from scripts.run_pd_baselines import OwnedEngine,wait_idle,get,ENDPOINT
from scripts import run_concurrency_campaign as previous


def configuration_snapshot(runtime,fetch):
    """The initialized native scheduler is authoritative when dev APIs are absent."""
    try:
        return dict(native_runtime=runtime,server_info=fetch('/server_info?config_format=json'))
    except HTTPError as exc:
        if exc.code!=404:raise
        return dict(native_runtime=runtime,server_info_http_status=404,
                    runtime_origin='initialized native scheduler, before model requests')


def verify_inputs(plan,hardware_run,mb_run,overlay):
    hardware=json.loads((hardware_run/'hardware-result.json').read_text())
    validate_hardware_result(hardware,plan)
    if hardware['status']!='complete':raise ValueError('hardware calibration incomplete')
    params=json.loads((hardware_run/'pd-hardware-parameters.json').read_text())
    if any(params[k]!=v for k,v in hardware['parameters'].items()):raise ValueError('hardware parameter bytes changed')
    mb=json.loads((mb_run/'mb-grid-result.json').read_text())
    if mb['identity']!=plan['identity'] or mb['status']!='complete':raise ValueError('MB calibration identity/incomplete')
    if fit_mb_cost(mb['rows'],plan)!=mb['fit']:raise ValueError('MB fit differs from raw independent grid')
    manifest=json.loads((overlay/'manifest.json').read_text())
    if manifest.get('overlay_directory_name',overlay.name)!=overlay.name:
        raise ValueError('overlay directory differs from prepared source manifest')
    if (manifest['author_sha256']!=AUTHOR_SHA256 or manifest['native_sha256']!=NATIVE_SHA256
        or manifest['calibration_sha256']!=CALIBRATION_SHA256):raise ValueError('overlay source mismatch')
    for name,digest in manifest['files'].items():
        if Path(name).name!=name or hashlib.sha256((overlay/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('overlay file mismatch')
    return hardware,mb


def run_model_gates(plan,root,out,progress,hardware_run,mb_run,overlay):
    hardware,mb=verify_inputs(plan,hardware_run,mb_run,overlay)
    templates,_=load_frozen_workloads(root/'concurrency-roofline')
    contract=verify_prompt_count_contract(root/plan['base_case']['model'],backend='vllm031')
    code=Path(__file__).resolve().parents[1]
    report=dict(status='running',identity=plan['identity'],prompt_count_contract=contract,cases=[],
                scope='Full model/code/tools/actual policy qualification; no joint performance conclusion.')
    def flush():(out/'model-qualification.json').write_text(json.dumps(report,indent=2)+'\n')
    def requests(name,count):
        t=templates[name]
        return [dict(id=f'{name}-{i}',prompt_id=name,payload=t['payload'],expected_tool=t['expected_tool'])
                for i in range(count)]
    def execute(server,name,prompt_id,count):
        wait_idle()
        with (server.directory/(name+'.jsonl')).open('x') as raw:
            rows=asyncio.run(run_load(ENDPOINT,requests(prompt_id,count),raw,'closed',4))
        return qualify_tool_rows(rows,templates,[prompt_id]*count)
    try:
        for mode in ['native','eb','auto']:
            previous.gpu_free()
            case=copy.deepcopy(plan['base_case']);case.update(id='qualify-'+mode,async_scheduling=False,shutdown_timeout=30,
                num_gpu_blocks_override=hardware['native_runtime']['num_gpu_blocks'],
                scheduler_class='inferx_ebplus_scheduler.'+('ObservedNativeScheduler' if mode=='native' else 'EBPlusScheduler'))
            directory=out/case['id']
            case['environment'].update(PYTHONPATH=str(overlay)+':'+str(code),
                INFERX_VLLM_RUNTIME=str(directory/'native-runtime.json'))
            if mode!='native':
                case['environment'].update(mb['fit']['env_vars'])
                case['environment'].update(VLLM_PD_SCHEDULER_MODE=mode,VLLM_PD_K_MODE='ifr',
                    VLLM_PD_CALIBRATION_FILE=str(hardware_run/'pd-hardware-parameters.json'),
                    VLLM_COLLECT_SCHEDULE_STATS='1',VLLM_SCHEDULE_STATS_FILE=str(directory/'schedule-stats.json'),
                    INFERX_EBPLUS_EVIDENCE=str(directory/'policy-evidence.json'))
            entry=dict(case=case['id'],state='starting');progress(entry)
            item=dict(mode=mode,configuration=case);report['cases'].append(item);flush()
            with OwnedEngine(case,root,directory) as server:
                runtime=json.loads((directory/'native-runtime.json').read_text())
                info=configuration_snapshot(runtime,get)
                (directory/'configuration-evidence.json').write_text(json.dumps(info,indent=2)+'\n')
                expected=dict(context_window=131328,concurrency=4,max_active=4,chunk=case['chunk'],
                    long_prefill_token_threshold=case['long_prefill_token_threshold'],async_scheduling=False,
                    enforce_eager=True,cudagraph_mode='NONE',speculative=False,quantization='compressed-tensors',
                    kv_dtype=case['kv_dtype'],mamba_cache_mode=case['mamba_cache_mode'],
                    num_gpu_blocks=case['num_gpu_blocks_override'],block_size=hardware['native_runtime']['block_size'])
                if any(runtime.get(k)!=v for k,v in expected.items()):
                    raise ValueError('actual engine configuration differs from qualification')
                item['native_runtime']=runtime
                previous.quality_gate(server,templates)
                if json.loads((directory/'quality-result.json').read_text())['status']!='passed':
                    raise RuntimeError('full code gate failed')
                item['warmup']=execute(server,'warmup','tool-24576',4);flush()
                if mode=='auto':
                    item['independent_policy_priming']=execute(server,'policy-prime','tool-24576',100)
                    flush()
                item['cold']=execute(server,'full-cold','cold-131072',1);flush()
                item['resume']=execute(server,'full-resume','resume-131072',1);flush()
            if mode!='native':
                item['policy']=json.loads((directory/'policy-evidence.json').read_text())
                validate_policy_evidence(item['policy'])
            entry['state']='measured';item['status']='passed';progress(None);flush()
        report['status']='passed'
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));raise
    finally:flush()
    return report
