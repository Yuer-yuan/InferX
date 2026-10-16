"""Independent real27B completed-engine-step profiling on the owned GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import time
import uuid


def validate_profile_plan(plan):
    from inferx.ebplus_validation import NATIVE_SHA256
    identity=plan['identity'];case=plan['base_case'];p=plan['profile']
    if identity.get('native_scheduler_sha256')!=NATIVE_SHA256 or any(
            case.get(k)!=v for k,v in identity.items() if k!='native_scheduler_sha256'):
        raise ValueError('profile identity differs from runtime')
    if (identity['concurrency'],identity['draft'],identity['cuda_graph'])!=(4,0,False):
        raise ValueError('profile requires C4 MTP0 eager')
    expected=dict(num_gpu_blocks=98,prefill_targets=[128,512,2048,2552],decoder_counts=[0,1,3],
        holdout_decoder_counts=[0,2],candidates=[128,256,384,512,768,1024,1280,1536,1792,2048,2552],
        long_prompt_len=131072,decoder_prompt_len=24557,decoder_max_tokens=1024,warmup_steps=5,
        initial_pps_warmup=5,initial_pps_samples=20,stop_on_preemption_with_decoders=True)
    if p!=expected or case['chunk']!=4096 or case['long_prefill_token_threshold']!=0:
        raise ValueError('profile geometry differs from predeclared protocol')
    if case.get('scheduler_reserve_full_isl') is not False:
        raise ValueError('profile requires actual native checkFalse')


def rollout_specs(plan,engine_id):
    validate_profile_plan(plan);p=plan['profile'];result=[]
    def add(kind,target,decoders,length,warmup):
        index=len(result);rid=f'{kind}-{index:02d}-p{target}-d{decoders}'
        result.append(dict(rollout_id=rid,kind=kind,phase='holdout' if kind=='holdout' else 'train',
            prefill_target=target,decoders=decoders,prompt_len=length,warmup_steps=warmup,
            long_salt=engine_id+'/'+rid+'/long',decoder_salt=engine_id+'/'+rid+'/decoder',token_seed=42+index))
    for i in range(p['initial_pps_warmup']+p['initial_pps_samples']):
        add('initial-pps',512,0,512,int(i<p['initial_pps_warmup']))
    for target in p['prefill_targets']:
        for decoders in p['decoder_counts']:add('train',target,decoders,p['long_prompt_len'],p['warmup_steps'])
    for decoders in p['holdout_decoder_counts']:add('holdout',128,decoders,p['long_prompt_len'],p['warmup_steps'])
    return result


def repair_rollout_specs(plan,engine_id,definition):
    """One declared supplementary matrix; old holdout never enters training."""
    validate_profile_plan(plan)
    expected_candidates=[32,64,96,128,256,384,512,768,1024,1280,1536,1792,2048,2552]
    training=[dict(decoders=d,context=c) for d,c in
        ((0,24557),(1,512),(1,24557),(2,24557),(3,24557),(1,65536),(1,131072))]
    holdout=[dict(decoders=d,context=24557) for d in (0,2)]
    cycles=[dict(protocol='original-cycle-v1',candidates=plan['profile']['candidates']),
            dict(protocol='remainder-cycle-v1',candidates=expected_candidates)]
    if (definition.get('schema_version')!=1 or definition.get('identity')!=plan['identity']
            or definition.get('calibration_candidates')!=expected_candidates
            or definition.get('train_decoder_geometries')!=training
            or definition.get('holdout_decoder_geometries')!=holdout
            or definition.get('holdout_candidate_cycles')!=cycles
            or definition.get('diagnostic_homogeneous_prefill_target')!=128
            or definition.get('separate_protocol_guard_required') is not True
            or definition.get('retain_initial_training') is not True
            or definition.get('replace_initial_holdout') is not True or definition.get('warmup_steps')!=5):
        raise ValueError('repair definition differs from the bounded independent protocol')
    for key in ('basis_fit_sha256','basis_archive_sha256'):
        value=definition.get(key)
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('repair needs exact initial calibration hashes')
    path=Path(definition['basis_fit_path'])
    if path.is_absolute() or '..' in path.parts:raise ValueError('repair basis must remain on HDD')
    result=[s for s in rollout_specs(plan,engine_id) if s['kind']=='initial-pps']
    def add(kind,geometry,candidates,protocol):
        index=len(result);d=geometry['decoders'];context=geometry['context']
        phase='train' if kind=='train' else 'holdout'
        rid=f'repair-{kind}-{index:02d}-d{d}-ctx{context}'
        result.append(dict(rollout_id=rid,kind=kind,phase=phase,budget_mode='total-cycle',
            holdout_protocol=protocol,candidates=candidates,prefill_target=128,decoders=d,prompt_len=131072,warmup_steps=5,
            decoder_prompt_len=context,decoder_max_tokens=min(1024,plan['identity']['context_window']-context),
            long_salt=engine_id+'/'+rid+'/long',decoder_salt=engine_id+'/'+rid+'/decoder',token_seed=142+index))
    for geometry in training:add('train',geometry,expected_candidates,'training-cycle-v1')
    add('diagnostic',dict(decoders=0,context=24557),[128],'diagnostic-homogeneous-v1')
    for cycle in cycles:
        for geometry in holdout:add('holdout',geometry,cycle['candidates'],cycle['protocol'])
    return result


def verify_repair_basis(definition,root):
    root=Path(root).resolve();basis=(root/definition['basis_fit_path']).resolve()
    if not basis.is_relative_to(root):raise ValueError('repair basis escapes HDD root')
    if hashlib.sha256(basis.read_bytes()).hexdigest()!=definition['basis_fit_sha256']:
        raise ValueError('repair basis fit hash mismatch')
    result=json.loads(basis.read_text())
    inventory=json.loads((root/'qoserve-validation/exports/independent-profile-v5-inventory.json').read_text())
    if (result.get('qualified') is not False or result.get('identity')!=definition['identity']
            or result['input_evidence']['archive_sha256']!=definition['basis_archive_sha256']
            or inventory['archive_sha256']!=definition['basis_archive_sha256']):
        raise ValueError('repair basis is not the declared initial calibration')


def calibration_engine_args(plan,root):
    from vllm.engine.arg_utils import EngineArgs
    validate_profile_plan(plan);case=plan['base_case']
    return EngineArgs(model=str(root/case['model']),dtype='bfloat16',max_model_len=case['context_window'],
        max_num_batched_tokens=4096,max_num_seqs=4,long_prefill_token_threshold=0,
        gpu_memory_utilization=case['gpu_memory_utilization'],kv_cache_dtype=case['kv_dtype'],
        mamba_cache_mode=case['mamba_cache_mode'],enable_chunked_prefill=True,enable_prefix_caching=True,
        enforce_eager=True,async_scheduling=False,distributed_executor_backend='uni',language_model_only=True,
        scheduler_cls='inferx_qoserve_scheduler.ProfilingNativeScheduler',scheduler_reserve_full_isl=False,
        num_gpu_blocks_override=plan['profile']['num_gpu_blocks'])


def calibration_request(spec,role,index,plan):
    from vllm import SamplingParams
    from vllm.v1.engine import EngineCoreRequest
    if role not in ('long','decoder'):raise ValueError('unknown profile request role')
    is_long=role=='long';p=plan['profile']
    rng=random.Random(spec['token_seed']+int(is_long)*10000)
    length=spec['prompt_len'] if is_long else spec.get('decoder_prompt_len',p['decoder_prompt_len'])
    return EngineCoreRequest(request_id=spec['rollout_id']+'/'+role+'/'+str(index),
        prompt_token_ids=[rng.randrange(100,10000) for _ in range(length)],mm_features=None,
        sampling_params=SamplingParams(max_tokens=1 if is_long else spec.get('decoder_max_tokens',p['decoder_max_tokens']),
            ignore_eos=True,temperature=0.),pooling_params=None,arrival_time=time.time(),lora_request=None,
        cache_salt=spec['long_salt'] if is_long else spec['decoder_salt'],data_parallel_rank=None)


def read_gpu_snapshot(nvml,handle):
    """Read management counters outside the completed-step timing interval."""
    gpu_uuid=nvml.nvmlDeviceGetUUID(handle)
    if isinstance(gpu_uuid,bytes):gpu_uuid=gpu_uuid.decode()
    return dict(time_unix_s=time.time(),monotonic_s=time.perf_counter(),gpu_uuid=gpu_uuid,
        temperature_c=nvml.nvmlDeviceGetTemperature(handle,nvml.NVML_TEMPERATURE_GPU),
        sm_clock_mhz=nvml.nvmlDeviceGetClockInfo(handle,nvml.NVML_CLOCK_SM),
        memory_clock_mhz=nvml.nvmlDeviceGetClockInfo(handle,nvml.NVML_CLOCK_MEM),
        power_w=nvml.nvmlDeviceGetPowerUsage(handle)/1000.,pstate=nvml.nvmlDeviceGetPerformanceState(handle))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True,type=Path)
    parser.add_argument('--repair-definition',type=Path)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    plan=json.loads(args.plan.read_text());validate_profile_plan(plan)
    assert not args.output.exists()
    root=Path(os.environ['INFERX_ROOT']);assert os.environ['CUDA_VISIBLE_DEVICES']==plan['identity']['gpu_uuid']
    result=dict(status='starting',identity=plan['identity'],engine_id=uuid.uuid4().hex,rollouts=[],
        started_unix=time.time(),plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        measurement_type='completed-engine-wall',raw_rows='steps.jsonl')
    if args.repair_definition:
        definition=json.loads(args.repair_definition.read_text());verify_repair_basis(definition,root)
        specs=repair_rollout_specs(plan,result['engine_id'],definition)
        result.update(matrix_kind='bounded-repair-v1',repair_definition_sha256=
                      hashlib.sha256(args.repair_definition.read_bytes()).hexdigest())
    else:
        specs=rollout_specs(plan,result['engine_id']);result['matrix_kind']='initial-v1'
    def flush():
        tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(args.output)
    flush();core=None;nvml=None;nvml_initialized=False
    try:
        import pynvml as nvml
        nvml.nvmlInit();nvml_initialized=True
        nvml_handle=nvml.nvmlDeviceGetHandleByUUID(plan['identity']['gpu_uuid'])
        import torch
        from vllm.v1.engine.core import EngineCore
        from vllm.v1.executor.uniproc_executor import UniProcExecutor
        from vllm.utils.torch_utils import set_default_torch_num_threads
        from vllm.v1.request import RequestStatus
        cfg=calibration_engine_args(plan,root).create_engine_config()
        assert cfg.speculative_config is None and cfg.max_concurrent_batches==1
        assert cfg.model_config.quantization=='compressed-tensors'
        with set_default_torch_num_threads(1):
            core=EngineCore(vllm_config=cfg,executor_class=UniProcExecutor,log_stats=False)
        scheduler=core.scheduler
        observed=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid',
                                         '--format=csv,noheader,nounits'],text=True)
        processes={(int(line.split(',')[0]),line.split(',')[1].strip())
                   for line in observed.splitlines() if line.strip()}
        assert processes=={(os.getpid(),plan['identity']['gpu_uuid'])},processes
        result.update(status='running',gpu_process=dict(pid=os.getpid(),uuid=plan['identity']['gpu_uuid']),
            gpu_telemetry_initial=read_gpu_snapshot(nvml,nvml_handle),
            host_process_cpu_time_boundary='around EngineCore.step; includes read-only output observers',
            device_name=torch.cuda.get_device_name(0),torch_version=torch.__version__,
            native_runtime=dict(scheduler_class=type(scheduler).__name__,max_model_len=scheduler.max_model_len,
                max_num_running_reqs=scheduler.max_num_running_reqs,block_size=scheduler.block_size,
                cache_block_size=cfg.cache_config.block_size,num_gpu_blocks=scheduler.kv_cache_config.num_blocks,
                full_isl=scheduler.scheduler_reserve_full_isl,cap=scheduler.scheduler_config.long_prefill_token_threshold,
                kv_dtype=cfg.cache_config.cache_dtype,quantization=cfg.model_config.quantization,
                mamba_cache_mode=cfg.cache_config.mamba_cache_mode,torch_num_threads=torch.get_num_threads(),
                kv_groups=[dict(type=type(g.kv_cache_spec).__name__,spec=str(g.kv_cache_spec),layers=g.layer_names)
                           for g in scheduler.kv_cache_config.kv_cache_groups]))
        assert scheduler.kv_cache_config.num_blocks==98 and scheduler.max_num_running_reqs==4
        flush()
        def add(request):core.add_request(*core.preprocess_add_request(request));return request.request_id
        pps=[]
        with args.output.with_name('steps.jsonl').open('x') as raw:
            for spec in specs:
                assert not scheduler.requests
                seeds=[add(calibration_request(spec,'decoder',i,plan)) for i in range(spec['decoders'])]
                for prep in range(1000):
                    if all(rid in scheduler.requests and scheduler.requests[rid].output_token_ids for rid in seeds):break
                    core.step()
                else:raise RuntimeError('decoder preparation did not complete')
                long_id=add(calibration_request(spec,'long',0,plan))
                entry=dict(spec=spec,status='running',rows=0,started_unix=time.time(),
                    decoder_preparation_steps=prep,gpu_telemetry=[],
                    free_blocks_before=scheduler.kv_cache_manager.block_pool.get_num_free_blocks())
                result['rollouts'].append(entry);flush()
                scheduler.begin_profile(dict(phase=spec['phase'],identity=plan['identity'],engine_id=result['engine_id'],
                    rollout_id=spec['rollout_id'],cache_salt=spec['long_salt'],long_request_id=long_id,
                    prefill_target=spec['prefill_target'],candidates=spec.get('candidates',plan['profile']['candidates']),
                    budget_mode=spec.get('budget_mode','prefill-target' if spec['phase']=='train' else 'total-cycle'),
                    calibration_kind=spec['kind'],holdout_protocol=spec.get('holdout_protocol','original-cycle-v1'),
                    warmup_steps=spec['warmup_steps']))
                row_cursor=0
                for step in range(20000):
                    cpu_start=time.process_time()
                    core.step()
                    cpu_elapsed=time.process_time()-cpu_start
                    new=scheduler._profile_rows[row_cursor:]
                    for row in new:
                        row['host_process_cpu_time_s']=cpu_elapsed
                        raw.write(json.dumps(row,allow_nan=False)+'\n')
                    row_cursor+=len(new)
                    if step%25==0:
                        entry['gpu_telemetry'].append(dict(after_step=scheduler.current_step,
                            **read_gpu_snapshot(nvml,nvml_handle)))
                        raw.flush();entry['rows']=row_cursor;flush()
                    if long_id not in scheduler.requests:
                        entry['stop_reason']='long-completed';break
                    if new and new[-1]['preempted_request_ids'] and any(rid in scheduler.requests for rid in seeds):
                        entry.update(stop_reason='native-preemption-with-held-decoders',
                            preempted_request_ids=new[-1]['preempted_request_ids'],
                            capacity_snapshot=dict(free_blocks=scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
                                requests={rid:dict(computed_tokens=r.num_computed_tokens,prompt_tokens=r.num_prompt_tokens,
                                    output_tokens=len(r.output_token_ids),preemptions=r.num_preemptions,
                                    blocks=scheduler.kv_cache_manager.get_blocks(rid).get_block_ids())
                                    for rid,r in scheduler.requests.items()}));break
                else:raise RuntimeError('profile exceeded declared model step bound')
                records=scheduler.end_profile();assert row_cursor==len(records['rows'])
                entry.update(status='collected',rows=row_cursor,events=records['events'],finished_unix=time.time())
                if spec['kind']=='initial-pps' and not records['rows'][0]['warmup']:
                    row=records['rows'][0]
                    assert len(records['rows'])==1 and row['features']['batch_num_prefill_tokens']==512
                    pps.append(512/row['timing']['batch_execution_time'])
                scheduler.finish_requests(list(scheduler.requests),RequestStatus.FINISHED_ABORTED)
                core.step()  # Drain native finished-ID notifications before the next salted rollout.
                assert not scheduler.requests
                raw.flush();flush()
                print(json.dumps(dict(event='rollout-collected',rollout=spec['rollout_id'],rows=row_cursor,
                    stop_reason=entry['stop_reason'],elapsed_s=entry['finished_unix']-entry['started_unix'])),flush=True)
        assert len(pps)==20
        result.update(status='complete',initial_pps=statistics.median(pps),initial_pps_samples=pps)
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:
        result['finished_unix']=time.time();flush()
        try:
            if core is not None:core.shutdown()
        finally:
            if nvml_initialized:nvml.nvmlShutdown()


if __name__=='__main__':main()
