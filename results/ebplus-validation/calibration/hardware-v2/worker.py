"""Actual pinned-author calibration on the matched modern model/GPU engine."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from inferx.ebplus_calibration import validate_plan,validate_hardware_result
from inferx.ebplus_validation import CALIBRATION_SHA256,NATIVE_SHA256


def calibration_engine_args(plan,root):
    """Retain hybrid alignment while request salts prohibit prefix reuse."""
    from vllm.engine.arg_utils import EngineArgs
    case=plan['base_case']
    return EngineArgs(model=str(root/case['model']),dtype='bfloat16',max_model_len=case['context_window'],
        max_num_batched_tokens=case['chunk'],max_num_seqs=case['concurrency'],
        long_prefill_token_threshold=case['long_prefill_token_threshold'],
        gpu_memory_utilization=case['gpu_memory_utilization'],kv_cache_dtype=case['kv_dtype'],
        mamba_cache_mode=case['mamba_cache_mode'],enable_chunked_prefill=True,
        enable_prefix_caching=True,enforce_eager=True,async_scheduling=False,
        distributed_executor_backend='uni',language_model_only=True)


def calibration_request(prompt_len,max_tokens):
    """Current request ABI; the author's random-token, fixed-batch protocol."""
    import numpy as np
    from vllm import SamplingParams
    from vllm.v1.engine import EngineCoreRequest
    rid='calibration_'+uuid.uuid4().hex
    return EngineCoreRequest(request_id=rid,
        prompt_token_ids=[int(n) for n in np.random.randint(100,10000,size=prompt_len)],
        mm_features=None,sampling_params=SamplingParams(max_tokens=max_tokens,ignore_eos=True),
        pooling_params=None,arrival_time=time.time(),lora_request=None,
        cache_salt=rid,data_parallel_rank=None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();plan=json.loads(a.plan.read_text());validate_plan(plan)
    assert not a.output.exists()
    root=Path(os.environ['INFERX_ROOT']);case=plan['base_case'];stage=plan['decomposed']
    assert os.environ['CUDA_VISIBLE_DEVICES']==plan['identity']['gpu_uuid']
    source=root/'sota-validation/sources/eb-vllm-repo/vllm/v1/core/sched/calibration.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==CALIBRATION_SHA256
    spec=importlib.util.spec_from_file_location('inferx_pinned_author_calibration',source)
    author=importlib.util.module_from_spec(spec);sys.modules[spec.name]=author;spec.loader.exec_module(author)
    import torch
    import vllm.v1.core.sched.scheduler as native
    assert hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()==NATIVE_SHA256
    assert torch.cuda.device_count()==1
    result=dict(status='starting',identity=plan['identity'],measurement_type='decomposed_cuda_events',
                source_sha256=CALIBRATION_SHA256,worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),raw_steps=[],started_unix=time.time())

    def flush():
        tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(a.output)

    class MatchedCalibrator(author.HardwareCalibrator):
        def setup(self):
            from vllm.v1.engine.core import EngineCore
            from vllm.v1.executor.uniproc_executor import UniProcExecutor
            from vllm.utils.torch_utils import set_default_torch_num_threads
            args=calibration_engine_args(plan,root)
            cfg=args.create_engine_config()
            assert cfg.speculative_config is None and cfg.max_concurrent_batches==1
            assert cfg.model_config.quantization=='compressed-tensors'
            assert cfg.cache_config.mamba_cache_mode==case['mamba_cache_mode']
            with set_default_torch_num_threads(1):
                self.engine_core=EngineCore(vllm_config=cfg,executor_class=UniProcExecutor,log_stats=False)
            current=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
            rows={(int(x.split(',')[0]),x.split(',')[1].strip()) for x in current.splitlines() if x.strip()}
            assert rows=={(os.getpid(),case['gpu_uuid'])},rows
            s=self.engine_core.scheduler
            result.update(status='running',gpu_process=dict(pid=os.getpid(),uuid=case['gpu_uuid']),
                device_name=torch.cuda.get_device_name(0),torch_version=torch.__version__,
                native_runtime=dict(max_model_len=s.max_model_len,max_num_running_reqs=s.max_num_running_reqs,
                    block_size=s.block_size,num_gpu_blocks=s.kv_cache_config.num_blocks,
                    kv_dtype=cfg.cache_config.cache_dtype,quantization=cfg.model_config.quantization,
                    mamba_cache_mode=cfg.cache_config.mamba_cache_mode,
                    kv_groups=[dict(type=type(g.kv_cache_spec).__name__,spec=str(g.kv_cache_spec),layers=g.layer_names)
                               for g in s.kv_cache_config.kv_cache_groups]))
            flush()

        def _add_request(self,prompt_len,max_tokens=1000):
            req=calibration_request(prompt_len,max_tokens)
            self.engine_core.add_request(*self.engine_core.preprocess_add_request(req))
            return req.request_id

        def measure_prefill_times(self):
            self.stage='prefill';data=super().measure_prefill_times();result['prefill_medians_ms']=data;flush();return data

        def measure_decode_times(self):
            self.stage='decode';data=super().measure_decode_times();result['decode_medians_ms']=data;flush();return data

        def _run_single_step(self):
            s=self.engine_core.scheduler
            if self.stage=='prefill':
                waiting=list(s.waiting);assert len(waiting)==1 and not s.running
                target=waiting[0].num_tokens-waiting[0].num_computed_tokens
            else:
                target=len(s.running);assert not s.waiting and target in stage['decode_counts']
            output=super()._run_single_step()  # Original CUDA-event timing and current executor RPC.
            nd,np,tokens,elapsed=output
            result['raw_steps'].append(dict(stage=self.stage,target=target,num_decode=nd,num_prefill=np,
                                            tokens=tokens,elapsed_ms=elapsed))
            flush();return output

    calibrator=MatchedCalibrator(model=str(root/case['model']),dtype='bfloat16',
        prefill_sizes=stage['prefill_sizes'],decode_counts=stage['decode_counts'],
        decode_context_len=stage['decode_context_len'],num_warmup=stage['num_warmup'],num_iterations=stage['num_iterations'],
        max_num_batched_tokens=case['chunk'],max_num_seqs=case['concurrency'],gpu_memory_utilization=case['gpu_memory_utilization'])
    flush()
    try:
        calibrator.setup();params=calibrator.calibrate()
        result.update(parameters={k:getattr(params,k) for k in ['alpha_p','beta_p','alpha_d','beta_d']},
                      prefill_sizes=params.prefill_sizes,decode_counts=params.decode_counts,
                      prefill_r2=params.prefill_r2,decode_r2=params.decode_r2,
                      author_hardware_params=author.asdict(params))
        import numpy as np
        result['unclipped_linear_fits']={
            key:list(author.linear_regression(np.array([x[0] for x in values]),
                                             np.array([x[1]/1000 for x in values])))
            for key,values in [('prefill',result['prefill_medians_ms']),('decode',result['decode_medians_ms'])]}
        validate_hardware_result(result,plan)
        params.save(a.output.with_name('pd-hardware-parameters.json'))
        result['status']='complete'
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:
        result['finished_unix']=time.time();flush()
        if calibrator.engine_core is not None:calibrator.engine_core.shutdown()


if __name__=='__main__':main()
