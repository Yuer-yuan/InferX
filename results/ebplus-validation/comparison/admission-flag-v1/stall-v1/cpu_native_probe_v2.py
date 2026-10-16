"""Diagnose all-partial-prefill progress on actual native CPU allocator APIs.

This is logical-cache diagnosis, not model performance or a source-ceiling claim.
Run only with no timed GPU campaign; keep the qualified source snapshot frozen.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


root=Path(os.environ['INFERX_ROOT'])
base=root/'ebplus-validation'
code=base/'code-task3-v14'
assert not Path('/proc/4056513').exists() and not Path('/proc/4073454').exists()
assert json.loads((base/'admission-flag-v1/stall-v1/after-stop-service.json').read_text())['status']=='verified'
sys.path.insert(0,str(code))
spec=importlib.util.spec_from_file_location('eb_probe',code/'tests/test_ebplus_native.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
fixture=module.NativeHybridContractTests;fixture.setUpClass()
case=fixture(methodName='runTest');observations=[]
try:
    for kind,reserve,lengths in [('native',False,[30720,20000]),('eb',False,[30720]),
                                ('eb',True,[30720,20000]),('eb',False,[30720,20000])]:
        scheduler=case.scheduler(mode='eb',blocks=128,reserve_full=reserve,
            cls=None if kind=='eb' else case.scheduler_cls.__bases__[-1])
        requests=[case.add(scheduler,f'p{i}',length=n,output=4,token_id=40+i) for i,n in enumerate(lengths)]
        empty=0;start=time.time()
        for step in range(500):
            if not scheduler.requests:break
            output=case.step(scheduler)
            empty=empty+1 if output.total_num_scheduled_tokens==0 else 0
            if empty==64:break
        record=dict(kind=kind,reserve_full=reserve,lengths=lengths,steps=step+1,
            empty_streak=empty,remaining_requests=len(scheduler.requests),elapsed_s=time.time()-start,
            phase=getattr(scheduler,'pd_phase',None),N=getattr(scheduler,'pd_batch_size_N',None),
            decoding=sorted(getattr(scheduler,'pd_decoding_requests',[])),
            free=scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
            running=[r.request_id for r in scheduler.running],
            waiting=[r.request_id for r in scheduler.waiting],
            requests=[dict(id=r.request_id,prompt=r.num_prompt_tokens,computed=r.num_computed_tokens,
                           output=len(r.output_token_ids),preemptions=r.num_preemptions) for r in requests])
        if empty==64:
            checks=[]
            for request in scheduler.running:
                tokens=scheduler._mamba_block_aligned_split(request,min(2048,request.num_tokens-request.num_computed_tokens))
                before=scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
                allocation=scheduler.kv_cache_manager.allocate_slots(request,tokens,num_lookahead_tokens=0)
                checks.append(dict(id=request.request_id,tokens=tokens,returned_none=allocation is None,
                    free_before=before,free_after=scheduler.kv_cache_manager.block_pool.get_num_free_blocks()))
            record['each_native_allocation_after_stall']=checks
        observations.append(record);print(json.dumps(record),flush=True)
finally:fixture.tearDownClass()
target=base/'admission-flag-v1/stall-v1/cpu-native-reproduction-v2.json'
with target.open('x') as stream:
    json.dump(dict(scope='CPU-native128block/256token logical hybrid cache; not27B performance',
                   source_snapshot=str(code),observations=observations),stream,indent=2)
    stream.write('\n')
