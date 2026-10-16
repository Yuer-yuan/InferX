"""MorphX engine port: shared hybrid state with two primary-context P/D streams.

SGLang supplies admission/batching and hybrid model execution. Resource control
belongs to the real MorphX compiler/profiler/runtime, with no GreenContext SM
partition. This is an adaptation, not the unavailable author's vLLM fork.
"""
import logging
import re


def initialize_groups(namespace,gpu_id,config,torch_module=None):
    if config.sm_group_num!=3:raise ValueError('MorphX port requires three logical groups and one actual P/D stream pair')
    if torch_module is None:
        import torch as torch_module
    total=torch_module.cuda.get_device_properties(gpu_id).multi_processor_count
    prefill=torch_module.cuda.Stream(gpu_id);decode=torch_module.cuda.Stream(gpu_id)
    pair=(prefill,decode)
    namespace.update(STREAM_GROUPS=[pair,pair,pair],SM_COUNTS=[(total,0),(total,total),(0,total)],SM_GROUP_NUM=3,CURRENT_STREAM_IDX=0,CURRENT_STREAM_GROUP=pair)
    logging.getLogger('inferx.morphx').info('MORPHX_PD_PRIMARY_STREAMS prefill=%s decode=%s available_sms=%s',prefill.cuda_stream,decode.cuda_stream,total)


def patch_primary_stream_groups(source):
    pattern=r'def initialize_stream_groups\(gpu_id: int, config: PDMuxConfig\):\n    from sgl_kernel import spatial\n.*?(?=\ndef set_current_stream_idx\()'
    replacement='''def initialize_stream_groups(gpu_id: int, config: PDMuxConfig):
    from inferx_morphx_runtime import initialize_groups
    initialize_groups(globals(), gpu_id, config)

'''
    changed,count=re.subn(pattern,replacement,source,flags=re.DOTALL)
    if count!=1:raise ValueError('unexpected or already patched PDMux stream initializer')
    return changed


def patch_marlin_loader(source):
    pattern=r'(def _jit_gptq_marlin_module\(dtype: torch.dtype\) -> Module:\n)    args = make_cpp_args\(dtype\)\n.*?(?=\ndef _or_empty\()'
    replacement='''    import os
    from tvm_ffi import load_module
    if dtype != torch.bfloat16:
        raise ValueError('custom MorphX Marlin qualified for BF16 only')
    return load_module(os.environ['INFERX_MORPHX_MARLIN'])

'''
    changed,count=re.subn(pattern,lambda m: m.group(1)+replacement,source,flags=re.DOTALL)
    if count!=1:raise ValueError('unexpected or already patched Marlin JIT loader')
    return changed
