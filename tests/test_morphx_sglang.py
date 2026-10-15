from types import SimpleNamespace
import pytest


def test_pd_streams_share_primary_context_and_report_available_sms():
    from inferx.morphx_sglang import initialize_groups
    streams=[]
    class CUDA:
        @staticmethod
        def get_device_properties(gpu):return SimpleNamespace(multi_processor_count=128)
        @staticmethod
        def Stream(gpu):
            obj=SimpleNamespace(device=gpu,cuda_stream=len(streams)+100);streams.append(obj);return obj
    ns={};config=SimpleNamespace(sm_group_num=3,manual_divisions=[[96,32,1]])
    initialize_groups(ns,0,config,SimpleNamespace(cuda=CUDA))
    assert len(streams)==2
    assert ns['STREAM_GROUPS']==[(streams[0],streams[1])]*3
    assert ns['SM_COUNTS']==[(128,0),(128,128),(0,128)]
    assert ns['CURRENT_STREAM_GROUP']==ns['STREAM_GROUPS'][0]
    assert ns['CURRENT_STREAM_IDX']==0 and ns['SM_GROUP_NUM']==3
    with pytest.raises(ValueError):initialize_groups({},0,SimpleNamespace(sm_group_num=4),SimpleNamespace(cuda=CUDA))


def test_stream_patch_changes_only_the_pinned_initializer():
    from inferx.morphx_sglang import patch_primary_stream_groups
    source='def initialize_stream_groups(gpu_id: int, config: PDMuxConfig):\n    from sgl_kernel import spatial\n    spatial.init()\n\ndef set_current_stream_idx(idx):\n    pass\n'
    changed=patch_primary_stream_groups(source)
    assert 'spatial.init' not in changed and 'initialize_groups(globals(), gpu_id, config)' in changed
    assert changed.endswith('def set_current_stream_idx(idx):\n    pass\n')
    with pytest.raises(ValueError):patch_primary_stream_groups(changed)


def test_custom_marlin_loader_uses_exact_library_and_rejects_other_dtypes(monkeypatch):
    import sys
    from inferx.morphx_sglang import patch_marlin_loader
    source='''def _jit_gptq_marlin_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit("original", *args)


def _or_empty(x):
    return x
'''
    changed=patch_marlin_loader(source)
    marker=object();seen=[];bf16=object();torch=SimpleNamespace(dtype=object,bfloat16=bf16)
    monkeypatch.setitem(sys.modules,'tvm_ffi',SimpleNamespace(load_module=lambda path:seen.append(path) or marker))
    monkeypatch.setenv('INFERX_MORPHX_MARLIN','/hdd/custom.so')
    ns={'torch':torch,'Module':object};exec(changed,ns)
    assert ns['_jit_gptq_marlin_module'](bf16) is marker
    assert seen==['/hdd/custom.so'] and ns['_or_empty'](3)==3
    with pytest.raises(ValueError):ns['_jit_gptq_marlin_module'](object())
    with pytest.raises(ValueError):patch_marlin_loader(changed)
