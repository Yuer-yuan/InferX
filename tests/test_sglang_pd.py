"""Admission accounting and source overlays for actual SGLang experiments."""
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def implementation():
    assert importlib.util.find_spec('inferx.sglang_pd') is not None, 'PD strategy implementation missing'
    return importlib.import_module('inferx.sglang_pd')


def adder(waiting=1, budget=512):
    return NS(waiting_queue_len=waiting, rem_chunk_tokens=budget, rem_total_tokens=10000,
              can_run_list=[])


@pytest.mark.parametrize('cap', [128,256])
def test_reserved_budget_can_admit_a_new_short_request(cap):
    m=implementation(); a=adder(); request=NS(remaining=10000)
    def original(req):
        consumed=min(req.remaining,a.rem_chunk_tokens)
        a.rem_chunk_tokens-=consumed; a.rem_total_tokens-=consumed
        a.can_run_list.append(req)
        return req
    assert m.cap_chunk_budget(a,request,original,cap) is request
    assert a.rem_chunk_tokens==512-cap
    assert a.rem_total_tokens==10000-cap
    assert a.can_run_list==[request]
    # The unspent round budget remains usable by a newly arriving short prefill.
    assert a.rem_chunk_tokens>=41


@pytest.mark.parametrize('waiting,budget,remaining,expected',[(0,512,10000,0),(1,64,10000,0),(1,512,20,492),(1,0,20,0)])
def test_original_progress_and_short_final_chunk_are_preserved(waiting,budget,remaining,expected):
    m=implementation();a=adder(waiting,budget)
    def original(req):
        a.rem_chunk_tokens-=min(remaining,a.rem_chunk_tokens)
        return 'original-result'
    assert m.cap_chunk_budget(a,object(),original,128)=='original-result'
    assert a.rem_chunk_tokens==expected


def test_exception_restores_unspent_reservation():
    m=implementation();a=adder()
    def original(req):raise RuntimeError('original failure')
    with pytest.raises(RuntimeError,match='original failure'):
        m.cap_chunk_budget(a,object(),original,128)
    assert a.rem_chunk_tokens==512


@pytest.mark.parametrize('cap',[0,-1,True,1.5])
def test_invalid_caps_fail_closed(cap):
    m=implementation()
    with pytest.raises(ValueError):m.cap_chunk_budget(adder(),object(),lambda _:None,cap)


def test_original_capacity_rejection_is_not_overridden():
    m=implementation();a=adder()
    token=object()
    assert m.cap_chunk_budget(a,object(),lambda _:token,128) is token
    assert a.rem_chunk_tokens==512 and a.can_run_list==[]


def test_overlay_is_independent_and_refuses_overwrite(tmp_path):
    m=implementation();src=tmp_path/'installed';src.mkdir()
    target=src/'srt/managers/schedule_policy.py';target.parent.mkdir(parents=True)
    original='class PrefillAdder:\n    pass\n';target.write_text(original)
    out=tmp_path/'overlay'
    result=m.prepare_overlay(src,out,'reserve',128)
    assert target.read_text()==original
    assert (out/'sglang/srt/managers/schedule_policy.py').read_text()!=original
    manifest=json.loads((out/'overlay-manifest.json').read_text())
    assert manifest==result and manifest['policy']=='reserve' and manifest['cap']==128
    assert manifest['files']['srt/managers/schedule_policy.py']['before_sha256']
    with pytest.raises(FileExistsError):m.prepare_overlay(src,out,'reserve',128)


def test_overlay_rejects_nested_destination(tmp_path):
    m=implementation()
    with pytest.raises(ValueError):m.prepare_overlay(tmp_path,tmp_path/'nested','control',None)


def test_instrumentation_distinguishes_admission_rejection(caplog):
    m=implementation()
    class Original:
        def __init__(self):self.__dict__.update(adder().__dict__)
        def add_chunked_req(self,req):self.rem_chunk_tokens-=128;return req
        def add_one_req(self,req,**kwargs):return NS(name='NO_TOKEN')
    m.install_admission_hooks(Original,128)
    a=Original();req=NS(rid='short',prefix_indices=[1,2],full_untruncated_fill_ids=[1,2,3])
    with caplog.at_level('INFO',logger='inferx.pd'):
        r=a.add_one_req(req,has_chunked_req=True,truncation_align_size=None)
    assert r.name=='NO_TOKEN' and a.can_run_list==[]
    event=json.loads(caplog.records[-1].message.split('PD_ADMISSION ',1)[1])
    assert event['result']=='NO_TOKEN' and event['cached_tokens']==2
    assert event['remaining_input']==1 and not event['admitted']


def split_model():
    calls=[]
    def layer(index):
        def execute(*,positions,hidden_states,residual,forward_batch):
            calls.append(('layer',index))
            return hidden_states+index+1,(residual or 0)+hidden_states
        return execute
    def norm(h,r):calls.append(('norm',h+r));return h+r,None
    def logits(ids,h,head,batch,aux=None):calls.append(('logits',h));return h*3
    model=NS(embed_tokens=lambda ids:ids,layers=[layer(i) for i in range(3)],
             norm=norm,flashinfer_mnnvl_cutedsl_fusion=None)
    return NS(model=model,config=NS(num_hidden_layers=3,num_experts=0),language_model_only=True,
              capture_aux_hidden_states=False,is_mrope_enabled=False,
              pp_group=NS(is_first_rank=True,is_last_rank=True,world_size=1),
              logits_processor=logits,lm_head=object()),calls


def test_split_layers_match_independent_monolithic_result_and_normalize_once():
    m=implementation();model,calls=split_model();batch=NS(spec_info=None,residual=999)
    assert m.qwen_split_prefill(model,2,None,batch,(0,1)) is None
    assert batch.hidden_states==3 and batch.residual==2
    assert m.qwen_split_prefill(model,2,None,batch,(1,3))==54
    assert calls==[('layer',0),('layer',1),('layer',2),('norm',18),('logits',18)]


def test_split_batches_keep_independent_residual_state_when_interleaved():
    m=implementation();model,_=split_model();a=NS(spec_info=None);b=NS(spec_info=None)
    m.qwen_split_prefill(model,2,None,a,(0,2))
    m.qwen_split_prefill(model,5,None,b,(0,1))
    assert m.qwen_split_prefill(model,2,None,a,(2,3))==54
    assert m.qwen_split_prefill(model,5,None,b,(1,3))==90


@pytest.mark.parametrize('interval',[(1,2),(-1,1),(0,4),(0,0),(2,1)])
def test_invalid_or_noncontiguous_split_rejected(interval):
    m=implementation();model,calls=split_model()
    with pytest.raises(ValueError):m.qwen_split_prefill(model,2,None,NS(spec_info=None),interval)
    assert calls==[]


@pytest.mark.parametrize('flag',['language_model_only','capture_aux_hidden_states','is_mrope_enabled','pp'])
def test_split_rejects_unsupported_execution_modes(flag):
    m=implementation();model,calls=split_model()
    if flag=='pp':model.pp_group.world_size=2
    else:setattr(model,flag,flag!='language_model_only')
    with pytest.raises(ValueError):m.qwen_split_prefill(model,2,None,NS(spec_info=None),(0,3))
    assert calls==[]


def test_installed_split_adapter_rejects_tensor_parallelism():
    m=implementation()
    class Model:pass
    m.install_qwen_split(Model,lambda:2)
    with pytest.raises(ValueError,match='TP1'):Model().forward_split_prefill(2,None,NS(),(0,3))


def test_pdmux_topology_uses_current_parallel_state():
    m=implementation()
    source='class Mix:\n    def init(self):\n        return self.gpu_id, self.tp_size\n'
    fixed=m.patch_pdmux_topology(source)
    namespace={};exec(fixed,namespace)
    scheduler=namespace['Mix']();scheduler.ps=NS(gpu_id=0,tp_size=1)
    assert scheduler.init()==(0,1)
    with pytest.raises(ValueError):m.patch_pdmux_topology('class Updated: pass\n')


def test_dense_model_is_not_rejected_by_unused_legacy_expert_count():
    m=implementation();model,_=split_model()
    model.config.num_experts=512;model.config.model_type='qwen3_5_text'
    assert m.qwen_split_prefill(model,2,None,NS(spec_info=None),(0,3))==54
    model.config.model_type='qwen3_5_moe_text'
    with pytest.raises(ValueError):m.qwen_split_prefill(model,2,None,NS(spec_info=None),(0,3))


def test_pdmux_chunk_transition_keeps_unfinished_request_out_of_decode():
    m=implementation();long=NS(prefix_indices=[0,1],extend_range=NS(end=5));short=object()
    class Batch:
        def __init__(self):self.reqs=[long,short]
        def filter_batch(self,chunked_req_to_exclude):
            self.reqs=[r for r in self.reqs if not any(r is x for x in chunked_req_to_exclude)]
    def stash(req):req.prefix_indices=list(range(req.extend_range.end))
    scheduler=NS(chunked_req=long,stash_chunked_request=stash)
    b=Batch();assert m.finish_pdmux_chunk(scheduler,b) is b
    assert b.reqs==[short] and long.prefix_indices==[0,1,2,3,4]
    assert scheduler.chunked_req is long


def test_token_chunking_pdmux_overlay_preserves_stock_other_validation():
    m=implementation()
    original='def check(cfg):\n    assert cfg.chunked_prefill_size == -1\n    assert cfg.disable_overlap_schedule\n'
    namespace={};exec(m.patch_pdmux_chunk_validation(original),namespace)
    namespace['check'](NS(chunked_prefill_size=512,disable_overlap_schedule=True))
    with pytest.raises(AssertionError):namespace['check'](NS(chunked_prefill_size=512,disable_overlap_schedule=False))
    with pytest.raises(AssertionError):namespace['check'](NS(chunked_prefill_size=1024,disable_overlap_schedule=True))


def test_reservation_never_creates_a_second_unfinished_chunk():
    m=implementation();other=NS(name='OTHER')
    class Original:
        def __init__(self):self.__dict__.update(adder().__dict__);self.page_size=1
        def add_chunked_req(self,req):return req
        def add_one_req(self,req,**kwargs):self.can_run_list.append(req);return 'accepted'
    m.install_admission_hooks(Original,128,blocked_result=other)
    a=Original();a.rem_chunk_tokens=384
    cold=NS(prefix_indices=[],full_untruncated_fill_ids=[0]*24000)
    assert a.add_one_req(cold,has_chunked_req=True,truncation_align_size=None) is other
    assert a.can_run_list==[]
    cached=NS(prefix_indices=[0]*23960,full_untruncated_fill_ids=[0]*24000)
    assert a.add_one_req(cached,has_chunked_req=True,truncation_align_size=None)=='accepted'
    assert a.can_run_list==[cached]


def test_active_decode_is_protected_after_waiting_queue_drains():
    m=implementation();a=adder(waiting=0)
    a.running_batch=NS(is_empty=lambda:False)
    def original(req):a.rem_chunk_tokens-=min(512,a.rem_chunk_tokens);return req
    req=object()
    assert m.cap_chunk_budget(a,req,original,128,protect_decode=True) is req
    assert a.rem_chunk_tokens==384
    a.rem_chunk_tokens=512;a.running_batch=NS(is_empty=lambda:True)
    m.cap_chunk_budget(a,req,original,128,protect_decode=True)
    assert a.rem_chunk_tokens==0


def test_prepared_overlay_actually_applies_decode_protection(tmp_path,monkeypatch):
    m=implementation();src=tmp_path/'source';target=src/'srt/managers/schedule_policy.py'
    target.parent.mkdir(parents=True)
    target.write_text('''class AddReqResult:
    OTHER = 'OTHER'
class PrefillAdder:
    def __init__(self):
        self.waiting_queue_len=0
        self.rem_chunk_tokens=512
        self.rem_total_tokens=10000
        self.can_run_list=[]
        self.running_batch=type('Batch',(),{'is_empty':lambda s:False})()
    def add_chunked_req(self,req):
        self.rem_chunk_tokens-=min(self.rem_chunk_tokens,10000)
        return req
    def add_one_req(self,req,**kwargs):
        return AddReqResult.OTHER
''')
    out=tmp_path/'overlay';manifest=m.prepare_overlay(src,out,'reserve',128,protect_decode=True)
    assert manifest['protect_decode'] is True
    monkeypatch.syspath_prepend(str(out));monkeypatch.delitem(__import__('sys').modules,'inferx_pd_runtime',raising=False)
    scope=__import__('runpy').run_path(str(out/'sglang/srt/managers/schedule_policy.py'))
    a=scope['PrefillAdder']();req=NS(prefix_indices=[],full_untruncated_fill_ids=[0]*10000)
    a.add_chunked_req(req)
    assert a.rem_chunk_tokens==384


def test_owned_chunk_row_is_not_charged_against_free_request_slots():
    m=implementation();held=NS(kv=NS(holds_kv=True));new=NS(kv=NS(holds_kv=False))
    assert m.owned_admission_limit(1,[held],2)==2
    assert m.owned_admission_limit(1,[held,new],2)==2
    assert m.owned_admission_limit(4,[held],4)==4  # retain hard concurrency bound
    assert m.owned_admission_limit(0,[held],2)==1  # no new row becomes available
    assert m.owned_admission_limit(2,[new],3)==2
    assert m.owned_admission_limit(1,[held],2,credit_existing=False)==1


def test_green_context_binding_restores_context_on_exception_and_nested_scopes():
    m=implementation()
    class Driver:
        value=10
        def current(self):return self.value
        def context_for_stream(self,stream):return {1:20,2:30,3:None}[stream]
        def set_current(self,value):self.value=value
    api=Driver()
    with pytest.raises(RuntimeError,match='body failure'):
        with m.bind_stream_context(1,api):
            assert api.current()==20
            with m.bind_stream_context(2,api):assert api.current()==30
            assert api.current()==20
            raise RuntimeError('body failure')
    assert api.current()==10
    with m.bind_stream_context(3,api):assert api.current()==10
