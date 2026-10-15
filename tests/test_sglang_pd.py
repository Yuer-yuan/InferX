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
