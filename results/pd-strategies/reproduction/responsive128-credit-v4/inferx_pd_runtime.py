"""Narrow, explicitly labeled SGLang source-overlay experiments."""
import functools
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time


def cap_chunk_budget(adder, req, original, cap, protect_decode=False):
    if type(cap) is not int or cap <= 0:
        raise ValueError('positive integer chunk cap required')
    budget = adder.rem_chunk_tokens
    running=getattr(adder,'running_batch',None)
    active_decode=protect_decode and running is not None and not running.is_empty()
    if (not adder.waiting_queue_len and not active_decode) or budget is None or budget <= cap:
        return original(req)
    adder.rem_chunk_tokens = cap
    try:
        return original(req)
    finally:
        # Preserve original accounting, including capacity decisions and actual
        # consumption; release only the temporary reservation for other work.
        adder.rem_chunk_tokens += budget-cap


def _admission_event(adder, req, method, before, result):
    prefix = len(getattr(req, 'prefix_indices', ()))
    remaining = len(getattr(req, 'full_untruncated_fill_ids', ()))-prefix
    event = dict(unix=time.time(), method=method, rid=getattr(req, 'rid', None),
        waiting=adder.waiting_queue_len, cached_tokens=prefix, remaining_input=remaining,
        chunk_before=before, chunk_after=adder.rem_chunk_tokens,
        total_after=adder.rem_total_tokens,
        result=getattr(result, 'name', str(result) if result is None else type(result).__name__),
        admitted=any(item is req for item in adder.can_run_list))
    logging.getLogger('inferx.pd').info('PD_ADMISSION %s', json.dumps(event,sort_keys=True))


def install_admission_hooks(cls, cap=None, blocked_result=None, protect_decode=False):
    if getattr(cls, '_inferx_admission_installed', False):
        raise RuntimeError('admission hooks already installed')
    if cap is not None and (type(cap) is not int or cap <= 0):
        raise ValueError('positive integer chunk cap required')
    chunk = cls.add_chunked_req
    one = cls.add_one_req
    @functools.wraps(chunk)
    def add_chunked(self, req):
        before=self.rem_chunk_tokens
        result=(chunk(self,req) if cap is None else
                cap_chunk_budget(self,req,lambda r:chunk(self,r),cap,protect_decode=protect_decode))
        _admission_event(self,req,'chunk',before,result)
        return result
    @functools.wraps(one)
    def add_one(self, req, *args, **kwargs):
        before=self.rem_chunk_tokens
        has_chunk=kwargs.get('has_chunked_req',args[0] if args else False)
        remaining=len(req.full_untruncated_fill_ids)-len(req.prefix_indices)
        page=getattr(self,'page_size',1)
        needed=-(-remaining//page)*page
        if cap is not None and has_chunk and before is not None and needed>before:
            # The stock scheduler tracks one unfinished chunk only. Its usual
            # all-budget consumption makes this branch unreachable; reserving
            # tokens must not let add_one_req start a second partial request.
            if blocked_result is None:
                raise RuntimeError('reserved admission requires original OTHER result')
            result=blocked_result
        else:
            result=one(self,req,*args,**kwargs)
        _admission_event(self,req,'new',before,result)
        return result
    cls.add_chunked_req=add_chunked
    cls.add_one_req=add_one
    cls._inferx_admission_installed=True


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def qwen_split_prefill(self, input_ids, positions, forward_batch, split_interval, input_embeds=None):
    """Dense Qwen3.5 text path, preserving the upstream layer calling convention."""
    model=self.model
    if (not self.language_model_only or getattr(self,'capture_aux_hidden_states',False)
            or getattr(self,'is_mrope_enabled',False) or self.pp_group.world_size!=1
            or 'moe' in getattr(self.config,'model_type','').lower()
            or 'Moe' in type(model).__name__
            or getattr(forward_batch,'spec_info',None) is not None
            or getattr(model,'flashinfer_mnnvl_cutedsl_fusion',None) is not None):
        details=dict(text_only=self.language_model_only,aux=getattr(self,'capture_aux_hidden_states',False),
            mrope=getattr(self,'is_mrope_enabled',False),pp=self.pp_group.world_size,
            experts=getattr(self.config,'num_experts',None),
            spec=type(getattr(forward_batch,'spec_info',None)).__name__,
            fusion=type(getattr(model,'flashinfer_mnnvl_cutedsl_fusion',None)).__name__)
        raise ValueError('unsupported split-prefill execution mode: '+repr(details))
    start,end=split_interval
    total=self.config.num_hidden_layers
    if not (0<=start<end<=total) or (start and getattr(forward_batch,'_inferx_split_next',None)!=start):
        raise ValueError('invalid or noncontiguous layer interval')
    if start==0:
        forward_batch.hidden_states=(model.embed_tokens(input_ids) if input_embeds is None else input_embeds)
        forward_batch.residual=None
    for index in range(start,end):
        forward_batch.hidden_states,forward_batch.residual=model.layers[index](
            positions=positions,hidden_states=forward_batch.hidden_states,
            residual=forward_batch.residual,forward_batch=forward_batch)
    forward_batch._inferx_split_next=end
    if start==0 or end==total:
        details=dict(unix=time.time(),start=start,end=end,layers=total)
        if getattr(input_ids,'is_cuda',False):
            import torch
            details['cuda_stream']=torch.cuda.current_stream().cuda_stream
        logging.getLogger('inferx.pd').info('PD_SPLIT %s',json.dumps(details,sort_keys=True))
    if end!=total:
        return None
    hidden,_=model.norm(forward_batch.hidden_states,forward_batch.residual)
    forward_batch.hidden_states=hidden
    return self.logits_processor(input_ids,hidden,self.lm_head,forward_batch,None)


def install_qwen_split(cls, parallel_size):
    def forward(self,input_ids,positions,forward_batch,split_interval,input_embeds=None):
        if parallel_size()!=1:
            raise ValueError('adapted Qwen split prefill supports TP1 only')
        return qwen_split_prefill(self,input_ids,positions,forward_batch,split_interval,input_embeds)
    cls.forward_split_prefill=forward


def patch_pdmux_topology(text):
    # Pinned wheel's scheduler stores topology under ps; legacy mixin still
    # references two removed aliases. Fail if the expected source has changed.
    if text.count('self.gpu_id')!=1 or text.count('self.tp_size')!=1:
        raise ValueError('unexpected PD-Multiplexing topology source')
    return text.replace('self.gpu_id','self.ps.gpu_id').replace('self.tp_size','self.ps.tp_size')


def finish_pdmux_chunk(scheduler, batch):
    """Mirror the normal scheduler's stash/exclude transition before decode."""
    chunk=scheduler.chunked_req
    if chunk is not None:
        if not any(req is chunk for req in batch.reqs):
            raise RuntimeError('ongoing chunk does not belong to completed prefill batch')
        if chunk.extend_range.end>len(chunk.prefix_indices):
            scheduler.stash_chunked_request(chunk)
    batch.filter_batch(chunked_req_to_exclude=[] if chunk is None else [chunk])
    return batch


def patch_pdmux_chunk_validation(text):
    anchor='cfg.chunked_prefill_size == -1'
    if text.count(anchor)!=1:
        raise ValueError('unexpected token-chunk validation source')
    return text.replace(anchor,'cfg.chunked_prefill_size in (-1, 512)')


def _patch_chunk_transition(text):
    original='''                        if running_batch and not running_batch.is_empty():
                            running_batch.merge_batch(self.split_prefill_batch)
                        else:
                            running_batch = self.split_prefill_batch
                        self.running_batch = running_batch'''
    modified='''                        ready = _inferx_finish_chunk(self, self.split_prefill_batch)
                        if not ready.is_empty():
                            if running_batch and not running_batch.is_empty():
                                running_batch.merge_batch(ready)
                            else:
                                running_batch = ready
                        self.running_batch = running_batch
                        # Stash/filter/merge enqueue metadata on the prefill
                        # stream; make it visible before next decode preparation.
                        decode_stream.wait_stream(prefill_stream)'''
    if text.count(original)!=1:
        raise ValueError('unexpected PDMux prefill-to-decode merge source')
    return text.replace(original,modified)+'\nfrom inferx_pd_runtime import finish_pdmux_chunk as _inferx_finish_chunk\n'


def owned_admission_limit(free_limit, requests, hard_limit, credit_existing=True):
    owned=sum(bool(getattr(getattr(req,'kv',None),'holds_kv',False)) for req in requests) if credit_existing else 0
    limit=min(max(0,hard_limit),free_limit+owned)
    if owned and limit!=free_limit:
        logging.getLogger('inferx.pd').info('PD_CAPACITY %s',json.dumps(dict(
            unix=time.time(),free_limit=free_limit,owned_rows=owned,hard_limit=hard_limit,
            can_run=len(requests),corrected_limit=limit),sort_keys=True))
    return limit


def _patch_owned_capacity(text):
    original='''            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(
                running_bs,
                candidate_beam_width,
                running_batch=running_batch,
            ):'''
    replacement='''            if len(adder.can_run_list) >= _inferx_owned_limit(
                self.get_num_allocatable_reqs(
                    running_bs, candidate_beam_width, running_batch=running_batch),
                adder.can_run_list,
                min(self.max_running_requests, get_parallel().pp_max_micro_batch_size)-running_bs,
                credit_existing=(candidate_beam_width is None),
            ):'''
    if text.count(original)!=1:
        raise ValueError('unexpected scheduler admission-capacity source')
    return text.replace(original,replacement)+'\nfrom inferx_pd_runtime import owned_admission_limit as _inferx_owned_limit\n'


def prepare_overlay(source: Path, destination: Path, policy: str, cap=None, protect_decode=False, credit_owned_slots=False) -> dict:
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('overlay must be independent of installed source')
    if destination.exists():
        raise FileExistsError(destination)
    if policy not in {'control','reserve','responsive','pdmux','pdmux-chunked'}:
        raise ValueError('unknown overlay policy')
    if policy=='reserve' and (type(cap) is not int or cap not in (128,256)):
        raise ValueError('reserve cap must be128 or256')
    if policy=='responsive' and (type(cap) is not int or cap not in (64,128,256)):
        raise ValueError('responsive cap must be64,128 or256')
    if policy=='pdmux-chunked' and cap is not None and (type(cap) is not int or cap not in (64,128,256)):
        raise ValueError('chunked PDMux cap must be64,128 or256')
    if policy not in {'reserve','responsive','pdmux-chunked'} and cap is not None:
        raise ValueError('only reserve policy accepts a cap')
    protect_decode=protect_decode or policy=='responsive'
    if protect_decode and cap is None:
        raise ValueError('decode protection requires a cap')
    destination.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.prepare-',dir=destination.parent))
    try:
        shutil.copytree(source,staging/'sglang',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        manifest=dict(policy=policy,cap=cap,protect_decode=protect_decode,
                      credit_owned_slots=credit_owned_slots,source=str(source),files={})
        target=staging/'sglang/srt/managers/schedule_policy.py'
        original=target.read_bytes()
        if b'class PrefillAdder' not in original:
            raise ValueError('expected PrefillAdder source anchor missing')
        footer=('\nfrom inferx_pd_runtime import install_admission_hooks as _inferx_install\n'
                f'_inferx_install(PrefillAdder, cap={cap!r}, blocked_result=AddReqResult.OTHER, '
                f'protect_decode={protect_decode!r})\n')
        target.write_bytes(original+footer.encode())
        manifest['files']['srt/managers/schedule_policy.py']=dict(
            before_sha256=_sha(original),after_sha256=_sha(target.read_bytes()))
        if credit_owned_slots:
            target=staging/'sglang/srt/managers/scheduler.py';original=target.read_bytes()
            target.write_text(_patch_owned_capacity(original.decode()))
            manifest['files']['srt/managers/scheduler.py']=dict(
                before_sha256=_sha(original),after_sha256=_sha(target.read_bytes()))
        if policy.startswith('pdmux'):
            target=staging/'sglang/srt/models/qwen3_5.py'
            original=target.read_bytes()
            if b'class Qwen3_5ForConditionalGeneration' not in original:
                raise ValueError('expected dense Qwen wrapper source anchor missing')
            target.write_bytes(original+(
                '\nfrom inferx_pd_runtime import install_qwen_split as _inferx_split\n'
                '_inferx_split(Qwen3_5ForConditionalGeneration, lambda: get_parallel().tp_size)\n').encode())
            manifest['files']['srt/models/qwen3_5.py']=dict(
                before_sha256=_sha(original),after_sha256=_sha(target.read_bytes()))
            target=staging/'sglang/srt/multiplex/multiplexing_mixin.py'
            original=target.read_bytes()
            changed=patch_pdmux_topology(original.decode())
            if policy=='pdmux-chunked':changed=_patch_chunk_transition(changed)
            target.write_text(changed)
            manifest['files']['srt/multiplex/multiplexing_mixin.py']=dict(
                before_sha256=_sha(original),after_sha256=_sha(target.read_bytes()))
            if policy=='pdmux-chunked':
                target=staging/'sglang/srt/arg_groups/validation_hook.py'
                original=target.read_bytes()
                target.write_text(patch_pdmux_chunk_validation(original.decode()))
                manifest['files']['srt/arg_groups/validation_hook.py']=dict(
                    before_sha256=_sha(original),after_sha256=_sha(target.read_bytes()))
        runtime=Path(__file__).read_bytes()
        (staging/'inferx_pd_runtime.py').write_bytes(runtime)
        manifest['runtime_sha256']=_sha(runtime)
        inventory={str(p.relative_to(staging)):_sha(p.read_bytes())
                   for p in sorted(staging.rglob('*')) if p.is_file()}
        (staging/'source-hashes.json').write_text(json.dumps(inventory,indent=2)+'\n')
        manifest['source_hashes_sha256']=_sha((staging/'source-hashes.json').read_bytes())
        (staging/'overlay-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        os.rename(staging,destination)
        return manifest
    finally:
        if staging.exists():shutil.rmtree(staging)
