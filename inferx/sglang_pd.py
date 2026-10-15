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


def cap_chunk_budget(adder, req, original, cap):
    if type(cap) is not int or cap <= 0:
        raise ValueError('positive integer chunk cap required')
    budget = adder.rem_chunk_tokens
    if not adder.waiting_queue_len or budget is None or budget <= cap:
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


def install_admission_hooks(cls, cap=None):
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
                cap_chunk_budget(self,req,lambda r:chunk(self,r),cap))
        _admission_event(self,req,'chunk',before,result)
        return result
    @functools.wraps(one)
    def add_one(self, req, *args, **kwargs):
        before=self.rem_chunk_tokens
        result=one(self,req,*args,**kwargs)
        _admission_event(self,req,'new',before,result)
        return result
    cls.add_chunked_req=add_chunked
    cls.add_one_req=add_one
    cls._inferx_admission_installed=True


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare_overlay(source: Path, destination: Path, policy: str, cap=None) -> dict:
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('overlay must be independent of installed source')
    if destination.exists():
        raise FileExistsError(destination)
    if policy not in {'control','reserve','pdmux'}:
        raise ValueError('unknown overlay policy')
    if policy=='reserve' and (type(cap) is not int or cap not in (128,256)):
        raise ValueError('reserve cap must be128 or256')
    if policy!='reserve' and cap is not None:
        raise ValueError('only reserve policy accepts a cap')
    destination.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.prepare-',dir=destination.parent))
    try:
        shutil.copytree(source,staging/'sglang',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        manifest=dict(policy=policy,cap=cap,source=str(source),files={})
        target=staging/'sglang/srt/managers/schedule_policy.py'
        original=target.read_bytes()
        if b'class PrefillAdder' not in original:
            raise ValueError('expected PrefillAdder source anchor missing')
        footer=('\nfrom inferx_pd_runtime import install_admission_hooks as _inferx_install\n'
                f'_inferx_install(PrefillAdder, cap={cap!r})\n')
        target.write_bytes(original+footer.encode())
        manifest['files']['srt/managers/schedule_policy.py']=dict(
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
