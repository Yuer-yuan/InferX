"""Source-bound EB+ policy extraction and a modern vLLM engine boundary.

The policy is the pinned author's code. Native hybrid allocation/state code
remains native; a generated subclass adds eligibility gates without hiding
requests or rewriting the installed vLLM package.
"""
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path

AUTHOR_SHA256 = '86b6f0fb3e7d4dc5b4cbd468df290d787aa743efb3c29f2b541cc19fa991b9c8'
CALIBRATION_SHA256 = '5628303f69d6fefef94338995c8e9ae2a6d7f0bea5ae0d24030d770bc8472658'
NATIVE_SHA256 = 'f820bee4f42b878b262a30f3728a97d7413477b2a277748e20f2528ac2ec2f2a'
AUTHOR_COMMIT = '5b6a619b732b8eeaea93c8c726dbe13948920599'
LICENSE_SHA256 = 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'


def phase_allows_running(phase, request, decoding_ids):
    """Exclusive phases change eligibility, never request/cache ownership."""
    is_prefill = request.num_computed_tokens < request.num_prompt_tokens
    if phase in (0, 2):
        return is_prefill
    if phase == 1:
        return not is_prefill and request.request_id in decoding_ids
    raise ValueError('invalid EB phase')


def phase_allows_admission(phase, target, prefilled, requests, scheduled):
    """Count prefills completing in this native step before admitting more."""
    if phase == 1:
        return False
    if phase not in (0, 2):
        raise ValueError('invalid EB phase')
    completing = sum(
        request.num_computed_tokens < request.num_prompt_tokens
        <= request.num_computed_tokens + scheduled_tokens
        for request_id, scheduled_tokens in scheduled.items()
        for request in (requests[request_id],)
    )
    return prefilled + completing < target


def record_native_runtime(scheduler):
    """Read actual initialized allocator/configuration once, before measurement."""
    import os
    path=os.environ.get('INFERX_VLLM_RUNTIME')
    if not path:return
    cfg=scheduler.vllm_config
    data=dict(model=cfg.model_config.model,context_window=scheduler.max_model_len,
              concurrency=scheduler.max_num_running_reqs,max_active=scheduler.max_num_active_reqs,
              chunk=scheduler.max_num_scheduled_tokens,
              long_prefill_token_threshold=scheduler.scheduler_config.long_prefill_token_threshold,
              async_scheduling=scheduler.scheduler_config.async_scheduling,
              enforce_eager=cfg.model_config.enforce_eager,
              cudagraph_mode=cfg.compilation_config.cudagraph_mode.name,
              speculative=cfg.speculative_config is not None,quantization=cfg.model_config.quantization,
              kv_dtype=cfg.cache_config.cache_dtype,mamba_cache_mode=cfg.cache_config.mamba_cache_mode,
              block_size=scheduler.block_size,num_gpu_blocks=scheduler.kv_cache_config.num_blocks,
              kv_groups=[dict(type=type(g.kv_cache_spec).__name__,spec=str(g.kv_cache_spec),layers=g.layer_names)
                         for g in scheduler.kv_cache_config.kv_cache_groups])
    target=Path(path);temporary=target.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(target)


def _verified(path, expected, label):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f'{label} source hash mismatch')
    return ast.parse(raw.decode())


def _scheduler(tree):
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Scheduler')


def extract_author_policy(scheduler, calibration):
    """Return importable stdlib-only policy code, including original defaults."""
    tree = _verified(scheduler, AUTHOR_SHA256, 'author')
    calibration_tree = _verified(calibration, CALIBRATION_SHA256, 'calibration')
    cls = _scheduler(tree)
    original_init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    # This is the whole original EB/EB+ initialization, after native engine setup.
    init = ast.parse('def _initialize_author_policy(self): pass').body[0]
    init.body = [copy.deepcopy(n) for n in original_init.body if n.lineno >= 234]
    if ast.unparse(init.body[0]).splitlines()[0] != "_mode_env = os.environ.get('VLLM_PD_SCHEDULER_MODE', '')":
        raise ValueError('author initialization boundary mismatch')
    policy = ast.ClassDef(name='AuthorPolicy', bases=[], keywords=[], decorator_list=[], body=[init])
    policy.body += [copy.deepcopy(n) for n in cls.body
                    if 616 <= n.lineno <= 2125 or 2431 <= n.lineno <= 2472
                    or 4127 <= n.lineno <= 4409]
    data = [copy.deepcopy(n) for n in calibration_tree.body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef))
            and n.name in ('HardwareParams', 'load_hardware_params')]
    header = '''# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Policy extracted from weifang231/eb-vllm; exact source pin in manifest.json.
from __future__ import annotations
import atexit
import json
import logging
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
logger = logging.getLogger(__name__)
_PD_THETA_HARD_MIN = 0.01
'''
    module = ast.fix_missing_locations(ast.Module(body=data + [policy], type_ignores=[]))
    return header + '\n' + ast.unparse(module) + '\n'


def _native_adapter(scheduler, native_scheduler):
    """Gate native eligibility scans, retaining the entire ownership graph."""
    native = _scheduler(_verified(native_scheduler, NATIVE_SHA256, 'native'))
    method = copy.deepcopy(next(n for n in native.body
                               if isinstance(n, ast.FunctionDef) and n.name == 'schedule'))
    method.name = '_schedule_exclusive_native'
    running_gate = ast.parse('''
if not phase_allows_running(self.pd_phase, request, self.pd_decoding_requests):
    req_index += 1
    continue
''').body[0]
    waiting_gate = ast.parse('''
target = self.pd_batch_size_N if self.pd_phase == 0 else self.pd_refill_target
if not phase_allows_admission(self.pd_phase, target, self.pd_prefilled_count,
                              self.requests, num_scheduled_tokens):
    break
''').body
    counts = {'running': 0, 'waiting': 0, 'prefill_allocation': 0, 'threshold': 0}

    class Gates(ast.NodeTransformer):
        def visit_While(self, node):
            self.generic_visit(node)
            if ast.unparse(node.test) == 'req_index < len(self.running) and token_budget > 0':
                if ast.unparse(node.body[0]) != 'request = self.running[req_index]':
                    raise ValueError('native running boundary mismatch')
                node.body.insert(1, running_gate)
                counts['running'] += 1
            elif ast.unparse(node.test) == 'token_budget > 0':
                node.body = waiting_gate + node.body
                counts['waiting'] += 1
            elif ast.unparse(node.test) == 'True' and any(
                    isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'new_blocks'
                    for n in node.body):
                # Original EB prefills never evict decoders to allocate a chunk.
                success = next(i for i, n in enumerate(node.body)
                               if isinstance(n, ast.If) and ast.unparse(n.test) == 'new_blocks is not None')
                node.body.insert(success + 1, ast.parse('if self.pd_phase in (0, 2): break').body[0])
                counts['prefill_allocation'] += 1
            return node

        def visit_Assign(self, node):
            if ast.unparse(node.targets[0]) == 'long_prefill_token_threshold':
                # Author applies the configured cap even to a single prefill.
                # The MB path calls untouched NativeScheduler.schedule instead.
                node.value = ast.parse('self.scheduler_config.long_prefill_token_threshold', mode='eval').body
                counts['threshold'] += 1
            return node

    method = ast.fix_missing_locations(Gates().visit(method))
    if counts != {'running': 1, 'waiting': 1, 'prefill_allocation': 1, 'threshold': 2}:
        raise ValueError(f'native boundary census mismatch: {counts}')
    author = _scheduler(_verified(scheduler, AUTHOR_SHA256, 'author'))
    update = next(n for n in author.body if isinstance(n, ast.FunctionDef) and n.name == 'update_from_output')
    stopped = next(n for n in ast.walk(update) if isinstance(n, ast.If) and ast.unparse(n.test) == 'stopped')
    completion = ast.parse('def _ebplus_record_completion(self, request): pass').body[0]
    completion.body = [copy.deepcopy(stopped.body[0])]
    bridge = ast.parse(_BRIDGE_CLASS).body[0]
    bridge.body.extend([method, completion])
    bridge = ast.fix_missing_locations(bridge)
    header = f'''# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Native scheduler body plus InferX eligibility/state boundary.
from __future__ import annotations
import hashlib
from pathlib import Path
import vllm.v1.core.sched.scheduler as _native
_NativeScheduler = _native.Scheduler
if hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest() != {NATIVE_SHA256!r}:
    raise RuntimeError('installed native scheduler hash mismatch')
globals().update({{k:v for k,v in vars(_native).items() if not k.startswith('__')}})
from inferx_ebplus_policy import AuthorPolicy
import time
import os
import json
'''
    return (header + '\n' + inspect.getsource(phase_allows_running) + '\n'
            + inspect.getsource(phase_allows_admission) + '\n' + inspect.getsource(record_native_runtime)
            + '\nclass ObservedNativeScheduler(_NativeScheduler):\n'
            + '    def __init__(self,*args,**kwargs):\n'
            + '        super().__init__(*args,**kwargs)\n'
            + '        record_native_runtime(self)\n\n' + ast.unparse(bridge) + '\n')


_BRIDGE_CLASS = '''
class EBPlusScheduler(AuthorPolicy, _NativeScheduler):
    def __init__(self, *args, **kwargs):
        self._ebplus_exclusive = False
        _NativeScheduler.__init__(self, *args, **kwargs)
        record_native_runtime(self)
        if self.max_num_running_reqs > 4 or self.max_num_active_reqs > 4:
            raise ValueError('EB+ validation requires a hard C4 ceiling')
        if self.vllm_config.max_concurrent_batches != 1:
            raise ValueError('qualify EB+ synchronous scheduling before overlapping batches')
        self._initialize_author_policy()
        if self.scheduler_mode not in ('auto', 'eb'):
            raise ValueError('EB+ validation requires explicit auto or eb mode')
        self._ebplus_execution_counts = {'mb': 0, 'prefill': 0, 'decode': 0}
        self._ebplus_evaluations = 0

    def _evaluate_mode_switch(self):
        eligible = (self.pd_param_initialized and self._avg_prompt_len + self.pd_avg_output_tokens > 0
                    and 0 < self.pd_k_ratio < 1)
        result = AuthorPolicy._evaluate_mode_switch(self)
        if eligible:
            self._ebplus_evaluations += 1
            self._ebplus_dump_evidence()
        return result

    def _ebplus_dump_evidence(self):
        path = os.environ.get('INFERX_EBPLUS_EVIDENCE')
        if not path:
            return
        state = dict(mode=self.scheduler_mode, active_scheduler=getattr(self,'_active_scheduler',self.scheduler_mode),
            evaluations=self._ebplus_evaluations, parameters_initialized=self.pd_param_initialized,
            mb_cost_profiled=getattr(self,'_mb_cost_profiled',False), execution_counts=self._ebplus_execution_counts,
            mode_switch_history=getattr(self,'_mode_switch_history',[]),
            completed=self.pd_total_completed + self.pd_batch_completed_count,
            max_num_running_reqs=self.max_num_running_reqs,
            block_size=self.block_size, num_gpu_blocks=self.kv_cache_config.num_blocks,
            kv_groups=[dict(type=type(g.kv_cache_spec).__name__,spec=str(g.kv_cache_spec),layers=g.layer_names)
                       for g in self.kv_cache_config.kv_cache_groups])
        target = Path(path)
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(state,indent=2)+'\\n')
        temporary.replace(target)

    def shutdown(self):
        self._ebplus_dump_evidence()
        return _NativeScheduler.shutdown(self)

    def schedule(self, throttle_prefills=False):
        if throttle_prefills:
            raise ValueError('DP prefill throttling is outside this single-GPU qualification')
        return AuthorPolicy.schedule(self)

    def _schedule_default(self):
        self._ebplus_execution_counts['mb'] += 1
        return _NativeScheduler.schedule(self)

    def schedule_pd(self):
        # Carry current native in-progress prefills across an MB->EB switch.
        self.chunk_prefilling = [r for r in self.running if self._is_prefill(r)]
        self._handle_phase_transition()
        self._ebplus_exclusive = True
        try:
            self._ebplus_execution_counts['decode' if self.pd_phase == 1 else 'prefill'] += 1
            return self._schedule_exclusive_native()
        finally:
            self._ebplus_exclusive = False

    def _compute_adaptive_N(self):
        # The author's large-server heuristic has a minimum of16. Respect C4.
        return min(self.max_num_running_reqs, AuthorPolicy._compute_adaptive_N(self))

    def _update_after_schedule(self, output):
        before = {}
        if self._ebplus_exclusive:
            before = {rid: self._is_prefill(self.requests[rid]) for rid in output.num_scheduled_tokens}
        _NativeScheduler._update_after_schedule(self, output)
        if self._ebplus_exclusive:
            for rid, was_prefill in before.items():
                request = self.requests[rid]
                if was_prefill and not self._is_prefill(request):
                    self.pd_prefilled_count += 1
                    self.pd_decoding_requests.add(rid)
            self.chunk_prefilling = [r for r in self.running if self._is_prefill(r)]

    def _preempt_request(self, request, timestamp, drop_stale_output=False):
        _NativeScheduler._preempt_request(self, request, timestamp, drop_stale_output)
        if self.use_pd_scheduler:
            self.pd_oom_event_count += 1
            self.pd_decoding_requests.discard(request.request_id)
        if request in self.chunk_prefilling:
            self.chunk_prefilling.remove(request)

    def _preempt_chunk_prefilling(self):
        count, tokens = 0, 0
        for request in list(self.chunk_prefilling):
            # Keep the reference owned until the native transition succeeds.
            # No temporary hiding, fabricated cache API or direct old free().
            if not self._request_blocks_can_be_freed(request):
                raise RuntimeError('cannot synchronously preempt in-flight prefill')
            computed_before = request.num_computed_tokens
            _NativeScheduler._preempt_request(self, request, time.monotonic())
            self.running.remove(request)
            self.pd_decoding_requests.discard(request.request_id)
            count += 1
            tokens += computed_before
        self.chunk_prefilling.clear()
        return count, tokens

    def _free_request(self, request):
        if request.status in (RequestStatus.FINISHED_STOPPED, RequestStatus.FINISHED_LENGTH_CAPPED,
                              RequestStatus.FINISHED_REPETITION):
            self._ebplus_record_completion(request)
        self.pd_decoding_requests.discard(request.request_id)
        if request in self.chunk_prefilling:
            self.chunk_prefilling.remove(request)
        return _NativeScheduler._free_request(self, request)
'''


def prepare_adapter(scheduler, calibration, native_scheduler, destination):
    """Never overwrite a prepared overlay or the original engine/source tree."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    policy = extract_author_policy(scheduler, calibration)
    adapter = _native_adapter(scheduler, native_scheduler)
    license_bytes = None
    for parent in Path(scheduler).resolve().parents:
        for name in ('author_LICENSE', 'LICENSE'):
            candidate = parent / name
            if candidate.is_file():
                raw = candidate.read_bytes()
                if hashlib.sha256(raw).hexdigest() == LICENSE_SHA256:
                    license_bytes = raw
                    break
        if license_bytes is not None:
            break
    if license_bytes is None:
        raise ValueError('pinned author license missing')
    compile(policy, 'inferx_ebplus_policy.py', 'exec')
    compile(adapter, 'inferx_ebplus_scheduler.py', 'exec')
    destination.mkdir(parents=True)
    (destination / 'inferx_ebplus_policy.py').write_text(policy)
    (destination / 'inferx_ebplus_scheduler.py').write_text(adapter)
    (destination / 'LICENSE').write_bytes(license_bytes)
    manifest = {'author_commit': AUTHOR_COMMIT,
                'author_sha256': AUTHOR_SHA256,
                'calibration_sha256': CALIBRATION_SHA256,
                'native_sha256': NATIVE_SHA256,
                'adapted_engine_boundaries': [
                    'schedule_pd', '_schedule_default', '_update_after_schedule',
                    '_free_request', '_preempt_request', '_preempt_chunk_prefilling',
                    '_compute_adaptive_N', 'schedule throttle guard',
                    '_evaluate_mode_switch observational counter', 'shutdown evidence snapshot',
                ],
                'extracted_method_sha256': {
                    n.name: hashlib.sha256(ast.get_source_segment(Path(scheduler).read_text(), n).encode()).hexdigest()
                    for n in _scheduler(_verified(scheduler, AUTHOR_SHA256, 'author')).body
                    if isinstance(n, ast.FunctionDef)
                    and (619 <= n.lineno <= 2125 or 2431 <= n.lineno <= 2472
                         or 4127 <= n.lineno <= 4409)
                },
                'files': {'inferx_ebplus_policy.py': hashlib.sha256(policy.encode()).hexdigest(),
                          'inferx_ebplus_scheduler.py': hashlib.sha256(adapter.encode()).hexdigest(),
                          'LICENSE': LICENSE_SHA256},
                'qualification': 'source-bound adapter; actual current-stack CPU/GPU gates pending'}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest
