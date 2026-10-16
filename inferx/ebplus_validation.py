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
    counts = {'running': 0, 'waiting': 0, 'prefill_allocation': 0,
              'waiting_allocation': 0, 'threshold': 0}

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
                refused = [n for n in node.body if isinstance(n, ast.If)
                           and ast.unparse(n.test) == 'new_blocks is None']
                if len(refused) != 1:
                    raise ValueError('native waiting allocation-refusal boundary mismatch')
                refused[0].body.insert(0, ast.parse(
                    'self._ebplus_note_admission_block(request)').body[0])
                counts['waiting_allocation'] += 1
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
    if counts != {'running': 1, 'waiting': 1, 'prefill_allocation': 1,
                  'waiting_allocation': 1, 'threshold': 2}:
        raise ValueError(f'native boundary census mismatch: {counts}')
    author = _scheduler(_verified(scheduler, AUTHOR_SHA256, 'author'))
    # Preserve all author branches. Modern full-sequence admission can refuse a
    # request while raw free-block fraction is high, so feed that actual native
    # resource refusal into the same existing memory-pressure escape predicate.
    phase = copy.deepcopy(next(n for n in author.body if isinstance(n, ast.FunctionDef)
                               and n.name == '_handle_phase_transition'))
    phase_start = next(i for i, n in enumerate(phase.body) if isinstance(n, ast.If)
                       and ast.unparse(n.test) == 'self.pd_phase == 0')
    phase.body[phase_start:phase_start] = ast.parse('''
native_admission_pressure = self._ebplus_admission_pressure()
kv_cache_full = kv_cache_full or native_admission_pressure
''').body
    phase.body.extend(ast.parse('''
if native_admission_pressure and prev_phase in (0, 2) and self.pd_phase == 1:
    self._ebplus_admission_escapes += 1
''').body)
    update = next(n for n in author.body if isinstance(n, ast.FunctionDef) and n.name == 'update_from_output')
    stopped = next(n for n in ast.walk(update) if isinstance(n, ast.If) and ast.unparse(n.test) == 'stopped')
    completion = ast.parse('def _ebplus_record_completion(self, request): pass').body[0]
    completion.body = [copy.deepcopy(stopped.body[0])]
    bridge = ast.parse(_BRIDGE_CLASS).body[0]
    bridge.body.extend([method, completion, phase])
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
        self._last_evaluation_inputs = None
        self._ebplus_admission_blocked = None
        self._ebplus_admission_failures = 0
        self._ebplus_admission_escapes = 0
        self._ebplus_last_admission_failure = None
        self._ebplus_empty_schedule_streak = 0
        self._ebplus_empty_snapshot = None

    def _ebplus_note_admission_block(self, request):
        free = self.kv_cache_manager.block_pool.get_num_free_blocks()
        self._ebplus_admission_blocked = (request.request_id, free)
        self._ebplus_admission_failures += 1
        self._ebplus_last_admission_failure = dict(request_id=request.request_id,
            prompt_tokens=request.num_prompt_tokens,free_blocks=free,phase=self.pd_phase,
            full_sequence_must_fit=self.scheduler_reserve_full_isl,
            decoders=len(self.pd_decoding_requests),running=len(self.running))

    def _ebplus_admission_pressure(self):
        blocked = self._ebplus_admission_blocked
        if blocked is None:
            return False
        request_id, failed_free = blocked
        queue = self.kv_holding_waiting or self.waiting
        free = self.kv_cache_manager.block_pool.get_num_free_blocks()
        # Retry when native capacity or queue head changes. While neither does,
        # decode must release capacity instead of immediately retrying refill.
        if (not self.pd_decoding_requests or not queue
                or queue.peek_request().request_id != request_id or free > failed_free):
            self._ebplus_admission_blocked = None
            return False
        return True

    def _evaluate_mode_switch(self):
        eligible = (self.pd_param_initialized and self._avg_prompt_len + self.pd_avg_output_tokens > 0
                    and 0 < self.pd_k_ratio < 1)
        inputs = None
        if eligible:
            inputs = dict(mu_L=self._avg_prompt_len,mu_o=self.pd_avg_output_tokens,
                N_obs=max(1.,self._n_obs),theta0=self.pd_k_ratio,alpha_mb=self._alpha_mb,
                alpha_p=self.pd_alpha_p,beta_p=self.pd_beta_p,alpha_d=self.pd_alpha_d,beta_d=self.pd_beta_d,
                mb_cost_a=self._mb_cost_a,mb_cost_b=self._mb_cost_b,mb_cost_c=self._mb_cost_c,
                delta=self._mode_switch_delta,cooldown_before=self._mode_cooldown,
                active_before=self._active_scheduler)
        result = AuthorPolicy._evaluate_mode_switch(self)
        if eligible:
            self._ebplus_evaluations += 1
            self._last_evaluation_inputs = inputs
            self._ebplus_dump_evidence()
        return result

    def _ebplus_dump_evidence(self):
        path = os.environ.get('INFERX_EBPLUS_EVIDENCE')
        if not path:
            return
        # The author's cold path increments total before evaluating and only
        # clears the batch afterwards. Count that in-flight batch once.
        pending = self.pd_batch_completed_count
        completed = self.pd_total_completed + (pending if pending < self.pd_param_update_interval else 0)
        state = dict(mode=self.scheduler_mode, active_scheduler=getattr(self,'_active_scheduler',self.scheduler_mode),
            evaluations=self._ebplus_evaluations, parameters_initialized=self.pd_param_initialized,
            mb_cost_profiled=getattr(self,'_mb_cost_profiled',False), execution_counts=self._ebplus_execution_counts,
            mode_switch_history=getattr(self,'_mode_switch_history',[]),
            completed=completed,last_evaluation_inputs=self._last_evaluation_inputs,
            native_admission_failures=self._ebplus_admission_failures,
            native_admission_escapes=self._ebplus_admission_escapes,
            last_native_admission_failure=self._ebplus_last_admission_failure,
            zero_schedule_streak=self._ebplus_empty_schedule_streak,
            scheduler_snapshot=self._ebplus_empty_snapshot,
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
            output = self._schedule_exclusive_native()
            self._ebplus_empty_schedule_streak = (self._ebplus_empty_schedule_streak + 1
                if output.total_num_scheduled_tokens == 0 else 0)
            if self._ebplus_empty_schedule_streak == 64 and self.pd_decoding_requests:
                self._ebplus_empty_snapshot = self._ebplus_scheduler_snapshot()
                self._ebplus_dump_evidence()
            return output
        finally:
            self._ebplus_exclusive = False

    def _ebplus_scheduler_snapshot(self):
        # Observe ordinary fields only: reading a grammar future/property can
        # itself advance readiness and would change the state being diagnosed.
        return dict(phase=self.pd_phase,current_step=self.current_step,
            N=self.pd_batch_size_N,k=self.pd_switch_threshold_k,
            refill_target=self.pd_refill_target,prefilled=self.pd_prefilled_count,
            free_blocks=self.kv_cache_manager.block_pool.get_num_free_blocks(),
            running=[r.request_id for r in self.running],
            waiting=[r.request_id for r in self.waiting],
            kv_holding_waiting=[r.request_id for r in self.kv_holding_waiting],
            deferred_waiting=[r.request_id for r in self.deferred_waiting],
            decoding=sorted(self.pd_decoding_requests),
            inflight_prefills=[r.request_id for r in self._inflight_prefills],
            requests={rid:dict(status=r.status.name,prompt_tokens=r.num_prompt_tokens,
                computed_tokens=r.num_computed_tokens,output_tokens=len(r.output_token_ids),
                next_decode_eligible_step=r.next_decode_eligible_step,
                output_placeholders=r.num_output_placeholders)
                for rid,r in self.requests.items()})

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
                'overlay_directory_name': destination.name,
                'author_sha256': AUTHOR_SHA256,
                'calibration_sha256': CALIBRATION_SHA256,
                'native_sha256': NATIVE_SHA256,
                'adapted_engine_boundaries': [
                    'schedule_pd', '_schedule_default', '_update_after_schedule',
                    '_free_request', '_preempt_request', '_preempt_chunk_prefilling',
                    '_compute_adaptive_N', 'schedule throttle guard',
                    '_handle_phase_transition native admission-pressure predicate',
                    'waiting allocation-refusal observation',
                    'sustained-empty-schedule state snapshot',
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
