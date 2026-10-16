# SPDX-License-Identifier: Apache-2.0
# Native vLLM boundary; see LICENSE.native.
from __future__ import annotations
import hashlib, json, math, os, time, heapq
from pathlib import Path
from types import SimpleNamespace
import vllm.v1.core.sched.scheduler as _native
_NativeScheduler = _native.Scheduler
_site = Path(_native.__file__).parents[4]
for _name, _sha in {'vllm/v1/core/sched/scheduler.py': 'f820bee4f42b878b262a30f3728a97d7413477b2a277748e20f2528ac2ec2f2a', 'vllm/v1/core/sched/request_queue.py': 'db8d25daf9580d581f18c6b631a6576dfaa7d1474398cd8dc4db1d29745c214d', 'vllm/v1/core/kv_cache_manager.py': 'bd9bf46a2101f9368d4941867415a239d18d95e0b956f82f885ce25e10e69f3b', 'vllm/v1/core/kv_cache_coordinator.py': '6a1e00227b981a7bb6108a42db0e446c716f789274f40c63eb9a6ce4d699ddcb', 'vllm/v1/core/single_type_kv_cache_manager.py': '3a3cb20bd6aa1e0ad46471fc9e5e01ee5797ab4bb759e00ef5fb309d2830db13', 'vllm/v1/core/block_pool.py': '05c73052934a9d4664ccd5df562ce71011278d9a511932da102ef2efff00e934', 'vllm/v1/core/kv_cache_utils.py': '1b20c836aece4982dd87f7e06e14f5ae07a1fc2636101074cfad11584b1a0dcf', 'vllm/v1/request.py': 'dd7c30bf8b0894dc123a739c299c0133566f8cc075b081e45905f1c29506e69d', 'vllm/v1/outputs.py': 'a4c73e7125781c339d42bdb7f8fdd3324a489742a4588d8c1b1638aa335fd138', 'vllm/v1/core/sched/output.py': '972e0bc8e7af1f4262ac73f660f6b8f798ba780f86c031a9c8a7f10cbfaad3a8', 'vllm/v1/engine/core.py': 'f116962c35d9d312b20b0b76c34a8cdaa6d3f1d31f2036db4e123957d2d1555b', 'vllm/config/scheduler.py': '82bc6805ecf30a9b3d1b678d3270dad919abed011782ad16f1e805a78aa6c241'}.items():
    if hashlib.sha256((_site / _name).read_bytes()).hexdigest() != _sha:
        raise RuntimeError('installed native source hash mismatch: ' + _name)
globals().update({k: v for k, v in vars(_native).items() if not k.startswith('__')})
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from inferx_qoserve_author_policy import AuthorPolicy
from inferx.qoserve_profile import snapshot_step_geometry, validate_step_row, TIMING_BOUNDARY, proposed_profile_budget

def parse_contract(extra_args):
    """Validate flat vllm_xargs; only untagged code checks receive defaults."""
    tags = {key: value for key, value in (extra_args or {}).items() if key.startswith('qoserve_')}
    if not tags:
        return dict(tier=0, deadline_s=6.0, tbt_s=0.1, trace_id=None)
    required = {'qoserve_tier', 'qoserve_deadline_s', 'qoserve_tbt_s'}
    if not required <= tags.keys() or tags.keys() - required - {'qoserve_trace_id'}:
        raise ValueError('incomplete or unknown QoServe contract fields')
    tier = tags['qoserve_tier']
    if type(tier) is not int or tier not in (0, 1):
        raise ValueError('QoServe tier must be integer 0 or 1')
    values = []
    for key in ('qoserve_deadline_s', 'qoserve_tbt_s'):
        value = tags[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(value)) or (value <= 0):
            raise ValueError('QoServe deadlines and TBT must be positive finite seconds')
        values.append(float(value))
    trace = tags.get('qoserve_trace_id')
    if 'qoserve_trace_id' in tags and (not isinstance(trace, str) or not trace):
        raise ValueError('QoServe trace ID must be a nonempty string')
    return dict(tier=tier, deadline_s=values[0], tbt_s=values[1], trace_id=trace)

def sequence_view_type(policy):
    """Map real fields to the extracted author methods, retaining request age.

    The view owns policy metadata only. It never owns, allocates or frees KV.
    Its inherited comparison/deadline methods are the pinned author's code.
    """

    class NativeSequenceView(policy.Sequence):

        def __init__(self, request, *, wall_to_monotonic):
            self.request = request
            self.contract = parse_contract(request.sampling_params.extra_args)
            self.seq_id = request.request_id
            self.arrival_time = request.arrival_time + wall_to_monotonic
            self.state = SimpleNamespace(_arrived_at=self.arrival_time, e2e_prefill_time=0.0)
            self.scheduler_type = policy.scheduler_type
            self.hybrid_prioritization_param = policy.hybrid_prioritization_param
            self.drop = 0
            self.cached_prefix_tokens = 0
            self.last_slack = 0.0
            self.tier = self.contract['tier']
            self.tier_deadline = self.contract['deadline_s']
            self.ttft_deadline = self.contract['deadline_s']
            self.num_decode_tokens = -1
            context = SimpleNamespace(output_len_pred=policy.output_len_pred, execution_threshold_batched=self.contract['tbt_s'])
            policy._qoserve_ttft_budget(context, self)

        @property
        def output_token_ids(self):
            return self.request.output_token_ids

        @property
        def mem_allocated(self):
            return self.request.num_computed_tokens > 0

        @property
        def prompt_processing_finished(self):
            return self.request.num_computed_tokens >= self.get_prompt_len()

        def get_prompt_len(self):
            return max(self.request.num_prompt_tokens, self.request.num_tokens - 1)

        def get_num_prompt_tokens_processed(self):
            return min(max(self.request.num_computed_tokens, self.cached_prefix_tokens), self.get_prompt_len())

        def _get_context_size(self):
            return self.request.num_tokens if self.prompt_processing_finished else self.get_num_prompt_tokens_processed()
    return NativeSequenceView

def record_native_runtime(scheduler):
    """Read actual initialized allocator/configuration once, before measurement."""
    import os
    path = os.environ.get('INFERX_VLLM_RUNTIME')
    if not path:
        return
    cfg = scheduler.vllm_config
    data = dict(model=cfg.model_config.model, context_window=scheduler.max_model_len, concurrency=scheduler.max_num_running_reqs, max_active=scheduler.max_num_active_reqs, chunk=scheduler.max_num_scheduled_tokens, long_prefill_token_threshold=scheduler.scheduler_config.long_prefill_token_threshold, scheduler_reserve_full_isl=scheduler.scheduler_reserve_full_isl, async_scheduling=scheduler.scheduler_config.async_scheduling, enforce_eager=cfg.model_config.enforce_eager, cudagraph_mode=cfg.compilation_config.cudagraph_mode.name, speculative=cfg.speculative_config is not None, quantization=cfg.model_config.quantization, kv_dtype=cfg.cache_config.cache_dtype, mamba_cache_mode=cfg.cache_config.mamba_cache_mode, block_size=scheduler.block_size, num_gpu_blocks=scheduler.kv_cache_config.num_blocks, kv_groups=[dict(type=type(g.kv_cache_spec).__name__, spec=str(g.kv_cache_spec), layers=g.layer_names) for g in scheduler.kv_cache_config.kv_cache_groups])
    target = Path(path)
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(target)

def _record_qoserve_runtime(scheduler):
    record_native_runtime(scheduler)
    path = os.environ.get('INFERX_VLLM_RUNTIME')
    if not path:
        return
    target = Path(path)
    data = json.loads(target.read_text())
    data.update(scheduler_class=type(scheduler).__name__, full_input_commitments=isinstance(scheduler, CompatibleNativeScheduler), qoserve_policy_enabled=scheduler._qoserve_policy_enabled, profiling_mode=getattr(scheduler, '_qoserve_profiling_mode', 'off'), wall_to_monotonic_offset=scheduler._qoserve_clock_offset)
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(target)

class ObservedNativeScheduler(_NativeScheduler):
    _qoserve_policy_enabled = False

    def __init__(self, *args, **kwargs):
        _NativeScheduler.__init__(self, *args, **kwargs)
        if self.max_num_running_reqs > 4 or self.max_num_active_reqs > 4:
            raise ValueError('QoServe qualification requires hard C4')
        if self.vllm_config.max_concurrent_batches != 1 or self.scheduler_config.async_scheduling or self.vllm_config.speculative_config is not None:
            raise ValueError('QoServe qualification requires synchronous MTP0')
        if self.connector is not None or self.ec_connector is not None:
            raise ValueError('external connectors are outside this native qualification')
        self._qoserve_clock_offset = time.monotonic() - time.time()
        self._qoserve_view_type = sequence_view_type(AuthorPolicy)
        self._qoserve_views = {}
        self._qoserve_records = {}
        self._qoserve_trace_ids = set()
        self._qoserve_decisions = []
        self._qoserve_quotas = {}
        _record_qoserve_runtime(self)

    def add_request(self, request):
        if request.request_id in self.requests:
            raise ValueError('duplicate native request ID')
        view = self._qoserve_view_type(request, wall_to_monotonic=self._qoserve_clock_offset)
        trace = view.contract['trace_id']
        if trace is not None and trace in self._qoserve_trace_ids:
            raise ValueError('duplicate QoServe trace ID')
        _NativeScheduler.add_request(self, request)
        self._qoserve_views[request.request_id] = view
        self._qoserve_records[request.request_id] = dict(request_id=request.request_id, trace_id=trace, contract=view.contract, arrival_monotonic=view.arrival_time, model_token_times_s=[], finish_status=None)
        if trace is not None:
            self._qoserve_trace_ids.add(trace)

    def update_from_output(self, scheduler_output, model_runner_output):
        before = {rid: self._qoserve_views[rid] for rid in scheduler_output.num_scheduled_tokens if rid in self._qoserve_views}
        result = _NativeScheduler.update_from_output(self, scheduler_output, model_runner_output)
        native_update_end = time.perf_counter()
        now = time.monotonic()
        for batch in result.values():
            for output in batch.outputs:
                if not output.new_token_ids:
                    continue
                if len(output.new_token_ids) != 1:
                    raise RuntimeError('unqualified multiple model tokens in synchronous MTP0')
                view = before[output.request_id]
                record = self._qoserve_records[output.request_id]
                if not record['model_token_times_s']:
                    view.state.e2e_prefill_time = now - view.arrival_time
                record['model_token_times_s'].append(now - view.arrival_time)
        profile_callback = getattr(self, '_qoserve_after_native_update', None)
        if profile_callback is not None:
            profile_callback(scheduler_output, native_update_end)
        return result

    def _free_request(self, request, *args, **kwargs):
        result = _NativeScheduler._free_request(self, request, *args, **kwargs)
        if request.request_id in self._qoserve_records:
            self._qoserve_records[request.request_id]['finish_status'] = request.status.name
        self._qoserve_views.pop(request.request_id, None)
        return result

    def shutdown(self):
        path = os.environ.get('INFERX_QOSERVE_EVIDENCE')
        if path:
            target = Path(path)
            tmp = target.with_suffix('.tmp')
            tmp.write_text(json.dumps(dict(scheduler_class=type(self).__name__, requests=self._qoserve_records, decisions=self._qoserve_decisions, remaining_views=sorted(self._qoserve_views)), indent=2) + '\n')
            tmp.replace(target)
        return _NativeScheduler.shutdown(self)

class ProfilingNativeScheduler(ObservedNativeScheduler):

    def __init__(self, *args, **kwargs):
        self._qoserve_profiling_mode = 'off'
        self._profile_config = None
        self._profile_pending = None
        self._profile_rows = []
        self._profile_events = []
        super().__init__(*args, **kwargs)
        if self.scheduler_reserve_full_isl:
            raise ValueError('calibration profile requires official admission checkFalse')
        if self.scheduler_config.long_prefill_token_threshold != 0:
            raise ValueError('calibration controls total budget with native long-prefill cap0')

    def begin_profile(self, config):
        if self._profile_config is not None or self._profile_pending is not None:
            raise ValueError('finish the previous profiling rollout first')
        cfg = json.loads(json.dumps(config, allow_nan=False))
        if cfg['phase'] not in ('train', 'holdout'):
            raise ValueError('invalid profiling phase')
        for key in ('engine_id', 'rollout_id', 'cache_salt', 'long_request_id'):
            if not isinstance(cfg.get(key), str) or not cfg[key]:
                raise ValueError('missing profiling identity')
        if cfg['long_request_id'] not in self.requests or self.requests[cfg['long_request_id']].cache_salt != cfg['cache_salt']:
            raise ValueError('profile must bind an actual salted native request')
        if type(cfg.get('prefill_target')) is not int or not 1 <= cfg['prefill_target'] <= 4096:
            raise ValueError('invalid profiling prefill target')
        if type(cfg.get('warmup_steps')) is not int or cfg['warmup_steps'] < 0:
            raise ValueError('invalid profiling warmup count')
        if not cfg.get('candidates') or any((type(n) is not int or not 1 <= n <= 4096 for n in cfg['candidates'])):
            raise ValueError('invalid profiling candidate list')
        proposed_profile_budget(cfg, 0, 0)
        self._profile_config = cfg
        self._profile_rows = []
        self._profile_events = []
        self._qoserve_profiling_mode = cfg['phase']
        _record_qoserve_runtime(self)

    def end_profile(self):
        if self._profile_pending is not None:
            raise ValueError('model step has no matching native update')
        result = dict(rows=self._profile_rows, events=self._profile_events)
        self._profile_config = None
        self._qoserve_profiling_mode = 'off'
        self._profile_rows = []
        self._profile_events = []
        _record_qoserve_runtime(self)
        return result

    def schedule(self, throttle_prefills=False):
        started = time.perf_counter()
        cfg = self._profile_config
        if cfg is None:
            return _NativeScheduler.schedule(self, throttle_prefills)
        if self._profile_pending is not None:
            raise ValueError('overlapping profiling steps are unqualified')
        if throttle_prefills:
            raise ValueError('DP profiling is outside this single GPU protocol')
        metadata = {rid: dict(num_prompt_tokens=r.num_prompt_tokens, num_tokens=r.num_tokens, output_tokens_before=len(r.output_token_ids)) for rid, r in self.requests.items()}
        is_decode = lambda r: r.num_computed_tokens >= max(r.num_prompt_tokens, r.num_tokens - 1)
        decoders = sum((is_decode(r) for r in self.running))
        self.running.sort(key=lambda r: not is_decode(r))
        ceiling = self.max_num_scheduled_tokens
        proposed = proposed_profile_budget(cfg, decoders, len(self._profile_rows), ceiling)
        try:
            self.max_num_scheduled_tokens = min(ceiling, proposed)
            output = _NativeScheduler.schedule(self)
        finally:
            self.max_num_scheduled_tokens = ceiling
        if output.total_num_scheduled_tokens == 0:
            self._profile_events.append(dict(step=self.current_step, reason='empty-native-step'))
        else:
            self._profile_pending = dict(output=output, metadata=metadata, started=started, proposed=proposed, step=self.current_step)
        return output

    def _qoserve_after_native_update(self, output, ended):
        pending = self._profile_pending
        if pending is None:
            return
        if pending['output'] is not output:
            raise ValueError('profiling update paired with different schedule')
        cfg = self._profile_config
        requests, features = snapshot_step_geometry(output, pending['metadata'])
        row = dict(row_id=cfg['engine_id'] + '/' + cfg['rollout_id'] + '/' + str(pending['step']), phase=cfg['phase'], engine_id=cfg['engine_id'], rollout_id=cfg['rollout_id'], cache_salt=cfg['cache_salt'], step_index=pending['step'], warmup=len(self._profile_rows) < cfg['warmup_steps'], identity=cfg['identity'], budget_mode=cfg.get('budget_mode', 'prefill-target' if cfg['phase'] == 'train' else 'total-cycle'), calibration_kind=cfg.get('calibration_kind', cfg['phase']), holdout_protocol=cfg.get('holdout_protocol', 'original-cycle-v1'), proposed_total_tokens=pending['proposed'], requests=requests, features=features, preempted_request_ids=sorted(output.preempted_req_ids), timing=dict(start_s=pending['started'], end_s=ended, batch_execution_time=ended - pending['started'], boundary=TIMING_BOUNDARY, completed_model_step=True))
        validate_step_row(row, cfg['identity'])
        self._profile_rows.append(row)
        self._profile_pending = None

class CompatibleNativeScheduler(ObservedNativeScheduler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.scheduler_reserve_full_isl:
            raise ValueError('full-input commitments require native full-sequence admission check')

    def _qoserve_owners(self):
        running = {r.request_id for r in self.running}
        return [r for rid, r in self.requests.items() if not r.is_finished() and (rid in running or self._holds_kv_blocks(r))]

    def _qoserve_other_commitments(self, request):
        excluded = getattr(request, 'request_id', None)
        return sum((self._request_remaining_blocks(r) for r in self._qoserve_owners() if r.request_id != excluded))

    def _qoserve_effective_free_blocks(self):
        return self.kv_cache_manager.block_pool.get_num_free_blocks() - self._qoserve_other_commitments(None)

    def _qoserve_can_admit(self, request):
        if request in self._qoserve_owners():
            return True
        if len(self._qoserve_owners()) + self.num_waiting_for_streaming_input >= self.max_num_active_reqs:
            return False
        if request.num_computed_tokens == 0:
            blocks, local, _, _ = self._get_local_prefix_cache_hit(request)
        else:
            blocks = self.kv_cache_manager.empty_kv_cache_blocks
            local = 0
        view = self._qoserve_views[request.request_id]
        view.cached_prefix_tokens = local
        computed = request.num_computed_tokens + local
        full = min(request.num_tokens, self.max_model_len)
        tables = [manager.req_to_blocks for manager in self.kv_cache_manager.coordinator.single_type_managers]
        existed = [request.request_id in table for table in tables]
        try:
            need = self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(request_id=request.request_id, num_tokens=full, new_computed_blocks=blocks.blocks, num_encoder_tokens=0, total_computed_tokens=computed, num_local_computed_tokens=computed, num_tokens_main_model=full, apply_admission_cap=True, prefill_end=max(request.num_prompt_tokens, request.num_tokens - 1))
        finally:
            for table, present in zip(tables, existed):
                if not present and request.request_id in table:
                    if table[request.request_id]:
                        raise RuntimeError('capacity forecast unexpectedly acquired native blocks')
                    del table[request.request_id]
        watermark = self.kv_cache_manager.watermark_blocks if self.running else 0
        return need + self._qoserve_other_commitments(request) + watermark <= self.kv_cache_manager.block_pool.get_num_free_blocks()

    def _qoserve_quota(self, request):
        return self._qoserve_quotas.get(request.request_id, 0) if self._qoserve_policy_enabled else self.max_num_scheduled_tokens

    def schedule(self, throttle_prefills=False):
        if throttle_prefills:
            raise ValueError('DP prefill throttling is outside qualification')
        return self._qoserve_schedule_native()

    def _qoserve_schedule_native(self, throttle_prefills: bool=False) -> SchedulerOutput:
        self.current_step += 1
        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []
        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        spec = self.vllm_config.speculative_config
        draft_slots = spec.max_num_new_slots_for_drafting if spec is not None else 0
        input_budget = self.scheduler_config.max_num_batched_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            token_budget = 0
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        prefill_scheduled = False
        has_sync_kv_loads = False
        scheduled_timestamp = time.monotonic()
        self.kv_cache_manager.new_step_starts()
        defer_prefills = (throttle_prefills and (not self.prefill_capacity_bound)) and any((not r.is_prefill_chunk for r in self.running))
        num_running, num_waiting = self.get_request_counts()
        num_eligible_reqs = num_running + num_waiting
        long_prefill_token_threshold = self.scheduler_config.long_prefill_token_threshold if num_eligible_reqs > 1 else 0
        if long_prefill_token_threshold > 0 and self.adaptive_long_prefill_threshold:
            long_prefill_token_threshold = max(long_prefill_token_threshold, input_budget // num_eligible_reqs)
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
            if self._qoserve_policy_enabled and self._qoserve_quota(request) == 0:
                req_index += 1
                continue
            if input_budget <= draft_slots:
                break
            if request.num_output_placeholders > 0 and request.num_computed_tokens + 2 - request.num_output_placeholders >= request.num_prompt_tokens + request.max_tokens:
                req_index += 1
                continue
            if self.current_step < request.next_decode_eligible_step:
                req_index += 1
                continue
            if defer_prefills and request.is_prefill_chunk:
                req_index += 1
                continue
            if self.ec_connector is not None and request.mm_features and (not self.ec_connector.ensure_cache_available(request, request.num_computed_tokens - request.num_output_placeholders)):
                req_index += 1
                continue
            num_new_tokens = request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens
            if 0 < long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget, input_budget - draft_slots, self._qoserve_quota(request))
            num_new_tokens = min(num_new_tokens, self.max_model_len - request.num_computed_tokens - self.num_sampled_tokens_per_step)
            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(request, num_new_tokens)
            encoder_inputs_to_schedule = None
            external_load_encoder_input: list[int] = []
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                encoder_inputs_to_schedule, num_new_tokens, new_encoder_compute_budget, external_load_encoder_input = self._try_schedule_encoder_inputs(request, request.num_computed_tokens, num_new_tokens, encoder_compute_budget, shift_computed_tokens=self.num_prefill_lookahead)
            num_new_tokens = self._reserve_prefill_lookahead(request, request.num_computed_tokens, num_new_tokens)
            if num_new_tokens == 0:
                req_index += 1
                continue
            with record_function_or_nullcontext('schedule: allocate_slots'):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, num_lookahead_tokens=self.num_lookahead_tokens, reserved_blocks=self._qoserve_other_commitments(request))
                    if new_blocks is not None:
                        break
                    break
                    if self.connector is not None and self.connector.has_pending_block_frees():
                        break
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
                    else:
                        preempted_req = self.running[-1]
                    if not self._request_blocks_can_be_freed(preempted_req):
                        break
                    if self.policy == SchedulingPolicy.PRIORITY:
                        victim_index = self.running.index(preempted_req)
                        del self.running[victim_index]
                        if victim_index < req_index:
                            req_index -= 1
                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            restored = num_scheduled_tokens.pop(preempted_req_id)
                            token_budget += restored
                            input_budget += restored + draft_slots
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(preempted_req_id, None)
                            if preempted_encoder_inputs:
                                num_embeds_to_restore = sum((preempted_req.get_num_encoder_embeds(i) for i in preempted_encoder_inputs))
                                encoder_compute_budget += num_embeds_to_restore
                    else:
                        preempted_req = self.running.pop()
                    self._preempt_request(preempted_req, scheduled_timestamp, drop_stale_output=self.requires_kv_delivery)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        break
            if new_blocks is None:
                break
            scheduled_running_reqs.append(request)
            prefill_scheduled |= request.is_prefill_chunk
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            input_budget -= num_new_tokens + draft_slots
            req_index += 1
            if request.spec_token_ids:
                num_scheduled_spec_tokens = num_new_tokens + request.num_computed_tokens - request.num_tokens - request.num_output_placeholders
                if num_scheduled_spec_tokens > 0:
                    spec_token_ids = request.spec_token_ids
                    if len(spec_token_ids) > num_scheduled_spec_tokens:
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids
                request.spec_token_ids = []
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)
                encoder_compute_budget = new_encoder_compute_budget
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set((req.lora_request.lora_int_id for req in scheduled_running_reqs if req.lora_request and req.lora_request.lora_int_id > 0))
            assert len(scheduled_loras) <= self.lora_config.max_loras
        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
            step_skipped_waiting: deque[Request] = deque()
            step_skipped_kv_holding: deque[Request] = deque()

            def skip_request(from_queue: RequestQueue) -> None:
                request_to_skip = from_queue.pop_request()
                self.deferred_waiting.add(request_to_skip)
                if self._holds_kv_blocks(request_to_skip):
                    step_skipped_kv_holding.appendleft(request_to_skip)
                else:
                    step_skipped_waiting.appendleft(request_to_skip)
            while token_budget > 0:
                request_queue = self.kv_holding_waiting or self.waiting
                if not request_queue or input_budget <= draft_slots:
                    break
                num_running = len(self.running) + self.num_waiting_for_streaming_input
                if num_running >= self.max_num_active_reqs:
                    break
                request = request_queue.peek_request()
                request_id = request.request_id
                ready_to_schedule = self._handle_blocked_waiting_request(request)
                if not ready_to_schedule:
                    skip_request(request_queue)
                    continue
                if self._qoserve_policy_enabled and self._qoserve_quota(request) == 0:
                    skip_request(request_queue)
                    continue
                if request.num_stale_output_tokens and (not request.drop_stale_output):
                    skip_request(request_queue)
                    continue
                if self.lora_config and request.lora_request and (len(scheduled_loras) == self.lora_config.max_loras and request.lora_request.lora_int_id not in scheduled_loras):
                    skip_request(request_queue)
                    continue
                num_external_computed_tokens = 0
                load_kv_async = False
                connector_prefix_cache_queries, connector_prefix_cache_hits = (0, 0)
                did_prefix_cache_lookup = False
                if request.num_computed_tokens == 0:
                    did_prefix_cache_lookup = True
                    new_computed_blocks, num_new_local_computed_tokens, request.shared_prefix_boundary, hit_diverged = self._get_local_prefix_cache_hit(request)
                    if self.connector is not None:
                        partial_tail = num_new_local_computed_tokens % self.block_size
                        block_aligned_local = num_new_local_computed_tokens - partial_tail
                        ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(request, block_aligned_local)
                        if request.skip_reading_prefix_cache:
                            ext_tokens, load_kv_async = (0, False)
                        if ext_tokens is None:
                            skip_request(request_queue)
                            continue
                        if self.prefix_replay_tokens:
                            ext_tokens -= ext_tokens % self.block_size
                            load_kv_async = load_kv_async and ext_tokens > 0
                        if partial_tail and ext_tokens > partial_tail:
                            new_computed_blocks = self.kv_cache_manager.truncate_computed_blocks(new_computed_blocks, block_aligned_local)
                            num_new_local_computed_tokens = block_aligned_local
                            num_external_computed_tokens = ext_tokens
                        elif partial_tail:
                            num_external_computed_tokens = 0
                            load_kv_async = False
                        else:
                            num_external_computed_tokens = ext_tokens
                        if hit_diverged and num_external_computed_tokens == 0:
                            new_computed_blocks, num_new_local_computed_tokens, request.shared_prefix_boundary = self.kv_cache_manager.get_computed_blocks(request)
                        connector_prefix_cache_queries = request.num_tokens - num_new_local_computed_tokens
                        connector_prefix_cache_hits = num_external_computed_tokens
                    num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens
                    assert num_computed_tokens <= request.num_tokens
                    if 0 < num_computed_tokens <= self.prefix_replay_tokens:
                        new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                        num_new_local_computed_tokens = 0
                        num_external_computed_tokens = 0
                        connector_prefix_cache_hits = 0
                        num_computed_tokens = 0
                        load_kv_async = False
                    if self._ec_transfer_pending(request, num_computed_tokens):
                        skip_request(request_queue)
                        continue
                    if request.prefill_stats and request.num_preemptions <= 0:
                        assert num_computed_tokens <= request.num_prompt_tokens
                        request.prefill_stats.set(num_prompt_tokens=request.num_prompt_tokens, num_local_cached_tokens=num_new_local_computed_tokens, num_external_cached_tokens=num_external_computed_tokens)
                else:
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens
                    if self._ec_transfer_pending(request, num_computed_tokens):
                        skip_request(request_queue)
                        continue
                encoder_inputs_to_schedule = None
                external_load_encoder_input = []
                new_encoder_compute_budget = encoder_compute_budget
                pad_spec_decode = False
                num_replay_tokens = 0
                if load_kv_async:
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                elif defer_prefills and num_computed_tokens < request.num_tokens - 1:
                    break
                else:
                    request_token_budget = min(token_budget, input_budget - draft_slots)
                    if did_prefix_cache_lookup:
                        num_replay_tokens = self._mark_prefix_replay(request, num_computed_tokens)
                        num_computed_tokens -= num_replay_tokens
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    if (self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None) and self.num_sampled_tokens_per_step > 0 and (num_new_tokens == 1) and (not prefill_scheduled) and (scheduled_running_reqs or num_computed_tokens > 0):
                        padded_num_tokens = 1 + self.num_spec_tokens
                        if num_computed_tokens + padded_num_tokens + self.num_sampled_tokens_per_step <= self.max_model_len:
                            if padded_num_tokens > request_token_budget:
                                break
                            num_new_tokens = padded_num_tokens
                            pad_spec_decode = True
                    if 0 < long_prefill_token_threshold < num_new_tokens:
                        num_new_tokens = long_prefill_token_threshold
                    if not self.scheduler_config.enable_chunked_prefill and num_new_tokens > request_token_budget:
                        break
                    num_new_tokens = min(num_new_tokens, request_token_budget, self._qoserve_quota(request))
                    assert num_new_tokens > 0
                    if self.need_mamba_block_aligned_split:
                        num_new_tokens = self._mamba_block_aligned_split(request, num_new_tokens, num_new_local_computed_tokens, num_external_computed_tokens)
                        if num_new_tokens == 0:
                            break
                        if pad_spec_decode and num_new_tokens != 1 + self.num_spec_tokens:
                            num_new_tokens = 1
                            pad_spec_decode = False
                    if request.has_encoder_inputs:
                        encoder_inputs_to_schedule, num_new_tokens, new_encoder_compute_budget, external_load_encoder_input = self._try_schedule_encoder_inputs(request, num_computed_tokens, num_new_tokens, encoder_compute_budget, shift_computed_tokens=self.num_prefill_lookahead)
                    num_new_tokens = self._reserve_prefill_lookahead(request, num_computed_tokens, num_new_tokens)
                    if num_new_tokens == 0:
                        break
                limit_lookahead_tokens = load_kv_async and self.num_lookahead_tokens > 0
                effective_lookahead_tokens = 0 if limit_lookahead_tokens else self.num_lookahead_tokens
                num_encoder_tokens = 0
                if self.is_encoder_decoder and request.has_encoder_inputs and encoder_inputs_to_schedule:
                    num_encoder_tokens = sum((request.get_num_encoder_embeds(i) for i in encoder_inputs_to_schedule))
                reserved_blocks = 0
                if load_kv_async:
                    reserved_blocks = self._inflight_prefill_reserved_blocks() + self._spec_decode_step_blocks()
                num_tokens_past_hit = max(num_new_tokens - num_replay_tokens, 0)
                new_blocks = self.kv_cache_manager.allocate_slots(request, num_tokens_past_hit, num_new_computed_tokens=num_new_local_computed_tokens, new_computed_blocks=new_computed_blocks, num_lookahead_tokens=effective_lookahead_tokens, num_external_computed_tokens=num_external_computed_tokens, delay_cache_blocks=load_kv_async, num_encoder_tokens=num_encoder_tokens, full_sequence_must_fit=self.scheduler_reserve_full_isl, reserved_blocks=reserved_blocks + self._qoserve_other_commitments(request), has_scheduled_reqs=bool(self.running))
                if new_blocks is None:
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break
                if self.connector is not None:
                    self.connector.update_state_after_alloc(request, self.kv_cache_manager.get_blocks(request_id), num_external_computed_tokens)
                    if self.connector_prefix_cache_stats is not None and connector_prefix_cache_queries != 0:
                        self.connector_prefix_cache_stats.record(num_tokens=connector_prefix_cache_queries, num_hits=connector_prefix_cache_hits, preempted=request.num_preemptions > 0)
                if did_prefix_cache_lookup:
                    self.kv_cache_manager.record_prefix_cache_stats(request, num_new_local_computed_tokens)
                if load_kv_async:
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    request.num_computed_tokens = num_computed_tokens
                    self._inflight_prefills.add(request)
                    if self.needs_kv_cache_zeroing:
                        self._skip_zero_block_ids.update(self.kv_cache_manager.get_zeroing_block_ids_in_range(request_id, num_new_local_computed_tokens, num_computed_tokens))
                    skip_request(request_queue)
                    continue
                request = request_queue.pop_request()
                self.deferred_waiting.discard(request)
                self.running.append(request)
                if num_external_computed_tokens > 0:
                    has_sync_kv_loads = True
                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED, scheduled_timestamp)
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f'Invalid request status: {request.status}')
                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(request_id)
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                input_budget -= num_new_tokens + draft_slots
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                if pad_spec_decode:
                    assert num_new_tokens == 1 + self.num_spec_tokens
                    scheduled_spec_decode_tokens[request_id] = [-1] * self.num_spec_tokens
                if num_computed_tokens + num_new_tokens < request.num_tokens:
                    self._inflight_prefills.add(request)
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)
                    encoder_compute_budget = new_encoder_compute_budget
                if external_load_encoder_input:
                    for i in external_load_encoder_input:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)
            if step_skipped_kv_holding:
                self.kv_holding_waiting.prepend_requests(step_skipped_kv_holding)
            if step_skipped_waiting:
                self.waiting.prepend_requests(step_skipped_waiting)
            if not defer_prefills:
                _, num_waiting = self.get_request_counts()
                self.prefill_capacity_bound = num_waiting > len(self.deferred_waiting)
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert input_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(scheduled_running_reqs) <= len(self.running)
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        with record_function_or_nullcontext('schedule: get_num_common_prefix_blocks'):
            if self.running:
                any_request_id = self.running[0].request_id
                num_common_prefix_blocks = self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
        if self.use_v2_model_runner:
            scheduled_new_reqs.extend(scheduled_resumed_reqs)
            scheduled_resumed_reqs.clear()
            new_reqs_data = [NewRequestData.from_request(req, req_to_new_blocks[req.request_id].get_block_ids(), req._all_token_ids, uses_mrope=self.model_uses_mrope) for req in scheduled_new_reqs]
        else:
            new_reqs_data = [NewRequestData.from_request(req, req_to_new_blocks[req.request_id].get_block_ids(), uses_mrope=self.model_uses_mrope) for req in scheduled_new_reqs]
        with record_function_or_nullcontext('schedule: make_cached_request_data'):
            cached_reqs_data = self._make_cached_request_data(scheduled_running_reqs, scheduled_resumed_reqs, num_scheduled_tokens, scheduled_spec_decode_tokens, req_to_new_blocks)
        if not self.use_v2_model_runner:
            self.prev_step_scheduled_req_ids.clear()
            self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())
        boundary_state_offloads = self.kv_cache_manager.take_boundary_state_offloads()
        kv_connector_block_state = None
        if self.connector is not None:
            block_state_req_ids = set(num_scheduled_tokens)
            block_state_req_ids.update((req_id for req_id in boundary_state_offloads if req_id in self.requests))
            kv_connector_block_state = KVConnectorBlockState(req_ids=block_state_req_ids, resolve_block_ids=self.kv_cache_manager.get_block_ids, boundary_state_offloads=boundary_state_offloads)
        kv_cache_block_copies, cow_retained_blocks = self.kv_cache_manager.take_kv_cache_block_copies()
        if kv_cache_block_copies:
            self._free_cow_retained_blocks(cow_retained_blocks, self.sched_step_seq + 1)
        pending_kv_cache_block_copies = kv_cache_block_copies or None
        num_spec_tokens_to_schedule = self.num_spec_tokens
        if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[len(num_scheduled_tokens)]
        scheduled_encoder_input_stats = None
        if self.log_stats and self.observability_config.enable_logging_iteration_details:
            scheduled_encoder_input_stats = self._make_scheduled_encoder_input_stats(scheduled_encoder_inputs)
        scheduler_output = SchedulerOutput(scheduled_new_reqs=new_reqs_data, scheduled_cached_reqs=cached_reqs_data, num_scheduled_tokens=num_scheduled_tokens, total_num_scheduled_tokens=total_num_scheduled_tokens, scheduled_spec_decode_tokens=scheduled_spec_decode_tokens, scheduled_encoder_inputs=scheduled_encoder_inputs, scheduled_encoder_input_stats=scheduled_encoder_input_stats, num_common_prefix_blocks=num_common_prefix_blocks, preempted_req_ids=self.reset_preempted_req_ids, finished_req_ids=self.finished_req_ids, free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(), new_block_ids_to_zero=self._get_new_block_ids_to_zero(), has_sync_kv_loads=has_sync_kv_loads, kv_cache_block_copies=pending_kv_cache_block_copies, kv_connector_block_state=kv_connector_block_state, num_spec_tokens_to_schedule=num_spec_tokens_to_schedule, ec_manager_metadata=self.encoder_cache_manager.get_manager_metadata())
        if self.connector is not None:
            meta = self._build_kv_connector_meta(self.connector, scheduler_output)
            scheduler_output.kv_connector_metadata = meta
        if self.aux_output_connector is not None:
            scheduler_output.aux_output_connector_metadata = self.aux_output_connector.build_connector_meta(scheduler_output, self.requests)
        if self.ec_connector is not None:
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(scheduler_output)
            scheduler_output.ec_connector_metadata = ec_meta
        scheduler_output.kv_connector_block_state = None
        if self.defer_block_free and total_num_scheduled_tokens > 0:
            self.sched_step_seq += 1
        with record_function_or_nullcontext('schedule: update_after_schedule'):
            self._update_after_schedule(scheduler_output)
        return scheduler_output

class QoServeScheduler(AuthorPolicy, CompatibleNativeScheduler):
    _qoserve_policy_enabled = True

    def __init__(self, *args, **kwargs):
        self.require_deployment_coefficients()
        CompatibleNativeScheduler.__init__(self, *args, **kwargs)
        if self.scheduler_config.long_prefill_token_threshold != 0:
            raise ValueError('QoServe dynamic candidates require native long-prefill cap0')
        if self.policy != SchedulingPolicy.FCFS:
            raise ValueError('native policy must be FCFS before QoServe ordering')
        initial = float(os.environ['INFERX_QOSERVE_INITIAL_PPS'])
        if not math.isfinite(initial) or initial <= 0:
            raise ValueError('invalid calibrated initial prefill throughput')
        self.expected_prefill_throughput = initial
        self.chunk_size = self.max_num_scheduled_tokens
        self._prefill_queue = []
        self.last_prefill_size = 0
        self.prev_call_time = -1

    def _qoserve_key(self, view, now):
        context = SimpleNamespace(execution_threshold=view.contract['tbt_s'], execution_threshold_batched=view.contract['tbt_s'])
        return AuthorPolicy._sorting_key(context, view, now)

    def schedule(self, throttle_prefills=False):
        if throttle_prefills:
            raise ValueError('DP prefill throttling is outside qualification')
        self.kv_cache_manager.new_step_starts()
        now = time.monotonic()
        self._qoserve_update_prefill_throughput(now)
        decoders = [self._qoserve_views[r.request_id] for r in self.running if self._qoserve_views[r.request_id].prompt_processing_finished]
        decoders.sort(key=lambda view: self._qoserve_key(view, now))
        prefills = []
        running = {r.request_id for r in self.running}
        for rid, view in self._qoserve_views.items():
            if view.prompt_processing_finished:
                continue
            view.cached_prefix_tokens = 0
            request = view.request
            if rid in running or (request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED) and self._qoserve_can_admit(request)):
                prefills.append(view)
        heapq.heapify(prefills)
        self._prefill_queue = prefills
        min_slack = min((view._get_TBT_deadline(view.contract['tbt_s'], view.contract['tbt_s']) - now for view in decoders), default=1e+18)
        decode_context = sum((view._get_context_size() for view in decoders))
        allowance = 0
        predicted = None
        if prefills:
            allowance, predicted = self._get_prefill_size_by_slack(min_slack, decode_context, prefills[0]._get_context_size(), len(decoders))
        self._qoserve_quotas = {view.seq_id: 1 for view in decoders}
        ranks = {}
        used = 0
        while self._prefill_queue:
            view = heapq.heappop(self._prefill_queue)
            if self._qoserve_relegate_once(view, now):
                continue
            count = min(allowance - used, view.get_prompt_len() - view.get_num_prompt_tokens_processed())
            if count <= 0:
                break
            ranks[view.seq_id] = len(ranks)
            self._qoserve_quotas[view.seq_id] = count
            used += count
        decode_order = {view.seq_id: i for i, view in enumerate(decoders)}
        self.running = sorted(self.running, key=lambda request: (0, decode_order[request.request_id]) if request.request_id in decode_order else (1, ranks.get(request.request_id, 10 ** 9)))
        for name in ('waiting', 'kv_holding_waiting'):
            setattr(self, name, FCFSRequestQueue(sorted(getattr(self, name), key=lambda request: ranks.get(request.request_id, 10 ** 9))))
        ceiling = self.max_num_scheduled_tokens
        selected = min(ceiling, max(1, used + len(decoders)))
        prefill_ids = {view.seq_id for view in self._qoserve_views.values() if not view.prompt_processing_finished}
        try:
            self.max_num_scheduled_tokens = selected
            output = self._qoserve_schedule_native()
        finally:
            self.max_num_scheduled_tokens = ceiling
        self.last_prefill_size = sum((count for rid, count in output.num_scheduled_tokens.items() if rid in prefill_ids))
        self._qoserve_decisions.append(dict(step=self.current_step, min_slack=min_slack, allowance=allowance, predicted_s=predicted, selected_total=selected, actual=dict(output.num_scheduled_tokens), effective_free_blocks=self._qoserve_effective_free_blocks()))
        return output
