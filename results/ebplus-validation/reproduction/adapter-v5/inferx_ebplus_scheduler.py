# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Native scheduler body plus InferX eligibility/state boundary.
from __future__ import annotations
import hashlib
from pathlib import Path
import vllm.v1.core.sched.scheduler as _native
_NativeScheduler = _native.Scheduler
if hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest() != 'f820bee4f42b878b262a30f3728a97d7413477b2a277748e20f2528ac2ec2f2a':
    raise RuntimeError('installed native scheduler hash mismatch')
globals().update({k:v for k,v in vars(_native).items() if not k.startswith('__')})
from inferx_ebplus_policy import AuthorPolicy
import time
import os
import json

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

class ObservedNativeScheduler(_NativeScheduler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        record_native_runtime(self)

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
        self._ebplus_last_admission_failure = dict(request_id=request.request_id, prompt_tokens=request.num_prompt_tokens, free_blocks=free, phase=self.pd_phase, full_sequence_must_fit=self.scheduler_reserve_full_isl, decoders=len(self.pd_decoding_requests), running=len(self.running))

    def _ebplus_admission_pressure(self):
        blocked = self._ebplus_admission_blocked
        if blocked is None:
            return False
        request_id, failed_free = blocked
        queue = self.kv_holding_waiting or self.waiting
        free = self.kv_cache_manager.block_pool.get_num_free_blocks()
        if not self.pd_decoding_requests or not queue or queue.peek_request().request_id != request_id or (free > failed_free):
            self._ebplus_admission_blocked = None
            return False
        return True

    def _evaluate_mode_switch(self):
        eligible = self.pd_param_initialized and self._avg_prompt_len + self.pd_avg_output_tokens > 0 and (0 < self.pd_k_ratio < 1)
        inputs = None
        if eligible:
            inputs = dict(mu_L=self._avg_prompt_len, mu_o=self.pd_avg_output_tokens, N_obs=max(1.0, self._n_obs), theta0=self.pd_k_ratio, alpha_mb=self._alpha_mb, alpha_p=self.pd_alpha_p, beta_p=self.pd_beta_p, alpha_d=self.pd_alpha_d, beta_d=self.pd_beta_d, mb_cost_a=self._mb_cost_a, mb_cost_b=self._mb_cost_b, mb_cost_c=self._mb_cost_c, delta=self._mode_switch_delta, cooldown_before=self._mode_cooldown, active_before=self._active_scheduler)
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
        pending = self.pd_batch_completed_count
        completed = self.pd_total_completed + (pending if pending < self.pd_param_update_interval else 0)
        state = dict(mode=self.scheduler_mode, active_scheduler=getattr(self, '_active_scheduler', self.scheduler_mode), evaluations=self._ebplus_evaluations, parameters_initialized=self.pd_param_initialized, mb_cost_profiled=getattr(self, '_mb_cost_profiled', False), execution_counts=self._ebplus_execution_counts, mode_switch_history=getattr(self, '_mode_switch_history', []), completed=completed, last_evaluation_inputs=self._last_evaluation_inputs, native_admission_failures=self._ebplus_admission_failures, native_admission_escapes=self._ebplus_admission_escapes, last_native_admission_failure=self._ebplus_last_admission_failure, zero_schedule_streak=self._ebplus_empty_schedule_streak, scheduler_snapshot=self._ebplus_empty_snapshot, max_num_running_reqs=self.max_num_running_reqs, block_size=self.block_size, num_gpu_blocks=self.kv_cache_config.num_blocks, kv_groups=[dict(type=type(g.kv_cache_spec).__name__, spec=str(g.kv_cache_spec), layers=g.layer_names) for g in self.kv_cache_config.kv_cache_groups])
        target = Path(path)
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
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
        self.chunk_prefilling = [r for r in self.running if self._is_prefill(r)]
        self._handle_phase_transition()
        self._ebplus_exclusive = True
        try:
            self._ebplus_execution_counts['decode' if self.pd_phase == 1 else 'prefill'] += 1
            output = self._schedule_exclusive_native()
            self._ebplus_empty_schedule_streak = self._ebplus_empty_schedule_streak + 1 if output.total_num_scheduled_tokens == 0 else 0
            if self._ebplus_empty_schedule_streak == 64 and self.pd_decoding_requests:
                self._ebplus_empty_snapshot = self._ebplus_scheduler_snapshot()
                self._ebplus_dump_evidence()
            return output
        finally:
            self._ebplus_exclusive = False

    def _ebplus_scheduler_snapshot(self):
        return dict(phase=self.pd_phase, current_step=self.current_step, N=self.pd_batch_size_N, k=self.pd_switch_threshold_k, refill_target=self.pd_refill_target, prefilled=self.pd_prefilled_count, free_blocks=self.kv_cache_manager.block_pool.get_num_free_blocks(), running=[r.request_id for r in self.running], waiting=[r.request_id for r in self.waiting], kv_holding_waiting=[r.request_id for r in self.kv_holding_waiting], deferred_waiting=[r.request_id for r in self.deferred_waiting], decoding=sorted(self.pd_decoding_requests), inflight_prefills=[r.request_id for r in self._inflight_prefills], requests={rid: dict(status=r.status.name, prompt_tokens=r.num_prompt_tokens, computed_tokens=r.num_computed_tokens, output_tokens=len(r.output_token_ids), next_decode_eligible_step=r.next_decode_eligible_step, output_placeholders=r.num_output_placeholders) for rid, r in self.requests.items()})

    def _compute_adaptive_N(self):
        return min(self.max_num_running_reqs, AuthorPolicy._compute_adaptive_N(self))

    def _update_after_schedule(self, output):
        before = {}
        if self._ebplus_exclusive:
            before = {rid: self._is_prefill(self.requests[rid]) for rid in output.num_scheduled_tokens}
        _NativeScheduler._update_after_schedule(self, output)
        if self._ebplus_exclusive:
            for rid, was_prefill in before.items():
                request = self.requests[rid]
                if was_prefill and (not self._is_prefill(request)):
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
        count, tokens = (0, 0)
        for request in list(self.chunk_prefilling):
            if not self._request_blocks_can_be_freed(request):
                raise RuntimeError('cannot synchronously preempt in-flight prefill')
            computed_before = request.num_computed_tokens
            _NativeScheduler._preempt_request(self, request, time.monotonic())
            self.running.remove(request)
            self.pd_decoding_requests.discard(request.request_id)
            count += 1
            tokens += computed_before
        self.chunk_prefilling.clear()
        return (count, tokens)

    def _free_request(self, request):
        if request.status in (RequestStatus.FINISHED_STOPPED, RequestStatus.FINISHED_LENGTH_CAPPED, RequestStatus.FINISHED_REPETITION):
            self._ebplus_record_completion(request)
        self.pd_decoding_requests.discard(request.request_id)
        if request in self.chunk_prefilling:
            self.chunk_prefilling.remove(request)
        return _NativeScheduler._free_request(self, request)

    def _schedule_exclusive_native(self, throttle_prefills: bool=False) -> SchedulerOutput:
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
        long_prefill_token_threshold = self.scheduler_config.long_prefill_token_threshold
        if long_prefill_token_threshold > 0 and self.adaptive_long_prefill_threshold:
            long_prefill_token_threshold = self.scheduler_config.long_prefill_token_threshold
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
            if not phase_allows_running(self.pd_phase, request, self.pd_decoding_requests):
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
            num_new_tokens = min(num_new_tokens, token_budget, input_budget - draft_slots)
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
                    new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, num_lookahead_tokens=self.num_lookahead_tokens)
                    if new_blocks is not None:
                        break
                    if self.pd_phase in (0, 2):
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
                target = self.pd_batch_size_N if self.pd_phase == 0 else self.pd_refill_target
                if not phase_allows_admission(self.pd_phase, target, self.pd_prefilled_count, self.requests, num_scheduled_tokens):
                    break
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
                    num_new_tokens = min(num_new_tokens, request_token_budget)
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
                new_blocks = self.kv_cache_manager.allocate_slots(request, num_tokens_past_hit, num_new_computed_tokens=num_new_local_computed_tokens, new_computed_blocks=new_computed_blocks, num_lookahead_tokens=effective_lookahead_tokens, num_external_computed_tokens=num_external_computed_tokens, delay_cache_blocks=load_kv_async, num_encoder_tokens=num_encoder_tokens, full_sequence_must_fit=self.scheduler_reserve_full_isl, reserved_blocks=reserved_blocks, has_scheduled_reqs=bool(self.running))
                if new_blocks is None:
                    self._ebplus_note_admission_block(request)
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

    def _ebplus_record_completion(self, request):
        if self.use_pd_scheduler:
            if request.request_id in self.pd_decoding_requests:
                self.pd_completed_decode_count += 1
                self.pd_decoding_requests.discard(request.request_id)
                output_tokens = request.num_tokens - request.num_prompt_tokens
                if output_tokens > 0:
                    self._update_params_online(output_tokens)
                pl = float(request.num_prompt_tokens)
                a = self.pd_avg_prompt_ema_alpha
                self.pd_avg_prompt_len = a * pl + (1 - a) * self.pd_avg_prompt_len
                if self.scheduler_mode == 'auto':
                    self._avg_prompt_len = self.pd_avg_prompt_len
            elif self.scheduler_mode == 'auto' and self._active_scheduler == 'mb':
                output_tokens = request.num_tokens - request.num_prompt_tokens
                if output_tokens > 0:
                    self._update_params_online(output_tokens)
                prompt_len = float(request.num_prompt_tokens)
                a = self.pd_avg_prompt_ema_alpha
                self.pd_avg_prompt_len = a * prompt_len + (1 - a) * self.pd_avg_prompt_len
                self._avg_prompt_len = self.pd_avg_prompt_len

    def _handle_phase_transition(self) -> None:
        """
        Handle P/D phase transitions based on current state.

        Updates self.pd_phase based on:
        - Phase 0 -> 1: When N prefilled or no more waiting
        - Phase 1 -> 2: When k decoded AND k waiting available
        - Phase 2 -> 1: When k prefilled or no more waiting
        - Reset to 0: When idle (no decode work, no running, has waiting)
        """
        num_running = len(self.running)
        if len(self.pd_decoding_requests) > num_running:
            running_ids = {req.request_id for req in self.running}
            orphaned_ids = self.pd_decoding_requests - running_ids
            if orphaned_ids:
                logger.warning('[P/D] Cleaning %d orphaned decoding IDs', len(orphaned_ids))
                self.pd_decoding_requests -= orphaned_ids
        num_pending_chunks = len(self.chunk_prefilling)
        if num_pending_chunks > 0:
            running_set = set(self.running)
            orphaned_chunks = [r for r in self.chunk_prefilling if r not in running_set or r.num_computed_tokens >= r.num_prompt_tokens]
            if orphaned_chunks:
                logger.warning('[P/D] Cleaning %d orphaned/completed chunk_prefilling', len(orphaned_chunks))
                for req in orphaned_chunks:
                    self.chunk_prefilling.remove(req)
                    if req in running_set and req.num_computed_tokens >= req.num_prompt_tokens and (req.request_id not in self.pd_decoding_requests):
                        self.pd_decoding_requests.add(req.request_id)
                        self.pd_prefilled_count += 1
                num_pending_chunks = len(self.chunk_prefilling)
        num_decoding = len(self.pd_decoding_requests)
        has_decoding = num_decoding > 0
        waiting_count = len(self.waiting) + num_pending_chunks
        has_waiting = waiting_count > 0
        has_pending_chunks = num_pending_chunks > 0
        prev_phase = self.pd_phase
        kv_cache_full = False
        adaptive_threshold = self._compute_adaptive_kv_threshold()
        if hasattr(self.kv_cache_manager, 'block_pool'):
            total_blocks = self.kv_cache_manager.block_pool.num_gpu_blocks
            free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
            kv_cache_full = free_blocks < total_blocks * adaptive_threshold
        native_admission_pressure = self._ebplus_admission_pressure()
        kv_cache_full = kv_cache_full or native_admission_pressure
        if self.pd_phase == 0:
            can_transition = not has_pending_chunks or kv_cache_full
            if self.pd_prefilled_count >= self.pd_batch_size_N:
                if can_transition:
                    self.pd_phase = 1
                    self.pd_completed_decode_count = 0
                else:
                    logger.info(f'[P/D] Phase 0: waiting for {num_pending_chunks} chunked prefills to complete before decode')
            elif kv_cache_full and has_decoding:
                adaptive_n = self._compute_adaptive_N()
                _floor_div = int(os.environ.get('VLLM_PD_MIN_N_FLOOR_DIV', '10'))
                min_n = max(16, self.max_num_running_reqs // _floor_div)
                new_n = min(adaptive_n, self.pd_prefilled_count)
                if new_n >= min_n:
                    self._update_batch_size_n(new_n, 'kv_escape')
                preempted_chunks, preempted_tokens = self._preempt_chunk_prefilling()
                logger.info(f'[P/D] KV cache threshold ({adaptive_threshold:.2%}) escape: phase 0->1 with {self.pd_prefilled_count} prefilled, {waiting_count} waiting, adaptive_N={adaptive_n}, avg_output={self.pd_avg_output_tokens:.1f}, preempted_chunks={preempted_chunks} (freed {preempted_tokens} computed tokens)')
                self.pd_phase = 1
                self.pd_completed_decode_count = 0
            elif not has_waiting and has_decoding and can_transition:
                min_n = max(16, self.max_num_running_reqs // 10)
                if self.pd_prefilled_count >= min_n:
                    self._update_batch_size_n(self.pd_prefilled_count, 'cold_start')
                self.pd_phase = 1
                self.pd_completed_decode_count = 0
        elif self.pd_phase == 1:
            target_n = self.max_num_running_reqs
            if self.pd_batch_size_N < target_n and waiting_count >= target_n // 2 and (time.monotonic() - self.pd_last_n_update_time >= self.pd_n_update_cooldown):
                old_n = self.pd_batch_size_N
                self.pd_batch_size_N = target_n
                self._update_k_star()
                self.pd_last_n_update_time = time.monotonic()
                self._record_n_update(old_n, self.pd_batch_size_N, 'recovery')
                logger.info(f'[P/D] N RECOVERY: {old_n} -> {self.pd_batch_size_N} (queue filled, k*={self.pd_switch_threshold_k}, avg_out={self.pd_avg_output_tokens:.1f})')
            if num_decoding > 0 and (not kv_cache_full):
                N = self.pd_batch_size_N
                k_star = self.pd_switch_threshold_k
                fillable = min(waiting_count, max(0, N - num_decoding))
                denom = N - k_star
                if denom > 0 and fillable * denom >= num_decoding * k_star:
                    self.pd_refill_target = fillable
                    self.pd_phase = 2
                    self.pd_prefilled_count = 0
            elif not has_decoding and has_waiting:
                self._reset_pd_to_initial()
        elif self.pd_phase == 2:
            ready_to_decode = self.pd_prefilled_count >= self.pd_refill_target or (not has_waiting and has_decoding) or kv_cache_full
            if ready_to_decode:
                if not has_pending_chunks or kv_cache_full:
                    if kv_cache_full and has_pending_chunks:
                        preempted_chunks, preempted_tokens = self._preempt_chunk_prefilling()
                        logger.info(f'[P/D] Phase 2->1: preempted {preempted_chunks} chunks (freed {preempted_tokens} computed tokens, KV full)')
                    self.pd_phase = 1
                    self.pd_completed_decode_count = 0
                else:
                    logger.info(f'[P/D] Phase 2: waiting for {num_pending_chunks} chunked prefills to complete before decode')
        if prev_phase != self.pd_phase:
            logger.info(f'[P/D] {self.PD_PHASE_NAMES[prev_phase]} -> {self.PD_PHASE_NAMES[self.pd_phase]} | prefilled={self.pd_prefilled_count}, decoded={self.pd_completed_decode_count}, k*={self.pd_switch_threshold_k}, refill_target={self.pd_refill_target}, decoding={num_decoding}, N={self.pd_batch_size_N}, avg_out={self.pd_avg_output_tokens:.1f}, kv_thresh={adaptive_threshold:.2%}')
        if native_admission_pressure and prev_phase in (0, 2) and (self.pd_phase == 1):
            self._ebplus_admission_escapes += 1
