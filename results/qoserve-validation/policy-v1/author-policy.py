# SPDX-License-Identifier: MIT
# Copyright (c) Microsoft Corporation.
# QoServe source commit 72339db5b120614033b317880ad3a05a5e4c9415; see bundled author LICENSE.
import heapq

TOKEN_SEARCH_SPACE = [128] + list(range(256, 2049, 256)) + [2552]
LOW_MEMORY_TOKEN_SEARCH_SPACE = list(range(128, 513, 128))

class AuthorSequence:

    def __lt__(self, other: object) -> bool:
        if self.scheduler_type == 'fcfs':
            return self.arrival_time < other.arrival_time
        if self.scheduler_type == 'edf':
            return self.ttft_deadline + self.arrival_time < other.ttft_deadline + other.arrival_time
        self_remaining_tokens = self.get_prompt_len() - self.get_num_prompt_tokens_processed()
        other_remaining_tokens = other.get_prompt_len() - other.get_num_prompt_tokens_processed()
        if self.scheduler_type == 'srpf':
            return self_remaining_tokens < other_remaining_tokens
        if self.drop != other.drop:
            return self.drop < other.drop
        self_deadline = self.ttft_deadline + self.arrival_time + self.hybrid_prioritization_param * self_remaining_tokens
        other_deadline = other.ttft_deadline + other.arrival_time + other.hybrid_prioritization_param * other_remaining_tokens
        return self_deadline < other_deadline

    def _get_TBT_deadline(self, execution_threshold: float, execution_threshold_batched: float) -> float:
        ttft = self.state._arrived_at + max(self.state.e2e_prefill_time, self.ttft_deadline)
        decode_iteration = len(self.output_token_ids) + 1
        if self.tier > 0:
            execution_threshold = execution_threshold_batched
        return ttft + execution_threshold * decode_iteration

class AuthorPolicy:
    chunk_size = 512
    execution_threshold = 0.1
    execution_threshold_batched = 0.4
    scheduler_type = 'deadline'
    output_len_pred = 15
    hybrid_prioritization_param = 0.008
    expected_prefill_throughput = 8500.0
    prefill_throughput_weight_decay = 0.995
    count = 0
    max_time = 0
    min_time = 1e+18
    pred_thres = 1.2
    prev_call_time = -1
    last_prefill_size = 0
    Sequence = AuthorSequence
    _qoserve_coefficients = {'tokens_0_512': {'coefficients': {'batch_num_tokens': 5.882475e-05, 'batch_num_prefill_tokens': 2.247956e-05, 'batch_num_decode_tokens': 3.634519e-05, 'batch_decode_context': 1.111258e-07, 'batch_prefill_context': 7.554773e-07}, 'intercept': 0.014936}, 'tokens_513_plus': {'coefficients': {'batch_num_tokens': 2.71905e-05, 'batch_num_prefill_tokens': 6.863043e-05, 'batch_num_decode_tokens': -4.143993e-05, 'batch_decode_context': 1.663409e-07, 'batch_prefill_context': 2.727223e-06}, 'intercept': 0.005004}}
    deployment_coefficients_supplied = False
    coefficient_source = 'author-reference'

    @classmethod
    def require_deployment_coefficients(cls):
        if not cls.deployment_coefficients_supplied:
            raise ValueError('deployment requires hardware coefficients, not author A100 reference tables')

    def _predict_batch_time(self, batch_decode_context, batch_prefill_context, batch_num_prefill_tokens, batch_num_decode_tokens):
        """
        Predict execution time using two stratified linear models:
        - tokens_0_512:      0 < total_tokens <= 512
        - tokens_513_plus:   total_tokens > 512
        """
        batch_num_tokens = batch_num_prefill_tokens + batch_num_decode_tokens
        tokens_0_512 = self._qoserve_coefficients['tokens_0_512']
        tokens_513_plus = self._qoserve_coefficients['tokens_513_plus']
        model = tokens_0_512 if batch_num_tokens <= 512 else tokens_513_plus
        c = model['coefficients']
        y = model['intercept']
        y += c['batch_num_tokens'] * batch_num_tokens
        y += c['batch_num_prefill_tokens'] * batch_num_prefill_tokens
        y += c['batch_num_decode_tokens'] * batch_num_decode_tokens
        y += c['batch_decode_context'] * batch_decode_context
        y += c['batch_prefill_context'] * batch_prefill_context
        return y

    def _get_prefill_size_by_slack(self, min_slack, batch_decode_context, batch_prefill_context, batch_num_decode_tokens):
        """Find max prefill tokens that can be processed within min_slack time.
        Search based on total tokens (prefill + decode) but return prefill tokens,
        while respecting memory budget constraints."""
        token_search_space = TOKEN_SEARCH_SPACE
        free_blocks = self._qoserve_effective_free_blocks()
        block_size = self.block_size
        memory_budget = free_blocks * block_size // 32
        if memory_budget < 1000:
            token_search_space = LOW_MEMORY_TOKEN_SEARCH_SPACE
        best_total_tokens = token_search_space[0]
        best_total_tokens_pred_time = self._predict_batch_time(batch_decode_context, batch_prefill_context, max(0, best_total_tokens - batch_num_decode_tokens), batch_num_decode_tokens)
        if min_slack <= 0:
            return (max(0, best_total_tokens - batch_num_decode_tokens), best_total_tokens_pred_time)
        lo, hi = (0, len(token_search_space) - 1)
        while lo <= hi:
            mid = (lo + hi) // 2
            total_tokens = token_search_space[mid]
            prefill_tokens = max(0, total_tokens - batch_num_decode_tokens)
            predicted_time = self._predict_batch_time(batch_decode_context, batch_prefill_context, prefill_tokens, batch_num_decode_tokens)
            adjusted_predicted_time = predicted_time * self.pred_thres
            if adjusted_predicted_time <= min_slack:
                best_total_tokens = total_tokens
                best_total_tokens_pred_time = predicted_time
                lo = mid + 1
            else:
                hi = mid - 1
        enable_logging = False
        if enable_logging:
            if not hasattr(self, '_log_dc_buffer'):
                self._log_dc_buffer = []
            log_line = f'{batch_decode_context},{batch_prefill_context},{batch_num_decode_tokens},{min_slack},{best_total_tokens},{best_total_tokens_pred_time}\n'
            self._log_dc_buffer.append(log_line)
            if len(self._log_dc_buffer) >= 200:
                with open(f'dc_{self.pred_thres}.csv', 'a') as f:
                    f.writelines(self._log_dc_buffer)
                self._log_dc_buffer.clear()
        return (max(0, best_total_tokens - batch_num_decode_tokens), best_total_tokens_pred_time)

    def _sorting_key(self, seq, now):
        if seq.prompt_processing_finished:
            slack = seq._get_TBT_deadline(self.execution_threshold, self.execution_threshold_batched) - now
            seq.last_slack = slack
            return (0, -slack)
        else:
            return (1, seq.arrival_time + seq.ttft_deadline)

    def _qoserve_ttft_budget(self, seq):
        if seq.tier >= 1:
            if seq.num_decode_tokens == -1:
                seq.ttft_deadline = seq.tier_deadline - self.output_len_pred * self.execution_threshold_batched
            else:
                seq.ttft_deadline = seq.tier_deadline - seq.num_decode_tokens * self.execution_threshold_batched
        return seq.ttft_deadline

    def _qoserve_relegate_once(self, seq, now):
        remaining_prefill_tokens = seq.get_prompt_len() - seq.get_num_prompt_tokens_processed()
        if self.scheduler_type == 'deadline' or self.scheduler_type == 'deadline_no_dynamic_chunking':
            service_time_required = remaining_prefill_tokens / self.expected_prefill_throughput
            if seq.drop == 0 and seq.mem_allocated == False and (now + service_time_required > seq.arrival_time + seq.ttft_deadline):
                seq.drop += 1
                heapq.heappush(self._prefill_queue, seq)
                return True
        return False

    def _qoserve_update_prefill_throughput(self, now):
        if self.last_prefill_size != 0:
            throughput_observation = self.last_prefill_size / (now - self.prev_call_time)
            self.expected_prefill_throughput = self.prefill_throughput_weight_decay * self.expected_prefill_throughput + (1 - self.prefill_throughput_weight_decay) * throughput_observation
        self.last_prefill_size = 0
        self.prev_call_time = now
