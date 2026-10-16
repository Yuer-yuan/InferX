"""QoServe caller contracts and source-policy views of native requests.

Native cache/dispatch integration is qualified separately in the pinned vLLM
environment. This module's lightweight part has no CUDA/runtime imports.
"""
import math
from types import SimpleNamespace


def parse_contract(extra_args):
    """Validate flat vllm_xargs; only untagged code checks receive defaults."""
    tags = {key:value for key,value in (extra_args or {}).items() if key.startswith('qoserve_')}
    if not tags:
        return dict(tier=0, deadline_s=6., tbt_s=.1, trace_id=None)
    required = {'qoserve_tier', 'qoserve_deadline_s', 'qoserve_tbt_s'}
    if not required <= tags.keys() or tags.keys() - required - {'qoserve_trace_id'}:
        raise ValueError('incomplete or unknown QoServe contract fields')
    tier = tags['qoserve_tier']
    if type(tier) is not int or tier not in (0, 1):
        raise ValueError('QoServe tier must be integer 0 or 1')
    values = []
    for key in ('qoserve_deadline_s', 'qoserve_tbt_s'):
        value = tags[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
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
            self.state = SimpleNamespace(_arrived_at=self.arrival_time, e2e_prefill_time=0.)
            self.scheduler_type = policy.scheduler_type
            self.hybrid_prioritization_param = policy.hybrid_prioritization_param
            self.drop = 0
            self.cached_prefix_tokens = 0
            self.last_slack = 0.
            self.tier = self.contract['tier']
            self.tier_deadline = self.contract['deadline_s']
            self.ttft_deadline = self.contract['deadline_s']
            self.num_decode_tokens = -1  #max_tokens is a cap, not known actual output length.
            context = SimpleNamespace(output_len_pred=policy.output_len_pred,
                                      execution_threshold_batched=self.contract['tbt_s'])
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
            # Native replay of past output is prefill work too.
            return max(self.request.num_prompt_tokens, self.request.num_tokens - 1)

        def get_num_prompt_tokens_processed(self):
            return min(max(self.request.num_computed_tokens, self.cached_prefix_tokens), self.get_prompt_len())

        def _get_context_size(self):
            # Original decode context includes the latest sampled input token;
            # prefill context is only the already-computed prompt/cache prefix.
            return (self.request.num_tokens if self.prompt_processing_finished
                    else self.get_num_prompt_tokens_processed())

    return NativeSequenceView
