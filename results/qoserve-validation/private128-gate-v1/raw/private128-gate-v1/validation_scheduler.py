"""Functional probe only: native worker4096 with a temporary total budget128."""
import json
import os
from pathlib import Path
from inferx_qoserve_scheduler import ObservedNativeScheduler
from inferx.qoserve_profile import snapshot_step_geometry


class Private128Scheduler(ObservedNativeScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_scheduled_tokens == 4096
        assert self.scheduler_config.long_prefill_token_threshold == 0
        assert not self.scheduler_reserve_full_isl
        self.validation_geometry = []

    def schedule(self, throttle_prefills=False):
        metadata = {rid: dict(num_prompt_tokens=r.num_prompt_tokens,
                             num_tokens=r.num_tokens,
                             output_tokens_before=len(r.output_token_ids))
                    for rid, r in self.requests.items()}
        ceiling = self.max_num_scheduled_tokens
        try:
            self.max_num_scheduled_tokens = 128
            output = super().schedule(throttle_prefills)
        finally:
            self.max_num_scheduled_tokens = ceiling
        requests, features = snapshot_step_geometry(output, metadata)
        assert features['batch_num_tokens'] <= 128
        assert sum(output.num_scheduled_tokens.values()) == features['batch_num_tokens']
        self.validation_geometry.append(dict(step=self.current_step,
            requests=requests, features=features,
            preempted_request_ids=sorted(output.preempted_req_ids)))
        return output

    def shutdown(self):
        Path(os.environ['INFERX_PRIVATE128_GEOMETRY']).write_text(json.dumps(dict(
            scope='Functional/numerical probe, not SLO or throughput evidence.',
            worker_budget=4096, runtime_step_budget=128, rows=self.validation_geometry),
            indent=2) + '\n')
        return super().shutdown()
