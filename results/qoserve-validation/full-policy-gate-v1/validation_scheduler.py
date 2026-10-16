"""Read-only functional observations around the actual adapted QoServe policy."""
import json
import os
from pathlib import Path
from inferx_qoserve_scheduler import QoServeScheduler
from inferx.qoserve_profile import snapshot_step_geometry


class PolicyProbeScheduler(QoServeScheduler):
    def __init__(self, *args, **kwargs):
        self.validation_geometry = []
        self.validation_requeues = []
        super().__init__(*args, **kwargs)

    def _qoserve_relegate_once(self, view, now):
        before = view.drop
        result = super()._qoserve_relegate_once(view, now)
        if result or before != view.drop:
            self.validation_requeues.append(dict(request_id=view.seq_id,
                trace_id=view.contract['trace_id'], step=self.current_step,
                before=before, after=view.drop, requeued=result))
        return result

    def schedule(self, throttle_prefills=False):
        metadata = {rid: dict(num_prompt_tokens=r.num_prompt_tokens,
            num_tokens=r.num_tokens, output_tokens_before=len(r.output_token_ids))
            for rid, r in self.requests.items()}
        output = super().schedule(throttle_prefills)
        assert self.max_num_scheduled_tokens == 4096
        requests, features = snapshot_step_geometry(output, metadata)
        self.validation_geometry.append(dict(step=self.current_step, requests=requests,
            features=features, order=list(output.num_scheduled_tokens),
            preempted_request_ids=sorted(output.preempted_req_ids)))
        return output

    def shutdown(self):
        Path(os.environ['INFERX_POLICY_VALIDATION']).write_text(json.dumps(dict(
            scope='Functional qualification only; instrumentation excluded from efficacy.',
            geometry=self.validation_geometry, requeues=self.validation_requeues), indent=2)+'\n')
        return super().shutdown()
