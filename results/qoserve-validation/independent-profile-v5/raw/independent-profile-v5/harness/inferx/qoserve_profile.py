"""Identity and completed-step geometry contracts for QoServe calibration.

These checks reject mislabeled data. They do not establish that a GPU was used:
that additionally requires the owned run, pinned collector and runtime evidence.
"""
import math


FEATURES=('batch_num_tokens','batch_num_prefill_tokens','batch_num_decode_tokens',
          'batch_decode_context','batch_prefill_context')
TIMING_BOUNDARY='schedule-entry-to-native-update-exit'


def snapshot_step_geometry(output,metadata):
    """Read pre-step counters from the actual native SchedulerOutput ABI."""
    starts={}
    for request in output.scheduled_new_reqs:
        if request.req_id in starts:raise ValueError('duplicate new request geometry')
        starts[request.req_id]=request.num_computed_tokens
    cached=output.scheduled_cached_reqs
    if len(cached.req_ids)!=len(cached.num_computed_tokens):
        raise ValueError('cached request geometry lengths differ')
    for rid,computed in zip(cached.req_ids,cached.num_computed_tokens):
        if rid in starts:raise ValueError('duplicate cached request geometry')
        starts[rid]=computed
    if set(starts)!=set(output.num_scheduled_tokens):
        raise ValueError('native scheduled request geometry is not fully joined')
    rows=[];features=dict.fromkeys(FEATURES,0)
    for rid,tokens in output.num_scheduled_tokens.items():
        source=metadata[rid];computed=starts[rid]
        boundary=max(source['num_prompt_tokens'],source['num_tokens']-1)
        prefill=computed<boundary
        rows.append(dict(request_id=rid,computed_before=int(computed),prefill_end=boundary,
                         scheduled_tokens=int(tokens),stage='prefill' if prefill else 'decode',
                         output_tokens_before=source['output_tokens_before']))
        features['batch_num_tokens']+=int(tokens)
        if prefill:
            features['batch_num_prefill_tokens']+=int(tokens)
            features['batch_prefill_context']+=int(computed)
        else:
            features['batch_num_decode_tokens']+=int(tokens)
            features['batch_decode_context']+=int(computed+tokens)
    return rows,features


def _finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def validate_step_row(row,identity):
    """Return independently reconstructed features of one completed model step."""
    if row.get('identity')!=identity:
        raise ValueError('calibration hardware/model/source identity mismatch')
    if row.get('phase') not in ('train','holdout'):
        raise ValueError('only independent train/holdout calibration rows are allowed')
    for key in ('row_id','engine_id','rollout_id','cache_salt'):
        if not isinstance(row.get(key),str) or not row[key]:
            raise ValueError('missing calibration provenance: '+key)
    if type(row.get('step_index')) is not int or row['step_index']<0 or type(row.get('warmup')) is not bool:
        raise ValueError('invalid calibration step/warmup provenance')
    timing=row.get('timing',{})
    if timing.get('completed_model_step') is not True:
        raise ValueError('a completed model step is required for calibration')
    if timing.get('boundary')!=TIMING_BOUNDARY:
        raise ValueError('calibration timing boundary mismatch')
    if any(not _finite(timing.get(key)) for key in ('start_s','end_s','batch_execution_time')):
        raise ValueError('finite engine timing is required')
    start,end,duration=(timing[key] for key in ('start_s','end_s','batch_execution_time'))
    if start<0 or end<=start or duration<=0 or not math.isclose(end-start,duration,rel_tol=1e-9,abs_tol=1e-6):
        raise ValueError('engine timing duration/units mismatch')
    requests=row.get('requests')
    if not isinstance(requests,list) or not 1<=len(requests)<=min(4,identity['concurrency']):
        raise ValueError('nonempty hard-C4 step geometry required')
    features=dict.fromkeys(FEATURES,0)
    seen=set()
    for request in requests:
        rid=request.get('request_id')
        if not isinstance(rid,str) or not rid or rid in seen:
            raise ValueError('duplicate or missing step request identity')
        seen.add(rid)
        for key in ('computed_before','prefill_end','scheduled_tokens','output_tokens_before'):
            if type(request.get(key)) is not int or request[key]<0:
                raise ValueError('nonnegative integer step geometry required')
        computed=request['computed_before'];boundary=request['prefill_end'];tokens=request['scheduled_tokens']
        if (tokens<=0 or not computed<=boundary<=identity['context_window']
                or computed+tokens>identity['context_window']):
            raise ValueError('invalid model-context step geometry')
        prefill=computed<boundary
        if request.get('stage')!=('prefill' if prefill else 'decode'):
            raise ValueError('step geometry stage differs from actual prefill boundary')
        if prefill:
            # Replaying already generated output can include the latest known
            # input token beyond num_tokens-1. It remains a prefill batch.
            limit=boundary+int(request['output_tokens_before']>0)
            if computed+tokens>limit:raise ValueError('prefill geometry exceeds known input')
            features['batch_num_prefill_tokens']+=tokens
            features['batch_prefill_context']+=computed
        else:
            if tokens!=1:raise ValueError('decode geometry must be synchronous MTP0')
            features['batch_num_decode_tokens']+=tokens
            features['batch_decode_context']+=computed+tokens
        features['batch_num_tokens']+=tokens
    proposed=row.get('proposed_total_tokens')
    if type(proposed) is not int or not features['batch_num_tokens']<=proposed<=4096:
        raise ValueError('actual step geometry exceeds proposed native token budget')
    supplied=row.get('features')
    if (not isinstance(supplied,dict) or set(supplied)!=set(FEATURES)
            or any(type(value) is not int for value in supplied.values()) or supplied!=features):
        raise ValueError('recorded geometry differs from actual request execution')
    return features
