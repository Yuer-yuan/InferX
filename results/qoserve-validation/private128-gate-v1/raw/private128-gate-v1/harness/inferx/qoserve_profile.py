"""Identity and completed-step geometry contracts for QoServe calibration.

These checks reject mislabeled data. They do not establish that a GPU was used:
that additionally requires the owned run, pinned collector and runtime evidence.
"""
import math
import copy
import statistics


FEATURES=('batch_num_tokens','batch_num_prefill_tokens','batch_num_decode_tokens',
          'batch_decode_context','batch_prefill_context')
TIMING_BOUNDARY='schedule-entry-to-native-update-exit'


def proposed_profile_budget(config,decoders,step_index,ceiling=4096):
    """Calibration proposals; phase labels never change to enable cycling."""
    if (type(decoders) is not int or not 0<=decoders<=4 or type(step_index) is not int
            or step_index<0 or type(ceiling) is not int or not 1<=ceiling<=4096):
        raise ValueError('invalid profiling budget counters')
    phase=config['phase']
    if phase not in ('train','holdout'):raise ValueError('invalid calibration phase')
    mode=config.get('budget_mode','prefill-target' if phase=='train' else 'total-cycle')
    if mode=='prefill-target' and phase=='train':proposed=config['prefill_target']+decoders
    elif mode=='total-cycle':proposed=config['candidates'][step_index%len(config['candidates'])]
    else:raise ValueError('invalid profiling budget mode')
    if type(proposed) is not int or proposed<=0:raise ValueError('invalid proposed budget')
    return min(ceiling,proposed)


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


def _training_bin(row):
    total=row['features']['batch_num_tokens']
    if total<=512:return 'tokens_0_512'
    if total>513:return 'tokens_513_plus'
    return None  # This gap is in the original training script.


def _runtime_bin(row):
    return 'tokens_0_512' if row['features']['batch_num_tokens']<=512 else 'tokens_513_plus'


CONTEXT_BANDS=('0-4K','4-24K','24-64K','64-128K')


def _context_band(row):
    contexts=[r['computed_before'] for r in row['requests'] if r['stage']=='prefill']
    if not contexts:return None
    context=max(contexts)
    return CONTEXT_BANDS[sum(context>=edge for edge in (4096,24576,65536))]


def _coverage_signature(row):
    return (_training_bin(row),row['proposed_total_tokens'],
            row['features']['batch_num_decode_tokens'],_context_band(row))


def fit_profile(rows,holdout,plan):
    """Fit the pinned author's OLS; independently gate, never repair with holdout.

    A returned unqualified result is diagnostic evidence, not a deployable model.
    Scientific dependencies are deliberately lazy and isolated from vLLM.
    """
    rows=list(rows);holdout=list(holdout);identity=plan['identity'];protocol=plan['fit']
    expected=dict(min_train_per_bin=100,min_holdout_per_bin=30,min_holdout_per_candidate=5,
        guard=1.2,minimum_guard_coverage=.95,source_cutoff_s=1.,compatibility_cutoff_s=5.,
        random_state=42,test_size=.2)
    if protocol!=expected:raise ValueError('fit protocol differs from predeclared source/qualification rules')
    seen=set()
    for phase,records in (('train',rows),('holdout',holdout)):
        for row in records:
            validate_step_row(row,identity)
            if row['phase']!=phase:raise ValueError('formal data or incorrect phase entered calibration')
            if row.get('calibration_kind',phase)!=phase:
                raise ValueError('diagnostic/throughput-only rows cannot train or qualify the predictor')
            if row['row_id'] in seen:raise ValueError('duplicate calibration row identity')
            seen.add(row['row_id'])
    for field in ('cache_salt','rollout_id'):
        if {r[field] for r in rows}&{r[field] for r in holdout}:
            raise ValueError('holdout contamination: '+field)
    known_protocols=['original-cycle-v1','remainder-cycle-v1']
    tagged_protocols={r['holdout_protocol'] for r in holdout if 'holdout_protocol' in r}
    if not tagged_protocols.issubset(known_protocols):raise ValueError('unknown holdout protocol')
    required_protocols=plan.get('required_holdout_protocols')
    if required_protocols is None:
        required_protocols=(known_protocols if 'remainder-cycle-v1' in tagged_protocols else
                            ['original-cycle-v1'] if tagged_protocols else [])
    if required_protocols not in ([],['original-cycle-v1'],known_protocols):
        raise ValueError('invalid required holdout protocols')
    if 'remainder-cycle-v1' in tagged_protocols and required_protocols!=known_protocols:
        raise ValueError('extended-cycle qualification must retain the original-cycle gate')
    warmed=[r for r in rows if not r['warmup']]
    source=[r for r in warmed if r['timing']['batch_execution_time']<=1. and _training_bin(r)]
    compatible=[r for r in warmed if r['timing']['batch_execution_time']<=5. and _training_bin(r)]
    missing={_coverage_signature(r) for r in compatible}-{_coverage_signature(r) for r in source}
    bin_names=('tokens_0_512','tokens_513_plus')
    recovered_counts=[name for name in bin_names if sum(_training_bin(r)==name for r in source)<100
                      <=sum(_training_bin(r)==name for r in compatible)]
    fallback=bool(missing or recovered_counts);cutoff=5. if fallback else 1.
    eligible=compatible if fallback else source
    result=dict(identity=identity,qualified=False,qualification_reasons=[],cutoff_s=cutoff,
        fit_protocol=dict(algorithm='sklearn.linear_model.LinearRegression',test_size=.2,random_state=42,
            training_upper_excludes_513=True,runtime_upper_starts_at_513=True),
        fallback=dict(activated=fallback,source_censored_row_ids=[r['row_id'] for r in warmed
            if r['timing']['batch_execution_time']>1.],recovered_bin_counts=recovered_counts,
            missing_geometry_context_signatures=sorted([list(s) for s in missing],key=repr)),
        training_exclusions={'warmup':[r['row_id'] for r in rows if r['warmup']],
            'source-edge-513':[r['row_id'] for r in warmed if _training_bin(r) is None],
            'above-cutoff':[r['row_id'] for r in warmed if r['timing']['batch_execution_time']>cutoff]},
        holdout_warmup_exclusions=[r['row_id'] for r in holdout if r['warmup']],
        coefficients={},bins={},candidates={},prediction_failures=[],deployment_prediction_failures=[])
    reasons=result['qualification_reasons']
    import numpy as np
    import sklearn
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    result['scientific_versions']=dict(numpy=np.__version__,sklearn=sklearn.__version__)
    models={}
    for name in bin_names:
        records=[r for r in eligible if _training_bin(r)==name]
        info=dict(eligible_training_rows=len(records),training_context_bands=sorted({_context_band(r)
            for r in records if _context_band(r) is not None}))
        result['bins'][name]=info
        if len(records)<100:
            reasons.append(name+' training coverage below 100');continue
        x=np.array([[r['features'][f] for f in FEATURES] for r in records],dtype=float)
        y=np.array([r['timing']['batch_execution_time'] for r in records])
        fit_indices,test_indices=train_test_split(np.arange(len(records)),test_size=.2,random_state=42)
        model=LinearRegression();model.fit(x[fit_indices],y[fit_indices]);models[name]=model
        predicted=model.predict(x[test_indices]);errors=predicted-y[test_indices]
        info.update(rank=int(model.rank_),singular_values=[float(v) for v in model.singular_],
            fit_row_ids=[records[i]['row_id'] for i in fit_indices],
            internal_test_row_ids=[records[i]['row_id'] for i in test_indices],
            internal_test=dict(count=len(test_indices),rmse_s=float(np.sqrt(np.mean(errors**2))),
                max_absolute_error_s=float(np.max(np.abs(errors))),
                r2=float(model.score(x[test_indices],y[test_indices])) if np.var(y[test_indices])>0 else None))
        result['coefficients'][name]=dict(coefficients={f:float(c) for f,c in zip(FEATURES,model.coef_)},
                                         intercept=float(model.intercept_))
        if set(info['training_context_bands'])!=set(CONTEXT_BANDS):reasons.append(name+' training context coverage incomplete')
    held=[r for r in holdout if not r['warmup']]
    predictions={}
    for row in warmed+held:
        name=_runtime_bin(row)
        if name not in models:continue
        predicted=float(models[name].predict([[row['features'][f] for f in FEATURES]])[0])
        if not math.isfinite(predicted) or predicted<=0:
            result['prediction_failures'].append(row['row_id'])
        if row['phase']=='holdout':predictions[row['row_id']]=predicted
    def metrics(records):
        complete=[r for r in records if r['row_id'] in predictions]
        if not complete:return dict(count=len(records),predicted_count=0,guard_coverage=0.)
        actual=np.array([r['timing']['batch_execution_time'] for r in complete])
        predicted=np.array([predictions[r['row_id']] for r in complete])
        covered=np.isfinite(predicted)&(predicted>0)&(predicted*1.2>=actual)
        return dict(count=len(records),predicted_count=len(complete),guard_coverage=float(covered.sum()/len(records)),
            rmse_s=float(np.sqrt(np.mean((predicted-actual)**2))),
            mean_absolute_relative_error=float(np.mean(np.abs(predicted-actual)/actual)),
            p95_prediction_ratio=float(np.quantile(predicted/actual,.95)),
            overprediction_above_2x_fraction=float(np.mean(predicted>2*actual)),
            row_ids=[r['row_id'] for r in complete],predictions_s=[float(v) for v in predicted],
            actual_s=[float(v) for v in actual])
    for name in bin_names:
        records=[r for r in held if _runtime_bin(r)==name];info=result['bins'][name]
        info['holdout']=metrics(records)
        bands=sorted({_context_band(r) for r in records if _context_band(r) is not None})
        info['holdout_context_bands']=bands
        if len(records)<30:reasons.append(name+' holdout coverage below 30')
        if info['holdout']['guard_coverage']<.95:reasons.append(name+' holdout guard below 95%')
        if set(bands)!=set(CONTEXT_BANDS):reasons.append(name+' holdout context coverage incomplete')
    for candidate in plan['profile']['candidates']:
        records=[r for r in held if r['proposed_total_tokens']==candidate]
        info=metrics(records);result['candidates'][str(candidate)]=info
        if len(records)<5:reasons.append(f'candidate {candidate} holdout coverage below 5')
        if info['guard_coverage']<.95:reasons.append(f'candidate {candidate} guard below 95%')
        # Check the counterfactual numerical domain queried by the source search.
        for decoders in range(4):
            for context in (0,4096,24576,65536,131072):
                for decoder_context in (24558,25600,131072):
                    features=(candidate,candidate-decoders,decoders,decoders*decoder_context,context)
                    name='tokens_0_512' if candidate<=512 else 'tokens_513_plus'
                    if name not in models:continue
                    predicted=float(models[name].predict([features])[0])
                    if not math.isfinite(predicted) or predicted<=0:
                        result['deployment_prediction_failures'].append(dict(features=list(features),prediction_s=predicted))
    if required_protocols:
        result['required_holdout_protocols']=required_protocols
        result['holdout_protocols']={}
        for protocol_name in required_protocols:
            subset=[r for r in held if r.get('holdout_protocol','original-cycle-v1')==protocol_name]
            group=dict(bins={},candidates={});result['holdout_protocols'][protocol_name]=group
            for name in bin_names:
                records=[r for r in subset if _runtime_bin(r)==name];info=metrics(records)
                info['context_bands']=sorted({_context_band(r) for r in records if _context_band(r) is not None})
                group['bins'][name]=info
                if len(records)<30:reasons.append(f'protocol {protocol_name} bin {name} coverage below 30')
                if info['guard_coverage']<.95:reasons.append(f'protocol {protocol_name} bin {name} guard below 95%')
            for candidate in plan['profile']['candidates']:
                records=[r for r in subset if r['proposed_total_tokens']==candidate];info=metrics(records)
                group['candidates'][str(candidate)]=info
                if len(records)<5:reasons.append(f'protocol {protocol_name} candidate {candidate} coverage below 5')
                if info['guard_coverage']<.95:reasons.append(f'protocol {protocol_name} candidate {candidate} guard below 95%')
    if result['prediction_failures'] or result['deployment_prediction_failures']:
        reasons.append('nonpositive or nonfinite observed/deployment predictions')
    result['qualified']=not reasons
    return result


def freeze_common_budgets(calibration,identity):
    """Check labels and derive fixed proxy budgets from separately owned solos.

    The caller must additionally archive actual model/runtime and strict tool
    evidence. This function does not turn an asserted record into GPU proof.
    """
    if (calibration.get('identity')!=identity or calibration.get('phase')!='independent-native-solo'
            or calibration.get('strategy')!='plain-native'):
        raise ValueError('common SLOs require independent matched native calibration')
    if any(len(calibration.get(key,[]))!=count for key,count in (('short',24),('cold',3),('resume',3))):
        raise ValueError('common SLO calibration needs 24 short and 3+3 long solos')
    seen=set();intervals=[]
    for kind in ('short','cold','resume'):
        for row in calibration[kind]:
            rid=row.get('request_id')
            if not isinstance(rid,str) or not rid or rid in seen:raise ValueError('solo request join is not unique')
            seen.add(rid)
            if row.get('strict_ok') is not True:raise ValueError('solo strict completion gate failed')
            if any(not _finite(row.get(k)) or row[k]<=0 for k in ('ttft_s','latency_s')):
                raise ValueError('solo timing must be finite and positive')
            times=row.get('model_token_times_s',[])
            if (len(times)<2 or any(not _finite(t) or t<=0 for t in times)
                    or any(b<=a for a,b in zip(times,times[1:]))
                    or times[-1]>row['latency_s']+1e-6 or row['ttft_s']>row['latency_s']):
                raise ValueError('solo model token times must describe completed ordered outputs')
            if kind=='short':intervals.extend(b-a for a,b in zip(times,times[1:]))
    def p95(values):
        values=sorted(values);return values[math.ceil(.95*len(values))-1]
    short=calibration['short'];ttft=p95([r['ttft_s'] for r in short]);tbt=p95(intervals)
    e2e=p95([r['latency_s'] for r in short])
    return dict(identity=identity,label='independent proxy SLOs; not enterprise SLAs',
        quantile='sorted[ceil(0.95*n)-1]',source_request_ids={k:[r['request_id'] for r in calibration[k]]
            for k in ('short','cold','resume')},
        independent_native=dict(short_ttft_p95_s=ttft,short_model_tbt_p95_s=tbt,short_e2e_p95_s=e2e,
            long_median_s={k:statistics.median(r['latency_s'] for r in calibration[k]) for k in ('cold','resume')}),
        interactive=dict(ttft_s=1.5*ttft,tbt_s=1.5*tbt,practical_e2e_s=1.5*e2e),
        long={k:dict(ttlt_s=1.25*statistics.median(r['latency_s'] for r in calibration[k]),tbt_s=4*1.5*tbt)
              for k in ('cold','resume')})


def attach_common_contract(payload,budgets,*,tier,kind,trace_id):
    """Add only the four numeric caller tags understood by the pinned adapter."""
    from inferx.qoserve_validation import parse_contract
    if type(tier) is not int or tier not in (0,1):raise ValueError('caller tier must be 0 or 1')
    if kind not in ('cold','resume'):raise ValueError('unknown long workload group')
    policy=budgets['interactive'] if tier==0 else budgets['long'][kind]
    tags=dict(qoserve_tier=tier,qoserve_deadline_s=policy['ttft_s' if tier==0 else 'ttlt_s'],
              qoserve_tbt_s=policy['tbt_s'],qoserve_trace_id=trace_id)
    parse_contract(tags)
    result=copy.deepcopy(payload);extra=result.setdefault('vllm_xargs',{})
    if not isinstance(extra,dict) or any(key in extra for key in tags):
        raise ValueError('payload already contains caller contract tags or invalid vllm_xargs')
    extra.update(tags)
    return result
