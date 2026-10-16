"""Own-model EB+ calibration contracts; no generic/sample hardware constants."""
from collections import Counter
import math

from inferx.ebplus_validation import NATIVE_SHA256


def make_calibration_plan(case):
    identity={k:case[k] for k in ['model','model_revision','gpu_uuid','quantization','kv_dtype',
                                  'context_window','concurrency','draft','cuda_graph','wheel_sha256']}
    identity['native_scheduler_sha256']=NATIVE_SHA256
    plan=dict(identity=identity,base_case=case,
              decomposed=dict(prefill_sizes=[256,512,1024,1568,3136],decode_counts=[1,2,4],
                              decode_context_len=512,num_warmup=5,num_iterations=20,sampling_temperature=0.0),
              mb_grid=dict(counts=[1,2,4],budgets=[1024,2048,4096],prefill_caps=[0,2048],
                           scenarios=[dict(name='decode-heavy',input=128,output=1024),
                                      dict(name='balanced',input=512,output=512),
                                      dict(name='prefill-heavy',input=1024,output=128)]),
              scope='Author decomposed GPU-event calibration and separate inverse-best-total-token-throughput MB fit. Calibration does not use formal mixed outcomes. Context512 is author protocol; long-context model-fit validation remains explicit.')
    validate_plan(plan)
    return plan


def validate_plan(plan):
    identity=plan['identity'];stage=plan['decomposed'];grid=plan['mb_grid']
    if (identity.get('native_scheduler_sha256')!=NATIVE_SHA256 or
        any(plan['base_case'].get(key)!=value for key,value in identity.items()
            if key!='native_scheduler_sha256')):
        raise ValueError('plan runtime/model/GPU identity mismatch')
    if identity['concurrency']!=4 or identity['draft']!=0 or identity['cuda_graph'] is not False:
        raise ValueError('calibration requires C4/MTP0/eager')
    for counts in (stage['decode_counts'],grid['counts']):
        if counts!=[1,2,4]:raise ValueError('calibration requires feasible C4 counts1/2/4')
    sizes=stage['prefill_sizes']
    if not sizes or any(type(n) is not int or n<=1 or n>plan['base_case']['chunk'] for n in sizes):
        raise ValueError('prefill must complete in one calibrated native step')
    if stage['num_warmup']<1 or stage['num_iterations']<2:
        raise ValueError('calibration needs warmup and independent samples')
    if stage['decode_context_len']+100000>=identity['context_window']:
        raise ValueError('decode preparation exceeds model context')
    if any(b<=0 or b>plan['base_case']['chunk'] for b in grid['budgets']):
        raise ValueError('invalid MB budget')
    for scene in grid['scenarios']:
        if min(scene['input'],scene['output'])<=0 or scene['input']+scene['output']>identity['context_window']:
            raise ValueError('infeasible MB geometry')


def validate_hardware_result(result,plan):
    validate_plan(plan)
    if result['identity']!=plan['identity']:raise ValueError('calibration identity mismatch')
    if result['measurement_type']!='decomposed_cuda_events':
        raise ValueError('actual GPU completion-event timing required')
    for name,value in result['parameters'].items():
        if name not in {'alpha_p','beta_p','alpha_d','beta_d'} or not math.isfinite(value) or value<0:
            raise ValueError('invalid hardware cost')
    if set(result['parameters'])!={'alpha_p','beta_p','alpha_d','beta_d'} or result['parameters']['alpha_d']<=0:
        raise ValueError('incomplete hardware parameters')
    stage=plan['decomposed']
    if 'sampling_temperature' in stage and (result.get('sampling_temperature')!=stage['sampling_temperature']
                                           or result.get('ignore_eos') is not True):
        raise ValueError('calibration sampler differs from deployment')
    if result['prefill_sizes']!=stage['prefill_sizes'] or result['decode_counts']!=stage['decode_counts']:
        raise ValueError('incomplete calibration geometries')
    counts=Counter()
    for row in result['raw_steps']:
        if not math.isfinite(row['elapsed_ms']) or row['elapsed_ms']<=0:
            raise ValueError('invalid GPU duration')
        if row['stage']=='prefill':
            if row['num_prefill']!=1 or row['num_decode']!=0 or row['tokens']!=row['target']:
                raise ValueError('prefill measurement truncated or mixed')
        elif row['stage']=='decode':
            if row['num_prefill']!=0 or row['num_decode']!=row['target'] or row['tokens']!=row['target']:
                raise ValueError('decode batch not fully retained')
        else:raise ValueError('invalid calibration stage')
        counts[row['stage'],row['target']]+=1
    expected={(kind,n):stage['num_warmup']+stage['num_iterations']
              for kind,names in [('prefill',stage['prefill_sizes']),('decode',stage['decode_counts'])]
              for n in names}
    if dict(counts)!=expected:raise ValueError('missing raw GPU timing samples')


def require_restored(campaign,restored,current,*,expected_count):
    if (campaign.get('status')!='restored' or campaign.get('restored') is not True
        or len(campaign['cases'])!=expected_count or any(x['state']!='measured' for x in campaign['cases'])):
        raise ValueError('prerequisite campaign not restored and complete')
    require_terminal_restoration(campaign,restored,current)


def require_terminal_restoration(campaign,restored,current):
    """A failed owned attempt may be retried only after actual restoration."""
    if campaign.get('status') not in ('restored','failed') or campaign.get('restored') is not True:
        raise ValueError('previous campaign has not completed restoration')
    keys=['pid','proc_start_ticks','uid','executable']
    if any(restored.get(k)!=current.get(k) for k in keys):
        raise ValueError('restoration record is stale against current service')


def validate_policy_evidence(state):
    counts=state['execution_counts']
    if state['mode']=='auto':
        if (state['evaluations']<1 or state['parameters_initialized'] is not True
            or state['mb_cost_profiled'] is not True or sum(counts.values())<1):
            raise ValueError('actual calibrated auto policy evaluation not observed')
    elif state['mode']=='eb':
        if counts.get('prefill',0)<1 or counts.get('decode',0)<1:
            raise ValueError('exclusive prefill/decode policy not observed')
    else:raise ValueError('not an EB+ policy')


def fit_mb_cost(rows,plan):
    """Author Phase C: inverse best total TPS at each of three ratios."""
    validate_plan(plan);grid=plan['mb_grid']
    scenes={x['name']:x for x in grid['scenarios']}
    expected={(s,b,cap,n) for s in scenes for b in grid['budgets']
              for cap in grid['prefill_caps'] for n in grid['counts']}
    seen=set()
    for row in rows:
        key=row['scenario'],row['budget'],row['cap'],row['count']
        if key not in expected or key in seen:raise ValueError('invalid or duplicate MB grid point')
        seen.add(key);scene=scenes[row['scenario']];n=row['count']
        if (row['completed']!=n or row['prompt_tokens']!=n*scene['input']
            or row['completion_tokens']!=n*scene['output']):
            raise ValueError('incomplete MB tokens')
        duration=row['duration_s'];tps=row['total_token_throughput']
        if not all(math.isfinite(x) and x>0 for x in (duration,tps)):
            raise ValueError('invalid MB timing')
        if not math.isclose(tps,(row['prompt_tokens']+row['completion_tokens'])/duration,rel_tol=1e-9):
            raise ValueError('MB total tokens throughput mismatch')
    if seen!=expected:raise ValueError('missing MB grid points')
    points=[]
    for name,scene in scenes.items():
        best=max((row for row in rows if row['scenario']==name),key=lambda row:row['total_token_throughput'])
        points.append(dict(r=scene['output']/(scene['input']+scene['output']),
                           seconds_per_total_token=1/best['total_token_throughput'],best=best))
    if len(points)!=3 or len({p['r'] for p in points})!=3:
        raise ValueError('three independent MB ratios required')
    # Lagrange interpolation equals the author's full-rank 3x3 least-squares fit.
    a=b=c=0.
    for i,p in enumerate(points):
        others=[q['r'] for j,q in enumerate(points) if j!=i];u,v=others
        scale=p['seconds_per_total_token']/((p['r']-u)*(p['r']-v))
        a+=scale*u*v;b-=scale*(u+v);c+=scale
    return dict(coefficients=dict(a=a,b=b,c=c),points=points,
                env_vars={f'VLLM_PD_MB_COST_{key.upper()}':str(value)
                          for key,value in dict(a=a,b=b,c=c).items()},
                method='author inverse best total-token TPS quadratic, no clipping',
                limitation='Three points fit exactly; residual zero is not held-out or long-context validation.')


def qualify_tool_rows(records,templates,expected_prompt_ids):
    """Full-input qualification, reconstructed from original SSE and payloads."""
    from inferx.sota_results import prompt_count_contract
    from inferx.concurrency import reconstruct_tool_calls
    from inferx.tuning import summarize_response
    if Counter(r['prompt_id'] for r in records)!=Counter(expected_prompt_ids):
        raise ValueError('incomplete qualification requests')
    if len({r['id'] for r in records})!=len(records):raise ValueError('duplicate qualification request')
    counts=prompt_count_contract('vllm031')['prompt_tokens'];tokens=[]
    for row in records:
        template=templates[row['prompt_id']];summary=row['summary']
        if row['request']!=template['payload']:raise ValueError('qualification payload changed')
        actual=summarize_response(row['events'],summary['wall_s'],summary['ttft_s'],'natural')
        if not actual['ok'] or actual['truncated'] or actual['prompt_tokens']!=counts[row['prompt_id']]:
            raise ValueError('incomplete qualification input or response')
        if any(actual[k]!=summary[k] for k in ['prompt_tokens','completion_tokens','finish_reason','cached_tokens']):
            raise ValueError('inconsistent qualification summary')
        calls=reconstruct_tool_calls(row['events']);expected=template['expected_tool']
        if (len(calls)!=1 or calls[0]['name']!=expected['name'] or calls[0]['arguments']!=expected['arguments']):
            raise ValueError('qualification strict tool mismatch')
        tokens.append(actual['prompt_tokens'])
    return dict(strict_requests=len(records),actual_prompt_tokens=tokens)
