"""Frozen, independently conditioned EB+ comparisons and policy reconstruction."""
import copy
import math

from inferx.ebplus_calibration import validate_plan


def comparison_cases(plan):
    validate_plan(plan)
    cases=[]
    for name in ['native','auto-default','forced-eb']:
        case=copy.deepcopy(plan['base_case'])
        case.update(id=name,async_scheduling=False,shutdown_timeout=30,num_gpu_blocks_override=98,
                    scheduler_class='inferx_ebplus_scheduler.'+
                    ('ObservedNativeScheduler' if name=='native' else 'EBPlusScheduler'))
        if name!='native':case['environment'].update(VLLM_PD_SCHEDULER_MODE='auto',VLLM_PD_K_MODE='ifr')
        if name=='forced-eb':case['environment']['VLLM_PD_SCHEDULER_MODE']='eb'
        cases.append(case)
    return cases


def comparison_specs(templates,group,label,*,kind):
    if group not in {'short-control','cold-low','resume-high'} or kind not in {'mixed','isolated-serial-C4'}:
        raise ValueError('unfrozen comparison group/kind')
    if kind=='mixed' and group=='short-control':raise ValueError('mixed requires full long')
    specs=[]
    names=['tool-24576']*24 if kind=='mixed' or group=='short-control' else []
    if group!='short-control':names.append(('cold' if group=='cold-low' else 'resume')+'-131072')
    for i,name in enumerate(names):
        t=templates[name]
        row=dict(id=f'{label}-{i}',prompt_id=name,payload=copy.deepcopy(t['payload']),expected_tool=t['expected_tool'])
        if kind=='mixed':row['offset_s']=2*i if name=='tool-24576' else 16
        else:row['producer']=0
        specs.append(row)
    return specs


def crossover_decision(x):
    if (any(not math.isfinite(v) for v in x.values() if isinstance(v,(float,int)))
        or x['mu_L']+x['mu_o']<=0 or not 0<x['theta0']<1 or x['N_obs']<1
        or x['delta']<0 or x['active_before'] not in {'mb','eb'}):
        raise ValueError('invalid original crossover inputs')
    total=x['mu_L']+x['mu_o'];r=x['mu_o']/total
    mb=x['mb_cost_a']+x['mb_cost_b']*r+x['mb_cost_c']*r*r
    eb=(x['beta_p']*x['mu_L']+x['beta_d']*x['mu_o'])/total
    lhs=mb-eb
    rhs=((x['alpha_p']-x['alpha_d']*math.log(1-x['theta0'])*x['mu_o'])/(x['theta0']*x['N_obs'])
         -x['alpha_mb']*(1+x['mu_o'])/x['N_obs'])/total
    active=x['active_before']
    if x['cooldown_before']==0:
        if lhs>rhs+x['delta']:active='eb'
        elif lhs<rhs-x['delta']:active='mb'
    return dict(r=r,beta_mb=mb,beta_eb=eb,lhs=lhs,rhs=rhs,margin=lhs-rhs,
                delta=x['delta'],expected_active=active)


def audit_policy(state,hardware,mb,*,delta):
    from inferx.ebplus_calibration import validate_policy_evidence
    validate_policy_evidence(state)
    if state.get('max_num_running_reqs')!=4 or state.get('num_gpu_blocks')!=98 or state.get('completed',0)<100:
        raise ValueError('policy capacity or actual completion count mismatch')
    if state['mode']=='eb':
        if state.get('active_scheduler')!='eb':raise ValueError('forced policy decision mismatch')
        return dict(expected_active='eb',scope='Author forced-exclusive policy; original phase execution observed')
    x=state['last_evaluation_inputs']
    expected=dict(hardware['parameters'],**{'mb_cost_'+k:v for k,v in mb['fit']['coefficients'].items()},delta=delta)
    if any(x.get(k)!=v for k,v in expected.items()):raise ValueError('policy calibration/knob mismatch')
    decision=crossover_decision(x)
    if decision['expected_active']!=state['active_scheduler']:raise ValueError('observed policy decision mismatch')
    return decision


def audit_native_runtime(runtime,plan,hardware,*,root):
    native=hardware['native_runtime'];case=plan['base_case']
    expected=dict(model=str(root/case['model']),context_window=131328,concurrency=4,max_active=4,
        chunk=case['chunk'],long_prefill_token_threshold=case['long_prefill_token_threshold'],
        async_scheduling=False,enforce_eager=True,cudagraph_mode='NONE',speculative=False,
        quantization='compressed-tensors',kv_dtype=case['kv_dtype'],mamba_cache_mode=case['mamba_cache_mode'],
        block_size=native['block_size'],num_gpu_blocks=native['num_gpu_blocks'],kv_groups=native['kv_groups'])
    if runtime!=expected:raise ValueError('actual native runtime differs from calibrated model/precision/capacity')
