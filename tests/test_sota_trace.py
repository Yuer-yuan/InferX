"""Only GPU kernel intervals establish P/D lifetime overlap, with union accounting."""
import pytest


def kernel(start,duration,stream,name='native'):
    return dict(cat='kernel',ph='X',ts=start,dur=duration,name=name,args={'stream':stream,'device':0})


def test_gpu_kernel_union_overlap_does_not_count_cpu_or_double_count_nested_work():
    from inferx.sota_trace import analyze_trace
    events=[kernel(0,10,10,'matmul_pk'),kernel(5,10,10,'gdn_pk'),kernel(8,4,20),
            dict(cat='cpu_op',ph='X',ts=0,dur=100,name='CPU',args={'stream':20})]
    r=analyze_trace(events,{'prefill':10,'decode':20})
    assert r['kernel_count']==3 and r['status']=='observed'
    assert r['roles']['prefill']['kernel_union_us']==15
    assert r['roles']['decode']['kernel_union_us']==4
    assert r['pd_kernel_lifetime_overlap_us']==4
    assert r['transformed_kernel_sum_us']==20 and r['all_kernel_sum_us']==24
    assert r['transformed_kernel_time_fraction']==pytest.approx(20/24)


def test_no_actual_two_stream_gpu_evidence_stays_unobserved():
    from inferx.sota_trace import analyze_trace
    r=analyze_trace([kernel(0,5,10,'a_pk')],{'prefill':10,'decode':20})
    assert r['status']=='unobserved' and r['pd_kernel_lifetime_overlap_us']==0
    r=analyze_trace([kernel(0,2,10),kernel(3,2,20)],{'prefill':10,'decode':20})
    assert r['status']=='observed' and r['pd_kernel_lifetime_overlap_us']==0


@pytest.mark.parametrize('events,mapping',[
    ([kernel(0,float('nan'),10)],{'prefill':10,'decode':20}),
    ([dict(cat='kernel',ph='X',ts=1,dur=2,name='a',args={})],{'prefill':10,'decode':20}),
    ([kernel(0,2,10)],{'prefill':10,'decode':10}),
    ([kernel(0,2,10),{**kernel(0,2,20),'args':{'stream':20,'device':1}}],{'prefill':10,'decode':20}),
])
def test_ambiguous_trace_is_rejected_instead_of_inventing_overlap(events,mapping):
    from inferx.sota_trace import analyze_trace
    with pytest.raises(ValueError):analyze_trace(events,mapping)
