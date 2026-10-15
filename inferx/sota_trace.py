"""Conservative CUDA trace evidence; kernel lifetime overlap is not SM occupancy."""
import math


def _number(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError('finite trace number required')
    return value


def _union(intervals):
    result=[]
    for start,end in sorted(intervals):
        if end==start:continue
        if result and start<=result[-1][1]:result[-1]=(result[-1][0],max(result[-1][1],end))
        else:result.append((start,end))
    return result


def _overlap(left,right):
    total=0;i=j=0
    while i<len(left) and j<len(right):
        a,b=left[i];c,d=right[j];total+=max(0,min(b,d)-max(a,c))
        if b<=d:i+=1
        else:j+=1
    return total


def analyze_trace(events,stream_ids):
    if set(stream_ids)!={'prefill','decode'} or any(type(v) is not int for v in stream_ids.values()) or len(set(stream_ids.values()))!=2:
        raise ValueError('distinct measured P/D stream IDs required')
    kernels=[];devices=set();copies=[]
    for event in events:
        if event.get('cat')=='gpu_memcpy' and event.get('ph')=='X':
            duration=_number(event['dur'])
            if duration<0:raise ValueError('negative copy duration')
            copies.append(duration)
        if event.get('cat')!='kernel' or event.get('ph')!='X':continue
        args=event.get('args',{})
        if type(args.get('stream')) is not int or not isinstance(event.get('name'),str):
            raise ValueError('GPU kernel stream/name absent')
        start=_number(event['ts']);duration=_number(event['dur'])
        if duration<0:raise ValueError('negative kernel duration')
        devices.add(args.get('device'))
        kernels.append(dict(start=start,end=start+duration,duration=duration,stream=args['stream'],name=event['name']))
    if len(devices)>1:raise ValueError('trace mixes GPU device identities')
    roles={};unions={}
    for role,stream in stream_ids.items():
        selected=[k for k in kernels if k['stream']==stream];unions[role]=_union([(k['start'],k['end']) for k in selected])
        roles[role]=dict(stream_id=stream,kernel_count=len(selected),kernel_union_us=sum(b-a for a,b in unions[role]),kernel_sum_us=sum(k['duration'] for k in selected))
    transformed=[k for k in kernels if k['name'].endswith('_pk')]
    total=sum(k['duration'] for k in kernels);transformed_us=sum(k['duration'] for k in transformed)
    all_union=_union([(k['start'],k['end']) for k in kernels]);union_us=sum(b-a for a,b in all_union)
    envelope=all_union[-1][1]-all_union[0][0] if all_union else 0
    costs={}
    for k in kernels:
        cost=costs.setdefault(k['name'],dict(name=k['name'],count=0,sum_us=0,transformed=k['name'].endswith('_pk')))
        cost['count']+=1;cost['sum_us']+=k['duration']
    return dict(status='observed' if all(roles[r]['kernel_count'] for r in roles) else 'unobserved',
        scope='GPU kernel lifetime overlap; does not establish instantaneous SM occupancy or measured memory bandwidth. Profiled lane excluded from latency comparisons.',
        kernel_count=len(kernels),device=next(iter(devices)) if devices else None,roles=roles,
        kernel_lifetime_union_us=union_us,kernel_envelope_us=envelope,
        non_kernel_time_within_envelope_us=envelope-union_us,
        gap_scope='No recorded kernel during these intervals; not proof of GPU idle or a host-control root cause. Copies and other GPU operations may still run.',
        kernel_costs=sorted(costs.values(),key=lambda row:(-row['sum_us'],row['name'])),
        pd_kernel_lifetime_overlap_us=_overlap(unions['prefill'],unions['decode']),
        all_kernel_sum_us=total,transformed_kernel_count=len(transformed),transformed_kernel_sum_us=transformed_us,
        transformed_kernel_time_fraction=transformed_us/total if total else None,
        transformed_name_rule='Actual compiler-generated raw kernel name suffix _pk only; no inferred transformation of native libraries.',
        memcpy_event_count=len(copies),memcpy_sum_us=sum(copies),
        by_stream={str(s):dict(kernel_count=sum(k['stream']==s for k in kernels),kernel_sum_us=sum(k['duration'] for k in kernels if k['stream']==s)) for s in sorted({k['stream'] for k in kernels})})
