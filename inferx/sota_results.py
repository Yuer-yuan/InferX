"""Strict evidence normalization for independent SOTA comparison waves."""
import hashlib
import json
import math
import statistics

from inferx.concurrency import distribution, reconstruct_tool_calls
from inferx.tuning import summarize_response


def audit_wave(specifications, records, span_s, *, kind):
    if kind not in {'mixed','isolated-serial-C4'}:
        raise ValueError('unknown measurement kind')
    if not math.isfinite(span_s) or span_s<=0:
        raise ValueError('positive measured span required')
    specs={s['id']:s for s in specifications}
    if len(specs)!=len(specifications) or len({r['id'] for r in records})!=len(records):
        raise ValueError('duplicate request identity')
    if set(specs)!={r['id'] for r in records}:
        raise ValueError('missing or unexpected requests')
    if kind=='mixed':
        if any(type(s.get('offset_s')) not in (int,float) or not math.isfinite(s['offset_s']) for s in specifications):
            raise ValueError('finite frozen arrivals required')
        offsets=lambda short:sorted(s['offset_s'] for s in specifications if (s['prompt_id']=='tool-24576')==short)
        if offsets(True)!=list(range(0,48,2)) or offsets(False)!=[16] or span_s<48:
            raise ValueError('frozen .5RPS/48s/long-at16 arrival schedule changed')
    if kind=='isolated-serial-C4':
        ordered=sorted(records,key=lambda r:r['dispatched_s'])
        if any('offset_s' in s or s.get('producer')!=0 for s in specifications):
            raise ValueError('serial solo producer required')
        if any(b['dispatched_s']<a['finished_s'] for a,b in zip(ordered,ordered[1:])):
            raise ValueError('serial solo dispatch overlapped')
    short=[];long=[];verified=[]
    for row in records:
        spec=specs[row['id']];summary=row['summary']
        if row['request']!=spec['payload']:
            raise ValueError('payload differs from frozen specification')
        digest=hashlib.sha256(json.dumps(row['request'],sort_keys=True).encode()).hexdigest()
        if row['request_sha256']!=digest or row['prompt_id']!=spec['prompt_id']:
            raise ValueError('payload hash or prompt identity mismatch')
        actual=summarize_response(row['events'],summary['wall_s'],summary['ttft_s'],'natural')
        if (not actual['ok'] or actual['truncated'] or not summary['ok'] or
            summary.get('truncated') or any(summary.get(k)!=actual[k] for k in
            ['prompt_tokens','completion_tokens','cached_tokens','finish_reason'])):
            raise ValueError('incomplete or inconsistent response summary')
        calls=reconstruct_tool_calls(row['events']);expected=spec['expected_tool']
        if (row.get('functional_ok') is not True or row.get('expected_tool')!=expected or
            len(calls)!=1 or calls[0]['name']!=expected['name'] or calls[0]['arguments']!=expected['arguments']):
            raise ValueError('strict reconstructed tool mismatch')
        if not (0<=row['scheduled_s']<=row['dispatched_s']<=row['finished_s']<=span_s):
            raise ValueError('invalid dispatch/completion timing')
        if kind=='mixed' and row['scheduled_s']!=spec['offset_s']:
            raise ValueError('frozen arrival changed')
        lag=row['dispatched_s']-row['scheduled_s']
        if (abs(row['latency_s']-(lag+actual['wall_s']))>1e-6 or
            abs(row['finished_s']-row['dispatched_s']-actual['wall_s'])>.1):
            raise ValueError('inconsistent latency')
        if actual['ttft_s'] is None or row['arrival_ttft_s'] is None:
            raise ValueError('missing observed TTFT')
        if abs(row['arrival_ttft_s']-(lag+actual['ttft_s']))>1e-6:
            raise ValueError('inconsistent TTFT')
        is_short=spec['prompt_id']=='tool-24576'
        if type(actual['prompt_tokens']) is not int or actual['prompt_tokens']<(24000 if is_short else 130000):
            raise ValueError('shortened input or missing actual prompt count')
        (short if is_short else long).append(row)
        verified.append(actual)
    if ((kind=='mixed' and (len(short),len(long))!=(24,1)) or
        (kind=='isolated-serial-C4' and (len(short),len(long)) not in {(24,0),(0,1)})):
        raise ValueError('incomplete full workload')
    tokens=sum(s['completion_tokens'] for s in verified)
    return dict(strict_requests=len(records),short_p95_s=distribution([r['latency_s'] for r in short])['p95'],
        short_ttft_p95_s=distribution([r['arrival_ttft_s'] for r in short])['p95'],
        long_s=long[0]['latency_s'] if long else None,long_ttft_s=long[0]['arrival_ttft_s'] if long else None,
        span_s=span_s,completion_tokens=tokens,output_tokens_per_s=tokens/span_s,
        completed_within_48s=sum(r['finished_s']<=48 for r in records),
        api_cached_tokens=[s['cached_tokens'] for s in verified],physical_prefill_tokens=None,
        actual_prompt_tokens=[s['prompt_tokens'] for s in verified],
        max_dispatch_lag_s=max(r['dispatch_lag_s'] for r in records))


def compare_lanes(waves,solos):
    isolated={(s['case'],s['group']):s for s in solos}
    if len(isolated)!=len(solos):raise ValueError('duplicate solo diagnostic identity')
    groups={}
    for wave in waves:groups.setdefault((wave['case'],wave['group']),[]).append(wave)
    output=[]
    for (case,group),repeats in sorted(groups.items()):
        if len(repeats)!=3 or {r['repeat'] for r in repeats}!={0,1,2}:
            raise ValueError('three independent repeat identities required')
        short_solo=isolated[(case,'short-control')]['short_p95_s'];long_solo=isolated[(case,group)]['long_s']
        if not short_solo or not long_solo:raise ValueError('positive matched solo required')
        median=lambda key:statistics.median(r[key] for r in repeats)
        short=median('short_p95_s');long=median('long_s')
        output.append(dict(case=case,group=group,independent_mixed_repeats=3,independent_solo_repeats=1,
            short_p95_s=short,long_s=long,short_solo_p95_s=short_solo,long_solo_s=long_solo,
            short_slowdown=short/short_solo,long_slowdown=long/long_solo,
            output_tokens_per_s=median('output_tokens_per_s'),completed_within_48s=median('completed_within_48s'),
            repeat_values=[{k:r[k] for k in ['repeat','short_p95_s','long_s','output_tokens_per_s','completed_within_48s']} for r in sorted(repeats,key=lambda r:r['repeat'])],
            scope='Median of three fresh mixed waves divided by one lane-specific serial-C4 solo diagnostic; no request pooling or enterprise SLA inferred.'))
    return output
