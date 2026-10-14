"""Recompute compact observations from explicit raw campaign inputs only."""
import argparse
import csv
import hashlib
import json
import sqlite3
from pathlib import Path

from inferx.concurrency import summarize_load, summarize_native


def summarize_cuda_timeline(path: Path) -> dict:
    connection=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)
    try:
        tables={row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'CUPTI_ACTIVITY_KIND_KERNEL' not in tables:
            return dict(status='no_kernel_table')
        intervals=connection.execute('SELECT start,end,deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start').fetchall()
    finally:
        connection.close()
    if not intervals:
        return dict(status='no_kernels')
    if any(end<=start for start,end,_ in intervals):
        raise ValueError('invalid kernel interval')
    duration=sum(end-start for start,end,_ in intervals)
    union=0
    first,tail=intervals[0][:2]
    left,right=first,tail
    for start,end,_ in intervals[1:]:
        if start>right:
            union+=right-left;left,right=start,end
        else:
            right=max(right,end)
    union+=right-left
    envelope=right-first
    return dict(status='observed',kernel_count=len(intervals),device_ids=sorted({x[2] for x in intervals}),
        kernel_duration_sum_ns=duration,kernel_union_ns=union,kernel_envelope_ns=envelope,
        no_kernel_fraction=1-union/envelope,
        limitation='Kernel envelope includes warmup; gaps can include copies and are not automatically GPU idle. No DRAM counters.')


def export(root: Path, destination: Path, allow_running: bool = False) -> dict:
    root, destination = Path(root), Path(destination)
    sources = {}
    def read(path, lines=False):
        path = root/path
        data = path.read_bytes()
        sources[str(path.relative_to(root))] = dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
        return [json.loads(line) for line in data.splitlines() if line.strip()] if lines else json.loads(data)
    grid, campaign = read('grid.json'), read('campaign.json')
    manifest = read('workload-manifest.json')
    for entry in manifest['files']:
        path = root/entry['file']
        if path.parent != root or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('frozen workload hash mismatch')
        sources[entry['file']] = dict(sha256=entry['sha256'],bytes=path.stat().st_size)
    if campaign['status'] == 'running' and not allow_running:
        raise ValueError('campaign is still running')
    configurations = grid['configs']
    ids = [c['id'] for c in configurations]
    observed_ids = [c['configuration']['id'] for c in campaign['cases']]
    if len(set(ids)) != len(ids) or len(set(observed_ids)) != len(observed_ids):
        raise ValueError('duplicate case identity')
    if not set(observed_ids) <= set(ids):
        raise ValueError('unexpected case identity')
    cases, groups, requests, incomplete = [], [], [], []
    for configuration in configurations:
        name = configuration['id']
        if Path(name).name != name:
            raise ValueError('invalid case path')
        directory = root/name
        if not (directory/'case.json').exists():
            cases.append(dict(configuration=configuration,status='not_finished'))
        else:
            case = read(Path(name)/'case.json')
            if case['configuration'] != configuration:
                raise ValueError('case configuration mismatch')
            if (directory/'quality-result.json').exists():
                case['quality_gate'] = read(Path(name)/'quality-result.json')
            cases.append(case)
        if not directory.exists():
            continue
        # Only immediate children of frozen case directories. Derived exports are never scanned.
        for path in sorted(directory.iterdir()):
            if not path.is_dir() or not any((path/f).exists() for f in ['result.json','requests.jsonl','warmup.jsonl']):
                continue
            relative = path.relative_to(root)
            rows = read(relative/'requests.jsonl',True) if (path/'requests.jsonl').exists() else []
            if len({r['id'] for r in rows}) != len(rows):
                raise ValueError('duplicate request identity')
            warm = read(relative/'warmup.jsonl',True) if (path/'warmup.jsonl').exists() else []
            native = read(relative/'native.jsonl',True) if (path/'native.jsonl').exists() else []
            if not (path/'result.json').exists():
                incomplete.append(dict(case=name,group=path.name,warmup_attempts=len(warm),measurement_attempts=len(rows)))
                result = dict(group=path.name,case=name,status='incomplete')
            else:
                result = read(relative/'result.json')
                if result['case'] != name or result['group'] != path.name:
                    raise ValueError('group identity mismatch')
                if result['requested'] != len(rows):
                    raise ValueError('request count mismatch')
                result['summary'] = summarize_load(rows,result['summary']['measured_span_s'])
                result['native'] = summarize_native(native)
                result['status'] = 'completed'
                result['classes'] = {}
                class_rows = {}
                for kind in sorted({(r.get('prompt_id') or 'unknown').split('-')[0] for r in rows}):
                    subset = [r for r in rows if (r.get('prompt_id') or 'unknown').split('-')[0]==kind]
                    class_rows[kind] = subset
                    result['classes'][kind] = summarize_load(subset,result['summary']['measured_span_s'])
                sizes = {kind:{r['summary'].get('prompt_tokens') for r in subset
                               if r['summary']['ok'] and r['summary'].get('prompt_tokens') is not None}
                         for kind,subset in class_rows.items()}
                result['class_native'] = {}
                for kind, subset in class_rows.items():
                    disjoint = all(not sizes[kind].intersection(other) for key,other in sizes.items() if key!=kind)
                    matched = [e for e in native if e.get('event')=='request_done'
                               and e['result']['prompt_tokens'] in sizes[kind]]
                    valid = disjoint and len(matched)==sum(r['summary']['ok'] for r in subset) and bool(matched)
                    result['class_native'][kind] = dict(matching='disjoint_prompt_usage_sizes' if valid else 'unavailable_or_ambiguous',
                        observations=summarize_native(matched) if valid else None)
                result['native_count_audit'] = dict(client_attempts=len(rows),client_protocol_ok=result['summary']['protocol_ok'],
                    native_done=result['native']['request_done'],native_errors=result['native']['request_errors'])
                result['warmup_protocol_ok'] = sum(r['summary']['ok'] for r in warm)
                groups.append(result)
            for phase, observations in [('measurement',rows),('warmup',warm)]:
                for row in observations:
                    summary=row['summary']
                    requests.append(dict(case=name,group=path.name,phase=phase,id=row['id'],
                        prompt_id=row.get('prompt_id'),producer=row.get('producer'),ok=summary['ok'],
                        functional_ok=row.get('functional_ok'),errors=summary.get('errors',[]),
                        truncated=summary.get('truncated'),prompt_tokens=summary.get('prompt_tokens'),
                        completion_tokens=summary.get('completion_tokens'),latency_s=row.get('latency_s'),
                        ttft_s=row.get('arrival_ttft_s'),dispatch_lag_s=row.get('dispatch_lag_s')))
    bandwidth=read('bandwidth.json') if (root/'bandwidth.json').exists() else None
    profiles=[]
    for name in ['profile-d0-c1','profile-d7-c4']:
        path=Path('profiles')/name/'profile.json'
        if (root/path).exists():
            profile=read(path)
            sqlite=path.parent/'trace.sqlite'
            if (root/sqlite).exists():
                data=(root/sqlite).read_bytes()
                sources[str(sqlite)]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
                try:
                    profile['kernel_timeline']=summarize_cuda_timeline(root/sqlite)
                except (ValueError,sqlite3.Error) as exc:
                    profile['kernel_timeline']=dict(status='analysis_error',error=str(exc))
            for csv_path in (root/path.parent).glob('stats_*.csv'):
                data=csv_path.read_bytes()
                sources[str(csv_path.relative_to(root))]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
            profiles.append(profile)
    restored=read('restored-service.json') if (root/'restored-service.json').exists() else None
    result=dict(campaign=campaign,grid=grid,cases=cases,groups=groups,requests=requests,
        incomplete_groups=incomplete,bandwidth=bandwidth,profiles=profiles,restored_service=restored,
        roofline=dict(classification='undetermined',
            reason='No valid inference-kernel DRAM traffic counters; copy bandwidth and utilization are insufficient.',
            counter_permission=campaign.get('hardware_counter_permission')),
        source_hashes=sources,
        limitations=['Synthetic serving snapshots, not live agent accuracy.',
            'Request timers overlap; sums do not represent GPU wall time.',
            'P95 requires20 and P99 requires100 successful observations.',
            'Configured concurrency and warmup do not prove batching or cache hits.',
            'C1 mixed-context is serial grouped order; C8/near256 have separate memory budgets.'])
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (destination/'requests.csv').open('w',newline='') as output:
        fields=['case','group','phase','id','prompt_id','producer','ok','functional_ok','errors','truncated',
                'prompt_tokens','completion_tokens','latency_s','ttft_s','dispatch_lag_s']
        writer=csv.DictWriter(output,fieldnames=fields);writer.writeheader();writer.writerows(requests)
    with (destination/'groups.csv').open('w',newline='') as output:
        fields=['case','group','offered','errors','tool_valid','aggregate_tokens_per_s','latency_median_s',
            'latency_p95_s','latency_p99_s','ttft_median_s','cached_fraction','computed_prefill_tokens',
            'acceptance','accepted_per_round','realized_batch_size']
        writer=csv.DictWriter(output,fieldnames=fields);writer.writeheader()
        for group in groups:
            s,n=group['summary'],group['native']
            writer.writerow(dict(case=group['case'],group=group['group'],
                **{k:s[k] for k in ['offered','errors','tool_valid','aggregate_tokens_per_s']},
                latency_median_s=s['latency_s']['median'],latency_p95_s=s['latency_s']['p95'],
                latency_p99_s=s['latency_s']['p99'],ttft_median_s=s['ttft_s']['median'],
                cached_fraction=n['cached_tokens']/n['prompt_tokens'] if n['prompt_tokens'] else None,
                **{k:n[k] for k in ['computed_prefill_tokens','acceptance','accepted_per_round','realized_batch_size']}))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('destination',type=Path)
    parser.add_argument('--allow-running',action='store_true')
    args=parser.parse_args()
    result=export(args.root,args.destination,args.allow_running)
    print(json.dumps(dict(cases=len(result['cases']),groups=len(result['groups']),
                         incomplete_groups=len(result['incomplete_groups']),requests=len(result['requests']),
                         campaign_status=result['campaign']['status'])))


if __name__=='__main__':
    main()
