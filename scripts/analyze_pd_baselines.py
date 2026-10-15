"""Recompute PD observations from explicitly indexed raw cases, retaining failures."""
import argparse
import collections
import copy
import csv
import hashlib
import io
import json
import math
import statistics
from pathlib import Path

from inferx.concurrency import distribution, reconstruct_tool_calls, summarize_load


def sampled_gpu_memory(data):
    used, free, invalid = [], [], 0
    if data is not None:
        for row in csv.reader(io.StringIO(data.decode(errors='replace'))):
            try:
                if len(row) != 6:
                    raise ValueError('expected nvidia-smi six-column sample')
                values = float(row[1]), float(row[2])
                if not all(math.isfinite(v) and v >= 0 for v in values):
                    raise ValueError('invalid memory sample')
                used.append(values[0]); free.append(values[1])
            except ValueError:
                invalid += 1
    return dict(status=('partial' if invalid else 'observed') if used else 'unobserved',
        valid_samples=len(used), invalid_rows=invalid, sampling_interval_ms=200,
        sampled_peak_mib=max(used) if used else None,
        sampled_min_free_mib=min(free) if free else None,
        scope='Healthy server lifetime including quality/warmups/measurement; startup excluded. '
              '200 ms samples can miss short peaks; values cover the selected GPU device.')


def prefix_observations(rows):
    good=[r for r in rows if r['summary']['ok']]
    observed=[r for r in good if type(r['summary'].get('cached_tokens')) is int]
    all_known=bool(good) and len(observed)==len(good)
    return dict(status='observed' if all_known else 'partial' if observed else 'unobserved',
        scope='API prompt minus cached tokens; excludes possible internal preemption replay.',
        successful_requests=len(good),counter_observations=len(observed),
        cached_tokens=sum(r['summary']['cached_tokens'] for r in observed) if all_known else None,
        computed_tokens=sum(r['summary']['prompt_tokens']-r['summary']['cached_tokens']
            for r in observed) if all_known and all(type(r['summary'].get('prompt_tokens')) is int
                                                  for r in observed) else None)


def preemption_observations(result):
    metric='vllm:num_preemptions_total'
    result=result or {}
    before=result.get('metrics_before',{}).get(metric)
    after=result.get('metrics_after',{}).get(metric)
    if not all(type(v) in (int,float) and math.isfinite(v) and v>=0
               for v in (before,after)):
        return dict(status='unobserved',count=None,metric=metric)
    if after<before:
        return dict(status='counter_reset',count=None,metric=metric)
    return dict(status='observed',count=after-before,metric=metric)


def fixed_window_completion(rows, complete):
    known=complete and all(type(r.get('finished_s')) in (int,float)
        and math.isfinite(r['finished_s']) and r['finished_s']>=0 for r in rows)
    within=[r for r in rows if r['finished_s']<=48] if known else []
    return dict(status='observed' if known else 'unobserved',duration_s=48,
        protocol=sum(bool(r['summary']['ok']) for r in within) if known else None,
        strict_tool=sum(r.get('functional_ok') is True and r['summary']['ok']
            and not r['summary'].get('truncated') for r in within) if known else None)


def _summarize_repeats(groups):
    buckets=collections.defaultdict(list)
    for group in groups:
        if group['comparison_eligible']:
            key=(group['case'],group['configuration_sha256'],group['group'],group['rate'])
            buckets[key].append(group)
    result=[]
    for key,waves in sorted(buckets.items(),key=lambda x:str(x[0])):
        result.append(dict(case=key[0],configuration_sha256=key[1],group=key[2],rate=key[3],
            independent_repetitions=len(waves),wave_ids=[w['identity'] for w in waves],
            short_p95_s=distribution([w['short']['latency_s']['p95'] for w in waves]),
            short_errors=sum(w['short']['errors'] for w in waves),
            short_5s_goodput_per_s=distribution([w['short']['slo']['5']['tool_goodput_per_s'] for w in waves]),
            long_completed=sum(w['full_long_completed'] for w in waves),
            long_offered=sum(w['long']['offered'] for w in waves),
            limitation='Per-wave statistics; request samples within a wave are correlated.'))
    return result


def export(root: Path, destination: Path, allow_running=False) -> dict:
    root,destination=Path(root).resolve(),Path(destination).resolve()
    if root==destination:
        raise ValueError('destination cannot replace raw root')
    sources={}
    def read(path):
        data=path.read_bytes()
        sources[str(path.relative_to(root))]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
        return json.loads(data)
    def read_rows(path):
        if not path.exists():
            return [],[]
        data=path.read_bytes()
        sources[str(path.relative_to(root))]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
        rows,errors=[],[]
        for lineno,line in enumerate(data.splitlines(),1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except (ValueError,UnicodeError) as exc:
                errors.append(dict(line=lineno,error=type(exc).__name__))
        if len({r['id'] for r in rows})!=len(rows):
            raise ValueError(f'duplicate request identity: {path}')
        return rows,errors
    campaigns,groups,requests=[],[],[]
    for campaign_path in sorted(root.glob('*/campaign.json')):
        if campaign_path.is_relative_to(destination):
            continue
        run=campaign_path.parent
        campaign=read(campaign_path)
        if campaign['status']=='running' and not allow_running:
            raise ValueError(f'campaign still running: {run.name}')
        configuration=read(run/'config.json')
        configs={c['id']:c for c in configuration}
        if len(configs)!=len(configuration):
            raise ValueError('duplicate case configuration')
        campaigns.append(dict(run=run.name,**campaign))
        definitions={d['name']:d for d in campaign.get('workload_definitions',[])}
        seen=set()
        for state in campaign['cases']:
            if 'group' not in state:  # Kernel compatibility probes have no serving groups.
                continue
            key=(state['case'],state['repetition'],state['group'])
            if key in seen:
                raise ValueError('duplicate case/group/repetition')
            seen.add(key)
            if any(Path(str(part)).name!=str(part) or str(part) in {'.','..'} for part in [key[0],key[2]]):
                raise ValueError('invalid raw case path')
            case=configs[state['case']]
            case_dir=run/f'{key[0]}-r{key[1]}-{key[2]}'
            # Capacity/backend claims come from these files, not requested CLI budgets.
            for name in ('startup.json','server.log'):
                path=case_dir/name
                if path.exists():
                    data=path.read_bytes()
                    sources[str(path.relative_to(root))]=dict(
                        sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
            directory=case_dir/key[2]
            payloads=read(directory/'specifications.json') if (directory/'specifications.json').exists() else None
            planned=len(payloads) if payloads is not None else None
            rows,parse_errors=read_rows(directory/'requests.jsonl')
            for row in rows:
                expected=row.get('expected_tool')
                if expected is not None:
                    calls=reconstruct_tool_calls(row.get('events',[]))
                    row['functional_ok']=bool(row['summary']['ok'] and not row['summary'].get('truncated')
                        and len(calls)==1 and calls[0]['name']==expected['name']
                        and calls[0]['arguments']==expected['arguments'])
            result=read(directory/'result.json') if (directory/'result.json').exists() else None
            quality=read(case_dir/'quality-result.json') if (case_dir/'quality-result.json').exists() else {}
            complete=(result is not None and not parse_errors and planned==len(rows))
            configuration_hash=hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest()
            entry=dict(identity=f'{run.name}/{case_dir.name}/{key[2]}',run=run.name,
                case=key[0],repetition=key[1],group=key[2],configuration=case,
                configuration_sha256=configuration_hash,rate=definitions.get(key[2],{}).get('rate'),
                state=state['state'],error=state.get('error'),complete=complete,
                planned=planned,observed_responses=len(rows),offered=planned if complete else None,
                parse_errors=parse_errors,quality_status=quality.get('status'),
                summary=None,short=None,long=None)
            short=[r for r in rows if r.get('prompt_id')=='tool-24576']
            long=[r for r in rows if r.get('prompt_id')!='tool-24576']
            entry['prefix']=dict(short=prefix_observations(short),long=prefix_observations(long))
            entry['preemptions']=preemption_observations(result)
            entry['fixed_window_completion']=fixed_window_completion(rows,complete)
            coverage_path=directory/'morphx-coverage.json'
            entry['morphx_coverage']=read(coverage_path) if coverage_path.exists() else None
            entry['comparison_eligible']=bool(complete and quality.get('status')=='passed'
                and (not case.get('morphx',{}).get('enabled') or
                     (entry['morphx_coverage'] or {}).get('status')=='observed'))
            gpu_path=case_dir/'gpu.csv'
            gpu_data=gpu_path.read_bytes() if gpu_path.exists() else None
            if gpu_data is not None:
                sources[str(gpu_path.relative_to(root))]=dict(
                    sha256=hashlib.sha256(gpu_data).hexdigest(),bytes=len(gpu_data))
            entry['gpu_memory']=sampled_gpu_memory(gpu_data)
            if complete:
                span=result['summary']['measured_span_s']
                entry.update(summary=summarize_load(rows,span),short=summarize_load(short,span),
                             long=summarize_load(long,span))
                entry['full_long_completed']=sum(r['summary']['ok'] and not r['summary'].get('truncated')
                    and (r['summary'].get('prompt_tokens') or 0)>=130000 for r in long)
            groups.append(entry)
            for row in rows:
                requests.append(dict(identity=entry['identity']+'/'+row['id'],run=run.name,
                    case=key[0],group=key[2],repetition=key[1],id=row['id'],prompt_id=row.get('prompt_id'),
                    ok=row['summary']['ok'],functional_ok=row.get('functional_ok'),
                    truncated=row['summary'].get('truncated',False),errors=row['summary'].get('errors',[]),
                    latency_s=row['latency_s'],ttft_s=row.get('arrival_ttft_s'),
                    prompt_tokens=row['summary'].get('prompt_tokens'),cached_tokens=row['summary'].get('cached_tokens'),
                    completion_tokens=row['summary'].get('completion_tokens')))
    summary=dict(campaigns=campaigns,groups=groups,requests=requests,
        repeats=_summarize_repeats(groups),source_hashes=dict(sorted(sources.items())),
        limitation='Cross-engine quantization/templating/capacity differences are not isolated scheduler effects.')
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    group_rows=[]
    for g in groups:
        group_rows.append(dict(run=g['run'],case=g['case'],group=g['group'],repetition=g['repetition'],
            comparison_eligible=g['comparison_eligible'],
            rate=g['rate'],complete=g['complete'],planned=g['planned'],observed=g['observed_responses'],
            errors=g['summary']['errors'] if g['summary'] else None,
            short_p95_s=g['short']['latency_s']['p95'] if g['short'] else None,
            short_5s_goodput_per_s=g['short']['slo']['5']['tool_goodput_per_s'] if g['short'] else None,
            gpu_sampled_peak_mib=g['gpu_memory']['sampled_peak_mib'],
            preemptions=g['preemptions']['count'],
            preemptions_status=g['preemptions']['status'],
            completed_within_48s_protocol=g['fixed_window_completion']['protocol'],
            completed_within_48s_strict_tool=g['fixed_window_completion']['strict_tool'],
            full_long_completed=g.get('full_long_completed'),quality=g['quality_status']))
    for name,records in [('groups',group_rows),('requests',requests)]:
        with (destination/f'{name}.csv').open('w') as output:
            fields=list(records[0]) if records else ['identity']
            writer=csv.DictWriter(output,fields);writer.writeheader()
            for record in records:
                writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(list,dict)) else v
                                 for k,v in record.items()})
    audit=dict(campaigns=len(campaigns),complete_groups=sum(g['complete'] for g in groups),
        incomplete_groups=sum(not g['complete'] for g in groups),observed_responses=len(requests),
        input_files=len(sources),running_campaigns=[c['run'] for c in campaigns if c['status']=='running'])
    (destination/'audit.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('destination',type=Path)
    parser.add_argument('--allow-running',action='store_true');args=parser.parse_args()
    result=export(args.root,args.destination,args.allow_running)
    print(json.dumps(dict(groups=len(result['groups']),requests=len(result['requests']))))
