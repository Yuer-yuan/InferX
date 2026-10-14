"""Paired episode analysis with failed tasks retained and dev-only selection."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import statistics


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    index = (len(values)-1)*q
    lo = int(index); hi = min(lo+1, len(values)-1)
    return values[lo] + (values[hi]-values[lo])*(index-lo)


def capped(task):
    return task['solution_latency'] if task['solved'] else task['deadline']


def summarize(episodes):
    tasks = [t for e in episodes for t in e['tasks']]
    if not tasks:
        raise ValueError('empty group')
    events = [c for t in tasks for c in t['candidates']]
    admission = []; verification = []; queue_fractions = []; feedback = []; busy = 0
    used_tokens = 0
    for task in tasks:
        if task['solved']:
            winner = next(c for c in task['candidates'] if c['candidate_id'] == task['winner'])
            used_tokens += winner['tokens']
            feedback.append((winner['verify_end'] - winner['generation_end']) / task['solution_latency'])
        for c in task['candidates']:
            a = v = 0
            if 'admission_queued' in c:
                a = c.get('generation_admitted', c.get('cancel_at', c['admission_queued'])) - c['admission_queued']
                admission.append(a)
            if 'verify_queued' in c:
                v = c.get('verify_start', c.get('cancel_at', c['verify_queued'])) - c['verify_queued']
                verification.append(v)
            queue_fractions.append((a+v)/task['total'] if task['total'] else 0)
            if 'verify_start' in c:
                busy += c.get('verify_end', c.get('cancel_done', c['verify_start'])) - c['verify_start']
    observed = sum(t['observed_tokens'] for t in tasks)
    wall = sum(e['makespan_with_drain'] for e in episodes)
    successes = [t for t in tasks if t['solved']]
    return dict(episodes=len(episodes), trials=len(tasks), solved=len(successes),
        success_rate=len(successes)/len(tasks), mean_capped_solution_s=statistics.mean(map(capped, tasks)),
        solved_median_s=percentile([t['solution_latency'] for t in successes], .5),
        solved_p95_s=percentile([t['solution_latency'] for t in successes], .95),
        observed_tokens=observed, observed_tokens_per_task=observed/len(tasks),
        unused_observed_tokens=observed-used_tokens,
        unused_observed_fraction=(observed-used_tokens)/observed if observed else 0,
        cancelled_observed_tokens=sum(c['tokens'] for c in events if c['status'] == 'cancelled'),
        generated_candidates=len(events), format_errors=sum('format_error' in c for c in events),
        verifier_timeouts=sum(c.get('verification', {}).get('status') == 'timeout' for c in events),
        admission_queue_median_s=percentile(admission, .5), admission_queue_p95_s=percentile(admission, .95),
        verification_queue_median_s=percentile(verification, .5), verification_queue_p95_s=percentile(verification, .95),
        admission_cancelled=sum('admission_queued' in c and 'generation_admitted' not in c for c in events),
        verification_queue_cancelled=sum('verify_queued' in c and 'verify_start' not in c for c in events),
        candidate_queue_fraction_median=percentile(queue_fractions, .5) or 0,
        feedback_fraction_median=percentile(feedback, .5), feedback_fraction_p95=percentile(feedback, .95),
        verifier_occupancy=busy/(2*wall) if wall else 0,
        server_drain_median_s=percentile([e['server_drain_s'] for e in episodes], .5),
        server_drain_p95_s=percentile([e['server_drain_s'] for e in episodes], .95),
        mean_makespan_with_drain_s=wall/len(episodes), total_wall_s=wall,
        observed_tokens_per_wall_s=observed/wall if wall else 0,
        solved_tasks_per_wall_s=len(successes)/wall if wall else 0)


def validate_grid(rows, *, dev_only=False):
    dev = [e for e in rows if e['split'] == 'dev']
    evaluation = [e for e in rows if e['split'] == 'eval']
    if len(dev) + len(evaluation) != len(rows):
        raise ValueError('unknown split')
    dev_ids = {t['task_id'] for e in dev for t in e['tasks']}
    eval_ids = {t['task_id'] for e in evaluation for t in e['tasks']}
    if dev_ids & eval_ids:
        raise ValueError('development/evaluation task overlap')
    expected_dev = {(t, p, r) for t in dev_ids for p in ('fixed1', 'fixed2', 'fixed4') for r in (0, 1)}
    actual_dev = [(e['episode_id'], e['policy'], e['round']) for e in dev]
    if (len(dev_ids) != 8 or any(len(e['tasks']) != 1 or e['mode'] != 'single'
            or e['tasks'][0]['task_id'] != e['episode_id'] for e in dev)
            or len(actual_dev) != len(expected_dev) or set(actual_dev) != expected_dev):
        raise ValueError('incomplete or duplicated development grid')
    if dev_only:
        if evaluation:
            raise ValueError('dev-only selection input contains evaluation')
        return
    ids = {e['episode_id'] for e in evaluation}
    expected = {(eid, m, p, r) for eid in ids for m in ('burst', 'stagger')
                for p in ('fixed1', 'fixed2', 'fixed4', 'feedback-fair') for r in (0, 1)}
    actual = [(e['episode_id'], e['mode'], e['policy'], e['round']) for e in evaluation]
    if len(ids) != 4 or len(eval_ids) != 16 or len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('incomplete or duplicated evaluation grid')
    membership = []
    for eid in ids:
        matching = [e for e in evaluation if e['episode_id'] == eid]
        tuples = {tuple(t['task_id'] for t in e['tasks']) for e in matching}
        if len(tuples) != 1 or len(next(iter(tuples))) != 4 or len(set(next(iter(tuples)))) != 4:
            raise ValueError('paired episode membership changed')
        membership.extend(next(iter(tuples)))
    if len(set(membership)) != len(membership):
        raise ValueError('evaluation tasks reused across episode templates')
    for e in rows:
        for t in e['tasks']:
            if t['solved'] and (t['solution_latency'] is None or t['solution_latency'] > t['deadline']):
                raise ValueError('success outside deadline')
            if t['observed_tokens'] != sum(c['tokens'] for c in t['candidates']):
                raise ValueError('token accounting mismatch')


def select_baseline(rows):
    groups = {p: [e for e in rows if e['split'] == 'dev' and e['policy'] == p]
              for p in ('fixed1', 'fixed2', 'fixed4')}
    coverage = [{(e['episode_id'], e['round']) for e in v} for v in groups.values()]
    if not coverage[0] or any(c != coverage[0] for c in coverage):
        raise ValueError('unbalanced development coverage')
    scores = {p: summarize(v) for p, v in groups.items()}
    return min(scores, key=lambda p: (-scores[p]['solved'], scores[p]['mean_capped_solution_s'], int(p[-1])))


def paired_intervals(rows, baseline, mode, draws=2000):
    policies = (baseline, 'feedback-fair')
    groups = {p: {(e['episode_id'], e['round']): e for e in rows
                  if e['split'] == 'eval' and e['mode'] == mode and e['policy'] == p} for p in policies}
    a, b = (groups[p] for p in policies)
    if not a or set(a) != set(b):
        raise ValueError('paired policies/rounds required')
    clusters = sorted({k[0] for k in a})
    rounds = sorted({k[1] for k in a})
    if set(a) != {(c, r) for c in clusters for r in rounds}:
        raise ValueError('incomplete cluster rounds')

    def values(sample):
        ar = [t for c in sample for r in rounds for t in a[(c, r)]['tasks']]
        br = [t for c in sample for r in rounds for t in b[(c, r)]['tasks']]
        at = sum(t['observed_tokens'] for t in ar); bt = sum(t['observed_tokens'] for t in br)
        return (statistics.mean(map(capped, br))-statistics.mean(map(capped, ar)),
                statistics.mean(t['solved'] for t in br)-statistics.mean(t['solved'] for t in ar),
                1-bt/at if at else 0)
    estimate = values(clusters)
    rng = random.Random(2026101402)
    samples = [values(rng.choices(clusters, k=len(clusters))) for _ in range(draws)]
    result = dict(episode_clusters=len(clusters), rounds_per_cluster=len(rounds), draws=draws,
        method='paired episode-template percentile bootstrap; all tasks and both rounds retained')
    for i, name in enumerate(('capped_solution_delta_s', 'success_rate_delta', 'token_reduction_fraction')):
        result[name] = dict(estimate=estimate[i], ci95=[percentile([v[i] for v in samples], q) for q in (.025, .975)])
    return result


def screening_decision(groups, baseline):
    pair = {(m, r): [groups[f'eval/{m}/{p}/round{r}'] for p in (baseline, 'feedback-fair')]
            for m in ('burst', 'stagger') for r in (0, 1)}
    if any(s['solved'] < 8 or s['trials'] != 16 for v in pair.values() for s in v):
        return 'quality_or_sample_insufficient'
    no_quality_loss = all(b['solved'] >= a['solved'] for a, b in pair.values())
    modes = {}
    for m in ('burst', 'stagger'):
        modes[m] = [dict(time=statistics.mean(pair[(m, r)][i]['mean_capped_solution_s'] for r in (0, 1)),
                         tokens=sum(pair[(m, r)][i]['observed_tokens'] for r in (0, 1))) for i in (0, 1)]
    faster = all(b['time'] <= .8*a['time'] for a, b in modes.values())
    cheaper = all(b['tokens'] <= .7*a['tokens'] and b['time'] <= 1.05*a['time'] for a, b in modes.values())
    if no_quality_loss and (faster or cheaper):
        return 'ordinary_feedback_fair_baseline_sufficient'
    queue = all(b['candidate_queue_fraction_median'] >= .2 for a, b in pair.values())
    unused = all(b['unused_observed_fraction'] >= .5
                 and b['mean_capped_solution_s'] >= 1.2*a['mean_capped_solution_s'] for a, b in pair.values())
    if queue or unused:
        return 'residual_pressure_requires_followup'
    return 'mixed_or_unsupported'


def analyze(rows):
    validate_grid(rows)
    baseline = select_baseline(rows)
    grouped = defaultdict(list)
    for e in rows:
        prefix = f'{e["split"]}/{e["mode"]}/{e["policy"]}'
        grouped[prefix + f'/round{e["round"]}'].append(e)
        grouped[prefix + '/all'].append(e)
    groups = {k: summarize(v) for k, v in sorted(grouped.items())}
    return dict(selected_baseline=baseline, decision=screening_decision(groups, baseline), groups=groups,
        paired_intervals={m: paired_intervals(rows, baseline, m) for m in ('burst', 'stagger')},
        note='Observed tokens are lower bounds. Unused tokens include failed candidates and are not all avoidable. Queue includes cancelled waiters. Failures cost 120s; makespan includes drain. Four episode templates limit inference. Candidate queue fraction is each candidate admission+verification queue divided by its task elapsed time.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('files', nargs='+')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dev-only', action='store_true')
    args = p.parse_args()
    rows = [json.loads(line) for file in args.files for line in Path(file).read_text().splitlines()]
    if args.dev_only:
        validate_grid(rows, dev_only=True)
        result = dict(selected_baseline=select_baseline(rows),
            groups={p: summarize([e for e in rows if e['policy'] == p]) for p in ('fixed1', 'fixed2', 'fixed4')})
    else:
        result = analyze(rows)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    with args.output.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, ['group'] + list(next(iter(result['groups'].values()))))
        writer.writeheader()
        writer.writerows(dict(group=k, **v) for k, v in result['groups'].items())
    print(json.dumps({k: v for k, v in result.items() if k not in ('groups', 'paired_intervals', 'note')}))
