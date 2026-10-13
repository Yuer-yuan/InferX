"""Analyze live records; select width only on development tasks."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    i = (len(values) - 1) * q
    lo = int(i)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (i - lo)


def summarize(rows):
    successes = [x for x in rows if x['solved']]
    events = [e for x in rows for e in x['candidates']]
    feedback = []
    for row in successes:
        winner = next(e for e in row['candidates'] if e['candidate_id'] == row['winner'])
        feedback.append((winner['verify_end'] - winner['generation_end']) / row['solution_latency'])
    queue = [e['verify_start'] - e['verify_queued'] for e in events if 'verify_start' in e]
    service = [e['verify_end'] - e['verify_start'] for e in events if 'verify_end' in e]
    busy = sum((e.get('verify_end', e.get('cancel_done', e['verify_start'])) - e['verify_start'])
               for e in events if 'verify_start' in e)
    test_cost = [e['verification']['test_elapsed'] for e in events
                 if e.get('verification', {}).get('test_elapsed') is not None]
    observed = sum(x['observed_tokens'] for x in rows)
    used = sum(next(e['tokens'] for e in x['candidates'] if e['candidate_id'] == x['winner']) for x in successes)
    return dict(trials=len(rows), solved=len(successes), success_rate=len(successes)/len(rows),
                mean_capped_solution_s=statistics.mean(x['solution_latency'] if x['solved'] else x['deadline'] for x in rows),
                solved_median_s=percentile([x['solution_latency'] for x in successes], .5),
                solved_p95_s=percentile([x['solution_latency'] for x in successes], .95),
                feedback_fraction_median=percentile(feedback, .5), feedback_fraction_p95=percentile(feedback, .95),
                verification_queue_p95_s=percentile(queue, .95),
                verification_service_median_s=percentile(service, .5),
                verification_service_p95_s=percentile(service, .95),
                test_execution_median_s=percentile(test_cost, .5),
                verifier_occupancy=busy/(2*sum(x['total'] for x in rows)),
                observed_tokens=observed, unused_observed_tokens=observed-used,
                cancelled_observed_tokens=sum(e['tokens'] for e in events if e['status'] == 'cancelled'),
                observed_tokens_per_wall_s=observed/sum(x['total'] for x in rows),
                total_wall_s=sum(x['total'] for x in rows),
                verifier_timeouts=sum(e.get('verification', {}).get('status') == 'timeout' for e in events))


def select_width(rows):
    groups = defaultdict(list)
    for row in rows:
        if row['split'] == 'dev':
            groups[row['width']].append(row)
    if not groups:
        raise ValueError('no development results')
    # Same number and identity of tasks/rounds for each candidate width.
    coverage = [{(r['task_id'], r['round']) for r in group} for group in groups.values()]
    if any(c != coverage[0] for c in coverage):
        raise ValueError('unbalanced development coverage')
    scores = {width: summarize(group) for width, group in groups.items()}
    return min(scores, key=lambda w: (-scores[w]['solved'], scores[w]['mean_capped_solution_s'], w))


def analyze(rows):
    seen, splits = set(), defaultdict(set)
    for row in rows:
        key = (row['task_id'], row['width'], row['round'])
        if key in seen:
            raise ValueError('duplicate trial: ' + str(key))
        seen.add(key)
        splits[row['task_id']].add(row['split'])
    if any(len(s) > 1 for s in splits.values()):
        raise ValueError('development/evaluation task overlap')
    width = select_width(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[f"{row['split']}/width{row['width']}/round{row['round']}"] .append(row)
        groups[f"{row['split']}/width{row['width']}/all"].append(row)
    summaries = {k: summarize(v) for k, v in sorted(groups.items())}
    selected = [summaries[k] for k in summaries if k.startswith(f'eval/width{width}/round')]
    decision = 'insufficient_evidence'
    if len(selected) >= 2 and all(x['solved'] >= 12 for x in selected):
        if all(x['feedback_fraction_median'] < .05 and x['feedback_fraction_p95'] < .1
               and x['verification_queue_p95_s'] < .1 for x in selected):
            decision = 'no_material_verification_pressure_on_this_workload'
        elif all(x['feedback_fraction_median'] >= .1 for x in selected):
            decision = 'pressure_present_test_backpressure_policy'
        else:
            decision = 'mixed_or_borderline'
    return dict(selected_width=width, decision=decision, groups=summaries,
                note='Cancelled token counts are observed lower bounds. Occupancy assumes two verifier workers. This is a screening experiment, not a publication claim.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for file in args.files for line in Path(file).read_text().splitlines()]
    result = analyze(rows)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'selected_width': result['selected_width'], 'decision': result['decision'], 'trials': len(rows)}))
