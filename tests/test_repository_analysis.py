import copy
import unittest

try:
    from scripts.analyze_repository import (summarize, select_baseline, validate_grid,
                                           paired_intervals, screening_decision)
except ImportError:
    summarize = None


def task(task_id, solved=True, latency=10, tokens=20):
    event = dict(candidate_id=0, generation_start=0, admission_queued=0, generation_admitted=0,
        generation_end=4, verify_queued=4, verify_start=5, verify_end=latency,
        tokens=tokens, status='passed' if solved else 'failed',
        verification={'status': 'passed' if solved else 'failed', 'test_elapsed': 2})
    return dict(task_id=task_id, solved=solved, winner=0 if solved else None,
        solution_latency=latency if solved else None, total=latency, deadline=120,
        candidates=[event], observed_tokens=tokens, arrival_lag=0)


def episode(split, eid, policy, round_id, rows, mode='burst'):
    return dict(split=split, episode_id=eid, policy=policy, round=round_id, mode=mode,
                tasks=rows, makespan=max(t['total'] for t in rows), server_drain_s=2,
                makespan_with_drain=max(t['total'] for t in rows)+2)


def grid():
    dev = [episode('dev', f'dev/{i}', p, r, [task(f'dev/{i}')], 'single')
           for i in range(8) for p in ('fixed1', 'fixed2', 'fixed4') for r in (0, 1)]
    evaluation = [episode('eval', f'episode/{i}', p, r,
                  [task(f'eval/{i*4+j}') for j in range(4)], m)
                  for i in range(4) for m in ('burst', 'stagger')
                  for p in ('fixed1', 'fixed2', 'fixed4', 'feedback-fair') for r in (0, 1)]
    return dev + evaluation


class RepositoryAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(summarize, 'repository analyzer missing')

    def test_failure_penalty_empty_deadline_and_censored_queue_are_retained(self):
        passed = task('x/0')
        waiting = task('x/1', False, latency=12, tokens=3)
        waiting['candidates'] = [dict(candidate_id=0, tokens=3, status='cancelled',
            generation_start=0, admission_queued=0, cancel_at=12, cancel_done=12)]
        expired = task('x/2', False, latency=130, tokens=0)
        expired.update(candidates=[], arrival_lag=130)
        rows = [episode('eval', str(i), 'fixed1', 0, [t]) for i, t in enumerate([passed, waiting, expired])]
        result = summarize(rows)
        self.assertAlmostEqual(result['mean_capped_solution_s'], 250/3)
        self.assertEqual(result['observed_tokens'], 23)
        self.assertEqual(result['unused_observed_tokens'], 3)
        self.assertEqual(result['cancelled_observed_tokens'], 3)
        self.assertEqual(result['admission_cancelled'], 1)
        self.assertAlmostEqual(result['candidate_queue_fraction_median'], .55)
        self.assertEqual(result['total_wall_s'], 158)

    def test_development_only_selection_quality_before_latency_and_width(self):
        rows = grid()
        for e in rows:
            if e['split'] == 'dev' and e['policy'] == 'fixed1':
                e['tasks'][0] = task(e['tasks'][0]['task_id'], False, latency=1)
        self.assertEqual(select_baseline(rows), 'fixed2')
        for e in rows:
            if e['split'] == 'eval' and e['policy'] == 'fixed2':
                e['tasks'] = [task(t['task_id'], False) for t in e['tasks']]
        self.assertEqual(select_baseline(rows), 'fixed2')

    def test_full_grid_missing_duplicate_overlap_and_changed_pair_rejected(self):
        rows = grid(); validate_grid(rows)
        for malformed in (rows[:-1], rows + [rows[-1]]):
            with self.assertRaises(ValueError):
                validate_grid(malformed)
        overlap = copy.deepcopy(rows)
        overlap[-1]['tasks'][0]['task_id'] = 'dev/0'
        with self.assertRaises(ValueError):
            validate_grid(overlap)
        changed = copy.deepcopy(rows)
        changed[-1]['tasks'][0]['task_id'] = 'eval/1000'
        with self.assertRaises(ValueError):
            validate_grid(changed)

    def test_bootstrap_resamples_episode_clusters_with_both_rounds(self):
        rows = grid()
        for e in rows:
            if e['split'] == 'eval' and e['policy'] == 'feedback-fair':
                e['tasks'] = [task(t['task_id'], latency=5 if e['round'] == 0 else 9,
                                   tokens=10 if e['round'] == 0 else 30) for t in e['tasks']]
        result = paired_intervals(rows, 'fixed1', 'burst', draws=100)
        self.assertEqual(result['episode_clusters'], 4)
        self.assertEqual(result['rounds_per_cluster'], 2)
        self.assertEqual(result['capped_solution_delta_s']['ci95'], [-3, -3])
        self.assertEqual(result['token_reduction_fraction']['ci95'], [0, 0])

    def test_development_only_rejects_corrupted_tokens_and_late_success(self):
        for field, value in [('observed_tokens', 1020), ('solution_latency', 121)]:
            with self.subTest(field=field):
                rows = [e for e in grid() if e['split'] == 'dev']
                validate_grid(rows, dev_only=True)
                rows[0]['tasks'][0][field] = value
                with self.assertRaises(ValueError):
                    validate_grid(rows, dev_only=True)

    def test_screening_gates_quality_first_and_both_modes_required(self):
        base = dict(solved=12, trials=16, mean_capped_solution_s=40,
                    observed_tokens=100, candidate_queue_fraction_median=.1,
                    unused_observed_fraction=.3)
        groups = {f'eval/{m}/{p}/round{r}': dict(base)
                  for m in ('burst', 'stagger') for p in ('fixed1', 'feedback-fair') for r in (0, 1)}
        for k, v in groups.items():
            if '/feedback-fair/' in k:
                v['mean_capped_solution_s'] = 30
        self.assertEqual(screening_decision(groups, 'fixed1'), 'ordinary_feedback_fair_baseline_sufficient')
        groups['eval/stagger/feedback-fair/round1']['solved'] = 7
        self.assertEqual(screening_decision(groups, 'fixed1'), 'quality_or_sample_insufficient')
        groups['eval/stagger/feedback-fair/round1']['solved'] = 12
        for k, v in groups.items():
            if '/feedback-fair/' in k:
                v.update(mean_capped_solution_s=45, candidate_queue_fraction_median=.25)
        self.assertEqual(screening_decision(groups, 'fixed1'), 'residual_pressure_requires_followup')
        for v in groups.values():
            v['candidate_queue_fraction_median'] = .05
        self.assertEqual(screening_decision(groups, 'fixed1'), 'mixed_or_unsupported')
