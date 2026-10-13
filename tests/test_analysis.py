import unittest

from scripts.analyze import summarize, select_width, analyze
try:
    from scripts.analyze import validate_coverage, paired_intervals
except ImportError:
    paired_intervals = None


def row(split, width, solved, latency=4, task='test/0', round_id=0):
    return dict(task_id=task, split=split, width=width, round=round_id, solved=solved,
                deadline=120, solution_latency=latency if solved else None,
                total=latency if solved else 9, winner=0 if solved else None,
                observed_tokens=15, candidates=[
                    dict(candidate_id=0, tokens=10, generation_start=0, generation_end=3,
                         verify_queued=3, verify_start=3.1, verify_end=4,
                         status='passed' if solved else 'failed', verification={'test_elapsed': .01}),
                    dict(candidate_id=1, tokens=5, generation_start=0, cancel_at=4,
                         cancel_done=4.02, status='cancelled')])


class AnalysisTests(unittest.TestCase):
    def test_cancelled_verification_wait_remains_visible(self):
        r = row('eval', 2, True)
        r['candidates'][1]['verify_queued'] = 3
        result = summarize([r])
        self.assertAlmostEqual(result['verification_queue_p95_s'], .955)
        self.assertEqual(result['verification_queue_cancelled'], 1)

    def test_paired_intervals_include_failed_task_penalty(self):
        self.assertIsNotNone(paired_intervals)
        rows = [row('eval', w, w == 2, latency=4, task=f'test/{t}', round_id=r)
                for t in range(3) for r in [0, 1] for w in [1, 2]]
        result = paired_intervals(rows, 2, draws=50)
        self.assertEqual(result['capped_solution_delta_s']['estimate'], -116)
        self.assertEqual(result['capped_solution_delta_s']['ci95'], [-116, -116])
        self.assertEqual(result['success_rate_delta']['ci95'], [1, 1])
        with self.assertRaises(ValueError):
            paired_intervals(rows[:-1], 2, draws=50)

    def test_partial_evaluation_grid_rejected(self):
        self.assertIsNotNone(validate_coverage)
        rows = [row(split, width, True, task=split + '/0', round_id=r)
                for split in ['dev', 'eval'] for width in [1, 2] for r in [0, 1]]
        validate_coverage(rows, {'dev': 1, 'eval': 1}, [1, 2], [0, 1])
        with self.assertRaises(ValueError):
            validate_coverage(rows[:-1], {'dev': 1, 'eval': 1}, [1, 2], [0, 1])

    def test_actual_workers_drain_and_unresolved_tail(self):
        r = row('eval', 1, False)
        r.update(verifier_workers=1, total_with_drain=10, server_drain_s=1)
        result = summarize([r])
        self.assertAlmostEqual(result['verifier_occupancy'], .9 / 10)
        self.assertEqual(result['total_wall_s'], 10)
        self.assertEqual(result['unresolved_verification_tail_p95_s'], 1)
        self.assertEqual(result['server_drain_p95_s'], 1)

    def test_failures_get_deadline_penalty(self):
        self.assertIsNotNone(summarize, 'analyzer missing')
        result = summarize([row('eval', 1, True), row('eval', 1, False, task='test/1')])
        self.assertEqual(result['solved'], 1)
        self.assertEqual(result['mean_capped_solution_s'], 62)
        self.assertEqual(result['observed_tokens'], 30)
        self.assertEqual(result['unused_observed_tokens'], 20)
        self.assertEqual(result['feedback_fraction_median'], .25)
        self.assertAlmostEqual(result['verification_queue_p95_s'], .1)

    def test_selection_does_not_read_eval_quality(self):
        self.assertIsNotNone(select_width, 'analyzer missing')
        rows = [row('dev', 1, True), row('dev', 2, False),
                row('eval', 1, False), row('eval', 2, True, latency=1)]
        self.assertEqual(select_width(rows), 1)

    def test_selection_prefers_quality_before_latency(self):
        self.assertIsNotNone(select_width, 'analyzer missing')
        rows = [row('dev', 1, True, 5), row('dev', 2, True, 2)]
        self.assertEqual(select_width(rows), 2)

    def test_overlap_and_duplicate_rejected(self):
        self.assertIsNotNone(analyze, 'analyzer missing')
        with self.assertRaises(ValueError):
            analyze([row('dev', 1, True), row('eval', 1, True)])
        with self.assertRaises(ValueError):
            analyze([row('dev', 1, True), row('dev', 1, True)])


if __name__ == '__main__':
    unittest.main()
