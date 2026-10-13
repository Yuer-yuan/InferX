import unittest

try:
    from scripts.analyze import summarize, select_width, analyze
except ImportError:
    summarize = select_width = analyze = None


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
