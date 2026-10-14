import asyncio
import time
import unittest

try:
    from inferx.scheduler import run_task
except ImportError:
    run_task = None


class SchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, generate, check, **kwargs):
        self.assertIsNotNone(run_task, "scheduler is not implemented")
        return await run_task({"task_id": "test/0"}, generate, check,
                              seed=7, **kwargs)

    async def test_verified_success_cancels_loser_and_counts_tokens(self):
        closed = []
        async def generate(i, seed, event):
            event['tokens'] = 9
            try:
                await asyncio.sleep(.02 if i == 0 else 10)
                return "valid"
            finally:
                closed.append(i)
        async def check(code, event):
            await asyncio.sleep(.01)
            return {'status': 'passed'}
        r = await self.run_case(generate, check, width=2, budget=2, deadline=1)
        self.assertTrue(r['solved'])
        self.assertEqual(sorted(closed), [0, 1])
        self.assertEqual(r['observed_tokens'], 18)
        self.assertEqual(r['winner'], 0)
        self.assertEqual(r['candidates'][1]['status'], 'cancelled')

    async def test_budget_exhaustion_and_refill_before_verification(self):
        async def generate(i, seed, event):
            await asyncio.sleep(.005)
            return 'bad'
        async def check(code, event):
            await asyncio.sleep(.04)
            return {'status': 'failed'}
        r = await self.run_case(generate, check, width=1, budget=3, deadline=1)
        self.assertFalse(r['solved'])
        self.assertEqual(r['stop_reason'], 'budget')
        self.assertEqual(len(r['candidates']), 3)
        self.assertLess(r['candidates'][1]['generation_start'], r['candidates'][0]['verify_end'])

    async def test_deadline_covers_verification_and_cleans_up(self):
        cleanup = []
        async def generate(i, seed, event):
            return 'code'
        async def check(code, event):
            try:
                await asyncio.sleep(10)
            finally:
                cleanup.append(True)
        start = time.monotonic()
        r = await self.run_case(generate, check, width=1, budget=1, deadline=.04)
        self.assertEqual(r['stop_reason'], 'deadline')
        self.assertFalse(r['solved'])
        self.assertEqual(cleanup, [True])
        self.assertLess(time.monotonic() - start, .5)

    async def test_engine_failure_is_not_a_model_failure(self):
        async def generate(i, seed, event):
            raise RuntimeError('engine unavailable')
        async def check(code, event):
            return {'status': 'failed'}
        with self.assertRaisesRegex(RuntimeError, 'engine unavailable'):
            await self.run_case(generate, check, width=1, budget=2, deadline=1)

    async def test_invalid_limits_rejected(self):
        with self.assertRaises(ValueError):
            await self.run_case(None, None, width=0, budget=2, deadline=1)

    async def test_feedback_gating_waits_for_failure_before_next_candidate(self):
        async def generate(i, seed, event):
            return 'code'
        async def check(code, event):
            await asyncio.sleep(.01)
            return {'status': 'passed' if event['candidate_id'] == 1 else 'failed'}
        result = await self.run_case(generate, check, width=1, budget=4,
                                     deadline=1, await_feedback=True)
        self.assertEqual(len(result['candidates']), 2)
        self.assertGreaterEqual(result['candidates'][1]['generation_start'],
                                result['candidates'][0]['verify_end'])


if __name__ == '__main__':
    unittest.main()
