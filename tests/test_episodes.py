import asyncio
import time
import unittest

try:
    from inferx.episodes import run_episode
except ImportError:
    run_episode = None


class EpisodeTests(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, generate, check, **kwargs):
        self.assertIsNotNone(run_episode, 'episode wrapper missing')
        arrivals = kwargs.pop('arrivals', [0, 0, 0])
        tasks = [dict(task_id=f'test/{i}') for i in range(len(arrivals))]
        return await run_episode(tasks, generate, check, arrivals=arrivals, round_id=0, **kwargs)

    async def test_capacity_and_independent_success(self):
        active = 0; peak = 0; finished = []
        async def generate(task, i, seed, event):
            nonlocal active, peak
            active += 1; peak = max(peak, active)
            try:
                await asyncio.sleep(.01 if task['task_id'] == 'test/0' else .03)
                event['tokens'] = 5
                finished.append(task['task_id'])
                return 'ok'
            finally:
                active -= 1
        async def check(task, code, event):
            return {'status': 'passed'}
        result = await self.run_case(generate, check, policy='fixed2', slots=2, deadline=1)
        self.assertEqual(active, 0)
        self.assertEqual(peak, 2)
        self.assertTrue(all(t['solved'] for t in result['tasks']))
        self.assertEqual(set(finished), {'test/0', 'test/1', 'test/2'})

    async def test_fifo_feedback_waits_and_preserves_seed_pairing(self):
        order = []
        async def generate(task, i, seed, event):
            order.append((task['task_id'], i, seed))
            await asyncio.sleep(.005)
            return 'bad'
        async def check(task, code, event):
            await asyncio.sleep(.01)
            return {'status': 'failed'}
        result = await self.run_case(generate, check, policy='feedback-fair', slots=1, budget=2, deadline=1)
        self.assertEqual([x[0] for x in order[:3]], ['test/0', 'test/1', 'test/2'])
        for task in result['tasks']:
            a, b = task['candidates']
            self.assertGreaterEqual(b['generation_start'], a['verify_end'])
            self.assertGreaterEqual(a['generation_admitted'], a['admission_queued'])

    async def test_waiting_deadline_cleans_permit_for_later_arrival(self):
        async def generate(task, i, seed, event):
            if task['task_id'] != 'test/2':
                try:
                    await asyncio.sleep(10)
                finally:
                    # Hold the permit through cancellation cleanup so both
                    # deadlines expire before the queued task can be admitted.
                    await asyncio.sleep(.08)
            return 'ok'
        async def check(task, code, event):
            return {'status': 'passed'}
        result = await self.run_case(generate, check, policy='fixed1', slots=1,
                                     budget=1, deadline=.04, arrivals=[0, 0, .2])
        first, waiting, later = result['tasks']
        self.assertFalse(first['solved']); self.assertFalse(waiting['solved'])
        self.assertNotIn('generation_admitted', waiting['candidates'][0])
        self.assertTrue(later['solved'])
        self.assertGreaterEqual(later['actual_arrival'], .2)
        self.assertLess(result['makespan'], 1)

    async def test_infrastructure_failure_cancels_other_tasks(self):
        closed = []
        async def generate(task, i, seed, event):
            try:
                if task['task_id'] == 'test/0':
                    await asyncio.sleep(.01)
                    raise RuntimeError('engine lost')
                await asyncio.sleep(10)
            finally:
                closed.append(task['task_id'])
        with self.assertRaisesRegex(RuntimeError, 'engine lost'):
            await self.run_case(generate, None, policy='fixed1', slots=3, deadline=1)
        self.assertEqual(set(closed), {'test/0', 'test/1', 'test/2'})

    async def test_external_cancellation_cleans_admitted_and_waiting_tasks(self):
        started = asyncio.Event(); closed = []
        async def generate(task, i, seed, event):
            started.set()
            try:
                await asyncio.sleep(10)
            finally:
                closed.append(task['task_id'])
        self.assertIsNotNone(run_episode, 'episode wrapper missing')
        job = asyncio.create_task(self.run_case(generate, None, policy='fixed1', slots=1))
        await started.wait()
        job.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await job
        self.assertEqual(len(closed), 1)

    async def test_invalid_policy_and_arrival_grid_rejected(self):
        self.assertIsNotNone(run_episode, 'episode wrapper missing')
        for args in [dict(policy='future'), dict(policy='fixed1', arrivals=[-1]),
                     dict(policy='fixed1', slots=0)]:
            with self.assertRaises(ValueError):
                await self.run_case(None, None, **args)

    async def test_late_arrival_is_expired_before_candidate_submission(self):
        async def generate(task, i, seed, event):
            time.sleep(.05)
            return 'code'
        async def check(task, code, event):
            return {'status': 'passed'}
        result = await self.run_case(generate, check, policy='fixed1', budget=1,
                                     arrivals=[0, .01], deadline=.02)
        self.assertEqual(result['tasks'][1]['candidates'], [])
        self.assertEqual(result['tasks'][1]['stop_reason'], 'deadline')
