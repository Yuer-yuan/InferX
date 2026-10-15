"""Concurrent task lifecycles sharing FIFO generation admission."""
import asyncio
import math
import time

from inferx.scheduler import run_task

POLICIES = {'fixed1': 1, 'fixed2': 2, 'fixed4': 4, 'feedback-fair': 1}


async def run_episode(tasks: list[dict], generate, verify_candidate, *, policy: str,
                      arrivals: list[float], round_id: int, slots: int = 8,
                      budget: int = 4, deadline: float = 120) -> dict:
    if (policy not in POLICIES or not tasks or len(tasks) != len(arrivals)
            or len({x['task_id'] for x in tasks}) != len(tasks)
            or slots < 1 or budget < 1 or deadline <= 0 or not math.isfinite(deadline)
            or any(a < 0 or not math.isfinite(a) for a in arrivals)):
        raise ValueError('invalid episode policy, capacity, tasks or arrivals')
    permits = asyncio.Semaphore(slots)
    start = time.monotonic()

    async def one(task, arrival):
        await asyncio.sleep(max(0, start + arrival - time.monotonic()))
        actual = time.monotonic() - start
        lag = actual - arrival
        seed = 100000 * round_id + int(task['task_id'].split('/')[-1]) * 32 + 1000

        async def admitted(i, seed, event):
            event['admission_queued'] = time.monotonic() - event['_origin']
            async with permits:
                event['generation_admitted'] = time.monotonic() - event['_origin']
                return await generate(task, i, seed, event)

        async def checked(code, event):
            return await verify_candidate(task, code, event)

        if lag >= deadline:
            result = dict(task_id=task['task_id'], solved=False, winner=None,
                solution_latency=None, total=lag, candidates=[], observed_tokens=0,
                stop_reason='deadline', width=POLICIES[policy], budget=budget, seed=seed)
        else:
            result = await run_task(task, admitted, checked, POLICIES[policy], budget,
                                    deadline - lag, seed, await_feedback=policy == 'feedback-fair')
            result['total'] += lag
            if result['solution_latency'] is not None:
                result['solution_latency'] += lag
        result.update(scheduled_arrival=arrival, actual_arrival=actual, arrival_lag=lag,
                      deadline=deadline, policy=policy, round=round_id)
        return result

    jobs = [asyncio.create_task(one(t, a)) for t, a in zip(tasks, arrivals)]
    try:
        results = await asyncio.gather(*jobs)
    finally:
        for job in jobs:
            if not job.done():
                job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
    return dict(policy=policy, round=round_id, arrivals=arrivals, tasks=results,
                slots=slots, budget=budget, deadline=deadline, makespan=time.monotonic() - start)
