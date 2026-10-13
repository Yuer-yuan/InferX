"""Fixed generation concurrency with asynchronous verification and cancellation."""
import asyncio
import time


async def run_task(task: dict, generate, verify_candidate, width: int, budget: int,
                   deadline: float, seed: int) -> dict:
    if min(width, budget, deadline) <= 0:
        raise ValueError("width, budget and deadline must be positive")
    start = time.monotonic()
    events = []
    pending = {}
    winner = None
    reason = "budget"

    async def gen(event):
        code = await generate(event['candidate_id'], event['seed'], event)
        event['generation_end'] = time.monotonic() - start
        event['code'] = code
        return code

    async def check(event):
        event['verify_queued'] = time.monotonic() - start
        result = await verify_candidate(event['code'], event)
        event.setdefault('verify_start', event['verify_queued'])
        event['verify_end'] = time.monotonic() - start
        event['verification'] = result
        event['status'] = result['status']
        return result

    def refill():
        active = sum(stage == 'generate' for stage, _ in pending.values())
        while active < width and len(events) < budget:
            i = len(events)
            event = dict(candidate_id=i, seed=seed + i, generation_start=time.monotonic() - start,
                         tokens=0, status='generating', _origin=start)
            events.append(event)
            pending[asyncio.create_task(gen(event))] = ('generate', event)
            active += 1

    try:
        refill()
        while pending:
            remaining = deadline - (time.monotonic() - start)
            if remaining <= 0:
                reason = 'deadline'
                break
            done, _ = await asyncio.wait(pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
            if not done:
                reason = 'deadline'
                break
            # Observe all exceptions before deciding success; infrastructure errors
            # must abort a trial, never silently become extra model failures.
            completed = [(future, future.result()) for future in done]
            passes = []
            for future, result in completed:
                stage, event = pending.pop(future)
                if stage == 'generate':
                    event['status'] = 'verifying'
                    pending[asyncio.create_task(check(event))] = ('verify', event)
                elif result['status'] == 'passed' and event['verify_end'] <= deadline:
                    passes.append(event)
            if passes:
                winner = min(passes, key=lambda x: x['verify_end'])
                reason = 'passed'
                break
            refill()
    finally:
        cancel_at = time.monotonic() - start
        for future, (_, event) in pending.items():
            if not future.done():
                event['cancel_at'] = cancel_at
                event['status'] = 'cancelled'
                future.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for _, event in pending.values():
            if 'cancel_at' in event:
                event['cancel_done'] = time.monotonic() - start
        for event in events:
            event.pop('_origin', None)
    return dict(task_id=task['task_id'], solved=winner is not None,
                winner=winner['candidate_id'] if winner else None,
                solution_latency=winner['verify_end'] if winner else None,
                total=time.monotonic() - start, deadline=deadline, width=width,
                budget=budget, seed=seed, stop_reason=reason, candidates=events,
                observed_tokens=sum(x['tokens'] for x in events))
