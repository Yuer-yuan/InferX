import asyncio
import io
import json
import math

import pytest
from aiohttp import web

from inferx.concurrency import bandwidth_bytes_per_s, reconstruct_tool_calls, run_load, summarize_load


def test_partial_tool_arguments_are_reassembled_and_not_accepted_early():
    events = [dict(choices=[dict(delta=dict(tool_calls=[dict(index=0, id='call-a',
        function=dict(name='record_status', arguments='{"status":"rea'))]))]),
        dict(choices=[dict(delta=dict(tool_calls=[dict(index=0,
            function=dict(arguments='dy","attempt":3}'))]))])]
    partial = reconstruct_tool_calls(events[:1])
    assert partial[0]['arguments'] is None
    complete = reconstruct_tool_calls(events)
    assert complete[0]['name'] == 'record_status'
    assert complete[0]['arguments'] == {'status': 'ready', 'attempt': 3}


def test_load_summary_keeps_errors_and_separates_tool_quality_from_protocol():
    rows = [dict(id='a', latency_s=.4, summary=dict(ok=True, completion_tokens=8),
                 functional_ok=True),
            dict(id='b', latency_s=.8, summary=dict(ok=True, completion_tokens=4),
                 functional_ok=False),
            dict(id='c', latency_s=.2, summary=dict(ok=False, completion_tokens=None,
                                                errors=['HTTP429']), functional_ok=False)]
    result = summarize_load(rows, 1.0)
    assert result['offered'] == 3 and result['protocol_ok'] == 2 and result['errors'] == 1
    assert result['completion_tokens'] == 12 and result['aggregate_tokens_per_s'] == 12
    assert result['tool_valid'] == 1
    assert result['latency_s']['p95'] is None and result['latency_s']['p99'] is None
    assert result['slo']['0.5']['protocol_goodput_per_s'] == 1
    assert result['slo']['1']['tool_goodput_per_s'] == 1
    many = [dict(id=str(i), latency_s=float(i + 1), summary=dict(ok=True,
            completion_tokens=1), functional_ok=None) for i in range(100)]
    summary = summarize_load(many, 100)
    assert summary['latency_s']['p95'] == 95 and summary['latency_s']['p99'] == 99
    with pytest.raises(ValueError, match='duplicate'):
        summarize_load(rows + rows[:1], 1)


def test_bandwidth_units_include_both_reads_and_writes():
    assert bandwidth_bytes_per_s(1000, 10, 1000) == 20000
    with pytest.raises(ValueError):
        bandwidth_bytes_per_s(1000, 10, 0)


def test_real_http_arrivals_overlap_and_closed_loop_obeys_worker_cap():
    async def exercise():
        active = peak = 0
        starts = []

        async def handler(request):
            nonlocal active, peak
            await request.json()
            active += 1
            peak = max(peak, active)
            starts.append(asyncio.get_running_loop().time())
            try:
                await asyncio.sleep(.15)
                event = dict(choices=[dict(delta=dict(content='ok'), finish_reason='stop')],
                             usage=dict(prompt_tokens=3, completion_tokens=2))
                return web.Response(text='data: ' + json.dumps(event) + '\n\ndata: [DONE]\n\n',
                                    content_type='text/event-stream')
            finally:
                active -= 1

        app = web.Application()
        app.router.add_post('/v1/chat/completions', handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        requests = [dict(id=str(i), offset_s=i * .03, payload={}) for i in range(3)]
        try:
            output = io.StringIO()
            rows = await run_load(f'http://127.0.0.1:{port}', requests, output,
                                  mode='open', workers=1)
            assert peak == 3 and starts[-1] - starts[0] < .13
            assert all(r['summary']['ok'] for r in rows)
            assert all(math.isclose(r['latency_s'], r['dispatch_lag_s'] + r['summary']['wall_s'])
                       for r in rows)
            assert len(output.getvalue().splitlines()) == 3
            peak = 0
            rows = await run_load(f'http://127.0.0.1:{port}', requests, io.StringIO(),
                                  mode='closed', workers=1)
            assert peak == 1 and all(r['dispatch_lag_s'] == 0 for r in rows)
        finally:
            await runner.cleanup()

    asyncio.run(exercise())


def test_runner_rejects_complete_tool_json_when_truncated_or_protocol_failed():
    async def exercise():
        expected = dict(name='record_status', arguments={'status': 'ready'})

        async def handler(request):
            body = await request.json()
            event = dict(choices=[dict(delta=dict(tool_calls=[dict(index=0,
                function=dict(name=expected['name'], arguments=json.dumps(expected['arguments'])))]),
                finish_reason=body['finish'])], usage=dict(prompt_tokens=3, completion_tokens=8))
            return web.Response(text='data: '+json.dumps(event)+'\n\ndata: [DONE]\n\n',
                                content_type='text/event-stream')

        app = web.Application()
        app.router.add_post('/v1/chat/completions', handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            rows = await run_load(f'http://127.0.0.1:{port}', [dict(id=str(i),
                payload={'finish': finish}, expected_tool=expected)
                for i,finish in enumerate(['length', None, 'tool_calls'])],
                io.StringIO(), mode='closed', workers=1)
            assert [r['functional_ok'] for r in rows] == [False, False, True]
            assert summarize_load(rows, 1)['tool_valid'] == 1
            # Defensive aggregation of historical/caller-provided classifications.
            rows[0]['functional_ok'] = True
            assert summarize_load(rows, 1)['tool_valid'] == 1
        finally:
            await runner.cleanup()
    asyncio.run(exercise())
