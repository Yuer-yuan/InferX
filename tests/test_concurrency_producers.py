import asyncio
import io
import json
from collections import Counter

from aiohttp import web

from inferx.concurrency import run_load


def test_closed_loop_preserves_each_producers_dependency_without_stalling_others():
    async def exercise():
        active, peak = Counter(), Counter()
        async def handler(request):
            payload = await request.json()
            producer = payload['producer']
            active[producer] += 1
            peak[producer] = max(peak[producer], active[producer])
            await asyncio.sleep(.02 if producer == 0 else .15)
            active[producer] -= 1
            event = dict(choices=[dict(delta=dict(content='ok'), finish_reason='stop')],
                         usage=dict(prompt_tokens=3, completion_tokens=2))
            return web.Response(text='data: '+json.dumps(event)+'\n\ndata: [DONE]\n\n',
                                content_type='text/event-stream')
        app = web.Application()
        app.router.add_post('/v1/chat/completions', handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        requests = [dict(id=str(i), producer=i%2, payload=dict(producer=i%2)) for i in range(4)]
        try:
            rows = await run_load(f'http://127.0.0.1:{port}', requests, io.StringIO(),
                                  mode='closed', workers=2)
            assert peak == {0:1, 1:1}
            by_id = {r['id']:r for r in rows}
            assert by_id['2']['finished_s'] < by_id['1']['finished_s']
        finally:
            await runner.cleanup()
    asyncio.run(exercise())
