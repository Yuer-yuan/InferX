"""The real HTTP boundary must not inject llama-only template options into NInfer."""
import asyncio
import io
import json

from aiohttp import web

from inferx.tuning import benchmark


def test_common_benchmark_accepts_strict_template_api():
    async def scenario():
        seen = []

        async def handler(request):
            payload = await request.json()
            seen.append(payload)
            # Observed NInfer response to the initial campaign's request.
            if 'enable_thinking' in payload.get('chat_template_kwargs', {}):
                return web.json_response({'error': {'code': 'chat_template_option_not_supported'}},
                                         status=400)
            event = {'choices': [{'delta': {'content': '5'}, 'finish_reason': 'stop'}],
                     'usage': {'prompt_tokens': 10, 'completion_tokens': 2}}
            return web.Response(text='data: ' + json.dumps(event) + '\n\ndata: [DONE]\n\n',
                                content_type='text/event-stream')

        app = web.Application()
        app.router.add_post('/v1/chat/completions', handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        port = runner.addresses[0][1]
        try:
            output = io.StringIO()
            rows = await benchmark(f'http://127.0.0.1:{port}', 'model',
                                   [{'id': 'sum', 'messages': [{'role': 'user', 'content': '2+3'}]}],
                                   output, 'fixture', 'cold', 1, 1)
            assert rows[0]['summary']['ok'], rows[0]['summary']['errors']
            assert rows[0]['summary']['completion_tokens'] == 2
            assert json.loads(output.getvalue())['summary']['text'] == '5'
            assert seen[0]['messages'] == [{'role': 'user', 'content': '2+3'}]
            assert seen[0]['temperature'] == 0
        finally:
            await runner.cleanup()

    asyncio.run(scenario())
