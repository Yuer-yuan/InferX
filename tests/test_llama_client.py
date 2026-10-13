import asyncio
import json
import time
import unittest

try:
    from aiohttp import web, ClientSession
except ImportError:
    web = None
try:
    from inferx.llama_client import LlamaClient, create_session
except ImportError:
    LlamaClient = None


@unittest.skipIf(web is None, 'aiohttp integration suite runs in the remote llm environment')
class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.assertIsNotNone(LlamaClient, 'llama client is not implemented')
        self.request = None
        self.closed = asyncio.Event()
        self.mode = 'normal'
        app = web.Application()
        app.router.add_post('/completion', self.completion)
        app.router.add_post('/probe', self.probe)
        self.transports = set()
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.session = create_session()
        self.client = LlamaClient(self.session, f'http://127.0.0.1:{port}')

    async def probe(self, request):
        # Model an HTTP peer closing a pooled connection before accepting
        # the next POST. Retrying POST would risk duplicated inference work.
        transport = request.transport
        if transport in self.transports:
            transport.abort()
        self.transports.add(transport)
        return web.json_response({'status': 'ok'})

    async def test_post_does_not_reuse_closing_connection(self):
        for _ in range(2):
            result = await self.client.request_json('POST', '/probe', {})
            self.assertEqual(result['status'], 'ok')
        self.assertEqual(len(self.transports), 2)

    async def asyncTearDown(self):
        if hasattr(self, 'session'):
            await self.session.close()
            await self.runner.cleanup()

    async def completion(self, request):
        self.request = await request.json()
        if self.mode == 'error':
            return web.json_response({'error': 'broken engine'}, status=500)
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        try:
            pieces = [dict(content='```python\ndef f(): return 1\n```', tokens=[10, 11], stop=False)]
            if self.mode == 'normal':
                pieces.append(dict(content='', tokens=[], stop=True, tokens_predicted=2,
                                   stop_type='eos', timings={'predicted_n': 2}))
            for piece in pieces:
                packet = ('data: ' + json.dumps(piece) + '\n\n').encode()
                await response.write(packet[:7])
                await response.write(packet[7:])
            if self.mode == 'infinite':
                while True:
                    await asyncio.sleep(.005)
                    await response.write(b': keepalive\n\n')
        except (ConnectionResetError, asyncio.CancelledError):
            self.closed.set()
        return response

    async def test_stream_tokens_code_and_sampling(self):
        event = {'_origin': time.monotonic(), 'tokens': 0}
        code = await self.client.generate('prompt', 13, event, 768)
        self.assertEqual(code, 'def f(): return 1')
        self.assertEqual(event['tokens'], 2)
        self.assertGreater(event['first_token'], 0)
        self.assertEqual(self.request['seed'], 13)
        self.assertEqual(self.request['min_p'], 0)
        self.assertEqual(self.request['n_predict'], 768)

    async def test_cancel_closes_network_stream(self):
        self.mode = 'infinite'
        event = {'_origin': time.monotonic(), 'tokens': 0}
        task = asyncio.create_task(self.client.generate('prompt', 13, event, 768))
        for _ in range(100):
            if event['tokens']:
                break
            await asyncio.sleep(.005)
        self.assertEqual(event['tokens'], 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(self.closed.wait(), 1)

    async def test_missing_terminal_and_server_error_fail_trial(self):
        for mode in ['truncated', 'error']:
            self.mode = mode
            with self.assertRaises(RuntimeError):
                await self.client.generate('prompt', 1, {'_origin': time.monotonic(), 'tokens': 0}, 768)
