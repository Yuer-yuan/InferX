"""Native llama.cpp streaming API, preserving observed tokens on cancellation."""
import asyncio
import hashlib
import json
import re
import time


def extract_code(text):
    block = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (block.group(1) if block else text).strip()


class LlamaClient:
    def __init__(self, session, url):
        self.session = session
        self.url = url.rstrip('/')

    async def request_json(self, method, path, payload=None):
        async with self.session.request(method, self.url + path, json=payload) as response:
            if response.status != 200:
                raise RuntimeError(f'engine HTTP {response.status}: ' + (await response.text())[:1000])
            return await response.json()

    async def prompt(self, specification):
        messages = [dict(role='system', content='You are a helpful coding assistant.'),
                    dict(role='user', content='Implement the following Python function according to its specification. Use only the Python standard library. Return a complete Python module including any imports, in a single Python code block. Do not include tests or explanations.\n\n' + specification)]
        result = await self.request_json('POST', '/apply-template', {
            'messages': messages, 'chat_template_kwargs': {'enable_thinking': False}})
        return result['prompt']

    async def generate(self, prompt, seed, event, max_tokens):
        payload = dict(prompt=prompt, seed=seed, n_predict=max_tokens, temperature=.8,
                       top_p=.95, top_k=0, min_p=0, repeat_penalty=1,
                       stream=True, return_tokens=True, cache_prompt=True)
        chunks = []
        stopped = False
        async with self.session.post(self.url + '/completion', json=payload) as response:
            if response.status != 200:
                raise RuntimeError(f'engine HTTP {response.status}: ' + (await response.text())[:1000])
            async for line in response.content:
                if not line.startswith(b'data: '):
                    continue
                item = json.loads(line[6:])
                if 'error' in item:
                    raise RuntimeError(str(item['error']))
                chunks.append(item.get('content', ''))
                event['tokens'] += len(item.get('tokens', []))
                if event['tokens'] and 'first_token' not in event:
                    event['first_token'] = time.monotonic() - event['_origin']
                if item.get('stop'):
                    stopped = True
                    event['tokens'] = item.get('tokens_predicted', event['tokens'])
                    event['engine_timings'] = item.get('timings', {})
                    event['finish_reason'] = item.get('stop_type')
        if not stopped:
            raise RuntimeError('engine stream ended without a terminal response')
        text = ''.join(chunks)
        event['raw_output'] = text
        code = extract_code(text)
        event['code_sha256'] = hashlib.sha256(code.encode()).hexdigest()
        return code

    async def wait_idle(self, timeout=10):
        start = time.monotonic()
        while True:
            slots = await self.request_json('GET', '/slots')
            if not any(slot['is_processing'] for slot in slots):
                return time.monotonic() - start
            if time.monotonic() - start >= timeout:
                raise RuntimeError('engine slots still active after cancellation')
            await asyncio.sleep(.01)
