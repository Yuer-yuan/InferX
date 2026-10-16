"""Live integration probe: chat API, streamed tokens and eight-way cancellation."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inferx.llama_client import LlamaClient, create_session


async def main(args):
    async with create_session() as session:
        client = LlamaClient(session, args.url)
        await client.wait_idle()
        chat = await client.request_json('POST', '/v1/chat/completions', {
            'messages': [{'role': 'user', 'content': 'Reply with exactly OK.'}],
            'max_tokens': 16, 'temperature': 0,
            'chat_template_kwargs': {'enable_thinking': False}})
        assert chat['choices'][0]['message']['content'].strip(), chat
        assert chat['usage']['completion_tokens'] > 0, chat
        prompt = await client.prompt('Write 100 separate functions named f001 through f100. '
            'Each returns its number. Explicitly write every function definition; do not use loops to construct them.')
        events = [dict(tokens=0, _origin=time.monotonic()) for _ in range(8)]
        tasks = [asyncio.create_task(client.generate(prompt, 8100 + i, e, 512))
                 for i, e in enumerate(events)]
        try:
            async with asyncio.timeout(30):
                while min(e['tokens'] for e in events) < 4:
                    for task in tasks:
                        if task.done():
                            task.result()
                    await asyncio.sleep(.01)
            before = await client.request_json('GET', '/slots')
            assert any(not task.done() for task in tasks), 'No running request to cancel'
        finally:
            start = time.monotonic()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await client.wait_idle()
        after = await client.request_json('GET', '/slots')
        result = dict(chat_reply=chat['choices'][0]['message']['content'], usage=chat['usage'],
            prompt_tail=prompt[-160:], observed_tokens=[e['tokens'] for e in events],
            active_before=sum(s['is_processing'] for s in before),
            active_after=sum(s['is_processing'] for s in after),
            cancellation_to_idle_s=time.monotonic() - start)
        assert result['active_after'] == 0, result
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:18080')
    parser.add_argument('--output', required=True)
    asyncio.run(main(parser.parse_args()))
