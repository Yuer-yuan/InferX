"""Measure an appended follow-up with the first response retained in conversation."""
import argparse
import asyncio
import json
from pathlib import Path
import aiohttp
from inferx.tuning import stream_request

parser=argparse.ArgumentParser()
parser.add_argument('--corpus',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--config',required=True)
args=parser.parse_args()

async def main():
    prompts=json.loads(args.corpus.read_text())
    timeout=aiohttp.ClientTimeout(total=600,sock_read=300)
    async with aiohttp.ClientSession(timeout=timeout,
                                    connector=aiohttp.TCPConnector(force_close=True)) as session:
        with args.output.open('x') as out:
            for prompt in prompts:
                messages=list(prompt['messages'])
                for turn in range(2):
                    payload=dict(model='qwen38-27b',messages=list(messages),temperature=0,top_p=1,
                        presence_penalty=0,frequency_penalty=0,seed=42,max_tokens=1024,
                        stream=True,stream_options={'include_usage':True},cache_prompt=True)
                    result=await stream_request(session,'http://127.0.0.1:18081',payload,'natural')
                    s=result['summary']
                    row=dict(kind='trial',config=args.config,prompt_id=prompt['id']+f'_turn{turn}',
                        repetition=0,concurrency=1,index=0,wave_s=s['wall_s'],**result)
                    out.write(json.dumps(row)+'\n');out.flush()
                    print(prompt['id'],turn,s['ok'],s['ttft_s'],s['cached_tokens'],s['prompt_tokens'],flush=True)
                    if not s['ok'] or s['truncated']:raise RuntimeError('turn failed or truncated')
                    messages += [dict(role='assistant',content=s['text']),dict(role='user',content=
                        'Update the parser to include the original 1-based source line number in each parsed dict, and reject records with negative timeout_ms. Return only the complete Python code.')]
asyncio.run(main())
