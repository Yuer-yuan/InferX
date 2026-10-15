"""Independent native-vLLM MB grid for the author's Phase-C cost fit."""
import asyncio
import copy
import hashlib
import json
import random
import time
import uuid
from pathlib import Path

import aiohttp

from inferx.ebplus_calibration import fit_mb_cost
from scripts.run_pd_baselines import OwnedEngine,wait_idle,ENDPOINT
from scripts import run_concurrency_campaign as previous


async def point(scene,count,raw):
    rng=random.Random(0)
    requests=[dict(model='qwen38-27b',prompt=[rng.randrange(100,10000) for _ in range(scene['input'])],
                   max_tokens=scene['output'],ignore_eos=True,temperature=0,stream=False,
                   cache_salt=uuid.uuid4().hex) for _ in range(count)]
    timeout=aiohttp.ClientTimeout(total=900)
    async with aiohttp.ClientSession(timeout=timeout) as client:
        async def one(body):
            started=time.perf_counter()
            async with client.post(ENDPOINT+'/v1/completions',json=body) as response:
                text=await response.text()
                item=dict(request=body,status=response.status,response_text=text,
                          duration_s=time.perf_counter()-started)
                raw.write(json.dumps(item)+'\n');raw.flush()
                if response.status!=200:raise RuntimeError('MB calibration request failed; keep raw')
                data=json.loads(text);usage=data['usage']
                if usage['prompt_tokens']!=scene['input'] or usage['completion_tokens']!=scene['output']:
                    raise RuntimeError('MB calibration token geometry changed; keep raw')
                return usage
        started=time.perf_counter();values=await asyncio.gather(*(one(body) for body in requests))
        duration=time.perf_counter()-started
    total_input=sum(x['prompt_tokens'] for x in values);total_output=sum(x['completion_tokens'] for x in values)
    return dict(scenario=scene['name'],count=count,completed=len(values),duration_s=duration,
                prompt_tokens=total_input,completion_tokens=total_output,
                total_token_throughput=(total_input+total_output)/duration)


def run_mb_grid(plan,root,out,progress):
    grid=plan['mb_grid'];rows=[]
    metadata=dict(status='running',identity=plan['identity'],rows=rows,
                  measurement_type='native_HTTP_total_token_throughput',started_unix=time.time(),
                  worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='C4 ceiling;1/2/4 simultaneous requests per point; one fresh wave per grid point; independent from formal traffic.')
    def flush():
        (out/'mb-grid-result.json').write_text(json.dumps(metadata,indent=2)+'\n')
    flush()
    try:
        for budget in grid['budgets']:
            for cap in grid['prefill_caps']:
                previous.gpu_free()
                case=copy.deepcopy(plan['base_case'])
                case.update(id=f'mb-b{budget}-cap{cap}',chunk=budget,long_prefill_token_threshold=cap,
                            async_scheduling=False)
                entry=dict(case=case['id'],state='starting');progress(entry)
                with OwnedEngine(case,root,out/case['id']) as server:
                    wait_idle()
                    with (server.directory/'warmup.jsonl').open('x') as raw:
                        asyncio.run(point(dict(name='warmup',input=512,output=32),4,raw))
                    for scene in grid['scenarios']:
                        for count in grid['counts']:
                            wait_idle()
                            name=f"{scene['name']}-n{count}.jsonl"
                            with (server.directory/name).open('x') as raw:
                                row=asyncio.run(point(scene,count,raw))
                            row.update(budget=budget,cap=cap,raw=str((server.directory/name).relative_to(out)))
                            rows.append(row);flush()
                entry['state']='measured';progress(None)
        metadata.update(status='complete',fit=fit_mb_cost(rows,plan))
    except BaseException as exc:
        metadata.update(status='failed',error=repr(exc));raise
    finally:
        metadata['finished_unix']=time.time();flush()
    return metadata
