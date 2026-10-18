"""Disk-only transport observer around untouched original EvalScope runner."""
import datetime
import hashlib
import itertools
import json
import os
import pathlib
import runpy
import sys
import threading
import time
import traceback

import httpx

ROOT = pathlib.Path(__file__).resolve().parent
ATTEMPTS = ROOT/'http-attempts'
ATTEMPTS.mkdir(exist_ok=False)
counter = itertools.count(1)
lock = threading.Lock()


def write(p, obj):
    temp = p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    os.replace(temp,p)


def begin(request):
    with lock:
        index=next(counter)
    folder=ATTEMPTS/f'{index:05d}'
    folder.mkdir(exist_ok=False)
    body=request.content
    (folder/'request.raw.bin').write_bytes(body)
    original_timeouts=request.extensions.get('timeout')
    timeout=dict(connect=120.0,read=7200.0,write=120.0,pool=7200.0)
    request.extensions['timeout']=timeout
    meta=dict(attempt=index,pid=os.getpid(),method=request.method,url=str(request.url),
        start_wall_ns=time.time_ns(),start_monotonic_ns=time.monotonic_ns(),
        request_body_sha256=hashlib.sha256(body).hexdigest(),
        headers={k:v for k,v in request.headers.items() if k.lower() not in ['authorization','api-key','x-api-key','cookie']},
        original_transport_timeout=original_timeouts,effective_transport_timeout=timeout,
        source='original httpx request forwarded once; body unchanged')
    if request.method=='POST' and request.url.path.endswith('/chat/completions'):
        payload=json.loads(body)
        write(folder/'payload.json',payload)
        assert payload.get('max_tokens')==32768,payload.keys()
        assert payload.get('model')=='qwen38-27b'
        for key in ['temperature','top_p','top_k','seed','reasoning_effort','chat_template_kwargs']:
            assert payload.get(key) is None,(key,payload.get(key))
    write(folder/'attempt.json',meta)
    return folder,meta


def complete(folder,meta,response):
    data=response.content
    (folder/'response.raw.bin').write_bytes(data)
    meta.update(end_wall_ns=time.time_ns(),end_monotonic_ns=time.monotonic_ns(),
                status_code=response.status_code,response_bytes=len(data),
                response_sha256=hashlib.sha256(data).hexdigest(),
                response_headers={k:v for k,v in response.headers.items() if k.lower() not in ['set-cookie']})
    try:
        obj=json.loads(data)
        write(folder/'response.json',obj)
        meta.update(native_response_id=obj.get('id'),usage=obj.get('usage'))
    except (ValueError,AttributeError):
        pass
    write(folder/'attempt.json',meta)


def failed(folder,meta,error):
    meta.update(end_wall_ns=time.time_ns(),end_monotonic_ns=time.monotonic_ns(),
                error=repr(error),traceback=traceback.format_exc())
    write(folder/'attempt.json',meta)


original_async=httpx.AsyncClient.send
async def observed_async(self,request,*args,**kwargs):
    folder,meta=begin(request)
    try:
        response=await original_async(self,request,*args,**kwargs)
        await response.aread()
        complete(folder,meta,response)
        return response
    except BaseException as error:
        failed(folder,meta,error)
        raise


original_sync=httpx.Client.send
def observed_sync(self,request,*args,**kwargs):
    folder,meta=begin(request)
    try:
        response=original_sync(self,request,*args,**kwargs)
        response.read()
        complete(folder,meta,response)
        return response
    except BaseException as error:
        failed(folder,meta,error)
        raise


httpx.AsyncClient.send=observed_async
httpx.Client.send=observed_sync
runner=pathlib.Path('/workspace/InferX/code/submission/organizer/runner.py')
assert hashlib.sha256(runner.read_bytes()).hexdigest()=='d80df390cf6d1d50129796b31fdd7f4d1e95d0043d3b942d47ac582118702fb0'
sys.argv=[str(runner),'_accuracy',str(ROOT/'task_config.json')]
runpy.run_path(str(runner),run_name='__main__')
