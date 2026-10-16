"""Small, independent numerical probes for Green Context runtime compatibility."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def child(mode):
    from contextlib import nullcontext
    import torch
    from sgl_kernel import spatial
    from inferx.sglang_pd import CudaContextAPI,green_stream,green_wait,green_sync
    assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-4d9bef9d-631c-bb36-3ff4-589d6a184189'
    torch.cuda.init();api=CudaContextAPI()
    host=torch.arange(64,dtype=torch.float32,pin_memory=True)
    shared=host.to('cuda');torch.cuda.synchronize()
    primary=api.current()
    streams=spatial.create_greenctx_stream_by_value(96,32,0)
    result=dict(mode=mode,torch=torch.__version__,primary_context=primary,
        context_after_create=api.current(),streams=[x.cuda_stream for x in streams],
        green_contexts=[api.context_for_stream(x.cuda_stream) for x in streams])
    if mode=='primary-restored':api.set_current(primary)
    before_scope=api.current()
    scope=green_stream if mode=='bound' else torch.cuda.stream
    try:
        with scope(streams[0]):
            result['copy_context']=api.current()
            value=host.to('cuda',non_blocking=True)
            observed=int((value*3+1).sum().cpu().item())
        assert observed==6112,observed
        with scope(streams[0]):shared.mul_(2)
        if mode=='bound':green_wait(streams[1],streams[0])
        else:streams[1].wait_stream(streams[0])
        with scope(streams[1]):second=int((shared+1).sum().cpu().item())
        assert second==4096,second
        if mode=='bound':
            for stream in streams:green_sync(stream)
        else:
            for stream in streams:stream.synchronize()
        result.update(status='passed',copy_oracle=observed,handoff_oracle=second,
                      restored_context=api.current())
        if mode=='bound':assert api.current()==before_scope,'context was not restored'
    except Exception as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    print(json.dumps(result),flush=True)
    return 0 if result['status']=='passed' else 1


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['raw','primary-restored','bound'])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.mode:return child(args.mode)
    if args.output is None:parser.error('--output required')
    rows=[]
    for mode in ['raw','primary-restored','bound']:
        command=[sys.executable,'-m','scripts.probe_pd_green_context','--mode',mode]
        result=subprocess.run(command,capture_output=True,text=True,timeout=75)
        item=dict(mode=mode,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
        try:item['result']=json.loads(result.stdout.splitlines()[-1])
        except (ValueError,IndexError):pass
        rows.append(item)
        args.output.write_text(json.dumps(dict(status='running',modes=rows),indent=2)+'\n')
    qualified=rows[-1].get('result',{}).get('status')=='passed'
    args.output.write_text(json.dumps(dict(status='passed' if qualified else 'failed',
        scope='CUDA host copy, scalar numerical oracle and cross-stream dependency; no model/overlap performance claim',
        modes=rows),indent=2)+'\n')
    print(json.dumps(dict(status='passed' if qualified else 'failed',
        modes=[dict(mode=r['mode'],returncode=r['returncode'],result=r.get('result')) for r in rows])),flush=True)
    return 0 if qualified else 1


if __name__=='__main__':raise SystemExit(main())
