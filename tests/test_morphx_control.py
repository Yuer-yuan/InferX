"""Behavioral control-transport contracts, with the pinned historical runtime."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

ROOT=Path(__file__).parents[1]


def historical_source():
    with tarfile.open(ROOT/'results/sota-validation/reproduction/source-capsule-v2.tar.gz') as t:
        return t.extractfile('./sota-validation/runtime-cuda-copy-v2/runtime/src/cuda-llm.cpp').read().decode()


def corrected_source():
    from inferx.morphx_validation import upgrade_proc_address_queries
    return upgrade_proc_address_queries(historical_source())


def test_native_async_fill_and_event_lifetime_against_driver_oracle(tmp_path):
    header=ROOT/'scripts/morphx_control_ops.hpp'
    assert header.exists(),'transport helper missing'
    prefix=r'''
#include <cassert>
#include <cstdint>
#include <functional>
#include <vector>
using CUresult=int;using CUdeviceptr=uintptr_t;using CUstream=void*;
struct Event{bool done=false;};using CUevent=Event*;
constexpr int CUDA_SUCCESS=0,CUDA_ERROR_INVALID_HANDLE=400,CUDA_ERROR_NOT_READY=600;
constexpr int CUDA_ERROR_NOT_SUPPORTED=801,CUDA_ERROR_INVALID_VALUE=1,CU_EVENT_DISABLE_TIMING=2;
CUstream model=(void*)0x123;std::vector<std::function<void()>> queue;
int fills=0,created=0,destroyed=0,queries=0,record_error=0,query_error=0;
bool launched=false;
CUresult fill(CUdeviceptr d,unsigned value,size_t n,CUstream s){
 assert(n==1 && s==model);++fills;queue.push_back([=]{*(uint32_t*)d=value;});return 0;}
CUresult create(CUevent* p,unsigned flags){assert(flags==CU_EVENT_DISABLE_TIMING);*p=new Event;++created;return 0;}
CUresult record(CUevent e,CUstream s){assert(launched && s==model);if(record_error)return record_error;queue.push_back([=]{e->done=true;});return 0;}
CUresult query(CUevent e){++queries;if(query_error)return query_error;return e->done?0:CUDA_ERROR_NOT_READY;}
CUresult destroy(CUevent e){delete e;++destroyed;return 0;}
void drain(){for(auto& op:queue)op();queue.clear();}
'''
    main=r'''
int main(){
 using namespace inferx_control;Driver driver{fill,create,record,query,destroy};assert(driver.valid());
 Driver missing=driver;missing.fill32=nullptr;assert(!missing.valid());
 for(unsigned value:{128u,768u,1024u}){
   uint32_t agents=0xdeadbeef,observed=0;
   assert(initialize_agents(driver,(CUdeviceptr)&agents,value,model)==0);
   assert(agents==0xdeadbeef); // exact value is queued, never a pageable host copy
   queue.push_back([&]{observed=agents;});drain();assert(observed==value);
 }
 Completion completion;bool done=true;
 assert(completion.poll(driver,done)==CUDA_ERROR_INVALID_HANDLE && !done && queries==0);
 assert(completion.record_after_launch(driver,model,7)==7 && created==0);
 launched=true;assert(completion.record_after_launch(driver,model,0)==0);
 assert(completion.poll(driver,done)==0 && !done);
 assert(completion.release(driver)==CUDA_ERROR_NOT_READY && destroyed==0);
 // SM scheduler may continue while kernel event is not ready.
 int sm_changes=0;if(!done)++sm_changes;assert(sm_changes==1);
 query_error=8;assert(completion.poll(driver,done)==8 && !done && destroyed==0);query_error=0;
 drain();assert(completion.poll(driver,done)==0 && done);
 assert(completion.release(driver)==0 && destroyed==1);
 assert(completion.release(driver)==CUDA_ERROR_INVALID_HANDLE && destroyed==1);
 Completion failed;record_error=9;
 assert(failed.record_after_launch(driver,model,0)==9 && created==2 && destroyed==2);
 assert(failed.poll(driver,done)==CUDA_ERROR_INVALID_HANDLE && !done);
 assert(initialize_agents(missing,0,128,model)==CUDA_ERROR_NOT_SUPPORTED);
 assert(fills==3);
}
'''
    source=tmp_path/'oracle.cpp';source.write_text(prefix+header.read_text()+main)
    subprocess.run(['g++','-std=c++17',str(source),'-o',str(tmp_path/'oracle')],check=True,capture_output=True,text=True)
    subprocess.run([str(tmp_path/'oracle')],check=True,capture_output=True,text=True,timeout=10)


@pytest.mark.parametrize('fill,event',[(True,False),(False,True),(True,True)])
def test_pinned_patch_keeps_profiler_native_dispatch_and_dynamic_scheduler(fill,event):
    from inferx.morphx_control import patch_control_runtime
    source=corrected_source();changed=patch_control_runtime(source,device_fill=fill,event_completion=event)
    for start,end in [('void profileThread(','bool isCI('),('void schedule_2(','void agentThread('),
                      ('static void inferxTraceNative(', 'CUresult cuModuleLoad(')]:
        # Full native/query section remains byte-identical, including correctedABI.
        before=source[source.index(start):source.index(end,source.index(start))]
        assert before in changed
    if fill:
        assert 'initialize_agents(inferxControlDriver(),' in changed
        assert 'cudaMemcpyAsync(launchInfo->d_agents, &agents' not in changed
    else:assert 'cudaMemcpyAsync(launchInfo->d_agents, &agents' in changed
    agent=changed[changed.index('void agentThread('):changed.index('StreamInfo* registerStream(')]
    if event:
        assert 'completion.poll(inferxControlDriver(), complete)' in agent
        assert 'gGdrPool->get(kernels[i]->fetched)' not in agent
        assert agent.index('if(complete)')<agent.index('gGdrPool->gdr_free')
        assert changed.index('completion.record_after_launch(')<changed.index('if(!profile && !streamInfo->pending->push(launchInfo))')
    else:assert 'gGdrPool->get(kernels[i]->fetched)' in agent
    with pytest.raises(ValueError):patch_control_runtime(changed,device_fill=fill,event_completion=event)


def source_tree(tmp_path):
    source=tmp_path/'source';(source/'runtime/src').mkdir(parents=True);(source/'include').mkdir()
    (source/'runtime/src/cuda-llm.cpp').write_text(historical_source());(source/'include/common.h').write_text('other pinned source')
    hashes={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
    (source/'manifest.json').write_text(json.dumps(dict(source_sha256=hashes,library_sha256='64f3268499a920e30db33d48b8ecee9137216e3dd494e0a93d0e2119a542af3a',command=[])))
    return source


def test_preparation_rejects_changed_sources_before_creating_output(tmp_path):
    from inferx.morphx_control import prepare_control_runtime
    source=source_tree(tmp_path);destination=tmp_path/'fresh'
    (source/'include/common.h').write_text('changed')
    with pytest.raises(ValueError,match='hash'):prepare_control_runtime(source,destination,device_fill=True,event_completion=True)
    assert not destination.exists()


def test_preparation_preserves_old_files_and_rejects_overwrite_and_overlap(tmp_path):
    from inferx.morphx_control import prepare_control_runtime
    source=source_tree(tmp_path);destination=tmp_path/'fresh'
    before=(source/'runtime/src/cuda-llm.cpp').read_bytes()
    result=prepare_control_runtime(source,destination,device_fill=True,event_completion=True)
    assert (source/'runtime/src/cuda-llm.cpp').read_bytes()==before
    assert (destination/'include/morphx_control_ops.hpp').is_file()
    assert result['base_library_sha256']=='64f3268499a920e30db33d48b8ecee9137216e3dd494e0a93d0e2119a542af3a'
    assert all(hashlib.sha256((destination/n).read_bytes()).hexdigest()==h for n,h in result['source_sha256'].items())
    with pytest.raises(FileExistsError):prepare_control_runtime(source,destination,device_fill=True,event_completion=True)
    with pytest.raises(ValueError,match='overlap'):prepare_control_runtime(source,source/'child',device_fill=True,event_completion=True)
