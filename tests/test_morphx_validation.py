"""CPU compilation exercises the patched real dispatch bodies, without CUDA."""
import hashlib
from pathlib import Path
import shutil
import subprocess

import pytest

FIXTURES = Path(__file__).parent/'fixtures'


def original():
    return (FIXTURES/'morphx_launch_original.cpp').read_text()


def test_extended_and_ptsz_dispatch_preserve_config_and_native_semantics(tmp_path):
    from inferx.morphx_validation import patch_extended_launch
    changed = patch_extended_launch(original())
    source = (FIXTURES/'morphx_launch_shim.cpp').read_text().replace('// PATCHED_DISPATCH', changed)
    path=tmp_path/'dispatch.cpp';path.write_text(source)
    compiler=shutil.which('g++')
    assert compiler, 'C++ compiler required for dispatch behavior regression'
    subprocess.run([compiler,'-std=c++17','-Wall','-Wextra','-Werror',str(path),'-o',str(tmp_path/'probe')],check=True,capture_output=True,text=True)
    result=subprocess.run([str(tmp_path/'probe')],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'dispatch oracle passed' in result.stdout


def test_patch_rejects_unknown_or_already_patched_dispatch():
    from inferx.morphx_validation import patch_extended_launch
    with pytest.raises(ValueError,match='launch source'):
        patch_extended_launch('unrelated source')
    with pytest.raises(ValueError,match='launch source'):
        patch_extended_launch(patch_extended_launch(original()))


def source_tree(tmp_path):
    root=tmp_path/'upstream';(root/'runtime/src').mkdir(parents=True);(root/'include').mkdir()
    (root/'runtime/src/cuda-llm.cpp').write_text(original())
    (root/'runtime/src/preload.c').write_text('unchanged preload')
    (root/'include/common.h').write_text('unchanged transport')
    return root


def test_preparation_is_fresh_hashed_and_leaves_upstream_unchanged(tmp_path):
    from inferx.morphx_validation import prepare_runtime
    src=source_tree(tmp_path);before=(src/'runtime/src/cuda-llm.cpp').read_bytes();sha=hashlib.sha256(before).hexdigest()
    dest=tmp_path/'isolated';manifest=prepare_runtime(src,dest,sha)
    assert (src/'runtime/src/cuda-llm.cpp').read_bytes()==before
    assert (dest/'runtime/src/preload.c').read_text()=='unchanged preload'
    assert manifest['before_sha256']==sha and manifest['after_sha256']!=sha
    assert manifest['attributed_launches']=='native; preserve original config'
    assert manifest['ptsz_launches']=='native; preserve per-thread stream semantics'
    assert manifest['normal_unattributed_launches']=='existing MorphX interception'
    for name,expected in manifest['source_sha256'].items():
        assert hashlib.sha256((dest/name).read_bytes()).hexdigest()==expected
    with pytest.raises(FileExistsError):prepare_runtime(src,dest,sha)
    assert (src/'runtime/src/cuda-llm.cpp').read_bytes()==before


def test_preparation_refuses_unpinned_source_before_creating_destination(tmp_path):
    from inferx.morphx_validation import prepare_runtime
    src=source_tree(tmp_path);dest=tmp_path/'isolated'
    with pytest.raises(ValueError,match='source hash'):prepare_runtime(src,dest,'0'*64)
    assert not dest.exists()
    with pytest.raises(ValueError,match='overlap'):
        prepare_runtime(src,src/'child',hashlib.sha256(original().encode()).hexdigest())


def test_profile_waves_cover_4090_nondivisible_sm_counts_exactly(tmp_path):
    from inferx.morphx_validation import patch_profile_wave_accounting
    original=(FIXTURES/'morphx_profile_waves_original.cpp').read_text()
    changed=patch_profile_wave_accounting(original)
    shim='''#include <algorithm>
#include <cassert>
#include <vector>
struct KernelInfo { int gridSize,occup; };
'''+changed+'''
int main() {
 for(int sms: {8,16,80,108,128,132,144})
  for(int occup: {1,2,4,8,16,32})
   for(int grid: {1,7,16,127,128,256,1024,8193,65536}) {
    KernelInfo info{grid,occup};std::vector<int> waves;
    generateProfileWaves(sms,&info,waves);
    int sum=0;for(int w:waves){assert(w>0);sum+=w;}assert(sum==grid);
    waves.clear();generateProfileWaves(sms,&info,waves,0,sms);
    sum=0;for(int w:waves){assert(w>0);sum+=w;}assert(sum==grid);
   }
 KernelInfo info{648*4,4};std::vector<int> waves;
 generateProfileWaves(108,&info,waves);
 assert((waves==std::vector<int>{432,432,384,336,288,240,192,144,96,48}));
}
'''
    source=tmp_path/'profile.cpp';source.write_text(shim)
    subprocess.run(['g++','-std=c++17',str(source),'-o',str(tmp_path/'probe')],check=True,capture_output=True,text=True)
    result=subprocess.run([str(tmp_path/'probe')],capture_output=True,text=True,timeout=5)
    assert result.returncode==0,result.stderr
    with pytest.raises(ValueError):patch_profile_wave_accounting(changed)


def test_float32_named_kernel_is_not_silently_skipped(tmp_path):
    from inferx.morphx_validation import remove_debug_kernel_skip
    block='''    if(strstr(funcName, "float32")) {
        launchMutex.unlock();
        return CUDA_SUCCESS;
        return real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
    }
'''
    changed=remove_debug_kernel_skip(block)
    source=tmp_path/'skip.cpp'
    source.write_text('#include <cstring>\nint calls=0; int run(const char* funcName) {\n'+changed+'++calls;return 73;}\nint main(){return (run("convert_float32")==73 && calls==1)?0:1;}\n')
    subprocess.run(['g++',str(source),'-o',str(tmp_path/'probe')],check=True,capture_output=True,text=True)
    assert subprocess.run([str(tmp_path/'probe')]).returncode==0
    with pytest.raises(ValueError):remove_debug_kernel_skip(changed)


def test_proc_queries_preserve_per_thread_flags_and_native_rejections(tmp_path):
    import re
    from inferx.morphx_validation import patch_proc_address_queries
    original=(FIXTURES/'morphx_proc_address_original.cpp').read_text();changed=patch_proc_address_queries(original)
    names=set(re.findall(r'\*pfn = \(void\*\)(\w+);',original)) - {'cuGetProcAddress','cuGetProcAddress_v2'}
    prefix='''#include <string>
#include <cstring>
#include <cassert>
using CUresult=int;using cuuint64_t=unsigned long;using CUdriverProcAddressQueryResult=int;
const int CUDA_SUCCESS=0, CUDA_ERROR_INVALID_VALUE=1, CU_GET_PROC_ADDRESS_SUCCESS=0;
const int CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM=2;
void* shared_lib=(void*)1;void ld_init(){};void native(){};
CUresult query(const char* s,void** f,int v,cuuint64_t flags,CUdriverProcAddressQueryResult* status){
 if(!s||!f||v==99999||flags>2)return 1;
 *f=(std::string(s)=="missing")?nullptr:(void*)native;
 if(status)*status=(*f)?0:2;return 0;
}
auto real_cuGetProcAddress=query;auto real_cuGetProcAddress_v2=query;
CUresult cuGetProcAddress(const char*,void**,int,cuuint64_t,CUdriverProcAddressQueryResult*);
CUresult cuGetProcAddress_v2(const char*,void**,int,cuuint64_t,CUdriverProcAddressQueryResult*);
'''
    for name in sorted(names):prefix+='void '+name+'(){}\n'
    main='''int main(){
 for(auto fn:{cuGetProcAddress,cuGetProcAddress_v2}) {
  void* f=nullptr;int status=-1;
  assert(fn("cuLaunchKernel",&f,12080,0,&status)==0 && f==(void*)cuLaunchKernel);
  assert(fn("cuLaunchKernel",&f,12080,2,&status)==0 && f==(void*)cuLaunchKernel_ptsz);
  assert(fn("cuLaunchKernelEx",&f,12080,2,&status)==0 && f==(void*)cuLaunchKernelEx_ptsz);
  assert(fn("cuLaunchKernel",&f,99999,0,&status)==1);
  assert(fn("cuLaunchKernel",nullptr,12080,0,&status)==1);
  assert(fn("missing",&f,12080,2,&status)==0 && f==nullptr && status==2);
  assert(fn("cuLaunchKernel",&f,12080,99,&status)==1);
 }
}
'''
    source=tmp_path/'query.cpp';source.write_text(prefix+changed+main)
    subprocess.run(['g++','-std=c++17',str(source),'-o',str(tmp_path/'probe')],check=True,capture_output=True,text=True)
    assert subprocess.run([str(tmp_path/'probe')]).returncode==0
    with pytest.raises(ValueError):patch_proc_address_queries(changed)
