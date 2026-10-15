"""Explicit MorphX launch-semantics qualification, not a full PD reimplementation."""
import hashlib
import json
from pathlib import Path
import shutil


_ORIGINALS = {'cuLaunchKernelEx': 'CUresult cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra) {\n    if (shared_lib == NULL) {\n        // fprintf(event_log, "[mod] cuLaunchKernelEx called before ld_init, calling ld_init now...\\n");\n        ld_init();\n    }\n\n    if(config == nullptr) {\n        return CUDA_ERROR_INVALID_VALUE;\n    }\n    return cuLaunchKernel(\n        f,\n        config->gridDimX,\n        config->gridDimY,\n        config->gridDimZ,\n        config->blockDimX,\n        config->blockDimY,\n        config->blockDimZ,\n        config->sharedMemBytes,\n        config->hStream,\n        kernelParams,\n        extra);\n}', 'cuLaunchKernel_ptsz': 'CUresult cuLaunchKernel_ptsz(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,\n    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes,\n    CUstream hStream, void** kernelParams, void** extra) {\n    return cuLaunchKernel(\n        f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);\n}', 'cuLaunchKernelEx_ptsz': 'CUresult cuLaunchKernelEx_ptsz(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra) {\n    return cuLaunchKernelEx(config, f, kernelParams, extra);\n}'}

_TRACE = r"""
static void inferxTraceNative(const char* phase, const CUlaunchConfig* config, CUfunction f) {
    const char* name = "unresolved";
    std::lock_guard<std::mutex> guard(launchMutex);
    if (real_cuFuncGetName) real_cuFuncGetName(&name, f);
    traceLLMKernel(phase, name,
        config->gridDimX * config->gridDimY * config->gridDimZ,
        config->blockDimX, config->blockDimY, config->blockDimZ,
        config->sharedMemBytes, config->hStream);
}
"""

_NORMAL_EX = r"""CUresult cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra) {
    if (shared_lib == NULL) ld_init();
    if (config == nullptr) return CUDA_ERROR_INVALID_VALUE;
    if (config->numAttrs != 0) {
        // Cooperative/grid-sync and unknown attributes must retain native launch semantics.
        if (!real_cuLaunchKernelEx) return CUDA_ERROR_NOT_SUPPORTED;
        inferxTraceNative("attrs-native", config, f);
        return real_cuLaunchKernelEx(config, f, kernelParams, extra);
    }
    return cuLaunchKernel(f, config->gridDimX, config->gridDimY, config->gridDimZ,
        config->blockDimX, config->blockDimY, config->blockDimZ,
        config->sharedMemBytes, config->hStream, kernelParams, extra);
}"""

_PTSZ = r"""CUresult cuLaunchKernel_ptsz(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes,
    CUstream hStream, void** kernelParams, void** extra) {
    if (shared_lib == NULL) ld_init();
    static auto native = reinterpret_cast<cuLaunchKernel_func_t>(dlsym(shared_lib, "cuLaunchKernel_ptsz"));
    if (!native) return CUDA_ERROR_NOT_SUPPORTED;
    CUlaunchConfig config = {};
    config.gridDimX=gridDimX; config.gridDimY=gridDimY; config.gridDimZ=gridDimZ;
    config.blockDimX=blockDimX; config.blockDimY=blockDimY; config.blockDimZ=blockDimZ;
    config.sharedMemBytes=sharedMemBytes; config.hStream=hStream;
    inferxTraceNative("ptsz-native", &config, f);
    return native(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
        sharedMemBytes, hStream, kernelParams, extra);
}"""

_PTSZ_EX = r"""CUresult cuLaunchKernelEx_ptsz(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra) {
    if (shared_lib == NULL) ld_init();
    if (config == nullptr) return CUDA_ERROR_INVALID_VALUE;
    static auto native = reinterpret_cast<cuLaunchKernelEx_func_t>(dlsym(shared_lib, "cuLaunchKernelEx_ptsz"));
    if (!native) return CUDA_ERROR_NOT_SUPPORTED;
    inferxTraceNative("ptsz-ex-native", config, f);
    return native(config, f, kernelParams, extra);
}"""


def patch_extended_launch(text: str) -> str:
    for original in _ORIGINALS.values():
        if text.count(original) != 1:
            raise ValueError('unexpected or already patched launch source')
    replacements = [_TRACE + _NORMAL_EX, _PTSZ, _PTSZ_EX]
    for original, replacement in zip(_ORIGINALS.values(), replacements):
        text = text.replace(original, replacement)
    return text


def prepare_runtime(source: Path, destination: Path, expected_sha256: str) -> dict:
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError('source and destination overlap')
    target = Path('runtime/src/cuda-llm.cpp')
    before = (source/target).read_bytes()
    if hashlib.sha256(before).hexdigest() != expected_sha256:
        raise ValueError('runtime source hash mismatch')
    patched = patch_extended_launch(before.decode())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    try:
        (destination/'runtime').mkdir()
        shutil.copytree(source/'runtime/src', destination/'runtime/src')
        shutil.copytree(source/'include', destination/'include')
        (destination/target).write_text(patched)
        profile = Path(__file__).parents[1]/'scripts/morphx_profile_table.hpp'
        shutil.copyfile(profile, destination/'include/morphx_profile_table.hpp')
        hashes = {str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(destination.rglob('*')) if p.is_file()}
        manifest = dict(source=str(source), before_sha256=expected_sha256,
            after_sha256=hashes[str(target)], source_sha256=hashes,
            attributed_launches='native; preserve original config',
            ptsz_launches='native; preserve per-thread stream semantics',
            normal_unattributed_launches='existing MorphX interception',
            coverage='Trace phases identify observed native/transformed kernel names, not a complete launch/time fraction.',
            limitations=['Mapped-host control transport differs from author GDRCopy.',
                'A kernel runtime correction alone is not author PD admission/scheduling.'])
        (destination/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        return manifest
    except BaseException:
        shutil.rmtree(destination)
        raise


def patch_profile_wave_accounting(text: str) -> str:
    """Preserve author's descending SM samples; count their exact integer sum.

    128SM/9 truncates to14: the original loop has ten levels, while its
    closed-form sum assumes nine. This aborts profiling on an RTX4090.
    """
    replacements={
        '        int dec = nrSM / baseWaves;\n        int baseBlocks = (dec + nrSM) * info->occup * baseWaves / 2;':
        '        int dec = std::max(1, nrSM / baseWaves);\n        int baseBlocks = 0;\n        for(int k = nrSM; k > 0; k -= dec) baseBlocks += k * info->occup;',
        '        for(int k = nrSM * info->occup; k; k -= dec * info->occup) {':
        '        for(int k = nrSM * info->occup; k > 0; k -= dec * info->occup) {',
        '        profileWaves.push_back(gridSize);\n        return;':
        '        if(gridSize > 0) profileWaves.push_back(gridSize);\n        return;',
    }
    for old in replacements:
        if text.count(old)!=1:raise ValueError('unexpected or already patched profile source')
    for old,new in replacements.items():text=text.replace(old,new)
    return text


def remove_debug_kernel_skip(text):
    old='''    if(strstr(funcName, "float32")) {
        launchMutex.unlock();
        return CUDA_SUCCESS;
        return real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
    }
'''
    if text.count(old)!=1:raise ValueError('unexpected or already patched debug skip')
    return text.replace(old,'    // All kernel names retain execution; no upstream debug-only success return.\n')


def patch_proc_address_queries(text):
    """Honor native version/flag validation before substituting hooked entries."""
    for name in ['cuGetProcAddress_v2','cuGetProcAddress']:
        signature=f'CUresult {name}(const char* symbol, void** pfn, int'
        start=text.find(signature)
        if start<0:raise ValueError('unexpected query source')
        point=text.find('    std::string sym_str(symbol);',start)
        if point<0 or 'inferx_query_validation' in text[start:point]:raise ValueError('unexpected or already patched query source')
        validation=f'''    // inferx_query_validation: retain native ABI/version/flag errors and status.
    CUresult validated = real_{name}(symbol, pfn, cudaVersion, flags, symbolStatus);
    if(validated != CUDA_SUCCESS || pfn == nullptr || *pfn == nullptr) return validated;
    if(flags == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM) {{
        if(strcmp(symbol, "cuLaunchKernel") == 0) {{
            *pfn = (void*)cuLaunchKernel_ptsz;
            return CUDA_SUCCESS;
        }}
        if(strcmp(symbol, "cuLaunchKernelEx") == 0) {{
            *pfn = (void*)cuLaunchKernelEx_ptsz;
            return CUDA_SUCCESS;
        }}
    }}
'''
        text=text[:point]+validation+text[point:]
    return text
