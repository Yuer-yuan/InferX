// Launch dispatch excerpts from MorphX4404aed, including prior local default/capture port.

CUresult cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra) {
    if (shared_lib == NULL) {
        // fprintf(event_log, "[mod] cuLaunchKernelEx called before ld_init, calling ld_init now...\n");
        ld_init();
    }

    if(config == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return cuLaunchKernel(
        f,
        config->gridDimX,
        config->gridDimY,
        config->gridDimZ,
        config->blockDimX,
        config->blockDimY,
        config->blockDimZ,
        config->sharedMemBytes,
        config->hStream,
        kernelParams,
        extra);
}

CUresult cuLaunchKernel_ptsz(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes,
    CUstream hStream, void** kernelParams, void** extra) {
    return cuLaunchKernel(
        f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
}

CUresult cuLaunchKernelEx_ptsz(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra) {
    return cuLaunchKernelEx(config, f, kernelParams, extra);
}
