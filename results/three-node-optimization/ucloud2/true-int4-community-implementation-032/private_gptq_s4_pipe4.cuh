#pragma once
// Private fixed four-stage S4/H512/G512 prototype; parent C128 arithmetic unchanged.
// Mechanism reference: efeslab/Atom 7e3618b1a7a7c86e1c93cc909b1510c046d76ac6,
// STAGE-1 prologue / wait_group(STAGE-2) pipeline. No external source lines copied.
// Atom's FP16 scales/output/default stream and INT8keepers are not imported.
// Paired H512 and group512 W4/A4. Exact S32 per group, ten FP32 FMA updates.
#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ucloud2::group512_s4h512_pipe4 {
using ninfer::ops::cp_commit;
using ninfer::ops::cp_wait;
using ninfer::ops::ldmatrix_x4;

using ninfer::ops::smem_addr;
constexpr int K=5120, N=34816, I=17408;
constexpr int Rows=128, Tokens=128, StepK=128, StepBytes=64, Threads=256, Steps=40;
constexpr int OutRows=64, OutLd=72, MT=4;
struct alignas(128) Stage { std::uint8_t w[Rows*StepBytes]; std::uint8_t x[Tokens*StepBytes]; };
struct alignas(128) Shared { Stage stage[4]; uint8_t output_pad[2048]; float ws[10*Rows]; float xs[10*Tokens]; };
constexpr std::size_t SharedBytes=sizeof(Shared);
static_assert(sizeof(Stage)==16384 && SharedBytes==77824);
static_assert(offsetof(Shared,ws)==67584 && offsetof(Shared,xs)==72704);
static_assert(sizeof(Stage)+Tokens*OutLd*sizeof(__nv_bfloat16)<=offsetof(Shared,ws));
static_assert(std::int64_t(K)*7*7<INT32_MAX);

__host__ __device__ constexpr int weight_row(int block,int local) {
    return block*64+(local>>6)*32+(local&31)+((local>>5)&1)*I;
}
__host__ __device__ constexpr unsigned swizzled(int line,int chunk) {
    return unsigned(line*StepBytes+((chunk^((line>>1)&3))<<4));
}

__device__ __forceinline__ void mma_s4(int& d0,int& d1,int& d2,int& d3,unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1){
 asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n":"+r"(d0),"+r"(d1),"+r"(d2),"+r"(d3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

// Debug is the same computation plus complete raw-S32 stores. Timing instantiation
// Debug=false has no global raw tensor read or write, and receives raw=nullptr.
template<bool Debug>
__global__ void __launch_bounds__(Threads,1) kernel(
        const std::uint8_t* __restrict__ W, const float* __restrict__ ws,
        const std::uint8_t* __restrict__ X, const float* __restrict__ xs,
        __nv_bfloat16* __restrict__ Y, std::int32_t* __restrict__ raw,
        float* __restrict__ pre,int debug_group,int tokens,int token_tiles,int total_tiles) {
    extern __shared__ __align__(128) std::uint8_t storage[];
    auto& sh=*reinterpret_cast<Shared*>(storage);
    const int tid=int(threadIdx.x),warp=tid>>5,lane=tid&31;
    const int wm=warp%2,wn=warp/2,gid=lane>>2,lid=lane&3;
    const int line0=tid>>2,chunk=tid&3;
    int tile=int(blockIdx.x),row_block=0,token0=0,live=0;
    // Fixed source sizes fit uint32; pointer addition still uses the full GPU pointer.
    std::uint32_t woff[2],xoff[2];
    const auto set_tile=[&](int t) {
        row_block=t/token_tiles;token0=(t%token_tiles)*Tokens;
        live=min(Tokens,tokens-token0);
#pragma unroll
        for(int i=0;i<2;++i){
            int line=line0+i*64;
            woff[i]=std::uint32_t(weight_row(row_block,line)*(K/2)+chunk*16);
            xoff[i]=std::uint32_t((token0+(line<live?line:0))*(K/2)+chunk*16);
        }
    };
    const auto issue_step=[&](int step,Stage& s) {
        const unsigned wbase=smem_addr(s.w)+swizzled(line0,chunk);
        const unsigned xbase=smem_addr(s.x)+swizzled(line0,chunk);
#pragma unroll
        for(int i=0;i<2;++i){
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::
                "r"(wbase+i*64*StepBytes),"l"(W+woff[i]+step*StepBytes));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"::
                "r"(xbase+i*64*StepBytes),"l"(X+xoff[i]+step*StepBytes),
                "r"(line0+i*64<live?16:0));
        }
    };
    const auto issue_scales=[&] {
        // Group-major shared cache, row-major global scales. All ten groups live
        // through this tile; next-tile scale publication follows the epilogue barrier.
        for(int group=0;group<10;++group){
            if(tid<Rows){
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::
                    "r"(smem_addr(sh.ws+group*Rows+tid)),"l"(ws+weight_row(row_block,tid)*10+group));
            }else{
                int token=tid-Rows;
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"::
                    "r"(smem_addr(sh.xs+group*Tokens+token)),"l"(xs+(token0+(token<live?token:0))*10+group),"r"(token<live?4:0));
            }
        }
    };
    // Same canonical C128 ldmatrix address mapping, packed signed4 operands with64logical K per32bytes.
    const unsigned a_off=unsigned(offsetof(Stage,w))+
        swizzled(wm*64+(lane&7)+((lane>>3)&1)*8,lane>>4);
    const unsigned b_off=unsigned(offsetof(Stage,x))+
        swizzled(wn*32+(lane>>4)*8+(lane&7),(lane>>3)&1);
    set_tile(tile);issue_step(0,sh.stage[0]);issue_scales();cp_commit();
    issue_step(1,sh.stage[1]);cp_commit();
    issue_step(2,sh.stage[2]);cp_commit();
    for(;;){
        int g[MT][4][4]; float f[MT][4][4];
#pragma unroll
        for(int mt=0;mt<MT;++mt)
#pragma unroll
            for(int nt=0;nt<4;++nt)
#pragma unroll
                for(int e=0;e<4;++e){g[mt][nt][e]=0;f[mt][nt][e]=0.f;}
        for(int step=0;step<Steps;++step){
            // Three committed stages ahead; tail drains remaining groups explicitly.
            // The barrier retires all readers before the producer wraps modulo four.
            if(step<Steps-2)cp_wait<2>();
            else if(step==Steps-2)cp_wait<1>();
            else cp_wait<0>();
            __syncthreads();
            if(step+3<Steps){issue_step(step+3,sh.stage[(step+3)&3]);cp_commit();}
            const unsigned base=smem_addr(&sh.stage[step&3]);
#pragma unroll
            for(int ks=0;ks<2;++ks){
                const unsigned slice=unsigned(ks)<<5;
                unsigned b[4][2];
#pragma unroll
                for(int np=0;np<2;++np)
                    ldmatrix_x4(b[2*np][0],b[2*np][1],b[2*np+1][0],b[2*np+1][1],
                        base+(b_off^slice)+np*16*StepBytes);
#pragma unroll
                for(int mt=0;mt<MT;++mt){
                    unsigned a0,a1,a2,a3;
                    ldmatrix_x4(a0,a1,a2,a3,base+(a_off^slice)+mt*16*StepBytes);
#pragma unroll
                    for(int nt=0;nt<4;++nt){int* d=g[mt][nt];
                        mma_s4(d[0],d[1],d[2],d[3],a0,a1,a2,a3,b[nt][0],b[nt][1]);
                    }
                }
            }
            if((step&3)==3){
                const int group=step>>2;
#pragma unroll
                for(int mt=0;mt<MT;++mt)
#pragma unroll
                    for(int nt=0;nt<4;++nt)
#pragma unroll
                        for(int e=0;e<4;++e){
                            int row=wm*64+mt*16+gid+(e>>1)*8;
                            int token=wn*32+nt*8+2*lid+(e&1);
                            if constexpr(Debug)if(group==debug_group&&token<live)
                                raw[std::int64_t(token0+token)*N+weight_row(row_block,row)]=g[mt][nt][e];
                            float unit=__fmul_rn(sh.ws[group*Rows+row],sh.xs[group*Tokens+token]);
                            f[mt][nt][e]=__fmaf_rn(__int2float_rn(g[mt][nt][e]),unit,f[mt][nt][e]);
                            g[mt][nt][e]=0;
                        }
            }
        }
        // All K-stage reads are complete before stage1 becomes BF16 output staging.
        __syncthreads();
        const int out_row_block=row_block,out_token0=token0,out_live=live;
        if constexpr(Debug){
#pragma unroll
            for(int mt=0;mt<MT;++mt)
#pragma unroll
                for(int nt=0;nt<4;++nt)
#pragma unroll
                    for(int e=0;e<4;++e){
                        int row=wm*64+mt*16+gid+(e>>1)*8;
                        int token=wn*32+nt*8+2*lid+(e&1);
                        if(token<out_live)pre[std::int64_t(out_token0+token)*N+
                            weight_row(out_row_block,row)]=f[mt][nt][e];
                    }
        }
        tile+=int(gridDim.x);const bool more=tile<total_tiles;
        if(more){set_tile(tile);issue_step(0,sh.stage[0]);cp_commit();}
        // Next-tile copies touch only stage0. Current scales remain intact.
        auto* staged=reinterpret_cast<__nv_bfloat16*>(storage+sizeof(Stage));
#pragma unroll
        for(int mt=0;mt<2;++mt)
#pragma unroll
            for(int nt=0;nt<4;++nt)
#pragma unroll
                for(int e=0;e<4;++e){
                    const int row=wm*64+mt*16+gid+(e>>1)*8;
                    const int token=wn*32+nt*8+2*lid+(e&1);
                    const float gate=f[mt][nt][e];
                    const float up=f[mt+2][nt][e];
                    staged[token*OutLd+wm*32+mt*16+gid+(e>>1)*8]=
                        __float2bfloat16_rn(__fmul_rn(ninfer::ops::silu(gate),up));
                }
        __syncthreads(); // every old scale read and output staging write is complete
#pragma unroll
        for(int i=0;i<4;++i){
            const int item=tid+i*Threads,token=item/8,chunk_out=item%8;
            const uint4 value=*reinterpret_cast<const uint4*>(staged+token*OutLd+chunk_out*8);
            if(token<out_live)*reinterpret_cast<uint4*>(Y+
                std::int64_t(out_token0+token)*I+out_row_block*OutRows+chunk_out*8)=value;
        }
        // Output spans retired stage1 plus the first2048B of stage2. Finish every
        // warp's output read before prefetching either of those stages for next tile.
        __syncthreads();
        if(!more)break;
        // Next stage0 was issued before epilogue. Scale and stage1 loads form one
        // group, then stage2 another. Scales are consumed only after step3's wait.
        issue_scales();issue_step(1,sh.stage[1]);cp_commit();
        issue_step(2,sh.stage[2]);cp_commit();
    }
}

// Future harness must call once before any cold Graph capture. No hidden first-use setup.
inline cudaError_t prepare_attributes(){
    auto e=cudaFuncSetAttribute(kernel<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(SharedBytes));
    if(e!=cudaSuccess)return e;
    return cudaFuncSetAttribute(kernel<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(SharedBytes));
}
template<bool Debug>
inline void launch(const std::uint8_t* W,const float* ws,const std::uint8_t* X,const float* xs,
                   __nv_bfloat16* Y,std::int32_t* raw,float* pre,int debug_group,int tokens,int sm_count,cudaStream_t stream){
    if((tokens<256||tokens>2048)||sm_count<=0||(Debug?(raw==nullptr||pre==nullptr||debug_group<0||debug_group>=10):(raw!=nullptr||pre!=nullptr)))
        throw std::runtime_error("fixed fused diagnostic shape/pointer contract");
    const int tiles=(tokens+Tokens-1)/Tokens,total=(I/OutRows)*tiles;
    kernel<Debug><<<std::min(total,sm_count),Threads,SharedBytes,stream>>>(W,ws,X,xs,Y,raw,pre,debug_group,tokens,tiles,total);
}
} // namespace ucloud2::group512_s4h512_pipe4
