/** Based on Huawei Ascend C 3.4 Matmul sample; pure A@B without bias. */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;
using namespace matmul;
extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tilingGm)
{
    TCubeTiling t;
    uint32_t *p = reinterpret_cast<uint32_t*>(&t);
    for (int i=0; i<sizeof(t)/sizeof(uint32_t); ++i)
        p[i] = reinterpret_cast<__gm__ uint32_t*>(tilingGm)[i];
    TPipe pipe;
    Matmul<MatmulType<TPosition::GM,CubeFormat::ND,half>,
           MatmulType<TPosition::GM,CubeFormat::ND,half>,
           MatmulType<TPosition::GM,CubeFormat::ND,float>,
           MatmulType<TPosition::GM,CubeFormat::ND,float>> mm;
    // Generated KernelLaunch wrapper initializes and clears the system workspace.
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &t);
    int id=GetBlockIdx();
    if (id>=t.usedCoreNum) return;
    int mb=(t.M+t.singleCoreM-1)/t.singleCoreM;
    int mi=id%mb, ni=id/mb;
    GlobalTensor<half> ag,bg;
    GlobalTensor<float> cg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(a)+mi*t.singleCoreM*t.Ka,t.M*t.Ka);
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(b)+ni*t.singleCoreN,t.Kb*t.N);
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c)+mi*t.singleCoreM*t.N+ni*t.singleCoreN,t.M*t.N);
    mm.SetTensorA(ag); mm.SetTensorB(bg);
    mm.IterateAll(cg); mm.End();
}
#ifndef ASCENDC_CPU_DEBUG
void matmul_custom_do(uint32_t dim,void* stream,uint8_t*a,uint8_t*b,uint8_t*c,uint8_t*w,uint8_t*t)
{matmul_custom<<<dim,nullptr,stream>>>(a,b,c,w,t);}
#endif
