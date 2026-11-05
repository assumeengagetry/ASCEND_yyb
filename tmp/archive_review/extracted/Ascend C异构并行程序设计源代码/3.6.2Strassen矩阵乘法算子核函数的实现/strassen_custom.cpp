#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;
constexpr int32_t m = 32;   
constexpr int32_t n = 32;   
constexpr int32_t k = 32;   
constexpr int32_t quarterSize = 256;

class strassenKernel{
public:
__aicore__ inline strassenKernel(){};
__aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tilingGm, TPipe* pipe);
__aicore__ inline void Process(TPipe* pipe);

__aicore__ inline void CopyIn();
__aicore__ inline void AddandSub1();
__aicore__ inline void MatmulCompute();
__aicore__ inline void AddandSub2();
__aicore__ inline void CopyOut();

Matmul<MatmulType<TPosition::VECOUT, CubeFormat::ND, half>,
        MatmulType<TPosition::VECOUT, CubeFormat::ND, half>,
        MatmulType<TPosition::VECIN, CubeFormat::ND, float>,
        MatmulType<TPosition::VECOUT, CubeFormat::ND, float>>
mm1;

TCubeTiling tiling;
GlobalTensor<half> GMmatrixA, GMmatrixB;
GlobalTensor<float> GMmatrixC;
TQue<TPosition::VECIN,1> A;
TQue<TPosition::VECIN,1> B;
TQue<TPosition::VECOUT,1> S1234;
TQue<TPosition::VECOUT,1> S5678;
TQue<TPosition::VECOUT,1> S910;
TQue<TPosition::VECOUT,1> N;
TQue<TPosition::VECIN,1> P1234;
TQue<TPosition::VECIN,1> P567;
TQue<TPosition::VECOUT,1> C;
};

__aicore__ inline void strassenKernel::Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tilingGm, TPipe* pipe){
    auto tempTilingGM = (__gm__ uint32_t*)tilingGm;
    auto tempTiling = (uint32_t*)&tiling;
    for (int32_t i = 0; i < sizeof(TCubeTiling) / sizeof(int32_t); ++i, ++tempTilingGM, ++tempTiling)
    {
        *tempTiling = *tempTilingGM;
    }

    GMmatrixA.SetGlobalBuffer((__gm__ half*)a);
    GMmatrixB.SetGlobalBuffer((__gm__ half*)b);
    GMmatrixC.SetGlobalBuffer((__gm__ float*)c);

    SetSysWorkspace(workspace);
    if(GetSysWorkSpacePtr() == nullptr){
        return;
    }

    pipe->InitBuffer(A, 1,  4 * quarterSize * sizeof(half));
    pipe->InitBuffer(B, 1,  4 * quarterSize * sizeof(half));
    pipe->InitBuffer(S1234, 1, 4 * quarterSize * sizeof(half));
    pipe->InitBuffer(S5678, 1, 4 * quarterSize * sizeof(half));
    pipe->InitBuffer(S910, 1, 2 * quarterSize * sizeof(half));
    pipe->InitBuffer(N, 1, 4 * quarterSize * sizeof(half));
    pipe->InitBuffer(P1234, 1, 4 * quarterSize * sizeof(float));
    pipe->InitBuffer(P567, 1, 3 * quarterSize * sizeof(float));
    pipe->InitBuffer(C, 1, 4 * quarterSize * sizeof(float));
}

__aicore__ inline void strassenKernel::Process(TPipe* pipe){
  CopyIn();
  AddandSub1();
  MatmulCompute();
  AddandSub2();
  CopyOut();
}

__aicore__ inline void strassenKernel::CopyIn(){
    LocalTensor<half> ALOCAL = A.AllocTensor<half>();
    LocalTensor<half> BLOCAL = B.AllocTensor<half>();
    DataCopy(ALOCAL[0], GMmatrixA[0], {16, 1, 1, 0});//A11
    DataCopy(ALOCAL[quarterSize], GMmatrixA[16], {16, 1, 1, 0});//A12
    DataCopy(ALOCAL[2*quarterSize], GMmatrixA[512], {16, 1, 1, 0});//A21
    DataCopy(ALOCAL[3*quarterSize], GMmatrixA[528], {16, 1, 1, 0});//A22
    DataCopy(BLOCAL[0], GMmatrixB[0], {16, 1, 1, 0});//B11
    DataCopy(BLOCAL[quarterSize], GMmatrixB[16], {16, 1, 1, 0});//B12
    DataCopy(BLOCAL[2*quarterSize], GMmatrixB[512], {16, 1, 1, 0});//B21
    DataCopy(BLOCAL[3*quarterSize], GMmatrixB[528], {16, 1, 1, 0});//B22

    A.EnQue<half>(ALOCAL);
    B.EnQue<half>(BLOCAL);
}

__aicore__ inline void strassenKernel::AddandSub1(){
    LocalTensor<half> ALOCAL = A.DeQue<half>();
    LocalTensor<half> BLOCAL = B.DeQue<half>();
    LocalTensor<half> S1234LOCAL = S1234.AllocTensor<half>();
    LocalTensor<half> S5678LOCAL = S5678.AllocTensor<half>();
    LocalTensor<half> S910LOCAL = S910.AllocTensor<half>();
    LocalTensor<half> NLOCAL = N.AllocTensor<half>();
    half scalar = 0;
    Sub(S1234LOCAL[0], BLOCAL[quarterSize], BLOCAL[3*quarterSize], 256);//S1
    Add(S1234LOCAL[quarterSize], ALOCAL[0], ALOCAL[quarterSize], 256);//S2
    Add(S1234LOCAL[2*quarterSize], ALOCAL[2*quarterSize], ALOCAL[3*quarterSize], 256);//S3
    Sub(S1234LOCAL[3*quarterSize], BLOCAL[2*quarterSize], BLOCAL[0], 256);//S4
    Add(S5678LOCAL[0], ALOCAL[0], ALOCAL[3*quarterSize], 256);//S5
    Add(S5678LOCAL[quarterSize], BLOCAL[0], BLOCAL[3*quarterSize], 256);//S6
    Sub(S5678LOCAL[2*quarterSize], ALOCAL[quarterSize], ALOCAL[3*quarterSize], 256);//S7
    Add(S5678LOCAL[3*quarterSize], BLOCAL[2*quarterSize], BLOCAL[3*quarterSize], 256);//S8
    Sub(S910LOCAL[0], ALOCAL[0], ALOCAL[2*quarterSize], 256);//S9
    Add(S910LOCAL[quarterSize], BLOCAL[0], BLOCAL[quarterSize], 256);//S10
    Adds(NLOCAL[0], ALOCAL[0], scalar, 256);//NA11
    Adds(NLOCAL[quarterSize], ALOCAL[3*quarterSize], scalar, 256);//NA22
    Adds(NLOCAL[2*quarterSize], BLOCAL[0], scalar, 256);//NB11
    Adds(NLOCAL[3*quarterSize], BLOCAL[3*quarterSize], scalar, 256);//NB22

    S1234.EnQue<half>(S1234LOCAL);
    S5678.EnQue<half>(S5678LOCAL);
    S910.EnQue<half>(S910LOCAL);
    N.EnQue<half>(NLOCAL);

    A.FreeTensor<half>(ALOCAL);
    B.FreeTensor<half>(BLOCAL);
}

__aicore__ inline void strassenKernel::MatmulCompute(){
    LocalTensor<half> S1234LOCAL = S1234.DeQue<half>();
    LocalTensor<half> S5678LOCAL = S5678.DeQue<half>();
    LocalTensor<half> S910LOCAL = S910.DeQue<half>();
    LocalTensor<half> NLOCAL = N.DeQue<half>();
    LocalTensor<float> P1234LOCAL = P1234.AllocTensor<float>();
    LocalTensor<float> P567LOCAL = P567.AllocTensor<float>();
    
    mm1.SetTensorA(NLOCAL[0]);
    mm1.SetTensorB(S1234LOCAL[0]);
    while (mm1.Iterate()) {   
    mm1.GetTensorC(P1234LOCAL[0]);//P1 
    }
    mm1.End();

    mm1.SetTensorA(S1234LOCAL[quarterSize]);
    mm1.SetTensorB(NLOCAL[3*quarterSize]);
    while (mm1.Iterate()) {   
    mm1.GetTensorC(P1234LOCAL[quarterSize]); //P2
    }
    mm1.End();

    mm1.SetTensorA(S1234LOCAL[2*quarterSize]);
    mm1.SetTensorB(NLOCAL[2*quarterSize]);
    while (mm1.Iterate()) {   
    mm1.GetTensorC(P1234LOCAL[2*quarterSize]); //P3
    }
    mm1.End();

    mm1.SetTensorA(NLOCAL[quarterSize]);
    mm1.SetTensorB(S1234LOCAL[3*quarterSize]);
    while (mm1.Iterate()) {   
    mm1.GetTensorC(P1234LOCAL[3*quarterSize]); //P4
    }
    mm1.End();

    mm1.SetTensorA(S5678LOCAL[0]);
    mm1.SetTensorB(S5678LOCAL[quarterSize]);
    while (mm1.Iterate()) {   
    mm1.GetTensorC(P567LOCAL[0]);//P5 
    }
    mm1.End();

    mm1.SetTensorA(S5678LOCAL[2*quarterSize]);
    mm1.SetTensorB(S5678LOCAL[3*quarterSize]);
    while (mm1.Iterate()) {   
    mm1.GetTensorC(P567LOCAL[quarterSize]); //P6
    }
    mm1.End();

    mm1.SetTensorA(S910LOCAL[0]);
    mm1.SetTensorB(S910LOCAL[quarterSize]);
    while (mm1.Iterate()) {   
    mm1.GetTensorC(P567LOCAL[2*quarterSize]); //P7
    }
    mm1.End();

    S1234.FreeTensor<half>(S1234LOCAL);
    S5678.FreeTensor<half>(S5678LOCAL);
    S910.FreeTensor<half>(S910LOCAL);
    N.FreeTensor<half>(NLOCAL);

    P1234.EnQue<float>(P1234LOCAL);
    P567.EnQue<float>(P567LOCAL);
    DumpTensor(P1234LOCAL,1,1024);
}

__aicore__ inline void strassenKernel::AddandSub2(){
    LocalTensor<float> P1234LOCAL = P1234.DeQue<float>();
    LocalTensor<float> P567LOCAL = P567.DeQue<float>();
    LocalTensor<float> CLOCAL = C.AllocTensor<float>();

    Add(CLOCAL[0], P567LOCAL[quarterSize], P567LOCAL[0], 256);
    Add(CLOCAL[0], CLOCAL[0], P1234LOCAL[3*quarterSize], 256);
    Sub(CLOCAL[0], CLOCAL[0], P1234LOCAL[quarterSize], 256);//C11
    Add(CLOCAL[quarterSize], P1234LOCAL[0], P1234LOCAL[quarterSize], 256);//C12
    Add(CLOCAL[2*quarterSize], P1234LOCAL[3*quarterSize], P1234LOCAL[2*quarterSize], 256);//C21
    Add(CLOCAL[3*quarterSize], P1234LOCAL[0], P567LOCAL[0], 256);
    Sub(CLOCAL[3*quarterSize], CLOCAL[3*quarterSize], P1234LOCAL[2*quarterSize], 256);
    Sub(CLOCAL[3*quarterSize], CLOCAL[3*quarterSize], P567LOCAL[2*quarterSize], 256);//C22

    C.EnQue<float>(CLOCAL);
    P1234.FreeTensor<float>(P1234LOCAL);
    P567.FreeTensor<float>(P567LOCAL);
}

__aicore__ inline void strassenKernel::CopyOut(){
    LocalTensor<float> CLOCAL = C.DeQue<float>();
    DataCopy(GMmatrixC[0], CLOCAL[0], {16, 2, 0, 2});
    DataCopy(GMmatrixC[16], CLOCAL[quarterSize], {16, 2, 0, 2});
    DataCopy(GMmatrixC[512], CLOCAL[2*quarterSize], {16, 2, 0, 2});
    DataCopy(GMmatrixC[528], CLOCAL[3*quarterSize], {16, 2, 0, 2});

    C.FreeTensor<float>(CLOCAL);
}

extern "C" __global__ __aicore__ void strassen_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tilingGm)
{
  strassenKernel op;
  TPipe pipe;
  op.Init(a,b,c,workspace,tilingGm,&pipe);
  REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.mm1, &op.tiling);
  op.Process(&pipe);
}
