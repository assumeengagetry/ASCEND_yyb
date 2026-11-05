#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

template <typename T1, typename T2, const bool IS_BF16 = false>  // T1 for data, T2 for vecClc
class FlashAttentionScoreGradSplitBNG {
 public:
  __aicore__ inline FlashAttentionScoreGradSplitBNG(){};
  __aicore__ inline void Init(__gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* dx, __gm__ uint8_t* query,
                              __gm__ uint8_t* pse_shift, __gm__ uint8_t* drop_mask, __gm__ uint8_t* atten_mask,
                              __gm__ uint8_t* forward_res, __gm__ uint8_t* softmax_max, __gm__ uint8_t* softmax_sum,
                              __gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv, __gm__ uint8_t* workspace,
                              const FlashAttentionScoreGradTilingDataBNGgs1s2* __restrict ordTilingData, TPipe* pipe_in);
  __aicore__ inline void Process();

  // 待补充：声明矩阵乘从？搬数据进？，数据类型是？
  using aType1 = MatmulType<TPosition::GM, CubeFormat::ND, T1, false>;
  using bType1 = MatmulType<TPosition::GM, CubeFormat::ND, T1, true>;
  using cType1 = MatmulType<TPosition::VECCALC, CubeFormat::ND_ALIGN, T2>;
  using biasType1 = MatmulType<TPosition::GM, CubeFormat::ND, float>;

  using aType2 = MatmulType<TPosition::VECCALC, CubeFormat::ND, T1, true>;
  using bType2 = MatmulType<TPosition::GM, CubeFormat::ND, T1, true>;
  using cType2 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
  Matmul<aType1, bType1, cType1, biasType1> mm1;
  Matmul<aType1, bType1, cType1, biasType1> mm2;
  Matmul<aType2, bType2, cType2, biasType1> mm3;
  Matmul<aType2, bType2, cType2, biasType1> mm4;  // 暂时都用mm3

 protected:
  /* define the que */
  TQue<QuePosition::VECIN, 2> vecInQue1;
  TQue<QuePosition::VECIN, 1> vecInQue2;
  TBuf<> vecClc1;
  TBuf<> vecClc2;
  TQue<QuePosition::VECOUT, 1> apiClcQue;

  const FlashAttentionScoreGradTilingDataBNGgs1s2* __restrict ordTilingData;

  TPipe* pipe;
  /* 输入 */
  GlobalTensor<uint8_t> attenMaskU8Gm;
  GlobalTensor<T1> keyGm, valueGm, dxGm, queryGm, attenMaskGm, forwardResGm, pseGm;

  /* 输出 */
  // GlobalTensor<T1> dqGm, dkGm, dvGm, dpseGm; // to do
  GlobalTensor<float> dqGm, dkGm, dvGm;
  GlobalTensor<float> softmaxMaxGm, softmaxSumGm;
  GlobalTensor<T2> workspaceGm;
  GlobalTensor<uint8_t> dropMaskGm;
  GlobalTensor<int32_t> syncGlobal;

  uint32_t inShapes1D[2];
  uint32_t inShapes11p[2];
  uint32_t inShapes1s2[2];
  uint32_t inShapes1s2T1[2];

  uint64_t mBlockIdx;
  uint64_t coreNum;

  T2 mulsValue = -10000.0;
  /* ---- yuankun parameter ---- */

  // The size of each Dimension
  int32_t B;
  int32_t G;
  int32_t s1;
  int32_t s2;
  int32_t D;
  int32_t N;

  // The core calculate number
  uint32_t gInner;
  int32_t Go;
  int32_t A;
  int32_t Af;
  int32_t CoreUsed;
  uint32_t splitedBatchRange;
  uint32_t s1s2T1AlignSize;  // s1*Ceil(s2, 32/T1)*32/T1*T2
  uint32_t s1DT1AlignSize;   // s1*Ceil(D, 32/T1)*32/T1*T2

  // The position of each Dimension at iteration i
  int32_t Bpos;
  int32_t Npos;
  int32_t Gpos;

  // The position at iteration i
  int32_t Offsets1D;
  int32_t Offsets2D;
  int32_t Offsets1s2;
  int32_t Offsets11p;

  // The calculate length at iteration i
  uint32_t inSize;
  uint32_t inSizes1D;
  uint32_t inSizes1s2;
  uint32_t inSizes11;
  /* ---- yuankun parameter ---- */

  bool isDrop;

  int32_t layoutSBH;

  __aicore__ inline void InitGm();

  __aicore__ inline void InitSplitParams();

  __aicore__ inline void ClcSoftMaxGrad(LocalTensor<T2>& SoftMaxGradOut);

  // 拷贝进UB并pading tiling
  __aicore__ inline void CopyInPadT1(LocalTensor<T1>& dst, const GlobalTensor<T1>& src, int32_t innerReduceNum,
                                     uint32_t* innerReduceShape);

  // 拷贝进UB并pading tiling
  __aicore__ inline void CopyInPadUint8(LocalTensor<uint8_t>& dst, const GlobalTensor<uint8_t>& src,
                                        int32_t innerReduceNum, uint32_t* innerReduceShape);

  __aicore__ inline void CopyInPadTranspose(LocalTensor<T1>& dst, const GlobalTensor<T1>& src, int32_t innerReduceNum,
                                            uint32_t* innerReduceShape);

  // PseAdd
  __aicore__ inline void PseAdd(LocalTensor<T2>& AddOut, LocalTensor<T2>& BMM2Out);

  // Mask
  __aicore__ inline void ClcMask(LocalTensor<T2>& MaskOut, LocalTensor<T2>& AddOut);

  // SimpleSoftMax
  __aicore__ inline void ClcSimpleSoftMax(LocalTensor<T2>& softmaxOut, LocalTensor<T2>& MaskOut);

  // Dropout
  __aicore__ inline void ClcDropout(LocalTensor<T2>& DropOutOut, LocalTensor<T2>& softmaxOut);
  // Sub
  __aicore__ inline void ClcSub(LocalTensor<T2>& AddResult, LocalTensor<T2>& Add1, LocalTensor<T2>& Add2,
                                int32_t innerMatResNum);

  // 矩阵乘MatMul
  __aicore__ inline void ClcMatMul(const LocalTensor<T2>& tensorC, const GlobalTensor<T1>& tensorA,
                                   const GlobalTensor<T1>& tensorB, bool transA = false, bool transB = false);

  // 矩阵乘MatMul
  __aicore__ inline void ClcMatMul2(const GlobalTensor<float>& tensorC, const LocalTensor<T1>& tensorA,
                                    const GlobalTensor<T1>& tensorB, bool transA = false, bool transB = false);

  // 矩阵乘MatMul
  __aicore__ inline void ClcMatMul3(const GlobalTensor<float>& tensorC, const LocalTensor<T1>& tensorA,
                                    const GlobalTensor<T1>& tensorB, bool transA = false, bool transB = false);

  __aicore__ inline void CastOutIfBF16(LocalTensor<T1>& outT1, LocalTensor<T2>& inT2);

  // 计算分块分核
  __aicore__ inline void DetermineLoopParams(uint64_t batchIdx);
};

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradSplitBNG<T1, T2, IS_BF16>::Init(
    __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* dx, __gm__ uint8_t* query, __gm__ uint8_t* pse_shift,
    __gm__ uint8_t* drop_mask, __gm__ uint8_t* atten_mask, __gm__ uint8_t* forward_res, __gm__ uint8_t* softmax_max,
    __gm__ uint8_t* softmax_sum, __gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv, __gm__ uint8_t* workspace,
    const FlashAttentionScoreGradTilingDataBNGgs1s2* __restrict ordTilingData, TPipe* pipe_in) {
  this->keyGm.SetGlobalBuffer((__gm__ T1*)key);
  this->valueGm.SetGlobalBuffer((__gm__ T1*)value);
  this->dxGm.SetGlobalBuffer((__gm__ T1*)dx);
  this->queryGm.SetGlobalBuffer((__gm__ T1*)query);
  this->pseGm.SetGlobalBuffer((__gm__ T1*)pse_shift);
  this->dropMaskGm.SetGlobalBuffer((__gm__ uint8_t*)drop_mask);
  this->attenMaskGm.SetGlobalBuffer((__gm__ T1*)atten_mask);
  this->attenMaskU8Gm.SetGlobalBuffer((__gm__ uint8_t*)atten_mask);  // second path
  this->forwardResGm.SetGlobalBuffer((__gm__ T1*)forward_res);
  this->softmaxMaxGm.SetGlobalBuffer((__gm__ float*)softmax_max);
  this->softmaxSumGm.SetGlobalBuffer((__gm__ float*)softmax_sum);
  this->dqGm.SetGlobalBuffer((__gm__ float*)dq);
  this->dkGm.SetGlobalBuffer((__gm__ float*)dk);
  this->dvGm.SetGlobalBuffer((__gm__ float*)dv);
  this->workspaceGm.SetGlobalBuffer((__gm__ T2*)workspace);
  uint64_t syncCoresBuffSize = 100 * 8;
  this->syncGlobal.SetGlobalBuffer((__gm__ int32_t*)workspace, syncCoresBuffSize);
  InitOutput<int32_t>(this->syncGlobal[GetBlockIdx() * 8], 8, 0);

  this->mBlockIdx = GetBlockIdx();
  this->ordTilingData = ordTilingData;
  this->pipe = pipe_in;

  // mm2TilingData参数设置，之后移动到tiling
  InitSplitParams();

  // 变量小写
  // pipe and matrix offset and size (还需要实现)
  uint64_t inQuesize0 = s1DT1AlignSize;
  uint64_t inQuesize1 = s1s2T1AlignSize;
  uint64_t outQuesize = s1DT1AlignSize;
  pipe->InitBuffer(vecInQue1, 2, inQuesize0);
  pipe->InitBuffer(vecInQue2, 1, inQuesize1);
  pipe->InitBuffer(vecClc1, inQuesize0);
  pipe->InitBuffer(vecClc2, inQuesize0);
  pipe->InitBuffer(apiClcQue, 1, outQuesize);

  this->pipe->InitSpmBuffer(this->workspaceGm[800 + this->mBlockIdx * inQuesize1], inQuesize1);  // [s_out, s_fla]
  uint64_t buff_offset = 800 + GetBlockNum() * GetTaskRation() * inQuesize1;
  this->mm1.SetWorkspace((__gm__ uint8_t*)workspaceGm[buff_offset + mBlockIdx * inQuesize1].GetPhyAddr(), inQuesize1);
  this->mm2.SetWorkspace(
      (__gm__ uint8_t*)workspaceGm[buff_offset + GetBlockNum() * GetTaskRation() * inQuesize1 + mBlockIdx * inQuesize1]
          .GetPhyAddr(),
      inQuesize1);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradSplitBNG<T1, T2, IS_BF16>::Process() {
  // 从输入参数keepProb判断是否需要走drop
  isDrop = this->ordTilingData->gBNBaseParams.keepProb < 1 ? true : false;
  // 初始化GM
  InitGm();
  /*
      输入Cube                shape                Cubesize         Cubeshape        Offset[mBlockIdx]
      dxGM                    [B,N2,G,s1,D   ]     inSizes1D[R]     inShapes1D       Offsets1D
      queryGm                 [B,N2,G,s1,D   ]     inSizes1D[R]     inShapes1D       Offsets1D
      dqGm                    [B,N2,G,s1,D   ]     inSizes1D[R]     inShapes1D       Offsets1D
      valueGm                 [B,N2,1,s2,D   ]     inSizes2D[R]     inShapes2D       Offsets2D
      keyGm                   [B,N2,1,s2,D   ]     inSizes2D[R]     inShapes2D       Offsets2D
      dvGm                    [B,N2,1,s2,D   ]     inSizes2D[R]     inShapes2D       Offsets2D
      dkGm                    [B,N2,1,s2,D   ]     inSizes2D[R]     inShapes2D       Offsets2D
  */

  /*
      输入UB                  shape                UBsize           UBshape          使用que     Offset[mBlockIdx]
      dxGM                    [B,N2,G,s1,D   ]     inSizes1D[R]     inShapes1D       vecInQue1   Offsets1D
      forwardResGm            [B,N2,G,s1,D   ]     inSizes1D[R]     inShapes1D       vecInQue1   Offsets1D
      dropMaskGm[bit]         [B*N2*G*s1,s2  ]     inSizes1s2[R]    inShapes1s2      vecInQue2   Offsets1s2
      pseGm                   [B,N2,G,s1,s2  ]     inSizes1s2[R]    inShapes1s2T1    vecInQue2   Offsets1s2
      attenMaskGM[bool]       [B,N2,G,s1,s2  ]     inSizes1s2[R]    inShapes1s2      vecInQue2   Offsets1s2
      softmaxMaxGm[fp32]      [B,N2,G,s1,1[p]]     inSizes11[R]     inShapes11p       vecInQue1   Offsets11p
      softmaxSumGm[fp32]      [B,N2,G,s1,1[p]]     inSizes11[R]     inShapes11p       vecInQue1   Offsets11p
  */

  /*
      中间量                  shape                UBsize           UBshape          使用que      Offset[mBlockIdx]
      SoftMaxGradOut          [B,N2,G,s1,1[p]]     inSizes11[R]     inShapes11p      vecClc2     Offsets11p
      BMM1Out                 [B,N2,G,s1,s2  ]     inSizes1s2[R]    inShapes1s2      vecClc1      Offsets1s2
      BMM2Out                 [B,N2,G,s1,s2  ]     inSizes1s2[R]    inShapes1s2      vecClc2      Offsets1s2
      MulOut                  [B,N2,G,s1,s2  ]     inSizes1s2[R]    inShapes1s2      vecClc1      Offsets1s2
  */
  for (uint64_t batchIdx = 0; batchIdx < this->splitedBatchRange; batchIdx++) {
    // 搬入的 offset 大小
    DetermineLoopParams(batchIdx);

    /* ---------------------- stage1 -------------------- */
    // SoftMaxGradOut : 存储SoftMaxGrad算子结果
    LocalTensor<T2> SoftMaxGradOut = vecClc2.Get<T2>(inSizes11 * 32 / sizeof(T2));
    SoftMaxGradOut.SetShapeInfo(ShapeInfo(2, inShapes11p, DataFormat::ND));
    ClcSoftMaxGrad(SoftMaxGradOut);

    // BMM1Out : 存储BMM1算子结果
    LocalTensor<T2> BMM1Out = vecClc1.Get<T2>(inSizes1s2);
    BMM1Out.SetShapeInfo(ShapeInfo(2, inShapes1s2T1, DataFormat::ND));

    pipe_barrier(PIPE_V);  // SoftMaxGradOut
    ClcMatMul(BMM1Out, dxGm[Offsets1D], valueGm[Offsets2D], false, true);

    // Dropout
    if (isDrop) {
      ClcDropout(BMM1Out, BMM1Out);
    }

    ClcSub(BMM1Out, BMM1Out, SoftMaxGradOut, inSizes1s2);
    /* ---------------------- stage1 -------------------- */

    /* ---------------------- stage2 -------------------- */
    // BMM2 搬运和计算
    LocalTensor<T2> BMM2Out = vecClc2.Get<T2>(inSizes1s2);  // vecClc2
    LocalTensor<T2>& AddOut = BMM2Out;
    LocalTensor<T2>& MaskOut = BMM2Out;
    LocalTensor<T2>& softmaxOut = BMM2Out;
    LocalTensor<T2>& DropOutOut = BMM2Out;
    BMM2Out.SetShapeInfo(ShapeInfo(2, inShapes1s2T1, DataFormat::ND));

    ClcMatMul(BMM2Out, queryGm[Offsets1D], keyGm[Offsets2D], false, true);

    Muls(BMM2Out, BMM2Out, (T2)(this->ordTilingData->gBNBaseParams.scaleValue), inSizes1s2);

    // Add 搬运和计算
    if (pseGm.GetPhyAddr() != nullptr) {
      PseAdd(AddOut, BMM2Out);
    }

    // MASK 搬运和计算
    ClcMask(MaskOut, AddOut);
    // SimpleSoftMax搬运和计算
    ClcSimpleSoftMax(softmaxOut, MaskOut);
    // Mul搬运和计算
    LocalTensor<T2>& frontResInner = BMM1Out;
    LocalTensor<T2>& mulResInner = frontResInner;

    Mul(mulResInner, softmaxOut, frontResInner, inSizes1s2);

    Muls(mulResInner, mulResInner, (T2)(this->ordTilingData->gBNBaseParams.scaleValue), inSizes1s2);

    LocalTensor<T1> castedMulResInner;
    CastOutIfBF16(castedMulResInner, mulResInner);

    // ---------------------- stage3 --------------------
    // BMM3.1 (C, A, B, A.Trans?, B.Trans?) 暂时都用MM3(ClcMatMul2), 之后换MM4(ClcMatMul3)
    ClcMatMul2(this->dqGm[Offsets1D], castedMulResInner, this->keyGm[Offsets2D], false, false);

    // BMM3.2
    ClcMatMul3(this->dkGm[Offsets2D], castedMulResInner, this->queryGm[Offsets1D], true, false);

    if (isDrop) {
      ClcDropout(DropOutOut, softmaxOut);
    }
    LocalTensor<T1> castedDropOutOut;
    CastOutIfBF16(castedDropOutOut, softmaxOut);
    // BMM4
    ClcMatMul3(this->dvGm[Offsets2D], castedDropOutOut, this->dxGm[Offsets1D], true, false);
  }
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradSplitBNG<T1, T2, IS_BF16>::InitSplitParams() {
  // split info, 之后放到tiling
  this->coreNum = 48;
  this->B = ordTilingData->gBNBaseParams.B;
  this->N = ordTilingData->gBNBaseParams.N1;
  this->G = 1;
  this->gInner = 1;
  this->s1 = ordTilingData->gBNBaseParams.S1;
  this->s2 = ordTilingData->gBNBaseParams.S2;
  this->D = ordTilingData->gBNBaseParams.D;
  this->layoutSBH = this->ordTilingData->gBNBaseParams.layout;

  this->Go = Ceil((uint32_t)G, gInner);
  this->A = B * N * Go;
  this->Af = Ceil(A, coreNum);
  this->CoreUsed = Ceil(A, Af);
  this->splitedBatchRange = A - mBlockIdx * Af < Af ? A - mBlockIdx * Af : Af;  // 多少baseG块，循环多少次
  this->s1s2T1AlignSize = s1 * Ceil(s2, 32 / sizeof(T1)) * 32 / sizeof(T1) * sizeof(T2);
  this->s1DT1AlignSize = s1 * Ceil(D, 32 / sizeof(T1)) * 32 / sizeof(T1) * sizeof(T2);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradSplitBNG<T1, T2, IS_BF16>::InitGm() {
  uint64_t allSize = this->A;
  uint32_t perSize = this->Af;
  uint32_t initUsedCore = this->CoreUsed;
  uint32_t tailSize = allSize - perSize * (initUsedCore - 1);

  uint32_t initSize = perSize;
  if (mBlockIdx == initUsedCore - 1) {
    initSize = tailSize;
  }

  uint32_t offset = mBlockIdx * perSize;

  if (mBlockIdx < initUsedCore) {
    InitOutput<float>(dqGm[offset], initSize, 0.0);
    InitOutput<float>(dkGm[offset], initSize, 0.0);
    InitOutput<float>(dvGm[offset], initSize, 0.0);
  }

  LocalTensor<int32_t> workLocal = vecInQue1.AllocTensor<int32_t>();
  SyncAll<true>(syncGlobal, workLocal);
  vecInQue1.FreeTensor(workLocal);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradSplitBNG<T1, T2, IS_BF16>::DetermineLoopParams(uint64_t batchIdx) {
  int32_t BNGTotal = mBlockIdx * Af + batchIdx;
  this->Bpos = BNGTotal / (N * Go);
  this->Npos = (BNGTotal % (N * Go)) / Go;
  this->Gpos = (BNGTotal % (N * Go)) % Go;

  this->inSize = (Gpos == Go) ? (G % gInner) : gInner;
  this->inSizes1D = this->inSize * s1 * Ceil(D, 32 / sizeof(T1)) * 32 / sizeof(T1);
  this->inSizes1s2 = this->inSize * s1 * Ceil(s2, 32 / sizeof(T2)) * 32 / sizeof(T2);
  this->inSizes11 = this->inSize * s1;

  this->inShapes1D[0] = this->inSizes11;
  this->inShapes1D[1] = Ceil(D, 32 / sizeof(T1)) * 32 / sizeof(T1);
  this->inShapes1s2[0] = this->inSizes11;
  this->inShapes1s2[1] = Ceil(s2, 32 / sizeof(uint8_t)) * 32 / sizeof(uint8_t);
  this->inShapes1s2T1[0] = this->inSizes11;
  this->inShapes1s2T1[1] = Ceil(s2, 32 / sizeof(T1)) * 32 / sizeof(T1);
  this->inShapes11p[0] = this->inSizes11;
  this->inShapes11p[1] = 32 / sizeof(T2);

  this->Offsets1s2 = this->Bpos * N * G * s1 * s2 + this->Npos * G * s1 * s2 + this->Gpos * gInner * s1 * s2;
  this->Offsets11p = this->Bpos * N * G * s1 * 8 + this->Npos * G * s1 * 8 + this->Gpos * gInner * s1 * 8;

  if (layoutSBH == 0) {  // shape BSH
    this->Offsets1D = this->Bpos * s1 * N * G * D + this->Npos * G * D + this->Gpos * gInner * D;
    this->Offsets2D = this->Bpos * s2 * N * G * D + this->Npos * G * D + this->Gpos * gInner * D;

  } else if (layoutSBH == 1) {  // shape SBH
    this->Offsets1D = this->Bpos * N * G * D + this->Npos * G * D + this->Gpos * gInner * D;
    this->Offsets2D = this->Offsets1D;

  } else {  // shape BNGSD
    this->Offsets1D = this->Bpos * N * G * s1 * D + this->Npos * G * s1 * D + this->Gpos * gInner * s1 * D;
    this->Offsets2D = this->Bpos * N * G * s2 * D + this->Npos * G * s2 * D + this->Gpos * gInner * s2 * D;
  }
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradSplitBNG<T1, T2, IS_BF16>::ClcSoftMaxGrad(
    LocalTensor<T2>& SoftMaxGradOut) {
  // dxGm
  LocalTensor<T1> dxIn = vecInQue1.AllocTensor<T1>();
  CopyInPadTranspose(dxIn, dxGm[Offsets1D], inSizes1D, inShapes1D);

  // forwardResGm
  LocalTensor<T1> forwardResIn = vecInQue1.AllocTensor<T1>();
  CopyInPadTranspose(forwardResIn, forwardResGm[Offsets1D], inSizes1D, inShapes1D);

  vecInQue1.EnQue(forwardResIn);
  vecInQue1.DeQue<T1>();

  SoftMaxTiling softmaxGradTilingData;

  LocalTensor<uint8_t> apiClcTensor = apiClcQue.AllocTensor<uint8_t>();
  apiClcTensor.SetSize(inSizes1D * 2);

  if constexpr (IS_BF16) {
    LocalTensor<T2> casteddxIn = vecClc1.Get<T2>(inSizes1D);
    casteddxIn.SetShapeInfo(ShapeInfo(2, inShapes1D, DataFormat::ND));
    Cast(casteddxIn, dxIn, RoundMode::CAST_NONE, inSizes1D);
    LocalTensor<T2> castedforwardResIn = vecClc2.Get<T2>(inSizes1D);
    castedforwardResIn.SetShapeInfo(ShapeInfo(2, inShapes1D, DataFormat::ND));
    Cast(castedforwardResIn, forwardResIn, RoundMode::CAST_NONE, inSizes1D);
    pipe_barrier(PIPE_V);

    SoftmaxGradFront<T2, false>(SoftMaxGradOut, castedforwardResIn, casteddxIn, apiClcTensor, softmaxGradTilingData);
  } else {
    SoftmaxGradFront<T2, false>(SoftMaxGradOut, forwardResIn, dxIn, apiClcTensor, softmaxGradTilingData);
  }

  vecInQue1.FreeTensor(dxIn);
  vecInQue1.FreeTensor(forwardResIn);
  apiClcQue.FreeTensor(apiClcTensor);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradSplitBNG<T1, T2, IS_BF16>::PseAdd(LocalTensor<T2>& AddOut,
                                                                                LocalTensor<T2>& BMM2Out) {
  LocalTensor<T1> pseShiftIn = vecInQue2.AllocTensor<T1>();
  CopyInPadT1(pseShiftIn, pseGm[Offsets1s2], inSizes1s2, inShapes1s2T1);
  vecInQue2.EnQue(pseShiftIn);
  vecInQue2.DeQue<T1>();

  if constexpr (IS_BF16) {
    LocalTensor<T2> castedpseShiftIn = vecInQue1.AllocTensor<T2>();
    castedpseShiftIn.SetSize(inSizes1s2);
    uint32_t eleNum = pseShiftIn.GetSize();
    castedpseShiftIn.SetShapeInfo(ShapeInfo(2, inShapes1s2, DataFormat::ND));
    Cast(castedpseShiftIn, pseShiftIn, RoundMode::CAST_NONE, eleNum);
    pipe_barrier(PIPE_V);
    Add(AddOut, BMM2Out, castedpseShiftIn, inSizes1s2);
    vecInQue1.FreeTensor(castedpseShiftIn);
  } else {
    Add(AddOut, BMM2Out, pseShiftIn, inSizes1s2);
  }
  vecInQue2.FreeTensor(pseShiftIn);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradSplitBNG<T1, T2, IS_BF16>::ClcMask(LocalTensor<T2>& MaskOut,
                                                                                 LocalTensor<T2>& AddOut) {
  LocalTensor<T1> attenMaskIn = vecInQue2.AllocTensor<T1>();
  attenMaskIn.SetSize(inSizes1s2);
  uint32_t temp = 0;  // 111S1S2, to do for B11S1S2 and BNGS1S1
  DataCopyPad(attenMaskIn, attenMaskGm[temp], {static_cast<uint16_t>(s1), static_cast<uint16_t>(s2 * sizeof(T1)), 0, 0},
              {false, 0, 0, 0});
  vecInQue2.EnQue(attenMaskIn);
  vecInQue2.DeQue<T1>();

  if constexpr (IS_BF16) {
    LocalTensor<T2> castedAttenMaskIn = vecInQue1.AllocTensor<T2>();
    castedAttenMaskIn.SetSize(inSizes1s2);
    uint32_t eleNum = attenMaskIn.GetSize();
    castedAttenMaskIn.SetShapeInfo(ShapeInfo(2, inShapes1s2, DataFormat::ND));
    Cast(castedAttenMaskIn, attenMaskIn, RoundMode::CAST_NONE, eleNum);
    pipe_barrier(PIPE_V);
    Muls(castedAttenMaskIn, castedAttenMaskIn, mulsValue, inSizes1s2);  // -10000
    pipe_barrier(PIPE_V);
    Add(MaskOut, castedAttenMaskIn, AddOut, inSizes1s2);
    vecInQue1.FreeTensor(castedAttenMaskIn);
  } else {
    Muls(attenMaskIn, attenMaskIn, mulsValue, inSizes1s2);  // -10000
    pipe_barrier(PIPE_V);
    Add(MaskOut, attenMaskIn, AddOut, inSizes1s2);
  }
  vecInQue2.FreeTensor(attenMaskIn);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradSplitBNG<T1, T2, IS_BF16>::ClcSimpleSoftMax(LocalTensor<T2>& softmaxOut,
                                                                                          LocalTensor<T2>& MaskOut) {
  LocalTensor<float> maxIn = vecInQue1.AllocTensor<float>();
  DataCopyPad(maxIn, softmaxMaxGm[Offsets11p], {1, (uint16_t)(inSizes11 * 32), 0, 0}, {false, 0, 0, 0});

  LocalTensor<float> sumIn = vecInQue1.AllocTensor<float>();
  DataCopyPad(sumIn, softmaxSumGm[Offsets11p], {1, (uint16_t)(inSizes11 * 32), 0, 0}, {false, 0, 0, 0});
  vecInQue1.EnQue(sumIn);
  vecInQue1.DeQue<float>();

  /*
      关于基本块：isBasicBlock， 输入和输出是否为基本块，可以使能该参数用于提升性能
      基本块定义为:满足last轴的大小为256Byte的倍数，非last轴乘积长度为8的倍数的输入tensor都可以称为基本块。
  */
  uint32_t shapeArray[2];
  shapeArray[0] = this->s1;
  shapeArray[1] = Ceil(this->s2, 32 / sizeof(T2)) * 32 / sizeof(T2);
  softmaxOut.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));

  LocalTensor<uint8_t> apiClcTensor = apiClcQue.AllocTensor<uint8_t>();
  bool isBasicBlock = (this->s1 % 8 == 0) && (this->s2 * sizeof(T2) % 256 == 0);
  if (isBasicBlock) {
    SimpleSoftMax<T2, true, true>(softmaxOut, sumIn, maxIn, MaskOut, apiClcTensor,
                                  this->ordTilingData->softmaxTilingData);
  } else {
    SimpleSoftMax<T2, true, false>(softmaxOut, sumIn, maxIn, MaskOut, apiClcTensor,
                                   this->ordTilingData->softmaxTilingData);
  }
  vecInQue1.FreeTensor(maxIn);
  vecInQue1.FreeTensor(sumIn);
  apiClcQue.FreeTensor(apiClcTensor);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradSplitBNG<T1, T2, IS_BF16>::ClcDropout(LocalTensor<T2>& DropOutOut,
                                                                                    LocalTensor<T2>& softmaxOut) {
  LocalTensor<uint8_t> dpMaskIn = vecInQue2.AllocTensor<uint8_t>();
  CopyInPadUint8(dpMaskIn, dropMaskGm[Offsets1s2], inSizes1s2, inShapes1s2);
  vecInQue2.EnQue(dpMaskIn);
  vecInQue2.DeQue<uint8_t>();

  DropOutShapeInfo info;
  info.firstAxis = s1;
  info.srcLastAxis = s2;
  info.maskLastAxis = (s2 / 8 + 31) / 32 * 32 / 8;
  LocalTensor<uint8_t> apiClcTensor = apiClcQue.AllocTensor<uint8_t>();
  apiClcTensor.SetSize(inSizes1s2);
  DropOut<T2, true>(DropOutOut, softmaxOut, dpMaskIn, apiClcTensor, ordTilingData->gBNBaseParams.keepProb, info);

  apiClcQue.FreeTensor(apiClcTensor);
  vecInQue2.FreeTensor(dpMaskIn);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradSplitBNG<T1, T2, IS_BF16>::CopyInPadUint8(
    LocalTensor<uint8_t>& dst, const GlobalTensor<uint8_t>& src, int32_t innerReduceNum, uint32_t* innerReduceShape) {
  dst.SetSize(innerReduceNum);
  dst.SetShapeInfo(ShapeInfo(2, innerReduceShape, DataFormat::ND));
  struct DataCopyParams intriParams;
  // 对齐部分的block数，默认BNSD
  intriParams.blockCount = 1;
  intriParams.blockLen = innerReduceNum / 32;
  intriParams.dstStride = 0;
  intriParams.srcStride = 0;
  DataCopyPadParams padParams;
  if (innerReduceNum % 32 == 0) {
    padParams.isPad = false;
  } else {
    padParams.isPad = true;
    padParams.paddingValue = 0;
    padParams.leftPadding = 0;
    padParams.rightPadding = (innerReduceNum % 32);
  }
  DataCopyPad(dst, src, intriParams, padParams);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradSplitBNG<T1, T2, IS_BF16>::CopyInPadT1(LocalTensor<T1>& dst,
                                                                                     const GlobalTensor<T1>& src,
                                                                                     int32_t innerReduceNum,
                                                                                     uint32_t* innerReduceShape) {
  dst.SetSize(innerReduceNum);
  dst.SetShapeInfo(ShapeInfo(2, innerReduceShape, DataFormat::ND));

  uint32_t blockCount;
  uint32_t blockLen;
  uint32_t srcStride;
  uint32_t padding;

  struct DataCopyParams intriParams;
  DataCopyPadParams padParams;

  // 对齐部分的block数，默认BNSD
  blockCount = 1;
  blockLen = innerReduceNum / innerReduceShape[1];
  srcStride = 0;
  padding = innerReduceNum % innerReduceShape[1];

  intriParams.blockCount = blockCount;
  intriParams.blockLen = blockLen;
  intriParams.dstStride = 0;
  intriParams.srcStride = srcStride;

  if (padding > 0) {
    padParams.isPad = true;
    padParams.paddingValue = 0;
    padParams.leftPadding = 0;
    padParams.rightPadding = padding * sizeof(T1);
  } else {
    padParams.isPad = false;
  }
  DataCopyPad(dst, src, intriParams, padParams);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradSplitBNG<T1, T2, IS_BF16>::CopyInPadTranspose(
    LocalTensor<T1>& dst, const GlobalTensor<T1>& src, int32_t innerReduceNum, uint32_t* innerReduceShape) {
  dst.SetSize(innerReduceNum);
  dst.SetShapeInfo(ShapeInfo(2, innerReduceShape, DataFormat::ND));

  uint32_t blockCount;
  uint32_t blockLen;
  uint32_t srcStride;
  uint32_t padding;

  struct DataCopyParams intriParams;
  DataCopyPadParams padParams;

  if (layoutSBH == 0) {
    // BSH
    blockCount = s1;
    blockLen = D * sizeof(T1);
    srcStride = (N * G * D - D) * sizeof(T1);
    padding = Ceil(D, 32 / sizeof(T1)) * 32 / sizeof(T1) - D;
  } else if (layoutSBH == 1) {
    // SBH
    blockCount = s1;
    blockLen = D * sizeof(T1);
    srcStride = (B * N * G * D - D) * sizeof(T1);
    padding = Ceil(D, 32 / sizeof(T1)) * 32 / sizeof(T1) - D;
  } else {
    // shape BNGSD
    blockCount = 1;
    blockLen = innerReduceNum * sizeof(T1);
    srcStride = 0;
    padding = innerReduceNum * sizeof(T1) % 32;
  }
  intriParams.blockCount = blockCount;
  intriParams.blockLen = blockLen;
  intriParams.dstStride = 0;
  intriParams.srcStride = srcStride;

  if (padding > 0) {
    padParams.isPad = true;
    padParams.paddingValue = 0;
    padParams.leftPadding = 0;
    padParams.rightPadding = padding;
  } else {
    padParams.isPad = false;
  }
  DataCopyPad(dst, src, intriParams, padParams);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradSplitBNG<T1, T2, IS_BF16>::ClcSub(LocalTensor<T2>& AddResult,
                                                                                LocalTensor<T2>& Add1,
                                                                                LocalTensor<T2>& Add2,
                                                                                int32_t innerMatResNum) {
  uint8_t blockNum = Ceil(this->s2, (32 / sizeof(T2)));
  uint32_t subMask = this->s2;
  int32_t repNum = this->s1;

  // dst_blk_stride, src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride, src1_rep_stride
  Sub(AddResult[0], Add1[0], Add2[0], subMask, repNum, {1, 1, 0, blockNum, blockNum, 1});
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradSplitBNG<T1, T2, IS_BF16>::ClcMatMul(const LocalTensor<T2>& tensorC,
                                                                                   const GlobalTensor<T1>& tensorA,
                                                                                   const GlobalTensor<T1>& tensorB,
                                                                                   bool transA, bool transB) {
  mm1.SetTail(s1, s2, D);  // M N K
  mm1.SetTensorA(tensorA, transA);
  mm1.SetTensorB(tensorB, transB);
  mm1.template IterateAll<true>(tensorC, 0);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradSplitBNG<T1, T2, IS_BF16>::ClcMatMul2(const GlobalTensor<float>& tensorC,
                                                                                    const LocalTensor<T1>& tensorA,
                                                                                    const GlobalTensor<T1>& tensorB,
                                                                                    bool transA, bool transB) {
  // batched matmul
  for (int i = 0; i < this->inSize; i++) {
    mm4.SetTail(s1, D, s2);  // M N K
    mm4.SetTensorA(tensorA, transA);
    mm4.SetTensorB(tensorB, transB);
    mm4.template IterateAll<true>(tensorC, 0);
  }
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradSplitBNG<T1, T2, IS_BF16>::ClcMatMul3(const GlobalTensor<float>& tensorC,
                                                                                    const LocalTensor<T1>& tensorA,
                                                                                    const GlobalTensor<T1>& tensorB,
                                                                                    bool transA, bool transB) {
  // batched matmul
  for (int i = 0; i < this->inSize; i++) {
    mm3.SetTail(s2, D, s1);  // M N K
    mm3.SetTensorA(tensorA, transA);
    mm3.SetTensorB(tensorB, transB);
    mm3.template IterateAll<true>(tensorC, 0);
  }
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradSplitBNG<T1, T2, IS_BF16>::CastOutIfBF16(LocalTensor<T1>& outT1,
                                                                                       LocalTensor<T2>& inT2) {
  if constexpr (IS_BF16) {
    pipe_barrier(PIPE_V);
    uint32_t eleNum = inT2.GetSize();
    outT1 = vecClc1.Get<T1>(eleNum);
    outT1.SetShapeInfo(ShapeInfo(2, inShapes1s2, DataFormat::ND));
    Cast(outT1, inT2, RoundMode::CAST_RINT, eleNum);
    pipe_barrier(PIPE_V);
  } else {
    outT1 = inT2;
  }
}