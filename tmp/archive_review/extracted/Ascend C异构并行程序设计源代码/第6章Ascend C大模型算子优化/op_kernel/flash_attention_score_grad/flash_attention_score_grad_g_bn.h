#ifndef FLASH_ATTENTION_SCORE_GRAD_UG_BBN_H_
#define FLASH_ATTENTION_SCORE_GRAD_UG_BBN_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

template <typename TOUT, typename TIN, const bool NEED_CAST>
__aicore__ inline void ClcCast(LocalTensor<TOUT>& outputTensor, LocalTensor<TIN>& inputTensor, uint32_t eleNum,
                               const RoundMode& roundMode);

template <typename T1, typename T2, const bool NEED_CAST = false>
class FlashAttentionScoreGradUgBbn {
 public:
  __aicore__ inline FlashAttentionScoreGradUgBbn(){};
  __aicore__ inline void Init(__gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* dx, __gm__ uint8_t* query,
                              __gm__ uint8_t* pse_shift, __gm__ uint8_t* drop_mask, __gm__ uint8_t* atten_mask,
                              __gm__ uint8_t* forward_res, __gm__ uint8_t* softmax_max, __gm__ uint8_t* softmax_sum,
                              __gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv, __gm__ uint8_t* workspace,
                              const FlashAttentionScoreGradTilingDataUgBbn* __restrict ordTilingData, TPipe* pipe_in);
  __aicore__ inline void Process();

  using AType1 = MatmulType<TPosition::GM, CubeFormat::ND, T1, false>;
  using BType1 = MatmulType<TPosition::GM, CubeFormat::ND, T1, true>;
  using CType1 = MatmulType<TPosition::VECCALC, CubeFormat::ND, T2>;
  using BiasType1 = MatmulType<TPosition::GM, CubeFormat::ND, float>;

  using AType2 = MatmulType<TPosition::VECCALC, CubeFormat::ND, T1, true>;
  using BType2 = MatmulType<TPosition::GM, CubeFormat::ND, T1, true>;
  using CType2 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
  Matmul<AType1, BType1, CType1, BiasType1> mm1;
  Matmul<AType1, BType1, CType1, BiasType1> mm2;
  Matmul<AType2, BType2, CType2, BiasType1> mm3;
  Matmul<AType2, BType2, CType2, BiasType1> mm4;

  /* common function */
  __aicore__ inline void InitGm();
  __aicore__ inline void InitSplitParams();
  __aicore__ inline void ClcSoftMaxGrad(LocalTensor<T2>& softMaxGradOut);

  __aicore__ inline void CopyInPadForPse(LocalTensor<T1>& dst, const GlobalTensor<T1>& src, int32_t innerReduceNum,
                                         uint32_t* innerReduceShape);

  __aicore__ inline void CopyInPadUint8(LocalTensor<uint8_t>& dst, const GlobalTensor<uint8_t>& src,
                                        int32_t innerReduceNum, uint32_t* innerReduceShape);

  __aicore__ inline void CopyInPadTranspose(LocalTensor<T1>& dst, const GlobalTensor<T1>& src, int32_t innerReduceNum,
                                            uint32_t* innerReduceShape);

  __aicore__ inline void ClcDropout(LocalTensor<T2>& dst, LocalTensor<T2>& src1, LocalTensor<uint8_t>& src2);

  __aicore__ inline void PseAdd(LocalTensor<T2>& AddOut, LocalTensor<T2>& BMM2Out);

  __aicore__ inline void ClcMask(LocalTensor<T2>& MaskOut, LocalTensor<T2>& AddOut);

  __aicore__ inline void ClcSimpleSoftMax(LocalTensor<T2>& softmaxOut, LocalTensor<T2>& MaskOut);

  __aicore__ inline void ClcDropoutForBitsMode(LocalTensor<T2>& dropOutOut, LocalTensor<T2>& softmaxOut);

  __aicore__ inline void ClcSub(LocalTensor<T2>& subResult, LocalTensor<T2>& subLeft, LocalTensor<T2>& subRight);

  __aicore__ inline void ClcMatMul(const LocalTensor<T2>& tensorC, const GlobalTensor<T1>& tensorA,
                                   const GlobalTensor<T1>& tensorB, bool transA = false, bool transB = false);

  __aicore__ inline void ClcMatMul2(const GlobalTensor<float>& tensorC, const LocalTensor<T1>& tensorA,
                                    const GlobalTensor<T1>& tensorB, bool transA = false, bool transB = false);

  __aicore__ inline void ClcMatMul3(const GlobalTensor<float>& tensorC, const LocalTensor<T1>& tensorA,
                                    const GlobalTensor<T1>& tensorB, bool transA = false, bool transB = false);

  __aicore__ inline void CalculateBlockInputParamOffeset(uint64_t gBatchIdx);

  __aicore__ inline void DetermineLoopParams(uint64_t batchIdx);

  __aicore__ inline void ClcCastT1T2(LocalTensor<T1>& outputTensor, LocalTensor<T2>& inputTensor, uint32_t eleNum,
                                     const RoundMode& roundMode);

  __aicore__ inline void ClcCastT2T1(LocalTensor<T2>& outputTensor, LocalTensor<T1>& inputTensor, uint32_t eleNum,
                                     const RoundMode& roundMode);

 protected:
  /* 1. define the que */
  TPipe* pipe;
  TQue<QuePosition::VECIN, 1> vecInQue1;
  TQue<QuePosition::VECIN, 1> vecInQue2;
  TQue<QuePosition::VECOUT, 1> vecOutQue;
  TBuf<> vecClc1;
  TBuf<> vecClc2;
  TBuf<> vecClc3;

  /* 2. tiling data */
  const FlashAttentionScoreGradTilingDataUgBbn* __restrict ordTilingData;

  /* 3. input and output params */
  GlobalTensor<T1> keyGm, valueGm, dxGm, queryGm, forwardResGm, pseShiftGm;
  GlobalTensor<float> softmaxMaxGm, softmaxSumGm;
  GlobalTensor<uint8_t> dropMaskGm;
  GlobalTensor<T1> attenMaskGm;
  GlobalTensor<float> dqGm, dkGm, dvGm;
  GlobalTensor<T2> dpseGm;

  GlobalTensor<T2> workspaceGm;
  GlobalTensor<int32_t> syncGlobal;

  constexpr static uint32_t block_count = 1;

  /* datacopy shape */
  uint32_t inShapeS1D[2];
  uint32_t inShapeS11[2];
  uint32_t inShapeS1S2[2];
  uint32_t inShapeS1S2T1[2];
  uint32_t inShapeS1S2T2[2];
  uint32_t innerMatOutShape[2];
  uint32_t innerReduceShape[2];

  uint64_t mSFlaForClcOffset;
  uint64_t mSOutForClcOffset;

  /* split info */
  // current core index and used core nums
  uint64_t mBlockIdx;
  uint64_t coreNum;

  // original input dims
  int32_t B;
  int32_t N;
  int32_t N2;
  int32_t G;
  int32_t S1;
  int32_t S2;
  int32_t D;

  // The position of each Dimension at iteration i
  int32_t bPos;
  int32_t nPos;

  // current batch block offset for per format input parameters
  int32_t OffsetS1D;  // q, dy, attention
  int32_t OffsetS2D;
  int32_t OffsetS1S2;
  int32_t OffsetS11;
  int32_t batchOffsetS1D;  // q, dy, attention
  int32_t batchOffsetS2D;
  int32_t batchOffsetS1S2;
  int32_t batchOffsetS11;

  // datacopy size for per format input parameters
  uint32_t inSize;
  uint32_t inSizeS1D;
  uint32_t inSizeS1S2;
  uint32_t inSizeS11;

  // split base param
  uint32_t batchRange;
  uint32_t splitBatchRange;
  uint32_t gOut;
  uint32_t gInner;
  uint32_t gTail;

  T2 mulsValue = -10000.0;

  int32_t innerReduceNum;

  int32_t innerMatResNum;
  bool isDrop;

  int32_t layout;
};

template <typename T1, typename T2, const bool NEED_CAST>
__aicore__ inline void FlashAttentionScoreGradUgBbn<T1, T2, NEED_CAST>::Init(
    __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* dx, __gm__ uint8_t* query, __gm__ uint8_t* pse_shift,
    __gm__ uint8_t* drop_mask, __gm__ uint8_t* atten_mask, __gm__ uint8_t* forward_res, __gm__ uint8_t* softmax_max,
    __gm__ uint8_t* softmax_sum, __gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv, __gm__ uint8_t* workspace,
    const FlashAttentionScoreGradTilingDataUgBbn* __restrict ordTilingData, TPipe* pipe_in) {
  this->keyGm.SetGlobalBuffer((__gm__ T1*)key);
  this->valueGm.SetGlobalBuffer((__gm__ T1*)value);
  this->dxGm.SetGlobalBuffer((__gm__ T1*)dx);
  this->queryGm.SetGlobalBuffer((__gm__ T1*)query);
  this->pseShiftGm.SetGlobalBuffer((__gm__ T1*)pse_shift);
  this->dropMaskGm.SetGlobalBuffer((__gm__ uint8_t*)drop_mask);
  this->attenMaskGm.SetGlobalBuffer((__gm__ T1*)atten_mask);
  this->forwardResGm.SetGlobalBuffer((__gm__ T1*)forward_res);
  this->softmaxMaxGm.SetGlobalBuffer((__gm__ float*)softmax_max);
  this->softmaxSumGm.SetGlobalBuffer((__gm__ float*)softmax_sum);

  this->dqGm.SetGlobalBuffer((__gm__ float*)dq);
  this->dkGm.SetGlobalBuffer((__gm__ float*)dk);
  this->dvGm.SetGlobalBuffer((__gm__ float*)dv);

  this->workspaceGm.SetGlobalBuffer((__gm__ T2*)workspace);
  uint64_t syncCoresBuffSize = 100 * 8;
  syncGlobal.SetGlobalBuffer((__gm__ int32_t*)workspace, syncCoresBuffSize);

  this->mBlockIdx = GetBlockIdx();
  this->ordTilingData = ordTilingData;
  this->pipe = pipe_in;

  this->B = ordTilingData->gBNBaseParams.B;
  this->N = ordTilingData->gBNBaseParams.N1;
  this->N2 = ordTilingData->gBNBaseParams.N2;
  this->G = ordTilingData->gBNBaseParams.G;
  this->S1 = ordTilingData->gBNBaseParams.S1;
  this->S2 = ordTilingData->gBNBaseParams.S2;
  this->D = ordTilingData->gBNBaseParams.D;

  this->gInner = ordTilingData->gBNSplitCoreParams.gInner;
  this->gOut = ordTilingData->gBNSplitCoreParams.gOut;
  this->gTail = ordTilingData->gBNSplitCoreParams.gTail;

  this->coreNum = ordTilingData->gBNBaseParams.coreNum;
  this->layout = this->ordTilingData->gBNBaseParams.layout;

  uint32_t maxN = (S2 > D) ? S2 : D;
  uint64_t maxAlignSize = Ceil(maxN, 32 / sizeof(T1)) * (32 / sizeof(T1));
  uint64_t bufferSize = this->gInner * S1 * maxAlignSize * sizeof(T2);

  pipe->InitBuffer(vecInQue1, 1, bufferSize);
  pipe->InitBuffer(vecInQue2, 1, bufferSize);
  pipe->InitBuffer(vecOutQue, 1, bufferSize);
  pipe->InitBuffer(vecClc1, bufferSize);
  pipe->InitBuffer(vecClc2, bufferSize);
  pipe->InitBuffer(vecClc3, bufferSize);

  uint64_t workspaceSize = this->gInner * S1 * Ceil(S2, 32 / sizeof(T2)) * 32 / sizeof(T2);
  uint64_t mmWorkspaceSize = workspaceSize * sizeof(T2);
  uint64_t offset0 = 800 + this->mBlockIdx * mmWorkspaceSize;
  uint64_t offsetWorkSpace = 800 + GetBlockNum() * GetTaskRation() * mmWorkspaceSize;
  uint64_t offset1 = offsetWorkSpace + this->mBlockIdx * mmWorkspaceSize;
  uint64_t offset2 = offset1 + GetBlockNum() * GetTaskRation() * mmWorkspaceSize;

  pipe->InitSpmBuffer(workspaceGm[offset0], mmWorkspaceSize);
  this->mm1.SetWorkspace((__gm__ uint8_t*)this->workspaceGm[offset1].GetPhyAddr(), mmWorkspaceSize);
  this->mm2.SetWorkspace((__gm__ uint8_t*)this->workspaceGm[offset2].GetPhyAddr(), mmWorkspaceSize);

  this->isDrop = (this->ordTilingData->gBNBaseParams.keepProb < 1) ? true : false;
}

template <typename T1, typename T2, const bool NEED_CAST>
__aicore__ inline void FlashAttentionScoreGradUgBbn<T1, T2, NEED_CAST>::Process() {
  InitGm();

  InitSplitParams();

  for (uint64_t batchIdx = 0; batchIdx < this->splitBatchRange; batchIdx++) {
    // 搬入的 offset 大小
    DetermineLoopParams(batchIdx);

    for (uint64_t gBatchIdx = 0; gBatchIdx < this->gOut; gBatchIdx++) {
      // calculate g dim input parameters offset
      CalculateBlockInputParamOffeset(gBatchIdx);
      /* ---------------------- stage1 -------------------- */
      // softMaxGradOut : 存储SoftMaxGrad算子结果
      LocalTensor<T2> softMaxGradOut = vecClc3.Get<T2>(inSizeS11 * 32 / sizeof(T2));
      softMaxGradOut.SetShapeInfo(ShapeInfo(2, inShapeS11, DataFormat::ND));
      ClcSoftMaxGrad(softMaxGradOut);

      // BMM1Out : 存储BMM1算子结果
      LocalTensor<T2> BMM1Out = vecClc1.Get<T2>(inSizeS1S2);
      BMM1Out.SetShapeInfo(ShapeInfo(2, inShapeS1S2T1, DataFormat::ND));
      pipe_barrier(PIPE_V);
      ClcMatMul(BMM1Out, dxGm[OffsetS1D], valueGm[OffsetS2D], false, true);

      if (isDrop) {
        ClcDropoutForBitsMode(BMM1Out, BMM1Out);
      }

      ClcSub(BMM1Out, BMM1Out, softMaxGradOut);

      /* ---------------------- stage1 -------------------- */

      /* ---------------------- stage2 -------------------- */

      // BMM2 搬运和计算
      LocalTensor<T2> BMM2Out = vecClc2.Get<T2>();
      LocalTensor<T2>& AddOut = BMM2Out;
      LocalTensor<T2>& MaskOut = BMM2Out;
      LocalTensor<T2>& softmaxOut = BMM2Out;
      LocalTensor<T2>& dropOut = BMM2Out;

      BMM2Out.SetShapeInfo(ShapeInfo(2, inShapeS1S2T2, DataFormat::ND));
      ClcMatMul(BMM2Out, queryGm[OffsetS1D], keyGm[OffsetS2D], false, true);
      Muls(BMM2Out, BMM2Out, (T2)(this->ordTilingData->gBNBaseParams.scaleValue), inSizeS1S2);

      // Add 搬运和计算
      if (pseShiftGm.GetPhyAddr() != nullptr) {
        PseAdd(AddOut, BMM2Out);
      }

      // MASK 搬运和计算
      ClcMask(MaskOut, AddOut);

      // SimpleSoftMax搬运和计算
      ClcSimpleSoftMax(softmaxOut, MaskOut);

      // Mul搬运和计算
      LocalTensor<T2>& frontResInner = BMM1Out;
      LocalTensor<T2>& mulResInner = frontResInner;

      Mul(mulResInner, softmaxOut, frontResInner, inSizeS1S2);
      Muls(mulResInner, mulResInner, (T2)(this->ordTilingData->gBNBaseParams.scaleValue), inSizeS1S2);

      LocalTensor<T1> castedMulResInner = vecClc1.Get<T1>();
      castedMulResInner.SetShapeInfo(ShapeInfo(2, inShapeS1S2T1, DataFormat::ND));
      ClcCastT1T2(castedMulResInner, mulResInner, inSizeS1S2, RoundMode::CAST_RINT);
      /* ---------------------- stage2 -------------------- */

      /* ---------------------- stage3 -------------------- */
      ClcMatMul3(this->dqGm[OffsetS1D], castedMulResInner, this->keyGm[OffsetS2D], false, false);

      ClcMatMul2(this->dkGm[OffsetS2D], castedMulResInner, this->queryGm[OffsetS1D], true, false);

      if (isDrop) {
        ClcDropoutForBitsMode(dropOut, softmaxOut);
      }

      LocalTensor<T1> castedDropOut = vecClc3.Get<T1>();
      castedDropOut.SetShapeInfo(ShapeInfo(2, inShapeS1S2T1, DataFormat::ND));
      ClcCastT1T2(castedDropOut, dropOut, inSizeS1S2, RoundMode::CAST_RINT);

      ClcMatMul2(this->dvGm[OffsetS2D], castedDropOut, this->dxGm[OffsetS1D], true, false);
      /* ---------------------- stage3 -------------------- */
    }
  }
}

template <typename T1, typename T2, const bool NEED_CAST>
__aicore__ inline void FlashAttentionScoreGradUgBbn<T1, T2, NEED_CAST>::InitGm() {
  uint64_t dQSize = this->B * this->N2 * this->G * this->S1 * this->D;

  uint64_t dKdVSize = this->B * this->N2 * this->S2 * this->D;

  // dq size per core
  uint32_t dQPerSize = (dQSize + this->coreNum - 1) / this->coreNum;  // ceil(dQSize, coreNum)
  uint32_t dQUsedCore = (dQSize + dQPerSize - 1) / dQPerSize;
  uint32_t dQTailSize = dQSize - dQPerSize * (dQUsedCore - 1);

  uint32_t dQInitSize = (mBlockIdx == dQUsedCore - 1) ? dQTailSize : dQPerSize;
  uint32_t dQOffset = mBlockIdx * dQPerSize;

  // dv/dk size per core
  uint32_t dKdVPerSize = (dKdVSize + this->coreNum - 1) / this->coreNum;  // ceil(dQSize, coreNum)
  uint32_t dKdVUsedCore = (dKdVSize + dKdVPerSize - 1) / dKdVPerSize;
  uint32_t dKdVTailSize = dKdVSize - dKdVPerSize * (dKdVUsedCore - 1);

  uint32_t dKdVInitSize = (mBlockIdx == dKdVUsedCore - 1) ? dKdVTailSize : dKdVPerSize;
  uint32_t dKdVOffset = mBlockIdx * dKdVPerSize;

  // clear output space
  InitOutput<float>(dqGm[dQOffset], dQInitSize, 0.0);
  InitOutput<float>(dkGm[dKdVOffset], dKdVInitSize, 0.0);
  InitOutput<float>(dvGm[dKdVOffset], dKdVInitSize, 0.0);

  // 全局同步
  LocalTensor<int32_t> workLocal = vecInQue1.AllocTensor<int32_t>();
  SyncAll<false>(syncGlobal, workLocal);
  vecInQue1.FreeTensor(workLocal);
}

template <typename T1, typename T2, const bool NEED_CAST>
__aicore__ inline void FlashAttentionScoreGradUgBbn<T1, T2, NEED_CAST>::InitSplitParams() {
  uint32_t batchRange = ordTilingData->gBNSplitCoreParams.batchRange;
  uint32_t splitedBatchRange = ordTilingData->gBNSplitCoreParams.splitedBatchRange;
  uint32_t remainBatchRange = ordTilingData->gBNSplitCoreParams.splitedRemainBatchRange;

  this->batchRange = batchRange;
  this->splitBatchRange = (this->mBlockIdx == this->coreNum - 1) ? splitedBatchRange : remainBatchRange;
}

template <typename T1, typename T2, const bool NEED_CAST>
__aicore__ inline void FlashAttentionScoreGradUgBbn<T1, T2, NEED_CAST>::ClcSoftMaxGrad(
    LocalTensor<T2>& softMaxGradOut) {
  // dxGm
  LocalTensor<T1> dxIn = vecInQue1.AllocTensor<T1>();
  CopyInPadTranspose(dxIn, dxGm[OffsetS1D], inSizeS1D, inShapeS1D);
  vecInQue1.EnQue(dxIn);
  vecInQue1.DeQue<T1>();

  // forwardResGm
  LocalTensor<T1> forwardResIn = vecInQue2.AllocTensor<T1>();
  CopyInPadTranspose(forwardResIn, forwardResGm[OffsetS1D], inSizeS1D, inShapeS1D);
  vecInQue2.EnQue(forwardResIn);
  vecInQue2.DeQue<T1>();

  LocalTensor<T2> casteddxIn = vecClc1.Get<T2>();
  casteddxIn.SetShapeInfo(ShapeInfo(2, inShapeS1D, DataFormat::ND));
  ClcCastT2T1(casteddxIn, dxIn, inSizeS1D, RoundMode::CAST_NONE);

  LocalTensor<T2> castedforwardResIn = vecClc2.Get<T2>();
  castedforwardResIn.SetShapeInfo(ShapeInfo(2, inShapeS1D, DataFormat::ND));
  ClcCastT2T1(castedforwardResIn, forwardResIn, inSizeS1D, RoundMode::CAST_NONE);

  LocalTensor<uint8_t> apiClcTensor = vecOutQue.AllocTensor<uint8_t>();
  SoftmaxGradFront<T2, false>(softMaxGradOut, castedforwardResIn, casteddxIn, apiClcTensor,
                              this->ordTilingData->softmaxGradTilingData);

  vecInQue1.FreeTensor(dxIn);
  vecInQue2.FreeTensor(forwardResIn);
  vecOutQue.FreeTensor(apiClcTensor);
}

template <typename T1, typename T2, const bool NEED_CAST>
__aicore__ inline void FlashAttentionScoreGradUgBbn<T1, T2, NEED_CAST>::ClcSub(LocalTensor<T2>& subResult,
                                                                               LocalTensor<T2>& subLeft,
                                                                               LocalTensor<T2>& subRight) {
  uint8_t repeatStride = Ceil(this->S2, (32 / sizeof(T2)));
  uint32_t vecCalNum = 256 / sizeof(T2);
  uint32_t subVecCalCount = Ceil((uint32_t)S2, vecCalNum);
  uint32_t tailVecCalNum = S2 - vecCalNum * (subVecCalCount - 1);

  for (uint32_t subIdx = 0; subIdx < subVecCalCount; subIdx++) {
    uint32_t subMask = (subIdx == subVecCalCount - 1) ? tailVecCalNum : vecCalNum;
    uint32_t srcOffset = 0;
    uint32_t dstOffset = subIdx * vecCalNum;
    // dst_blk_stride, src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride, src1_rep_stride
    Sub(subResult[dstOffset], subLeft[dstOffset], subRight[srcOffset], subMask, S1,
        {1, 1, 0, repeatStride, repeatStride, 1});
  }
}

template <typename T1, typename T2, const bool NEED_CAST>
__aicore__ inline void FlashAttentionScoreGradUgBbn<T1, T2, NEED_CAST>::PseAdd(LocalTensor<T2>& AddOut,
                                                                               LocalTensor<T2>& BMM2Out) {
  LocalTensor<T1> pseShiftIn = vecInQue2.AllocTensor<T1>();
  CopyInPadForPse(pseShiftIn, pseShiftGm[OffsetS1S2], inSizeS1S2, inShapeS1S2T1);
  vecInQue2.EnQue(pseShiftIn);
  vecInQue2.DeQue<T1>();
  LocalTensor<T2> castedpseShiftIn = vecClc1.Get<T2>();
  castedpseShiftIn.SetShapeInfo(ShapeInfo(2, inShapeS1S2T2, DataFormat::ND));
  ClcCastT2T1(castedpseShiftIn, pseShiftIn, inSizeS1S2, RoundMode::CAST_NONE);
  Add(AddOut, BMM2Out, castedpseShiftIn, inSizeS1S2);
  vecInQue2.FreeTensor(pseShiftIn);
}

template <typename T1, typename T2, const bool NEED_CAST>
__aicore__ inline void FlashAttentionScoreGradUgBbn<T1, T2, NEED_CAST>::ClcMask(LocalTensor<T2>& MaskOut,
                                                                                LocalTensor<T2>& AddOut) {
  LocalTensor<T1> attenMaskIn = vecInQue2.AllocTensor<T1>();
  DataCopyPad(attenMaskIn, this->attenMaskGm[0],
              {static_cast<uint16_t>(S1), static_cast<uint16_t>(S2 * sizeof(T1)), 0, 0}, {false, 0, 0, 0});
  vecInQue2.EnQue(attenMaskIn);
  vecInQue2.DeQue<T1>();

  LocalTensor<T2> castedAttenMaskIn = vecInQue1.AllocTensor<T2>();
  castedAttenMaskIn.SetSize(inSizeS1S2);
  castedAttenMaskIn.SetShapeInfo(ShapeInfo(2, inShapeS1S2T2, DataFormat::ND));
  ClcCastT2T1(castedAttenMaskIn, attenMaskIn, inSizeS1S2, RoundMode::CAST_NONE);
  Muls(castedAttenMaskIn, castedAttenMaskIn, mulsValue, inSizeS1S2);  // -10000
  pipe_barrier(PIPE_V);
  Add(MaskOut, castedAttenMaskIn, AddOut, inSizeS1S2);
  vecInQue1.FreeTensor(attenMaskIn);
  vecInQue2.FreeTensor(attenMaskIn);
}

template <typename T1, typename T2, const bool NEED_CAST>
__aicore__ inline void FlashAttentionScoreGradUgBbn<T1, T2, NEED_CAST>::ClcSimpleSoftMax(LocalTensor<T2>& softmaxOut,
                                                                                         LocalTensor<T2>& MaskOut) {
  LocalTensor<float> maxIn = vecInQue1.AllocTensor<float>();
  DataCopyPad(maxIn, this->softmaxMaxGm[OffsetS11], {1, (uint16_t)(inSizeS11 * 32), 0, 0}, {false, 0, 0, 0});
  vecInQue1.EnQue(maxIn);
  vecInQue1.DeQue<float>();
  LocalTensor<float> sumIn = vecInQue2.AllocTensor<float>();
  DataCopyPad(sumIn, this->softmaxSumGm[OffsetS11], {1, (uint16_t)(inSizeS11 * 32), 0, 0}, {false, 0, 0, 0});
  vecInQue2.EnQue(sumIn);
  vecInQue2.DeQue<float>();
  /*
      关于基本块：isBasicBlock， 输入和输出是否为基本块，可以使能该参数用于提升性能
      基本块定义为:满足last轴的大小为256Byte的倍数，非last轴乘积长度为8的倍数的输入tensor都可以称为基本块。
  */
  uint32_t shapeArray[2];
  shapeArray[0] = this->S1;
  shapeArray[1] = Ceil(this->S2, (32 / sizeof(T2)) * 32 / sizeof(T2));
  softmaxOut.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));

  LocalTensor<uint8_t> apiClcTensor = vecOutQue.AllocTensor<uint8_t>();
  bool isBasicBlock = (this->S1 % 8 == 0) && (this->S2 * sizeof(T2) % 256 == 0);
  if (isBasicBlock) {
    SimpleSoftMax<T2, true, true>(softmaxOut, sumIn, maxIn, MaskOut, apiClcTensor,
                                  this->ordTilingData->softmaxTilingData);
  } else {
    SimpleSoftMax<T2, true, false>(softmaxOut, sumIn, maxIn, MaskOut, apiClcTensor,
                                   this->ordTilingData->softmaxTilingData);
  }
  vecInQue1.FreeTensor(maxIn);
  vecInQue2.FreeTensor(sumIn);
  vecOutQue.FreeTensor(apiClcTensor);
}

template <typename T1, typename T2, const bool NEED_CAST>
__aicore__ inline void FlashAttentionScoreGradUgBbn<T1, T2, NEED_CAST>::ClcDropoutForBitsMode(
    LocalTensor<T2>& dropOut, LocalTensor<T2>& softmaxOut) {
  LocalTensor<uint8_t> dpMaskIn = vecInQue2.AllocTensor<uint8_t>();
  CopyInPadUint8(dpMaskIn, dropMaskGm[OffsetS1S2], inSizeS1S2, inShapeS1S2);
  vecInQue2.EnQue(dpMaskIn);
  vecInQue2.DeQue<uint8_t>();
  ClcDropout(dropOut, softmaxOut, dpMaskIn);
  vecInQue2.FreeTensor(dpMaskIn);
}

template <typename T1, typename T2, const bool NEED_CAST>
__aicore__ inline void FlashAttentionScoreGradUgBbn<T1, T2, NEED_CAST>::CopyInPadForPse(LocalTensor<T1>& dst,
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

template <typename T1, typename T2, const bool NEED_CAST>
__aicore__ inline void FlashAttentionScoreGradUgBbn<T1, T2, NEED_CAST>::CopyInPadUint8(LocalTensor<uint8_t>& dst,
                                                                                       const GlobalTensor<uint8_t>& src,
                                                                                       int32_t innerReduceNum,
                                                                                       uint32_t* innerReduceShape) {
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

template <typename T1, typename T2, const bool NEED_CAST>
__aicore__ inline void FlashAttentionScoreGradUgBbn<T1, T2, NEED_CAST>::CopyInPadTranspose(LocalTensor<T1>& dst,
                                                                                           const GlobalTensor<T1>& src,
                                                                                           int32_t innerReduceNum,
                                                                                           uint32_t* innerReduceShape) {
  dst.SetSize(innerReduceNum);
  dst.SetShapeInfo(ShapeInfo(2, innerReduceShape, DataFormat::ND));
  uint32_t blockCount;
  uint32_t blockLen;
  uint32_t srcStride;
  uint32_t alignLen;
  uint32_t padding;

  struct DataCopyParams intriParams;
  DataCopyPadParams padParams;

  if (layout == 0 || layout == 3) {  // 0:BSH  3:BSND
    blockCount = S1;
    blockLen = D * sizeof(T1);
    srcStride = (N * G * D - D) * sizeof(T1);
    alignLen = Ceil(D, 32 / sizeof(T1)) * 32 / sizeof(T1);
    padding = alignLen - D;
  } else if (layout == 1) {  // SBH
    blockCount = S1;
    blockLen = D * sizeof(T1);
    srcStride = (B * N * G * D - D) * sizeof(T1);
    alignLen = Ceil(D, 32 / sizeof(T1)) * 32 / sizeof(T1);
    padding = alignLen - D;
  } else {  // BNSD
    // 对齐部分的block数，默认BNSD
    blockCount = 1;
    blockLen = innerReduceNum * sizeof(T1);
    srcStride = 0;
    alignLen = Ceil(innerReduceNum, 32 / sizeof(T1)) * 32 / sizeof(T1);
    padding = alignLen - innerReduceNum;
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

template <typename T1, typename T2, const bool NEED_CAST>
__aicore__ inline void FlashAttentionScoreGradUgBbn<T1, T2, NEED_CAST>::ClcDropout(LocalTensor<T2>& dst,
                                                                                   LocalTensor<T2>& src1,
                                                                                   LocalTensor<uint8_t>& src2) {
  DropOutShapeInfo info;
  info.firstAxis = S1;
  info.srcLastAxis = S2;
  info.maskLastAxis = (S2 + 31) / 32 * 32 / 8;
  LocalTensor<uint8_t> apiClcTensor = vecOutQue.AllocTensor<uint8_t>();
  DropOut<T2, true>(dst, src1, src2, apiClcTensor, ordTilingData->gBNBaseParams.keepProb, info);

  vecOutQue.FreeTensor(apiClcTensor);
}

template <typename T1, typename T2, const bool NEED_CAST>
__aicore__ inline void FlashAttentionScoreGradUgBbn<T1, T2, NEED_CAST>::ClcMatMul(const LocalTensor<T2>& tensorC,
                                                                                  const GlobalTensor<T1>& tensorA,
                                                                                  const GlobalTensor<T1>& tensorB,
                                                                                  bool transA, bool transB) {
  mm1.SetTail(S1, S2, D);  // M N K
  mm1.SetTensorA(tensorA, transA);
  mm1.SetTensorB(tensorB, transB);
  mm1.template IterateAll<true>(tensorC, 0);
}

template <typename T1, typename T2, const bool NEED_CAST>
__aicore__ inline void FlashAttentionScoreGradUgBbn<T1, T2, NEED_CAST>::ClcMatMul2(const GlobalTensor<float>& tensorC,
                                                                                   const LocalTensor<T1>& tensorA,
                                                                                   const GlobalTensor<T1>& tensorB,
                                                                                   bool transA, bool transB) {
  // batched matmul
  for (int i = 0; i < this->inSize; i++) {
    mm3.SetTail(S2, D, S1);
    mm3.SetTensorA(tensorA, transA);
    mm3.SetTensorB(tensorB, transB);
    mm3.template IterateAll<true>(tensorC, 0);
  }
}

template <typename T1, typename T2, const bool NEED_CAST>
__aicore__ inline void FlashAttentionScoreGradUgBbn<T1, T2, NEED_CAST>::ClcMatMul3(const GlobalTensor<float>& tensorC,
                                                                                   const LocalTensor<T1>& tensorA,
                                                                                   const GlobalTensor<T1>& tensorB,
                                                                                   bool transA, bool transB) {
  // batched matmul
  for (int i = 0; i < this->inSize; i++) {
    mm4.SetTail(S1, D, S2);
    mm4.SetTensorA(tensorA, transA);
    mm4.SetTensorB(tensorB, transB);
    mm4.template IterateAll<false>(tensorC, 0);
  }
}

template <typename T1, typename T2, const bool NEED_CAST>
__aicore__ inline void FlashAttentionScoreGradUgBbn<T1, T2, NEED_CAST>::CalculateBlockInputParamOffeset(
    uint64_t gBatchIdx) {
  bool isTailBatch = (this->mBlockIdx == this->coreNum - 1 && gBatchIdx == this->gOut - 1);
  this->inSize = isTailBatch ? this->gTail : this->gInner;
  this->inSizeS1D = this->inSize * S1 * Ceil(D, 32 / sizeof(T1) * 32 / sizeof(T1));
  ;
  this->inSizeS1S2 = this->inSize * S1 * Ceil(S2, 32 / sizeof(T2) * 32 / sizeof(T2));
  ;
  this->inSizeS11 = this->inSize * S1;

  this->inShapeS1D[0] = static_cast<uint32_t>(this->inSizeS11);
  this->inShapeS1D[1] = Ceil(D, 32 / sizeof(T1)) * 32 / sizeof(T1);
  this->inShapeS1S2[0] = static_cast<uint32_t>(this->inSizeS11);
  this->inShapeS1S2[1] = Ceil(S2, 32 / sizeof(uint8_t)) * 32 / sizeof(uint8_t);
  this->inShapeS1S2T1[0] = static_cast<uint32_t>(this->inSizeS11);
  this->inShapeS1S2T1[1] = Ceil(S2, 32 / sizeof(T1)) * 32 / sizeof(T1);
  this->inShapeS1S2T2[0] = static_cast<uint32_t>(this->inSizeS11);
  this->inShapeS1S2T2[1] = Ceil(S2, 32 / sizeof(T2)) * 32 / sizeof(T2);
  this->inShapeS11[0] = static_cast<uint32_t>(this->inSizeS11);
  this->inShapeS11[1] = 32 / sizeof(T2);

  if (layout == 0) {  // shape BSNGD or SBNGD
    this->OffsetS1D = this->batchOffsetS1D + gBatchIdx * this->gInner * D;
    this->OffsetS2D = this->batchOffsetS2D + gBatchIdx * this->gInner * D;
  } else {  // shape BNGSD
    this->OffsetS1D = this->batchOffsetS1D + gBatchIdx * this->gInner * S1 * D;
    this->OffsetS2D = this->batchOffsetS2D + gBatchIdx * this->gInner * S2 * D;
  }

  // 内部格式都是BNGSS格式
  this->OffsetS1S2 = this->batchOffsetS1S2 + gBatchIdx * this->gInner * S1 * S2;
  this->OffsetS11 = (this->batchOffsetS11 + gBatchIdx * this->gInner * S1) * 8;
}

template <typename T1, typename T2, const bool NEED_CAST>
__aicore__ inline void FlashAttentionScoreGradUgBbn<T1, T2, NEED_CAST>::DetermineLoopParams(uint64_t batchIdx) {
  uint64_t bIdx = (this->mBlockIdx * this->batchRange + batchIdx) / this->N;
  uint64_t nIdx = (this->mBlockIdx * this->batchRange + batchIdx) % this->N;

  if (layout == 0) {  // shape BSH / BSND
    this->batchOffsetS1D = bIdx * S1 * N * G * D + nIdx * G * D;
    this->batchOffsetS2D = bIdx * S2 * N * G * D + nIdx * G * D;
  } else if (layout == 1) {  // shape SBH
    this->batchOffsetS1D = bIdx * N * G * D + nIdx * G * D;
    this->batchOffsetS2D = this->batchOffsetS1D;
  } else {  // shape BNSD
    this->batchOffsetS1D = bIdx * N * G * S1 * D + nIdx * G * S1 * D;
    this->batchOffsetS2D = bIdx * N * G * S2 * D + nIdx * G * S2 * D;
  }

  // 内部格式都是BNGSS格式
  this->batchOffsetS1S2 = bIdx * N * G * S1 * S2 + nIdx * G * S1 * S2;
  this->batchOffsetS11 = bIdx * N * G * S1 + nIdx * G * S1;
}

template <typename T1, typename T2, const bool NEED_CAST>
__aicore__ inline void FlashAttentionScoreGradUgBbn<T1, T2, NEED_CAST>::ClcCastT1T2(LocalTensor<T1>& outputTensor,
                                                                                    LocalTensor<T2>& inputTensor,
                                                                                    uint32_t eleNum,
                                                                                    const RoundMode& mode) {
  if constexpr (NEED_CAST) {
    pipe_barrier(PIPE_V);
    Cast(outputTensor, inputTensor, mode, eleNum);
    pipe_barrier(PIPE_V);
  } else {
    outputTensor = inputTensor;
  }
}

template <typename T1, typename T2, const bool NEED_CAST>
__aicore__ inline void FlashAttentionScoreGradUgBbn<T1, T2, NEED_CAST>::ClcCastT2T1(LocalTensor<T2>& outputTensor,
                                                                                    LocalTensor<T1>& inputTensor,
                                                                                    uint32_t eleNum,
                                                                                    const RoundMode& mode) {
  if constexpr (NEED_CAST) {
    pipe_barrier(PIPE_V);
    Cast(outputTensor, inputTensor, mode, eleNum);
    pipe_barrier(PIPE_V);
  } else {
    outputTensor = inputTensor;
  }
}
#endif
