/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _FLASH_ATTENTION_SCORE_GRAD_US1S2_BBN2GS1S2_H_
#define _FLASH_ATTENTION_SCORE_GRAD_US1S2_BBN2GS1S2_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

template <typename T1, typename T2, const bool IS_BF16 = false>
class FlashAttentionScoreGradUs1s2Bbn2gs1s2 {
 public:
  __aicore__ inline FlashAttentionScoreGradUs1s2Bbn2gs1s2(){};

  __aicore__ inline void Init(__gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* dx, __gm__ uint8_t* query,
                              __gm__ uint8_t* pse_shift, __gm__ uint8_t* drop_mask, __gm__ uint8_t* atten_mask,
                              __gm__ uint8_t* forward_res, __gm__ uint8_t* softmax_max, __gm__ uint8_t* softmax_sum,
                              __gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv, __gm__ uint8_t* dpse,
                              __gm__ uint8_t* workspace,
                              const FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2* __restrict ordTilingData,
                              TPipe* pipe_in);
  __aicore__ inline void CopyInSoftMaxGrad(LocalTensor<T1>& dstTensor, uint64_t softmaxGradOffset, uint32_t s1Extend,
                                           uint32_t dExtend, uint32_t dExtendAlign);
  __aicore__ inline void CalcSoftMaxGrad(LocalTensor<float>& sfmgClc3, uint64_t aTensorOffset, uint32_t s1Extend);
  __aicore__ inline void CalcDropoutBit(LocalTensor<T2>& dstTensor, uint64_t maskOffset, uint32_t s1Extend,
                                        uint32_t s2Extend);
  __aicore__ inline void CalcDropoutBool(LocalTensor<T2>& dstTensor, uint64_t maskOffset, uint32_t s1Extend,
                                         uint32_t s2Extend);
  __aicore__ inline void CopyInSoftMax(LocalTensor<float>& dstTensor, uint64_t softMaxOffset, uint32_t s1Extend);
  __aicore__ inline void CalcSoftMax(LocalTensor<T2>& dstTensor, LocalTensor<float>& srcTensor, uint32_t s1Extend,
                                     const SoftMaxTiling& tiling);
  __aicore__ inline void CopyInPseShift(LocalTensor<T2>& dstTensor, LocalTensor<T1> vecInTensor, uint64_t pseOffset,
                                        uint32_t s2Extend);
  __aicore__ inline void CalcPseShift(LocalTensor<T2>& dstTensor, LocalTensor<T2>& srcTensor, uint32_t s1Extend,
                                      uint32_t s2ExtendAlign);
  __aicore__ inline void CopyInAttenMaskT1(LocalTensor<T1>& dstTensor, uint64_t attenMaskOffset, uint32_t s1Extend,
                                           uint32_t s2Extend);
  __aicore__ inline void CalcAttenMaskT1(LocalTensor<T2>& dstTensor, LocalTensor<T1> srcTensor, uint32_t s1Extend,
                                         uint32_t s2Extend);
  __aicore__ inline void CopyInAttenMaskBool(LocalTensor<uint8_t>& dstTensor, uint64_t attenMaskOffset,
                                             uint32_t s1Extend, uint32_t s2Extend);
  __aicore__ inline void CalcAttenMaskBool(LocalTensor<T2>& dstTensor, LocalTensor<uint8_t> srcTensor);
  __aicore__ inline void CalcAttenMaskOffset(uint64_t& attenMaskOffset, const int64_t delta);
  __aicore__ inline void CalcCausalAttenMaskOffset(uint64_t& attenMaskOffset, const int64_t delta);
  __aicore__ inline void Process();
  __aicore__ inline bool IsSparse(uint32_t baseIndex);
  __aicore__ inline void InitIndex(uint32_t index);
  __aicore__ inline void Compute(uint32_t preIndex, uint32_t nextIndex);
  __aicore__ inline void SyncALLCores();
  __aicore__ inline bool CheckIsValidBlock(uint32_t baseIdx);

  using aType1 = MatmulType<TPosition::GM, CubeFormat::ND, T1>;
  using bType1 = MatmulType<TPosition::GM, CubeFormat::ND, T1, true>;
  using cType1 = MatmulType<TPosition::VECCALC, CubeFormat::ND, T2>;
  using biasType1 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
  using aType2 = MatmulType<TPosition::VECCALC, CubeFormat::ND, T1, true>;
  using bType2 = MatmulType<TPosition::GM, CubeFormat::ND, T1>;
  using cType2 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
  using biasType2 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
  Matmul<aType1, bType1, cType1, biasType1> mm1;
  Matmul<aType1, bType1, cType1, biasType1> mm2;
  Matmul<aType2, bType2, cType2, biasType2> mm3;
  Matmul<aType2, bType2, cType2, biasType2> mm4;

 protected:
  TPipe* pipe;
  TQue<QuePosition::VECIN, 1> vecInQue;
  TQue<QuePosition::VECOUT, 1> vecOutQue;
  TQue<QuePosition::VECIN, 1> vecClc1;
  TQue<QuePosition::VECIN, 1> vecClc2;
  TBuf<> vecClc3;

  uint32_t coreNum;
  uint64_t cBlockIdx;
  const FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2* __restrict TilingData;

  // input
  GlobalTensor<T1> keyGm, valueGm, dxGm, queryGm, forwardResGm, attenMaskGm, pseGm;
  GlobalTensor<uint8_t> maskWorkSpaceGm, attenMaskU8Gm, dropMaskGm;
  GlobalTensor<float> softmaxMaxGm, softmaxSumGm;

  // output
  GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;
  GlobalTensor<T1> dropWorkSpaceGm, mulWorkSpaceGm;

  // workspace
  GlobalTensor<uint8_t> workspaceGm;

  // AscendC
  GlobalTensor<int32_t> syncGlobal;

  // matmal1/matmal2 result buffer
  GlobalTensor<float> matmalResultBuffer1;
  GlobalTensor<float> matmalResultBuffer2;

  int32_t layout;
  constexpr static uint32_t BNGSD = 0;
  constexpr static uint32_t SBNGD = 1;
  constexpr static uint32_t BSNGD = 2;

  T2 mulsValue = -10000.0;

  // optional control
  bool isDrop;
  float keepProb;
  bool isPse;
  uint32_t s1Token;
  uint32_t s2Token;
  uint32_t sparseMode;
  bool isSparse;

  // org shape info
  uint32_t b;
  uint32_t n2;
  uint32_t g;
  uint32_t s1;
  uint32_t s2;
  uint32_t d;
  uint32_t attenMaksDimS2;

  uint32_t baseMN;

  // split info
  uint32_t s1Outer;
  uint32_t s1Inner;
  uint32_t s1Tail;
  uint32_t s1TailAlign;
  uint32_t s2Outer;
  uint32_t s2CvRatio;
  uint32_t s2CvExtend;
  uint32_t s2Inner;
  uint32_t s2Tail;
  uint32_t s2TailAlign;
  bool isPadABlock;
  uint32_t sfmgdOuter;
  uint32_t sfmgdInner;
  uint32_t sfmgdTail;
  uint32_t sfmgdTailAlign;
  uint32_t mm1CalRatio;

  uint32_t blockOuter;

  // sparse block info
  uint32_t blockStarts[50];
  uint32_t blockEnds[50];
  uint32_t prefixN[64];

  // buferinfo
  uint32_t baseUbSize;
  uint32_t matmalWorkspaceSize;
  uint32_t tmpBufferSize;

  // base info
  uint32_t n2gs1os2o;
  uint32_t gs1os2o;
  uint32_t s1os2o;
  uint32_t s2o;

  uint32_t baseIdx;
  uint32_t bDimIdx;
  uint32_t bDimTail;
  uint32_t n2DimIdx;
  uint32_t n2DimTail;
  uint32_t gDimIdx;
  uint32_t gDimTail;
  uint32_t s1oDimIdx;
  uint32_t s2oCvDimIdx;
  uint32_t s2oDimIdx;
  uint32_t s1Extend;
  uint32_t s2Extend;
  uint32_t cvExtend;
  uint32_t s1ExtendAlign;
  uint32_t s2ExtendAlign;

  int32_t isStart = 1;
  uint32_t pingpongIdx = 1;

  // calDtype
  constexpr static uint32_t calDtypeBytes = 4;

  // other const
  constexpr static uint32_t cal_block_num = 32 / sizeof(T2);
  constexpr static uint32_t cal_repeat_num = 256 / sizeof(T2);
  constexpr static uint32_t input_block_num = 32 / sizeof(T1);
  constexpr static uint32_t SYNC_GLOBAL_WORKSPACE_SIZE = 16 * 1024;
  constexpr static uint32_t ADDR_ALIGN_SIZE = 512;
  constexpr static uint32_t INPUT_NUMS = 2;
};

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2<T1, T2, IS_BF16>::Init(
    __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* dx, __gm__ uint8_t* query, __gm__ uint8_t* pse_shift,
    __gm__ uint8_t* drop_mask, __gm__ uint8_t* atten_mask, __gm__ uint8_t* forward_res, __gm__ uint8_t* softmax_max,
    __gm__ uint8_t* softmax_sum, __gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv, __gm__ uint8_t* dpse,
    __gm__ uint8_t* workspace, const FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2* __restrict ordTilingData,
    TPipe* pipe_in) {
  keyGm.SetGlobalBuffer((__gm__ T1*)key);
  valueGm.SetGlobalBuffer((__gm__ T1*)value);
  dxGm.SetGlobalBuffer((__gm__ T1*)dx);
  queryGm.SetGlobalBuffer((__gm__ T1*)query);
  forwardResGm.SetGlobalBuffer((__gm__ T1*)forward_res);
  pseGm.SetGlobalBuffer((__gm__ T1*)pse_shift);

  dropMaskGm.SetGlobalBuffer((__gm__ uint8_t*)drop_mask);
  attenMaskGm.SetGlobalBuffer((__gm__ T1*)atten_mask);
  attenMaskU8Gm.SetGlobalBuffer((__gm__ uint8_t*)atten_mask);
  softmaxMaxGm.SetGlobalBuffer((__gm__ float*)softmax_max);
  softmaxSumGm.SetGlobalBuffer((__gm__ float*)softmax_sum);

  // init current core tilingInfo
  cBlockIdx = GetBlockIdx();
  TilingData = ordTilingData;
  pipe = pipe_in;
  coreNum = TilingData->s1s2BNGS1S2BaseParams.coreNum;

  // shape info
  b = TilingData->s1s2BNGS1S2BaseParams.b;
  n2 = TilingData->s1s2BNGS1S2BaseParams.n2;
  g = TilingData->s1s2BNGS1S2BaseParams.g;
  s1 = TilingData->s1s2BNGS1S2BaseParams.s1;
  s2 = TilingData->s1s2BNGS1S2BaseParams.s2;
  d = TilingData->s1s2BNGS1S2BaseParams.d;
  attenMaksDimS2 = TilingData->s1s2BNGS1S2BaseParams.attenMaskS2Size;

  s1Token = TilingData->s1s2BNGS1S2BaseParams.s1Token;
  s2Token = TilingData->s1s2BNGS1S2BaseParams.s2Token;
  sparseMode = TilingData->s1s2BNGS1S2BaseParams.sparseMode;
  isSparse = false;
  if (TilingData->s1s2BNGS1S2BaseParams.isSparse == 1) {
    isSparse = true;
  }

  // split info
  s1Outer = TilingData->s1s2BNGS1S2SplitCoreParams.s1Outer;
  s1Inner = TilingData->s1s2BNGS1S2SplitCoreParams.s1Inner;
  s1Tail = TilingData->s1s2BNGS1S2SplitCoreParams.s1Tail;
  s1TailAlign = (s1Tail + 15) / 16 * 16;
  s2Outer = TilingData->s1s2BNGS1S2SplitCoreParams.s2Outer;
  s2CvRatio = TilingData->s1s2BNGS1S2SplitCoreParams.s2CvRatio;
  s2Inner = TilingData->s1s2BNGS1S2SplitCoreParams.s2Inner;
  s2Tail = TilingData->s1s2BNGS1S2SplitCoreParams.s2Tail;
  s2TailAlign = (s2Tail + 15) / 16 * 16;
  if ((s2TailAlign - s2Tail) >= cal_block_num) {
    isPadABlock = true;
  }else {
    isPadABlock = false;
  }

  sfmgdOuter = TilingData->s1s2BNGS1S2SplitCoreParams.sfmgdOuter;
  sfmgdInner = TilingData->s1s2BNGS1S2SplitCoreParams.sfmgdFactor;
  sfmgdTail = TilingData->s1s2BNGS1S2SplitCoreParams.sfmgdTail;
  sfmgdTailAlign = (sfmgdTail + 15) / 16 * 16;
  mm1CalRatio = TilingData->s1s2BNGS1S2SplitCoreParams.mm1CalRatio;

  blockOuter = TilingData->s1s2BNGS1S2SplitCoreParams.blockOuter;
  tmpBufferSize = TilingData->s1s2BNGS1S2SplitCoreParams.tmpBufferSize;
  baseMN = TilingData->s1s2BNGS1S2SplitCoreParams.baseMN;

  for (uint32_t i = 0; i < 50; i++) {
    blockStarts[i] = TilingData->s1s2BNGS1S2BlockNumList.blockStarts[i];
    blockEnds[i] = TilingData->s1s2BNGS1S2BlockNumList.blockEnds[i];
  }

  for (uint32_t i = 0; i < 64; i++) {
    prefixN[i] = TilingData->s1s2BNGS1S2BaseParams.prefixN[i];
  }

  isDrop = false;
  if (TilingData->s1s2BNGS1S2BaseParams.keepProb < 1) {
    isDrop = true;
  }
  keepProb = TilingData->s1s2BNGS1S2BaseParams.keepProb;
  layout = TilingData->s1s2BNGS1S2BaseParams.layout;

  // idx info
  n2gs1os2o = n2 * g * s1Outer * s2Outer;
  gs1os2o = g * s1Outer * s2Outer;
  s1os2o = s1Outer * s2Outer;
  s2o = s2Outer;

  uint64_t qPreBlockTotal = TilingData->preTilingData.qPreBlockTotal;
  uint64_t kvPreBlockTotal = TilingData->preTilingData.kvPreBlockTotal;
  uint64_t maskPreBlockTotal = TilingData->preTilingData.maskPreBlockTotal;
  uint64_t qPostBlockTotal = TilingData->postTilingData.qPostBlockTotal;
  uint64_t kvPostBlockTotal = TilingData->postTilingData.kvPostBlockTotal;

  // init workspace address
  syncGlobal.SetGlobalBuffer((__gm__ int32_t*)workspace);
  workspaceGm.SetGlobalBuffer((__gm__ uint8_t*)workspace);
  uint64_t workspaceOffsets = SYNC_GLOBAL_WORKSPACE_SIZE;
  dqWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace + workspaceOffsets / sizeof(T2));
  workspaceOffsets =
      (workspaceOffsets + qPostBlockTotal * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
  dkWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace + workspaceOffsets / sizeof(T2));
  workspaceOffsets =
      (workspaceOffsets + kvPostBlockTotal * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
  dvWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace + workspaceOffsets / sizeof(T2));
  workspaceOffsets =
      (workspaceOffsets + kvPostBlockTotal * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
  
  if (isDrop && s2 % 8 != 0) {
    maskWorkSpaceGm.SetGlobalBuffer((__gm__ uint8_t*)workspace + workspaceOffsets);
    workspaceOffsets =
        (workspaceOffsets + maskPreBlockTotal * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
  }

  // matmal1 and matmal2 workspace size
  matmalWorkspaceSize = s2CvRatio * baseMN * sizeof(float);
  mm1.SetWorkspace((__gm__ uint8_t*)workspaceGm[workspaceOffsets + cBlockIdx * matmalWorkspaceSize].GetPhyAddr(),
                   matmalWorkspaceSize);

  mm2.SetWorkspace((__gm__ uint8_t*)this
                       ->workspaceGm[workspaceOffsets + coreNum * matmalWorkspaceSize + cBlockIdx * matmalWorkspaceSize]
                       .GetPhyAddr(),
                   matmalWorkspaceSize);

  // drop workspace offset
  workspaceOffsets = (workspaceOffsets + coreNum * s2CvRatio * baseMN * sizeof(float) * INPUT_NUMS
                      + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
  dropWorkSpaceGm.SetGlobalBuffer((__gm__ T1*)workspace + workspaceOffsets / sizeof(T1));

  // mul workspace offset
  workspaceOffsets = (workspaceOffsets + coreNum * s2CvRatio * baseMN * sizeof(half) * 2
                      + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
  mulWorkSpaceGm.SetGlobalBuffer((__gm__ T1*)workspace + workspaceOffsets / sizeof(T1));

  InitOutput<int32_t>(syncGlobal[GetBlockIdx() * 8], 8, 0);
  baseUbSize = baseMN * calDtypeBytes;

  pipe->InitBuffer(vecInQue, 1, baseUbSize);
  pipe->InitBuffer(vecOutQue, 1, tmpBufferSize);
  pipe->InitBuffer(vecClc1, 1, baseUbSize);
  pipe->InitBuffer(vecClc2, 1, baseUbSize);
  pipe->InitBuffer(vecClc3, baseUbSize);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2<T1, T2, IS_BF16>::CopyInSoftMaxGrad(
    LocalTensor<T1>& dstTensor, uint64_t softmaxGradFrontOffset, uint32_t s1Extend, uint32_t dExtend,
    uint32_t dExtendAlign) {
  bool isBasicBlock = (s1Extend % 8 == 0) && (dExtend % 64 == 0);

  uint64_t transpse_stride = 0;
  if (layout == BNGSD) {
    transpse_stride = (d - dExtend) * sizeof(T1);
  } else if (layout == SBNGD) {
    transpse_stride = (b * n2 * g * d - dExtend) * sizeof(T1);
  } else if (layout == BSNGD) {
    transpse_stride = (n2 * g * d - dExtend) * sizeof(T1);
  }

  if (transpse_stride <= 65535) {
    // GM stride and length use org
    DataCopyPad(dstTensor, forwardResGm[softmaxGradFrontOffset],
                {static_cast<uint16_t>(s1Extend), static_cast<uint16_t>(dExtend * sizeof(T1)),
                 static_cast<uint16_t>(transpse_stride), 0},
                {true, 0, static_cast<uint8_t>((dExtendAlign - dExtend)), 0});
    DataCopyPad(dstTensor[s1Extend * dExtendAlign], dxGm[softmaxGradFrontOffset],
                {static_cast<uint16_t>(s1Extend), static_cast<uint16_t>(dExtend * sizeof(T1)),
                 static_cast<uint16_t>(transpse_stride), 0},
                {true, 0, static_cast<uint8_t>((dExtendAlign - dExtend)), 0});
  } else {
    uint64_t instr_stride = 0;
    if (layout == BNGSD) {
      instr_stride = d;
    } else if (layout == SBNGD) {
      instr_stride = b * n2 * g * d;
    } else if (layout == BSNGD) {
      instr_stride = n2 * g * d;
    }
    for (uint32_t i = 0; i < s1Extend; i++) {
      DataCopyPad(dstTensor[i * dExtendAlign], forwardResGm[softmaxGradFrontOffset],
                  {static_cast<uint16_t>(1), static_cast<uint16_t>(dExtend * sizeof(T1)), 0, 0},
                  {true, 0, static_cast<uint8_t>((dExtendAlign - dExtend)), 0});
      DataCopyPad(dstTensor[s1Extend * dExtendAlign + i * dExtendAlign], dxGm[softmaxGradFrontOffset],
                  {static_cast<uint16_t>(1), static_cast<uint16_t>(dExtend * sizeof(T1)), 0, 0},
                  {true, 0, static_cast<uint8_t>((dExtendAlign - dExtend)), 0});
      softmaxGradFrontOffset = softmaxGradFrontOffset + instr_stride;
    }
  }
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2<T1, T2, IS_BF16>::CalcSoftMaxGrad(
    LocalTensor<float>& sfmgClc3, uint64_t aTensorOffset, uint32_t s1Extend) {
  LocalTensor<float> sfmgClc1 = vecClc1.AllocTensor<float>();
  LocalTensor<float> sfmgClc2 = vecClc2.AllocTensor<float>();
  Duplicate<float>(sfmgClc3, 0.0, s1Extend * 32 / sizeof(float));

  for (uint32_t sfmgdIdx = 0; sfmgdIdx < sfmgdOuter; sfmgdIdx++) {
    LocalTensor<T1> vecInBuffer = vecInQue.AllocTensor<T1>();
    LocalTensor<uint8_t> vecOutBuffer = vecOutQue.AllocTensor<uint8_t>();
    uint64_t softmaxGradFrontOffset = aTensorOffset + sfmgdIdx * sfmgdInner;
    uint32_t dExtend = (sfmgdIdx == sfmgdOuter - 1) ? sfmgdTail : sfmgdInner;
    uint32_t dExtendAlign = (sfmgdIdx == sfmgdOuter - 1) ? sfmgdTailAlign : sfmgdInner;
    bool isBasicBlock = (s1Extend % 8 == 0) && (dExtend % 64 == 0);

    CopyInSoftMaxGrad(vecInBuffer, softmaxGradFrontOffset, s1Extend, dExtend, dExtendAlign);
    vecInQue.EnQue(vecInBuffer);
    vecInQue.DeQue<T1>();

    Cast(sfmgClc1, vecInBuffer, RoundMode::CAST_NONE, s1Extend * dExtendAlign);
    Cast(sfmgClc2, vecInBuffer[s1Extend * dExtendAlign], RoundMode::CAST_NONE, s1Extend * dExtendAlign);
    pipe_barrier(PIPE_V);
    uint32_t shapeArray[2];
    shapeArray[0] = s1Extend;
    shapeArray[1] = dExtendAlign;
    sfmgClc1.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    sfmgClc2.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    uint32_t shapeArray1[2];
    shapeArray1[0] = s1Extend;
    shapeArray1[1] = 32 / sizeof(float);
    sfmgClc3.SetShapeInfo(ShapeInfo(2, shapeArray1, DataFormat::ND));

    if (isBasicBlock) {
      SoftmaxGradFront<float, true>(sfmgClc3[s1Extend * 32 / sizeof(float)], sfmgClc1, sfmgClc2, vecOutBuffer,
                                    TilingData->softmaxGradTilingData);
    } else {
      SoftmaxGradFront<float, false>(sfmgClc3[s1Extend * 32 / sizeof(float)], sfmgClc1, sfmgClc2, vecOutBuffer,
                                     TilingData->softmaxGradTilingData);
    }
    pipe_barrier(PIPE_ALL);
    Add(sfmgClc3, sfmgClc3[s1Extend * 32 / sizeof(float)], sfmgClc3, s1Extend * 32 / sizeof(float));
    vecOutQue.FreeTensor(vecOutBuffer);
    vecInQue.FreeTensor(vecInBuffer);
  }
  vecClc1.FreeTensor(sfmgClc1);
  vecClc2.FreeTensor(sfmgClc2);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2<T1, T2, IS_BF16>::CalcDropoutBool(
    LocalTensor<T2>& dstTensor, uint64_t maskOffset, uint32_t s1Extend, uint32_t s2Extend) {
  LocalTensor<uint8_t> vecInBoolBuffer = vecInQue.AllocTensor<uint8_t>();
  DataCopyPad(
      vecInBoolBuffer, maskWorkSpaceGm[maskOffset],
      {static_cast<uint16_t>(s1Extend), static_cast<uint16_t>(s2Extend), static_cast<uint16_t>(s2 - s2Extend), 0},
      {false, 0, 0, 0});
  vecInQue.EnQue(vecInBoolBuffer);
  vecInQue.DeQue<uint8_t>();

  uint32_t shapeArray[2];
  shapeArray[0] = s1Extend;
  shapeArray[1] = s2Extend;
  vecInBoolBuffer.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));

  DropOutShapeInfo info;
  info.firstAxis = s1Inner;
  info.srcLastAxis = s2Extend;
  info.maskLastAxis = s2Extend;

  LocalTensor<uint8_t> vecOutBuffer = vecOutQue.AllocTensor<uint8_t>();
  DropOut<T2, true>(dstTensor, dstTensor, vecInBoolBuffer, vecOutBuffer, keepProb, info);
  vecOutQue.FreeTensor(vecOutBuffer);
  vecInQue.FreeTensor(vecInBoolBuffer);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2<T1, T2, IS_BF16>::CalcDropoutBit(
    LocalTensor<T2>& dstTensor, uint64_t maskOffset, uint32_t s1Extend, uint32_t s2Extend) {
  LocalTensor<uint8_t> vecInBitBuffer = vecInQue.AllocTensor<uint8_t>();
  DataCopyPad(vecInBitBuffer, dropMaskGm[maskOffset],
              {static_cast<uint16_t>(s1Extend), static_cast<uint16_t>(s2Extend / 8),
               static_cast<uint16_t>((s2 - s2Extend) / 8), 0},
              {false, 0, 0, 0});
  vecInQue.EnQue(vecInBitBuffer);
  vecInQue.DeQue<uint8_t>();

  uint32_t shapeArray[2];
  shapeArray[0] = s1Extend;
  shapeArray[1] = s2Extend;
  vecInBitBuffer.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));

  DropOutShapeInfo info;
  info.firstAxis = s1Extend;
  info.srcLastAxis = s2Extend;
  info.maskLastAxis = (s2Extend / 8 + 31) / 32 * 32;

  LocalTensor<uint8_t> vecOutBuffer = vecOutQue.AllocTensor<uint8_t>();
  DropOut(dstTensor, dstTensor, vecInBitBuffer, vecOutBuffer, keepProb, info);
  vecOutQue.FreeTensor(vecOutBuffer);
  vecInQue.FreeTensor(vecInBitBuffer);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2<T1, T2, IS_BF16>::CopyInSoftMax(
    LocalTensor<float>& dstTensor, uint64_t softMaxOffset, uint32_t s1Extend) {
  DataCopyPad(dstTensor, softmaxSumGm[softMaxOffset], {1, static_cast<uint16_t>(s1Extend * 32), 0, 0},
              {false, 0, 0, 0});
  DataCopyPad(dstTensor[s1Extend * 32 / sizeof(float)], softmaxMaxGm[softMaxOffset],
              {1, static_cast<uint16_t>(s1Extend * 32), 0, 0}, {false, 0, 0, 0});
  vecInQue.EnQue(dstTensor);
  vecInQue.DeQue<float>();
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2<T1, T2, IS_BF16>::CalcSoftMax(
    LocalTensor<T2>& dstTensor, LocalTensor<float>& srcTensor, uint32_t s1Extend, const SoftMaxTiling& tiling) {
  LocalTensor<uint8_t> vecOutBuffer = vecOutQue.AllocTensor<uint8_t>();

  uint32_t shapeArray1[2];
  shapeArray1[0] = s1Extend;
  shapeArray1[1] = s2ExtendAlign;
  dstTensor.SetShapeInfo(ShapeInfo(2, shapeArray1, DataFormat::ND));

  bool isBasicBlock = (s1Inner % 8 == 0) && (s2ExtendAlign % 64 == 0);
  if (isBasicBlock) {
    SimpleSoftMax<T2, true, true>(dstTensor, srcTensor, srcTensor[s1Extend * 32 / sizeof(float)], dstTensor,
                                  vecOutBuffer, tiling);
  } else {
    SimpleSoftMax<T2, true, false>(dstTensor, srcTensor, srcTensor[s1Extend * 32 / sizeof(float)], dstTensor,
                                   vecOutBuffer, tiling);
  }
  vecOutQue.FreeTensor(vecOutBuffer);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2<T1, T2, IS_BF16>::CopyInPseShift(
    LocalTensor<T2>& dstTensor, LocalTensor<T1> vecInTensor, uint64_t pseOffset, uint32_t s2Extend) {
  if constexpr (IsSameType<T1, T2>::value) {
    DataCopyPad(
        vecInTensor, pseGm[pseOffset],
        {1, static_cast<uint16_t>(s2Extend * sizeof(T1)), static_cast<uint16_t>((s2 - s2Extend) * sizeof(T1)), 0},
        {false, 0, 0, 0});
    vecInQue.EnQue(vecInTensor);
    vecInQue.DeQue<T1>();
    DataCopy(dstTensor, vecInTensor, {1, static_cast<uint16_t>(s2Extend), 0, 0});
  } else {
    uint16_t repeatTimes = 1;
    if (TilingData->s1s2BNGS1S2BaseParams.pseShapeType == 1) {
      repeatTimes = static_cast<uint16_t>(s1Extend);
    }
    DataCopyPad(vecInTensor, pseGm[pseOffset],
                {repeatTimes, static_cast<uint16_t>(s2Extend * sizeof(T1)),
                 static_cast<uint16_t>((s2 - s2Extend) * sizeof(T1)), 0},
                {false, 0, 0, 0});
    vecInQue.EnQue(vecInTensor);
    vecInQue.DeQue<T1>();
    Cast(dstTensor, vecInTensor, RoundMode::CAST_NONE, repeatTimes * s2Extend);
  }
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2<T1, T2, IS_BF16>::CalcPseShift(LocalTensor<T2>& dstTensor,
                                                                                            LocalTensor<T2>& srcTensor,
                                                                                            uint32_t s1Extend,
                                                                                            uint32_t s2ExtendAlign) {
  if (TilingData->s1s2BNGS1S2BaseParams.pseShapeType == 0) {
    for (uint32_t subIdx = 0; subIdx < s1Extend; subIdx++) {
      pipe_barrier(PIPE_V);
      Add(dstTensor[subIdx * s2ExtendAlign], srcTensor, dstTensor[subIdx * s2ExtendAlign], s2ExtendAlign);
    }
  } else if (TilingData->s1s2BNGS1S2BaseParams.pseShapeType == 1) {
    Add(dstTensor, srcTensor, dstTensor, s1Extend * s2ExtendAlign);
  }
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2<T1, T2, IS_BF16>::CopyInAttenMaskT1(
    LocalTensor<T1>& dstTensor, uint64_t attenMaskOffset, uint32_t s1Extend, uint32_t s2Extend) {
  DataCopyPad(dstTensor, attenMaskGm[attenMaskOffset],
              {static_cast<uint16_t>(s1Extend), static_cast<uint16_t>(s2Extend * sizeof(T1)),
               static_cast<uint16_t>((attenMaksDimS2 - s2Extend) * sizeof(T1)), 0},
              {false, 0, 0, 0});
  vecInQue.EnQue(dstTensor);
  vecInQue.DeQue<T1>();
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2<T1, T2, IS_BF16>::CopyInAttenMaskBool(
    LocalTensor<uint8_t>& dstTensor, uint64_t attenMaskOffset, uint32_t s1Extend, uint32_t s2Extend) {
  DataCopyPad(dstTensor, attenMaskU8Gm[attenMaskOffset],
              {static_cast<uint16_t>(s1Extend), static_cast<uint16_t>(s2Extend * sizeof(uint8_t)),
               static_cast<uint16_t>((attenMaksDimS2 - s2Extend) * sizeof(uint8_t)), 0},
              {false, 0, 0, 0});
  vecInQue.EnQue(dstTensor);
  vecInQue.DeQue<uint8_t>();
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2<T1, T2, IS_BF16>::CalcAttenMaskT1(
    LocalTensor<T2>& dstTensor, LocalTensor<T1> srcTensor, uint32_t s1Extend, uint32_t s2Extend) {
  if constexpr (IsSameType<T1, T2>::value) {
    Muls(srcTensor, srcTensor, mulsValue, s1Extend * s2Extend);  // -10000
    pipe_barrier(PIPE_V);
    Add(dstTensor, srcTensor, dstTensor, s1Extend * s2Extend);
  } else {
    LocalTensor<T2> vecOutBuffer = vecOutQue.AllocTensor<T2>();
    Cast(vecOutBuffer, srcTensor, RoundMode::CAST_NONE, s1Extend * s2Extend);
    pipe_barrier(PIPE_V);

    Muls(vecOutBuffer, vecOutBuffer, mulsValue, s1Extend * s2Extend);  // -10000
    pipe_barrier(PIPE_V);
    Add(dstTensor, vecOutBuffer, dstTensor, s1Extend * s2Extend);
    vecOutQue.FreeTensor(vecOutBuffer);
  }
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2<T1, T2, IS_BF16>::CalcAttenMaskBool(
    LocalTensor<T2>& dstTensor, LocalTensor<uint8_t> srcTensor) {
  LocalTensor<uint8_t> tmpUbBuffer = vecOutQue.AllocTensor<uint8_t>();

  T2 scalar;
  if constexpr (IsSameType<T2, float>::value) {
    uint32_t tmp = 0xFF7F0000;
    scalar = *((float*)&tmp);
  } else {
    uint16_t tmp = 0xFBF0;
    scalar = *((half*)&tmp);
  }

  SelectWithBytesMaskShapeInfo info;
  info.firstAxis = s1Extend;
  info.srcLastAxis = s2ExtendAlign;
  info.maskLastAxis = (s2Extend * sizeof(uint8_t) + 31) / 32 * 32 / sizeof(uint8_t);
  dstTensor.SetSize(info.firstAxis * info.srcLastAxis);
  srcTensor.SetSize(info.firstAxis * info.maskLastAxis);
  SelectWithBytesMask(dstTensor, dstTensor, scalar, srcTensor, tmpUbBuffer, info);
  vecOutQue.FreeTensor(tmpUbBuffer);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2<T1, T2, IS_BF16>::Process() {
  uint32_t preIndex = blockStarts[cBlockIdx];
  if (blockEnds[cBlockIdx] == 0) {
    return;
  }
  for (uint32_t blockInnerIdx = blockStarts[cBlockIdx] + 1; blockInnerIdx < blockEnds[cBlockIdx]; blockInnerIdx++) {
    if (IsSparse(blockInnerIdx)) {
      uint32_t nextIndex = blockInnerIdx;
      Compute(preIndex, nextIndex);
      preIndex = nextIndex;
    }
  }
  Compute(preIndex, 0u);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline bool FlashAttentionScoreGradUs1s2Bbn2gs1s2<T1, T2, IS_BF16>::CheckIsValidBlock(uint32_t baseIdx) {
  uint32_t gDimTail = baseIdx % s1os2o;
  uint32_t s1oDimIdx = gDimTail / s2o;
  uint32_t s2oCvDimIdx = gDimTail % s2o;

  int64_t cvS2Inner = static_cast<int64_t>(s2CvRatio) * s2Inner;
  int64_t s2IgnoredEndLen = static_cast<int64_t>(s1) - static_cast<int64_t>(s1Inner * (s1oDimIdx + 1));
  int64_t s2EndLen = 0;
  if (static_cast<int64_t>(s2) > s2IgnoredEndLen) {
    s2EndLen = static_cast<int64_t>(s2) - s2IgnoredEndLen;
  } else {
    s2EndLen = 0;
  }

  if (sparseMode == 5) {
    uint32_t curBIdx = baseIdx / n2gs1os2o;
    s2EndLen = s2EndLen > static_cast<int64_t>(prefixN[curBIdx]) ? s2EndLen : static_cast<int64_t>(prefixN[curBIdx]);
    s2EndLen = min(s2EndLen, static_cast<int64_t>(s2));
  }

  uint32_t currS2Idx =(s2EndLen + cvS2Inner - 1) / cvS2Inner;
  return s2oCvDimIdx < currS2Idx ? true : false;
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline bool FlashAttentionScoreGradUs1s2Bbn2gs1s2<T1, T2, IS_BF16>::IsSparse(uint32_t baseIdx) {
  if (!isSparse) {
    return true;
  }
  if (sparseMode == 3 || sparseMode == 5) {
    return CheckIsValidBlock(baseIdx);
  } else {
    uint32_t gDimTail = baseIdx % s1os2o;
    uint32_t s1oDimIdx = gDimTail / s2o;
    uint32_t s2oCvDimIdx = gDimTail % s2o;

    uint32_t s2IdxLeft = s2oCvDimIdx * s2Inner * s2CvRatio;
    uint32_t s2IdxRight = (s2oCvDimIdx + 1) * s2Inner * s2CvRatio < s2 ? (s2oCvDimIdx + 1) * s2Inner * s2CvRatio : s2;
    uint32_t s2SparseLeft = int64_t(s1Inner * s1oDimIdx) - int64_t(s1Token) < 0 ? 0 : s1Inner * s1oDimIdx - s1Token;
    uint32_t s2SparseRight = s1Inner * (s1oDimIdx + 1) + s2Token < s2 ? s1Inner * (s1oDimIdx + 1) + s2Token : s2;
    return s2IdxLeft < s2SparseRight && s2IdxRight > s2SparseLeft;
  }
}


template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2<T1, T2, IS_BF16>::CalcAttenMaskOffset(
        uint64_t& attenMaskOffset, const int64_t delta) {
    if (delta == 0) {
        attenMaskOffset = 0;
    } else if (delta < 0) {
        if (-delta > s1Inner) {
            attenMaskOffset = s1Inner;
        } else {
            attenMaskOffset = -delta;
        }
    } else {
        if (delta > s2Inner) {
            attenMaskOffset = s2Inner * attenMaksDimS2;
        } else {
            attenMaskOffset = delta * attenMaksDimS2;
        }
    }
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2<T1, T2, IS_BF16>::CalcCausalAttenMaskOffset(
        uint64_t& attenMaskOffset, const int64_t delta) {
    CalcAttenMaskOffset(attenMaskOffset, delta);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2<T1, T2, IS_BF16>::InitIndex(uint32_t index) {
  baseIdx = index;
  bDimIdx = baseIdx / n2gs1os2o;
  bDimTail = baseIdx % n2gs1os2o;
  n2DimIdx = bDimTail / gs1os2o;
  n2DimTail = bDimTail % gs1os2o;
  gDimIdx = n2DimTail / s1os2o;
  gDimTail = n2DimTail % s1os2o;
  s1oDimIdx = gDimTail / s2o;
  s2oCvDimIdx = gDimTail % s2o;
  s1Extend = (s1oDimIdx == s1Outer - 1) ? s1Tail : s1Inner;
  s1ExtendAlign = (s1oDimIdx == s1Outer - 1) ? s1TailAlign : s1Inner;
  s2CvExtend = s2CvRatio;
  s2Extend = (s2oCvDimIdx == s2Outer - 1) ? s2Tail : s2Inner;
  s2ExtendAlign = (s2oCvDimIdx == s2Outer - 1) ? s2TailAlign : s2Inner;
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2<T1, T2, IS_BF16>::Compute(uint32_t preIndex,
                                                                                       uint32_t nextIndex) {
  pingpongIdx = 1 - pingpongIdx;
  if (isStart == 1) {
    InitIndex(preIndex);
  }
  uint64_t aTensorOffsetCv = 0;
  uint64_t bTensorOffsetCv = 0;
  if (layout == BNGSD) {
    aTensorOffsetCv = (((bDimIdx * n2 + n2DimIdx) * g + gDimIdx) * s1 + s1oDimIdx * s1Inner) * d;
    bTensorOffsetCv = ((bDimIdx * n2 + n2DimIdx) * s2 + s2oCvDimIdx * s2CvRatio * s2Inner) * d;
  } else if (layout == SBNGD) {
    aTensorOffsetCv = ((((s1oDimIdx * s1Inner) * b + bDimIdx) * n2 + n2DimIdx) * g + gDimIdx) * d;
    bTensorOffsetCv = (((s2oCvDimIdx * s2CvRatio * s2Inner) * b + bDimIdx) * n2 + n2DimIdx) * d;
  } else if (layout == BSNGD) {
    aTensorOffsetCv = (((bDimIdx * s1 + s1oDimIdx * s1Inner) * n2 + n2DimIdx) * g + gDimIdx) * d;
    bTensorOffsetCv = ((bDimIdx * s2 + s2oCvDimIdx * s2CvRatio * s2Inner) * n2 + n2DimIdx) * d;
  }

  if (isStart == 1) {
    mm1.SetTail(s1Extend, s2Extend * s2CvRatio, d);  // M N K
    mm1.SetTensorA(dxGm[aTensorOffsetCv]);
    mm1.SetTensorB(valueGm[bTensorOffsetCv], true);
    mm1.template Iterate<false>();

    mm2.SetTail(s1Extend, s2Extend * s2CvRatio, d);  // M N K
    mm2.SetTensorA(queryGm[aTensorOffsetCv]);
    mm2.SetTensorB(keyGm[bTensorOffsetCv], true);
    mm2.template Iterate<false>();

    isStart = 0;
  }

  uint64_t subMaskCout = (cal_repeat_num < s2ExtendAlign) ? cal_repeat_num : s2ExtendAlign;
  for (uint32_t curS2Idx = 0; curS2Idx < s2CvExtend; curS2Idx++) {
    s2oDimIdx = s2oCvDimIdx * s2CvRatio + curS2Idx;
    uint64_t aTensorOffset = 0;
    uint64_t bTensorOffset = 0;
    if (layout == BNGSD) {
      aTensorOffset = (((bDimIdx * n2 + n2DimIdx) * g + gDimIdx) * s1 + s1oDimIdx * s1Inner) * d;
      bTensorOffset = ((bDimIdx * n2 + n2DimIdx) * s2 + s2oDimIdx * s2Inner) * d;
    } else if (layout == SBNGD) {
      aTensorOffset = ((((s1oDimIdx * s1Inner) * b + bDimIdx) * n2 + n2DimIdx) * g + gDimIdx) * d;
      bTensorOffset = (((s2oDimIdx * s2Inner) * b + bDimIdx) * n2 + n2DimIdx) * d;
    } else if (layout == BSNGD) {
      aTensorOffset = (((bDimIdx * s1 + s1oDimIdx * s1Inner) * n2 + n2DimIdx) * g + gDimIdx) * d;
      bTensorOffset = ((bDimIdx * s2 + s2oDimIdx * s2Inner) * n2 + n2DimIdx) * d;
    }

    ///////////////////////////////////////////////////////////////
    // SoftmaxGradFront
    ///////////////////////////////////////////////////////////////
    LocalTensor<float> sfmgClc3 = vecClc3.Get<float>();
    if (curS2Idx == 0) {
      CalcSoftMaxGrad(sfmgClc3, aTensorOffset, s1Extend);
    }

    if (curS2Idx % mm1CalRatio == 0) {
      matmalResultBuffer1 = mm1.template GetTensorC<false>(false, false);
      mm1.End();
    }
    LocalTensor<T2> vecClc1Buffer = vecClc1.AllocTensor<T2>();

    uint8_t pad_num = 0;
    if (isPadABlock && (s2Extend == s2Tail) && (s2Tail != s2Inner)) {
      pad_num = 8;
    }

    DataCopyPad(vecClc1Buffer, matmalResultBuffer1[(curS2Idx % mm1CalRatio) * s2ExtendAlign],
                {static_cast<uint16_t>(s1Extend), static_cast<uint16_t>(s2Extend * sizeof(float)),
                 static_cast<uint16_t>((mm1CalRatio - 1) * s2ExtendAlign * sizeof(float)), 0},
                {true, 0, pad_num, 0});
    vecClc1.EnQue(vecClc1Buffer);
    vecClc1.DeQue<T2>();


    ///////////////////////////////////////////////////////////////
    // dropout
    ///////////////////////////////////////////////////////////////
    if (isDrop) {
      if (s2 % 8 == 0) {
        uint64_t maskOffset =
            ((((bDimIdx * n2 + n2DimIdx) * g + gDimIdx) * s1 + s1oDimIdx * s1Inner) * s2 + s2oDimIdx * s2Inner) / 8;
        CalcDropoutBit(vecClc1Buffer, maskOffset, s1Extend, s2Extend);
      } else {
        uint64_t maskOffset =
            (((bDimIdx * n2 + n2DimIdx) * g + gDimIdx) * s1 + s1oDimIdx * s1Inner) * s2 + s2oDimIdx * s2Inner;
        CalcDropoutBool(vecClc1Buffer, maskOffset, s1Extend, s2Extend);
      }
    }
    pipe_barrier(PIPE_V);

    ///////////////////////////////////////////////////////////////
    // sub to improve
    ///////////////////////////////////////////////////////////////
    uint32_t sub_block_cout = (s2ExtendAlign + cal_repeat_num - 1) / cal_repeat_num;
    for (uint32_t subIdx = 0; subIdx < sub_block_cout; subIdx++) {
      Sub(vecClc1Buffer[subIdx * cal_repeat_num], vecClc1Buffer[subIdx * cal_repeat_num], sfmgClc3, subMaskCout,
          s1Extend,
          {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0,
           static_cast<uint8_t>(s2ExtendAlign / 8), static_cast<uint8_t>(s2ExtendAlign / 8), 1});
    }

    LocalTensor<float> vecClc2Buffer = vecClc2.AllocTensor<float>();
    if (curS2Idx % mm1CalRatio == 0) {
      matmalResultBuffer2 = mm2.template GetTensorC<false>(false, false);
      mm2.End();
    }

    DataCopyPad(vecClc2Buffer, matmalResultBuffer2[(curS2Idx % mm1CalRatio) * s2ExtendAlign],
                {static_cast<uint16_t>(s1Extend), static_cast<uint16_t>(s2Extend * sizeof(float)),
                 static_cast<uint16_t>((mm1CalRatio - 1) * s2ExtendAlign * sizeof(float)), 0},
                {true, 0, pad_num, 0});
    vecClc2.EnQue(vecClc2Buffer);

    vecClc2.DeQue<float>();

    ///////////////////////////////////////////////////////////////
    // pse + muls
    ///////////////////////////////////////////////////////////////
    // pse shape  0--BN2G1S2    1--BN2GS1S2
    if (TilingData->s1s2BNGS1S2BaseParams.pseOptional) {
      uint64_t pseOffset = 0;
      if (TilingData->s1s2BNGS1S2BaseParams.pseShapeType == 0) {
        pseOffset = ((bDimIdx * n2 + n2DimIdx) * g + gDimIdx) * s2 + s2oDimIdx * s2Inner;
      } else {
        pseOffset = (((bDimIdx * n2 + n2DimIdx) * g + gDimIdx) * s1 + s1oDimIdx * s1Inner) * s2 + s2oDimIdx * s2Inner;
      }
      LocalTensor<T2> vecClc3Buffer = vecOutQue.AllocTensor<T2>();
      LocalTensor<T1> vecInBuffer1 = vecInQue.AllocTensor<T1>();
      CopyInPseShift(vecClc3Buffer, vecInBuffer1, pseOffset, s2Extend);
      CalcPseShift(vecClc2Buffer, vecClc3Buffer, s1Extend, s2ExtendAlign);
      vecInQue.FreeTensor(vecInBuffer1);
      vecOutQue.FreeTensor(vecClc3Buffer);
    }

    Muls(vecClc2Buffer, vecClc2Buffer, (T2)(TilingData->s1s2BNGS1S2BaseParams.scaleValue), s1Extend * s2ExtendAlign);

    ///////////////////////////////////////////////////////////////
    // attenMask
    ///////////////////////////////////////////////////////////////
    // attenMaskOffset     attenMaskShapeType  0--111S1S2        1--B11S1S2         2--BN2GS1S2
    if (TilingData->s1s2BNGS1S2BaseParams.attenMaskOptional) {
      uint64_t attenMaskOffset = 0;
      uint64_t compressMode = TilingData->s1s2BNGS1S2BaseParams.attenMaskCompressMode;
      int64_t causal_delta = (int64_t)(s1oDimIdx * s1Inner) - (int64_t)(s2oDimIdx * s2Inner);
      if (compressMode == 1) {
        // causal s1==s2
        CalcCausalAttenMaskOffset(attenMaskOffset, causal_delta);
      } else if (compressMode == 2) {
        // causal s1!=s2
        causal_delta = causal_delta - s1 + s2;
        CalcCausalAttenMaskOffset(attenMaskOffset, causal_delta);
      } else if (TilingData->s1s2BNGS1S2BaseParams.attenMaskShapeType == 0) {
        attenMaskOffset = s1oDimIdx * s1Inner * s2 + s2oDimIdx * s2Inner;
      } else if (TilingData->s1s2BNGS1S2BaseParams.attenMaskShapeType == 1) {
        attenMaskOffset = (bDimIdx * s1 + s1oDimIdx * s1Inner) * s2 + s2oDimIdx * s2Inner;
      } else {
        attenMaskOffset =
            (((bDimIdx * n2 + n2DimIdx) * g + gDimIdx) * s1 + s1oDimIdx * s1Inner) * s2 + s2oDimIdx * s2Inner;
      }

      if (TilingData->s1s2BNGS1S2BaseParams.attenMaskDtype == 0) {  // 0 表示输入数据类型与 qkv 一致
        LocalTensor<T1> attenMaskUb = vecInQue.AllocTensor<T1>();
        CopyInAttenMaskT1(attenMaskUb, attenMaskOffset, s1Extend, s2Extend);
        CalcAttenMaskT1(vecClc2Buffer, attenMaskUb, s1Extend, s2ExtendAlign);
        vecInQue.FreeTensor(attenMaskUb);
      } else {  //   uint8_t
        LocalTensor<uint8_t> attenMaskUb = vecInQue.AllocTensor<uint8_t>();
        CopyInAttenMaskBool(attenMaskUb, attenMaskOffset, s1Extend, s2Extend);
        CalcAttenMaskBool(vecClc2Buffer, attenMaskUb);
        vecInQue.FreeTensor(attenMaskUb);
      }
    }
    ///////////////////////////////////////////////////////////////
    // simpleSoftMax
    ///////////////////////////////////////////////////////////////
    uint64_t softMaxOffset = (((bDimIdx * n2 + n2DimIdx) * g + gDimIdx) * s1 + s1oDimIdx * s1Inner) * 32 / sizeof(float);
    LocalTensor<float> vecInBuffer3 = vecInQue.AllocTensor<float>();
    CopyInSoftMax(vecInBuffer3, softMaxOffset, s1Extend);
    CalcSoftMax(vecClc2Buffer, vecInBuffer3, s1Extend, TilingData->softmaxTilingData);
    vecInQue.FreeTensor(vecInBuffer3);

    ///////////////////////////////////////////////////////////////
    // mul
    ///////////////////////////////////////////////////////////////
    pipe_barrier(PIPE_V);
    LocalTensor<T1> vecOutBuffer = vecOutQue.AllocTensor<T1>();
    if constexpr (IsSameType<T1, T2>::value) {
      Mul(vecOutBuffer, vecClc1Buffer, vecClc2Buffer, s1Extend * s2ExtendAlign);
    } else {
      Mul(vecClc1Buffer, vecClc1Buffer, vecClc2Buffer, s1Extend * s2ExtendAlign);
      pipe_barrier(PIPE_V);
      Cast(vecOutBuffer, vecClc1Buffer, RoundMode::CAST_ROUND, s1Extend * s2ExtendAlign);
    }
    vecOutQue.EnQue(vecOutBuffer);
    vecOutQue.DeQue<T1>();
    DataCopyPad(mulWorkSpaceGm[pingpongIdx * coreNum * s2CvRatio * baseMN + cBlockIdx * s2CvRatio * baseMN
                               + curS2Idx * s2ExtendAlign], vecOutBuffer,
                {static_cast<uint16_t>(s1Extend), static_cast<uint16_t>(s2ExtendAlign * sizeof(T1)), 0,
                 static_cast<uint16_t>((s2CvRatio - 1) * s2ExtendAlign * sizeof(T1))});
    vecOutQue.FreeTensor(vecOutBuffer);
    vecClc1.FreeTensor(vecClc1Buffer);

    ///////////////////////////////////////////////////////////////
    // dropout
    ///////////////////////////////////////////////////////////////
    if (isDrop) {
      if (s2 % 8 == 0) {
        uint64_t maskOffset =
            ((((bDimIdx * n2 + n2DimIdx) * g + gDimIdx) * s1 + s1oDimIdx * s1Inner) * s2 + s2oDimIdx * s2Inner) / 8;
        CalcDropoutBit(vecClc2Buffer, maskOffset, s1Extend, s2Extend);
      } else {
        uint64_t maskOffset =
            (((bDimIdx * n2 + n2DimIdx) * g + gDimIdx) * s1 + s1oDimIdx * s1Inner) * s2 + s2oDimIdx * s2Inner;
        CalcDropoutBool(vecClc2Buffer, maskOffset, s1Extend, s2Extend);
      }
    }

    ///////////////////////////////////////////////////////////////
    // cast fp322bf16
    ///////////////////////////////////////////////////////////////
    LocalTensor<T1> vecOut1Buffer1 = vecOutQue.AllocTensor<T1>();
    if constexpr (IsSameType<T1, T2>::value) {
      Muls(vecOut1Buffer1, vecClc2Buffer, (T1)1.0, s1Extend * s2ExtendAlign);
    } else {
      Cast(vecOut1Buffer1, vecClc2Buffer, RoundMode::CAST_ROUND, s1Extend * s2ExtendAlign);
    }
    vecOutQue.EnQue(vecOut1Buffer1);
    vecOutQue.DeQue<T1>();
    DataCopyPad(dropWorkSpaceGm[pingpongIdx * coreNum * s2CvRatio * baseMN + cBlockIdx * s2CvRatio * baseMN
                                + curS2Idx * s2ExtendAlign], vecOut1Buffer1,
                {static_cast<uint16_t>(s1Extend), static_cast<uint16_t>(s2ExtendAlign * sizeof(T1)), 0,
                 static_cast<uint16_t>((s2CvRatio - 1) * s2ExtendAlign * sizeof(T1))});
    vecOutQue.FreeTensor(vecOut1Buffer1);
    vecClc2.FreeTensor(vecClc2Buffer);
    pipe_barrier(PIPE_ALL);
  }

  uint32_t preS1Extend = s1Extend;
  uint32_t preS2Extend = s2Extend;
  if (nextIndex != 0) {
    InitIndex(nextIndex);
    uint64_t aTensorOffsetCv1 = 0;
    uint64_t bTensorOffsetCv1 = 0;
    if (layout == BNGSD) {
      aTensorOffsetCv1 = (((bDimIdx * n2 + n2DimIdx) * g + gDimIdx) * s1 + s1oDimIdx * s1Inner) * d;
      bTensorOffsetCv1 = ((bDimIdx * n2 + n2DimIdx) * s2 + s2oCvDimIdx * s2CvRatio * s2Inner) * d;
    } else if (layout == SBNGD) {
      aTensorOffsetCv1 = ((((s1oDimIdx * s1Inner) * b + bDimIdx) * n2 + n2DimIdx) * g + gDimIdx) * d;
      bTensorOffsetCv1 = (((s2oCvDimIdx * s2CvRatio * s2Inner) * b + bDimIdx) * n2 + n2DimIdx) * d;
    } else if (layout == BSNGD) {
      aTensorOffsetCv1 = (((bDimIdx * s1 + s1oDimIdx * s1Inner) * n2 + n2DimIdx) * g + gDimIdx) * d;
      bTensorOffsetCv1 = ((bDimIdx * s2 + s2oCvDimIdx * s2CvRatio * s2Inner) * n2 + n2DimIdx) * d;
    }

    mm1.SetTail(s1Extend, s2Extend * s2CvRatio, d);
    mm1.SetTensorA(dxGm[aTensorOffsetCv1]);
    mm1.SetTensorB(valueGm[bTensorOffsetCv1], true);
    mm1.template Iterate<false>();

    mm2.SetTail(s1Extend, s2Extend * s2CvRatio, d);
    mm2.SetTensorA(queryGm[aTensorOffsetCv1]);
    mm2.SetTensorB(keyGm[bTensorOffsetCv1], true);
    mm2.template Iterate<false>();
  }

  ///////////////////////////////////////////////////////////////
  // Matmal4 dq
  ///////////////////////////////////////////////////////////////
  // left [B, N2, G, S1, s2] right [B, N2, 1, S2, D] output [B, N2, G, S1, D]
  if (layout == BNGSD) {
    mm4.SetOrgShape(s1, d, s2);
  } else if (layout == SBNGD) {
    mm4.SetOrgShape(s1, b * n2 * d, s2, s2, b * n2 * g * d);
  } else if (layout == BSNGD) {
    mm4.SetOrgShape(s1, n2 * d, s2, s2, n2 * g * d);
  }
  mm4.SetTail(preS1Extend, -1, s2CvRatio * preS2Extend);  // M N K
  mm4.SetTensorA(mulWorkSpaceGm[pingpongIdx * coreNum * s2CvRatio * baseMN + cBlockIdx * s2CvRatio * baseMN]);
  mm4.SetTensorB(keyGm[bTensorOffsetCv]);
  mm4.template IterateAll<false>(dqWorkSpaceGm[aTensorOffsetCv], true);
  mm4.End();

  ///////////////////////////////////////////////////////////////
  // Matmal4 dk
  ///////////////////////////////////////////////////////////////
  // left [B, N2, G, S1, S2] right [B, N2, 1, S1, D] output [B, N2, G, S2, D]
  if (layout == BNGSD) {
    mm3.SetOrgShape(s2, d, s1);
  } else if (layout == SBNGD) {
    mm3.SetOrgShape(s2, b * n2 * g * d, s1, s1, b * n2 * d);
  } else if (layout == BSNGD) {
    mm3.SetOrgShape(s2, n2 * g * d, s1, s1, n2 * d);
  }
  mm3.SetTail(s2CvRatio * preS2Extend, -1, preS1Extend);
  mm3.SetTensorA(mulWorkSpaceGm[pingpongIdx * coreNum * s2CvRatio * baseMN + cBlockIdx * s2CvRatio * baseMN], true);
  mm3.SetTensorB(queryGm[aTensorOffsetCv]);
  mm3.template IterateAll<false>(dkWorkSpaceGm[bTensorOffsetCv], true);
  mm3.End();

  ///////////////////////////////////////////////////////////////
  // Matmal5 dv
  ///////////////////////////////////////////////////////////////
  // left [B, N2, G, S1, S2] right [B, N2, G, S1, D] output [B, N2, 1, S2, D]
  mm3.SetTensorA(dropWorkSpaceGm[pingpongIdx * coreNum * s2CvRatio * baseMN + cBlockIdx * s2CvRatio * baseMN], true);
  mm3.SetTensorB(dxGm[aTensorOffsetCv]);
  if (nextIndex == 0) {
    mm3.template IterateAll<true>(dvWorkSpaceGm[bTensorOffsetCv], true);
  } else {
    mm3.template IterateAll<false>(dvWorkSpaceGm[bTensorOffsetCv], true);
  }
  mm3.End();
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn2gs1s2<T1, T2, IS_BF16>::SyncALLCores() {
  SyncAll();
}

#endif  // _FLASH_ATTENTION_SCORE_GRAD_US1S2_BBN2GS1S2_H_