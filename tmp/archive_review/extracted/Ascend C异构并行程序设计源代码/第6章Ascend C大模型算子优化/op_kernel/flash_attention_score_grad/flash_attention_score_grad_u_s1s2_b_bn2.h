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

#ifndef _FLASH_ATTENTION_SCORE_GRAD_U_S1S2_B_BN2_H_
#define _FLASH_ATTENTION_SCORE_GRAD_U_S1S2_B_BN2_H_

#include "lib/matmul_intf.h"
#include "kernel_operator.h"
using namespace matmul;

template <typename T1, typename T2>
class FlashAttentionScoreGradUs1s2Bbn {
 public:
  __aicore__ inline FlashAttentionScoreGradUs1s2Bbn(){};
  __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR pse_shift,
                              GM_ADDR drop_mask, GM_ADDR padding_mask, GM_ADDR atten_mask, GM_ADDR softmax_max,
                              GM_ADDR softmax_sum, GM_ADDR softmax_in, GM_ADDR attention_in, GM_ADDR dq, GM_ADDR dk,
                              GM_ADDR dv, GM_ADDR dpse, GM_ADDR workspace,
                              const FlashAttentionScoreGradTilingDataUs1s2Bbn2* __restrict ordTilingData,
                              TPipe* pipe_in);
  __aicore__ inline void Process();

  using aType = MatmulType<TPosition::GM, CubeFormat::ND, T1>;
  using bType = MatmulType<TPosition::GM, CubeFormat::ND, T1>;
  using aTypeUB = MatmulType<TPosition::VECCALC, CubeFormat::ND, T1>;
  using bTypeUB = MatmulType<TPosition::VECCALC, CubeFormat::ND, T1>;
  using aTypeTranspose = MatmulType<TPosition::GM, CubeFormat::ND, T1, true>;
  using bTypeTranspose = MatmulType<TPosition::GM, CubeFormat::ND, T1, true>;
  using aTypeUBTranspose = MatmulType<TPosition::VECCALC, CubeFormat::ND, T1, true>;
  using bTypeUBTranspose = MatmulType<TPosition::VECCALC, CubeFormat::ND, T1, true>;

  using cType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
  using cTypeUB = MatmulType<TPosition::VECCALC, CubeFormat::ND_ALIGN, T2>;
  using biasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

  Matmul<aType, bTypeTranspose, cTypeUB, biasType> mm1;
  Matmul<aType, bTypeTranspose, cTypeUB, biasType> mm2;
  Matmul<aTypeTranspose, bType, cType, biasType> mm4; // mm3.2 reused mm4
  Matmul<aType, bType, cType, biasType> mm3_1;

 protected:
  // init
  __aicore__ inline void InitSoftSyncBuffer(GM_ADDR workspace);
  __aicore__ inline void InitOutputBuffer(GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dpse);
  __aicore__ inline void InitRequireInputBuffer(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy);
  __aicore__ inline void InitOptionInputBuffer(GM_ADDR pse_shift, GM_ADDR drop_mask, GM_ADDR padding_mask,
                                               GM_ADDR atten_mask, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                               GM_ADDR softmax_in, GM_ADDR attention_in);
  __aicore__ inline void InitParams(const FlashAttentionScoreGradTilingDataUs1s2Bbn2* __restrict ordTilingData);
  __aicore__ inline void InitUB(TPipe* pipe_in);
  __aicore__ inline void InitBmmWorkspace(GM_ADDR workspace);
  __aicore__ inline void InitCastWorkspace(GM_ADDR workspace);
  __aicore__ inline void InitDropWorkspace(GM_ADDR workspace);
  __aicore__ inline void AtomicClean();
  __aicore__ inline void DumpGmZero(GlobalTensor<float>& gm, uint64_t num);

  // process
  __aicore__ inline void CalcProcess();
  __aicore__ inline void DoVector(int64_t s2_o_o);
  __aicore__ inline void SendMatmul2(const int64_t m, const int64_t n, const int64_t a_addr, const int64_t b_addr);
  __aicore__ inline void SendMatmul1(const int64_t m, const int64_t n, const int64_t a_addr, const int64_t b_addr);
  __aicore__ inline void SendMatmul4(const uint32_t real_n, const uint32_t s1_inner, const uint64_t a_in_addr, const uint64_t b_in_addr,
                                     const uint64_t out_addr, const bool is_sync);
  __aicore__ inline void SendMatmul3_1(const uint32_t real_n, const uint32_t s1_inner, const uint64_t a_in_addr, const uint64_t b_in_addr,
                                       const uint64_t out_addr, const bool is_sync);
  __aicore__ inline void SendMatmul3_2(const uint32_t real_n, const uint32_t s1_inner, const uint64_t a_in_addr, const uint64_t b_in_addr,
                                       const uint64_t out_addr, const bool is_sync);
  __aicore__ inline void MTE2_PSE(TQue<QuePosition::VECIN, 1>& que);  // CopyInPSE
  __aicore__ inline void MTE2_ATMask(TQue<QuePosition::VECIN, 1>& que, const uint64_t attenMaskOffset);
  __aicore__ inline void MTE2_SFT(TQue<QuePosition::VECIN, 2>& que);
  __aicore__ inline void MTE2_DropMask(TQue<QuePosition::VECIN, 1>& que);
  __aicore__ inline void MTE2_STFGrad(GlobalTensor<T1>& gmTensor, int64_t addr, LocalTensor<T1>& localTensor,
                                      int64_t num);
  __aicore__ inline void Mte2ForMm2(int64_t s2_o_i);
  __aicore__ inline void Mte2ForMm1(int64_t s2_o_i);
  __aicore__ inline void DoPse(LocalTensor<T2>& dstTensor, LocalTensor<T2>& pseTensor);
  __aicore__ inline void DoMask(LocalTensor<T2>& dstTensor, LocalTensor<T2>& attenMaskTensor);
  __aicore__ inline void DoMaskU8(LocalTensor<T2>& dstTensor);
  __aicore__ inline void DoSimpleSoftMax(LocalTensor<T2>& dstTensor, LocalTensor<float>& sumTensor,
                                         LocalTensor<float>& maxTensor);
  __aicore__ inline void DoSoftmaxGrad(LocalTensor<T2>& dstTensor, TQue<QuePosition::VECIN, 1>& que);
  __aicore__ inline void FullGrad(LocalTensor<T2>& dstTensor, TQue<QuePosition::VECIN, 1>& que);
  __aicore__ inline void SplitGrad(LocalTensor<T2>& dstTensor, TQue<QuePosition::VECIN, 1>& que);
  __aicore__ inline void DoSub(LocalTensor<T2>& dstTensor, LocalTensor<T2>& srcTensor);
  __aicore__ inline void DoMul(LocalTensor<T2>& dstTensor, LocalTensor<T2>& srcTensor);
  __aicore__ inline void OutputDpse(LocalTensor<T2>& dstTensor);
  __aicore__ inline void MmOffsetTensorA(const uint32_t s1_idx, uint64_t& a_addr);
  __aicore__ inline void MmOffsetTensorB(const uint32_t s2_idx, uint64_t& b_addr);
  __aicore__ inline void CalcAttenMaskOffset(uint64_t& attenMaskOffset, const int64_t delta);
  __aicore__ inline void CalcCausalAttenMaskOffset(uint64_t& attenMaskOffset, const int64_t delta);
  __aicore__ inline void CopyInOffsetForSimpleSoftmax();
  __aicore__ inline void CopyInOffset();
  __aicore__ inline void DoDropout(LocalTensor<T2>& dstTensor, LocalTensor<T2>& srcTensor,
                                   LocalTensor<uint8_t>& dropMaskTensor);
  __aicore__ inline void InnerT2Process(int64_t s2_o_i);
  __aicore__ inline void CastTo32(LocalTensor<T2>& dstTensor, LocalTensor<T1>& srcTensor);
  __aicore__ inline void CastTo16(LocalTensor<T1>& dstTensor, LocalTensor<T2>& srcTensor);
  __aicore__ inline void DoMulsScale(LocalTensor<T2>& dstTensor);
  __aicore__ inline void CalcSparseIdx(const uint32_t bIndex, const uint32_t s1Idx, uint32_t& s2_start_idx,
                                       uint32_t& s2_end_idx);
  __aicore__ inline void CopyoutWorkspace(const GlobalTensor<T1>& dstGm, const LocalTensor<T1>& srcTensor);
  __aicore__ inline void AssureUsefulDataBySingleN(int64_t s2_o_o);

 protected:
  GlobalTensor<T1> queryGm;
  GlobalTensor<T1> keyGm;
  GlobalTensor<T1> valueGm;
  GlobalTensor<T1> dyGm;

  GlobalTensor<T1> pseShiftGm;
  GlobalTensor<uint8_t> dropMaskGm;
  GlobalTensor<T1> paddingMaskGm;
  GlobalTensor<T1> attenMaskGm;
  GlobalTensor<uint8_t> attenMaskU8Gm;
  GlobalTensor<float> softmaxMaxGm;
  GlobalTensor<float> softmaxSumGm;
  GlobalTensor<T1> softmaxInGm;
  GlobalTensor<T1> attentionInGm;

  /*
  FP16-HighPerformance: res is workspace
  FP16-HighPrecision:: res is workspace
  BFP16: res is workspace
  FP32: res is dxGm
  */
  GlobalTensor<float> dqGm;
  GlobalTensor<float> dkGm;
  GlobalTensor<float> dvGm;
  GlobalTensor<T1> dpseGm;

  GlobalTensor<T1> mm4InputWorkspaceGm;
  GlobalTensor<T1> mm3InputWorkspaceGm;

  GlobalTensor<uint8_t> dropoutWorkspaceGm;
  GlobalTensor<float> dqWorkspaceGm;
  GlobalTensor<float> dkWorkspaceGm;
  GlobalTensor<float> dvWorkspaceGm;

  GlobalTensor<int32_t> syncAtomicCleanGlobal;
  GlobalTensor<int32_t> syncCastGlobal;

  // matmul1/matmul2 result buffer
  GlobalTensor<float> mm1ResultBuffer;
  GlobalTensor<float> mm2ResultBuffer;

  TPipe* pipe;
  int64_t blockIdx;
  const FlashAttentionScoreGradTilingDataUs1s2Bbn2* __restrict tilingData;

  TQue<QuePosition::VECOUT, 1> bufQue0;
  TQue<QuePosition::VECIN, 1> copyBufQue1;
  TQue<QuePosition::VECIN, 1> copyBufQue2;
  TQue<QuePosition::VECIN, 1> bufQue2;           // matmul1
  TQue<QuePosition::VECIN, 1> bufQue3;           // matmul2
  TQue<QuePosition::VECIN, 1> softmaxGradOutQue;
  TQue<QuePosition::VECIN, 2> simpleSoftmaxQue;  // (S1, 1[p])  2 * 4K
  TBuf<> helpQue;

  // softmax_grad计算外提
  LocalTensor<T2> softmaxGradOutTensor;

  // simplesoftmax计算外提
  LocalTensor<float> simpleSoftmaxSumTensor;
  LocalTensor<float> simpleSoftmaxMaxTensor;

  // core
  int64_t usedCoreNum;
  int64_t formerCoreNum;
  int64_t remainCoreNum;

  // Shape
  uint32_t dimB;
  uint32_t dimN2;
  uint32_t dimS1;
  uint32_t dimS2;
  uint32_t dimG;
  uint32_t dimD;
  uint32_t attenMaskDimS2;

  // attr
  float scaleValue;
  float keepProb;
  uint32_t layout;
  uint32_t existPse;
  uint32_t existAttenMask;
  uint32_t maskDataType;
  uint32_t maskShapeType;
  uint32_t pseShapeType;

  // ub space
  uint32_t inputBufferLen;
  uint32_t helpBufferLen;

  // workspace (couldn't bigger than 4G)
  uint32_t syncDropoutLen;
  uint32_t syncAtomicLen;
  uint32_t syncCastLen;
  uint32_t mm1WorkspaceLen;
  uint32_t mm2WorkspaceLen;
  uint32_t mm3InputWorkspaceLen;
  uint32_t mm4InputWorkspaceLen;
  uint32_t dqWorkspaceLen;
  uint32_t dkWorkspaceLen;
  uint32_t dvWorkspaceLen;
  uint32_t dropoutWorkspaceLen;

  // split_params
  uint32_t baseM;
  uint32_t baseN;
  uint32_t singleN;
  uint32_t baseMmm;    // base block for matmul
  uint32_t baseNmm;
  uint32_t singleNmm;
  uint32_t cvCalcuRatio;
  uint32_t s1OuterNum;
  uint32_t s2OuterOuterNum;
  uint32_t dInner;

  // Index
  uint32_t beginBNByCore;
  uint32_t endBNByCore;
  uint32_t s2OuterInnerNum;
  uint32_t bIndex;
  uint32_t s1Index;
  uint32_t s2Index;
  uint32_t n2Index;
  uint32_t gIndex;

  // 基本块
  uint32_t processM;
  uint32_t processN;
  uint32_t processMTail;
  uint32_t processNTail;
  uint32_t processMv;
  uint32_t processNv;
  uint32_t s1Inner;
  uint32_t s2Inner;
  uint32_t s2InnerAlign;
  uint32_t s2InnerAlignT2;
  uint32_t s2Tail;
  uint32_t dTail;
  uint32_t mmWidthGetSize;

  // 地址相关
  uint64_t inputMMLeftMatrixAddr;   // MM1's addr as same as MM2
  uint64_t inputMMRighMatrixtAddr;  // MM1's addr as same as MM2
  uint64_t dyGmAddr;
  uint64_t attentionInGmAddr;
  uint64_t pseInGmAddr;
  uint64_t attenmaskGmAddr;
  uint64_t dropmaskGmAddr;
  uint64_t dpseGmAddr;
  uint64_t softmaxMaxSumGmAddr;  // max's addr as same as sum 修改变量命名
  uint64_t mm3_4_tensor_g_s1_addr;
  uint64_t mm3_4_tensor_1_s2_addr;
  uint64_t mm3PangInputWspOffset;
  uint64_t mm4PangInputWspOffset;
  uint64_t cal_num;
  bool need_mm_sync;
  bool is_s1s2_last;

  // sparse相关
  uint32_t sparse_s2_start_idx;
  uint32_t sparse_s2_end_idx;
  uint32_t preTokens;
  uint32_t nextTokens;
  uint32_t isSparse;
  uint32_t prefixN[64];

  uint32_t usedSingleNBegin;
  uint32_t usedSingleNEnd;
  uint32_t usedSingleNNum;
  uint32_t realProcessN;
  uint64_t mm3InputWorkspaceAddr;
  uint64_t mm4InputWorkspaceAddr;
  uint64_t lastMm3InputWorkspaceAddr;
  uint64_t lastMm4InputWorkspaceAddr;
  uint32_t lastS1Index;
  uint32_t lastS2Index;
  uint32_t lastUsedSingleNNum;
  uint32_t lastRealProcessN;
  uint32_t lastS1Inner;
  uint32_t not_start = 0;

  // 寄存已使用的workspaceLen
  uint64_t usedWorkspaceLen;
  bool is_fp32;

  constexpr static uint64_t BSH = 0;
  constexpr static uint64_t SBH = 1;
  constexpr static uint64_t BNSD = 2;
  constexpr static uint64_t BSND = 3;

  constexpr static uint64_t BLOCK = 32;
  constexpr static uint64_t STRIDE_LIMIT = 65535;
  constexpr static uint64_t VALUE_ZERO = 0;
  constexpr static uint64_t B32_BLOCK_NUM = BLOCK / sizeof(int32_t);

  // others
  constexpr static uint64_t calcBlockNum = BLOCK / sizeof(T2);
  constexpr static uint64_t dataCopyBlockNum = BLOCK / sizeof(T1);
};

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::Init(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR pse_shift, GM_ADDR drop_mask, GM_ADDR padding_mask,
    GM_ADDR atten_mask, GM_ADDR softmax_max, GM_ADDR softmax_sum, GM_ADDR softmax_in, GM_ADDR attention_in, GM_ADDR dq,
    GM_ADDR dk, GM_ADDR dv, GM_ADDR dpse, GM_ADDR workspace,
    const FlashAttentionScoreGradTilingDataUs1s2Bbn2* __restrict ordTilingData, TPipe* pipe_in) {
  InitOutputBuffer(dq, dk, dv, dpse);

  InitRequireInputBuffer(query, key, value, dy);

  InitOptionInputBuffer(pse_shift, drop_mask, padding_mask, atten_mask, softmax_max, softmax_sum, softmax_in,
                        attention_in);

  InitParams(ordTilingData);

  InitSoftSyncBuffer(workspace);

  InitDropWorkspace(workspace);

  InitBmmWorkspace(workspace);

  InitCastWorkspace(workspace);

  AtomicClean();

  InitUB(pipe_in);
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::InitRequireInputBuffer(GM_ADDR query, GM_ADDR key,
                                                                                       GM_ADDR value, GM_ADDR dy) {
  // 必选输入初始化
  queryGm.SetGlobalBuffer((__gm__ T1*)query);
  keyGm.SetGlobalBuffer((__gm__ T1*)key);
  valueGm.SetGlobalBuffer((__gm__ T1*)value);
  dyGm.SetGlobalBuffer((__gm__ T1*)dy);
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::InitOptionInputBuffer(
    GM_ADDR pse_shift, GM_ADDR drop_mask, GM_ADDR padding_mask, GM_ADDR atten_mask, GM_ADDR softmax_max,
    GM_ADDR softmax_sum, GM_ADDR softmax_in, GM_ADDR attention_in) {
  // 可选输入初始化
  pseShiftGm.SetGlobalBuffer((__gm__ T1*)pse_shift);
  dropMaskGm.SetGlobalBuffer((__gm__ uint8_t*)drop_mask);
  paddingMaskGm.SetGlobalBuffer((__gm__ T1*)padding_mask);
  attenMaskGm.SetGlobalBuffer((__gm__ T1*)atten_mask);
  attenMaskU8Gm.SetGlobalBuffer((__gm__ uint8_t*)atten_mask);
  softmaxMaxGm.SetGlobalBuffer((__gm__ float*)softmax_max);
  softmaxSumGm.SetGlobalBuffer((__gm__ float*)softmax_sum);
  softmaxInGm.SetGlobalBuffer((__gm__ T1*)softmax_in);
  attentionInGm.SetGlobalBuffer((__gm__ T1*)attention_in);
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::InitOutputBuffer(GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                                                                                 GM_ADDR dpse) {
  // 输出初始化
  dqGm.SetGlobalBuffer((__gm__ float*)dq);
  dkGm.SetGlobalBuffer((__gm__ float*)dk);
  dvGm.SetGlobalBuffer((__gm__ float*)dv);
  dpseGm.SetGlobalBuffer((__gm__ T1*)dpse);
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::InitParams(
    const FlashAttentionScoreGradTilingDataUs1s2Bbn2* __restrict ordTilingData) {
  // op_info
  blockIdx = GetBlockIdx();
  tilingData = ordTilingData;
  usedCoreNum = tilingData->opInfo.usedCoreNum;
  formerCoreNum = tilingData->opInfo.formerCoreNum;
  remainCoreNum = tilingData->opInfo.remainCoreNum;

  dimB = tilingData->opInfo.B;
  dimN2 = tilingData->opInfo.N2;
  dimS1 = tilingData->opInfo.S1;
  dimS2 = tilingData->opInfo.S2;
  dimG = tilingData->opInfo.G;
  dimD = tilingData->opInfo.D;
  attenMaskDimS2 = tilingData->opInfo.attenMaskS2Size;

  scaleValue = tilingData->opInfo.scaleValue;
  keepProb = tilingData->opInfo.keepProb;
  layout = tilingData->opInfo.layout;
  preTokens = tilingData->opInfo.preTokens;
  nextTokens = tilingData->opInfo.nextTokens;
  isSparse = tilingData->opInfo.isSparse;
  maskShapeType = tilingData->opInfo.maskShapeType;
  maskDataType = tilingData->opInfo.maskDataType;
  pseShapeType = tilingData->opInfo.pseShapeType;
  existPse = tilingData->opInfo.existPse;
  existAttenMask = tilingData->opInfo.existAttenMask;
  inputBufferLen = tilingData->opInfo.inputBufferLen;
  helpBufferLen = tilingData->opInfo.helpBufferLen;
  syncDropoutLen = tilingData->opInfo.syncDropoutLen;
  syncAtomicLen = tilingData->opInfo.syncAtomicLen;
  syncCastLen = tilingData->opInfo.syncCastLen;
  mm1WorkspaceLen = tilingData->opInfo.mm1WorkspaceLen;
  mm2WorkspaceLen = tilingData->opInfo.mm2WorkspaceLen;
  mm3InputWorkspaceLen = tilingData->opInfo.mm3InputWorkspaceLen;
  mm4InputWorkspaceLen = tilingData->opInfo.mm4InputWorkspaceLen;
  dqWorkspaceLen = tilingData->castParams.dqWorkspaceLen;
  dkWorkspaceLen = tilingData->castParams.dkWorkspaceLen;
  dvWorkspaceLen = tilingData->castParams.dvWorkspaceLen;
  dropoutWorkspaceLen = tilingData->dropoutCastParams.dropoutWorkspaceLen;
  for (uint32_t i = 0; i < 64; ++i) {
    prefixN[i] = tilingData->opInfo.prefixN[i];
  }
  mm3PangInputWspOffset = mm3InputWorkspaceLen / sizeof(T1) / 2;
  mm4PangInputWspOffset = mm4InputWorkspaceLen / sizeof(T1) / 2;

  // split_params
  baseM = tilingData->splitCoreParams.baseM;
  baseN = tilingData->splitCoreParams.baseN;
  singleN = tilingData->splitCoreParams.singleN;
  baseMmm = tilingData->splitCoreParams.baseMmm;
  baseNmm = tilingData->splitCoreParams.baseNmm;
  singleNmm = tilingData->splitCoreParams.singleNmm;
  cvCalcuRatio = baseNmm / baseN;
  s1OuterNum = tilingData->splitCoreParams.s1Outer;
  s2OuterOuterNum = tilingData->splitCoreParams.s2OuterOuter;
  dInner = tilingData->splitCoreParams.dInner;
  mmWidthGetSize = baseNmm;

  // calc
  dTail = dimD % dInner;
  processM = baseMmm;
  processN = singleNmm;
  processMv = baseM;
  processNv = singleN;
  processMTail = (dimS1 % baseMmm == 0) ? baseMmm : dimS1 % baseMmm;
  processNTail = (dimS2 % singleNmm == 0) ? singleNmm : dimS2 % singleNmm;
  s2Tail = dimS2 % baseN == 0 ? baseN : dimS2 % baseN;

  int64_t former_core_process_n_nums = (dimB * dimN2 + usedCoreNum - 1) / usedCoreNum;
  beginBNByCore =
      blockIdx * former_core_process_n_nums - ((blockIdx - formerCoreNum) > 0 ? (blockIdx - formerCoreNum) : 0);
  endBNByCore = (blockIdx + 1) * former_core_process_n_nums -
                ((blockIdx + 1 - formerCoreNum) > 0 ? (blockIdx + 1 - formerCoreNum) : 0);
  is_fp32 = sizeof(T1) == sizeof(float);
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::InitUB(TPipe* pipe_in) {
  pipe = pipe_in;
  pipe->InitBuffer(bufQue0, 1, inputBufferLen);
  pipe->InitBuffer(copyBufQue1, 1, inputBufferLen / 2);     // 2: 16K
  pipe->InitBuffer(copyBufQue2, 1, inputBufferLen / 2);     // 2: 16K
  pipe->InitBuffer(bufQue2, 1, inputBufferLen);
  pipe->InitBuffer(bufQue3, 1, inputBufferLen);
  pipe->InitBuffer(softmaxGradOutQue, 1, baseM * BLOCK);
  pipe->InitBuffer(simpleSoftmaxQue, 2, baseM * BLOCK);
  pipe->InitBuffer(helpQue, helpBufferLen);
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::DumpGmZero(GlobalTensor<float>& gm, uint64_t num) {
  // dump 0 to gm by blockIdx
  uint64_t perSize = (num + usedCoreNum - 1) / usedCoreNum;
  uint64_t coreNum = (num + perSize - 1) / perSize;
  uint64_t tailSize = num - perSize * (coreNum - 1);
  uint64_t initSize = perSize;

  if (blockIdx == coreNum - 1) {
    initSize = tailSize;
  }

  if (blockIdx < coreNum) {
    InitOutput<float>(gm[blockIdx * perSize], initSize, 0);
  }
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::AtomicClean() {
  // FP32 clean
  // Input is B16 clean workspace
  // Input is B32 clean output-gm
  // Used All UB before InitUB
  uint64_t dvSize = dimB * dimN2 * dimS2 * dimD;
  uint64_t dqSize = dimB * dimN2 * dimG * dimS1 * dimD;
  uint64_t dkSize = dimB * dimN2 * dimS2 * dimD;
  dvSize = (dvSize + B32_BLOCK_NUM - 1) / B32_BLOCK_NUM * B32_BLOCK_NUM;
  dqSize = (dqSize + B32_BLOCK_NUM - 1) / B32_BLOCK_NUM * B32_BLOCK_NUM;
  dkSize = (dkSize + B32_BLOCK_NUM - 1) / B32_BLOCK_NUM * B32_BLOCK_NUM;

  if (is_fp32) {
    DumpGmZero(dvGm, dvSize);
    DumpGmZero(dqGm, dqSize);
    DumpGmZero(dkGm, dkSize);
  } else {
    DumpGmZero(dvWorkspaceGm, dvSize);  // FP32 FUNC:T1 FP16
    DumpGmZero(dqWorkspaceGm, dqSize);
    DumpGmZero(dkWorkspaceGm, dkSize);
  }
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::InitSoftSyncBuffer(GM_ADDR workspace) {
  // 软同步初始化(dropout->atomic->cast)
  auto atomicAddr = syncDropoutLen / sizeof(int32_t);
  auto castAddr = atomicAddr + syncAtomicLen / sizeof(int32_t);

  syncAtomicCleanGlobal.SetGlobalBuffer((__gm__ int32_t*)workspace + atomicAddr, syncAtomicLen / sizeof(int32_t));
  syncCastGlobal.SetGlobalBuffer((__gm__ int32_t*)workspace + castAddr, syncCastLen / sizeof(int32_t));

  InitOutput<int32_t>(syncAtomicCleanGlobal[blockIdx * B32_BLOCK_NUM], B32_BLOCK_NUM, VALUE_ZERO);
  InitOutput<int32_t>(syncCastGlobal[blockIdx * B32_BLOCK_NUM], B32_BLOCK_NUM, VALUE_ZERO);

  usedWorkspaceLen = syncDropoutLen + syncAtomicLen + syncCastLen;
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::InitDropWorkspace(GM_ADDR workspace) {
  if (dimS2 % baseN == 0) {
    return;
  }

  // dropout op
  auto dropoutAddr = usedWorkspaceLen / sizeof(uint8_t);
  dropoutWorkspaceGm.SetGlobalBuffer((__gm__ uint8_t*)workspace + dropoutAddr);

  usedWorkspaceLen += dropoutWorkspaceLen;
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::InitBmmWorkspace(GM_ADDR workspace) {
  // bmm used T2
  // mm1WorkspaceLen should be as same as mm2
  auto mm1Addr = usedWorkspaceLen / sizeof(T2) + blockIdx * mm1WorkspaceLen / sizeof(T2);
  auto mm2Addr = usedWorkspaceLen / sizeof(T2) + usedCoreNum * mm1WorkspaceLen / sizeof(T2) +
                 blockIdx * mm2WorkspaceLen / sizeof(T2);
  GlobalTensor<T2> workspaceGm;
  workspaceGm.SetGlobalBuffer((__gm__ T2*)workspace);
  mm1.SetWorkspace((__gm__ uint8_t*)workspaceGm[mm1Addr].GetPhyAddr(), mm1WorkspaceLen);
  mm2.SetWorkspace((__gm__ uint8_t*)workspaceGm[mm2Addr].GetPhyAddr(), mm2WorkspaceLen);

  usedWorkspaceLen += (mm1WorkspaceLen + mm2WorkspaceLen) * usedCoreNum;

  // mm3.1 mm3.2 mm4 used workspace as input of Bmm
  auto mm4Addr = usedWorkspaceLen / sizeof(T1) + blockIdx * mm4InputWorkspaceLen / sizeof(T1);
  usedWorkspaceLen += mm4InputWorkspaceLen * usedCoreNum;
  auto mm3Addr = usedWorkspaceLen / sizeof(T1) + blockIdx * mm3InputWorkspaceLen / sizeof(T1);
  usedWorkspaceLen += mm3InputWorkspaceLen * usedCoreNum;

  mm4InputWorkspaceGm.SetGlobalBuffer((__gm__ T1*)workspace + mm4Addr);
  mm3InputWorkspaceGm.SetGlobalBuffer((__gm__ T1*)workspace + mm3Addr);
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::InitCastWorkspace(GM_ADDR workspace) {
  // fp32 don't need workspace
  if (is_fp32) {
    return;
  }

  auto dqAddr = usedWorkspaceLen / sizeof(float);
  auto dkAddr = dqAddr + dqWorkspaceLen / sizeof(float);
  auto dvAddr = dkAddr + dkWorkspaceLen / sizeof(float);
  dqWorkspaceGm.SetGlobalBuffer((__gm__ float*)workspace + dqAddr);
  dkWorkspaceGm.SetGlobalBuffer((__gm__ float*)workspace + dkAddr);
  dvWorkspaceGm.SetGlobalBuffer((__gm__ float*)workspace + dvAddr);

  usedWorkspaceLen += dqWorkspaceLen + dkWorkspaceLen + dvWorkspaceLen;
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::CopyoutWorkspace(const GlobalTensor<T1>& dstGm,
                                                                                 const LocalTensor<T1>& srcTensor) {
  // send data to workspace which used as bmm's input
  // send shape is [s1Inner, s2Inner]
  // need consider DB
  DataCopyParams intriParams;
  intriParams.blockCount = s1Inner;
  intriParams.blockLen = s2Inner * sizeof(T1);
  intriParams.srcStride = 0;
  intriParams.dstStride = (realProcessN - s2Inner) * sizeof(T1);
  DataCopyPad(dstGm, srcTensor, intriParams);
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::SendMatmul2(const int64_t m, const int64_t n,
                                                                            const int64_t a_addr,
                                                                            const int64_t b_addr) {
  mm2.SetTail(m, n, -1);
  mm2.SetTensorA(queryGm[a_addr]);
  mm2.SetTensorB(keyGm[b_addr], true);
  mm2.template Iterate<false>();
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::SendMatmul1(const int64_t m, const int64_t n,
                                                                            const int64_t a_addr,
                                                                            const int64_t b_addr) {
  mm1.SetTail(m, n, -1);
  mm1.SetTensorA(dyGm[a_addr]);
  mm1.SetTensorB(valueGm[b_addr], true);
  mm1.template Iterate<false>();
}

/*-----------------------------NewMatmulBEGIN---------------------------------------*/
template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::SendMatmul4(const uint32_t real_n,
                                                                            const uint32_t s1_inner,
                                                                            const uint64_t a_in_addr,
                                                                            const uint64_t b_in_addr,
                                                                            const uint64_t out_addr,
                                                                            const bool is_sync) {
  /*
  BNSD:
    A: [s1Inner, realSingleN]
    B: [B, N1, S1, D]
    C: [s1Inner, realSingleN] x [B, N1, S1, D] = [B, N2, S2, D]
    For A, m = realSingleN, k = s1Inner
    For B, n = D, k = S1
    For C, m = S2, n = D
  */
  if (layout == BNSD) {
    mm4.SetOrgShape(real_n, dimD, dimS1, dimS1);
  } else if (layout == BSH || layout == BSND) {
    mm4.SetOrgShape(real_n, dimD * dimN2 * dimG, dimS1, dimS1, dimD * dimN2);
  } else {
    mm4.SetOrgShape(real_n, dimD * dimN2 * dimG * dimB, dimS1, dimS1, dimD * dimN2 * dimB);
  }
  mm4.SetTail(real_n, -1, s1_inner);
  mm4.SetTensorA(mm4InputWorkspaceGm[a_in_addr], true);
  mm4.SetTensorB(dyGm[b_in_addr]);
  if (is_sync) {
    mm4.template IterateAll<true>(dvWorkspaceGm[out_addr], true);
  } else {
    mm4.template IterateAll<false>(dvWorkspaceGm[out_addr], true);
  }
  mm4.End();
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::SendMatmul3_1(const uint32_t real_n,
                                                                              const uint32_t s1_inner,
                                                                              const uint64_t a_in_addr,
                                                                              const uint64_t b_in_addr,
                                                                              const uint64_t out_addr,
                                                                              const bool is_sync) {
  /*
  BSH:
    A: [s1Inner, realSingleN]
    B: [B, N2, S2, D]
    C: [s1Inner, realSingleN] x [B, S2, N2, D] = [B, S1, N1, D]
    For A, m = s1Inner, k = realSingleN
    For B, n = N2 * D, k = S2
    For C, m = S1, n = N1 * D
  */
  if (layout == BNSD) {
    mm3_1.SetOrgShape(dimS1, dimD, real_n, real_n);
  } else if (layout == BSH || layout == BSND) {
    mm3_1.SetOrgShape(dimS1, dimD * dimN2, real_n, real_n, dimD * dimN2 * dimG);
  } else {
    mm3_1.SetOrgShape(dimS1, dimD * dimN2 * dimB, real_n, real_n, dimD * dimN2 * dimG * dimB);
  }
  mm3_1.SetTail(s1_inner, -1, real_n);
  mm3_1.SetTensorA(mm3InputWorkspaceGm[a_in_addr]);
  mm3_1.SetTensorB(keyGm[b_in_addr]);
  if (is_sync) {
    mm3_1.template IterateAll<true>(dqWorkspaceGm[out_addr], true);
  } else {
    mm3_1.template IterateAll<false>(dqWorkspaceGm[out_addr], true);
  }
  mm3_1.End();
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::SendMatmul3_2(const uint32_t real_n,
                                                                              const uint32_t s1_inner,
                                                                              const uint64_t a_in_addr,
                                                                              const uint64_t b_in_addr,
                                                                              const uint64_t out_addr,
                                                                              const bool is_sync) {
  if (layout == BNSD) {
    mm4.SetOrgShape(real_n, dimD, dimS1, dimS1);
  } else if (layout == BSH || layout == BSND) {
    mm4.SetOrgShape(real_n, dimD * dimN2 * dimG, dimS1, dimS1, dimD * dimN2);
  } else {
    mm4.SetOrgShape(real_n, dimD * dimN2 * dimG * dimB, dimS1, dimS1, dimD * dimN2 * dimB);
  }
  mm4.SetTail(real_n, -1, s1_inner);
  mm4.SetTensorA(mm3InputWorkspaceGm[a_in_addr], true);
  mm4.SetTensorB(queryGm[b_in_addr]);
  if (is_sync) {
    mm4.template IterateAll<true>(dkWorkspaceGm[out_addr], true);
  } else {
    mm4.template IterateAll<false>(dkWorkspaceGm[out_addr], true);
  }
  mm4.End();
}
/*-----------------------------NewMatmulEND---------------------------------------*/

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::MTE2_PSE(TQue<QuePosition::VECIN, 1>& que) {
  /*
  Func: move pse_shift to UB
  pseShapeType: 0 means [B, N2, G, S1, S2]
  pseShapeType: 1 means [B, N2, G, 1, S2]
  */
  if (existPse == 0) {
    return;
  }
  // malloc
  auto pseTensor = que.AllocTensor<T1>();

  // params
  DataCopyPadParams padParams;
  DataCopyParams params(s1Inner, s2Inner * sizeof(T1), (dimS2 - s2Inner) * sizeof(T1), 0);
  if (pseShapeType == 1) {
    params.blockCount = 1;
    params.srcStride = 0;
  }

  // move
  pseTensor.SetSize(params.blockCount * s2InnerAlign);  // UB内对齐
  DataCopyPad(pseTensor, pseShiftGm[pseInGmAddr], params, padParams);
  que.EnQue(pseTensor);
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::MTE2_ATMask(TQue<QuePosition::VECIN, 1>& que,
                                                                            const uint64_t attenMaskOffset) {
  /*
  Func: move atten_mask to UB
  maskShapeType: 0 means: [S1, S2]
  maskShapeType: 1 means: [B, 1, 1, S1, S2]
  maskShapeType: 2 means: [B, N2, G, S1, S2]

  maskDataType: 0 means: dtype as same as T1
  maskDataType: 1 means: uint8
  */
  if (existAttenMask == 0) {
    return;
  }

  if (maskDataType == 0) {
    auto attenMaskTensor = que.AllocTensor<T1>();
    attenMaskTensor.SetSize(s1Inner * s2InnerAlign);

    DataCopyParams params(s1Inner, s2Inner * sizeof(T1), (attenMaskDimS2 - s2Inner) * sizeof(T1), 0);
    DataCopyPad(attenMaskTensor, attenMaskGm[attenMaskOffset], params, {false, 0, 0, 0});
    que.EnQue(attenMaskTensor);
  } else {
    auto attenMaskTensor = que.AllocTensor<uint8_t>();
    attenMaskTensor.SetSize(s1Inner * s2InnerAlign);

    DataCopyParams params(s1Inner, s2Inner, (attenMaskDimS2 - s2Inner), 0);
    DataCopyPad(attenMaskTensor, attenMaskU8Gm[attenMaskOffset], params, {false, 0, 0, 0});
    que.EnQue(attenMaskTensor);
  }
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::MTE2_SFT(TQue<QuePosition::VECIN, 2>& que) {
  /*
  Func: move softmax_sum(fp32) to UB
  max_shape is fixed that [B, N2, G, S1, 8]
  */
  auto sumTensor = que.AllocTensor<float>();
  auto maxTensor = que.AllocTensor<float>();

  sumTensor.SetSize(s1Inner * 8);
  maxTensor.SetSize(s1Inner * 8);

  DataCopy(sumTensor, softmaxSumGm[softmaxMaxSumGmAddr], s1Inner * 8);
  DataCopy(maxTensor, softmaxMaxGm[softmaxMaxSumGmAddr], s1Inner * 8);

  que.EnQue(sumTensor);
  que.EnQue(maxTensor);
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::MTE2_DropMask(TQue<QuePosition::VECIN, 1>& que) {
  /*
  DropMask: 输入的每一个UINT8代表mask当前位置
  DropMaskShape: [B,N2,G,S1,S2]
  ubShape: [s1Inner, align(s2Inner, 32)]
  */
  if (keepProb >= 1) {
    return;
  }

  auto dropMaskTensor = que.AllocTensor<uint8_t>();

  if (dimS2 % baseN != 0) {
    DataCopyParams intriParams;
    intriParams.blockCount = s1Inner;
    intriParams.blockLen = s2Inner;
    intriParams.srcStride = dimS2 - s2Inner;
    intriParams.dstStride = 0;

    DataCopyPadParams padParams;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = (s2Inner + BLOCK - 1) / BLOCK * BLOCK - s2Inner;
    padParams.paddingValue = 0;

    DataCopyPad(dropMaskTensor, dropoutWorkspaceGm[dropmaskGmAddr], intriParams, padParams);
  } else {
    DataCopyPad(dropMaskTensor, dropMaskGm[dropmaskGmAddr / 8],
                {static_cast<uint16_t>(s1Inner), static_cast<uint16_t>(s2Inner / 8),
                static_cast<uint16_t>((dimS2 - s2Inner) / 8), 0},
                {false, 0, 0, 0});
  }

  que.EnQue(dropMaskTensor);
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::MTE2_STFGrad(GlobalTensor<T1>& gmTensor, int64_t addr,
                                                                             LocalTensor<T1>& localTensor,
                                                                             int64_t num) {
  // dy and attentionIn had same address
  int64_t srcStride = 0;
  int64_t offsetStride = 0;
  int64_t num_align = (num + dataCopyBlockNum - 1) / dataCopyBlockNum * dataCopyBlockNum;
  if (layout == BNSD) {
    offsetStride = dimD;
    srcStride = (dimD - num) * sizeof(T1);
  } else if (layout == SBH) {
    offsetStride = dimB * dimN2 * dimG * dimD;
    srcStride = (dimB * dimN2 * dimG * dimD - num) * sizeof(T1);
  } else {
    offsetStride = dimN2 * dimG * dimD;
    srcStride = (dimN2 * dimG * dimD - num) * sizeof(T1);
  }

  DataCopyPadParams padParams;
  DataCopyParams intriParams;
  if (srcStride <= STRIDE_LIMIT) {
    intriParams.blockCount = s1Inner;
    intriParams.blockLen = num * sizeof(T1);
    intriParams.srcStride = srcStride;
    intriParams.dstStride = 0;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = num_align - num;
    padParams.paddingValue = 0;
    DataCopyPad(localTensor, gmTensor[addr], intriParams, padParams);
  } else {
    intriParams.blockCount = 1;
    intriParams.blockLen = num * sizeof(T1);
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = num_align - num;
    padParams.paddingValue = 0;
    for (int64_t i = 0; i < s1Inner; i++) {
      DataCopyPad(localTensor[i * num_align], gmTensor[addr], intriParams, padParams);
      addr += offsetStride;
    }
  }
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::CastTo32(LocalTensor<T2>& dstTensor,
                                                                         LocalTensor<T1>& srcTensor) {
  pipe_barrier(PIPE_V);
  Cast(dstTensor, srcTensor, RoundMode::CAST_NONE, srcTensor.GetSize());  // 以前是s1_inner * s2InnerAlign
  pipe_barrier(PIPE_V);
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::CastTo16(LocalTensor<T1>& dstTensor,
                                                                         LocalTensor<T2>& srcTensor) {
  pipe_barrier(PIPE_V);
  Cast(dstTensor, srcTensor, RoundMode::CAST_ROUND, srcTensor.GetSize());  // 以前是s1_inner * s2InnerAlign
  pipe_barrier(PIPE_V);
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::DoPse(LocalTensor<T2>& dstTensor,
                                                                      LocalTensor<T2>& pseTensor) {
  /*
  Func: mm2 = mm2 + pse
  1.pseShapeType: 0 means [B,N,G,S1,S2]
  2.pseShapeType: 1 means [B,N,G,1, S2]
  */
  // setSize
  pseTensor.SetSize(s1Inner * s2InnerAlign);
  dstTensor.SetSize(s1Inner * s2InnerAlign);

  // broadcast [1, n'] -> [m, n']
  if (pseShapeType == 1) {
    for (uint32_t i = 1; i < s1Inner; i++) {
      DataCopy(pseTensor[i * s2InnerAlign], pseTensor[0], s2InnerAlign);
    }
  }

  // ADD
  pipe_barrier(PIPE_V);
  Add(dstTensor, dstTensor, pseTensor, pseTensor.GetSize());
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::DoMask(LocalTensor<T2>& dstTensor,
                                                                       LocalTensor<T2>& attenMaskTensor) {
  /*
  Func:
  1. 仅支持T1场景，待支持UINT8场景
  2. attenMaskTensor FP16 使用 DataMovePad 完成脏数据清0
  maskDataType: 0 means: dtype as same as T1
  maskDataType: 1 means: uint8
  */
  // B16 or B32
  T2 value = -10000.0;
  attenMaskTensor.SetSize(s1Inner * s2InnerAlign);
  pipe_barrier(PIPE_V);
  Muls(attenMaskTensor, attenMaskTensor, value, s1Inner * s2InnerAlign);
  pipe_barrier(PIPE_V);
  Add(dstTensor, dstTensor, attenMaskTensor, s1Inner * s2InnerAlign);
}
template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::DoMaskU8(LocalTensor<T2>& dstTensor) {
  /*
  Func:
  1. 仅支持T1场景，待支持UINT8场景
  2. attenMaskTensor FP16 使用 DataMovePad 完成脏数据清0
  maskDataType: 0 means: dtype as same as T1
  maskDataType: 1 means: uint8
  */
  // uint8
  auto attenMaskTensor = copyBufQue2.DeQue<uint8_t>();
  auto helpTensor = helpQue.Get<uint8_t>();
  T2 scalar;
  if constexpr (IsSameType<T2, float>::value) {
    uint32_t tmp = 0xFF7FFFFF;
    scalar = *((float*)&tmp);
  } else {
    uint16_t tmp = 0xFBFF;
    scalar = *((half*)&tmp);
  }

  SelectWithBytesMaskShapeInfo info;
  info.firstAxis = s1Inner;
  info.srcLastAxis = s2InnerAlign;
  info.maskLastAxis = (s2InnerAlign + 31) / 32 * 32;
  dstTensor.SetSize(info.firstAxis * info.srcLastAxis);
  attenMaskTensor.SetSize(info.firstAxis * info.maskLastAxis);
  SelectWithBytesMask(dstTensor, dstTensor, scalar, attenMaskTensor, helpTensor, info);
  copyBufQue2.FreeTensor(attenMaskTensor);
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::DoSimpleSoftMax(LocalTensor<T2>& dstTensor,
                                                                                LocalTensor<float>& sumTensor,
                                                                                LocalTensor<float>& maxTensor) {
  /*
  Func:
  1. sum/std::max shape is [S1, 8] FP32
  2. maxTensor had been stored
  */
  auto helpTensor = helpQue.Get<uint8_t>();
  // set size
  sumTensor.SetSize(s1Inner * 8);
  maxTensor.SetSize(s1Inner * 8);

  // set shape
  uint32_t softmaxShape[2] = {static_cast<uint32_t>(s1Inner), 8};
  sumTensor.SetShapeInfo(ShapeInfo(2, softmaxShape, DataFormat::ND));
  maxTensor.SetShapeInfo(ShapeInfo(2, softmaxShape, DataFormat::ND));

  uint32_t dstSoftShape[2] = {static_cast<uint32_t>(s1Inner), static_cast<uint32_t>(s2InnerAlign)};
  dstTensor.SetShapeInfo(ShapeInfo(2, dstSoftShape, DataFormat::ND));

  // calc reused buffer
  if ((s1Inner % 8 == 0) && (s2InnerAlign % 64 == 0)) {
    SimpleSoftMax<T2, true, true>(dstTensor, sumTensor, maxTensor, dstTensor, helpTensor,
                                  tilingData->softmaxTilingData);
  } else {
    SimpleSoftMax<T2, true, false>(dstTensor, sumTensor, maxTensor, dstTensor, helpTensor,
                                   tilingData->softmaxTilingData);
  }
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::DoDropout(LocalTensor<T2>& dstTensor,
                                                                          LocalTensor<T2>& srcTensor,
                                                                          LocalTensor<uint8_t>& dropMaskTensor) {
  // 每个UINT8元素代表是否进行掩码 [B,N2,G,S1,S2]
  auto helpTensor = helpQue.Get<uint8_t>();
  srcTensor.SetSize(s1Inner * s2InnerAlign);
  dropMaskTensor.SetSize(s1Inner * (s2Inner + BLOCK - 1) / BLOCK * BLOCK);
  dstTensor.SetSize(s1Inner * s2InnerAlign);
  DropOutShapeInfo info;
  info.firstAxis = s1Inner;
  info.srcLastAxis = s2InnerAlign; // srcTensor maybe 16 or 8 align

  if (dimS2 % baseN != 0) {
    info.maskLastAxis = (s2Inner + BLOCK - 1) / BLOCK * BLOCK; // 32 align
    DropOut<T2, true>(dstTensor, srcTensor, dropMaskTensor, helpTensor, keepProb, info);
  } else {
    info.maskLastAxis = (s2Inner / 8 + BLOCK - 1) / BLOCK * BLOCK; // 32 align
    DropOut(dstTensor, srcTensor, dropMaskTensor, helpTensor, keepProb, info);
  }
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::FullGrad(LocalTensor<T2>& dstTensor,
                                                                         TQue<QuePosition::VECIN, 1>& que) {
  // 1. D通道不切分场景
  // 2. FP16/BFP16 cast 为FP32计算
  // 3. dstTensor为引用，用来指向最终结果
  // 4. que:softmaxGradOutQue

  auto d_align = (dimD + dataCopyBlockNum - 1) / dataCopyBlockNum * dataCopyBlockNum;
  // copyIn + cast dyGm
  auto dyT1Tensor = copyBufQue1.AllocTensor<T1>();
  dyT1Tensor.SetSize(s1Inner * d_align);
  MTE2_STFGrad(dyGm, dyGmAddr, dyT1Tensor, dimD);
  copyBufQue1.EnQue(dyT1Tensor);
  copyBufQue1.DeQue<T1>();
  auto dyT2Tensor = bufQue0.AllocTensor<T2>();
  dyT2Tensor.SetSize(s1Inner * d_align);
  CastTo32(dyT2Tensor, dyT1Tensor);
  copyBufQue1.FreeTensor(dyT1Tensor);

  // copyIn + cast attentionGm
  auto attentionT1Tensor = bufQue3.AllocTensor<T1>();
  attentionT1Tensor.SetSize(s1Inner * d_align);
  MTE2_STFGrad(attentionInGm, attentionInGmAddr, attentionT1Tensor, dimD);
  bufQue3.EnQue(attentionT1Tensor);
  bufQue3.DeQue<T1>();
  auto attentionT2Tensor = bufQue2.AllocTensor<T2>();
  attentionT2Tensor.SetSize(s1Inner * d_align);
  CastTo32(attentionT2Tensor, attentionT1Tensor);
  bufQue3.FreeTensor(attentionT1Tensor);

  // calc softmax
  auto helpTensor = helpQue.Get<uint8_t>();
  // setshape size
  uint32_t dyShape[2] = {static_cast<uint32_t>(s1Inner), static_cast<uint32_t>(d_align)};
  uint32_t attentionInShape[2] = {static_cast<uint32_t>(s1Inner), static_cast<uint32_t>(d_align)};
  uint32_t softmaxGradShape[2] = {static_cast<uint32_t>(s1Inner), static_cast<uint32_t>(calcBlockNum)};
  dstTensor = que.AllocTensor<T2>();
  dstTensor.SetSize(s1Inner * calcBlockNum);
  dstTensor.SetShapeInfo(ShapeInfo(2, softmaxGradShape, DataFormat::ND));
  dyT2Tensor.SetShapeInfo(ShapeInfo(2, dyShape, DataFormat::ND));
  attentionT2Tensor.SetShapeInfo(ShapeInfo(2, attentionInShape, DataFormat::ND));
  if ((s1Inner % 8 == 0) && (dimD % 64 == 0)) {
    SoftmaxGradFront<T2, true>(dstTensor, dyT2Tensor, attentionT2Tensor, helpTensor, tilingData->softmaxGradTilingData);
  } else {
    SoftmaxGradFront<T2, false>(dstTensor, dyT2Tensor, attentionT2Tensor, helpTensor, tilingData->softmaxGradTilingData);
  }
  // release
  bufQue2.FreeTensor(attentionT2Tensor);
  bufQue0.FreeTensor(dyT2Tensor);
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::SplitGrad(LocalTensor<T2>& dstTensor,
                                                                          TQue<QuePosition::VECIN, 1>& que) {
  // 1. D通道切分场景
  // 2. FP16/BFP16 cast 为FP32计算
  // 3. dstTensor为引用，用来指向最终结果
  // 4. que:copyBufQue1
  // define sharedBuffer
  auto helpTensorB = helpQue.GetWithOffset<T2>(s1Inner * calcBlockNum, 0);
  auto tempLen = helpBufferLen - s1Inner * calcBlockNum * sizeof(T2);
  auto helpTensorA = helpQue.GetWithOffset<uint8_t>(tempLen, s1Inner * calcBlockNum * sizeof(T2));
  helpTensorB.SetSize(s1Inner * calcBlockNum);
  helpTensorA.SetSize(tempLen);

  // init zero
  Duplicate<T2>(helpTensorB, 0, s1Inner * calcBlockNum);

  // processMain
  auto d_align = (dInner + dataCopyBlockNum - 1) / dataCopyBlockNum * dataCopyBlockNum;
  uint32_t softmaxGradShape[2] = {static_cast<uint32_t>(s1Inner), static_cast<uint32_t>(calcBlockNum)};
  uint32_t dyShape[2] = {static_cast<uint32_t>(s1Inner), static_cast<uint32_t>(d_align)};
  uint32_t attentionInShape[2] = {static_cast<uint32_t>(s1Inner), static_cast<uint32_t>(d_align)};

  // set localBuffer
  auto dyT2Tensor = bufQue0.AllocTensor<T2>();
  auto attentionT2Tensor = bufQue2.AllocTensor<T2>();
  dyT2Tensor.SetSize(s1Inner * d_align);
  attentionT2Tensor.SetSize(s1Inner * d_align);
  dyT2Tensor.SetShapeInfo(ShapeInfo(2, dyShape, DataFormat::ND));
  attentionT2Tensor.SetShapeInfo(ShapeInfo(2, attentionInShape, DataFormat::ND));

  for (auto loop = 0; loop < dimD / dInner; loop++) {
    int64_t addr = dyGmAddr + loop * dInner;

    // copyIn + cast dyGm
    auto dyT1Tensor = copyBufQue1.AllocTensor<T1>();
    dyT1Tensor.SetSize(s1Inner * d_align);
    MTE2_STFGrad(dyGm, addr, dyT1Tensor, dInner);
    copyBufQue1.EnQue(dyT1Tensor);
    copyBufQue1.DeQue<T1>();
    CastTo32(dyT2Tensor, dyT1Tensor);
    copyBufQue1.FreeTensor(dyT1Tensor);

    // copyIn + cast attentionGm
    auto attentionT1Tensor = bufQue3.AllocTensor<T1>();
    attentionT1Tensor.SetSize(s1Inner * d_align);
    MTE2_STFGrad(attentionInGm, addr, attentionT1Tensor, dInner);
    bufQue3.EnQue(attentionT1Tensor);
    bufQue3.DeQue<T1>();
    CastTo32(attentionT2Tensor, attentionT1Tensor);
    bufQue3.FreeTensor(attentionT1Tensor);

    // calc
    auto stfTensor = que.AllocTensor<T2>();
    stfTensor.SetSize(s1Inner * calcBlockNum);
    stfTensor.SetShapeInfo(ShapeInfo(2, softmaxGradShape, DataFormat::ND));
    if ((s1Inner % 8 == 0) && (d_align % 64 == 0)) {
      SoftmaxGradFront<T2, true>(stfTensor, dyT2Tensor, attentionT2Tensor, helpTensorA, tilingData->softmaxGradTilingData);
    } else {
      SoftmaxGradFront<T2, false>(stfTensor, dyT2Tensor, attentionT2Tensor, helpTensorA, tilingData->softmaxGradTilingData);
    }

    // release
    pipe_barrier(PIPE_V);
    Add(helpTensorB, helpTensorB, stfTensor, s1Inner * calcBlockNum);
    que.FreeTensor(stfTensor);
  }

  // process-tail
  if (dTail != 0) {
    int64_t addr = dyGmAddr + dimD / dInner * dInner;
    d_align = (dTail + dataCopyBlockNum - 1) / dataCopyBlockNum * dataCopyBlockNum;

    // set shape size
    dyShape[1] = d_align;
    attentionInShape[1] = d_align;
    dyT2Tensor.SetSize(s1Inner * d_align);
    attentionT2Tensor.SetSize(s1Inner * d_align);
    dyT2Tensor.SetShapeInfo(ShapeInfo(2, dyShape, DataFormat::ND));
    attentionT2Tensor.SetShapeInfo(ShapeInfo(2, attentionInShape, DataFormat::ND));

    // copyIn + cast dyGm
    auto dyT1Tensor = copyBufQue1.AllocTensor<T1>();
    dyT1Tensor.SetSize(s1Inner * d_align);
    MTE2_STFGrad(dyGm, addr, dyT1Tensor, dTail);
    copyBufQue1.EnQue(dyT1Tensor);
    copyBufQue1.DeQue<T1>();
    CastTo32(dyT2Tensor, dyT1Tensor);
    copyBufQue1.FreeTensor(dyT1Tensor);

    // copyIn + cast attentionGm
    auto attentionT1Tensor = bufQue3.AllocTensor<T1>();
    attentionT1Tensor.SetSize(s1Inner * d_align);
    MTE2_STFGrad(attentionInGm, addr, attentionT1Tensor, dTail);
    bufQue3.EnQue(attentionT1Tensor);
    bufQue3.DeQue<T1>();
    CastTo32(attentionT2Tensor, attentionT1Tensor);
    bufQue3.FreeTensor(attentionT1Tensor);

    // calc
    auto stfTensor = que.AllocTensor<T2>();
    stfTensor.SetSize(s1Inner * calcBlockNum);
    stfTensor.SetShapeInfo(ShapeInfo(2, softmaxGradShape, DataFormat::ND));
    if ((s1Inner % 8 == 0) && (d_align % 64 == 0)) {
      SoftmaxGradFront<T2, true>(stfTensor, dyT2Tensor, attentionT2Tensor, helpTensorA, tilingData->softmaxGradTilingData);
    } else {
      SoftmaxGradFront<T2, false>(stfTensor, dyT2Tensor, attentionT2Tensor, helpTensorA, tilingData->softmaxGradTilingData);
    }

    // release
    pipe_barrier(PIPE_V);
    Add(helpTensorB, helpTensorB, stfTensor, s1Inner * calcBlockNum);
    que.FreeTensor(stfTensor);
  }

  // dataCopy
  dstTensor = que.AllocTensor<T2>();
  dstTensor.SetSize(s1Inner * calcBlockNum);
  DataCopy(dstTensor, helpTensorB, dstTensor.GetSize());

  // release
  bufQue2.FreeTensor(attentionT2Tensor);
  bufQue0.FreeTensor(dyT2Tensor);
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::DoSoftmaxGrad(LocalTensor<T2>& dstTensor,
                                                                              TQue<QuePosition::VECIN, 1>& que) {
  if (dInner == dimD) {
    FullGrad(dstTensor, que);
  } else {
    SplitGrad(dstTensor, que);
  }
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::DoSub(LocalTensor<T2>& dstTensor,
                                                                      LocalTensor<T2>& srcTensor) {
  /*
  需要支持：非对齐，bf16/fp16数据类型
  输入：a(S1,S2) b(S1,P)
  输出：out(S1,S2)
  计算逻辑：S1方向repeat，S2方向for
  当前仅支持（B，N2，G，S1，D) fp16, D不切分,S1<=255;
  */
  dstTensor.SetSize(s1Inner * s2InnerAlign);
  int64_t s2_for_time = s2InnerAlign / calcBlockNum / 8;
  int64_t s2_for_remain = s2InnerAlign / calcBlockNum % 8;
  int64_t mask_sub = 8 * calcBlockNum;
  for (int64_t i = 0; i < s2_for_time; i++) {
    Sub(dstTensor[i * mask_sub], dstTensor[i * mask_sub], srcTensor, mask_sub, s1Inner,
        {1, 1, 0, (uint8_t)(s2InnerAlign / calcBlockNum), (uint8_t)(s2InnerAlign / calcBlockNum), 1});
  }
  if (s2_for_remain) {
    Sub(dstTensor[s2_for_time * mask_sub], dstTensor[s2_for_time * mask_sub], srcTensor, s2_for_remain * 8, s1Inner,
        {1, 1, 0, (uint8_t)(s2InnerAlign / calcBlockNum), (uint8_t)(s2InnerAlign / calcBlockNum), 1});
  }
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::DoMul(LocalTensor<T2>& dstTensor,
                                                                      LocalTensor<T2>& srcTensor) {
  Mul(dstTensor, dstTensor, srcTensor, s1Inner * s2InnerAlign);
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::DoMulsScale(LocalTensor<T2>& dstTensor) {
  pipe_barrier(PIPE_V);
  Muls(dstTensor, dstTensor, scaleValue, s1Inner * s2InnerAlign);
  pipe_barrier(PIPE_V);
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::CalcSparseIdx(const uint32_t bIndex,
  const uint32_t s1_idx, uint32_t& s2_start_idx, uint32_t& s2_end_idx) {
  /*
           s2
     _____________________
     |              \    |
     |    ████       \   |
  s1 |\               \  |
     | \               \ |
     |  \               \|
     |___\_______________|

  ---------------sparse暂只支持s2方向MM配比1:1--------------------

  s2_start_idx 表示基本块上边对应行与左下对角线的交点
  s2_end_idx   表示基本块下边对应行与右上对角线的交点
  如果基本块右上角s2索引 <= s2_start_idx 该基本块无需计算
  如果基本块左下角s2索引 >= s2_end_idx 该基本块无需计算
  */

  if (tilingData->opInfo.sparseMode == 3 || tilingData->opInfo.sparseMode == 5) {
    s2_start_idx = 0;
    int32_t s2IgnoredEndLen = static_cast<int32_t>(tilingData->opInfo.S1) - static_cast<int32_t>(s1_idx + s1Inner);
    int32_t s2EndLen = 0;
    if (static_cast<int32_t>(tilingData->opInfo.S2) > s2IgnoredEndLen) {
        s2EndLen = static_cast<int32_t>(tilingData->opInfo.S2) - s2IgnoredEndLen;
    } else {
        s2EndLen = 0;
    }
    if (tilingData->opInfo.sparseMode == 5) {
        int32_t prefix_n = static_cast<int32_t>(prefixN[bIndex]);
        int32_t alignPrefixN = (prefix_n + baseNmm - 1) / baseNmm * baseNmm;
        s2EndLen = s2EndLen >= alignPrefixN ? s2EndLen : alignPrefixN;
    }
    s2_end_idx = static_cast<uint32_t>(s2EndLen);
    s2_end_idx = s2_end_idx > tilingData->opInfo.S2 ? tilingData->opInfo.S2 : s2_end_idx;
  } else {
    s2_start_idx = (s1_idx <= preTokens) ? 0 : (s1_idx - preTokens);
    // s1_idx + s1Inner表示基本块下边s1索引
    s2_end_idx = ((dimS2 < nextTokens) || ((s1_idx + s1Inner) >= (dimS2 - nextTokens))) ?
                dimS2 : (s1_idx + s1Inner + nextTokens);
  }
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::OutputDpse(LocalTensor<T2>& dstTensor) {
  bufQue0.EnQue(dstTensor);
  bufQue0.DeQue<T2>();
  DataCopyParams copyParams{s1Inner, s2Inner * sizeof(T1), 0, 0};  // D不切分
  DataCopyPadParams padParams{false, 0, 0, 0};
  DataCopyPad(dpseGm[dpseGmAddr], dstTensor, copyParams, padParams);
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::MmOffsetTensorA(const uint32_t s1_idx, uint64_t& a_addr) {
  /*
  BSH = 0, //  A: B S1 N2 G D
  SBH = 1, //  A: S1 B N2 G D
  BNSD = 2, // A: B N2 G S1 D
  BSND = 3, // A: B S1 N2 G D
  */

  if (layout == BSH || layout == BSND) {
    a_addr = bIndex * (dimS1 * dimN2 * dimG * dimD) + s1_idx * (dimN2 * dimG * dimD) + n2Index * (dimG * dimD) +
             gIndex * dimD;
    return;
  }
  if (layout == SBH) {
    a_addr = s1_idx * (dimB * dimN2 * dimG * dimD) + bIndex * (dimN2 * dimG * dimD) + n2Index * (dimG * dimD) +
             gIndex * dimD;
    return;
  }
  if (layout == BNSD) {
    a_addr = bIndex * (dimN2 * dimG * dimS1 * dimD) + n2Index * (dimG * dimS1 * dimD) + gIndex * (dimS1 * dimD) +
             s1_idx * dimD;
    return;
  }
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::MmOffsetTensorB(const uint32_t s2_idx, uint64_t& b_addr) {
  /*
  BSH = 0, //  B: B S2 N2 1 D
  SBH = 1, //  B: S2 B N2 1 D
  BNSD = 2, // B: B N2 1 S2 D
  BSND = 3, // B: B S2 N2 1 D
  */

  if (layout == BSH || layout == BSND) {
    b_addr = bIndex * (dimS2 * dimN2 * dimD) + s2_idx * (dimN2 * dimD) + n2Index * dimD;
    return;
  }
  if (layout == SBH) {
    b_addr = s2_idx * (dimB * dimN2 * dimD) + bIndex * (dimN2 * dimD) + n2Index * dimD;
    return;
  }
  if (layout == BNSD) {
    b_addr = bIndex * (dimN2 * dimS2 * dimD) + n2Index * (dimS2 * dimD) + s2_idx * dimD;
    return;
  }
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::CalcAttenMaskOffset(
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
            attenMaskOffset = s2Inner * attenMaskDimS2;
        } else {
            attenMaskOffset = delta * attenMaskDimS2;
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::CalcCausalAttenMaskOffset(
        uint64_t& attenMaskOffset, const int64_t delta) {
    CalcAttenMaskOffset(attenMaskOffset, delta);
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::CopyInOffsetForSimpleSoftmax() {
  int64_t softmax_max_sum_in_block_num = BLOCK / sizeof(float);
  softmaxMaxSumGmAddr = bIndex * (dimN2 * dimG * dimS1 * softmax_max_sum_in_block_num) +
                        n2Index * (dimG * dimS1 * softmax_max_sum_in_block_num) +
                        gIndex * (dimS1 * softmax_max_sum_in_block_num) + s1Index * softmax_max_sum_in_block_num;
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::CopyInOffset() {
  /*
  pseShapeType = 0: 代表PSE的输入是[B,N,G,S1,S2]
  pseShapeType = 1; 代表PSE的输入是[B,N,G,1,S2]

  maskShapeType = 0; 代表attenMask的输入是[S1,S2]
  maskShapeType = 1; 代表attenMask的输入是[B,1,1,S1,S2]
  maskShapeType = 2; 代表attenMask的输入是[B,N,G,S1,S2]

  maskDataType: 0 代表数据类型 与普通Q\K\V输入Type一致（他们要么一起是FP16, 要么一起FP32, 要么一起BFP16）
  maskDataType: 1 代表数据类型是UINT8

  drop_mask: 当前认为已经是bool 1:1对应mm输出
  */
  if (pseShapeType == 0) {
    pseInGmAddr = bIndex * (dimN2 * dimG * dimS1 * dimS2) + n2Index * (dimG * dimS1 * dimS2) + gIndex * dimS1 * dimS2 +
                  s1Index * dimS2 + s2Index;
  } else {
    pseInGmAddr = bIndex * (dimN2 * dimG * dimS2) + n2Index * (dimG * dimS2) + gIndex * dimS2 + s2Index;
  }

  uint64_t compressMode = tilingData->opInfo.attenMaskCompressMode;
  int64_t causal_delta = (int64_t)s1Index - (int64_t)s2Index;
  if (compressMode == 1) {
    // causal s1==s2
    CalcCausalAttenMaskOffset(attenmaskGmAddr, causal_delta);
  } else if (compressMode == 2) {
    // causal s1!=s2
    causal_delta = causal_delta - dimS1 + dimS2;
    CalcCausalAttenMaskOffset(attenmaskGmAddr, causal_delta);
  } else if (maskShapeType == 2) {
    attenmaskGmAddr = bIndex * (dimN2 * dimG * dimS1 * dimS2) + n2Index * (dimG * dimS1 * dimS2) +
                      gIndex * dimS1 * dimS2 + s1Index * dimS2 + s2Index;
  } else if (maskShapeType == 1) {
    attenmaskGmAddr = bIndex * (dimS1 * dimS2) + s1Index * dimS2 + s2Index;
  } else {
    attenmaskGmAddr = s1Index * dimS2 + s2Index;
  }
  dropmaskGmAddr = bIndex * (dimN2 * dimG * dimS1 * dimS2) + n2Index * (dimG * dimS1 * dimS2) + gIndex * dimS1 * dimS2 +
                   s1Index * dimS2 + s2Index;
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::Mte2ForMm2(int64_t s2_o_i) {
  // get mm2 single base offset addr
  if (s2_o_i % cvCalcuRatio == 0) {
    mmWidthGetSize = (s2_o_i + cvCalcuRatio >= s2OuterInnerNum && processN % baseNmm != 0) ? (processN % baseNmm) : baseNmm;
    mm2ResultBuffer = mm2.template GetTensorC<false>(false, true);
    mm2.End();
  }
  auto mm2TensorCurr = bufQue3.AllocTensor<T2>();
  DataCopyPad(mm2TensorCurr, mm2ResultBuffer[(s2_o_i % cvCalcuRatio) * baseN],
              {static_cast<uint16_t>(s1Inner), static_cast<uint16_t>(s2Inner * sizeof(float)),
               static_cast<uint16_t>((mmWidthGetSize - s2Inner) * sizeof(float)), 0},
               {false, 0, 0, 0});
  bufQue3.EnQue(mm2TensorCurr);
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::Mte2ForMm1(int64_t s2_o_i) {
  // get mm1 single base offset addr
  if (s2_o_i % cvCalcuRatio == 0) {
    mm1ResultBuffer = mm1.template GetTensorC<false>(false, true);
    mm1.End();
  }
  auto mm1TensorCurr = bufQue2.AllocTensor<T2>();
  DataCopyPad(mm1TensorCurr, mm1ResultBuffer[(s2_o_i % cvCalcuRatio) * baseN],
              {static_cast<uint16_t>(s1Inner), static_cast<uint16_t>(s2Inner * sizeof(float)),
               static_cast<uint16_t>((mmWidthGetSize - s2Inner) * sizeof(float)), 0},
               {false, 0, 0, 0});
  bufQue2.EnQue(mm1TensorCurr);
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::InnerT2Process(int64_t s2_o_i) {
  /*
  Branch:
  1. FP16输入 FP32计算
  2. FBP16输入 FP32计算
  */
  // do pse datacopy
  MTE2_PSE(copyBufQue1);

  // do attenmask datacopy
  MTE2_ATMask(copyBufQue2, attenmaskGmAddr);

  // do mm2 res datacopy
  Mte2ForMm2(s2_o_i);
  auto mm2Tensor = bufQue3.DeQue<T2>();

  // do pse
  if (existPse == 1) {
    auto pseB16Tensor = copyBufQue1.DeQue<T1>();
    pseB16Tensor.SetSize(pseShapeType == 1 ? s2InnerAlign : s1Inner * s2InnerAlign);  // BUG
    auto pseB32Tensor = bufQue0.AllocTensor<T2>();
    CastTo32(pseB32Tensor, pseB16Tensor);
    copyBufQue1.FreeTensor(pseB16Tensor);
    DoPse(mm2Tensor, pseB32Tensor);
    bufQue0.FreeTensor(pseB32Tensor);
  }

  // do muls scale
  DoMulsScale(mm2Tensor);

  // do mask
  if (existAttenMask == 1) {
    if (maskDataType == 1) {
      DoMaskU8(mm2Tensor);
    } else {
      auto attenMaskB16Tensor = copyBufQue2.DeQue<T1>();
      auto attenMaskB32Tensor = bufQue0.AllocTensor<T2>();
      attenMaskB16Tensor.SetSize(s1Inner * s2InnerAlign);  // BUG
      CastTo32(attenMaskB32Tensor, attenMaskB16Tensor);
      copyBufQue2.FreeTensor(attenMaskB16Tensor);
      DoMask(mm2Tensor, attenMaskB32Tensor);
      bufQue0.FreeTensor(attenMaskB32Tensor);
    }
  }

  // do simpleSoftmax
  DoSimpleSoftMax(mm2Tensor, simpleSoftmaxSumTensor, simpleSoftmaxMaxTensor);

  // dropout_0
  LocalTensor<T1> DpB16Tensor_0;
  if (keepProb < 1) {
    MTE2_DropMask(copyBufQue1);
    auto dropMaskTensor_0 = copyBufQue1.DeQue<uint8_t>();
    auto DpB32Tensor_0 = bufQue2.AllocTensor<T2>();
    DoDropout(DpB32Tensor_0, mm2Tensor, dropMaskTensor_0);
    copyBufQue1.FreeTensor(dropMaskTensor_0);
    // 输出到BMM需要保持BF16或FP16类型
    DpB16Tensor_0 = bufQue0.AllocTensor<T1>();
    CastTo16(DpB16Tensor_0, DpB32Tensor_0);
    bufQue2.FreeTensor(DpB32Tensor_0);
  } else {
    // 输出到BMM需要保持BF16或FP16类型
    DpB16Tensor_0 = bufQue0.AllocTensor<T1>();
    CastTo16(DpB16Tensor_0, mm2Tensor);
  }

  bufQue0.EnQue(DpB16Tensor_0);
  bufQue0.DeQue();
  CopyoutWorkspace(mm4InputWorkspaceGm[mm4InputWorkspaceAddr], DpB16Tensor_0);
  mm4InputWorkspaceAddr += baseN;
  bufQue0.FreeTensor(DpB16Tensor_0);

  // do dropout_1
  Mte2ForMm1(s2_o_i);
  auto mm1Tensor = bufQue2.DeQue<T2>();

  if (keepProb < 1) {
    MTE2_DropMask(copyBufQue1);
    auto dropMaskTensor_1 = copyBufQue1.DeQue<uint8_t>();
    DoDropout(mm1Tensor, mm1Tensor, dropMaskTensor_1);
    copyBufQue1.FreeTensor(dropMaskTensor_1);
  }

  // do sub
  DoSub(mm1Tensor, softmaxGradOutTensor);

  // do mul
  DoMul(mm2Tensor, mm1Tensor);
  bufQue2.FreeTensor(mm1Tensor);

  // 输出到BMM需要保持BF16或FP16类型
  auto mm2B16Tensor = bufQue0.AllocTensor<T1>();
  mm2Tensor.SetSize(s1Inner * s2InnerAlign);
  CastTo16(mm2B16Tensor, mm2Tensor);
  bufQue3.FreeTensor(mm2Tensor);
  bufQue0.EnQue(mm2B16Tensor);
  bufQue0.DeQue();
  CopyoutWorkspace(mm3InputWorkspaceGm[mm3InputWorkspaceAddr], mm2B16Tensor);
  mm3InputWorkspaceAddr += baseN;
  bufQue0.FreeTensor(mm2B16Tensor);
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::DoVector(int64_t s2_o_o) {
  mmWidthGetSize = baseNmm;
  for (int64_t s2_o_i = 0; s2_o_i < s2OuterInnerNum; s2_o_i++) {
    if ((s2_o_i == s2OuterInnerNum - 1) && (s2_o_o == s2OuterOuterNum - 1)) {
      s2Inner = s2Tail;
      s2InnerAlign = (s2Tail + dataCopyBlockNum - 1) / dataCopyBlockNum * dataCopyBlockNum;
      s2InnerAlignT2 = (s2Tail + calcBlockNum - 1) / calcBlockNum * calcBlockNum;
    }
    s2Index = s2_o_o * singleNmm + s2_o_i * baseN;
    // main
    if (s2_o_i >= usedSingleNBegin && s2_o_i <= usedSingleNEnd) {
      CopyInOffset();
      InnerT2Process(s2_o_i);
    }
  }
  s2Inner = baseN; // reback init-value that not effect other compute
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::AssureUsefulDataBySingleN(int64_t s2_o_o) {
  // return should calc data in SingleN
  uint32_t begin = 0;
  uint32_t num = 0;
  uint32_t mark = 0;
  for (int64_t s2_o_i = 0; s2_o_i < s2OuterInnerNum; s2_o_i++) {
    bool is_tail = (s2_o_i == s2OuterInnerNum - 1) && (s2_o_o == s2OuterOuterNum - 1);
    s2Inner = is_tail ? s2Tail : baseN;
    s2Index = s2_o_o * singleNmm + s2_o_i * baseN;
    bool exclude = (s2Index + s2Inner <= sparse_s2_start_idx) || (s2Index >= sparse_s2_end_idx);
    if (!(isSparse == 1 && exclude)) {
      num++;
      if (mark == 0) {
        begin = s2_o_i;
        mark = 1;
      }
    }
  }

  usedSingleNNum = num;
  usedSingleNBegin = begin;
  usedSingleNEnd = num == 0 ? begin : begin + num - 1;
  s2Inner = baseN; // reback init-value that not effect other compute
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::CalcProcess() {
  // BMM2 提前发射多份
  // DoVector: 每次UB处理的数据
  for (int64_t s2_o_o = 0; s2_o_o < s2OuterOuterNum; s2_o_o++) {
    bool isLastSingleN = (s2_o_o >= s2OuterOuterNum - 1);
    if (unlikely(isLastSingleN)) {
      processN = processNTail;
      processNv = processNTail;
      s2OuterInnerNum = (processN + baseN - 1) / baseN;
    }
    // calc useful data in SingleN
    s2Index = s2_o_o * singleN;
    AssureUsefulDataBySingleN(s2_o_o);
    if (usedSingleNNum > 0) {
      // do process by useful data
      realProcessN = usedSingleNNum * baseN;
      bool isLastBaseN = usedSingleNEnd == (processNv + baseN - 1) / baseN - 1;
      if (isLastSingleN && isLastBaseN) {
        if (processNv % baseN != 0) {
          realProcessN = realProcessN - baseN + (processNv % baseN);
        }
      }
      // 发当前轮的输入MM
      s2Index = s2_o_o * singleNmm + usedSingleNBegin * baseN;
      MmOffsetTensorB(s2Index, inputMMRighMatrixtAddr);
      SendMatmul2(processM, realProcessN, inputMMLeftMatrixAddr, inputMMRighMatrixtAddr);
      SendMatmul1(processM, realProcessN, inputMMLeftMatrixAddr, inputMMRighMatrixtAddr);
    }

    if ((not_start == 1) && (lastUsedSingleNNum > 0)) {
      // 发上一轮的输出MM, 从循环索引大于0 开始发
      MmOffsetTensorA(lastS1Index, mm3_4_tensor_g_s1_addr);
      MmOffsetTensorB(lastS2Index, mm3_4_tensor_1_s2_addr);
      SendMatmul4(lastRealProcessN, lastS1Inner, lastMm4InputWorkspaceAddr, mm3_4_tensor_g_s1_addr, mm3_4_tensor_1_s2_addr, false);
      SendMatmul3_1(lastRealProcessN, lastS1Inner, lastMm3InputWorkspaceAddr, mm3_4_tensor_1_s2_addr, mm3_4_tensor_g_s1_addr, false);
      SendMatmul3_2(lastRealProcessN, lastS1Inner, lastMm3InputWorkspaceAddr, mm3_4_tensor_g_s1_addr, mm3_4_tensor_1_s2_addr, false);
    }

    if (usedSingleNNum > 0) {
      // do vec process within singleN, note that the vec base block is [baseM, baseN]
      mm3InputWorkspaceAddr = (cal_num % 2) * mm3PangInputWspOffset;
      mm4InputWorkspaceAddr = (cal_num % 2) * mm4PangInputWspOffset;
      lastMm3InputWorkspaceAddr = mm3InputWorkspaceAddr;
      lastMm4InputWorkspaceAddr = mm4InputWorkspaceAddr;
      lastS1Index = s1Index;
      lastS2Index = s2Index;
      lastRealProcessN = realProcessN;
      lastS1Inner = s1Inner;
      DoVector(s2_o_o);
      not_start = 1;
      cal_num += 1;
    }
    lastUsedSingleNNum = usedSingleNNum;
  }
  // 循环外补发最后一次输出MM
  if ((lastUsedSingleNNum > 0) && is_s1s2_last) {
    MmOffsetTensorA(lastS1Index, mm3_4_tensor_g_s1_addr);
    MmOffsetTensorB(lastS2Index, mm3_4_tensor_1_s2_addr);
    SendMatmul4(lastRealProcessN, lastS1Inner, lastMm4InputWorkspaceAddr, mm3_4_tensor_g_s1_addr, mm3_4_tensor_1_s2_addr, need_mm_sync);
    SendMatmul3_1(lastRealProcessN, lastS1Inner, lastMm3InputWorkspaceAddr, mm3_4_tensor_1_s2_addr, mm3_4_tensor_g_s1_addr, need_mm_sync);
    SendMatmul3_2(lastRealProcessN, lastS1Inner, lastMm3InputWorkspaceAddr, mm3_4_tensor_g_s1_addr, mm3_4_tensor_1_s2_addr, need_mm_sync);
  }
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradUs1s2Bbn<T1, T2>::Process() {
  // clean
  auto workLocalClean = bufQue2.AllocTensor<int32_t>();
  SyncAll<false>(syncAtomicCleanGlobal, workLocalClean);
  bufQue2.FreeTensor(workLocalClean);
  bool is_for_bn_last;
  bool is_for_g_last;

  if (blockIdx < usedCoreNum) {
    // beginBNByCore: 当前核处理BN份数据的起始索引
    // endBNByCore：当前核处理BN份数据的结束索引
    // s1Inner,s2Inner: 基本块
    // processN,processM: Matmul SingleM SingleN
    // processNv,processMv: vector SingleM SingleN
    for (int64_t i = beginBNByCore; i < endBNByCore; i++) {
      bIndex = i / dimN2;
      n2Index = i % dimN2;
      if (unlikely(i == endBNByCore - 1)) {
        is_for_bn_last = true;
      } else {
        is_for_bn_last = false;
      }
      for (int64_t gi = 0; gi < dimG; gi++) {
        gIndex = gi;
        processM = baseMmm;
        s1Inner = baseM;
        is_s1s2_last = false;
        not_start = 0;
        cal_num = 0;
        if (unlikely(gi == dimG - 1)) {
          is_for_g_last = true;
        } else {
          is_for_g_last = false;
        }
        for (int64_t s1_o = 0; s1_o < s1OuterNum; s1_o++) {
          if (unlikely(s1_o == s1OuterNum - 1)) {
            processM = processMTail;
            s1Inner = processMTail;
            need_mm_sync = is_for_g_last && is_for_bn_last;
            is_s1s2_last = true;
          } else {
            need_mm_sync = false;
          }
          processN = singleNmm;  // Matmul配比
          processNv = singleN;
          s2Inner = baseN;
          s2InnerAlign = baseN;
          s2InnerAlignT2 = baseN;
          s2OuterInnerNum = (processN + baseN - 1) / baseN;
          s1Index = s1_o * baseM;
          if (isSparse == 1) {
            CalcSparseIdx(bIndex, s1Index, sparse_s2_start_idx, sparse_s2_end_idx);
            if (sparse_s2_start_idx == 0 && sparse_s2_end_idx == 0) {
              continue;
            }
          }
          MmOffsetTensorA(s1Index, dyGmAddr);
          attentionInGmAddr = dyGmAddr;
          inputMMLeftMatrixAddr = dyGmAddr;
          // do softmaxgrad
          DoSoftmaxGrad(softmaxGradOutTensor, softmaxGradOutQue);

          // prepare simplesoftmax input tensors
          CopyInOffsetForSimpleSoftmax();
          MTE2_SFT(simpleSoftmaxQue);
          simpleSoftmaxSumTensor = simpleSoftmaxQue.DeQue<float>();
          simpleSoftmaxMaxTensor = simpleSoftmaxQue.DeQue<float>();

          CalcProcess();

          simpleSoftmaxQue.FreeTensor(simpleSoftmaxSumTensor);
          simpleSoftmaxQue.FreeTensor(simpleSoftmaxMaxTensor);
          softmaxGradOutQue.FreeTensor(softmaxGradOutTensor);
        }
      }
    }
  }

  // cast
  auto workLocalCast = bufQue2.AllocTensor<int32_t>();
  SyncAll<false>(syncCastGlobal, workLocalCast);
  bufQue2.FreeTensor(workLocalCast);
}
#endif