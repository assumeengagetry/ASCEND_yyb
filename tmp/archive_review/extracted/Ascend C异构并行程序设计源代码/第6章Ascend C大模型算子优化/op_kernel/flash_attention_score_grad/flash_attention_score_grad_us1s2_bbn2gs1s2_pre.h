//
// Created by root on 9/18/23.
//

#ifndef _FLASH_ATTENTION_SCORE_GRAD_FLASH_ATTENTION_SCORE_GRAD_S1S2_BNGS1S2_PRE_KERNEL_H_
#define _FLASH_ATTENTION_SCORE_GRAD_FLASH_ATTENTION_SCORE_GRAD_S1S2_BNGS1S2_PRE_KERNEL_H_
#include "kernel_operator.h"

using namespace AscendC;
template <typename T1, typename T2>
class FlashAttentionScoreGradS1S2BNGS1S2Pre {
 public:
  __aicore__ inline FlashAttentionScoreGradS1S2BNGS1S2Pre(){};
  __aicore__ inline void Init(__gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv, __gm__ uint8_t* drop_mask,
                              __gm__ uint8_t* workspace,
                              const FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2* __restrict ordTilingData,
                              TPipe* pipe_in);
  __aicore__ inline void Process();
  __aicore__ inline void SyncALLCores();

  TPipe* pipe;
  TQue<QuePosition::VECIN, 1> inQueue;
  TQue<QuePosition::VECOUT, 1> outQueue;
  TBuf<> vecClc1;
  TBuf<> vecClc2;

  GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;
  GlobalTensor<uint8_t> maskWorkSpaceGm;
  GlobalTensor<uint8_t> drop_maskGm;
  GlobalTensor<int32_t> syncGlobal;

  const FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2* __restrict TilingData;
  constexpr static uint32_t SYNC_GLOBAL_WORKSPACE_SIZE = 16 * 1024;
  constexpr static uint32_t ADDR_ALIGN_SIZE = 512;

  uint32_t usedCoreNum;
  uint32_t cBlockIdx;
  // query
  uint32_t ubBaseSize;
  uint32_t qPreBlockFactor;
  uint32_t qPreBlockTotal;
  uint32_t qPreBlockTail;
  uint32_t qPostBlockTotal;
  uint32_t kvPreBlockFactor;
  uint32_t kvPreBlockTotal;
  uint32_t kvPreBlockTail;
  uint32_t kvPostBlockTotal;
  uint32_t maskPreBlockFactor;
  uint32_t maskPreBlockTotal;
  uint32_t maskPreTailNum;
  uint32_t maskPreBaseNum;
  bool isDrop;
};

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradS1S2BNGS1S2Pre<T1, T2>::Init(
    __gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv, __gm__ uint8_t* drop_mask, __gm__ uint8_t* workspace,
    const FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2* __restrict orgTilingData, TPipe* pipe_in) {
  cBlockIdx = GetBlockIdx();

  TilingData = orgTilingData;
  pipe = pipe_in;

  // tiling_data
  usedCoreNum = TilingData->s1s2BNGS1S2BaseParams.coreNum;
  qPreBlockFactor = TilingData->preTilingData.qPreBlockFactor;
  qPreBlockTotal = TilingData->preTilingData.qPreBlockTotal;
  qPreBlockTail = TilingData->preTilingData.qPreBlockTail;
  qPostBlockTotal = TilingData->postTilingData.qPostBlockTotal;
  kvPreBlockFactor = TilingData->preTilingData.kvPreBlockFactor;
  kvPreBlockTotal = TilingData->preTilingData.kvPreBlockTotal;
  kvPreBlockTail = TilingData->preTilingData.kvPreBlockTail;
  kvPostBlockTotal = TilingData->postTilingData.kvPostBlockTotal;
  maskPreBlockFactor = TilingData->preTilingData.maskPreBlockFactor;
  maskPreBlockTotal = TilingData->preTilingData.maskPreBlockTotal;
  maskPreTailNum = TilingData->preTilingData.maskPreTailNum;
  maskPreBaseNum = TilingData->preTilingData.maskPreBaseNum;
  ubBaseSize = maskPreBaseNum * sizeof(half);

  isDrop = false;
  if (TilingData->s1s2BNGS1S2BaseParams.keepProb < 1 && TilingData->s1s2BNGS1S2BaseParams.s2 % 8 != 0) {
    isDrop = true;
  }

  drop_maskGm.SetGlobalBuffer((__gm__ uint8_t*)drop_mask);

  syncGlobal.SetGlobalBuffer((__gm__ int32_t*)workspace);
  uint64_t workspaceOffsets = SYNC_GLOBAL_WORKSPACE_SIZE;
  dqWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace + workspaceOffsets / sizeof(T2));
  workspaceOffsets =
      (workspaceOffsets + qPostBlockTotal * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
  dkWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace + workspaceOffsets / sizeof(T2));
  workspaceOffsets =
      (workspaceOffsets + kvPostBlockTotal * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
  dvWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace + workspaceOffsets / sizeof(T2));
  if (isDrop) {
    workspaceOffsets =
        (workspaceOffsets + kvPostBlockTotal * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    maskWorkSpaceGm.SetGlobalBuffer((__gm__ uint8_t*)workspace + workspaceOffsets);
  }
  
  InitOutput<int32_t>(syncGlobal[GetBlockIdx() * 8], 8, 0);

  uint64_t initdqSize = cBlockIdx == qPreBlockTotal - 1 ? qPreBlockTail : qPreBlockFactor;
  uint64_t dqOffset = cBlockIdx * qPreBlockFactor;
  uint64_t initdkSize = cBlockIdx == kvPreBlockTotal - 1 ? kvPreBlockTail : kvPreBlockFactor;
  uint64_t dkvOffset = cBlockIdx * kvPreBlockFactor;
  if (cBlockIdx < usedCoreNum) {
    InitOutput<float>(dkWorkSpaceGm[dkvOffset], initdkSize, 0);
    InitOutput<float>(dvWorkSpaceGm[dkvOffset], initdkSize, 0);
    InitOutput<float>(dqWorkSpaceGm[dqOffset], initdqSize, 0);
  }

  pipe->InitBuffer(inQueue, 1, ubBaseSize);
  pipe->InitBuffer(outQueue, 1, ubBaseSize);
  pipe->InitBuffer(vecClc1, ubBaseSize);
  pipe->InitBuffer(vecClc2, ubBaseSize);
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradS1S2BNGS1S2Pre<T1, T2>::Process() {
  if (isDrop) {
    // drop mask
    LocalTensor<half> vecClcBuffer1 = vecClc1.Get<half>();
    LocalTensor<half> vecClcBuffer2 = vecClc2.Get<half>();

    Duplicate<half>(vecClcBuffer1, (half)1.0, (int32_t)maskPreBaseNum);
    uint64_t maskBegin = cBlockIdx * maskPreBlockFactor;
    bool isLast = cBlockIdx == maskPreBlockTotal - 1;
    uint64_t maskEnd = isLast ? maskBegin + maskPreTailNum : maskBegin + maskPreBlockFactor;
    uint32_t baseTail = maskPreTailNum % maskPreBaseNum == 0 ? maskPreBaseNum : maskPreTailNum % maskPreBaseNum;
    pipe_barrier(PIPE_V);
    for (uint64_t i = maskBegin; i < maskEnd; i = i + maskPreBlockFactor) {
      LocalTensor<uint8_t> vecIn = inQueue.AllocTensor<uint8_t>();
      LocalTensor<uint8_t> vecOut = outQueue.AllocTensor<uint8_t>();
      uint32_t dataSize = i + maskPreBaseNum < maskEnd ? maskPreBaseNum : baseTail;
      DataCopy(vecIn, drop_maskGm[i / 8], dataSize / 8);
      inQueue.EnQue(vecIn);
      inQueue.DeQue<uint8_t>();
      Select(vecClcBuffer2, vecIn, vecClcBuffer1, (half)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, dataSize);
      pipe_barrier(PIPE_V);
      Cast(vecOut, vecClcBuffer2, RoundMode::CAST_ROUND, dataSize);
      outQueue.EnQue(vecOut);
      outQueue.DeQue<uint8_t>();
      DataCopy(maskWorkSpaceGm[i], vecOut, dataSize);
      outQueue.FreeTensor(vecOut);
      inQueue.FreeTensor(vecIn);
    }
  }
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradS1S2BNGS1S2Pre<T1, T2>::SyncALLCores() {
  SyncAll();
}
#endif  // _FLASH_ATTENTION_SCORE_GRAD_FLASH_ATTENTION_SCORE_GRAD_S1S2_BNGS1S2_PRE_KERNEL_H_
