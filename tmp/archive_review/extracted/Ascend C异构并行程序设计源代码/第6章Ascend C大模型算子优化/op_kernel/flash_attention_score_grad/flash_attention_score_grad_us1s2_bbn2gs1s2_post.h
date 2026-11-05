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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either exPostss or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _FLASH_ATTENTION_SCORE_GRAD_FLASH_ATTENTION_SCORE_GRAD_S1S2_BNGS1S2_POST_KERNEL_H_
#define _FLASH_ATTENTION_SCORE_GRAD_FLASH_ATTENTION_SCORE_GRAD_S1S2_BNGS1S2_POST_KERNEL_H_
#include "kernel_operator.h"

template <typename T1, typename T2>
class FlashAttentionScoreGradS1S2BNGS1S2Post {
 public:
  __aicore__ inline FlashAttentionScoreGradS1S2BNGS1S2Post(){};
  __aicore__ inline void Init(__gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv, __gm__ uint8_t* workspace,
                              const FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2* __restrict ordTilingData,
                              TPipe* pipe_in);
  __aicore__ inline void Process();

  TPipe* pipe;
  TQue<QuePosition::VECIN, 1> inQueue;
  TQue<QuePosition::VECOUT, 1> outQueue;
  TBuf<> vecClc1;
  TBuf<> vecClc2;

  GlobalTensor<T1> dqGm, dkGm, dvGm;
  // input
  GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;

  const FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2* __restrict tilingData;
  constexpr static uint32_t SYNC_GLOBAL_WORKSPACE_SIZE = 16 * 1024;
  constexpr static uint32_t ADDR_ALIGN_SIZE = 512;

  uint32_t usedCoreNum;
  uint32_t cBlockIdx;
  // query
  uint32_t ubBaseSize;
  uint32_t qPostBlockFactor;
  uint32_t qPostBlockTotal;
  uint32_t qPostBaseNum;
  uint32_t qPostTailNum;
  uint32_t qSizeAlign;
  uint32_t kvPostBlockFactor;
  uint32_t kvPostBlockTotal;
  uint32_t kvPostBaseNum;
  uint32_t kvPostTailNum;
  uint32_t kvSizeAlign;
};

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradS1S2BNGS1S2Post<T1, T2>::Init(
    __gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv, __gm__ uint8_t* workspace,
    const FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2* __restrict ordTilingData, TPipe* pipe_in) {
  cBlockIdx = GetBlockIdx();

  tilingData = ordTilingData;
  pipe = pipe_in;

  dqGm.SetGlobalBuffer((__gm__ T1*)dq);
  dkGm.SetGlobalBuffer((__gm__ T1*)dk);
  dvGm.SetGlobalBuffer((__gm__ T1*)dv);

  // tiling_data
  usedCoreNum = tilingData->s1s2BNGS1S2BaseParams.coreNum;
  ubBaseSize = tilingData->postTilingData.postUbBaseSize;
  qPostBlockFactor = tilingData->postTilingData.qPostBlockFactor;
  qPostBlockTotal = tilingData->postTilingData.qPostBlockTotal;
  qPostBaseNum = tilingData->postTilingData.qPostBaseNum;
  qPostTailNum = tilingData->postTilingData.qPostTailNum;
  qSizeAlign = tilingData->postTilingData.qSizeAlign;
  kvPostBlockFactor = tilingData->postTilingData.kvPostBlockFactor;
  kvPostBlockTotal = tilingData->postTilingData.kvPostBlockTotal;
  kvPostBaseNum = tilingData->postTilingData.kvPostBaseNum;
  kvPostTailNum = tilingData->postTilingData.kvPostTailNum;
  kvSizeAlign = tilingData->postTilingData.kvSizeAlign;

  uint64_t qPreBlockTotal = tilingData->postTilingData.qPostBlockTotal;
  uint64_t kvPreBlockTotal = tilingData->postTilingData.kvPostBlockTotal;

  uint64_t workspaceOffsets = SYNC_GLOBAL_WORKSPACE_SIZE;
  dqWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace + workspaceOffsets / sizeof(T2));
  workspaceOffsets =
      (workspaceOffsets + qPreBlockTotal * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
  dkWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace + workspaceOffsets / sizeof(T2));
  workspaceOffsets =
      (workspaceOffsets + kvPreBlockTotal * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
  dvWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace + workspaceOffsets / sizeof(T2));

  pipe->InitBuffer(inQueue, 1, ubBaseSize * 2);
  pipe->InitBuffer(outQueue, 1, ubBaseSize);
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionScoreGradS1S2BNGS1S2Post<T1, T2>::Process() {
  bool isLastCore = (cBlockIdx == (usedCoreNum - 1));
  // init q
  uint64_t qBegin = cBlockIdx * qPostBlockFactor * qPostBaseNum;
  uint64_t qEnd = (cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum;

  if (((cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum) > qPostBlockTotal) {
    qEnd = qPostBlockTotal;
  }
  for (uint64_t i = qBegin; i < qEnd; i = i + qPostBaseNum) {
    LocalTensor<float> vecIn = inQueue.AllocTensor<float>();
    LocalTensor<T1> vecOut = outQueue.AllocTensor<T1>();
    uint32_t dataSize = i + qPostBaseNum < qPostBlockTotal ? qPostBaseNum : qPostTailNum;
    DataCopy(vecIn, dqWorkSpaceGm[i], dataSize);
    inQueue.EnQue(vecIn);
    inQueue.DeQue<float>();

    Muls(vecIn, vecIn, (float)tilingData->s1s2BNGS1S2BaseParams.scaleValue, dataSize);
    pipe_barrier(PIPE_V);
    Cast(vecOut, vecIn, RoundMode::CAST_ROUND, dataSize);
    outQueue.EnQue(vecOut);
    outQueue.DeQue<T1>();
    DataCopy(dqGm[i], vecOut, dataSize);
    inQueue.FreeTensor(vecIn);
    outQueue.FreeTensor(vecOut);
  }
  pipe_barrier(PIPE_ALL);
  // init k
  uint64_t kvBegin = cBlockIdx * kvPostBlockFactor * kvPostBaseNum;
  uint64_t kvEnd = (cBlockIdx + 1) * kvPostBlockFactor * kvPostBaseNum;
  if (((cBlockIdx + 1) * kvPostBlockFactor * kvPostBaseNum) > kvPostBlockTotal) {
    kvEnd = kvPostBlockTotal;
  }

  for (uint64_t i = kvBegin; i < kvEnd; i = i + kvPostBaseNum) {
    LocalTensor<float> vecIn = inQueue.AllocTensor<float>();
    LocalTensor<T1> vecOut = outQueue.AllocTensor<T1>();
    uint32_t dataSize = i + kvPostBaseNum < kvPostBlockTotal ? kvPostBaseNum : kvPostTailNum;
    DataCopy(vecIn, dkWorkSpaceGm[i], dataSize);
    inQueue.EnQue(vecIn);
    inQueue.DeQue<float>();
    Muls(vecIn, vecIn, (float)tilingData->s1s2BNGS1S2BaseParams.scaleValue, dataSize);
    pipe_barrier(PIPE_V);
    Cast(vecOut, vecIn, RoundMode::CAST_ROUND, dataSize);
    outQueue.EnQue(vecOut);
    outQueue.DeQue<T1>();
    DataCopy(dkGm[i], vecOut, dataSize);
    inQueue.FreeTensor(vecIn);
    outQueue.FreeTensor(vecOut);
  }
  pipe_barrier(PIPE_ALL);

  // init v
  for (uint64_t i = kvBegin; i < kvEnd; i = i + kvPostBaseNum) {
    LocalTensor<float> vecIn = inQueue.AllocTensor<float>();
    LocalTensor<T1> vecOut = outQueue.AllocTensor<T1>();
    uint32_t dataSize = i + kvPostBaseNum < kvPostBlockTotal ? kvPostBaseNum : kvPostTailNum;
    DataCopy(vecIn, dvWorkSpaceGm[i], dataSize);
    inQueue.EnQue(vecIn);
    inQueue.DeQue<float>();
    Cast(vecOut, vecIn, RoundMode::CAST_ROUND, dataSize);
    outQueue.EnQue(vecOut);
    outQueue.DeQue<T1>();
    DataCopy(dvGm[i], vecOut, dataSize);
    inQueue.FreeTensor(vecIn);
    outQueue.FreeTensor(vecOut);
  }
}

#endif  // _FLASH_ATTENTION_SCORE_GRAD_FLASH_ATTENTION_SCORE_GRAD_S1S2_BNGS1S2_POST_KERNEL_H_
