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

#ifndef _FLASH_ATTENTION_SCORE_GRAD_U_S1S2_B_BN2_CAST_H_
#define _FLASH_ATTENTION_SCORE_GRAD_U_S1S2_B_BN2_CAST_H_
#include "kernel_operator.h"

template <typename TypeOut>
class FlashAttentionScoreGradUs1s2BbnCast {
 public:
  __aicore__ inline FlashAttentionScoreGradUs1s2BbnCast(){};
  __aicore__ inline void Init(GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace,
                              const FlashAttentionScoreGradTilingDataUs1s2Bbn2* __restrict ordTilingData,
                              TPipe* pipe_in);
  __aicore__ inline void Process();

 protected:
  __aicore__ inline void CastTypeOut(const GlobalTensor<TypeOut>& dstGm, const GlobalTensor<float>& srcGm,
                                     const uint32_t& singleCoreLoopTimes, const uint32_t& lastLoopNum);
  __aicore__ inline void CastTypeAndScaleOut(const GlobalTensor<TypeOut>& dstGm, const GlobalTensor<float>& srcGm,
                                             const uint32_t& singleCoreLoopTimes, const uint32_t& lastLoopNum);
  constexpr static uint32_t BUFFER_NUM = 2;

  GlobalTensor<TypeOut> dqGm;
  GlobalTensor<TypeOut> dkGm;
  GlobalTensor<TypeOut> dvGm;

  GlobalTensor<float> dqWorkspaceGm;
  GlobalTensor<float> dkWorkspaceGm;
  GlobalTensor<float> dvWorkspaceGm;

  TPipe* pipe;
  int64_t blockIdx;
  const FlashAttentionScoreGradTilingDataUs1s2Bbn2* __restrict tilingData;

  TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

  uint32_t dqCoreLoop;
  uint32_t dqLastLoopNum;
  uint32_t kvCoreLoop;
  uint32_t kvLastLoopNum;
  float scaleValue;
};

template <typename TypeOut>
__aicore__ inline void FlashAttentionScoreGradUs1s2BbnCast<TypeOut>::Init(
    GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace,
    const FlashAttentionScoreGradTilingDataUs1s2Bbn2* __restrict ordTilingData, TPipe* pipe_in) {
  blockIdx = GetBlockIdx();
  tilingData = ordTilingData;
  pipe = pipe_in;

  dqGm.SetGlobalBuffer((__gm__ TypeOut*)dq);
  dkGm.SetGlobalBuffer((__gm__ TypeOut*)dk);
  dvGm.SetGlobalBuffer((__gm__ TypeOut*)dv);

  pipe->InitBuffer(inQueue, BUFFER_NUM, tilingData->castParams.inputBufferLen);
  pipe->InitBuffer(outQueue, BUFFER_NUM, tilingData->castParams.outputBufferLen);

  auto coreNum = tilingData->opInfo.usedCoreNum;
  auto syncDropoutLen = tilingData->opInfo.syncDropoutLen;
  auto syncAtomicLen = tilingData->opInfo.syncAtomicLen;
  auto syncCastLen = tilingData->opInfo.syncCastLen;
  auto mm1WorkspaceLen = tilingData->opInfo.mm1WorkspaceLen;
  auto mm2WorkspaceLen = tilingData->opInfo.mm2WorkspaceLen;
  auto mm4InputWorkspaceLen = tilingData->opInfo.mm4InputWorkspaceLen;
  auto mm3InputWorkspaceLen = tilingData->opInfo.mm3InputWorkspaceLen;
  auto dqWorkspaceLen = tilingData->castParams.dqWorkspaceLen;
  auto dkWorkspaceLen = tilingData->castParams.dkWorkspaceLen;
  auto dvWorkspaceLen = tilingData->castParams.dvWorkspaceLen;
  auto dropoutWorkspaceLen = tilingData->dropoutCastParams.dropoutWorkspaceLen;

  auto begin = syncDropoutLen + syncAtomicLen + syncCastLen;
  begin += dropoutWorkspaceLen + (mm1WorkspaceLen + mm2WorkspaceLen) * coreNum;
  begin += (mm4InputWorkspaceLen + mm3InputWorkspaceLen) * coreNum;
  auto dqAddr = begin / sizeof(float);
  auto dkAddr = dqAddr + dqWorkspaceLen / sizeof(float);
  auto dvAddr = dkAddr + dkWorkspaceLen / sizeof(float);
  scaleValue = tilingData->opInfo.scaleValue;

  dqWorkspaceGm.SetGlobalBuffer((__gm__ float*)workspace + dqAddr);
  dkWorkspaceGm.SetGlobalBuffer((__gm__ float*)workspace + dkAddr);
  dvWorkspaceGm.SetGlobalBuffer((__gm__ float*)workspace + dvAddr);
}

template <typename TypeOut>
__aicore__ inline void FlashAttentionScoreGradUs1s2BbnCast<TypeOut>::CastTypeOut(const GlobalTensor<TypeOut>& dstGm,
                                                                                 const GlobalTensor<float>& srcGm,
                                                                                 const uint32_t& singleCoreLoopTimes,
                                                                                 const uint32_t& lastLoopNum) {
  uint32_t castNum = tilingData->castParams.singleUBProcessNum;
  for (uint32_t idx = 0; idx < singleCoreLoopTimes; idx++) {
    uint64_t loopOffset = idx * castNum;
    if (idx == singleCoreLoopTimes - 1) {
      castNum = lastLoopNum;
    }
    LocalTensor<float> tensorIn = inQueue.AllocTensor<float>();
    DataCopy(tensorIn, srcGm[loopOffset], castNum);
    inQueue.EnQue(tensorIn);

    inQueue.DeQue<float>();
    LocalTensor<TypeOut> tensorOut = outQueue.AllocTensor<TypeOut>();

    Cast(tensorOut, tensorIn, RoundMode::CAST_ROUND, castNum);

    outQueue.EnQue(tensorOut);
    inQueue.FreeTensor(tensorIn);

    outQueue.DeQue<TypeOut>();
    DataCopy(dstGm[loopOffset], tensorOut, castNum);
    outQueue.FreeTensor(tensorOut);
  }
}

template <typename TypeOut>
__aicore__ inline void FlashAttentionScoreGradUs1s2BbnCast<TypeOut>::CastTypeAndScaleOut(
    const GlobalTensor<TypeOut>& dstGm, const GlobalTensor<float>& srcGm, const uint32_t& singleCoreLoopTimes,
    const uint32_t& lastLoopNum) {
  uint32_t castNum = tilingData->castParams.singleUBProcessNum;
  for (uint32_t idx = 0; idx < singleCoreLoopTimes; idx++) {
    uint64_t loopOffset = idx * castNum;
    if (idx == singleCoreLoopTimes - 1) {
      castNum = lastLoopNum;
    }
    LocalTensor<float> tensorIn = inQueue.AllocTensor<float>();
    DataCopy(tensorIn, srcGm[loopOffset], castNum);
    inQueue.EnQue(tensorIn);

    inQueue.DeQue<float>();
    LocalTensor<TypeOut> tensorOut = outQueue.AllocTensor<TypeOut>();

    Muls(tensorIn, tensorIn, scaleValue, castNum);
    pipe_barrier(PIPE_V);

    Cast(tensorOut, tensorIn, RoundMode::CAST_ROUND, castNum);

    outQueue.EnQue(tensorOut);
    inQueue.FreeTensor(tensorIn);

    outQueue.DeQue<TypeOut>();
    DataCopy(dstGm[loopOffset], tensorOut, castNum);
    outQueue.FreeTensor(tensorOut);
  }
}

template <typename TypeOut>
__aicore__ inline void FlashAttentionScoreGradUs1s2BbnCast<TypeOut>::Process() {
  if (blockIdx >= tilingData->opInfo.usedCoreNum) {
    return;
  }

  uint64_t dqOffset = blockIdx * tilingData->castParams.dqSingleCoreNum;
  uint64_t dkvOffset = blockIdx * tilingData->castParams.dkvSingleCoreNum;
  if (blockIdx == tilingData->opInfo.usedCoreNum - 1) {
    dqCoreLoop = tilingData->castParams.dqTailCoreLoop;
    dqLastLoopNum = tilingData->castParams.dqTailCoreLastLoopNum;
    kvCoreLoop = tilingData->castParams.kvTailCoreLoop;
    kvLastLoopNum = tilingData->castParams.kvTailCoreLastLoopNum;
  } else {
    dqCoreLoop = tilingData->castParams.dqSingleCoreLoop;
    dqLastLoopNum = tilingData->castParams.dqLastLoopNum;
    kvCoreLoop = tilingData->castParams.kvSingleCoreLoop;
    kvLastLoopNum = tilingData->castParams.kvLastLoopNum;
  }

  CastTypeAndScaleOut(dqGm[dqOffset], dqWorkspaceGm[dqOffset], dqCoreLoop, dqLastLoopNum);
  CastTypeAndScaleOut(dkGm[dkvOffset], dkWorkspaceGm[dkvOffset], kvCoreLoop, kvLastLoopNum);
  CastTypeOut(dvGm[dkvOffset], dvWorkspaceGm[dkvOffset], kvCoreLoop, kvLastLoopNum);
}
#endif