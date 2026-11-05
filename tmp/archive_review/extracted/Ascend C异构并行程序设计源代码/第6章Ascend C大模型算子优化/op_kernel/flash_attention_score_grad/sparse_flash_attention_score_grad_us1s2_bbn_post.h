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

#ifndef _SPARSE_FLASH_ATTENTION_SCORE_GRAD_US1S2_BBN_POST_H_
#define _SPARSE_FLASH_ATTENTION_SCORE_GRAD_US1S2_BBN_POST_H_
#include "kernel_operator.h"

template <typename OUT_TYPE, typename CALC_TYPE, const bool HIGH_PRECISE>
class FlashAttentionScoreGradUs1s2BbnPost {
public:
    __aicore__ inline FlashAttentionScoreGradUs1s2BbnPost(){};
    __aicore__ inline void Init(GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace,
                                const FlashAttentionScoreGradUs1s2BbnTilingData* __restrict ordTilingData,
                                TPipe* pipe_in, const uint64_t bmm31Offset, const uint64_t bmm32Offset,
                                const uint64_t bmm4Offset);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void CastTypeOut(const GlobalTensor<OUT_TYPE>& dstGm, const GlobalTensor<CALC_TYPE>& srcGm,
                                        const uint32_t& singleCoreLoopTimes, const uint32_t& lastLoopNum);
    __aicore__ inline void CastTypeAndScaleOut(const GlobalTensor<OUT_TYPE>& dstGm, const GlobalTensor<CALC_TYPE>& srcGm,
                                                const uint32_t& singleCoreLoopTimes, const uint32_t& lastLoopNum);
    __aicore__ inline void ScaleOut(const GlobalTensor<OUT_TYPE>& dstGm, const GlobalTensor<CALC_TYPE>& srcGm,
                                        const uint32_t& singleCoreLoopTimes, const uint32_t& lastLoopNum);

    constexpr static uint32_t BUFFER_NUM = 2;

    GlobalTensor<OUT_TYPE> dqGm;
    GlobalTensor<OUT_TYPE> dkGm;
    GlobalTensor<OUT_TYPE> dvGm;

    GlobalTensor<CALC_TYPE> dqWorkspaceGm;
    GlobalTensor<CALC_TYPE> dkWorkspaceGm;
    GlobalTensor<CALC_TYPE> dvWorkspaceGm;

    TPipe* pipe;
    int64_t blockIdx;
    const FlashAttentionScoreGradUs1s2BbnTilingData* __restrict tilingData;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    uint32_t dqCoreLoop;
    uint32_t dqLastLoopNum;
    uint32_t kvCoreLoop;
    uint32_t kvLastLoopNum;
    CALC_TYPE scaleValue;
};

template <typename OUT_TYPE, typename CALC_TYPE, const bool HIGH_PRECISE>
__aicore__ inline void FlashAttentionScoreGradUs1s2BbnPost<OUT_TYPE, CALC_TYPE, HIGH_PRECISE>::Init(
    GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace,
    const FlashAttentionScoreGradUs1s2BbnTilingData* __restrict ordTilingData, TPipe* pipe_in,
    const uint64_t bmm31Offset, const uint64_t bmm32Offset, const uint64_t bmm4Offset) {
    blockIdx = GetBlockIdx();
    tilingData = ordTilingData;
    pipe = pipe_in;

    dqGm.SetGlobalBuffer((__gm__ OUT_TYPE*)dq);
    dkGm.SetGlobalBuffer((__gm__ OUT_TYPE*)dk);
    dvGm.SetGlobalBuffer((__gm__ OUT_TYPE*)dv);

    pipe->InitBuffer(inQueue, BUFFER_NUM, tilingData->postOpTilingParams.inputBufferLen);
    pipe->InitBuffer(outQueue, BUFFER_NUM, tilingData->postOpTilingParams.outputBufferLen);

    // calc workspace offset
    scaleValue = (CALC_TYPE)tilingData->baseParams.scaleValue;

    dqWorkspaceGm.SetGlobalBuffer(((__gm__ CALC_TYPE*)workspace + bmm31Offset));
    dkWorkspaceGm.SetGlobalBuffer(((__gm__ CALC_TYPE*)workspace + bmm32Offset));
    dvWorkspaceGm.SetGlobalBuffer(((__gm__ CALC_TYPE*)workspace + bmm4Offset));
}

template <typename OUT_TYPE, typename CALC_TYPE, const bool HIGH_PRECISE>
__aicore__ inline void FlashAttentionScoreGradUs1s2BbnPost<OUT_TYPE, CALC_TYPE, HIGH_PRECISE>::CastTypeOut(const GlobalTensor<OUT_TYPE>& dstGm,
                                                                                const GlobalTensor<CALC_TYPE>& srcGm,
                                                                                const uint32_t& singleCoreLoopTimes,
                                                                                const uint32_t& lastLoopNum) {
  uint32_t castNum = tilingData->postOpTilingParams.singleUBProcessNum;
  for (uint32_t idx = 0; idx < singleCoreLoopTimes; idx++) {
    uint64_t loopOffset = idx * castNum;
    if (idx == singleCoreLoopTimes - 1) {
      castNum = lastLoopNum;
    }
    LocalTensor<CALC_TYPE> tensorIn = inQueue.AllocTensor<CALC_TYPE>();
    DataCopy(tensorIn, srcGm[loopOffset], castNum);
    inQueue.EnQue(tensorIn);

    inQueue.DeQue<float>();
    LocalTensor<OUT_TYPE> tensorOut = outQueue.AllocTensor<OUT_TYPE>();

    Cast(tensorOut, tensorIn, RoundMode::CAST_ROUND, castNum);

    outQueue.EnQue(tensorOut);
    inQueue.FreeTensor(tensorIn);

    outQueue.DeQue<OUT_TYPE>();
    DataCopy(dstGm[loopOffset], tensorOut, castNum);
    outQueue.FreeTensor(tensorOut);
  }
}

template <typename OUT_TYPE, typename CALC_TYPE, const bool HIGH_PRECISE>
__aicore__ inline void FlashAttentionScoreGradUs1s2BbnPost<OUT_TYPE, CALC_TYPE, HIGH_PRECISE>::CastTypeAndScaleOut(
    const GlobalTensor<OUT_TYPE>& dstGm, const GlobalTensor<CALC_TYPE>& srcGm, const uint32_t& singleCoreLoopTimes,
    const uint32_t& lastLoopNum) {
  uint32_t castNum = tilingData->postOpTilingParams.singleUBProcessNum;
  for (uint32_t idx = 0; idx < singleCoreLoopTimes; idx++) {
    uint64_t loopOffset = idx * castNum;
    if (idx == singleCoreLoopTimes - 1) {
      castNum = lastLoopNum;
    }

    LocalTensor<CALC_TYPE> tensorIn = inQueue.AllocTensor<CALC_TYPE>();
    DataCopy(tensorIn, srcGm[loopOffset], castNum);

    inQueue.EnQue(tensorIn);

    inQueue.DeQue<CALC_TYPE>();
    LocalTensor<OUT_TYPE> tensorOut = outQueue.AllocTensor<OUT_TYPE>();

    Muls(tensorIn, tensorIn, scaleValue, castNum);
    pipe_barrier(PIPE_V);
    Cast(tensorOut, tensorIn, RoundMode::CAST_ROUND, castNum);

    outQueue.EnQue(tensorOut);
    inQueue.FreeTensor(tensorIn);

    outQueue.DeQue<OUT_TYPE>();
    DataCopy(dstGm[loopOffset], tensorOut, castNum);
    outQueue.FreeTensor(tensorOut);
  }
}

template <typename OUT_TYPE, typename CALC_TYPE, const bool HIGH_PRECISE>
__aicore__ inline void FlashAttentionScoreGradUs1s2BbnPost<OUT_TYPE, CALC_TYPE, HIGH_PRECISE>::ScaleOut(
    const GlobalTensor<OUT_TYPE>& dstGm, const GlobalTensor<CALC_TYPE>& srcGm, const uint32_t& singleCoreLoopTimes,
    const uint32_t& lastLoopNum) {
    uint32_t castNum = tilingData->postOpTilingParams.singleUBProcessNum;
    for (uint32_t idx = 0; idx < singleCoreLoopTimes; idx++) {
        uint64_t loopOffset = idx * castNum;
        if (idx == singleCoreLoopTimes - 1) {
            castNum = lastLoopNum;
        }
        LocalTensor<CALC_TYPE> tensorIn = inQueue.AllocTensor<CALC_TYPE>();
        DataCopy(tensorIn, srcGm[loopOffset], castNum);
        inQueue.EnQue(tensorIn);

        inQueue.DeQue<CALC_TYPE>();
        LocalTensor<OUT_TYPE> tensorOut = outQueue.AllocTensor<OUT_TYPE>();

        Muls(tensorOut, tensorIn, scaleValue, castNum);

        outQueue.EnQue(tensorOut);
        inQueue.FreeTensor(tensorIn);

        outQueue.DeQue<OUT_TYPE>();
        DataCopy(dstGm[loopOffset], tensorOut, castNum);
        outQueue.FreeTensor(tensorOut);
    }
}

template <typename OUT_TYPE, typename CALC_TYPE, const bool HIGH_PRECISE>
__aicore__ inline void FlashAttentionScoreGradUs1s2BbnPost<OUT_TYPE, CALC_TYPE, HIGH_PRECISE>::Process() {
    if (g_coreType == AIV && blockIdx >= tilingData->baseParams.coreNum) {
        return;
    }

    uint64_t dqOffset = blockIdx * tilingData->postOpTilingParams.dqSingleCoreNum;
    uint64_t dkvOffset = blockIdx * tilingData->postOpTilingParams.dkvSingleCoreNum;
    if (blockIdx == tilingData->baseParams.coreNum - 1) {
        dqCoreLoop = tilingData->postOpTilingParams.dqTailCoreLoop;
        dqLastLoopNum = tilingData->postOpTilingParams.dqTailCoreLastLoopNum;
        kvCoreLoop = tilingData->postOpTilingParams.kvTailCoreLoop;
        kvLastLoopNum = tilingData->postOpTilingParams.kvTailCoreLastLoopNum;
    } else {
        dqCoreLoop = tilingData->postOpTilingParams.dqSingleCoreLoop;
        dqLastLoopNum = tilingData->postOpTilingParams.dqLastLoopNum;
        kvCoreLoop = tilingData->postOpTilingParams.kvSingleCoreLoop;
        kvLastLoopNum = tilingData->postOpTilingParams.kvLastLoopNum;
    }

    if constexpr (HIGH_PRECISE) {
        CastTypeOut(dvGm[dkvOffset], dvWorkspaceGm[dkvOffset], kvCoreLoop, kvLastLoopNum);
        CastTypeAndScaleOut(dqGm[dqOffset], dqWorkspaceGm[dqOffset], dqCoreLoop, dqLastLoopNum);
        CastTypeAndScaleOut(dkGm[dkvOffset], dkWorkspaceGm[dkvOffset], kvCoreLoop, kvLastLoopNum);
    } else {
        ScaleOut(dqGm[dqOffset], dqWorkspaceGm[dqOffset], dqCoreLoop, dqLastLoopNum);
        ScaleOut(dkGm[dkvOffset], dkWorkspaceGm[dkvOffset], kvCoreLoop, kvLastLoopNum);
    }
}

#endif
