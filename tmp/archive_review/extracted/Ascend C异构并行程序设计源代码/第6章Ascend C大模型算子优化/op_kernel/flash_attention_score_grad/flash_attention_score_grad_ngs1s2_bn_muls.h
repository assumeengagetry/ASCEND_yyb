/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#ifndef _FLASH_ATTENTION_SCORE_GRAD_NGS1S2_BN_MULS_H_
#define _FLASH_ATTENTION_SCORE_GRAD_NGS1S2_BN_MULS_H_
#include "kernel_operator.h"

template <typename TypeOut>
class FlashAttentionScoreGradUngs1s2BbnMuls {
public:
    __aicore__ inline FlashAttentionScoreGradUngs1s2BbnMuls(){};
    __aicore__ inline void Init(GM_ADDR dq, GM_ADDR dk, GM_ADDR workspace,
                                const FlashAttentionScoreGradTilingDataUngs1s2Bbn* __restrict ordTilingData,
                                TPipe* pipe_in);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void DoMuls(const GlobalTensor<TypeOut>& dstGm, const GlobalTensor<float>& srcGm,
                                  const uint32_t& singleCoreLoopTimes, const uint32_t& lastLoopNum);

    constexpr static uint32_t BUFFER_NUM = 1;

    GlobalTensor<TypeOut> dqGm;
    GlobalTensor<TypeOut> dkGm;

    GlobalTensor<float> dqWorkspaceGm;
    GlobalTensor<float> dkWorkspaceGm;

    TPipe* pipe;
    int64_t blockIdx;
    const FlashAttentionScoreGradTilingDataUngs1s2Bbn* __restrict tilingData;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    uint32_t dqCoreLoop;
    uint32_t dqLastLoopNum;
    uint32_t kvCoreLoop;
    uint32_t kvLastLoopNum;
};

template <typename TypeOut>
__aicore__ inline void FlashAttentionScoreGradUngs1s2BbnMuls<TypeOut>::Init(
    GM_ADDR dq, GM_ADDR dk, GM_ADDR workspace,
    const FlashAttentionScoreGradTilingDataUngs1s2Bbn* __restrict ordTilingData, TPipe* pipe_in) {
    blockIdx = GetBlockIdx();
    tilingData = ordTilingData;
    pipe = pipe_in;

    dqGm.SetGlobalBuffer((__gm__ TypeOut*)dq);
    dkGm.SetGlobalBuffer((__gm__ TypeOut*)dk);

    pipe->InitBuffer(inQueue, BUFFER_NUM, tilingData->mulsParams.inputBufferLen);
    pipe->InitBuffer(outQueue, BUFFER_NUM, tilingData->mulsParams.outputBufferLen);

    auto coreNum = tilingData->splitCoreParams.usedCoreNum;
    auto syncLen = tilingData->opInfo.syncLen;
    auto dropoutLen = tilingData->dropoutCastParams.dropoutWorkspaceLen;

    auto mm1WorkspaceLen = tilingData->opInfo.mm1WorkspaceLen;
    auto mm2WorkspaceLen = tilingData->opInfo.mm2WorkspaceLen;
    auto dqWorkspaceLen = tilingData->mulsParams.dqWorkspaceLen;
    auto dkWorkspaceLen = tilingData->mulsParams.dkWorkspaceLen;

    auto beginAddr = syncLen + dropoutLen + mm1WorkspaceLen + mm2WorkspaceLen;
    auto dqAddr = beginAddr / sizeof(float);
    auto dkAddr = dqAddr + dqWorkspaceLen / sizeof(float);
    dqWorkspaceGm.SetGlobalBuffer((__gm__ float*)workspace + dqAddr);
    dkWorkspaceGm.SetGlobalBuffer((__gm__ float*)workspace + dkAddr);
}

template <typename TypeOut>
__aicore__ inline void FlashAttentionScoreGradUngs1s2BbnMuls<TypeOut>::Process() {
    if (g_coreType == AIV && blockIdx >= tilingData->splitCoreParams.usedCoreNum) {
        return;
    }

    uint64_t dqOffset = blockIdx * tilingData->mulsParams.dqSingleCoreNum;
    uint64_t dkvOffset = blockIdx * tilingData->mulsParams.dkvSingleCoreNum;
    if (blockIdx == tilingData->splitCoreParams.usedCoreNum - 1) {
        dqCoreLoop = tilingData->mulsParams.dqTailCoreLoop;
        dqLastLoopNum = tilingData->mulsParams.dqTailCoreLastLoopNum;
        kvCoreLoop = tilingData->mulsParams.kvTailCoreLoop;
        kvLastLoopNum = tilingData->mulsParams.kvTailCoreLastLoopNum;
    } else {
        dqCoreLoop = tilingData->mulsParams.dqSingleCoreLoop;
        dqLastLoopNum = tilingData->mulsParams.dqLastLoopNum;
        kvCoreLoop = tilingData->mulsParams.kvSingleCoreLoop;
        kvLastLoopNum = tilingData->mulsParams.kvLastLoopNum;
    }

    DoMuls(dqGm[dqOffset], dqWorkspaceGm[dqOffset], dqCoreLoop, dqLastLoopNum);
    DoMuls(dkGm[dkvOffset], dkWorkspaceGm[dkvOffset], kvCoreLoop, kvLastLoopNum);
}

template <typename TypeOut>
__aicore__ inline void FlashAttentionScoreGradUngs1s2BbnMuls<TypeOut>::DoMuls(const GlobalTensor<TypeOut>& dstGm,
                                                                              const GlobalTensor<float>& srcGm,
                                                                              const uint32_t& singleCoreLoopTimes,
                                                                              const uint32_t& lastLoopNum) {
    uint32_t mulsNum = tilingData->mulsParams.singleUBProcessNum;
    float scaleValue = tilingData->opInfo.scaleValue;
    for (uint32_t idx = 0; idx < singleCoreLoopTimes; ++idx) {
        uint64_t loopOffset = idx * mulsNum;
        if (idx == singleCoreLoopTimes - 1) {
            mulsNum = lastLoopNum;
        }
        LocalTensor<float> tensorIn = inQueue.AllocTensor<float>();
        DataCopy(tensorIn, srcGm[loopOffset], mulsNum);
        inQueue.EnQue(tensorIn);
        inQueue.DeQue<float>();
        Muls(tensorIn, tensorIn, (float)scaleValue, mulsNum);
        pipe_barrier(PIPE_V);
        LocalTensor<TypeOut> tensorOut = outQueue.AllocTensor<TypeOut>();
        Cast(tensorOut, tensorIn, RoundMode::CAST_ROUND, mulsNum);
        outQueue.EnQue(tensorOut);
        inQueue.FreeTensor(tensorIn);
        outQueue.DeQue<TypeOut>();
        DataCopy(dstGm[loopOffset], tensorOut, mulsNum);
        outQueue.FreeTensor(tensorOut);
    }
}
#endif