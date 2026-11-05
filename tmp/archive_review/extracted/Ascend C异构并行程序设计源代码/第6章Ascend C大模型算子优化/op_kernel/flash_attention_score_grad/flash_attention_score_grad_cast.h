/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_CAST_
#define FLASH_ATTENTION_SCORE_GRAD_CAST_

#include "kernel_operator.h"

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t SYNC_GLOBAL_WORKSPACE_SIZE_CAST = 1024;
constexpr uint32_t ADDR_ALIGN_SIZE_CAST = 512;

template <typename TypeOut, typename TypeCompute> class FlashAttentionScoreGradCast {
public:
    __aicore__ inline FlashAttentionScoreGradCast() {};

    __aicore__ inline void Init(__gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *workspace,
        const FlashAttentionScoreGradTilingData *__restrict tilingData, TPipe *pipe_in)
    {
        mBlockIdx = GetBlockIdx();

        ordTilingData = tilingData;
        pipe = pipe_in;

        dqGm.SetGlobalBuffer((__gm__ TypeOut *)dq);
        dkGm.SetGlobalBuffer((__gm__ TypeOut *)dk);
        dvGm.SetGlobalBuffer((__gm__ TypeOut *)dv);

        pipe->InitBuffer(inQueue, BUFFER_NUM, tilingData->castTiling.baseOutNum * sizeof(float));
        pipe->InitBuffer(outQueue, BUFFER_NUM, tilingData->castTiling.baseOutNum * sizeof(TypeOut));

        uint64_t buff_offset =
            SYNC_GLOBAL_WORKSPACE_SIZE + GetBlockNum() * GetTaskRation() * ordTilingData->splitCoreParams.outMatResNum * FRONT_COMPUTE_TIMES
            + GetBlockNum() * GetTaskRation() * this->ordTilingData->splitCoreParams.s1OutRange * this->ordTilingData->splitCoreParams.sOut * 32 / sizeof(TypeCompute);
        buff_offset += (GetBlockNum() * GetTaskRation() * CUBE_PRELOAD_BLOCK_NUM *
            this->ordTilingData->splitCoreParams.outMatResNum);
        buff_offset = ((buff_offset + GetBlockNum() * GetTaskRation() * CUBE_PRELOAD_BLOCK_NUM2 *
            this->ordTilingData->splitCoreParams.outMatResNum) * 4 +
            ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;

        dqWorkSpace.SetGlobalBuffer((__gm__ float *)(workspace + buff_offset));

        buff_offset = (buff_offset + ordTilingData->castTiling.allNumQuery * 4 + ADDR_ALIGN_SIZE_CAST) /
            ADDR_ALIGN_SIZE_CAST * ADDR_ALIGN_SIZE_CAST;
        dkWorkSpace.SetGlobalBuffer((__gm__ float *)(workspace + buff_offset));

        buff_offset = (buff_offset + ordTilingData->castTiling.allNumKv * 4 + ADDR_ALIGN_SIZE_CAST) /
            ADDR_ALIGN_SIZE_CAST * ADDR_ALIGN_SIZE_CAST;
        dvWorkSpace.SetGlobalBuffer((__gm__ float *)(workspace + buff_offset));
    }

    __aicore__ inline void Process()
    {
        uint64_t dqOffset = mBlockIdx * ordTilingData->castTiling.singleCoreNumQuery;
        uint64_t dkvOffset = mBlockIdx * ordTilingData->castTiling.singleCoreNumKv;

        if (mBlockIdx == ordTilingData->baseParams.coreNum - 1) {
            // query
            coreNumQuery = ordTilingData->castTiling.tailCoreNumQuery;
            coreLoopTimesQuery = ordTilingData->castTiling.tailCoreLoopTimesQuery;
            lastLoopNumQuery = ordTilingData->castTiling.tailCoreLastLoopNumQuery;

            // K V
            coreNumKv = ordTilingData->castTiling.tailCoreNumKv;
            coreLoopTimesKv = ordTilingData->castTiling.tailCoreLoopTimesKv;
            lastLoopNumKv = ordTilingData->castTiling.tailCoreLastLoopNumKv;
        } else {
            // query
            coreNumQuery = ordTilingData->castTiling.singleCoreNumQuery;
            coreLoopTimesQuery = ordTilingData->castTiling.singleCoreLoopTimesQuery;
            lastLoopNumQuery = ordTilingData->castTiling.lastLoopNumQuery;

            // K V
            coreNumKv = ordTilingData->castTiling.singleCoreNumKv;
            coreLoopTimesKv = ordTilingData->castTiling.singleCoreLoopTimesKv;
            lastLoopNumKv = ordTilingData->castTiling.lastLoopNumKv;
        }

        CastClc(dqGm[dqOffset], dqWorkSpace[dqOffset], coreLoopTimesQuery, lastLoopNumQuery, true);
        CastClc(dkGm[dkvOffset], dkWorkSpace[dkvOffset], coreLoopTimesKv, lastLoopNumKv, true);
        CastClc(dvGm[dkvOffset], dvWorkSpace[dkvOffset], coreLoopTimesKv, lastLoopNumKv, false);
    }

protected:
    TPipe *pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    GlobalTensor<TypeOut> dqGm, dkGm, dvGm;
    GlobalTensor<float> dqWorkSpace, dkWorkSpace, dvWorkSpace;

    const FlashAttentionScoreGradTilingData *__restrict ordTilingData;

    uint64_t mBlockIdx;

    // query
    uint32_t coreNumQuery;
    uint32_t coreLoopTimesQuery;
    uint32_t lastLoopNumQuery;

    // K V
    uint32_t coreNumKv;
    uint32_t coreLoopTimesKv;
    uint32_t lastLoopNumKv;

    __aicore__ inline void CastClc(const GlobalTensor<TypeOut> &dstGm, const GlobalTensor<float> &srcGm,
        const uint32_t &singleCoreLoopTimes, const uint32_t &lastLoopNum, const bool isScale)
    {
        uint32_t castNum = ordTilingData->castTiling.baseOutNum;
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

            if (isScale) {
                Muls(tensorIn, tensorIn, (float)(ordTilingData->baseParams.scaleValue), castNum);
                pipe_barrier(PIPE_V);
            }
            Cast(tensorOut, tensorIn, RoundMode::CAST_ROUND, castNum);

            outQueue.EnQue(tensorOut);
            inQueue.FreeTensor(tensorIn);

            outQueue.DeQue<TypeOut>();
            DataCopy(dstGm[loopOffset], tensorOut, castNum);
            outQueue.FreeTensor(tensorOut);
        }
    }
};

#endif
