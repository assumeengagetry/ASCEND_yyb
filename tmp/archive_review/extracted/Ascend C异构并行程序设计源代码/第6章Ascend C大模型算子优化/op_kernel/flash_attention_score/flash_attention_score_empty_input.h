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

#ifndef FLASH_ATTENTION_SCORE_EMPTY_INPUT_H
#define FLASH_ATTENTION_SCORE_EMPTY_INPUT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"


template <typename INPUT_T> class FlashAttentionScoreEmptyInput {
public:
    __aicore__ inline FlashAttentionScoreEmptyInput() {};
    __aicore__ inline void Init(__gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *attentionOut,
        const FlashAttentionScoreTilingData *__restrict tiling);
    __aicore__ inline void Process();

protected:
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    GlobalTensor<INPUT_T> attentionOutGm;
    uint32_t tmpBlockIdx;
    const FlashAttentionScoreTilingData *__restrict tilingData;
    __aicore__ inline void ComputeEachCore();
};

template <typename INPUT_T>
__aicore__ inline void FlashAttentionScoreEmptyInput<INPUT_T>::Init(__gm__ uint8_t *softmaxMax,
    __gm__ uint8_t *softmaxSum, __gm__ uint8_t *attentionOut, const FlashAttentionScoreTilingData *__restrict tiling)
{
    tmpBlockIdx = GetBlockIdx();
    tilingData = tiling;
    softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
    softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);
    attentionOutGm.SetGlobalBuffer((__gm__ INPUT_T *)attentionOut);
}

template <typename INPUT_T> __aicore__ inline void FlashAttentionScoreEmptyInput<INPUT_T>::ComputeEachCore()
{
    uint32_t coreNum = tilingData->emptyInputTilingData.coreNum;
    uint32_t attentionOutFormerNum = tilingData->emptyInputTilingData.attentionOutFormerNum;
    uint32_t attentionOutTailNum = tilingData->emptyInputTilingData.attentionOutTailNum;
    uint32_t softmaxMaxFormerNum = tilingData->emptyInputTilingData.softmaxMaxFormerNum;
    uint32_t softmaxMaxTailNum = tilingData->emptyInputTilingData.softmaxMaxTailNum;
    uint64_t attentionOutSingleCoreDataSize = tilingData->emptyInputTilingData.attentionOutSingleCoreDataSize;
    uint64_t attentionOutTailCoreDataSize = tilingData->emptyInputTilingData.attentionOutTailCoreDataSize;
    uint64_t softmaxMaxSingleCoreDataSize = tilingData->emptyInputTilingData.softmaxMaxSingleCoreDataSize;
    uint64_t softmaxMaxTailCoreDataSize = tilingData->emptyInputTilingData.softmaxMaxTailCoreDataSize;
    uint64_t attentionOutLastCoreDataSize = tilingData->emptyInputTilingData.attentionOutLastCoreDataSize;
    uint64_t attentionOutLastCoreIndex = tilingData->emptyInputTilingData.attentionOutLastCoreIndex;

    // 初始化 attentionOut
    if (attentionOutFormerNum == coreNum || (attentionOutFormerNum + attentionOutTailNum) < coreNum) {
        if (tmpBlockIdx < attentionOutFormerNum - 1) {
            InitOutput<INPUT_T>(attentionOutGm[tmpBlockIdx * attentionOutSingleCoreDataSize],
                attentionOutSingleCoreDataSize, (INPUT_T)0.0);
        } else if (tmpBlockIdx == attentionOutFormerNum - 1) {
            InitOutput<INPUT_T>(attentionOutGm[attentionOutLastCoreIndex], attentionOutLastCoreDataSize, (INPUT_T)0.0);
        }
    } else {
        if (tmpBlockIdx < attentionOutFormerNum) {
            InitOutput<INPUT_T>(attentionOutGm[tmpBlockIdx * attentionOutSingleCoreDataSize],
                attentionOutSingleCoreDataSize, (INPUT_T)0.0);
        } else if (tmpBlockIdx >= attentionOutFormerNum && tmpBlockIdx < coreNum - 1) {
            InitOutput<INPUT_T>(attentionOutGm[attentionOutFormerNum * attentionOutSingleCoreDataSize +
                (tmpBlockIdx - attentionOutFormerNum) * attentionOutTailCoreDataSize],
                attentionOutTailCoreDataSize, (INPUT_T)0.0);
        } else if (tmpBlockIdx == coreNum - 1) {
            InitOutput<INPUT_T>(attentionOutGm[attentionOutLastCoreIndex], attentionOutLastCoreDataSize, (INPUT_T)0.0);
        }
    }

    // 初始化 softmaxMax 和 softmaxSum
    if (tmpBlockIdx >= (softmaxMaxFormerNum + softmaxMaxTailNum)) {
        return;
    } else if (tmpBlockIdx < softmaxMaxFormerNum) {
        InitOutput<float>(softmaxMaxGm[tmpBlockIdx * softmaxMaxSingleCoreDataSize],
            tilingData->emptyInputTilingData.softmaxMaxSingleCoreDataSize, (float)0.0);
        InitOutput<float>(softmaxSumGm[tmpBlockIdx * softmaxMaxSingleCoreDataSize],
            tilingData->emptyInputTilingData.softmaxMaxSingleCoreDataSize, (float)0.0);
    } else {
        InitOutput<float>(softmaxMaxGm[softmaxMaxFormerNum * softmaxMaxSingleCoreDataSize +
            (tmpBlockIdx - softmaxMaxFormerNum) * softmaxMaxTailCoreDataSize],
            tilingData->emptyInputTilingData.softmaxMaxTailCoreDataSize, (float)0.0);
        InitOutput<float>(softmaxSumGm[softmaxMaxFormerNum * softmaxMaxSingleCoreDataSize +
            (tmpBlockIdx - softmaxMaxFormerNum) * softmaxMaxTailCoreDataSize],
            tilingData->emptyInputTilingData.softmaxMaxTailCoreDataSize, (float)0.0);
    }
}

template <typename INPUT_T> __aicore__ inline void FlashAttentionScoreEmptyInput<INPUT_T>::Process()
{
    ComputeEachCore();
}

#endif // FLASH_ATTENTION_SCORE_H