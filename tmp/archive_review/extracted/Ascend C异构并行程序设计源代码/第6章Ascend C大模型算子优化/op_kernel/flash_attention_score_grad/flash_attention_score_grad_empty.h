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

#ifndef FLASH_ATTENTION_SCORE_GRAD_EMPTY_H_
#define FLASH_ATTENTION_SCORE_GRAD_EMPTY_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;
template <typename T> class FlashAttentionScoreGradEmpty {
public:
    __aicore__ inline FlashAttentionScoreGradEmpty() {};
    __aicore__ inline void Init(__gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dpse,
        const FlashAttentionScoreGradTilingData *__restrict tilingData);
    __aicore__ inline void Process();

protected:
    uint64_t m_blockIdx;
    GlobalTensor<T> m_dqGm, m_dkGm, m_dvGm, m_dpseGm;
    const EmptyTensorTilingData *__restrict m_tilingData;
};

template <typename T>
__aicore__ inline void FlashAttentionScoreGradEmpty<T>::Init(__gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv,
    __gm__ uint8_t *dpse, const FlashAttentionScoreGradTilingData *__restrict tilingData)
{
    m_blockIdx = GetBlockIdx();
    m_tilingData = &tilingData->emptyTensorTilingData;
    m_dqGm.SetGlobalBuffer((__gm__ T *)dq);
    m_dkGm.SetGlobalBuffer((__gm__ T *)dk);
    m_dvGm.SetGlobalBuffer((__gm__ T *)dv);
    m_dpseGm.SetGlobalBuffer((__gm__ T *)dpse);
}

template <typename T> __aicore__ inline void FlashAttentionScoreGradEmpty<T>::Process()
{
    if (m_tilingData->singleCoreDqNum > 0) {
        if (m_blockIdx < m_tilingData->formerDqNum) {
            InitOutput<T>(m_dqGm[m_blockIdx * m_tilingData->singleCoreDqNum], m_tilingData->singleCoreDqNum, 0);
        } else {
            InitOutput<T>(m_dqGm[m_tilingData->formerDqNum * m_tilingData->singleCoreDqNum +
                (m_blockIdx - m_tilingData->formerDqNum) * m_tilingData->tailCoreDqNum],
                m_tilingData->tailCoreDqNum, 0);
        }
    }
    if (m_tilingData->singleCoreDkNum > 0) {
        if (m_blockIdx < m_tilingData->formerDkNum) {
            InitOutput<T>(m_dkGm[m_blockIdx * m_tilingData->singleCoreDkNum], m_tilingData->singleCoreDkNum, 0);
            InitOutput<T>(m_dvGm[m_blockIdx * m_tilingData->singleCoreDkNum], m_tilingData->singleCoreDkNum, 0);
        } else {
            InitOutput<T>(m_dkGm[m_tilingData->formerDkNum * m_tilingData->singleCoreDkNum +
                (m_blockIdx - m_tilingData->formerDkNum) * m_tilingData->tailCoreDkNum],
                m_tilingData->tailCoreDkNum, 0);
            InitOutput<T>(m_dvGm[m_tilingData->formerDkNum * m_tilingData->singleCoreDkNum +
                (m_blockIdx - m_tilingData->formerDkNum) * m_tilingData->tailCoreDkNum],
                m_tilingData->tailCoreDkNum, 0);
        }
    }
    if (m_tilingData->singleCoreDpseNum > 0) {
        if (m_blockIdx < m_tilingData->formerDpseNum) {
            InitOutput<T>(m_dpseGm[m_blockIdx * m_tilingData->singleCoreDpseNum], m_tilingData->singleCoreDpseNum, 0);
        } else {
            InitOutput<T>(m_dpseGm[m_tilingData->formerDpseNum * m_tilingData->singleCoreDpseNum +
                (m_blockIdx - m_tilingData->formerDpseNum) * m_tilingData->tailCoreDpseNum],
                m_tilingData->tailCoreDpseNum, 0);
        }
    }
}

#endif