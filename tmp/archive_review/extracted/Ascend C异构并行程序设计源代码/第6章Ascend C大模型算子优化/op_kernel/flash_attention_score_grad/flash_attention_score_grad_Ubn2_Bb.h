/*
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

#ifndef FLASH_ATTENTION_SCORE_GRAD_UBN2_BB_H
#define FLASH_ATTENTION_SCORE_GRAD_UBN2_BB_H

#include "SplitBBase.h"

template <typename T, typename U, LayoutMode LAYOUT>
class FlashAttentionScoreGradUbn2Bb : public NewFlashAttentionScoreGradBase<T, U, LAYOUT> {
public:
    __aicore__ inline FlashAttentionScoreGradUbn2Bb() {};
    __aicore__ inline void Init(
        __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dout,
        __gm__ uint8_t *pseShift, __gm__ uint8_t *dropMask, __gm__ uint8_t *attenMask,
        __gm__ uint8_t *out, __gm__ uint8_t *softmaxMax,  __gm__ uint8_t *softmaxSum,
        __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dPse,
        __gm__ uint8_t *workspace, const FlashAttentionScoreGradUbngs1s2BbTilingData* __restrict tilingData, TPipe *pipeIn);
    __aicore__ inline void Process();
};

template <typename T, typename U, LayoutMode LAYOUT>
__aicore__ inline void FlashAttentionScoreGradUbn2Bb<T, U, LAYOUT>::Init(
      __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dout,
      __gm__ uint8_t *pseShift, __gm__ uint8_t *dropMask, __gm__ uint8_t *attenMask,
      __gm__ uint8_t *out, __gm__ uint8_t *softmaxMax,  __gm__ uint8_t *softmaxSum,
      __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dPse,
      __gm__ uint8_t *workspace, const FlashAttentionScoreGradUbngs1s2BbTilingData* __restrict tilingData, TPipe *pipeIn)
{
    this->m_queryGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(query));
    this->m_keyGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(key));
    this->m_valueGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(value));
    this->m_doutGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dout));

    this->m_pseGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(pseShift));

    this->m_dropMaskGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(dropMask));

    this->m_attenMaskGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(attenMask)); // TODO: uint8_t version?
    this->m_outGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
    this->m_softmaxMaxGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(softmaxMax));
    this->m_softmaxSumGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(softmaxSum));

    this->m_dqGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dq));
    this->m_dkGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dk));
    this->m_dvGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dv));
    this->m_dpseGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dPse));

    this->m_workspaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(workspace));
    // dunt need syncbuffer
    this->m_ordTilingData = tilingData;

    this->m_b = this->m_ordTilingData->opInfo.b;
    this->m_nQ= this->m_ordTilingData->opInfo.nQ;
    this->m_nKV = this->m_ordTilingData->opInfo.nKV;
    this->m_sQ = this->m_ordTilingData->opInfo.sQ;
    this->m_g = this->m_ordTilingData->opInfo.g;
    this->m_gSQ = this->m_ordTilingData->opInfo.gSQ;
    this->m_sKV = this->m_ordTilingData->opInfo.sKV;
    this->m_sKVAligned = this->m_ordTilingData->opInfo.sKVAlign;
    this->m_d = this->m_ordTilingData->opInfo.d;
    this->m_dInner = this->m_ordTilingData->opInfo.dInner;
    this->m_innerSize = this->m_ordTilingData->opInfo.innerSize;
    this->m_allInnerSize = this->m_ordTilingData->opInfo.allInnerSize;

    this->m_pipe = pipeIn;
    this->m_blockIdx = GetBlockIdx();
    uint32_t innerSize = 128 * 128 * 2; // TODO: add into tilingData

    this->m_pipe->InitBuffer(this->m_vecInQue1, 1, innerSize);
    this->m_pipe->InitBuffer(this->m_vecInQue2, 1, innerSize);
    this->m_pipe->InitBuffer(this->m_vecInQue3, 1, innerSize);
    this->m_pipe->InitBuffer(this->m_vecInQue4, 1, innerSize);
    this->m_pipe->InitBuffer(this->m_vecOutQue, 1, innerSize);

    this->m_pipe->InitBuffer(this->m_apiClcQue, 1, 16384);
    innerSize = 128 * 128 * 4; // TODO: add into tilingData

    uint64_t buffOffset = 800 + GetBlockNum() * innerSize;
    this->m_bmm1.SetWorkspace(
        (__gm__ uint8_t *)this->m_workspaceGm[800 + this->m_blockIdx * innerSize]
            .GetPhyAddr(),
        innerSize);
    this->m_bmm2.SetWorkspace(
        (__gm__ uint8_t *)this->m_workspaceGm[800 + GetBlockNum() * innerSize + this->m_blockIdx * innerSize]
            .GetPhyAddr(),
        innerSize);
    this->m_bmm3.SetWorkspace(
        (__gm__ uint8_t *)this->m_workspaceGm[800 + GetBlockNum() * innerSize * 2 + this->m_blockIdx * innerSize]
            .GetPhyAddr(),
        innerSize);
    this->m_bmm4.SetWorkspace(
        (__gm__ uint8_t *)this->m_workspaceGm[800 + GetBlockNum() * innerSize * 3 + this->m_blockIdx * innerSize]
            .GetPhyAddr(),
        innerSize);
}


template <typename T, typename U, LayoutMode LAYOUT>
__aicore__ inline void FlashAttentionScoreGradUbn2Bb<T, U, LAYOUT>::Process()
{
    uint64_t batchOffsetQ = this->m_blockIdx * this->m_ordTilingData->offsetInfo.batchOffsetQ;
    uint64_t batchOffsetKV = this->m_blockIdx * this->m_ordTilingData->offsetInfo.batchOffsetKV;
    uint64_t batchOffsetDrop = this->m_blockIdx * this->m_ordTilingData->offsetInfo.batchOffsetDrop;
    uint64_t batchOffsetMask = this->m_blockIdx * this->m_ordTilingData->offsetInfo.batchOffsetMask;
    uint64_t batchOffsetPse = this->m_blockIdx * this->m_ordTilingData->offsetInfo.batchOffsetPse;
    uint64_t batchOffsetSoftmax = this->m_blockIdx * this->m_ordTilingData->offsetInfo.batchOffsetSoftmax;

    uint32_t batchSize = this->m_blockIdx != (this->m_ordTilingData->splitCoreParams.usedCoreNum - 1) ?
        this->m_ordTilingData->singleCoreParams.bIn: this->m_ordTilingData->singleCoreParams.bInTail;
    for (int32_t bIdx = 0; bIdx < batchSize; bIdx++) {
        this->CalcDP(batchOffsetQ, batchOffsetKV, batchOffsetDrop);
        this->CalcDS(batchOffsetQ, batchOffsetKV, batchOffsetMask, batchOffsetSoftmax, batchOffsetPse);
        this->CalcDQDK(batchOffsetQ, batchOffsetKV);
        this->CalcDV(batchOffsetQ, batchOffsetKV, batchOffsetDrop);

        batchOffsetQ += this->m_ordTilingData->offsetInfo.matInSizeQ;
        batchOffsetKV += this->m_ordTilingData->offsetInfo.matInSizeKV;
        batchOffsetDrop += this->m_ordTilingData->offsetInfo.dropInSize;
        batchOffsetMask += this->m_ordTilingData->offsetInfo.maskInSize;
        batchOffsetSoftmax += this->m_ordTilingData->offsetInfo.softmaxSize;
    }
}

#endif
