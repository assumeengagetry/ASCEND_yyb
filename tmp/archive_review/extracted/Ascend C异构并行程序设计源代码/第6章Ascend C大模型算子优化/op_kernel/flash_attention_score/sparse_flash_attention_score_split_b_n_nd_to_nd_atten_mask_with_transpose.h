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

#ifndef SPARSE_FLASH_ATTENTION_SCORE_SPLIT_B_N_ND_TO_ND_ATTEN_MASK_WITH_TRANSPOSE_H
#define SPARSE_FLASH_ATTENTION_SCORE_SPLIT_B_N_ND_TO_ND_ATTEN_MASK_WITH_TRANSPOSE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "flash_attention_score_base.h"

using namespace AscendC;
using namespace matmul;
template <typename INPUT_T, typename T = INPUT_T, bool isBasicBlock = false>
class SparseFlashAttentionScoreAttenND2NDWithTrans : public FlashAttentionScore<INPUT_T, T, isBasicBlock> {
public:
    __aicore__ inline SparseFlashAttentionScoreAttenND2NDWithTrans()
    {
        this->reuseWorkspaceRatio = 1;
    };
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
        __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax,
        __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
        const FlashAttentionScoreTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

protected:
    int loopSOuterIdx;
    int startInBlock;
    int endInBlock;

    __aicore__ inline void ComputeEachCore(uint32_t coreIdx);

    __aicore__ inline void initOffset();

    __aicore__ inline void initTilingData();
};

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void SparseFlashAttentionScoreAttenND2NDWithTrans<INPUT_T, T, isBasicBlock>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse, __gm__ uint8_t *dropMask,
    __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
    __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const FlashAttentionScoreTilingData *__restrict tiling, TPipe *tPipe)
{
    this->tmp_block_idx = GetBlockIdx();
    // init global buffer
    this->queryGm.SetGlobalBuffer((__gm__ INPUT_T *)query);
    this->keyGm.SetGlobalBuffer((__gm__ INPUT_T *)key);
    this->valueGm.SetGlobalBuffer((__gm__ INPUT_T *)value);
    this->pseGm.SetGlobalBuffer((__gm__ INPUT_T *)pse);
    this->dropMaskGm.SetGlobalBuffer((__gm__ uint8_t *)dropMask);
    this->attenMaskGm.SetGlobalBuffer((__gm__ INPUT_T *)attenMask);
    this->attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
    this->softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
    this->softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);
    this->attentionOutGm.SetGlobalBuffer((__gm__ INPUT_T *)attentionOut);
    this->workspaceGm.SetGlobalBuffer((__gm__ T *)workspace);

    this->tilingData = tiling;
    this->pipe = tPipe;
    initOffset();
    initTilingData();
    if (this->attenMaskUbSize != 0) {
      this->pipe->InitBuffer(this->attenMaskQueue, 1, this->attenMaskUbSize * sizeof(INPUT_T));
    }
    this->pipe->InitBuffer(this->mmResQueue, 1, this->mmResUbSize * sizeof(T));
    this->pipe->InitBuffer(this->tmpMMResBmm2Queue, 1, this->bmm2ResUbSize * sizeof(T));
    this->pipe->InitBuffer(this->tmpSoftmaxUb, this->softmaxExpSize * sizeof(T));
    this->pipe->InitBuffer(this->tmpBmm2Ub, this->bmm2ResUbSize * sizeof(T));
    this->pipe->InitBuffer(this->softmaxOutQueue, 2, this->softmaxMaxSize * sizeof(float));
    this->pipe->InitBuffer(this->apiTmpBufferUb, this->tilingData->attentionScoreSingleCoreTensorSize.softmaxTmpBuffer);
    if constexpr (IsSameType<INPUT_T, bfloat16_t>::value ||
        (IsSameType<INPUT_T, half>::value && IsSameType<T, float>::value)) {
        this->tmpTypeCastUb = this->apiTmpBufferUb;
    }
    this->reuseWorkspaceRatio = this->maxInnerLoopTimes;
    this->mm.SetWorkspace((__gm__ uint8_t *)this->workspaceGm[this->tmp_block_idx * this->mmResUbSize *
                                                              this->reuseWorkspaceRatio].GetPhyAddr(),
                          this->mmResUbSize * this->reuseWorkspaceRatio);
    uint32_t buff_offset =
        GetBlockNum() * GetTaskRation() * this->mmResUbSize * this->reuseWorkspaceRatio;
    this->bmm2.SetWorkspace(
        (__gm__ uint8_t *)this->workspaceGm[buff_offset + this->tmp_block_idx * this->bmm2ResUbSize].GetPhyAddr(),
        this->bmm2ResUbSize);
    LocalTensor<T> apiTmpBuffer = this->apiTmpBufferUb.template Get<T>();
    if (this->tilingData->baseParams.keepProb != 1) {
      DropOutBitModeInit(apiTmpBuffer);
    }
    if constexpr (IsSameType<T, float>::value) {
        this->infValue = -2e38;
    } else {
        this->infValue = -10000.0;
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void SparseFlashAttentionScoreAttenND2NDWithTrans<INPUT_T, T, isBasicBlock>::initOffset()
{
    this->loopBNum = this->tilingData->attentionScoreCoreParams.loopBatchNum;
    this->loopNNum = this->tilingData->attentionScoreCoreParams.loopHeadNum;
    if (this->tmp_block_idx % this->tilingData->attentionScoreCoreParams.n1 >=
        this->tilingData->attentionScoreCoreParams.formerNum) {
        this->loopNNum = this->tilingData->attentionScoreCoreParams.loopHeadNumTail;
    }
    this->loopSNum = this->tilingData->attentionScoreSingleCoreParams.sOuterLoopTimes;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void SparseFlashAttentionScoreAttenND2NDWithTrans<INPUT_T, T, isBasicBlock>::initTilingData()
{
    this->singleProcessSOuterSize = this->tilingData->attentionScoreSingleCoreParams.singleProcessSOuterSize;
    this->singleProcessSInnerSize = this->tilingData->attentionScoreSingleCoreParams.singleProcessSInnerSize;
    this->sInnerLoopTimes = this->tilingData->attentionScoreSingleCoreParams.sInnerLoopTimes;
    this->singleProcessSInnerSizeTail = this->tilingData->attentionScoreSingleCoreParams.singleProcessSInnerSizeTail;
    this->typeByteNum = this->tilingData->attentionScoreOffestStrideParams.typeByteNum;

    this->mmResUbSize = this->tilingData->attentionScoreSingleCoreTensorSize.mmResUbSize;
    this->attenMaskUbSize = this->tilingData->attentionScoreSingleCoreTensorSize.attenMaskUbSize;
    this->maskSize = this->tilingData->attentionScoreSingleCoreTensorSize.maskSize;
    this->softmaxMaxSize = this->tilingData->attentionScoreSingleCoreTensorSize.softmaxMaxSize;
    this->softmaxSumSize = this->tilingData->attentionScoreSingleCoreTensorSize.softmaxSumSize;
    this->softmaxExpSize = this->tilingData->attentionScoreSingleCoreTensorSize.softmaxExpSize;
    this->bmm2ResUbSize = this->tilingData->attentionScoreSingleCoreTensorSize.bmm2ResUbSize;
    this->tmpMMResBmm2PreUbSize = this->tilingData->attentionScoreSingleCoreTensorSize.tmpMMResBmm2PreUbSize;
    this->tmpSoftmaxBmm2UbSize = this->tilingData->attentionScoreSingleCoreTensorSize.tmpSoftmaxBmm2UbSize;
    this->mmResUbSizeTail = this->tilingData->attentionScoreSingleCoreTensorSize.mmResUbSizeTailLoop;
    this->attenMaskUbSizeTail = this->tilingData->attentionScoreSingleCoreTensorSize.attenMaskUbSizeTailLoop;
    this->maskSizeTail = this->tilingData->attentionScoreSingleCoreTensorSize.maskSizeTailLoop;

    this->transposeTilingData = this->tilingData->transposeTilingData;
    this->softmaxTilingData = this->tilingData->softmaxTilingData;
    this->softmaxFlashTilingData = this->tilingData->softmaxFlashTilingData;
    this->softmaxMaxTypeByteNum = (sizeof(T) == sizeof(half)) ? (this->typeByteNum / 2) : this->typeByteNum;
    this->maxInnerLoopTimes = this->tilingData->attentionScoreSingleCoreParams.sInnerLoopTimes;
    this->sInnerLoopStartIndex = 0;
    this->sInnerLoopEndIndex = this->tilingData->attentionScoreSingleCoreParams.sInnerLoopTimes;
    this->attenMaskTypeIdx = this->tilingData->baseParams.attenMaskTypeIdx;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void SparseFlashAttentionScoreAttenND2NDWithTrans<INPUT_T, T, isBasicBlock>::Process()
{
    ComputeEachCore(this->tmp_block_idx);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void SparseFlashAttentionScoreAttenND2NDWithTrans<INPUT_T, T, isBasicBlock>::ComputeEachCore(
    uint32_t coreIdx)
{
    uint64_t bIdx = coreIdx / this->tilingData->attentionScoreCoreParams.n1 %
        this->tilingData->attentionScoreCoreParams.b1 * this->tilingData->attentionScoreCoreParams.singleCoreBatchSize;
    uint64_t nIdx = coreIdx % this->tilingData->attentionScoreCoreParams.n1;
    if (nIdx >= this->tilingData->attentionScoreCoreParams.formerNum) {
        nIdx = (nIdx - this->tilingData->attentionScoreCoreParams.formerNum) *
            this->tilingData->attentionScoreCoreParams.singleCoreHeadNumSizeTail +
            this->tilingData->attentionScoreCoreParams.formerNum *
            this->tilingData->attentionScoreCoreParams.singleCoreHeadNumSize;
    } else {
        nIdx = nIdx * this->tilingData->attentionScoreCoreParams.singleCoreHeadNumSize;
    }

    for (uint64_t loopBIdx = 0; loopBIdx < this->loopBNum; loopBIdx++) {
        // B offset
        this->batchBOffset = bIdx + this->tilingData->attentionScoreSingleCoreParams.singleProcessBatchSize * loopBIdx;
        for (uint64_t loopNIdx = 0; loopNIdx < this->loopNNum; loopNIdx++) {
            // N offset
            this->batchNOffset =
                nIdx + loopNIdx * this->tilingData->attentionScoreSingleCoreParams.singleProcessHeadNumSize;
            for (uint64_t sOuterLoopIdx = 0;
                sOuterLoopIdx < this->tilingData->attentionScoreSingleCoreParams.sOuterLoopTimes; sOuterLoopIdx++) {
                // Souter offset
                this->sOuterOffset = sOuterLoopIdx * this->singleProcessSOuterSize;
                this->LoopSOuterOffsetInit();
                this->sInnerLoopStartIndex =
                    (this->sOuterOffset - this->tilingData->baseParams.preTockens) / this->singleProcessSInnerSize;
                this->sInnerLoopEndIndex = (this->sOuterOffset + this->singleProcessSOuterSize +
                    this->tilingData->baseParams.nextTockens + this->singleProcessSInnerSize - 1) /
                    this->singleProcessSInnerSize;

                if (this->sInnerLoopStartIndex < 0) {
                    this->sInnerLoopStartIndex = 0;
                }
                if (this->sInnerLoopEndIndex > this->maxInnerLoopTimes) {
                    this->sInnerLoopEndIndex = this->maxInnerLoopTimes;
                }

                this->sInnerLoopTimes = this->sInnerLoopEndIndex - this->sInnerLoopStartIndex;

                this->SInnerLoopFunc(sOuterLoopIdx);
            }
        }
    }
}

#endif // SPARSE_FLASH_ATTENTION_SCORE_SPLIT_B_N_ND_TO_ND_ATTEN_MASK_WITH_TRANSPOSE_H
