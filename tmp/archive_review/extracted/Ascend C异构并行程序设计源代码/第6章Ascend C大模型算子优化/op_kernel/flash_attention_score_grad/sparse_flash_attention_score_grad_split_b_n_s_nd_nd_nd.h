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

#ifndef _SPARSE_FLASH_ATTENTION_SCORE_GRAD_SPLIT_B_N_S_ND_ND_ND_H_
#define _SPARSE_FLASH_ATTENTION_SCORE_GRAD_SPLIT_B_N_S_ND_ND_ND_H_

#include "flash_attention_score_grad_base_kernel.h"
#include "kernel_operator_common_intf.h"

template <typename T1, typename T2, const bool IS_BF16 = false>
class SparseFlashAttentionScoreGradBNSNdNdNd : public FlashAttentionScoreGradBase<T1, T2, IS_BF16> {
public:
    __aicore__ inline SparseFlashAttentionScoreGradBNSNdNdNd(){};
    __aicore__ inline void Process();

protected:
    int32_t flashKIdxFactor;
    int32_t s1OutIdxFactor;

    __aicore__ inline bool isBatchTailCore();
    __aicore__ inline bool isSeqTailCore();
    __aicore__ inline void getSplitCoreIdx(uint64_t &splitBatchCoreIdx, uint64_t &splitSeqCoreIdx);
};

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void SparseFlashAttentionScoreGradBNSNdNdNd<T1, T2, IS_BF16>::Process()
{
    this->isDrop = false;
    if (this->ordTilingData->baseParams.keepProb < 1) {
        this->isDrop = true;
    }

    this->InitGm();

    this->layoutSBH = this->ordTilingData->baseParams.layoutSBH;

    uint64_t splitBatchCoreIdx = 0, splitSeqCoreIdx = 0;
    getSplitCoreIdx(splitBatchCoreIdx, splitSeqCoreIdx);

    this->mSFlaForClcOffset = this->ordTilingData->splitCoreParams.sFla;
    this->mSOutForClcOffset = this->ordTilingData->splitCoreParams.sOut;

    // Before sOut&&sFla parameters
    this->splitedBatchRange = this->ordTilingData->splitCoreParams.splitedBatchRange;
    this->flashKLoopRange = this->ordTilingData->splitCoreParams.flashKLoopRange;
    this->s1OutRange = this->ordTilingData->splitCoreParams.s1OutRange;
    this->sFla = this->ordTilingData->splitCoreParams.sFla;
    this->sOut = this->ordTilingData->splitCoreParams.sOut;
    this->outMaskNum = this->ordTilingData->splitCoreParams.outMaskNum;
    this->outReduceNum = this->ordTilingData->splitCoreParams.outReduceNum;
    this->outMatResSize = this->ordTilingData->splitCoreParams.outMatResSize;

    if (isBatchTailCore()) {
        this->splitedBatchRange = this->ordTilingData->tailSplitCoreParams.splitedBatchRange;
    }

    if (isSeqTailCore()) { // Tail Core Engagement
        this->flashKLoopRange = this->ordTilingData->tailSplitCoreParams.flashKLoopRange;
        this->s1OutRange = this->ordTilingData->tailSplitCoreParams.s1OutRange;

        this->sFla = this->ordTilingData->tailSplitCoreParams.sFla;
        this->sOut = this->ordTilingData->tailSplitCoreParams.sOut;
        this->outMaskNum = this->ordTilingData->tailSplitCoreParams.outMaskNum;
        this->outReduceNum = this->ordTilingData->tailSplitCoreParams.outReduceNum;
    }

    int32_t nextTokenBlocks = this->ordTilingData->baseParams.nextTokensBlocks,
            preTokenBlocks = this->ordTilingData->baseParams.preTokensBlocks;

    flashKIdxFactor = this->sFla > this->sOut ? this->sFla / this->sOut : 1;
    s1OutIdxFactor = this->sFla > this->sOut ? 1 : this->sOut / this->sFla;
    if(this->sFla != this->sOut){
        if(s1OutIdxFactor == 1){
            preTokenBlocks += 1;
        }
        else{
            nextTokenBlocks += 1;
        }
    }

    for(int32_t batchIdx = 0; batchIdx < this->splitedBatchRange; batchIdx++) {
        int32_t bIdx = (this->mBlockIdx / this->ordTilingData->baseParams.splitSeqFactor *
            this->ordTilingData->splitCoreParams.splitedBatchRange + batchIdx) /
            this->ordTilingData->baseParams.N;
        int32_t nIdx = (this->mBlockIdx / this->ordTilingData->baseParams.splitSeqFactor *
            this->ordTilingData->splitCoreParams.splitedBatchRange + batchIdx) %
            this->ordTilingData->baseParams.N;

        uint64_t batchLoopOffset = bIdx * this->ordTilingData->baseParams.S * this->ordTilingData->baseParams.H +
                    nIdx * this->ordTilingData->baseParams.D +
                    splitSeqCoreIdx * this->ordTilingData->splitCoreParams.splitedCoreSeqNum * this->ordTilingData->baseParams.H;

        if (this->layoutSBH == 1) {
            batchLoopOffset = bIdx * this->ordTilingData->baseParams.H + nIdx * this->ordTilingData->baseParams.D +
                               splitSeqCoreIdx * this->ordTilingData->splitCoreParams.splitedCoreSeqNum * this->ordTilingData->baseParams.B * this->ordTilingData->baseParams.H;
        }

        uint64_t batchLoopMaskOffset = splitBatchCoreIdx * this->ordTilingData->splitCoreParams.splitedCoreBatchMaskNum +
                                batchIdx * this->ordTilingData->baseParams.allMatResNum +
                                splitSeqCoreIdx * this->ordTilingData->splitCoreParams.splitAllMatResNum;
        uint64_t batchLoopReduceOffset = splitBatchCoreIdx * this->ordTilingData->splitCoreParams.splitedCoreBatchReduceNum +
                                batchIdx * this->ordTilingData->baseParams.allReduceNum +
                                splitSeqCoreIdx * this->ordTilingData->splitCoreParams.splitAllReduceNum;

        for(int32_t flashKIdx = 0; flashKIdx < this->flashKLoopRange; flashKIdx++) {
            uint64_t flashKLoopOffset = bIdx * this->ordTilingData->baseParams.S * this->ordTilingData->baseParams.H +
                                    nIdx * this->ordTilingData->baseParams.D +
                                    flashKIdx * this->sFla * this->ordTilingData->baseParams.H;
            if (this->layoutSBH == 1) {
                flashKLoopOffset = bIdx * this->ordTilingData->baseParams.H + nIdx * this->ordTilingData->baseParams.D +
                                   flashKIdx * this->sFla * this->ordTilingData->baseParams.B * this->ordTilingData->baseParams.H;
            }

            uint32_t idx = 0;
            int32_t s1OutIdxStart = min(this->s1OutRange, max((int32_t)0, (int32_t)(flashKIdx * flashKIdxFactor - nextTokenBlocks - (int32_t)splitSeqCoreIdx * this->ordTilingData->splitCoreParams.s1OutRange))) / s1OutIdxFactor;
            int32_t s1OutIdxEnd = max(0, min((int32_t)this->s1OutRange, (int32_t)(flashKIdx * flashKIdxFactor + preTokenBlocks - (int32_t)splitSeqCoreIdx * this->ordTilingData->splitCoreParams.s1OutRange + 1))) / s1OutIdxFactor;

            int32_t mLength = s1OutIdxEnd - s1OutIdxStart;
            if (mLength <= 0) {
                continue;
            }

            this->mm1.SetTensorB(this->valueGm[flashKLoopOffset], true);
            this->mm2.SetTensorB(this->keyGm[flashKLoopOffset], true);

            for (int32_t s1OutIdx = s1OutIdxStart; s1OutIdx < s1OutIdxEnd; s1OutIdx++) {
                uint64_t s1OutLoopOffset =
                    batchLoopOffset + s1OutIdx * this->sOut * this->ordTilingData->baseParams.H;

                if (this->layoutSBH == 1) {
                    s1OutLoopOffset = batchLoopOffset + s1OutIdx * this->sOut * this->ordTilingData->baseParams.B * this->ordTilingData->baseParams.H;
                }
                uint64_t s1OutLoopMaskOffset = batchLoopMaskOffset + s1OutIdx * this->outMaskNum;
                uint64_t s1OutLoopReduceOffset = batchLoopReduceOffset + s1OutIdx * this->outReduceNum;

                this->pseOffset = bIdx * this->ordTilingData->baseParams.N * this->ordTilingData->baseParams.S +
                                    nIdx * this->ordTilingData->baseParams.S;  // 按 BN1S场景设置
                this->attenMaskOffset = s1OutIdx * this->outMaskNum +
                                        splitSeqCoreIdx * this->ordTilingData->splitCoreParams.splitAllMatResNum;
                if (this->ordTilingData->baseParams.attenMaskShapeType == 1) {  // B1SS
                    this->attenMaskOffset +=
                        bIdx * this->ordTilingData->baseParams.S * this->ordTilingData->baseParams.S;
                }

                if (isSeqTailCore()) {
                    this->DetermineLoopParams(this->ordTilingData->tailSplitCoreParams, flashKIdx, s1OutIdx);
                } else {
                    this->DetermineLoopParams(this->ordTilingData->splitCoreParams, flashKIdx, s1OutIdx);
                }

                if (idx == 0) {
                    this->curCubePreloadNum = (s1OutIdxEnd - s1OutIdx >= CUBE_PRELOAD_BLOCK_NUM ? CUBE_PRELOAD_BLOCK_NUM : s1OutIdxEnd - s1OutIdx);
                    this->mm1.SetTail(this->sOut * this->curCubePreloadNum, this->sFla, this->ordTilingData->baseParams.D); // M N K
                    this->mm1.SetTensorA(this->dxGm[s1OutLoopOffset]);
                    this->mm1.template Iterate<false>();
                }
                this->mm2.SetTail(this->sOut, this->sFla, this->ordTilingData->baseParams.D); // M N K
                this->mm2.SetTensorA(this->queryGm[s1OutLoopOffset]);
                this->mm2.template Iterate<false>(); // [sInner, H/N] * [H/N, sFla]

                this->FrontCompute(flashKIdx, s1OutIdx, flashKLoopOffset, s1OutLoopOffset, s1OutLoopMaskOffset, s1OutIdxStart, s1OutIdxEnd);
                this->ReCompute(batchIdx, flashKIdx, s1OutIdx, flashKLoopOffset, s1OutLoopOffset, s1OutLoopMaskOffset, s1OutLoopReduceOffset, s1OutIdxStart, s1OutIdxEnd);
                idx++;
            }
        }
    }
    // synaAll
    this->SyncAllLast();
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline bool SparseFlashAttentionScoreGradBNSNdNdNd<T1, T2, IS_BF16>::isBatchTailCore()
{
    return this->mBlockIdx != 0 &&
        (this->mBlockIdx / this->ordTilingData->baseParams.splitSeqFactor ==
        this->ordTilingData->baseParams.splitBatchFactor - 1);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline bool SparseFlashAttentionScoreGradBNSNdNdNd<T1, T2, IS_BF16>::isSeqTailCore() {
    return this->mBlockIdx != 0 && (this->mBlockIdx % this->ordTilingData->baseParams.splitSeqFactor == this->ordTilingData->baseParams.splitSeqFactor - 1);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void SparseFlashAttentionScoreGradBNSNdNdNd<T1, T2, IS_BF16>::getSplitCoreIdx(
    uint64_t &splitBatchCoreIdx, uint64_t &splitSeqCoreIdx)
{
    splitBatchCoreIdx = this->mBlockIdx / this->ordTilingData->baseParams.splitSeqFactor;
    splitSeqCoreIdx = this->mBlockIdx % this->ordTilingData->baseParams.splitSeqFactor;
}

#endif
