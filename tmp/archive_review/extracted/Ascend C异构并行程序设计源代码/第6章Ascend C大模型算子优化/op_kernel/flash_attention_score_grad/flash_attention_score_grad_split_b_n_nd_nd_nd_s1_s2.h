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

#ifndef FLASH_ATTENTION_SCORE_GRAD_SPLIT_B_N_ND_ND_ND_S1_S2_H_
#define FLASH_ATTENTION_SCORE_GRAD_SPLIT_B_N_ND_ND_ND_S1_S2_H_

#include "flash_attention_score_grad_s1_s2_base_kernel.h"
#include "kernel_operator.h"

template <typename T1, typename T2, const bool IS_BF16 = false>
class FlashAttentionScoreGradBNNdNdNdS1S2 : public FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16> {
public:
    __aicore__ inline FlashAttentionScoreGradBNNdNdNdS1S2(){};
    __aicore__ inline void Process();

protected:
    __aicore__ inline bool isBatchTailCore();
};

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradBNNdNdNdS1S2<T1, T2, IS_BF16>::Process()
{
    this->isDrop = false;
    if (this->ordTilingData->baseParams.keepProb < 1) {
        this->isDrop = true;
    }

    this->InitGm();

    this->layoutSBH = this->ordTilingData->baseParams.layoutSBH;

    this->splitedBatchRange = this->ordTilingData->splitCoreParams.splitedBatchRange;
    this->flashKLoopRange = this->ordTilingData->splitCoreParams.flashKLoopRange;
    this->s1OutRange = this->ordTilingData->splitCoreParams.s1OutRange;
    this->sFla = this->ordTilingData->splitCoreParams.sFla;
    this->sOut = this->ordTilingData->splitCoreParams.sOut;
    this->outMaskNum = this->ordTilingData->splitCoreParams.outMaskNum;
    this->outReduceNum = this->ordTilingData->splitCoreParams.outReduceNum;
    this->outMatResSize = this->ordTilingData->splitCoreParams.outMatResSize;

    this->mSFlaForClcOffset = this->ordTilingData->splitCoreParams.sFla;
    this->mSOutForClcOffset = this->ordTilingData->splitCoreParams.sOut;

    if (isBatchTailCore()) {  // Tail Core Engagement
        this->splitedBatchRange = this->ordTilingData->tailSplitCoreParams.splitedBatchRange;
        this->flashKLoopRange = this->ordTilingData->tailSplitCoreParams.flashKLoopRange;
        this->s1OutRange = this->ordTilingData->tailSplitCoreParams.s1OutRange;

        this->sFla = this->ordTilingData->tailSplitCoreParams.sFla;
        this->sOut = this->ordTilingData->tailSplitCoreParams.sOut;
        this->outMaskNum = this->ordTilingData->tailSplitCoreParams.outMaskNum;
        this->outReduceNum = this->ordTilingData->tailSplitCoreParams.outReduceNum;
    }

    for (int32_t batchIdx = 0; batchIdx < this->splitedBatchRange; batchIdx++) {
        uint64_t bIdx = (this->mBlockIdx * this->splitedBatchRange + batchIdx) / this->ordTilingData->baseParams.N;
        uint64_t nIdx = (this->mBlockIdx * this->splitedBatchRange + batchIdx) % this->ordTilingData->baseParams.N;

        uint64_t batchLoopOffset = bIdx * this->ordTilingData->baseParams.S * this->ordTilingData->baseParams.H +
                                nIdx * this->ordTilingData->baseParams.D;
        uint64_t batchLoopOffset2 = bIdx * this->ordTilingData->baseParams.Skv * this->ordTilingData->baseParams.H +
                                nIdx * this->ordTilingData->baseParams.D;
        if (this->layoutSBH == 1) {
            batchLoopOffset = bIdx * this->ordTilingData->baseParams.H + nIdx * this->ordTilingData->baseParams.D;
            batchLoopOffset2 = batchLoopOffset;
        }
        uint64_t batchLoopMaskOffset =
            this->mBlockIdx * this->ordTilingData->splitCoreParams.splitedCoreBatchMaskNum +
            batchIdx * this->ordTilingData->baseParams.allMatResNum;
        uint64_t batchLoopReduceOffset =
            this->mBlockIdx * this->ordTilingData->splitCoreParams.splitedCoreBatchReduceNum +
            batchIdx * this->ordTilingData->baseParams.allReduceNum;

        for (int32_t flashKIdx = 0; flashKIdx < this->flashKLoopRange; flashKIdx++) {  // s/flashK = s_fla
            uint64_t flashKLoopOffset = batchLoopOffset2 + flashKIdx * this->sFla * this->ordTilingData->baseParams.H;
            if (this->layoutSBH == 1) {
                flashKLoopOffset = batchLoopOffset2 +
                    flashKIdx * this->sFla * this->ordTilingData->baseParams.B * this->ordTilingData->baseParams.H;
            }

            uint32_t idx = 0;
            uint64_t s1OutIdxStart = 0;

            this->mm1.SetTensorB(this->valueGm[flashKLoopOffset], true);
            this->mm2.SetTensorB(this->keyGm[flashKLoopOffset], true);
            for (int32_t s1OutIdx = 0; s1OutIdx < this->s1OutRange; s1OutIdx++) {  // s/s1OutRges = s_out
                uint64_t s1OutLoopOffset = batchLoopOffset + s1OutIdx * this->sOut * this->ordTilingData->baseParams.H;
                if (this->layoutSBH == 1) {
                    s1OutLoopOffset =
                        batchLoopOffset +
                        s1OutIdx * this->sOut * this->ordTilingData->baseParams.B * this->ordTilingData->baseParams.H;
                }

                uint64_t s1OutLoopMaskOffset = batchLoopMaskOffset + s1OutIdx * this->outMaskNum;
                uint64_t s1OutLoopReduceOffset = batchLoopReduceOffset + s1OutIdx * this->outReduceNum;

                this->pseOffset = bIdx * this->ordTilingData->baseParams.N * this->ordTilingData->baseParams.Skv +
                                  nIdx * this->ordTilingData->baseParams.Skv;  // 按 BN1S场景设置
                this->attenMaskOffset = s1OutIdx * this->outMaskNum;
                if (this->ordTilingData->baseParams.attenMaskShapeType == 1) {  // B1SS
                    this->attenMaskOffset +=
                        bIdx * this->ordTilingData->baseParams.S * this->ordTilingData->baseParams.Skv;
                }

                if (isBatchTailCore()) {
                    this->DetermineLoopParams(this->ordTilingData->tailSplitCoreParams, flashKIdx, s1OutIdx);
                } else {
                    this->DetermineLoopParams(this->ordTilingData->splitCoreParams, flashKIdx, s1OutIdx);
                }

                if (idx == 0) {
                    this->curCubePreloadNum = this->s1OutRange - s1OutIdx >= CUBE_PRELOAD_BLOCK_NUM ? CUBE_PRELOAD_BLOCK_NUM : this->s1OutRange % CUBE_PRELOAD_BLOCK_NUM;
                    this->mm1.SetTail(this->sOut * this->curCubePreloadNum, this->sFla, this->ordTilingData->baseParams.D); // M N K
                    this->mm1.SetTensorA(this->dxGm[batchLoopOffset]);
                    this->mm1.template Iterate<false>();
                }
                this->mm2.SetTail(this->sOut, this->sFla, this->ordTilingData->baseParams.D); // M N K
                this->mm2.SetTensorA(this->queryGm[s1OutLoopOffset]);
                this->mm2.template Iterate<false>(); // [sInner, H/N] * [H/N, sFla]

                this->FrontCompute(flashKIdx, s1OutIdx, flashKLoopOffset,
                                   s1OutLoopOffset, s1OutLoopMaskOffset, s1OutIdxStart, this->s1OutRange);
                this->ReCompute(batchIdx, flashKIdx, s1OutIdx, flashKLoopOffset,
                                s1OutLoopOffset, s1OutLoopMaskOffset, s1OutLoopReduceOffset, s1OutIdxStart, this->s1OutRange);
                idx++;
            }
        }
    }
    // synaAll
    this->SyncAllLast();
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline bool FlashAttentionScoreGradBNNdNdNdS1S2<T1, T2, IS_BF16>::isBatchTailCore()
{
    return this->mBlockIdx != 0 && this->mBlockIdx == this->ordTilingData->baseParams.coreNum - 1;
}

#endif