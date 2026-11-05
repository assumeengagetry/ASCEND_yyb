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

#ifndef FLASH_ATTENTION_SCORE_SPLIT_B_H
#define FLASH_ATTENTION_SCORE_SPLIT_B_H

#include "flash_attention_score_common.h"
#include "kernel_data_copy_transpose.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

using matmul::MatmulType;

// INPUT_T - means data type for input
// T       - means data type when calc
template <typename INPUT_T, typename T = INPUT_T, bool isBasicBlock = false, LayoutMode layout = LayoutMode::BNGS1S2>
class FlashAttentionScoreSplitBBn2gs1s2 : public FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock> {
public:
    __aicore__ inline FlashAttentionScoreSplitBBn2gs1s2() {};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
        __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax,
        __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
        const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

    // define matmul
    using a1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false, layout>;
    using b1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, true, layout>;
    using bias1Type = MatmulType<TPosition::GM, CubeFormat::ND, float, false, LayoutMode::BNGS1S2>;
    using c1Type = MatmulType<TPosition::VECCALC, CubeFormat::ND_ALIGN, T, false, LayoutMode::BNGS1S2>;
    matmul::Matmul<a1Type, b1Type, c1Type, bias1Type> bmm1;

    // define batchmatmul
    using a2Type = MatmulType<TPosition::VECCALC, CubeFormat::ND, INPUT_T, false, LayoutMode::BNGS1S2>;
    using b2Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false, LayoutMode::BNGS1S2>;
    using bias2Type = MatmulType<TPosition::GM, CubeFormat::ND, float, false, layout>;
    using c2Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false, layout>;
    matmul::Matmul<a2Type, b2Type, c2Type, bias2Type> bmm2;

protected:
    __aicore__ inline void ProcessBlock();
    __aicore__ inline void InitSoftmaxTensor();
    __aicore__ inline void Bmm1SetTensorA();
    __aicore__ inline void Bmm1SetTensorB();
    __aicore__ inline void CalBatchSize();
    __aicore__ inline void ProcessStage1(LocalTensor<INPUT_T> &stage1ResUb);
    __aicore__ inline int64_t ComputePseOffset();
    __aicore__ inline int64_t ComputeAttenMaskOffset();
    __aicore__ inline int64_t ComputeOffsetForNoCompress();
    __aicore__ inline void SetResShape(LocalTensor<T> &bmm1ResUb, LocalTensor<INPUT_T> &stage1ResUb);
    __aicore__ inline void SoftmaxCompute(LocalTensor<T> &dstTensor, LocalTensor<T> &srcTensor);

    __aicore__ inline void ProcessStage2(LocalTensor<INPUT_T> &stage1ResUb);
    __aicore__ inline void Bmm2(LocalTensor<INPUT_T> &stage1ResUb);

    // bmm1TensorA的batch大小
    int64_t tensorABatchSize;

    // bmm1TensorB的batch大小
    int64_t tensorBBatchSize;

    // pse载入ub的batch大小
    int64_t pseBatchSize;

    // attenmask载入ub的batch大小
    int64_t attenMaskBatchSize;

    // dropout载入ub的batch大小
    int64_t dropoutBatchSize;

    // k和v矩阵在核间的offset
    int64_t kvCoreOffset;

    TBuf<> SoftmaxTmpTBuf; // 存softmax中间结果

    // 轴的乘积
    int64_t n2G;
    int64_t s1S2;

    // 切分轴的乘积
    int64_t bBaseN2D;
    int64_t bBaseS1S2;
    int64_t bBaseN2GD;
    int64_t bBaseN2GS1;
    int64_t bBaseN2GS2;
    int64_t bBaseN2S2D;
    int64_t bBaseN2GS1D;
    int64_t bBaseN2GS2D;
    int64_t bBaseN2GS1S2;
};

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBBn2gs1s2<INPUT_T, T, isBasicBlock, layout>::Init(__gm__ uint8_t *query,
    __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse, __gm__ uint8_t *dropMask,
    __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
    __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe)
{
    this->BaseInit(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
        attentionOut, workspace, tiling, tPipe);

    int64_t workspaceBaseSize = this->tilingData->tensorSizeParams.bmm1ResUbSize * this->tilingData->coreParams.nRatio;
    this->pipe->InitSpmBuffer(this->workspaceGm[this->blockIdx * workspaceBaseSize], workspaceBaseSize * sizeof(T));
    int64_t bmm1WorkspaceSize = this->tilingData->tensorSizeParams.bmm1ResUbSize * this->tilingData->coreParams.nRatio;
    this->bmm1.SetWorkspace(this->workspaceGm[this->blockIdx * workspaceBaseSize].GetPhyAddr(), bmm1WorkspaceSize);

    // 计算轴的乘积
    this->n2G = this->tilingData->inputParams.n2Size * this->tilingData->inputParams.gSize;
    this->s1S2 = this->tilingData->inputParams.s1Size * this->tilingData->inputParams.s2Size;

    // 计算切分轴的乘积
    this->bBaseN2D = this->tilingData->coreParams.bBaseSize * this->tilingData->inputParams.n2Size *
        this->tilingData->inputParams.dSize;
    this->bBaseS1S2 = this->tilingData->coreParams.bBaseSize * this->tilingData->inputParams.s1Size *
        this->tilingData->inputParams.s2Size;
    this->bBaseN2GD = this->bBaseN2D * this->tilingData->inputParams.gSize;
    this->bBaseN2GS1 = this->tilingData->coreParams.bBaseSize * this->n2G * this->tilingData->inputParams.s1Size;
    this->bBaseN2GS2 = this->tilingData->coreParams.bBaseSize * this->n2G * this->tilingData->inputParams.s2Size;
    this->bBaseN2S2D = this->bBaseN2D * this->tilingData->inputParams.s2Size;
    this->bBaseN2GS1D = this->bBaseN2GS1 * this->tilingData->inputParams.dSize;
    this->bBaseN2GS2D = this->bBaseN2GS2 * this->tilingData->inputParams.dSize;
    this->bBaseN2GS1S2 = this->bBaseS1S2 * this->n2G;
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBBn2gs1s2<INPUT_T, T, isBasicBlock, layout>::Process()
{
    // 确定核内切分起点
    int64_t multiCoreInnerOffset = this->blockIdx * this->tilingData->multiCoreParams.splitFactorSize;
    int64_t multiCoreInnerLimit = this->Min(multiCoreInnerOffset + this->tilingData->multiCoreParams.splitFactorSize,
        this->tilingData->multiCoreParams.totalSize);

    for (int64_t multiCoreInnerIdx = multiCoreInnerOffset; multiCoreInnerIdx < multiCoreInnerLimit;
        multiCoreInnerIdx++) {
        // 计算核间切分轴的起始idx
        this->boIdx = multiCoreInnerIdx;

        // 核内计算
        this->ProcessBlock();
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBBn2gs1s2<INPUT_T, T, isBasicBlock, layout>::ProcessBlock()
{
    this->Bmm1SetTensorA();
    this->Bmm1SetTensorB();
    this->CalBatchSize();

    LocalTensor<INPUT_T> stage1ResUb = this->stage1ResQueue.template AllocTensor<INPUT_T>();
    this->ProcessStage1(stage1ResUb);
    this->ProcessStage2(stage1ResUb);
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBBn2gs1s2<INPUT_T, T, isBasicBlock, layout>::Bmm1SetTensorA()
{
    int64_t qCoreOffset = 0;
    // B,S,H -> B,S,N,D -> B,S1,N2,G,D
    if (this->tilingData->inputParams.layoutType == 1) {
        qCoreOffset = this->boIdx * this->bBaseN2GS1D;
        // S,B,H -> S,B,N,D -> S1,B,N2,G,D
    } else if (this->tilingData->inputParams.layoutType == 2) {
        qCoreOffset = this->boIdx * this->bBaseN2GD;
        // B,N,S,D -> B,N2,G,S1,D
    } else if (this->tilingData->inputParams.layoutType == 3) {
        qCoreOffset = this->boIdx * this->bBaseN2GS1D;
    }
    this->bmm1.SetTensorA(this->queryGm[qCoreOffset]);
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBBn2gs1s2<INPUT_T, T, isBasicBlock, layout>::Bmm1SetTensorB()
{
    // B,S,H -> B,S,N,D -> B,S2,N2,1,D
    if (this->tilingData->inputParams.layoutType == 1) {
        this->kvCoreOffset = this->boIdx * this->bBaseN2S2D;
        // S,B,H -> S,B,N,D -> S2,B,N2,1,D
    } else if (this->tilingData->inputParams.layoutType == 2) {
        this->kvCoreOffset = this->boIdx * this->bBaseN2D;
        // B,N,S,D -> B,N2,1,S2,D
    } else if (this->tilingData->inputParams.layoutType == 3) {
        this->kvCoreOffset = this->boIdx * this->bBaseN2S2D;
    }
    this->bmm1.SetTensorB(this->keyGm[this->kvCoreOffset], true);
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBBn2gs1s2<INPUT_T, T, isBasicBlock, layout>::CalBatchSize()
{
    int64_t bBaseTailSize = this->tilingData->coreParams.bBaseSize;
    if (unlikely(this->boIdx == this->tilingData->coreParams.bOuterSize - 1)) {
        bBaseTailSize = this->tilingData->coreParams.bBaseTailSize;
    }
    this->tensorABatchSize = bBaseTailSize * this->n2G;
    this->tensorBBatchSize = bBaseTailSize * this->tilingData->inputParams.n2Size;

    this->pseBatchSize = this->tensorABatchSize;

    // attenMask的shape为(B,N2,G,S1,S2),attenMaskBatchSize为bBaseTailSize*n2*G
    if (this->tilingData->inputParams.attenMaskShapeType == 0) {
        this->attenMaskBatchSize = this->tensorABatchSize;
        // attenMask的shape为(B,1,1,S1,S2),attenMaskBatchSize为bBaseTailSize
    } else if (this->tilingData->inputParams.attenMaskShapeType == 1) {
        this->attenMaskBatchSize = bBaseTailSize;
        // attenMask的shape为(1,1,1,S1,S2)
    } else if (this->tilingData->inputParams.attenMaskShapeType == 2) {
        this->attenMaskBatchSize = 1;
    }

    this->dropoutBatchSize = this->tensorABatchSize;
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBBn2gs1s2<INPUT_T, T, isBasicBlock, layout>::InitSoftmaxTensor()
{
    uint32_t maxSumShape[] = {static_cast<uint32_t>(this->tensorABatchSize),
        static_cast<uint32_t>(this->tilingData->inputParams.s1Size), 8};
    this->softmaxSumUb = this->softmaxOutQueue.template AllocTensor<float>();
    this->softmaxMaxUb = this->softmaxOutQueue.template AllocTensor<float>();

    this->softmaxMaxUb.SetShapeInfo(ShapeInfo(3, maxSumShape, DataFormat::ND));
    this->softmaxSumUb.SetShapeInfo(ShapeInfo(3, maxSumShape, DataFormat::ND));
}

// stage1
template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBBn2gs1s2<INPUT_T, T, isBasicBlock, layout>::ProcessStage1(
    LocalTensor<INPUT_T> &stage1ResUb)
{
    // bmm1Compute
    LocalTensor<T> bmm1ResUb = this->bmm1ResQueue.template AllocTensor<T>();
    this->bmm1.IterateBatch(bmm1ResUb, this->tensorABatchSize, this->tensorBBatchSize, true);
    this->SetResShape(bmm1ResUb, stage1ResUb);
    this->bmm1.End();

    // pseCompute
    if (this->hasPse) {
        int64_t pseOffset = ComputePseOffset();
        this->PseCopyIn(pseOffset, this->tilingData->inputParams.s1Size, this->tilingData->inputParams.s2Size,
            this->pseBatchSize);
        this->PseComputeWithBatch(bmm1ResUb, this->pseBatchSize, this->tilingData->inputParams.s1Size,
            this->tilingData->inputParams.s2Size);
    }

    // attenMaskCompute
    int64_t attenMaskOffset = ComputeAttenMaskOffset();
    this->AttenMaskCopyIn(attenMaskOffset, this->tilingData->inputParams.s1Size, this->tilingData->inputParams.s2Size,
        this->attenMaskBatchSize);
    this->AttenMaskCompute(bmm1ResUb, this->tilingData->inputParams.s1Size, this->tilingData->inputParams.s2Size,
        this->attenMaskBatchSize);

    this->InitSoftmaxTensor();
    if (this->hasDropMask) {
        int64_t dropMaskOffset = this->boIdx * this->bBaseN2GS1S2;
        this->DropoutCopyIn(dropMaskOffset, this->tilingData->inputParams.s1Size, this->tilingData->inputParams.s2Size);
        this->SoftmaxCompute(bmm1ResUb, bmm1ResUb);
        if constexpr (AscendC::IsSameType<INPUT_T, T>::value) {
            this->DropoutCompute(stage1ResUb, bmm1ResUb, this->tilingData->inputParams.s1Size,
                this->tilingData->inputParams.s2Size, this->dropoutBatchSize);
        } else {
            this->DropoutCompute(bmm1ResUb, bmm1ResUb, this->tilingData->inputParams.s1Size,
                this->tilingData->inputParams.s2Size, this->dropoutBatchSize);
            this->Stage1Cast(stage1ResUb, bmm1ResUb);
        }
    } else {
        if constexpr (AscendC::IsSameType<INPUT_T, T>::value) {
            this->SoftmaxCompute(stage1ResUb, bmm1ResUb);
        } else {
            this->SoftmaxCompute(bmm1ResUb, bmm1ResUb);
            this->Stage1Cast(stage1ResUb, bmm1ResUb);
        }
    }

    this->bmm1ResQueue.FreeTensor(bmm1ResUb);
    return;
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBBn2gs1s2<INPUT_T, T, isBasicBlock, layout>::SetResShape(
    LocalTensor<T> &bmm1ResUb, LocalTensor<INPUT_T> &stage1ResUb)
{
    uint32_t resShape[] = {static_cast<uint32_t>(this->tensorABatchSize),
        static_cast<uint32_t>(this->tilingData->inputParams.s1Size),
        static_cast<uint32_t>(this->tilingData->inputParams.s2Size)};
    bmm1ResUb.SetShapeInfo(ShapeInfo(3, resShape, DataFormat::ND));
    stage1ResUb.SetShapeInfo(ShapeInfo(3, resShape, DataFormat::ND));
    return;
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline int64_t FlashAttentionScoreSplitBBn2gs1s2<INPUT_T, T, isBasicBlock, layout>::ComputePseOffset()
{
    // 0: (B,N2,G,S1,S2)
    if (this->tilingData->inputParams.pseShapeType == 0) {
        return this->boIdx * this->bBaseN2GS1S2;
    }

    // 1: (B,N2,G,1,S2)
    if (this->tilingData->inputParams.pseShapeType == 1) {
        return this->boIdx * this->bBaseN2GS2;
    }
    return 0;
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline int64_t FlashAttentionScoreSplitBBn2gs1s2<INPUT_T, T, isBasicBlock, layout>::ComputeAttenMaskOffset()
{
    if (this->tilingData->inputParams.attenMaskCompressMode == this->attenMaskNoCompress) {
        return this->ComputeOffsetForNoCompress();
    } else if (this->tilingData->inputParams.attenMaskCompressMode == this->attenMaskLeftUpCausalCompress) {
        return 0;
    } else if (this->tilingData->inputParams.attenMaskCompressMode == this->attenMaskRightDownCausalCompress) {
        int64_t delta = this->tilingData->inputParams.s2Size - this->tilingData->inputParams.s1Size;
        return this->ComputeOffsetForCausal(delta, this->tilingData->inputParams.s1Size,
                                            this->tilingData->inputParams.s2Size,
                                            this->tilingData->inputParams.attenMaskS2Size);
    }
    return 0;
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline int64_t FlashAttentionScoreSplitBBn2gs1s2<INPUT_T, T, isBasicBlock, layout>::ComputeOffsetForNoCompress()
{
    // 0: (B,N2,G,S1,S2)
    if (this->tilingData->inputParams.attenMaskShapeType == 0) {
        return this->boIdx * this->bBaseN2GS1S2;
    }
    // 1: (B,1,1,S1,S2)
    if (this->tilingData->inputParams.attenMaskShapeType == 1) {
        return this->boIdx * this->bBaseS1S2;
    }
    // 2: (1,1,1,S1,S2)
    if (this->tilingData->inputParams.attenMaskShapeType == 2) {
        return this->boIdx * this->s1S2;
    }
    return 0;
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBBn2gs1s2<INPUT_T, T, isBasicBlock, layout>::SoftmaxCompute(
    LocalTensor<T> &dstTensor, LocalTensor<T> &srcTensor)
{
    LocalTensor<uint8_t> apiTmpBuffer = this->apiTBuf.template Get<uint8_t>();
    SoftMax<T, true, isBasicBlock>(dstTensor, this->softmaxSumUb, this->softmaxMaxUb, srcTensor, apiTmpBuffer,
        this->tilingData->softmaxTilingData);
    int64_t softmaxGmOffset = this->boIdx * this->bBaseN2GS1 * 8;
    this->SoftmaxCopyOut(softmaxGmOffset, this->tensorABatchSize * this->tilingData->inputParams.s1Size * 8);
    this->softmaxOutQueue.FreeTensor(this->softmaxSumUb);
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBBn2gs1s2<INPUT_T, T, isBasicBlock, layout>::ProcessStage2(
    LocalTensor<INPUT_T> &stage1ResUb)
{
    Bmm2(stage1ResUb);
    this->stage1ResQueue.FreeTensor(stage1ResUb);
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBBn2gs1s2<INPUT_T, T, isBasicBlock, layout>::Bmm2(
    LocalTensor<INPUT_T> &stage1ResUb)
{
    this->bmm2.SetTensorA(stage1ResUb);
    this->bmm2.SetTensorB(this->valueGm[this->kvCoreOffset]);
    this->bmm2.template IterateBatch<true>(this->attentionOutGm[this->kvCoreOffset], this->tensorABatchSize,
        this->tensorBBatchSize, false);
    this->bmm2.End();
}
#endif // FLASH_ATTENTION_SCORE_SPLIT_B_H