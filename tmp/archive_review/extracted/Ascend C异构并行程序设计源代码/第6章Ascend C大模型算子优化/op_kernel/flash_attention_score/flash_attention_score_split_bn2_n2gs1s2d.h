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

#ifndef FLASH_ATTENTION_SCORE_SPLIT_B_N2
#define FLASH_ATTENTION_SCORE_SPLIT_B_N2

#include "flash_attention_score_common.h"
#include "kernel_data_copy_transpose.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

using matmul::MatmulType;

// INPUT_T - means data type for input
// T       - means data type when calc
template <typename INPUT_T, typename T = INPUT_T, bool isBasicBlock = false, LayoutMode layout = LayoutMode::BNGS1S2>
class FlashAttentionScoreSplitBn2gs1s2N2gs1s2 : public FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock> {
public:
    __aicore__ inline FlashAttentionScoreSplitBn2gs1s2N2gs1s2() {};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
        __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax,
        __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
        const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();
    // define batchmatmul
    using a1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false, layout>;
    using b1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, true, layout>;
    using bias1Type = MatmulType<TPosition::GM, CubeFormat::ND, float, false, LayoutMode::BNGS1S2>;
    using c1Type = MatmulType<TPosition::VECCALC, CubeFormat::ND_ALIGN, T, false, LayoutMode::BNGS1S2>;
    matmul::Matmul<a1Type, b1Type, c1Type, bias1Type> bmm1;

    using a2Type = MatmulType<TPosition::VECCALC, CubeFormat::ND, INPUT_T, false, LayoutMode::BNGS1S2>;
    using b2Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false, layout>;
    using bias2Type = MatmulType<TPosition::GM, CubeFormat::ND, float, false, layout>;
    using c2Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false, layout>;
    matmul::Matmul<a2Type, b2Type, c2Type, bias2Type> bmm2;

protected:
    __aicore__ inline void ComputeAxisIdx(int64_t multiCoreInnerIdx);
    __aicore__ inline void ProcessBlock();
    __aicore__ inline void InitSoftmaxTensor();
    __aicore__ inline void Bmm1SetTensorA();
    __aicore__ inline void Bmm1SetTensorB();
    __aicore__ inline void CalBatchSize();
    __aicore__ inline void ProcessStage1(LocalTensor<INPUT_T> &stage1ResUb);
    __aicore__ inline int64_t ComputePseOffset();
    __aicore__ inline int64_t ComputeDropOffset();
    __aicore__ inline int64_t ComputeAttenMaskOffset();
    __aicore__ inline int64_t ComputeOffsetForNoCompress();
    __aicore__ inline void SetResShape(LocalTensor<T> &bmm1ResUb, LocalTensor<INPUT_T> &stage1ResUb);
    __aicore__ inline void SoftMaxCompute(LocalTensor<T> &dstTensor, LocalTensor<T> &srcTensor);
    __aicore__ inline void ProcessStage2(LocalTensor<INPUT_T> &stage1ResUb);
    __aicore__ inline void Bmm2(LocalTensor<INPUT_T> &stage1ResUb);

    // bmmTensorA的batch大小
    int64_t tensorABatchSize;

    // bmmTensorB的batch大小
    int64_t tensorBBatchSize;

    // pse载入ub的batch大小
    int64_t pseBatchSize;

    // attenmask载入ub的batch大小
    int64_t attenMaskBatchSize;

    // dropout载入ub的batch大小
    int64_t dropoutBatchSize;

    // q矩阵在核间的offset
    int64_t qCoreOffset;

    // k和v矩阵在核间的offset
    int64_t kvCoreOffset;

    // 轴的乘积
    int64_t s1S2;
    int64_t gS1D;
    int64_t gS2D;
    int64_t gS1S2;
    int64_t gD;
    int64_t gS1;
    int64_t gS2;
    int64_t n2D;
    int64_t n2GD;
    int64_t n2S1D;
    int64_t n2GS1D;
    int64_t n2S2D;
    int64_t n2GS2D;
    int64_t n2GS1S2;
    int64_t n2GS1;
    int64_t n2GS2;

    // 切分轴的乘积
    int64_t n2BaseD;
    int64_t n2BaseS2D;
    int64_t n2BaseGS1;
    int64_t n2BaseGS2;
    int64_t n2BaseGD;
    int64_t n2BaseGS1D;
    int64_t n2BaseGS2D;
    int64_t n2BaseGS1S2;
};

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2N2gs1s2<INPUT_T, T, isBasicBlock, layout>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse, __gm__ uint8_t *dropMask,
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
    this->s1S2 = this->tilingData->inputParams.s1Size * this->tilingData->inputParams.s2Size;
    this->gS1D = this->tilingData->inputParams.gSize * this->tilingData->inputParams.s1Size *
        this->tilingData->inputParams.dSize;
    this->gS2D = this->tilingData->inputParams.gSize * this->tilingData->inputParams.s2Size *
        this->tilingData->inputParams.dSize;
    this->gS1S2 = this->tilingData->inputParams.gSize * this->s1S2;
    this->gD = this->tilingData->inputParams.gSize * this->tilingData->inputParams.dSize;
    this->gS1 = this->tilingData->inputParams.gSize * this->tilingData->inputParams.s1Size;
    this->gS2 = this->tilingData->inputParams.gSize * this->tilingData->inputParams.s2Size;
    this->n2D = this->tilingData->inputParams.n2Size * this->tilingData->inputParams.dSize;
    this->n2GD = this->tilingData->inputParams.n2Size * this->gD;
    this->n2S1D = this->tilingData->inputParams.n2Size;
    this->n2GS1D = this->tilingData->inputParams.n2Size * this->gS1D;
    this->n2S2D = this->tilingData->inputParams.n2Size * this->tilingData->inputParams.s2Size *
        this->tilingData->inputParams.dSize;
    this->n2GS2D = this->tilingData->inputParams.n2Size * this->gS2D;
    this->n2GS1S2 = this->tilingData->inputParams.n2Size * this->gS1S2;
    this->n2GS1 = this->tilingData->inputParams.n2Size * this->gS1;
    this->n2GS2 = this->tilingData->inputParams.n2Size * this->gS2;

    // 计算切分轴的乘积
    this->n2BaseD = this->tilingData->coreParams.n2BaseSize * this->tilingData->inputParams.dSize;
    this->n2BaseS2D = this->tilingData->coreParams.n2BaseSize * this->tilingData->inputParams.s2Size *
        this->tilingData->inputParams.dSize;
    this->n2BaseGS1 = this->tilingData->coreParams.n2BaseSize * this->gS1;
    this->n2BaseGS2 = this->tilingData->coreParams.n2BaseSize * this->gS2;
    this->n2BaseGD = this->tilingData->coreParams.n2BaseSize * this->gD;
    this->n2BaseGS1D = this->tilingData->coreParams.n2BaseSize * this->gS1D;
    this->n2BaseGS2D = this->tilingData->coreParams.n2BaseSize * this->gS2D;
    this->n2BaseGS1S2 = this->tilingData->coreParams.n2BaseSize * this->gS1S2;
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2N2gs1s2<INPUT_T, T, isBasicBlock, layout>::Process()
{
    // 确定核内切分起点
    int64_t multiCoreInnerOffset = this->blockIdx * this->tilingData->multiCoreParams.splitFactorSize;
    int64_t multiCoreInnerLimit = this->Min(multiCoreInnerOffset + this->tilingData->multiCoreParams.splitFactorSize,
        this->tilingData->multiCoreParams.totalSize);
    for (int64_t multiCoreInnerIdx = multiCoreInnerOffset; multiCoreInnerIdx < multiCoreInnerLimit;
        multiCoreInnerIdx++) {
        // 计算核间切分轴的起始idx
        this->ComputeAxisIdx(multiCoreInnerIdx);

        // 核内计算
        this->ProcessBlock();
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2N2gs1s2<INPUT_T, T, isBasicBlock, layout>::ComputeAxisIdx(
    int64_t multiCoreInnerIdx)
{
    this->boIdx = multiCoreInnerIdx / this->tilingData->coreParams.n2OuterSize;
    this->n2oIdx = multiCoreInnerIdx % this->tilingData->coreParams.n2OuterSize;
}


template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2N2gs1s2<INPUT_T, T, isBasicBlock, layout>::ProcessBlock()
{
    this->Bmm1SetTensorA();
    this->Bmm1SetTensorB();
    this->CalBatchSize();

    LocalTensor<INPUT_T> stage1ResUb = this->stage1ResQueue.template AllocTensor<INPUT_T>();
    this->ProcessStage1(stage1ResUb);
    this->ProcessStage2(stage1ResUb);
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2N2gs1s2<INPUT_T, T, isBasicBlock, layout>::InitSoftmaxTensor()
{
    uint32_t maxSumShape[] = {static_cast<uint32_t>(this->tensorABatchSize),
                            static_cast<uint32_t>(this->tilingData->inputParams.s1Size), 8};
    this->softmaxSumUb = this->softmaxOutQueue.template AllocTensor<float>();
    this->softmaxMaxUb = this->softmaxOutQueue.template AllocTensor<float>();

    this->softmaxMaxUb.SetShapeInfo(ShapeInfo(3, maxSumShape, DataFormat::ND));
    this->softmaxSumUb.SetShapeInfo(ShapeInfo(3, maxSumShape, DataFormat::ND));
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2N2gs1s2<INPUT_T, T, isBasicBlock, layout>::Bmm1SetTensorA()
{
    // 计算gm上的offset
    int64_t qBOffset = 0;
    int64_t qN2Offset = 0;

    if (this->tilingData->inputParams.layoutType == 1) {
        // BSH/BNGS1S2
        qBOffset = this->boIdx * this->n2GS1D;
        qN2Offset = this->n2oIdx * this->n2BaseGD;
    } else if (this->tilingData->inputParams.layoutType == 2) {
        // SBH/SBNGD
        qBOffset = this->boIdx * this->n2GD;
        qN2Offset = this->n2oIdx * this->n2BaseGD;
    } else if (this->tilingData->inputParams.layoutType == 3) {
        // BNGSD
        qBOffset = this->boIdx * this->n2GS1D;
        qN2Offset = this->n2oIdx * this->n2BaseGS1D;
    }

    this->qCoreOffset = qBOffset + qN2Offset;
    this->bmm1.SetTensorA(this->queryGm[this->qCoreOffset]);
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2N2gs1s2<INPUT_T, T, isBasicBlock, layout>::Bmm1SetTensorB()
{
    // 计算gm上的offset
    int64_t bOffset = 0;
    int64_t n2Offset = 0;

    if (this->tilingData->inputParams.layoutType == 1) {
        // BSH/BSND
        bOffset = this->boIdx * this->n2S2D;
        n2Offset = this->n2oIdx * this->n2BaseD;
    } else if (this->tilingData->inputParams.layoutType == 2) {
        // SBH/SBND
        bOffset = this->boIdx * this->n2D;
        n2Offset = this->n2oIdx * this->n2BaseD;
    } else if (this->tilingData->inputParams.layoutType == 3) {
        // BNSD
        bOffset = this->boIdx * this->n2S2D;
        n2Offset = this->n2oIdx * this->n2BaseS2D;
    }

    this->kvCoreOffset = bOffset + n2Offset;
    this->bmm1.SetTensorB(this->keyGm[this->kvCoreOffset], true);
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2N2gs1s2<INPUT_T, T, isBasicBlock, layout>::CalBatchSize()
{
    int64_t n2BaseTailSize = this->tilingData->coreParams.n2BaseSize;
    if (unlikely(this->n2oIdx == this->tilingData->coreParams.n2OuterSize - 1)) {
        n2BaseTailSize = this->tilingData->coreParams.n2BaseTailSize;
    }
    this->tensorABatchSize = n2BaseTailSize * this->tilingData->inputParams.gSize;
    this->tensorBBatchSize = n2BaseTailSize;

    this->pseBatchSize = this->tensorABatchSize;
    this->attenMaskBatchSize = this->tensorABatchSize;
    this->dropoutBatchSize = this->tensorABatchSize;
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2N2gs1s2<INPUT_T, T, isBasicBlock, layout>::ProcessStage1(
    LocalTensor<INPUT_T> &stage1ResUb)
{
    int64_t pseOffset = this->ComputePseOffset();
    this->PseCopyIn(pseOffset, this->tilingData->inputParams.s1Size, this->tilingData->inputParams.s2Size,
        this->pseBatchSize);

    // bmm1计算
    LocalTensor<T> bmm1ResUb = this->bmm1ResQueue.template AllocTensor<T>();
    this->bmm1.IterateBatch(bmm1ResUb, this->tensorABatchSize, this->tensorBBatchSize, true);
    this->SetResShape(bmm1ResUb, stage1ResUb);
    this->bmm1.End();

    // pse计算
    this->PseComputeWithBatch(bmm1ResUb, this->pseBatchSize, this->tilingData->inputParams.s1Size,
        this->tilingData->inputParams.s2Size);

    // attenmask计算
    int64_t attenMaskOffset = this->ComputeAttenMaskOffset();
    this->AttenMaskCopyIn(attenMaskOffset, this->tilingData->inputParams.s1Size, this->tilingData->inputParams.s2Size,
        this->attenMaskBatchSize);
    this->AttenMaskCompute(bmm1ResUb, this->tilingData->inputParams.s1Size, this->tilingData->inputParams.s2Size,
        this->attenMaskBatchSize);

    this->InitSoftmaxTensor();
    if (this->hasDropMask) {
        // softmax的计算结果需要执行dropout操作
        int64_t dropMaskOffset = this->ComputeDropOffset();
        this->DropoutCopyIn(dropMaskOffset, this->tilingData->inputParams.s1Size, this->tilingData->inputParams.s2Size,
                            this->dropoutBatchSize);
        // softmax当前max和expsum分开输出到gm
        this->SoftMaxCompute(bmm1ResUb, bmm1ResUb);

        // dropout计算
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
            // softmax当前max和expsum分开输出到gm
            this->SoftMaxCompute(stage1ResUb, bmm1ResUb);
        } else {
            this->SoftMaxCompute(bmm1ResUb, bmm1ResUb);
            this->Stage1Cast(stage1ResUb, bmm1ResUb);
        }
    }
    this->bmm1ResQueue.FreeTensor(bmm1ResUb);
    return;
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline int64_t FlashAttentionScoreSplitBn2gs1s2N2gs1s2<INPUT_T, T, isBasicBlock, layout>::ComputePseOffset()
{
    if (!this->hasPse) {
        return 0;
    }
    int64_t bOffset = this->boIdx * this->n2GS1S2;
    int64_t n2Offset = this->n2oIdx * this->n2BaseGS1S2;
    // 0: (B,N2,G,S1,S2), 1: (B,N2,G,1,S2)
    if (this->tilingData->inputParams.pseShapeType == 1) {
        bOffset = this->boIdx * this->n2GS2;
        n2Offset = this->n2oIdx * this->n2BaseGS2;
    }
    return bOffset + n2Offset;
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline int64_t
FlashAttentionScoreSplitBn2gs1s2N2gs1s2<INPUT_T, T, isBasicBlock, layout>::ComputeAttenMaskOffset()
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
__aicore__ inline int64_t
FlashAttentionScoreSplitBn2gs1s2N2gs1s2<INPUT_T, T, isBasicBlock, layout>::ComputeOffsetForNoCompress()
{
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    // 0 : (B,N2,G,S1,S2), 1: (B,1,1,S1,S2), 2: (1,1,1,S1,S2)
    if (this->tilingData->inputParams.attenMaskShapeType == 0) {
        bOffset = this->boIdx * this->n2GS1S2;
        n2Offset = this->n2oIdx * this->n2BaseGS1S2;
    } else if (this->tilingData->inputParams.attenMaskShapeType == 1) {
        bOffset = this->boIdx * this->s1S2;
    }
    return bOffset + n2Offset;
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline int64_t FlashAttentionScoreSplitBn2gs1s2N2gs1s2<INPUT_T, T, isBasicBlock, layout>::ComputeDropOffset()
{
    int64_t bOffset = this->boIdx * this->n2GS1S2;
    int64_t n2Offset = this->n2oIdx * this->n2BaseGS1S2;
    return bOffset + n2Offset;
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2N2gs1s2<INPUT_T, T, isBasicBlock, layout>::SoftMaxCompute(
    LocalTensor<T> &dstTensor, LocalTensor<T> &srcTensor)
{
    LocalTensor<uint8_t> apiTmpBuffer = this->apiTBuf.template Get<uint8_t>();
    SoftMax<T, true, isBasicBlock>(dstTensor, this->softmaxSumUb, this->softmaxMaxUb, srcTensor, apiTmpBuffer,
        this->tilingData->softmaxTilingData);
    int64_t softmaxBOffset = this->boIdx * this->n2GS1 * 8;
    int64_t softmaxN2Offset = this->n2oIdx * this->n2BaseGS1 * 8;
    int64_t softmaxGmOffset = softmaxBOffset + softmaxN2Offset;
    this->SoftmaxCopyOut(softmaxGmOffset, this->tensorABatchSize * this->tilingData->inputParams.s1Size * 8);
    this->softmaxOutQueue.FreeTensor(this->softmaxSumUb);
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2N2gs1s2<INPUT_T, T, isBasicBlock, layout>::SetResShape(
    LocalTensor<T> &bmm1ResUb, LocalTensor<INPUT_T> &stage1ResUb)
{
    uint32_t resShape[] = {static_cast<uint32_t>(this->tensorABatchSize),
                         static_cast<uint32_t>(this->tilingData->inputParams.s1Size),
                         static_cast<uint32_t>(this->tilingData->coreParams.s2BaseSize)};
    uint32_t resOrgShape[] = {static_cast<uint32_t>(this->tensorABatchSize),
                         static_cast<uint32_t>(this->tilingData->inputParams.s1Size),
                         static_cast<uint32_t>(this->tilingData->inputParams.s2Size)};
    bmm1ResUb.SetShapeInfo(ShapeInfo(3, resShape, 3, resOrgShape, DataFormat::ND));
    stage1ResUb.SetShapeInfo(ShapeInfo(3, resShape, DataFormat::ND));
    return;
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2N2gs1s2<INPUT_T, T, isBasicBlock, layout>::ProcessStage2(
    LocalTensor<INPUT_T> &stage1ResUb)
{
    Bmm2(stage1ResUb);
    this->stage1ResQueue.FreeTensor(stage1ResUb);
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2N2gs1s2<INPUT_T, T, isBasicBlock, layout>::Bmm2(
    LocalTensor<INPUT_T> &stage1ResUb)
{
    this->bmm2.SetTensorA(stage1ResUb);
    this->bmm2.SetTensorB(this->valueGm[this->kvCoreOffset]);
    this->bmm2.template IterateBatch(this->attentionOutGm[this->qCoreOffset], this->tensorABatchSize,
        this->tensorBBatchSize, false);
    this->bmm2.End();
}
#endif // FLASH_ATTENTION_SCORE_SPLIT_B_N2