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

#ifndef FLASH_ATTENTION_SCORE_COMMON_H
#define FLASH_ATTENTION_SCORE_COMMON_H

#include "kernel_data_copy_transpose.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "stdarg.h"

constexpr MatmulConfig CFG_EXCEED = GetNormalConfig(true);

// INPUT_T - means data type for input
// T       - means data type when calc
template <typename INPUT_T, typename T = INPUT_T, bool isBasicBlock = false>
class FlashAttentionScoreCommon {
public:
    __aicore__ inline FlashAttentionScoreCommon();
    __aicore__ inline void BaseInit(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
        __gm__ uint8_t *pse, __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask,
        __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut,
        __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
        const FlashAttentionScoreGeneralTilingData *__restrict tiling,
        TPipe *tPipe);

protected:
    __aicore__ inline void SetTiling(const FlashAttentionScoreGeneralTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffer();

    __aicore__ inline void ProcessBlock(int64_t multiCoreInnerIdx);

    __aicore__ inline void PseCopyIn(int64_t offset, int32_t s1Size, int32_t s2Size, int32_t batchSize = 1);
    __aicore__ inline void PseCompute(LocalTensor<T> &dstTensor, int32_t s1Size, int32_t s2Size);
    __aicore__ inline void PseBroadcastAdd(int32_t s1Size, int32_t s2Size, int32_t computeSize,
        const LocalTensor<T> &pseUb, const LocalTensor<T> &dstTensor);
    __aicore__ inline void BroadcastAdd(const LocalTensor<T> &src0Tensor, const LocalTensor<T> &src1Tensor,
        int64_t src0Offset, int32_t src1Size, int32_t repeatTimes);

    __aicore__ inline void DropoutCopyIn(int64_t offset, int32_t s1Size, int32_t s2Size, int32_t batchSize = 1);
    __aicore__ inline void DropoutCompute(LocalTensor<T> &dstTensor, LocalTensor<T> &srcTensor, int32_t s1Size,
        int32_t s2Size, int32_t batchSize = 1);

    __aicore__ inline void AttenMaskCopyIn(int64_t offset, int32_t s1Size, int32_t s2Size, int32_t batchSize = 1);
    __aicore__ inline int64_t ComputeOffsetForCausal(const int64_t &delta, const uint32_t &s1BaseSize,
                                                     const uint32_t &s2BaseSize, const uint32_t &attenMaskS2Size);
    __aicore__ inline void AttenMaskCompute(LocalTensor<T> &dstTensor, int32_t s1Size, int32_t s2Size,
        int32_t batchSize = 1);

    __aicore__ inline void Stage1Cast(LocalTensor<INPUT_T> &dstTensor, LocalTensor<T> &srcTensor);

    __aicore__ inline void Bit2Int8CopyIn(LocalTensor<uint8_t> &dstTensor, GlobalTensor<uint8_t> &srcTensor,
        int64_t offset, int32_t s1Size, int32_t s2Size, int32_t batchSize = 1);
    __aicore__ inline void DataCopyIn(LocalTensor<INPUT_T> &dstTensor, GlobalTensor<INPUT_T> &srcTensor, int64_t offset,
        int32_t s1Size, int32_t s2Size);

    __aicore__ inline void SoftmaxCopyOut(int64_t gmOffset, int64_t calculateSize);

    __aicore__ inline int32_t Align(int32_t shape);

    __aicore__ inline void BroadcastBatchCopyIn(LocalTensor<INPUT_T> &dstTensor, GlobalTensor<INPUT_T> &srcTensor,
                                                int64_t offset, int32_t batchSize, int32_t s1Size, int32_t s2Size);

    __aicore__ inline void PseBroadcastAddWithBatch(int32_t batchSize, int32_t s1Size, int32_t s2Size,
        int32_t computeSize, const LocalTensor<T> &pseUb, const LocalTensor<T> &dstTensor);
    __aicore__ inline void PseComputeWithBatch(LocalTensor<T> &dstTensor, int32_t batchSize, int32_t s1Size,
        int32_t s2Size);
    __aicore__ inline void BoolCopyIn(LocalTensor<uint8_t> &dstTensor, GlobalTensor<uint8_t> &srcTensor, int64_t offset,
        int32_t s1Size, int32_t s2Size, uint64_t totalS2Size, int32_t batchSize = 1);
    __aicore__ inline void DataCopyInWithBatch(LocalTensor<INPUT_T> &dstTensor, GlobalTensor<INPUT_T> &srcTensor,
        int64_t offset, int32_t batchSize, int32_t s1Size, int32_t s2Size);

    __aicore__ inline void AttenMaskInputTCompute(LocalTensor<T> &dstTensor, int32_t computeSize);

    __aicore__ inline void AttenMaskBoolCompute(LocalTensor<T> &dstTensor, int32_t s1Size, int32_t s2Size,
        int32_t batchSize = 1);

    __aicore__ inline void BoolCopyInBatchBroadcast(LocalTensor<uint8_t> &dstTensor, GlobalTensor<uint8_t> &srcTensor,
        int64_t offset, int32_t s1Size, int32_t s2Size, uint64_t totalS2Size, int32_t batchSize);

    __aicore__ inline void BoolCopyInWithBatch(LocalTensor<uint8_t> &dstTensor, GlobalTensor<uint8_t> &srcTensor,
        int64_t offset, int32_t s1Size, int32_t s2Size, int32_t batchSizem);

    template <typename T1, typename T2>
	__aicore__ inline T1 CeilDiv(T1 a, T2 b)
    {
        if (b == 0) {
            return 0;
        }
        return (a + b - 1) / b;
    };

    template <typename T1, typename T2>
	__aicore__ inline T1 Max(T1 a, T2 b)
    {
        return (a > b) ? (a) : (b);
    };

    template <typename T1, typename T2>
	__aicore__ inline T1 Min(T1 a, T2 b)
    {
        return (a > b) ? (b) : (a);
    };

    int32_t blockIdx;

    const FlashAttentionScoreGeneralTilingData *__restrict tilingData;

    int64_t boIdx;
    int64_t n2oIdx;
    int64_t goIdx;
    int64_t s1oIdx;

    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> optionalInputQueue; // 可选输入复用，包含pse, attenmask, dropmask
    TQue<QuePosition::VECIN, 1> bmm1ResQueue;
    TQue<QuePosition::VECOUT, 1> stage1ResQueue;
    TQue<QuePosition::VECOUT, 2> softmaxOutQueue; // 2个输出,sum和max

    TBuf<> apiTBuf;
    TBuf<> typeCastTBuf;

    GlobalTensor<INPUT_T> queryGm;
    GlobalTensor<INPUT_T> keyGm;
    GlobalTensor<T> workspaceGm;
    GlobalTensor<INPUT_T> pseGm;
    GlobalTensor<INPUT_T> valueGm;
    GlobalTensor<INPUT_T> attentionOutGm;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    GlobalTensor<uint8_t> dropMaskGm;
    GlobalTensor<INPUT_T> attenMaskGm;
    GlobalTensor<uint8_t> attenMaskGmInt;

    // 资源分配
    LocalTensor<float> softmaxSumUb;
    LocalTensor<float> softmaxMaxUb;

    constexpr static int32_t blockBytes = 32;
    constexpr static int32_t byteBitRatio = 8;
    constexpr static int32_t fp32BaseSize = 8;
    constexpr static int32_t repeatMaxBytes = 256;
    constexpr static int32_t repeatMaxTimes = 255;

    bool hasPse;
    bool hasAttenMask;
    bool hasDropMask;
    bool isInputTAttenMask;
    int32_t repeatMaxSize;

    // 0级接口的block间隔范围需要满足32B对齐
    int32_t blockSize = blockBytes / sizeof(T);

    // option input的layout
    int64_t pseS1S2 = 0;
    int64_t pse1S2 = 1;

    int64_t attenMaskBN2GS1S2 = 0;
    int64_t attenMaskBS1S2 = 1;
    int64_t attenMaskS1S2 = 2;

    uint32_t attenMaskNoCompress = 0;
    uint32_t attenMaskLeftUpCausalCompress = 1;
    uint32_t attenMaskRightDownCausalCompress = 2;
};

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock>::FlashAttentionScoreCommon() {}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock>::BaseInit(__gm__ uint8_t *query,
    __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse, __gm__ uint8_t *dropMask,
    __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
    __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe)
{
    this->blockIdx = GetBlockIdx();
    this->pipe = tPipe;
    this->repeatMaxSize = repeatMaxBytes / sizeof(T);
    this->SetTiling(tiling);

    // init global buffer
    this->queryGm.SetGlobalBuffer(reinterpret_cast<__gm__ INPUT_T *>(query));
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

    this->hasPse = pseGm.GetPhyAddr() != nullptr;
    this->hasAttenMask = attenMaskGm.GetPhyAddr() != nullptr;
    this->hasDropMask = dropMaskGm.GetPhyAddr() != nullptr;
    this->isInputTAttenMask = this->tilingData->inputParams.attenMaskDataType == 0;
    this->InitBuffer();

    LocalTensor<T> apiTmpBuffer = this->apiTBuf.template Get<T>();
    DropOutBitModeInit(apiTmpBuffer);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline int32_t FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock>::Align(int32_t shape)
{
    int32_t alignFactor = 16;
    int32_t alignedSize = this->CeilDiv(shape, alignFactor) * alignFactor;
    return alignedSize;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock>::SetTiling(
    const FlashAttentionScoreGeneralTilingData *__restrict tilingData)
{
    // copy base params
    this->tilingData = tilingData;
};

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock>::InitBuffer()
{
    if (this->hasPse || this->hasAttenMask || this->hasDropMask) {
        int32_t optionalUbSize = this->tilingData->tensorSizeParams.attenMaskUbSize;
        if (this->tilingData->tensorSizeParams.pseUbSize > optionalUbSize) {
            optionalUbSize = this->tilingData->tensorSizeParams.pseUbSize;
        }
        if (this->tilingData->tensorSizeParams.dropMaskUbSize > optionalUbSize) {
            optionalUbSize = this->tilingData->tensorSizeParams.dropMaskUbSize;
        }
        this->pipe->InitBuffer(this->optionalInputQueue, 1, optionalUbSize * sizeof(INPUT_T));
    }
    this->pipe->InitBuffer(this->bmm1ResQueue, 1, this->tilingData->tensorSizeParams.bmm1ResUbSize * sizeof(T));
    this->pipe->InitBuffer(this->stage1ResQueue, 1, this->tilingData->tensorSizeParams.bmm1ResUbSize * sizeof(INPUT_T));

    this->pipe->InitBuffer(this->softmaxOutQueue, 2,
        this->tilingData->tensorSizeParams.softmaxMaxUbSize * sizeof(float));
    this->pipe->InitBuffer(this->apiTBuf, this->tilingData->tensorSizeParams.apiTmpBufferBytes);
    if constexpr (!IsSameType<INPUT_T, T>::value) {
        this->pipe->InitBuffer(this->typeCastTBuf, this->tilingData->tensorSizeParams.castUbSize * sizeof(T)); // fp32
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock>::PseCopyIn(int64_t offset, int32_t s1Size,
    int32_t s2Size, int32_t batchSize)
{
    if (!this->hasPse) {
        return;
    }

    LocalTensor<INPUT_T> pseUb = this->optionalInputQueue.template AllocTensor<INPUT_T>();
    s1Size = this->tilingData->inputParams.pseShapeType == this->pse1S2 ? 1 : s1Size;
    if (batchSize > 1) {
        this->DataCopyInWithBatch(pseUb, this->pseGm, offset, batchSize, s1Size, s2Size);
    } else {
        this->DataCopyIn(pseUb, this->pseGm, offset, s1Size, s2Size);
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock>::PseCompute(LocalTensor<T> &dstTensor,
    int32_t s1Size, int32_t s2Size)
{
    if (!this->hasPse) {
        return;
    }

    LocalTensor<INPUT_T> pseUb = this->optionalInputQueue.template DeQue<INPUT_T>();
    s2Size = this->Align(s2Size);
    int32_t computeSize = this->tilingData->inputParams.pseShapeType == this->pseS1S2 ? s1Size * s2Size : s2Size;
    if constexpr (!IsSameType<INPUT_T, T>::value) {
        LocalTensor<T> tmpPseCastTensor = this->typeCastTBuf.template Get<T>();
        Cast(tmpPseCastTensor, pseUb, RoundMode::CAST_NONE, computeSize);
        pipe_barrier(PIPE_V);
        this->PseBroadcastAdd(s1Size, s2Size, computeSize, tmpPseCastTensor, dstTensor);
    } else {
        this->PseBroadcastAdd(s1Size, s2Size, computeSize, pseUb, dstTensor);
    }
    this->optionalInputQueue.FreeTensor(pseUb);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock>::PseBroadcastAdd(int32_t s1Size,
    int32_t s2Size, int32_t computeSize, const LocalTensor<T> &pseUb, const LocalTensor<T> &dstTensor)
{
    if (this->tilingData->inputParams.pseShapeType == this->pseS1S2) {
        Add(dstTensor, dstTensor, pseUb, computeSize);
    } else {
        /* Total repeated times should be <= this->repeatMaxTimes. If larger,
         * we need to do multiple inner loops. */
        int32_t s1OuterLoop = s1Size / this->repeatMaxTimes;
        int32_t s1OuterRemain = s1Size % this->repeatMaxTimes;
        for (int32_t s1OuterIdx = 0; s1OuterIdx < s1OuterLoop; s1OuterIdx++) {
            int32_t s1OuterOffset = (s1OuterIdx << 8) * s2Size;
            this->BroadcastAdd(dstTensor, pseUb, s1OuterOffset, s2Size, this->repeatMaxTimes);
        }
        if (s1OuterRemain > 0) {
            int32_t s1OuterOffset = (s1OuterLoop << 8) * s2Size;
            this->BroadcastAdd(dstTensor, pseUb, s1OuterOffset, s2Size, s1OuterRemain);
        }
    }
}

/*
dst = BroadcastAdd(src0, src1)
src0 shape: (s1, s2)
src1 shape: (1, s2)
dst  shape: (s1, s2)
*/
template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock>::BroadcastAdd(
    const LocalTensor<T> &src0Tensor, const LocalTensor<T> &src1Tensor, int64_t src0Offset, int32_t src1Size,
    int32_t repeatTimes)
{
    /* Total data number of single step should be smaller than 256bytes.
     * If larger, we need to do add multiple times. */
    int32_t innerLoop = src1Size / this->repeatMaxSize;   // s2轴整块计算次数
    int32_t innerRemain = src1Size % this->repeatMaxSize; // s2轴尾块计算量
    BinaryRepeatParams binaryRepeatParams;
    binaryRepeatParams.src0BlkStride = 1;
    binaryRepeatParams.src0RepStride = src1Size / blockSize;
    binaryRepeatParams.src1BlkStride = 1;
    binaryRepeatParams.src1RepStride = 0;
    binaryRepeatParams.dstRepStride = binaryRepeatParams.src0RepStride;
    binaryRepeatParams.blockNumber = binaryRepeatParams.src0RepStride;

    for (int32_t j = 0; j < innerLoop; j++) {
        auto innerOffset = j * this->repeatMaxSize;
        auto ubOffset = src0Offset + innerOffset;
        Add(src0Tensor[ubOffset], src0Tensor[ubOffset], src1Tensor[innerOffset], this->repeatMaxSize, repeatTimes,
            binaryRepeatParams);
    }
    if (innerRemain > 0) {
        auto innerOffset = innerLoop * this->repeatMaxSize;
        auto ubOffset = src0Offset + innerOffset;
        Add(src0Tensor[ubOffset], src0Tensor[ubOffset], src1Tensor[innerOffset], innerRemain, repeatTimes,
            binaryRepeatParams);
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock>::DropoutCopyIn(int64_t offset,
    int32_t s1Size, int32_t s2Size, int32_t batchSize)
{
    if (!this->hasDropMask) {
        return;
    }

    LocalTensor<uint8_t> dropMaskUb = this->optionalInputQueue.template AllocTensor<uint8_t>();
    // 仅支持对齐场景，非对齐使用boolCopyIn
    this->Bit2Int8CopyIn(dropMaskUb, this->dropMaskGm, offset, s1Size, s2Size, batchSize);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock>::DropoutCompute(LocalTensor<T> &dstTensor,
    LocalTensor<T> &srcTensor, int32_t s1Size, int32_t s2Size, int32_t batchSize)
{
    if (!this->hasDropMask) {
        return;
    }

    LocalTensor<uint8_t> dropMaskUb = this->optionalInputQueue.template DeQue<uint8_t>();
    DropOutShapeInfo dropOutShapeInfo;
    dropOutShapeInfo.firstAxis = batchSize * s1Size;
    dropOutShapeInfo.srcLastAxis = s2Size;

    dropOutShapeInfo.maskLastAxis = this->CeilDiv(s2Size / byteBitRatio, blockBytes) * blockBytes;
    LocalTensor<uint8_t> apiTmpBuffer = this->apiTBuf.template Get<uint8_t>();
    if (isInputTAttenMask) {
        DropOut<T, true>(dstTensor, srcTensor, dropMaskUb, apiTmpBuffer, tilingData->inputParams.keepProb,
            dropOutShapeInfo);
    } else {
        DropOut(dstTensor, srcTensor, dropMaskUb, apiTmpBuffer, tilingData->inputParams.keepProb,
            dropOutShapeInfo);
    }

    this->optionalInputQueue.FreeTensor(dropMaskUb);
}

// bit类型的datacopyin, 只支持对齐场景
template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock>::Bit2Int8CopyIn(
    LocalTensor<uint8_t>& dstTensor, GlobalTensor<uint8_t>& srcTensor, int64_t offset, int32_t s1Size, int32_t s2Size,
    int32_t batchSize)
{
    int32_t alignedS2Size = this->CeilDiv(s2Size, 32) * 32;
    uint32_t shapeArray[] = {static_cast<uint32_t>(batchSize * s1Size), static_cast<uint32_t>(alignedS2Size)};
    dstTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    dstTensor.SetSize(batchSize * s1Size * alignedS2Size);
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = batchSize * s1Size;
    dataCopyParams.blockLen = this->CeilDiv(s2Size / byteBitRatio, blockBytes);
    dataCopyParams.dstStride = 0;  // gap
    if (tilingData->inputParams.s2Size / byteBitRatio % blockBytes == 0) {
        dataCopyParams.srcStride = (tilingData->inputParams.s2Size /
        byteBitRatio - dataCopyParams.blockLen * blockBytes) / blockBytes;  // srcGap
        DataCopy(dstTensor, srcTensor[offset / byteBitRatio], dataCopyParams);
    } else {
        dataCopyParams.blockLen = s2Size / byteBitRatio;
        dataCopyParams.srcStride = (tilingData->inputParams.s2Size / byteBitRatio - dataCopyParams.blockLen);
        DataCopyPadParams dataCopyPadParams;
        DataCopyPad(dstTensor, srcTensor[offset / byteBitRatio], dataCopyParams, dataCopyPadParams);
    }
    optionalInputQueue.EnQue(dstTensor);
}

// 通用datacopyin
template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock>::DataCopyIn(LocalTensor<INPUT_T> &dstTensor,
    GlobalTensor<INPUT_T> &srcTensor, int64_t offset, int32_t s1Size, int32_t s2Size)
{
    int32_t dtypeSize = sizeof(INPUT_T);
    int32_t alignedS2Size = this->Align(s2Size);
    uint32_t shapeArray[] = {static_cast<uint32_t>(s1Size), static_cast<uint32_t>(alignedS2Size)};
    dstTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    dstTensor.SetSize(s1Size * alignedS2Size);
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = s1Size;
    dataCopyParams.blockLen = this->CeilDiv(s2Size * dtypeSize, blockBytes); // 单位32B
    dataCopyParams.dstStride = 0;                                            // gap
    if (tilingData->inputParams.s2Size * dtypeSize % blockBytes == 0) {
        dataCopyParams.srcStride =
            (tilingData->inputParams.s2Size * dtypeSize - dataCopyParams.blockLen * blockBytes) / blockBytes; // srcGap
        DataCopy(dstTensor, srcTensor[offset], dataCopyParams);
    } else {
        dataCopyParams.blockLen = s2Size * dtypeSize; // 单位Byte
        dataCopyParams.srcStride = (tilingData->inputParams.s2Size * dtypeSize - dataCopyParams.blockLen);
        dataCopyParams.dstStride = (alignedS2Size - s2Size) * dtypeSize;
        DataCopyPadParams dataCopyPadParams;
        dataCopyPadParams.rightPadding = (alignedS2Size - s2Size) * dtypeSize;
        DataCopyPad(dstTensor, srcTensor[offset], dataCopyParams, dataCopyPadParams);
    }
    optionalInputQueue.EnQue(dstTensor);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock>::AttenMaskCopyIn(int64_t offset,
    int32_t s1Size, int32_t s2Size, int32_t batchSize)
{
    if (!this->hasAttenMask) {
        return;
    }
    if (isInputTAttenMask) { // INPUT_T类型输入
        LocalTensor<INPUT_T> attenMaskUb = this->optionalInputQueue.template AllocTensor<INPUT_T>();
        if (batchSize > 1) {
            if (tilingData->inputParams.attenMaskShapeType == this->attenMaskBN2GS1S2) {
                this->DataCopyInWithBatch(attenMaskUb, attenMaskGm, offset, batchSize, s1Size, s2Size);
            } else {
                this->BroadcastBatchCopyIn(attenMaskUb, attenMaskGm, offset, batchSize, s1Size, s2Size);
            }
        } else {
            this->DataCopyIn(attenMaskUb, attenMaskGm, offset, s1Size, s2Size);
        }
    } else { // bool类型输入
        LocalTensor<uint8_t> attenMaskUb = this->optionalInputQueue.template AllocTensor<uint8_t>();
        if (batchSize > 1) {
            BoolCopyInWithBatch(attenMaskUb, attenMaskGmInt, offset, s1Size, s2Size, batchSize); // todo 待支持batch
        } else {
            BoolCopyIn(attenMaskUb, attenMaskGmInt, offset, s1Size, s2Size, tilingData->inputParams.attenMaskS2Size);
        }
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline int64_t FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock>::ComputeOffsetForCausal(
    const int64_t &delta, const uint32_t &s1BaseSize, const uint32_t &s2BaseSize, const uint32_t &attenMaskS2Size)
{
    if (delta <= 0) {
        return this->Min(-1 * delta, s1BaseSize);
    } else {
        return this->Min(delta, s2BaseSize) * attenMaskS2Size;
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock>::AttenMaskBoolCompute(
    LocalTensor<T> &dstTensor, int32_t s1Size, int32_t s2Size, int32_t batchSize)
{
    LocalTensor<uint8_t> attenMaskUb = optionalInputQueue.template DeQue<uint8_t>();
    int32_t alignedS2Size = this->CeilDiv(s2Size, blockBytes) * blockBytes;
    uint32_t shapeArray[] = {static_cast<uint32_t>(batchSize * s1Size), static_cast<uint32_t>(alignedS2Size)};
    attenMaskUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    attenMaskUb.SetSize(batchSize * s1Size * alignedS2Size);
    dstTensor.SetSize(batchSize * s1Size * s2Size);
    LocalTensor<uint8_t> apiTmpBuffer = apiTBuf.template Get<uint8_t>();
    T scalar;
    if constexpr (IsSameType<T, float>::value) {
        uint32_t tmp = 0xFF7FFFFF;
        scalar = *((float *)&tmp);
    } else {
        uint16_t tmp = 0xFBFF;
        scalar = *((half *)&tmp);
    }
    // 待实现batch
    SelectWithBytesMaskShapeInfo shapeInfo;
    shapeInfo.firstAxis = batchSize * s1Size;
    shapeInfo.srcLastAxis = s2Size;
    shapeInfo.maskLastAxis = alignedS2Size;
    SelectWithBytesMask(dstTensor, dstTensor, scalar, attenMaskUb, apiTmpBuffer, shapeInfo);
    optionalInputQueue.FreeTensor(attenMaskUb);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock>::AttenMaskInputTCompute(
    LocalTensor<T> &dstTensor, int32_t computeSize)
{
    LocalTensor<INPUT_T> attenMaskUb = optionalInputQueue.template DeQue<INPUT_T>();

    if constexpr (!IsSameType<INPUT_T, T>::value) {
        LocalTensor<T> attenCastTensor = this->typeCastTBuf.template Get<T>();
        Cast(attenCastTensor, attenMaskUb, RoundMode::CAST_NONE, computeSize);
        Muls(attenCastTensor, attenCastTensor, static_cast<T>(-10000.0), computeSize);
        pipe_barrier(PIPE_V);
        Add(dstTensor, dstTensor, attenCastTensor, computeSize);
    } else {
        Muls(attenMaskUb, attenMaskUb, static_cast<T>(-10000.0), computeSize);
        pipe_barrier(PIPE_V);
        Add(dstTensor, dstTensor, attenMaskUb, computeSize);
    }
    optionalInputQueue.FreeTensor(attenMaskUb);
}


template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock>::AttenMaskCompute(LocalTensor<T> &dstTensor,
    int32_t s1Size, int32_t s2Size, int32_t batchSize)
{
    s2Size = this->Align(s2Size);
    int32_t computeSize = batchSize * s1Size * s2Size;
    Muls(dstTensor, dstTensor, static_cast<T>(this->tilingData->inputParams.scaleValue), computeSize);
    if (!this->hasAttenMask) {
        return;
    }
    if (isInputTAttenMask) {
        AttenMaskInputTCompute(dstTensor, computeSize);
    } else {
        AttenMaskBoolCompute(dstTensor, s1Size, s2Size, batchSize);
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock>::SoftmaxCopyOut(int64_t gmOffset,
    int64_t calculateSize)
{
    this->softmaxOutQueue.EnQue(this->softmaxMaxUb);
    this->softmaxOutQueue.EnQue(this->softmaxSumUb);
    LocalTensor<float> maxTensor = this->softmaxOutQueue.template DeQue<float>();
    LocalTensor<float> sumTensor = this->softmaxOutQueue.template DeQue<float>();
    DataCopy(softmaxMaxGm[gmOffset], maxTensor, calculateSize);
    DataCopy(softmaxSumGm[gmOffset], sumTensor, calculateSize);
    this->softmaxOutQueue.FreeTensor(this->softmaxMaxUb);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock>::Stage1Cast(LocalTensor<INPUT_T> &dstTensor,
    LocalTensor<T> &srcTensor)
{
    LocalTensor<INPUT_T> castTBuf = this->typeCastTBuf.template Get<INPUT_T>(srcTensor.GetSize());
    Cast(castTBuf, srcTensor, RoundMode::CAST_ROUND, srcTensor.GetSize());
    DataCopy(dstTensor, castTBuf, srcTensor.GetSize());
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock>::PseComputeWithBatch(
    LocalTensor<T> &dstTensor, int32_t batchSize, int32_t s1Size, int32_t s2Size)
{
    if (!this->hasPse) {
        return;
    }
    LocalTensor<INPUT_T> pseUb = this->optionalInputQueue.template DeQue<INPUT_T>();
    s2Size = this->Align(s2Size);
    int32_t computeSize =
        this->tilingData->inputParams.pseShapeType == this->pseS1S2 ? batchSize * s1Size * s2Size : batchSize * s2Size;
    if constexpr (!IsSameType<INPUT_T, T>::value) {
        LocalTensor<T> tmpPseCastTensor = this->typeCastTBuf.template Get<T>();
        Cast(tmpPseCastTensor, pseUb, RoundMode::CAST_NONE, computeSize);
        this->PseBroadcastAddWithBatch(batchSize, s1Size, s2Size, computeSize, tmpPseCastTensor, dstTensor);
    } else {
        this->PseBroadcastAddWithBatch(batchSize, s1Size, s2Size, computeSize, pseUb, dstTensor);
    }
    this->optionalInputQueue.FreeTensor(pseUb);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock>::PseBroadcastAddWithBatch(int32_t batchSize,
    int32_t s1Size, int32_t s2Size, int32_t computeSize, const LocalTensor<T> &pseUb, const LocalTensor<T> &dstTensor)
{
    if (this->tilingData->inputParams.pseShapeType == 0) {
        Add(dstTensor, dstTensor, pseUb, computeSize);
    } else {
        int32_t dstBatchOffset = 0;
        int32_t pseBatchOffset = 0;
        BinaryRepeatParams binaryRepeatParams;
        binaryRepeatParams.src0BlkStride = 1;
        binaryRepeatParams.src0RepStride = s2Size / blockSize;
        binaryRepeatParams.src1BlkStride = 1;
        binaryRepeatParams.src1RepStride = 0;
        binaryRepeatParams.dstRepStride = binaryRepeatParams.src0RepStride;
        binaryRepeatParams.blockNumber = binaryRepeatParams.src0RepStride;
        for (int32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
            // mask参数可能需要拆分循环 mask支持范围fp16:[1,128], fp32(1,64)
            Add(dstTensor[dstBatchOffset], dstTensor[dstBatchOffset], pseUb[pseBatchOffset], s2Size, s1Size,
                binaryRepeatParams);
            dstBatchOffset += s1Size * s2Size;
            pseBatchOffset += s2Size;
        }
    }
}

/*
 只适用于不切S1,S2, 切Batch场景
*/
template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock>::DataCopyInWithBatch(
    LocalTensor<INPUT_T> &dstTensor, GlobalTensor<INPUT_T> &srcTensor, int64_t offset, int32_t batchSize,
    int32_t s1Size, int32_t s2Size)
{
    int32_t dtypeSize = sizeof(INPUT_T);
    int32_t alignedS2Size = this->Align(s2Size);
    uint32_t shapeArray[] = {static_cast<uint32_t>(batchSize), static_cast<uint32_t>(s1Size),
                           static_cast<uint32_t>(alignedS2Size)};
    dstTensor.SetShapeInfo(ShapeInfo(3, shapeArray, DataFormat::ND));
    dstTensor.SetSize(batchSize * s1Size * alignedS2Size);
    if (alignedS2Size == s2Size) { // 输入是对齐的, 可连续搬入
        int32_t computeSize = batchSize * s1Size * s2Size;
        DataCopy(dstTensor, srcTensor[offset], computeSize);
    } else {
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = batchSize * s1Size;
        dataCopyParams.blockLen = s2Size * dtypeSize; // 单位Byte
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (alignedS2Size - s2Size) * dtypeSize;
        DataCopyPadParams dataCopyPadParams;
        dataCopyPadParams.rightPadding = (alignedS2Size - s2Size) * dtypeSize;
        DataCopyPad(dstTensor, srcTensor[offset], dataCopyParams, dataCopyPadParams);
    }
    optionalInputQueue.EnQue(dstTensor);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock>::BoolCopyIn(LocalTensor<uint8_t> &dstTensor,
    GlobalTensor<uint8_t> &srcTensor, int64_t offset, int32_t s1Size, int32_t s2Size, uint64_t totalS2Size, int32_t batchSize)
{
    int32_t alignedS2Size = this->CeilDiv(s2Size, blockBytes) * blockBytes;
    uint32_t shapeArray[] = {static_cast<uint32_t>(batchSize * s1Size), static_cast<uint32_t>(alignedS2Size)};
    dstTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    dstTensor.SetSize(batchSize * s1Size * alignedS2Size);
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = batchSize * s1Size;
    dataCopyParams.dstStride = 0;
    if (s2Size % blockBytes == 0) {
        dataCopyParams.blockLen = alignedS2Size / blockBytes;
        dataCopyParams.srcStride = (totalS2Size - alignedS2Size) / blockBytes;
        DataCopy(dstTensor, srcTensor[offset], dataCopyParams);
    } else {
        dataCopyParams.blockLen = s2Size;
        dataCopyParams.srcStride = totalS2Size - s2Size;
        DataCopyPadParams dataCopyPadParams;
        DataCopyPad(dstTensor, srcTensor[offset], dataCopyParams, dataCopyPadParams);
    }
    optionalInputQueue.EnQue(dstTensor);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock>::BoolCopyInBatchBroadcast(
    LocalTensor<uint8_t> &dstTensor, GlobalTensor<uint8_t> &srcTensor, int64_t offset, int32_t s1Size, int32_t s2Size,
    uint64_t totalS2Size, int32_t batchSize)
{
    int32_t alignedS2Size = this->CeilDiv(s2Size, blockBytes) * blockBytes;
    uint32_t shapeArray[] = {static_cast<uint32_t>(batchSize), static_cast<uint32_t>(s1Size),
                             static_cast<uint32_t>(alignedS2Size)};
    dstTensor.SetShapeInfo(ShapeInfo(3, shapeArray, DataFormat::ND));
    dstTensor.SetSize(batchSize * s1Size * alignedS2Size);
    int32_t computeSize = s1Size * alignedS2Size;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = s1Size;
    dataCopyParams.dstStride = 0;
    if (s2Size % blockBytes == 0) {
        int32_t dstOffset = 0;
        dataCopyParams.blockLen = alignedS2Size / blockBytes;
        dataCopyParams.srcStride = (totalS2Size - alignedS2Size) / blockBytes;
        for (int32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
            dstOffset = batchIdx * computeSize;
            DataCopy(dstTensor[dstOffset], srcTensor[offset], dataCopyParams);
        }
    } else {
        int32_t dstOffset = 0;
        dataCopyParams.blockLen = s2Size;
        dataCopyParams.srcStride = totalS2Size - s2Size;
        DataCopyPadParams dataCopyPadParams;
        for (int32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
            dstOffset = batchIdx * computeSize;
            DataCopyPad(dstTensor[dstOffset], srcTensor[offset], dataCopyParams, dataCopyPadParams);
        }
    }
    optionalInputQueue.EnQue(dstTensor);
}

/**
 * 用于attenMask bool类型数据搬入
 * 支持2种shape type： (1, 1, 1, S1, S2), (B, N2, G, S1, S2).
 * (B, 1, 1, S1, S2)在切B模板中，因无法区分batchSize中是否包含B，需模板在B_i的循环中调用
 * atten mask计算不支持设置stride, 直接采用物理broadcast方式
*/
template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock>::BoolCopyInWithBatch(
    LocalTensor<uint8_t> &dstTensor, GlobalTensor<uint8_t> &srcTensor, int64_t offset, int32_t s1Size, int32_t s2Size,
    int32_t batchSize)
{
    if (this->tilingData->inputParams.attenMaskShapeType == this->attenMaskBN2GS1S2) {
        // 直接搬入不做broadcast
        BoolCopyIn(dstTensor, srcTensor, offset, s1Size, s2Size, tilingData->inputParams.attenMaskS2Size, batchSize);
    } else {
        // 在batch轴做bradcast
        BoolCopyInBatchBroadcast(dstTensor, srcTensor, offset, s1Size, s2Size, tilingData->inputParams.attenMaskS2Size,
                                 batchSize);
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock>::BroadcastBatchCopyIn(
    LocalTensor<INPUT_T> &dstTensor, GlobalTensor<INPUT_T> &srcTensor, int64_t offset, int32_t batchSize,
    int32_t s1Size, int32_t s2Size)
{
    // 用于attenmask输入，不支持B,1,1,S1,S2输入，切B轴场景,需要B轴循环调用
    int32_t dtypeSize = sizeof(INPUT_T);
    int32_t alignedS2Size = this->Align(s2Size);
    uint32_t shapeArray[] = {static_cast<uint32_t>(batchSize), static_cast<uint32_t>(s1Size),
                           static_cast<uint32_t>(alignedS2Size)};
    dstTensor.SetShapeInfo(ShapeInfo(3, shapeArray, DataFormat::ND));
    dstTensor.SetSize(batchSize * s1Size * alignedS2Size);
    int32_t computeSize = s1Size * alignedS2Size;
    if (alignedS2Size == s2Size) {
        int32_t dstOffset = 0;
        for (int32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
            dstOffset = batchIdx * computeSize;
            DataCopy(dstTensor[dstOffset], srcTensor[offset], computeSize);
        }
    } else {
        int32_t dstOffset = 0;
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = s1Size;
        dataCopyParams.blockLen = s2Size * dtypeSize; // 单位Byte
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (alignedS2Size - s2Size) * dtypeSize;
        DataCopyPadParams dataCopyPadParams;
        dataCopyPadParams.rightPadding = (alignedS2Size - s2Size) * dtypeSize;
        for (int32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
            dstOffset = batchIdx * computeSize;
            DataCopyPad(dstTensor[dstOffset], srcTensor[offset], dataCopyParams, dataCopyPadParams);
        }
    }
    optionalInputQueue.EnQue(dstTensor);
}

#endif // FLASH_ATTENTION_SCORE_COMMON_H
