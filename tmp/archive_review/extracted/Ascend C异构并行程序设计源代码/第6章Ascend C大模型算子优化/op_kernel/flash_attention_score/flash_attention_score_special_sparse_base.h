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

#ifndef FLASH_ATTENTION_SCORE_SPECIAL_SPARSE_BASE_H
#define FLASH_ATTENTION_SCORE_SPECIAL_SPARSE_BASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_data_copy_transpose.h"

using namespace AscendC;
using namespace matmul;

// INPUT_T - means data type for input
// T       - means data type when calc
template <typename INPUT_T, typename T = INPUT_T, bool isBasicBlock = false> class FlashAttentionScoreSpecialSparse {
public:
    __aicore__ inline FlashAttentionScoreSpecialSparse() {};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
        __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax,
        __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
        const FlashAttentionScoreTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();
    // define matmul
    static constexpr MatmulConfig mm1Config{false, true, false, 64, 128, 128};
    static constexpr MatmulConfig mm2Config{false, true, false, 64, 128, 128};
    using a1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T>;
    using b1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, true>;
    using bias1Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c1Type = MatmulType<TPosition::VECCALC, CubeFormat::ND, T>;
    matmul::Matmul<a1Type, b1Type, c1Type, bias1Type, mm1Config> mm;
    // define batchmatmul
    using a2Type = MatmulType<TPosition::VECCALC, CubeFormat::ND, INPUT_T>;
    using b2Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T>;
    using bias2Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c2Type = MatmulType<TPosition::VECCALC, CubeFormat::ND, T>;
    matmul::Matmul<a2Type, b2Type, c2Type, bias2Type, mm2Config> bmm2;

protected:
    const FlashAttentionScoreTilingData *__restrict tilingData;
    TPipe *pipe;
    // define the que
    TQue<QuePosition::VECIN, 1> attenMaskQueue;
    TQue<QuePosition::VECOUT, 1> mmResQueue;
    TQue<QuePosition::VECOUT, 1> tmpMMResBmm2Queue;
    TBuf<> tmpSoftmaxUb;
    TBuf<> tmpBmm2Ub;
    TBuf<> tmpTypeCastUb;
    TBuf<> apiTmpBufferUb;
    TQue<QuePosition::VECOUT, 2> softmaxOutQueue;  // softmax2个输出去：sum和max

    GlobalTensor<INPUT_T> queryGm;
    GlobalTensor<INPUT_T> keyGm;
    GlobalTensor<INPUT_T> valueGm;
    GlobalTensor<uint8_t> dropMaskGm;
    GlobalTensor<INPUT_T> attenMaskGm;
    GlobalTensor<uint8_t> attenMaskGmInt;
    GlobalTensor<INPUT_T> pseGm;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    GlobalTensor<INPUT_T> attentionOutGm;
    GlobalTensor<T> workspaceGm;

    uint32_t tmp_block_idx;
    // offset
    uint64_t loopBNum;
    uint64_t loopNNum;
    uint64_t loopSNum;
    uint64_t batchBOffset;
    uint64_t batchBOffsetDataSize;
    uint64_t batchNOffsetDataSize;
    uint64_t sOuterOffsetDataSize;
    uint64_t sInnerOffsetDataSize;
    uint64_t sOuterOffset;
    uint64_t batchNOffset;
    uint64_t maskOffset;
    uint64_t maskCoreOffset;
    uint64_t attenMaskCoreOffset;
    uint64_t pseCoreOffset;
    uint64_t valueOffset;
    uint64_t valueCoreOffset;
    uint64_t softmaxOutCoreOffset;
    uint64_t softmaxOutOffset;
    uint64_t attenMaskOffset;
    uint64_t pseOffset;
    uint64_t tensorAOffset;
    uint64_t tensorBOffset;
    uint64_t tensorACoreOffset;
    uint64_t tensorBCoreOffset;

    // tilingdata
    uint32_t singleProcessSOuterSize;
    uint32_t singleProcessSInnerSize;
    uint32_t sInnerLoopTimes;
    uint32_t singleProcessSInnerSizeTail;
    uint32_t mmResUbSize;
    uint32_t attenMaskUbSize;
    uint32_t pseUbSize;
    uint32_t maskSize;
    uint32_t softmaxMaxSize;
    uint32_t softmaxSumSize;
    uint32_t softmaxExpSize;
    uint32_t bmm2ResUbSize;
    uint32_t tmpMMResBmm2PreUbSize;
    uint32_t tmpSoftmaxBmm2UbSize;
    uint32_t typeByteNum;
    uint32_t mmResUbSizeTail;
    uint32_t attenMaskUbSizeTail;
    uint32_t pseUbSizeTail;
    uint32_t maskSizeTail;
    SoftMaxTiling softmaxTilingData;
    SoftMaxTiling softmaxFlashTilingData;
    CopyTransposeTiling transposeTilingData;

    uint64_t sInnerLoopCnt = 0;
    int32_t reuseWorkspaceRatio;

    int sInnerLoopStartIndex;
    int sInnerLoopEndIndex;
    uint32_t maxInnerLoopTimes;

    uint32_t softmaxMaxTypeByteNum;
    uint32_t attenMaskTypeIdx; // 0:fp, 1:int8, 2:int1

    T infValue;

    __aicore__ inline void AttenMaskCopyIn(uint64_t offset, uint32_t sinnerSize);

    __aicore__ inline bool PseAndAttenMaskCopyIn(uint64_t offset, uint32_t sinnerSize);

    __aicore__ inline void ElewiseCompute(LocalTensor<T> &mmResUb, uint32_t sinnerSize, uint32_t computeSize);

    __aicore__ inline void ComputePseRepParams(uint32_t innerSize, BinaryRepeatParams &repParam);

    __aicore__ inline void InnerBroadcastAdd(uint32_t repeatTimes, uint32_t outerOffset, uint32_t innerLoop,
        uint32_t innerRemain, const LocalTensor<T> &pseUb, const LocalTensor<T> &mmResUb, uint32_t stepCalcNum,
        const BinaryRepeatParams &repParams);

    __aicore__ inline void BroadcastAdd(uint32_t innerSize, uint32_t computeSize, const LocalTensor<T> &pseUb,
        const LocalTensor<T> &mmResUb);

    __aicore__ inline void PseAddCompute(bool hasPse, LocalTensor<T> &mmResUb, uint32_t innerSize,
        uint32_t computeSize);

    __aicore__ inline void SoftmaxComputeFirstLoop(LocalTensor<T> &mmResUb, LocalTensor<float> &softmaxMaxUb,
        LocalTensor<float> &softmaxSumUb, LocalTensor<T> &softmaxExpUb);

    __aicore__ inline void SoftmaxCompute(LocalTensor<T> &mmResUb, LocalTensor<float> &softmaxMaxUb,
        LocalTensor<float> &softmaxSumUb, LocalTensor<T> &softmaxExpUb, const bool isLast);

    __aicore__ inline void DropoutCopyIn(uint64_t sInnerOffset, uint32_t sinnerSize);

    __aicore__ inline void DropoutCompute(LocalTensor<T> &mmResUb, uint32_t sInnerSize);

    __aicore__ inline void Bmm2Compute(uint64_t offset, LocalTensor<T> &bmm2ResL1);

    __aicore__ inline void UpdateVmul(LocalTensor<T> &softmaxExpUb, LocalTensor<T> &bmm2ResPreUb);

    __aicore__ inline void Bmm2UpdateAdd(LocalTensor<T> &bmm2ResUb, LocalTensor<T> &bmm2ResPreUb);

    __aicore__ inline void Bmm2UpdateDiv(LocalTensor<T> &bmm2ResPreUb, LocalTensor<float> &softmaxSumUb,
        LocalTensor<T> &softmaxExpUb);

    __aicore__ inline void ComputeCubeOffset(uint64_t sInnerLoopIdx);

    __aicore__ inline void ComputeVecOffset();

    __aicore__ inline void Bmm1ResDoVecBmm2ComputeUpdate(LocalTensor<T> &bmm2ResPreUb, LocalTensor<float> &softmaxMaxUb,
        LocalTensor<float> &softmaxSumUb, LocalTensor<T> &softmaxExpUb, bool &if_not_second);

    __aicore__ inline void SInnerLoopFunc(uint64_t sOuterLoopIdx);

    __aicore__ inline void LoopSOuterOffsetInit();

    __aicore__ inline void ComputeEachCore(uint32_t coreIdx);

    __aicore__ inline void initOffset();

    __aicore__ inline void initTilingData();
};

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSpecialSparse<INPUT_T, T, isBasicBlock>::Init(__gm__ uint8_t *query, __gm__ uint8_t *key,
    __gm__ uint8_t *value, __gm__ uint8_t *pse, __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, const FlashAttentionScoreTilingData *__restrict tiling,
    TPipe *tPipe)
{
    tmp_block_idx = GetBlockIdx();
    // init global buffer
    tilingData = tiling;
    queryGm.SetGlobalBuffer((__gm__ INPUT_T *)query);
    keyGm.SetGlobalBuffer((__gm__ INPUT_T *)key);
    valueGm.SetGlobalBuffer((__gm__ INPUT_T *)value);
    dropMaskGm.SetGlobalBuffer((__gm__ uint8_t *)dropMask);
    attenMaskGm.SetGlobalBuffer((__gm__ INPUT_T *)attenMask);
    attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
    pseGm.SetGlobalBuffer((__gm__ INPUT_T *)pse);
    softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
    softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);
    attentionOutGm.SetGlobalBuffer((__gm__ INPUT_T *)attentionOut);
    workspaceGm.SetGlobalBuffer((__gm__ T *)workspace);

    tilingData = tiling;
    pipe = tPipe;
    initOffset();
    initTilingData();
    if (attenMaskUbSize != 0) {
      pipe->InitBuffer(attenMaskQueue, 1, attenMaskUbSize * sizeof(INPUT_T));
    }
    pipe->InitBuffer(mmResQueue, 1, mmResUbSize * sizeof(T));
    pipe->InitBuffer(tmpMMResBmm2Queue, 1, bmm2ResUbSize * sizeof(T));
    pipe->InitBuffer(tmpSoftmaxUb, softmaxExpSize * sizeof(T));
    pipe->InitBuffer(tmpBmm2Ub, bmm2ResUbSize * sizeof(T));
    pipe->InitBuffer(softmaxOutQueue, 2, softmaxMaxSize * sizeof(float));
    pipe->InitBuffer(apiTmpBufferUb, tilingData->attentionScoreSingleCoreTensorSize.softmaxTmpBuffer);
    if constexpr (IsSameType<INPUT_T, bfloat16_t>::value ||
        (IsSameType<INPUT_T, half>::value && IsSameType<T, float>::value)) {
        this->tmpTypeCastUb = apiTmpBufferUb;
    }
    reuseWorkspaceRatio = sInnerLoopTimes;
    mm.SetWorkspace((__gm__ uint8_t *)workspaceGm[tmp_block_idx * mmResUbSize * reuseWorkspaceRatio].GetPhyAddr(),
                    mmResUbSize * reuseWorkspaceRatio);
    uint64_t buff_offset = GetBlockNum() * GetTaskRation() * mmResUbSize * reuseWorkspaceRatio;
    bmm2.SetWorkspace((__gm__ uint8_t *)workspaceGm[buff_offset + tmp_block_idx * bmm2ResUbSize].GetPhyAddr(),
                      bmm2ResUbSize);
    LocalTensor<T> apiTmpBuffer = apiTmpBufferUb.Get<T>();
    if (tilingData->baseParams.keepProb != 1) {
      DropOutBitModeInit(apiTmpBuffer);
    }
    if constexpr (IsSameType<T, float>::value) {
        infValue = -2e38;
    } else {
        infValue = -10000.0;
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSpecialSparse<INPUT_T, T, isBasicBlock>::initOffset()
{
    loopBNum = tilingData->attentionScoreCoreParams.loopBatchNum;
    loopNNum = tilingData->attentionScoreCoreParams.loopHeadNum;
    if (attenMaskGm.GetPhyAddr() == nullptr) {
        loopSNum = tilingData->attentionScoreSingleCoreParams.sOuterLoopTimes;
        if (tmp_block_idx % tilingData->attentionScoreCoreParams.s1 >= tilingData->attentionScoreCoreParams.formerNum) {
            loopSNum = tilingData->attentionScoreSingleCoreParams.tailCoreSOuterLoopTimes;
        }
    } else {
        loopSNum = tilingData->attentionScoreSingleCoreParams.sOuterLoopTimesList[tmp_block_idx];
    }
    if (tmp_block_idx / tilingData->attentionScoreCoreParams.s1 % tilingData->attentionScoreCoreParams.n1 >=
        tilingData->attentionScoreCoreParams.formerHeadNum &&
        tilingData->baseParams.preTockens >= tilingData->baseParams.seqSizeK &&
        tilingData->baseParams.nextTockens >= tilingData->baseParams.seqSizeK) {
        loopNNum = tilingData->attentionScoreCoreParams.loopHeadNumTail;
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSpecialSparse<INPUT_T, T, isBasicBlock>::initTilingData()
{
    singleProcessSOuterSize = tilingData->attentionScoreSingleCoreParams.singleProcessSOuterSize;
    singleProcessSInnerSize = tilingData->attentionScoreSingleCoreParams.singleProcessSInnerSize;
    sInnerLoopTimes = tilingData->attentionScoreSingleCoreParams.sInnerLoopTimes;
    singleProcessSInnerSizeTail = tilingData->attentionScoreSingleCoreParams.singleProcessSInnerSizeTail;
    typeByteNum = tilingData->attentionScoreOffestStrideParams.typeByteNum;

    mmResUbSize = tilingData->attentionScoreSingleCoreTensorSize.mmResUbSize;
    attenMaskUbSize = tilingData->attentionScoreSingleCoreTensorSize.attenMaskUbSize;
    pseUbSize = tilingData->attentionScoreSingleCoreTensorSize.pseUbSize;
    maskSize = tilingData->attentionScoreSingleCoreTensorSize.maskSize;
    softmaxMaxSize = tilingData->attentionScoreSingleCoreTensorSize.softmaxMaxSize;
    softmaxSumSize = tilingData->attentionScoreSingleCoreTensorSize.softmaxSumSize;
    softmaxExpSize = tilingData->attentionScoreSingleCoreTensorSize.softmaxExpSize;
    bmm2ResUbSize = tilingData->attentionScoreSingleCoreTensorSize.bmm2ResUbSize;
    tmpMMResBmm2PreUbSize = tilingData->attentionScoreSingleCoreTensorSize.tmpMMResBmm2PreUbSize;
    tmpSoftmaxBmm2UbSize = tilingData->attentionScoreSingleCoreTensorSize.tmpSoftmaxBmm2UbSize;
    mmResUbSizeTail = tilingData->attentionScoreSingleCoreTensorSize.mmResUbSizeTailLoop;
    attenMaskUbSizeTail = tilingData->attentionScoreSingleCoreTensorSize.attenMaskUbSizeTailLoop;
    maskSizeTail = tilingData->attentionScoreSingleCoreTensorSize.maskSizeTailLoop;

    transposeTilingData = tilingData->transposeTilingData;
    softmaxTilingData = tilingData->softmaxTilingData;
    softmaxFlashTilingData = tilingData->softmaxFlashTilingData;

    maxInnerLoopTimes = tilingData->attentionScoreSingleCoreParams.sInnerLoopTimes;
    sInnerLoopStartIndex = 0;
    sInnerLoopEndIndex = maxInnerLoopTimes;

    softmaxMaxTypeByteNum = (sizeof(T) == sizeof(half)) ? (typeByteNum / 2) : typeByteNum;
    attenMaskTypeIdx = tilingData->baseParams.attenMaskTypeIdx;

    if (tmp_block_idx % tilingData->attentionScoreCoreParams.s1 >= tilingData->attentionScoreCoreParams.formerNum) {
        singleProcessSOuterSize = tilingData->attentionScoreSingleCoreParams.tailCoreSingleProcessSOuterSize;
        singleProcessSInnerSize = tilingData->attentionScoreSingleCoreParams.tailCoreSingleProcessSInnerSize;
        sInnerLoopTimes = tilingData->attentionScoreSingleCoreParams.tailCoreSInnerLoopTimes;
        singleProcessSInnerSizeTail = tilingData->attentionScoreSingleCoreParams.tailCoreSingleProcessSInnerSizeTail;
        mmResUbSize = tilingData->attentionScoreSingleCoreTensorSize.tailCoreMMResUbSize;
        attenMaskUbSize = tilingData->attentionScoreSingleCoreTensorSize.tailCoreAttenMaskUbSize;
        maskSize = tilingData->attentionScoreSingleCoreTensorSize.tailCoreMaskSize;
        softmaxMaxSize = tilingData->attentionScoreSingleCoreTensorSize.tailCoreSoftmaxMaxSize;
        softmaxSumSize = tilingData->attentionScoreSingleCoreTensorSize.tailCoreSoftmaxSumSize;
        softmaxExpSize = tilingData->attentionScoreSingleCoreTensorSize.tailCoreSoftmaxExpSize;
        bmm2ResUbSize = tilingData->attentionScoreSingleCoreTensorSize.tailCoreBmm2ResUbSize;
        tmpMMResBmm2PreUbSize = tilingData->attentionScoreSingleCoreTensorSize.tailCoreTmpMMResBmm2PreUbSize;
        tmpSoftmaxBmm2UbSize = tilingData->attentionScoreSingleCoreTensorSize.tailCoreTmpSoftmaxBmm2UbSize;
        mmResUbSizeTail = tilingData->attentionScoreSingleCoreTensorSize.tailCoreMMResUbSizeTailLoop;
        attenMaskUbSizeTail = tilingData->attentionScoreSingleCoreTensorSize.tailCoreAttenMaskUbSizeTailLoop;
        maskSizeTail = tilingData->attentionScoreSingleCoreTensorSize.tailCoreMaskSizeTailLoop;
        transposeTilingData = tilingData->transposeTilingDataTailCore;
        softmaxTilingData = tilingData->softmaxTilingDataTailCore;
        softmaxFlashTilingData = tilingData->softmaxFlashTilingDataTailCore;
        maxInnerLoopTimes = tilingData->attentionScoreSingleCoreParams.tailCoreSInnerLoopTimes;
        sInnerLoopEndIndex = maxInnerLoopTimes;
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSpecialSparse<INPUT_T, T, isBasicBlock>::Process()
{
    ComputeEachCore(tmp_block_idx);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSpecialSparse<INPUT_T, T, isBasicBlock>::AttenMaskCopyIn(uint64_t offset,
    uint32_t sinnerSize)
{
    if (attenMaskGm.GetPhyAddr() == nullptr) {
        return;
    }

    if (attenMaskTypeIdx == 1) {
        // int8
        LocalTensor<uint8_t> attenMaskUb = attenMaskQueue.AllocTensor<uint8_t>();
        DataCopyParams intriParams;
        intriParams.blockCount = singleProcessSOuterSize;
        intriParams.blockLen = (sinnerSize * sizeof(uint8_t) + 31) / 32;
        intriParams.dstStride = 0;
        if ((tilingData->baseParams.seqSizeK - intriParams.blockLen * 32) % 32 == 0) {
            intriParams.srcStride = (tilingData->baseParams.seqSizeK - sinnerSize) / (32 / sizeof(uint8_t));
            DataCopy(attenMaskUb, attenMaskGmInt[offset], intriParams);
        } else {
            intriParams.blockLen = sinnerSize * sizeof(uint8_t);
            intriParams.srcStride = (tilingData->baseParams.seqSizeK * sizeof(uint8_t) - intriParams.blockLen);
            DataCopyPadParams intriParams1;
            DataCopyPad(attenMaskUb, attenMaskGmInt[offset], intriParams, intriParams1);
        }
        attenMaskQueue.EnQue(attenMaskUb);
    } else {
        // INPUT_T
        LocalTensor<INPUT_T> attenMaskUb = attenMaskQueue.AllocTensor<INPUT_T>();
        attenMaskUb.SetSize(singleProcessSOuterSize * sinnerSize);
        uint32_t shapeArray[] = {singleProcessSOuterSize, sinnerSize};
        attenMaskUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        DataCopyParams intriParams;
        intriParams.blockCount = singleProcessSOuterSize;
        intriParams.blockLen = sinnerSize / (32 / sizeof(INPUT_T));
        intriParams.srcStride = (tilingData->baseParams.seqSizeK - sinnerSize) / (32 / sizeof(INPUT_T));
        intriParams.dstStride = 0;
        DataCopy(attenMaskUb, attenMaskGm[offset], intriParams);
        attenMaskQueue.EnQue(attenMaskUb);
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline bool FlashAttentionScoreSpecialSparse<INPUT_T, T, isBasicBlock>::PseAndAttenMaskCopyIn(uint64_t offset,
    uint32_t sinnerSize)
{
    if (pseGm.GetPhyAddr() == nullptr) {
        AttenMaskCopyIn(attenMaskOffset, sinnerSize);
        return false;
    }
    LocalTensor<INPUT_T> pseUb = attenMaskQueue.AllocTensor<INPUT_T>();
    DataCopyParams intriParams;
    intriParams.blockCount = singleProcessSOuterSize;
    if (tilingData->baseParams.pseShapeType == 1) {
        intriParams.blockCount = 1;
    }
    pseUb.SetSize(intriParams.blockCount * sinnerSize);
    uint32_t shapeArray[] = {intriParams.blockCount, sinnerSize};
    pseUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    intriParams.blockLen = sinnerSize / (32 / sizeof(INPUT_T));
    intriParams.srcStride = (tilingData->baseParams.seqSizeK - sinnerSize) / (32 / sizeof(INPUT_T));
    intriParams.dstStride = 0;
    DataCopy(pseUb, pseGm[offset], intriParams);
    attenMaskQueue.EnQue(pseUb);
    return true;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSpecialSparse<INPUT_T, T, isBasicBlock>::ElewiseCompute(LocalTensor<T> &mmResUb,
    uint32_t sinnerSize, uint32_t computeSize)
{
    Muls(mmResUb, mmResUb, static_cast<T>(tilingData->baseParams.scaleValue), computeSize);

    if (attenMaskGm.GetPhyAddr() == nullptr) {
        return;
    }

    if (attenMaskTypeIdx == 1) {
        // int8
        LocalTensor<uint8_t> attenMaskUb = attenMaskQueue.DeQue<uint8_t>();
        LocalTensor<uint8_t> ubWorkspace = apiTmpBufferUb.Get<uint8_t>();
        T scalar;
        if constexpr (IsSameType<T, float>::value) {
            uint32_t tmp = 0xFF800000;
            scalar = *((float *)&tmp);
        } else {
            uint16_t tmp = 0xFC00;
            scalar = *((half *)&tmp);
        }
        SelectWithBytesMaskShapeInfo s;
        s.firstAxis = singleProcessSOuterSize;
        s.srcLastAxis = sinnerSize;
        s.maskLastAxis = (sinnerSize * sizeof(uint8_t) + 31) / 32 * 32 / sizeof(uint8_t);
        SelectWithBytesMask(mmResUb, mmResUb, scalar, attenMaskUb, ubWorkspace, s);
        attenMaskQueue.FreeTensor(attenMaskUb);
    } else {
        // INPUT_T
        LocalTensor<INPUT_T> attenMaskUb = attenMaskQueue.DeQue<INPUT_T>();
        if constexpr (IsSameType<INPUT_T, bfloat16_t>::value ||
                      (IsSameType<INPUT_T, half>::value && IsSameType<T, float>::value)) {
            LocalTensor<T> tmpAttenCastTensor = tmpTypeCastUb.Get<T>();
            Cast(tmpAttenCastTensor, attenMaskUb, RoundMode::CAST_NONE, computeSize);
            Muls(tmpAttenCastTensor, tmpAttenCastTensor, static_cast<T>(infValue), computeSize);
            pipe_barrier(PIPE_V);
            Add(mmResUb, mmResUb, tmpAttenCastTensor, computeSize);
        } else {
            Muls(attenMaskUb, attenMaskUb, static_cast<T>(infValue), computeSize);
            pipe_barrier(PIPE_V);
            Add(mmResUb, mmResUb, attenMaskUb, computeSize);
        }
        attenMaskQueue.FreeTensor(attenMaskUb);
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSpecialSparse<INPUT_T, T, isBasicBlock>::ComputePseRepParams(uint32_t innerSize,
    BinaryRepeatParams &repParam)
{
    repParam.src0BlkStride = 1;
    repParam.src0RepStride = innerSize / typeByteNum;
    repParam.src1BlkStride = 1;
    repParam.src1RepStride = 0;
    repParam.dstRepStride = repParam.src0RepStride;
    repParam.blockNumber = repParam.src0RepStride;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSpecialSparse<INPUT_T, T, isBasicBlock>::InnerBroadcastAdd(uint32_t repeatTimes,
    uint32_t outerOffset, uint32_t innerLoop, uint32_t innerRemain, const LocalTensor<T> &pseUb,
    const LocalTensor<T> &mmResUb, uint32_t stepCalcNum, const BinaryRepeatParams &repParams)
{
    for (uint32_t j = 0; j < innerLoop; j++) {
        auto innerOffset = j * stepCalcNum;
        auto ubOffset = outerOffset + innerOffset;
        Add(mmResUb[ubOffset], mmResUb[ubOffset], pseUb[innerOffset], stepCalcNum, repeatTimes, repParams);
    }
    if (innerRemain > 0) {
        auto innerOffset = innerLoop * stepCalcNum;
        auto ubOffset = outerOffset + innerOffset;
        Add(mmResUb[ubOffset], mmResUb[ubOffset], pseUb[innerOffset], innerRemain, repeatTimes, repParams);
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSpecialSparse<INPUT_T, T, isBasicBlock>::BroadcastAdd(uint32_t innerSize,
    uint32_t computeSize, const LocalTensor<T> &pseUb, const LocalTensor<T> &mmResUb)
{
    if (tilingData->baseParams.pseShapeType == 0) {
        Add(mmResUb, mmResUb, pseUb, computeSize);
    } else {
        uint32_t stepCalcNum = 256 / sizeof(T);
        /* Total data number of single step should be smaller than 256bytes.
         * If larger, we need to do add multiple times. */
        stepCalcNum = innerSize > stepCalcNum ? stepCalcNum : innerSize;
        uint32_t innerLoop = innerSize / stepCalcNum;
        uint32_t innerRemain = innerSize % stepCalcNum;
        BinaryRepeatParams repParams;
        ComputePseRepParams(innerSize, repParams);
        /* Total repeated times should be <= 255. If larger,
         * we need to do multiple inner loops. */
        uint32_t outerLoop = singleProcessSOuterSize / 255;
        uint32_t outerRemain = singleProcessSOuterSize % 255;
        for (uint32_t i = 0; i < outerLoop; i++) {
            uint32_t outerOffset = (i >> 8) * innerSize;
            InnerBroadcastAdd(255, outerOffset, innerLoop, innerRemain, pseUb, mmResUb, stepCalcNum, repParams);
        }
        if (outerRemain > 0) {
            uint32_t outerOffset = (outerLoop >> 8) * innerSize;
            InnerBroadcastAdd(outerRemain, outerOffset, innerLoop, innerRemain, pseUb, mmResUb, stepCalcNum, repParams);
        }
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSpecialSparse<INPUT_T, T, isBasicBlock>::PseAddCompute(bool hasPse,
    LocalTensor<T> &mmResUb, uint32_t innerSize, uint32_t computeSize)
{
    if (hasPse) {
        LocalTensor<INPUT_T> pseUb = attenMaskQueue.DeQue<INPUT_T>();
        if constexpr (IsSameType<INPUT_T, bfloat16_t>::value ||
            (IsSameType<INPUT_T, half>::value && IsSameType<T, float>::value)) {
            LocalTensor<T> tmpPseCastTensor = tmpTypeCastUb.Get<T>();
            Cast(tmpPseCastTensor, pseUb, RoundMode::CAST_NONE, computeSize);
            BroadcastAdd(innerSize, computeSize, tmpPseCastTensor, mmResUb);
        } else {
            BroadcastAdd(innerSize, computeSize, pseUb, mmResUb);
        }
        attenMaskQueue.FreeTensor(pseUb);
        AttenMaskCopyIn(attenMaskOffset, innerSize);
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSpecialSparse<INPUT_T, T, isBasicBlock>::SoftmaxComputeFirstLoop(LocalTensor<T> &mmResUb,
    LocalTensor<float> &softmaxMaxUb, LocalTensor<float> &softmaxSumUb, LocalTensor<T> &softmaxExpUb)
{
    LocalTensor<uint8_t> apiTmpBuffer = apiTmpBufferUb.Get<uint8_t>();
    if (sInnerLoopTimes == 1) {
        SoftMax<T, true, isBasicBlock>(mmResUb, softmaxSumUb, softmaxMaxUb, mmResUb, apiTmpBuffer, softmaxTilingData);
        softmaxOutQueue.EnQue(softmaxMaxUb);
        softmaxOutQueue.EnQue(softmaxSumUb);
        softmaxMaxUb = softmaxOutQueue.DeQue<float>();
        softmaxSumUb = softmaxOutQueue.DeQue<float>();
        DataCopy(softmaxMaxGm[softmaxOutOffset], softmaxMaxUb, softmaxMaxSize);
        DataCopy(softmaxSumGm[softmaxOutOffset], softmaxSumUb, softmaxSumSize);
        softmaxOutQueue.FreeTensor(softmaxMaxUb);
        softmaxOutQueue.FreeTensor(softmaxSumUb);
    } else {
        SoftmaxFlashV2<T, false, true, isBasicBlock>(mmResUb, softmaxSumUb, softmaxMaxUb, mmResUb, softmaxExpUb,
            softmaxSumUb, softmaxMaxUb, apiTmpBuffer, softmaxFlashTilingData);
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSpecialSparse<INPUT_T, T, isBasicBlock>::SoftmaxCompute(LocalTensor<T> &mmResUb,
    LocalTensor<float> &softmaxMaxUb, LocalTensor<float> &softmaxSumUb, LocalTensor<T> &softmaxExpUb, const bool isLast)
{
    LocalTensor<uint8_t> apiTmpBuffer = apiTmpBufferUb.Get<uint8_t>();
    SoftmaxFlashV2<T, true, true, isBasicBlock>(mmResUb, softmaxSumUb, softmaxMaxUb, mmResUb, softmaxExpUb,
        softmaxSumUb, softmaxMaxUb, apiTmpBuffer, softmaxFlashTilingData);
    if (isLast) {
        // copy out
        softmaxOutQueue.EnQue(softmaxMaxUb);
        softmaxOutQueue.EnQue(softmaxSumUb);
        softmaxMaxUb = softmaxOutQueue.DeQue<float>();
        softmaxSumUb = softmaxOutQueue.DeQue<float>();
        DataCopy(softmaxMaxGm[softmaxOutOffset], softmaxMaxUb, softmaxMaxSize);
        DataCopy(softmaxSumGm[softmaxOutOffset], softmaxSumUb, softmaxSumSize);
        softmaxOutQueue.FreeTensor(softmaxMaxUb);
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSpecialSparse<INPUT_T, T, isBasicBlock>::DropoutCopyIn(uint64_t offset,
    uint32_t sinnerSize)
{
    LocalTensor<uint8_t> dropMaskUb = attenMaskQueue.AllocTensor<uint8_t>();
    uint32_t shapeArray[] = {singleProcessSOuterSize, sinnerSize};
    dropMaskUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    dropMaskUb.SetSize(singleProcessSOuterSize * sinnerSize);
    DataCopyParams intriParams;
    intriParams.blockCount = singleProcessSOuterSize;
    intriParams.blockLen = (sinnerSize / 8 + 31) / 32;
    intriParams.dstStride = 0;
    if ((tilingData->baseParams.seqSizeK / 8 - intriParams.blockLen * 32) % 32 == 0) {
        intriParams.srcStride = (tilingData->baseParams.seqSizeK / 8 - intriParams.blockLen * 32) / 32;
        DataCopy(dropMaskUb, dropMaskGm[offset / 8], intriParams);
    } else {
        intriParams.blockLen = sinnerSize / 8;
        intriParams.srcStride = (tilingData->baseParams.seqSizeK / 8 - intriParams.blockLen);
        DataCopyPadParams intriParams1;
        DataCopyPad(dropMaskUb, dropMaskGm[offset / 8], intriParams, intriParams1);
    }
    attenMaskQueue.EnQue(dropMaskUb);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSpecialSparse<INPUT_T, T, isBasicBlock>::DropoutCompute(LocalTensor<T> &mmResUb,
    uint32_t sInnerSize)
{
    LocalTensor<uint8_t> dropMaskUb = attenMaskQueue.DeQue<uint8_t>();
    dropMaskUb.SetSize(singleProcessSOuterSize * sInnerSize);
    DropOutShapeInfo dropOutShapeInfo;
    dropOutShapeInfo.firstAxis = singleProcessSOuterSize;
    dropOutShapeInfo.srcLastAxis = sInnerSize;
    dropOutShapeInfo.maskLastAxis = (sInnerSize / 8 + 31) / 32 * 32;
    LocalTensor<uint8_t> apiTmpBuffer = apiTmpBufferUb.Get<uint8_t>();
    if (attenMaskTypeIdx == 1) {
        // int8 SelectWithBytesMask has used cmp, DropOut can not use <T, true>
        DropOut(mmResUb, mmResUb, dropMaskUb, apiTmpBuffer, tilingData->baseParams.keepProb, dropOutShapeInfo);
    } else {
        DropOut<T, true>(mmResUb, mmResUb, dropMaskUb, apiTmpBuffer, tilingData->baseParams.keepProb, dropOutShapeInfo);
    }
    attenMaskQueue.FreeTensor(dropMaskUb);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSpecialSparse<INPUT_T, T, isBasicBlock>::Bmm2Compute(uint64_t offset,
    LocalTensor<T> &bmm2ResL1)
{
    if constexpr (IsSameType<INPUT_T, bfloat16_t>::value ||
        (IsSameType<INPUT_T, half>::value && IsSameType<T, float>::value)) {
        LocalTensor<INPUT_T> tmpBmm2ResCastTensor = tmpTypeCastUb.Get<INPUT_T>(bmm2ResL1.GetSize());
        Cast(tmpBmm2ResCastTensor, bmm2ResL1, RoundMode::CAST_ROUND, bmm2ResL1.GetSize());
        bmm2.SetTensorA(tmpBmm2ResCastTensor);
    } else {
        bmm2.SetTensorA(bmm2ResL1);
    }
    bmm2.SetTensorB(valueGm[offset]);
    bmm2.SetTail(-1, -1, singleProcessSInnerSize);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSpecialSparse<INPUT_T, T, isBasicBlock>::UpdateVmul(LocalTensor<T> &softmaxExpUb,
    LocalTensor<T> &bmm2ResPreUb)
{
    BinaryRepeatParams repeatParams;
    repeatParams.src0BlkStride = 0;
    repeatParams.src0RepStride = 1;
    repeatParams.src1RepStride = tilingData->baseParams.headSize / typeByteNum;
    repeatParams.dstRepStride = tilingData->baseParams.headSize / typeByteNum;

    int32_t loop = tilingData->baseParams.headSize / (256 / sizeof(T));
    int32_t remain = tilingData->baseParams.headSize % (256 / sizeof(T));
    for (int i = 0; i < loop; i++) {
        Mul(bmm2ResPreUb[i * 256 / sizeof(T)], softmaxExpUb, bmm2ResPreUb[i * 256 / sizeof(T)], 256 / sizeof(T),
            singleProcessSOuterSize, repeatParams);
    }
    if (remain) {
        Mul(bmm2ResPreUb[loop * 256 / sizeof(T)], softmaxExpUb, bmm2ResPreUb[loop * 256 / sizeof(T)], remain,
            singleProcessSOuterSize, repeatParams);
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSpecialSparse<INPUT_T, T, isBasicBlock>::Bmm2UpdateAdd(LocalTensor<T> &bmm2ResUb,
    LocalTensor<T> &bmm2ResPreUb)
{
    Add(bmm2ResPreUb, bmm2ResUb, bmm2ResPreUb, bmm2ResUbSize);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSpecialSparse<INPUT_T, T, isBasicBlock>::Bmm2UpdateDiv(LocalTensor<T> &bmm2ResPreUb,
    LocalTensor<float> &softmaxSumUb, LocalTensor<T> &softmaxExpUb)
{
    BinaryRepeatParams repeatParams;
    repeatParams.src0BlkStride = 1;
    repeatParams.src0RepStride = tilingData->baseParams.headSize / typeByteNum;
    repeatParams.src1BlkStride = 0;
    repeatParams.src1RepStride = 1;
    repeatParams.dstRepStride = tilingData->baseParams.headSize / typeByteNum;
    int32_t loop = tilingData->baseParams.headSize / (256 / sizeof(T));
    int32_t remain = tilingData->baseParams.headSize % (256 / sizeof(T));

    if constexpr (IsSameType<INPUT_T, half>::value && !IsSameType<T, float>::value) {
        int32_t calcSize = softmaxSumUb.GetSize();
        LocalTensor<float> tmpBuffer;
        if (attenMaskUbSize == 0) {
          tmpBuffer = tmpTypeCastUb.Get<float>();
        } else {
          tmpBuffer = attenMaskQueue.AllocTensor<float>();
        }
        Copy(tmpBuffer, softmaxSumUb, 64, calcSize / 64, { 2, 1, 16, 8 });
        Copy(tmpBuffer[8], softmaxSumUb, 64, calcSize / 64, { 2, 1, 16, 8 });
        pipe_barrier(PIPE_V);
        Cast(softmaxExpUb, tmpBuffer, RoundMode::CAST_ROUND, calcSize * 2);
        pipe_barrier(PIPE_V);
        if (attenMaskUbSize != 0) {
          attenMaskQueue.FreeTensor(tmpBuffer);
        }
        for (int i = 0; i < loop; i++) {
            Div(bmm2ResPreUb[i * 256 / sizeof(T)], bmm2ResPreUb[i * 256 / sizeof(T)], softmaxExpUb, 256 / sizeof(T),
                singleProcessSOuterSize, repeatParams);
        }
        if (remain) {
            Div(bmm2ResPreUb[loop * 256 / sizeof(T)], bmm2ResPreUb[loop * 256 / sizeof(T)], softmaxExpUb, remain,
                singleProcessSOuterSize, repeatParams);
        }
    } else {
        for (int i = 0; i < loop; i++) {
            Div(bmm2ResPreUb[i * 256 / sizeof(T)], bmm2ResPreUb[i * 256 / sizeof(T)], softmaxSumUb, 256 / sizeof(T),
                singleProcessSOuterSize, repeatParams);
        }
        if (remain) {
            Div(bmm2ResPreUb[loop * 256 / sizeof(T)], bmm2ResPreUb[loop * 256 / sizeof(T)], softmaxSumUb, remain,
                singleProcessSOuterSize, repeatParams);
        }
    }
    softmaxOutQueue.FreeTensor(softmaxSumUb);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSpecialSparse<INPUT_T, T, isBasicBlock>::ComputeVecOffset()
{
    sInnerOffsetDataSize = sInnerLoopCnt * singleProcessSInnerSize;
    attenMaskOffset = attenMaskCoreOffset + sInnerOffsetDataSize;
    pseOffset = pseCoreOffset + sInnerOffsetDataSize;
    maskOffset = maskCoreOffset + sInnerOffsetDataSize;
    valueOffset = valueCoreOffset + sInnerOffsetDataSize * tilingData->attentionScoreOffestStrideParams.matmulHead;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSpecialSparse<INPUT_T, T, isBasicBlock>::ComputeCubeOffset(uint64_t sInnerLoopIdx)
{
    sInnerOffsetDataSize = sInnerLoopIdx * singleProcessSInnerSize * reuseWorkspaceRatio;
    tensorAOffset = tensorACoreOffset;
    tensorBOffset = tensorBCoreOffset + sInnerOffsetDataSize * tilingData->attentionScoreOffestStrideParams.matmulHead;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSpecialSparse<INPUT_T, T, isBasicBlock>::Bmm1ResDoVecBmm2ComputeUpdate(
    LocalTensor<T> &bmm2ResPreUb, LocalTensor<float> &softmaxMaxUb, LocalTensor<float> &softmaxSumUb,
    LocalTensor<T> &softmaxExpUb, bool &if_not_second)
{
    bool isLast = (sInnerLoopCnt == (sInnerLoopEndIndex - 1));
    bool isFirst = (sInnerLoopCnt == sInnerLoopStartIndex);
    uint32_t innerSize = isLast ? singleProcessSInnerSizeTail : singleProcessSInnerSize;
    LocalTensor<T> mmResUb = mmResQueue.AllocTensor<T>();
    uint32_t shapeArray2[] = {singleProcessSOuterSize, innerSize};
    mmResUb.SetShapeInfo(ShapeInfo(2, shapeArray2, DataFormat::ND));
    // compute offset
    ComputeVecOffset();
    bool hasPse = PseAndAttenMaskCopyIn(pseOffset, innerSize);
    // start compute
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    set_flag(PIPE_MTE3, PIPE_MTE2, eventID);
    wait_flag(PIPE_MTE3, PIPE_MTE2, eventID);

    mm.template GetTensorC<false>(mmResUb, false, true);
    uint64_t computeSize = innerSize * singleProcessSOuterSize;
    PseAddCompute(hasPse, mmResUb, innerSize, computeSize);
    // vector compute
    ElewiseCompute(mmResUb, innerSize, computeSize);
    if (tilingData->baseParams.keepProb != 1) {
        DropoutCopyIn(maskOffset, innerSize);
    }
    if (unlikely(isFirst)) {
        // softmax
        SoftmaxComputeFirstLoop(mmResUb, softmaxMaxUb, softmaxSumUb, softmaxExpUb);
    } else {
        SoftmaxCompute(mmResUb, softmaxMaxUb, softmaxSumUb, softmaxExpUb, isLast);
    }
    if (tilingData->baseParams.keepProb != 1) {
        DropoutCompute(mmResUb, innerSize);
    }
    LocalTensor<T> bmm2ResUb;
    if (!isFirst) {
        if (if_not_second) {
            bmm2ResUb = tmpBmm2Ub.Get<T>(bmm2ResUbSize);
            bmm2.template GetTensorC<false>(bmm2ResUb, false, true);
            bmm2.End();
        } else {
            bmm2.template GetTensorC<false>(bmm2ResPreUb, false, true);
            bmm2.End();
        }
    }

    //  bmm2+confs_transopose
    if (unlikely(isLast)) {
        mmResUb.SetSize(singleProcessSOuterSize * singleProcessSInnerSizeTail);
    }
    Bmm2Compute(valueOffset, mmResUb);
    bmm2.template Iterate<false>();
    mmResQueue.FreeTensor(mmResUb);

    if (if_not_second) {
        Bmm2UpdateAdd(bmm2ResUb, bmm2ResPreUb);
    }
    if (unlikely(isLast)) {
        if (sInnerLoopTimes != 1) {
            UpdateVmul(softmaxExpUb, bmm2ResPreUb);
            LocalTensor<T> bmm2ResUb = tmpBmm2Ub.Get<T>(bmm2ResUbSize);
            bmm2.template GetTensorC<false>(bmm2ResUb, false, true);
            bmm2.End();
            Bmm2UpdateAdd(bmm2ResUb, bmm2ResPreUb);
            pipe_barrier(PIPE_V);
            Bmm2UpdateDiv(bmm2ResPreUb, softmaxSumUb, softmaxExpUb);
        } else {
            bmm2.template GetTensorC<false>(bmm2ResPreUb, false, true);
            bmm2.End();
        }
        TransposeParams transposeParams;
        transposeParams.bIndex = batchBOffset;
        transposeParams.nIndex = batchNOffset;
        transposeParams.sIndex = sOuterOffset;
        transposeParams.hNIndex = 0;
        if constexpr (IsSameType<INPUT_T, bfloat16_t>::value ||
            (IsSameType<INPUT_T, half>::value && IsSameType<T, float>::value)) {
            LocalTensor<INPUT_T> tmpAttenCastTensor = tmpTypeCastUb.Get<INPUT_T>();
            // fp32->bf16
            int32_t calcSize = bmm2ResPreUb.GetSize();
            Cast(tmpAttenCastTensor, bmm2ResPreUb, RoundMode::CAST_ROUND, calcSize);
            pipe_barrier(PIPE_V);
            tmpMMResBmm2Queue.FreeTensor(bmm2ResPreUb);
            LocalTensor<INPUT_T> bmm2ResUbCast = tmpMMResBmm2Queue.AllocTensor<INPUT_T>();
            DataCopy(bmm2ResUbCast, tmpAttenCastTensor, calcSize);
            tmpMMResBmm2Queue.EnQue(bmm2ResUbCast);
            bmm2ResUbCast = tmpMMResBmm2Queue.DeQue<INPUT_T>();
            if (tilingData->baseParams.transType == 0) {
                DataCopyTranspose<INPUT_T>(attentionOutGm, bmm2ResUbCast, CopyTransposeType::TRANSPOSE_ND_UB_GM,
                    transposeParams, transposeTilingData);
            } else if (tilingData->baseParams.transType == 1) {
                DataCopyTranspose<INPUT_T>(attentionOutGm, bmm2ResUbCast, CopyTransposeType::TRANSPOSE_ND_UB_GM_SBH,
                    transposeParams, transposeTilingData);
            }
            tmpMMResBmm2Queue.FreeTensor(bmm2ResUbCast);
        } else {
            tmpMMResBmm2Queue.EnQue(bmm2ResPreUb);
            bmm2ResPreUb = tmpMMResBmm2Queue.DeQue<T>();
            if (tilingData->baseParams.transType == 0) {
                DataCopyTranspose<T>(attentionOutGm, bmm2ResPreUb, CopyTransposeType::TRANSPOSE_ND_UB_GM,
                    transposeParams, transposeTilingData);
            } else if (tilingData->baseParams.transType == 1) {
                DataCopyTranspose<T>(attentionOutGm, bmm2ResPreUb, CopyTransposeType::TRANSPOSE_ND_UB_GM_SBH,
                    transposeParams, transposeTilingData);
            }
            tmpMMResBmm2Queue.FreeTensor(bmm2ResPreUb);
        }
    } else {
        if (!isFirst) {
            UpdateVmul(softmaxExpUb, bmm2ResPreUb);
        }
    }
    sInnerLoopCnt++;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSpecialSparse<INPUT_T, T, isBasicBlock>::SInnerLoopFunc(uint64_t sOuterLoopIdx)
{
    bool isAdd = false;

    uint32_t shapeMaxArray[] = {singleProcessSOuterSize, 8};
    uint32_t shapeSumArray[] = {singleProcessSOuterSize, 8};
    uint32_t shapeArray[] = {singleProcessSOuterSize, typeByteNum};
    LocalTensor<float> softmaxMaxUb = softmaxOutQueue.AllocTensor<float>();
    LocalTensor<float> softmaxSumUb = softmaxOutQueue.AllocTensor<float>();
    LocalTensor<T> softmaxExpUb = tmpSoftmaxUb.Get<T>(softmaxExpSize);

    softmaxMaxUb.SetShapeInfo(ShapeInfo(2, shapeMaxArray, DataFormat::ND));
    softmaxSumUb.SetShapeInfo(ShapeInfo(2, shapeSumArray, DataFormat::ND));
    softmaxExpUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));

    sInnerLoopCnt = sInnerLoopStartIndex;
    uint32_t mmRatio = tilingData->baseParams.seqSizeK / tilingData->attentionScoreSingleCoreParams.singleProcessSInnerSize;
    if (mmRatio > 32 && tilingData->baseParams.seqSizeK == 9216 && tilingData->baseParams.seqSizeQ == 9216 &&
        tilingData->baseParams.batchSize == 24 && tilingData->baseParams.headNumSize == 5) {
        mmRatio = 8;
    }
    // matmul basic block loop times
    uint32_t mm1LoopEndIdx = (sInnerLoopTimes + mmRatio - 1) / mmRatio;
    // tail loop matmul cache number
    uint32_t mmSinnerBasicBlockTail = sInnerLoopTimes - mmRatio * (mm1LoopEndIdx - 1);

    LocalTensor<T> bmm2ResPreUb = tmpMMResBmm2Queue.AllocTensor<T>();
    for (int64_t mm1LoopIdx = 0; mm1LoopIdx < mm1LoopEndIdx; mm1LoopIdx++) {
      uint32_t sInnerStart = sInnerLoopStartIndex + mm1LoopIdx * mmRatio;
      // tail or fore
      uint32_t sLoopTimeCur = mm1LoopIdx == (mm1LoopEndIdx - 1) ? mmSinnerBasicBlockTail : mmRatio;
      mm.SetTensorA(queryGm[tensorACoreOffset]);
      mm.SetTensorB(keyGm[tensorBCoreOffset + sInnerStart * singleProcessSInnerSize *
      tilingData->attentionScoreOffestStrideParams.matmulHead], true);
      // matmul singleN
      uint32_t mmNNum = singleProcessSInnerSize * sLoopTimeCur;
      // sinner tail basic block
      if (sInnerLoopEndIndex == maxInnerLoopTimes && mm1LoopIdx == (mm1LoopEndIdx - 1)) {
        mmNNum = tilingData->baseParams.seqSizeK - singleProcessSInnerSize * (mmRatio * (mm1LoopEndIdx - 1) +
                 sInnerLoopStartIndex);
      }

      // bmm1 tail
      mm.SetTail(singleProcessSOuterSize, mmNNum);
      mm.template Iterate<false>();
      for (int64_t sInnerLoopIdx = sInnerStart; sInnerLoopIdx < sInnerStart + sLoopTimeCur; sInnerLoopIdx++) {
        // bmm1+vmuls+vadd+softmax+dropout
        if (sInnerLoopIdx > sInnerLoopStartIndex + 1) {
          isAdd = true;
        }
        Bmm1ResDoVecBmm2ComputeUpdate(bmm2ResPreUb, softmaxMaxUb, softmaxSumUb, softmaxExpUb, isAdd);
      }
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSpecialSparse<INPUT_T, T, isBasicBlock>::LoopSOuterOffsetInit()
{
    // S, S
    attenMaskCoreOffset = sOuterOffset * tilingData->baseParams.seqSizeK;
    if (tilingData->baseParams.attenHeadNum == tilingData->baseParams.headNumSize &&
        tilingData->baseParams.attenBatchSize == 1) {
        // 1, N, S, S
        attenMaskCoreOffset += batchNOffset * tilingData->baseParams.seqSizeQ * tilingData->baseParams.seqSizeK;
    } else if (tilingData->baseParams.attenHeadNum == 1 &&
        tilingData->baseParams.attenBatchSize == tilingData->baseParams.batchSize) {
        // B, 1, S, S
        attenMaskCoreOffset += batchBOffset * tilingData->baseParams.seqSizeQ * tilingData->baseParams.seqSizeK;
    } else if (tilingData->baseParams.attenHeadNum == tilingData->baseParams.headNumSize &&
        tilingData->baseParams.attenBatchSize == tilingData->baseParams.batchSize) {
        attenMaskCoreOffset += batchBOffset * tilingData->baseParams.attenHeadNum * tilingData->baseParams.seqSizeQ *
            tilingData->baseParams.seqSizeK +
            batchNOffset * tilingData->baseParams.seqSizeQ * tilingData->baseParams.seqSizeK;
    }
    maskCoreOffset = batchBOffset * tilingData->baseParams.headNumSize * tilingData->baseParams.seqSizeQ *
        tilingData->baseParams.seqSizeK +
        batchNOffset * tilingData->baseParams.seqSizeQ * tilingData->baseParams.seqSizeK +
        sOuterOffset * tilingData->baseParams.seqSizeK;
    if (tilingData->baseParams.pseShapeType == 0) { // B, N, S, S
        pseCoreOffset = maskCoreOffset;
    } else {
        pseCoreOffset = batchBOffset * tilingData->baseParams.headNumSize * tilingData->baseParams.seqSizeK +
            batchNOffset * tilingData->baseParams.seqSizeK;
    }
    if (tilingData->baseParams.transType == 0) {
        tensorACoreOffset =
            batchBOffset * tilingData->attentionScoreOffestStrideParams.matmulHead * tilingData->baseParams.seqSizeQ +
            batchNOffset * tilingData->baseParams.headSize +
            sOuterOffset * tilingData->attentionScoreOffestStrideParams.matmulHead;
        tensorBCoreOffset =
            batchBOffset * tilingData->attentionScoreOffestStrideParams.matmulHead * tilingData->baseParams.seqSizeK +
            batchNOffset * tilingData->baseParams.headSize;
    } else if (tilingData->baseParams.transType == 1) {
        tensorACoreOffset = sOuterOffset * tilingData->attentionScoreOffestStrideParams.matmulHead +
            batchBOffset * tilingData->baseParams.headSize * tilingData->baseParams.headNumSize +
            batchNOffset * tilingData->baseParams.headSize;
        tensorBCoreOffset = batchBOffset * tilingData->baseParams.headSize * tilingData->baseParams.headNumSize +
            batchNOffset * tilingData->baseParams.headSize;
    }

    valueCoreOffset = tensorBCoreOffset;
    softmaxOutOffset =
        batchBOffset * tilingData->baseParams.headNumSize * tilingData->baseParams.seqSizeQ * softmaxMaxTypeByteNum +
        batchNOffset * tilingData->baseParams.seqSizeQ * softmaxMaxTypeByteNum + sOuterOffset * softmaxMaxTypeByteNum;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSpecialSparse<INPUT_T, T, isBasicBlock>::ComputeEachCore(uint32_t coreIdx)
{
    uint64_t bIdx = coreIdx / (tilingData->attentionScoreCoreParams.n1 * tilingData->attentionScoreCoreParams.s1) %
        tilingData->attentionScoreCoreParams.b1 * tilingData->attentionScoreCoreParams.singleCoreBatchSize;
    uint64_t nIdx = coreIdx / tilingData->attentionScoreCoreParams.s1 % tilingData->attentionScoreCoreParams.n1;
    if (nIdx >= tilingData->attentionScoreCoreParams.formerHeadNum) {
        nIdx = (nIdx - tilingData->attentionScoreCoreParams.formerHeadNum) *
            tilingData->attentionScoreCoreParams.singleCoreHeadNumSizeTail +
            tilingData->attentionScoreCoreParams.formerHeadNum *
            tilingData->attentionScoreCoreParams.singleCoreHeadNumSize;
    } else {
        nIdx = nIdx * tilingData->attentionScoreCoreParams.singleCoreHeadNumSize;
    }
    uint64_t sIdx = coreIdx % tilingData->attentionScoreCoreParams.s1;

    if (attenMaskGm.GetPhyAddr() == nullptr) {
        if (sIdx >= this->tilingData->attentionScoreCoreParams.formerNum) {
            sIdx = (sIdx - this->tilingData->attentionScoreCoreParams.formerNum) *
                this->tilingData->attentionScoreCoreParams.singleCoreSeqSizeTail +
                this->tilingData->attentionScoreCoreParams.formerNum *
                this->tilingData->attentionScoreCoreParams.singleCoreSeqSize;
        } else {
            sIdx = sIdx * this->tilingData->attentionScoreCoreParams.singleCoreSeqSize;
        }
    } else {
        int sOuterIdx = 0;
        for (int i = 0; i < sIdx; i++) {
            sOuterIdx += tilingData->attentionScoreCoreParams.eachCoreSeqSize[i];
        }
        sIdx = sOuterIdx;
    }

    for (uint64_t loopBIdx = 0; loopBIdx < loopBNum; loopBIdx++) {
        // B offset
        batchBOffset = bIdx + tilingData->attentionScoreSingleCoreParams.singleProcessBatchSize * loopBIdx;
        for (uint64_t loopNIdx = 0; loopNIdx < loopNNum; loopNIdx++) {
            // N offset
            batchNOffset = nIdx + loopNIdx * tilingData->attentionScoreSingleCoreParams.singleProcessHeadNumSize;
            for (uint64_t sOuterLoopIdx = 0; sOuterLoopIdx < loopSNum; sOuterLoopIdx++) {
                // Souter offset
                sOuterOffset = sIdx + sOuterLoopIdx * singleProcessSOuterSize;
                LoopSOuterOffsetInit();
                sInnerLoopStartIndex = 0;
                sInnerLoopEndIndex =
                    (sOuterOffset + singleProcessSOuterSize + singleProcessSInnerSize - 1) / singleProcessSInnerSize;
                if (sInnerLoopEndIndex > maxInnerLoopTimes) {
                    sInnerLoopEndIndex = maxInnerLoopTimes;
                }
                if (attenMaskGm.GetPhyAddr() == nullptr) {
                    sInnerLoopEndIndex = maxInnerLoopTimes;
                }
                sInnerLoopTimes = sInnerLoopEndIndex - sInnerLoopStartIndex;
                SInnerLoopFunc(sOuterLoopIdx);
            }
        }
    }
}

#endif // FLASH_ATTENTION_SCORE_SPECIAL_SPARSE_BASE_H
