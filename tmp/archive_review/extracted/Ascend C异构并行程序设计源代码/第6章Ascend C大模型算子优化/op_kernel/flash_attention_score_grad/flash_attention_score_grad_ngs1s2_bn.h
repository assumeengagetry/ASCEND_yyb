/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_NGS1S2_BN_H_
#define FLASH_ATTENTION_SCORE_GRAD_NGS1S2_BN_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

#if __CCE_KT_TEST__
#define MIX_LOG1(...)  \
    do {   \
        if (mBlockIdx == 0) {  \
            MIX_LOG(__VA_ARGS__);  \
        }   \
    } while(0)

#else
#define MIX_LOG1(format, ...)
#endif

template <typename T1, typename T2, const bool IS_BF16 = false>  // T1 for data, T2 for vecClc
class FlashAttentionScoreGradUngs1s2Bbn {
public:
    __aicore__ inline FlashAttentionScoreGradUngs1s2Bbn()
    {};
    __aicore__ inline void Init(__gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dx, __gm__ uint8_t *query,
                                __gm__ uint8_t *pse_shift, __gm__ uint8_t *drop_mask, __gm__ uint8_t *atten_mask,
                                __gm__ uint8_t *forward_res, __gm__ uint8_t *softmax_max, __gm__ uint8_t *softmax_sum,
                                __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *workspace,
                                const FlashAttentionScoreGradTilingDataUngs1s2Bbn *__restrict ordTilingData,
                                TPipe *pipe_in);

    __aicore__ inline void Process();

    using biasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

    using VecT1True = MatmulType<TPosition::VECCALC, CubeFormat::ND, T1, true>;
    using VecT2False = MatmulType<TPosition::VECCALC, CubeFormat::ND_ALIGN, T2, false>;
    using VecT1False = MatmulType<TPosition::VECCALC, CubeFormat::ND, T1, false>;

    using GmT1True = MatmulType<TPosition::GM, CubeFormat::ND, T1, true>;
    using GmT1False = MatmulType<TPosition::GM, CubeFormat::ND, T1>;
    using GmT2False = MatmulType<TPosition::GM, CubeFormat::ND, T2>;

    Matmul<GmT1False, GmT1True, VecT2False, biasType> mm1;
    Matmul<GmT1False, GmT1True, VecT2False, biasType> mm2;
    Matmul<VecT1False, GmT1False, GmT2False, biasType> mm31;
    Matmul<VecT1True, GmT1False, GmT2False, biasType> mm32;
    Matmul<VecT1True, GmT1False, GmT1False, biasType> mm4;

protected:
    /* define the que */
    TQue<QuePosition::VECIN, 1> vecInQue1;
    TQue<QuePosition::VECIN, 1> vecInQue2;
    TBuf<> vecClc1;
    TBuf<> vecClc2;
    TBuf<> vecCast;
    TQue<QuePosition::VECOUT, 1> apiClcQue;

    const FlashAttentionScoreGradTilingDataUngs1s2Bbn *__restrict ordTilingData_;

    TPipe *pipe;

    GlobalTensor <uint8_t> attenMaskU8Gm;
    GlobalTensor <T1> keyGm, valueGm, dxGm, queryGm, attenMaskGm, forwardResGm, pseGm;
    GlobalTensor <T1> dqGm, dkGm, dvGm;
    GlobalTensor <T2> dqWorkspaceGm, dkWorkspaceGm;

    GlobalTensor<float> softmaxMaxGm, softmaxSumGm;
    GlobalTensor <T2> workspaceGm;
    GlobalTensor <int32_t> syncGlobal;
    GlobalTensor <uint8_t> dropoutWorkspaceGm;

    int64_t b;
    int64_t n;
    int64_t g;
    int64_t sQ;
    int64_t pseSq;
    int64_t sKV;
    int64_t sKVAlign;
    int64_t sKVAlignVec;
    int64_t sKVAlignByte;
    int64_t hQ;
    int64_t hKV;
    int64_t d;
    int64_t originalDAlign;
    int64_t attenMaskDimS2;
    float scaleValue;
    float keepProb;
    uint32_t preTokens;
    uint32_t nextTokens;
    uint32_t isFLash;
    uint32_t headNum;
    uint32_t inputLayout;  // 0:BSH 1:SBH 2:BNSD
    uint32_t preTokensBlocks;
    uint32_t nextTokensBlocks;
    uint32_t inputDType;
    uint32_t inputDTypeSize;
    uint32_t vecCalcDTypeSize;
    uint32_t hasAttenMask;
    uint32_t attenMaskShapeType;
    uint32_t elementPerBlock;

    int64_t totalBatch;
    int64_t sKVAlignSize;
    int64_t nOut;
    int64_t apiClcQueueSize;
    uint64_t usedCoreNum;

    int64_t nIn;
    int64_t nInTail;
    uint32_t singleCoreBatchRange;
    uint32_t singleCoreBatchRangeTail;
    uint32_t syncLen;
    uint32_t mm1WorkspaceLen;
    uint32_t mm2WorkspaceLen;
    uint32_t dqWorkspaceLen;
    uint32_t dkWorkspaceLen;
    int64_t innerTmpBufSize;
    int64_t vecCastSize;
    int64_t splitedDAlign;
    int64_t dRange;

    int64_t subRange;
    int64_t subMask;
    int64_t subMaskTail;
    int64_t sKVAlignBlockNumVec;

    int32_t innerReduceNum;
    int32_t innerMaskNum;
    int32_t innerMatResNum;
    int64_t innerMatResNumVec;
    int64_t maskInputNum;

    uint32_t softmaxGradInputShape[2];
    uint32_t innerMatOutShape[2];
    uint32_t maskInputShape[2];
    uint32_t innerReduceShape[2];
    bool isDrop;
    int64_t dropoutWorkspaceLen;
    uint64_t mBlockIdx;

    __aicore__ inline void FrontCompute(const uint64_t &batchSqLoopOffset, const uint64_t &batchSkvLoopOffset,
                                        const uint64_t &dropMaskOffset);

    __aicore__ inline void ReCompute(const uint64_t &batchSqLoopOffset, const uint64_t &batchSkvLoopOffset,
                                     const uint64_t& attenMaskOffset, const uint64_t &dropMaskOffset,
                                     const uint64_t &batchSoftmaxInputOffset, const uint64_t &batchPseOffset);

    __aicore__ inline void SetFrontClcShape(LocalTensor<T2> &sftFrontResInner, LocalTensor<T2> &frontResInner,
                                            LocalTensor<uint8_t> &dpMaskInner);

    __aicore__ inline void CopyInSoftMaxGrad(LocalTensor <T1> &dxInner, const GlobalTensor <T1> &dxGmIn,
                                             LocalTensor <T1> &forwardResInner,
                                             const GlobalTensor <T1> &forwardResGmIn, int64_t thisDAlign,
                                             int64_t thisD, int64_t dAlignOffset);

    __aicore__ inline void ClcSoftMaxGrad(LocalTensor<T2> &softmaxGradOutUb, LocalTensor<T1> &dxUb,
                                          LocalTensor<T1> &attentionInUb, int64_t thisDAlign);

    __aicore__ inline void HandleSoftMaxGrad(const uint64_t &batchSqLoopOffset, LocalTensor<T2> &softmaxGradOutUb);

    __aicore__ inline void ClcSoftMaxGradSplitD(LocalTensor<T2> &softmaxGradOutUb, LocalTensor<T1> &dxUb,
                                                LocalTensor<T1> &attentionInUb, int64_t thisDAlign);

    __aicore__ inline void HandleSoftMaxGradSplitD(const uint64_t &batchSqLoopOffset,
                                                   LocalTensor<T2> &softmaxGradOutUb);
    __aicore__ inline void CopyInDrop(LocalTensor <uint8_t> &dst, const GlobalTensor <uint8_t> &src);

    __aicore__ inline void ClcDrop(LocalTensor <T2> &dst, LocalTensor <T2> &src1, LocalTensor <uint8_t> &src2);

    __aicore__ inline void ClcSub(LocalTensor <T2> &frontResInner, LocalTensor <T2> &dpResInner,
                                  LocalTensor <T2> &sftFrontResInner);

    __aicore__ inline void SetReClcShape(LocalTensor <T2> &mulResInner, LocalTensor<float> &maxInner,
                                         LocalTensor<float> &sumInner, LocalTensor <T2> &dvDropResInner);

    __aicore__ inline void CopyInSoftMax(LocalTensor<float> &maxInner, const GlobalTensor<float> &softmaxMaxGmIn,
                                         LocalTensor<float> &sumInner, const GlobalTensor<float> &softmaxSumGmIn);

    __aicore__ inline bool CopyInPse(const uint64_t &batchPseOffset);

    __aicore__ inline bool CopyInAttenMask(const uint64_t &attenMaskOffset);

    __aicore__ inline void ClcPse(LocalTensor <T2> &mmResUb);

    __aicore__ inline void ClcPseBroadcast(LocalTensor <T2> &mmResUb, LocalTensor <T2> pseUb, uint32_t eleNum);

    __aicore__ inline void ClcPseBroadcastInner(uint32_t batchIndex, uint32_t repeatTimes, uint32_t outerOffset,
                                                uint32_t innerLoop, uint32_t innerRemain, const LocalTensor <T2> &pseUb,
                                                const LocalTensor <T2> &mmResUb, uint32_t stepCalcNum,
                                                const BinaryRepeatParams &repParams);
    __aicore__ inline void CalcAttenMaskOffset(uint64_t& attenMaskOffset, const int64_t delta);

    __aicore__ inline void CalcCausalAttenMaskOffset(uint64_t& attenMaskOffset, const int64_t delta);

    __aicore__ inline void ClcAttenMask(LocalTensor <T2> &mmResUb);

    __aicore__ inline void ClcSoftMax(LocalTensor <T2> &softmaxResInner, LocalTensor <T2> &reMatmulResInner,
                                      LocalTensor<float> &maxInner, LocalTensor<float> &sumInner);

    __aicore__ inline void ClcMm31(const LocalTensor <T1> &tensorA, const GlobalTensor <T1> &tensorB,
                                   const GlobalTensor <T2> &tensorC);

    /* matmul 32 和 matmul31的区别在于输入TensorA是需要做Transpose的，且输出需要做G轴的reduce. */
    __aicore__ inline void ClcMm32(const LocalTensor <T1> &tensorA, const GlobalTensor <T1> &tensorB,
                                   const GlobalTensor <T2> &tensorC);

    __aicore__ inline void ClcMm4(const LocalTensor <T1> &tensorA, const GlobalTensor <T1> &tensorB,
                                  const GlobalTensor <T1> &tensorC);

    __aicore__ inline void ClcMatmul(const uint64_t &batchSqLoopOffset, const uint64_t &batchSkvLoopOffset,
                                     LocalTensor <T2> &mulResInner, LocalTensor <T2> &dvDropResInner);
};

#if __CCE_KT_TEST__
template <typename T1>
inline void Ngs1s2BnPrint(const GlobalTensor<T1> global, uint32_t len1, uint32_t printLen)
{
    std::ostringstream os_;
    os_.str("");
    uint32_t blockNum = ONE_BLK_SIZE / sizeof(T1);
    const int32_t width = 4;

    for (uint32_t i = len1; i < printLen; i++) {
        if (i % blockNum == 0) {
            os_ << std::setw(width) << std::setfill('0') << (i / blockNum) * blockNum << " : ";
        }

        if constexpr (std::is_same<T1, float>::value) {
            os_ << global.GetValue(i) << " ";
        } else {
            os_ << global.GetValue(i).ToFloat() << " ";
        }

        if ((i + 1) % blockNum == 0) {
            os_ << std::endl;
        }
    }
    os_ << std::endl;
    std::cout << os_.str();
}

template <typename T1>
inline void Ngs1s2BnPrint(const LocalTensor<T1> global, uint32_t len1, uint32_t printLen)
{
    std::ostringstream os_;
    os_.str("");
    uint32_t blockNum = ONE_BLK_SIZE / sizeof(T1);
    const int32_t width = 4;

    for (uint32_t i = len1; i < printLen; i++) {
        if (i % blockNum == 0) {
            os_ << std::setw(width) << std::setfill('0') << (i / blockNum) * blockNum << " : ";
        }

        if constexpr (std::is_same<T1, float>::value) {
            os_ << global.GetValue(i) << " ";
        } else {
            os_ << global.GetValue(i).ToFloat() << " ";
        }

        if ((i + 1) % blockNum == 0) {
            os_ << std::endl;
        }
    }
    os_ << std::endl;
    std::cout << os_.str();
}
template <typename T1>
inline void Ngs1s2BnRightPad(const LocalTensor<T1> global, uint32_t dim1, uint32_t dim2, uint32_t padNum, T1 value)
{
    for (uint32_t i = 0; i < dim1; i++) {
        for (uint32_t j = dim2 - padNum; j < dim2; j++) {
            global.SetValue(i * dim2 + j, value);
        }
    }
}
#endif  // __CCE_KT_TEST__

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUngs1s2Bbn<T1, T2, IS_BF16>::Init(
    __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* dx, __gm__ uint8_t* query, __gm__ uint8_t* pse_shift,
    __gm__ uint8_t* drop_mask, __gm__ uint8_t* atten_mask, __gm__ uint8_t* forward_res, __gm__ uint8_t* softmax_max,
    __gm__ uint8_t* softmax_sum, __gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv, __gm__ uint8_t* workspace,
    const FlashAttentionScoreGradTilingDataUngs1s2Bbn* __restrict ordTilingData, TPipe* pipe_in)
{
    mBlockIdx = GetBlockIdx();
    keyGm.SetGlobalBuffer((__gm__ T1 *) key);
    valueGm.SetGlobalBuffer((__gm__ T1 *) value);
    dxGm.SetGlobalBuffer((__gm__ T1 *) dx);
    queryGm.SetGlobalBuffer((__gm__ T1 *) query);
    pseGm.SetGlobalBuffer((__gm__ T1 *) pse_shift);
    attenMaskGm.SetGlobalBuffer((__gm__ T1 *) atten_mask);
    forwardResGm.SetGlobalBuffer((__gm__ T1 *) forward_res);
    attenMaskU8Gm.SetGlobalBuffer((__gm__ uint8_t *) atten_mask);
    softmaxMaxGm.SetGlobalBuffer((__gm__ float *) softmax_max);
    softmaxSumGm.SetGlobalBuffer((__gm__ float *) softmax_sum);
    dqGm.SetGlobalBuffer((__gm__ T1 *) dq);
    dkGm.SetGlobalBuffer((__gm__ T1 *) dk);
    dvGm.SetGlobalBuffer((__gm__ T1 *) dv);
    workspaceGm.SetGlobalBuffer((__gm__ T2 *) workspace);
    syncGlobal.SetGlobalBuffer((__gm__ int32_t *)workspace, 100 * 8);
    dropoutWorkspaceGm.SetGlobalBuffer((__gm__ uint8_t*)workspace + 3200);
    InitOutput<int32_t>(syncGlobal[GetBlockIdx() * 8], 8, 0);

    ordTilingData_ = ordTilingData;
    pipe = pipe_in;

    b = ordTilingData_->opInfo.b;
    n = ordTilingData_->opInfo.n;
    g = ordTilingData_->opInfo.g;
    sQ = ordTilingData_->opInfo.sQ;
    pseSq = ordTilingData_->opInfo.pseSq;
    sKV = ordTilingData_->opInfo.sKV;
    sKVAlign = ordTilingData_->opInfo.sKVAlign;
    sKVAlignVec = ordTilingData_->opInfo.sKVAlignVec;
    sKVAlignByte = ordTilingData_->opInfo.sKVAlignByte;
    hQ = ordTilingData_->opInfo.hQ;
    hKV = ordTilingData_->opInfo.hKV;
    d = ordTilingData_->opInfo.d;
    originalDAlign = ordTilingData_->opInfo.originalDAlign;
    attenMaskDimS2 = ordTilingData_->opInfo.attenMaskS2Size;
    scaleValue = ordTilingData_->opInfo.scaleValue;
    keepProb = ordTilingData_->opInfo.keepProb;
    preTokens = ordTilingData_->opInfo.preTokens;
    nextTokens = ordTilingData_->opInfo.nextTokens;
    headNum = ordTilingData_->opInfo.headNum;
    inputLayout = ordTilingData_->opInfo.inputLayout;
    preTokensBlocks = ordTilingData_->opInfo.preTokensBlocks;
    nextTokensBlocks = ordTilingData_->opInfo.nextTokensBlocks;
    inputDType = ordTilingData_->opInfo.inputDType;
    inputDTypeSize = ordTilingData_->opInfo.inputDTypeSize;
    vecCalcDTypeSize = ordTilingData_->opInfo.vecCalcDTypeSize;
    attenMaskShapeType = ordTilingData_->opInfo.attenMaskShapeType;
    hasAttenMask = ordTilingData_->opInfo.hasAttenMask;
    elementPerBlock = ordTilingData_->opInfo.elementPerBlock;
    sKVAlignSize = ordTilingData_->opInfo.sKVAlignSize;

    totalBatch = ordTilingData_->splitCoreParams.totalBatch;
    nOut = ordTilingData_->splitCoreParams.nOut;
    apiClcQueueSize = ordTilingData_->splitCoreParams.apiClcQueueSize;
    usedCoreNum = ordTilingData_->splitCoreParams.usedCoreNum;

    nIn = ordTilingData_->singleCoreParams.nIn;
    nInTail = ordTilingData_->singleCoreParams.nInTail;
    singleCoreBatchRange = ordTilingData_->singleCoreParams.singleCoreBatchRange;
    singleCoreBatchRangeTail = ordTilingData_->singleCoreParams.singleCoreBatchRangeTail;
    syncLen = ordTilingData_->opInfo.syncLen;
    mm1WorkspaceLen = ordTilingData_->opInfo.mm1WorkspaceLen;
    mm2WorkspaceLen = ordTilingData_->opInfo.mm2WorkspaceLen;
    dqWorkspaceLen = ordTilingData_->mulsParams.dqWorkspaceLen;
    dkWorkspaceLen = ordTilingData_->mulsParams.dkWorkspaceLen;
    innerTmpBufSize = ordTilingData_->singleCoreParams.innerTmpBufSize;
    vecCastSize = ordTilingData_->singleCoreParams.vecCastSize;
    splitedDAlign = ordTilingData_->singleCoreParams.splitedDAlign;
    dRange = ordTilingData_->singleCoreParams.dRange;

    subRange = ordTilingData_->singleCoreParams.subRange;
    subMask = ordTilingData_->singleCoreParams.subMask;
    subMaskTail = ordTilingData_->singleCoreParams.subMaskTail;
    sKVAlignBlockNumVec = ordTilingData_->singleCoreParams.sKVAlignBlockNumVec;

    dropoutWorkspaceLen = ordTilingData_->dropoutCastParams.dropoutWorkspaceLen;
    uint64_t usedWorkspaceLen = syncLen + dropoutWorkspaceLen + mm1WorkspaceLen + mm2WorkspaceLen;
    auto dqAddr = usedWorkspaceLen / sizeof(T2);
    auto dkAddr = dqAddr + dqWorkspaceLen / sizeof(T2);
    dqWorkspaceGm.SetGlobalBuffer((__gm__ T2*)workspace + dqAddr);
    dkWorkspaceGm.SetGlobalBuffer((__gm__ T2*)workspace + dkAddr);
    // 对于dropout mask其size是nIn * g * sQ * sKVAlignByte， sKVAlignByte这个值一定小于等于sKVAlignSize
    // 例如sKVALign是17，sKVAlignByte = 32， sKVAlignSize = 64
    // sKVAlign是32， sKVAlignByte = 32， sKVAlignSize = 64
    int64_t pseAndMaskInputSize = nIn * g * sQ * sKVAlignSize;
    pipe->InitBuffer(vecInQue1, 2, ordTilingData_->singleCoreParams.vecQueIn1Size);
    pipe->InitBuffer(vecInQue2, 1, pseAndMaskInputSize);
    // Bf16场景下vecClc1和vecClc2会被softmax的输入的Cast复用
    // tmpBufSize在计算时会取splitedDAlign和sKvAlign的较大值
    pipe->InitBuffer(vecClc1, innerTmpBufSize);
    pipe->InitBuffer(vecClc2, innerTmpBufSize);
    // D切分才需要申请额外的vecCast用于attenin的cast
    if (vecCastSize > 0) {
        pipe->InitBuffer(vecCast, vecCastSize);
    }
    pipe->InitBuffer(apiClcQue, 1, apiClcQueueSize);
    DropOutBitModeInit();
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUngs1s2Bbn<T1, T2, IS_BF16>::CopyInSoftMaxGrad(
    LocalTensor<T1>& dxInner, const GlobalTensor<T1>& dxGmIn, LocalTensor<T1>& forwardResInner,
    const GlobalTensor<T1>& forwardResGmIn, int64_t thisDAlign, int64_t thisD, int64_t dAlignOffset)
{
    DataCopyParams copyParams;
    DataCopyPadParams copyPadParams;
    uint64_t ubOffset = 0;
    uint64_t gmOffset = 0;
    copyParams.blockCount = sQ;
    copyParams.blockLen = thisD * sizeof(T1);
    copyParams.dstStride = 0;

    copyPadParams.isPad = true;
    copyPadParams.leftPadding = 0;
    copyPadParams.rightPadding = (thisDAlign - thisD);
    copyPadParams.paddingValue = 0;

    for (uint64_t copyIndex = 0; copyIndex < static_cast<uint64_t>(nIn * g); ++copyIndex) {
        if (inputLayout == 1) {  // SBH SBND
            gmOffset = copyIndex * d + dAlignOffset;
            copyParams.srcStride = (b * hQ - thisD) * sizeof(T1);
        } else if (inputLayout == 2) {  // BNSD
            gmOffset = copyIndex * sQ * d + dAlignOffset;
            copyParams.srcStride = (d - thisD) * sizeof(T1);
        } else {  // BSH BSDN
            gmOffset = copyIndex * d + dAlignOffset;
            copyParams.srcStride = (hQ - thisD) * sizeof(T1);
        }

        DataCopyPad(dxInner[ubOffset], dxGmIn[gmOffset], copyParams, copyPadParams);
        DataCopyPad(forwardResInner[ubOffset], forwardResGmIn[gmOffset], copyParams, copyPadParams);
        ubOffset += sQ * thisDAlign;
    }
    vecInQue1.EnQue(forwardResInner); // enque,等待数据copy，dxUb和dxUb同一个que的，共用一个enque操作
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUngs1s2Bbn<T1, T2, IS_BF16>::SetFrontClcShape(
    LocalTensor<T2> &sftFrontResUb, LocalTensor<T2> &frontResUb, LocalTensor<uint8_t> &dpMaskUb)
{
    uint32_t innerFrontReClcShape[2] = {static_cast<uint32_t>(sQ), static_cast<uint32_t>(nIn * g *(32 / sizeof(T2)))};
    sftFrontResUb.SetShapeInfo(ShapeInfo(2, innerFrontReClcShape, DataFormat::ND));
    frontResUb.SetShapeInfo(ShapeInfo(2, innerMatOutShape, DataFormat::ND));
    dpMaskUb.SetShapeInfo(ShapeInfo(2, innerMatOutShape, DataFormat::ND));
}

/* D不切分 直接复用softmaxGradOutUb结果内存，计算的结果放在softmaxGradOutUb */
template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUngs1s2Bbn<T1, T2, IS_BF16>::ClcSoftMaxGrad(
    LocalTensor<T2> &softmaxGradOutUb, LocalTensor<T1> &dxUb,
    LocalTensor<T1> &attentionInUb, int64_t thisDAlign)
{
    vecInQue1.DeQue<T1>(); // deque一把等待数据copy完，dxUb和dxUb同一个que的，共用一个deque操作
    auto softmaxGradInputNum = softmaxGradInputShape[0] * thisDAlign;
    bool isBasicBlock = (sQ % 8 == 0) && (thisDAlign % 64 == 0);
    LocalTensor <uint8_t> apiClcTensor = apiClcQue.AllocTensor<uint8_t>();
    apiClcTensor.SetSize(apiClcQueueSize);

    if constexpr (IS_BF16) {
        pipe_barrier(PIPE_V);
        LocalTensor<T2> castedAttentionIn = vecClc2.Get<T2>(softmaxGradInputNum); // 临时借用softmaxGradOutUb内存
        LocalTensor<T2> castedDx = vecClc1.Get<T2>(softmaxGradInputNum);
        castedAttentionIn.SetShapeInfo(ShapeInfo(2, softmaxGradInputShape, DataFormat::ND));
        castedDx.SetShapeInfo(ShapeInfo(2, softmaxGradInputShape, DataFormat::ND));
        Cast(castedAttentionIn, attentionInUb, RoundMode::CAST_NONE, softmaxGradInputNum);
        Cast(castedDx, dxUb, RoundMode::CAST_NONE, softmaxGradInputNum);
        pipe_barrier(PIPE_V);

        if (isBasicBlock) {
            SoftmaxGradFront<T2, true>(softmaxGradOutUb, castedAttentionIn, castedDx, apiClcTensor,
                                       ordTilingData_->softmaxGradTilingData);
        } else {
            SoftmaxGradFront<T2, false>(softmaxGradOutUb, castedAttentionIn, castedDx, apiClcTensor,
                                        ordTilingData_->softmaxGradTilingData);
        }
    } else {
        if (isBasicBlock) {
            SoftmaxGradFront<T2, true>(softmaxGradOutUb, attentionInUb, dxUb,
                                       apiClcTensor, ordTilingData_->softmaxGradTilingData);
        } else {
            SoftmaxGradFront<T2, false>(softmaxGradOutUb, attentionInUb, dxUb,
                                        apiClcTensor, ordTilingData_->softmaxGradTilingData);
        }
    }

    apiClcQue.FreeTensor(apiClcTensor);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUngs1s2Bbn<T1, T2, IS_BF16>::HandleSoftMaxGrad(
    const uint64_t &batchSqLoopOffset, LocalTensor<T2> &softmaxGradOutUb) {
    LocalTensor<T1> attentionInUb = vecInQue1.AllocTensor<T1>();
    LocalTensor<T1> dxUb = vecInQue1.AllocTensor<T1>();
    int64_t softmaxGradInputNum = softmaxGradInputNum = softmaxGradInputShape[0] * originalDAlign;
    dxUb.SetSize(softmaxGradInputNum);
    attentionInUb.SetSize(softmaxGradInputNum);

    softmaxGradInputShape[1] = originalDAlign;
    dxUb.SetShapeInfo(ShapeInfo(2, softmaxGradInputShape, DataFormat::ND));
    attentionInUb.SetShapeInfo(ShapeInfo(2, softmaxGradInputShape, DataFormat::ND));

    CopyInSoftMaxGrad(dxUb, dxGm[batchSqLoopOffset], attentionInUb, forwardResGm[batchSqLoopOffset],
                      originalDAlign, d, 0);

    ClcSoftMaxGrad(softmaxGradOutUb, dxUb, attentionInUb, originalDAlign);

    vecInQue1.FreeTensor(attentionInUb);
    vecInQue1.FreeTensor(dxUb);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUngs1s2Bbn<T1, T2, IS_BF16>::ClcSoftMaxGradSplitD(
    LocalTensor<T2> &softmaxGradOutUb, LocalTensor<T1> &dxUb,
    LocalTensor<T1> &attentionInUb, int64_t thisDAlign)
{
    vecInQue1.DeQue<T1>(); // deque一吧等待数据copyin完，dxUb和dxUb同一个que的，共用一个deque操作
    auto softmaxGradInputNum = softmaxGradInputShape[0] * thisDAlign;
    bool isBasicBlock = (sQ % 8 == 0) && (thisDAlign % 64 == 0);
    LocalTensor <uint8_t> apiClcTensor = apiClcQue.AllocTensor<uint8_t>(); // apiClcTensor可以外提
    apiClcTensor.SetSize(apiClcQueueSize);

    /* 这里复用了Calc1，这时mm还没有GetTensorC，Clc1处于闲置状态 */
    LocalTensor<T2> softmaxGradTempRes = vecClc1.Get<T2>(softmaxGradOutUb.GetSize());
    if constexpr (IS_BF16) {
        pipe_barrier(PIPE_V);
        LocalTensor<T2> castedAttentionIn = vecCast.Get<T2>(softmaxGradInputNum);
        LocalTensor<T2> castedDx = vecClc1.Get<T2>(softmaxGradInputNum);
        castedAttentionIn.SetShapeInfo(ShapeInfo(2, softmaxGradInputShape, DataFormat::ND));
        castedDx.SetShapeInfo(ShapeInfo(2, softmaxGradInputShape, DataFormat::ND));
        Cast(castedAttentionIn, attentionInUb, RoundMode::CAST_NONE, softmaxGradInputNum);
        Cast(castedDx, dxUb, RoundMode::CAST_NONE, softmaxGradInputNum);
        pipe_barrier(PIPE_V);

        if (isBasicBlock) {
            SoftmaxGradFront<T2, true>(softmaxGradTempRes, castedAttentionIn, castedDx, apiClcTensor,
                                       ordTilingData_->softmaxGradTilingData);
        } else {
            SoftmaxGradFront<T2, false>(softmaxGradTempRes, castedAttentionIn, castedDx, apiClcTensor,
                                        ordTilingData_->softmaxGradTilingData);
        }
    } else {
        if (isBasicBlock) {
            SoftmaxGradFront<T2, true>(softmaxGradOutUb, attentionInUb, dxUb,
                                       apiClcTensor, ordTilingData_->softmaxGradTilingData);
        } else {
            SoftmaxGradFront<T2, false>(softmaxGradOutUb, attentionInUb, dxUb,
                                        apiClcTensor, ordTilingData_->softmaxGradTilingData);
        }
    }

    pipe_barrier(PIPE_V);
    Add(softmaxGradOutUb, softmaxGradOutUb, softmaxGradTempRes, softmaxGradOutUb.GetSize());
    pipe_barrier(PIPE_V);

    apiClcQue.FreeTensor(apiClcTensor);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUngs1s2Bbn<T1, T2, IS_BF16>::HandleSoftMaxGradSplitD(
    const uint64_t &batchSqLoopOffset, LocalTensor<T2> &softmaxGradOutUb) {
    int64_t softmaxGradInputNum = 0;
    // 清空softmaxGradOutUb
    Duplicate<T2>(softmaxGradOutUb, 0, softmaxGradOutUb.GetSize());
    for (auto i = 0; i < dRange; ++i) {
        LocalTensor<T1> attentionInUb = vecInQue1.AllocTensor<T1>(); // 必须要放在循环内，做MTE和v之间的pipe等待
        LocalTensor<T1> dxUb = vecInQue1.AllocTensor<T1>();
        auto dAlignOffset = i * splitedDAlign;
        auto remainDAlign = originalDAlign - dAlignOffset;
        int64_t thisDAlign = 0;
        int64_t thisD = 0;
        if (remainDAlign < splitedDAlign) {
            thisDAlign = remainDAlign;
            thisD = d - dAlignOffset;
        } else {
            /* 非尾块都是对齐的 */
            thisDAlign = splitedDAlign;
            thisD = splitedDAlign;
        }
        softmaxGradInputNum = softmaxGradInputShape[0] * thisDAlign;
        dxUb.SetSize(softmaxGradInputNum);
        attentionInUb.SetSize(softmaxGradInputNum);

        softmaxGradInputShape[1] = thisDAlign;
        dxUb.SetShapeInfo(ShapeInfo(2, softmaxGradInputShape, DataFormat::ND));
        attentionInUb.SetShapeInfo(ShapeInfo(2, softmaxGradInputShape, DataFormat::ND));

        CopyInSoftMaxGrad(dxUb, dxGm[batchSqLoopOffset], attentionInUb, forwardResGm[batchSqLoopOffset],
                          thisDAlign, thisD, dAlignOffset);
        ClcSoftMaxGradSplitD(softmaxGradOutUb, dxUb, attentionInUb, thisDAlign);
        vecInQue1.FreeTensor(attentionInUb);
        vecInQue1.FreeTensor(dxUb);
    }
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUngs1s2Bbn<T1, T2, IS_BF16>::ClcSub(LocalTensor<T2>& frontResInner,
                                                                                  LocalTensor<T2>& dpResInner,
                                                                                  LocalTensor<T2>& sftFrontResInner)
{
    // [m,n] - [m,16] -> [m,n] 按n轴的block数repeat，每个指令repeat算[m,16] - [m,16], subRange循环处理 超过mask情况
    for (uint32_t batchIndex = 0; batchIndex < static_cast<uint32_t>(nIn * g); ++batchIndex) {
        for (int32_t subIdx = 0; subIdx < static_cast<int32_t>(subRange); ++subIdx) {
            uint64_t src0Offset = batchIndex * sQ * sKVAlignVec + subIdx * sKVAlignVec * 8;
            uint64_t src1Offset = batchIndex * sQ * (32 / sizeof(T2)) + subIdx * subMask;
            if (subIdx == subRange - 1 && subMaskTail != 0) {
                Sub(frontResInner[src0Offset], dpResInner[src0Offset], sftFrontResInner[src1Offset], subMaskTail,
                    sKVAlignBlockNumVec,
                    {(uint8_t) (sKVAlignBlockNumVec), (uint8_t) (sKVAlignBlockNumVec), 1, 1, 1, 0});
            } else {
                // dst_blk_stride, src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride, src1_rep_stride
                Sub(frontResInner[src0Offset], dpResInner[src0Offset], sftFrontResInner[src1Offset], subMask,
                    sKVAlignBlockNumVec,
                    {(uint8_t) (sKVAlignBlockNumVec), (uint8_t) (sKVAlignBlockNumVec), 1, 1, 1, 0});
            }
        }
    }
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUngs1s2Bbn<T1, T2, IS_BF16>::SetReClcShape(
    LocalTensor<T2>& mulResInner, LocalTensor<float>& maxInner, LocalTensor<float>& sumInner,
    LocalTensor<T2>& dvDropResInner)
{
    mulResInner.SetShapeInfo(ShapeInfo(2, innerMatOutShape, DataFormat::ND));
    maxInner.SetShapeInfo(ShapeInfo(2, innerReduceShape, DataFormat::ND));
    sumInner.SetShapeInfo(ShapeInfo(2, innerReduceShape, DataFormat::ND));
    dvDropResInner.SetShapeInfo(ShapeInfo(2, innerMatOutShape, DataFormat::ND));
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUngs1s2Bbn<T1, T2, IS_BF16>::CopyInSoftMax(
    LocalTensor<float>& maxInner, const GlobalTensor<float>& softmaxMaxGmIn, LocalTensor<float>& sumInner,
    const GlobalTensor<float>& softmaxSumGmIn)
{
    DataCopy(maxInner, softmaxMaxGmIn, innerReduceNum);
    DataCopy(sumInner, softmaxSumGmIn, innerReduceNum);
    vecInQue1.EnQue(sumInner);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline bool FlashAttentionScoreGradUngs1s2Bbn<T1, T2, IS_BF16>::CopyInPse(const uint64_t& batchPseOffset)
{
    if (pseSq == 0) {
        return false;
    }
    DataCopyParams copyParams;
    DataCopyPadParams copyPadParams;
    uint64_t ubOffset = 0;
    copyParams.blockCount = nIn * g * pseSq;
    copyParams.blockLen = sKV * sizeof(T1);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    copyPadParams.isPad = true;
    copyPadParams.leftPadding = 0;
    copyPadParams.rightPadding = (sKVAlign - sKV);
    copyPadParams.paddingValue = 0;
    LocalTensor<T1> pseUb = vecInQue2.AllocTensor<T1>();

    pseUb.SetSize(nIn * g * pseSq * sKVAlign);
    DataCopyPad(pseUb[ubOffset], pseGm[batchPseOffset], copyParams, copyPadParams);

    vecInQue2.EnQue(pseUb);
    return true;
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline bool FlashAttentionScoreGradUngs1s2Bbn<T1, T2, IS_BF16>::CopyInAttenMask(
    const uint64_t &attenMaskOffset)
{
    if (hasAttenMask != 1) {
        return false;
    }
    DataCopyParams copyParams;
    LocalTensor <uint8_t> attenMaskUb = vecInQue2.AllocTensor<uint8_t>();
    attenMaskUb.SetSize(nIn * g * sQ * sKVAlignByte);
    copyParams.blockCount = sQ;
    copyParams.blockLen = sKV;
    copyParams.srcStride = attenMaskDimS2 - sKV;
    copyParams.dstStride = 0;

    for (int64_t copyIndex = 0; copyIndex < nIn * g; ++copyIndex) {
        DataCopyPad(attenMaskUb[copyIndex * sQ * sKVAlignByte], attenMaskU8Gm[attenMaskOffset],
                                                                copyParams, {false, 0, 0, 0});
    }
    vecInQue2.EnQue(attenMaskUb);
    return true;
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUngs1s2Bbn<T1, T2, IS_BF16>::ClcPse(LocalTensor<T2>& mmResUb)
{
    LocalTensor<T1> pseUb = vecInQue2.DeQue<T1>();
    uint32_t eleNum = pseUb.GetSize();

    if constexpr (IS_BF16) {
        auto castedPseUb = vecInQue1.AllocTensor<T2>();
        castedPseUb.SetSize(eleNum);
        for (int64_t i = 0; i < nIn * g * pseSq; ++i) {
            Cast(castedPseUb[i * sKVAlignVec], pseUb[i * sKVAlign], RoundMode::CAST_NONE, sKVAlign);
        }
        pipe_barrier(PIPE_V);
        ClcPseBroadcast(mmResUb, castedPseUb, eleNum);
        vecInQue1.FreeTensor(castedPseUb);
    } else {
        ClcPseBroadcast(mmResUb, pseUb, eleNum);
    }
    vecInQue2.FreeTensor(pseUb);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUngs1s2Bbn<T1, T2, IS_BF16>::ClcPseBroadcast(LocalTensor<T2>& mmResUb,
                                                                                           LocalTensor<T2> pseUb,
                                                                                           uint32_t eleNum)
{
    if (pseSq == sQ) {  // BNSS
        Add(mmResUb, mmResUb, pseUb, eleNum);
    } else {  // BN1S
        int32_t stepCalcNum = 256 / sizeof(T2);
        stepCalcNum = sKVAlignVec > stepCalcNum ? stepCalcNum : sKVAlignVec;
        uint32_t innerLoop = sKVAlignVec / stepCalcNum;
        uint32_t innerRemain = sKVAlignVec % stepCalcNum;
        uint32_t outerLoop = sQ / 255;
        uint32_t outerRemain = sQ % 255;

        /* Total data number of single step should be smaller than 256bytes.
         * If larger, we need to do add multiple times. */
        BinaryRepeatParams repParams;
        repParams.src0BlkStride = 1;
        repParams.src0RepStride = (sKVAlignVec) / (32 / sizeof(T2));
        repParams.src1BlkStride = 1;
        repParams.src1RepStride = 0;
        repParams.dstRepStride = repParams.src0RepStride;
        repParams.blockNumber = repParams.src0RepStride;

        for (uint32_t batchIndex = 0; batchIndex < static_cast<uint32_t>(nIn * g); ++batchIndex) {
            /* Total repeated times should be <= 255. If larger,
            * we need to do multiple inner loops. */
            for (uint32_t i = 0; i < outerLoop; ++i) {
                uint32_t outerOffset = batchIndex * sQ * sKVAlignVec + 255 * i * sQ * sKVAlignVec;
                ClcPseBroadcastInner(batchIndex, 255, outerOffset, innerLoop, innerRemain, pseUb,
                                     mmResUb, stepCalcNum, repParams);
            }
            if (outerRemain > 0) {
                uint32_t outerOffset = batchIndex * sQ * sKVAlignVec + outerLoop * 255 * sQ * sKVAlignVec;
                ClcPseBroadcastInner(batchIndex, outerRemain, outerOffset, innerLoop, innerRemain,
                                     pseUb, mmResUb, stepCalcNum, repParams);
            }
        }
    }
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUngs1s2Bbn<T1, T2, IS_BF16>::ClcPseBroadcastInner(
    uint32_t batchIndex, uint32_t repeatTimes, uint32_t outerOffset, uint32_t innerLoop, uint32_t innerRemain,
    const LocalTensor<T2>& pseUb, const LocalTensor<T2>& mmResUb, uint32_t stepCalcNum,
    const BinaryRepeatParams& repParams)
{
    for (uint32_t j = 0; j < innerLoop; ++j) {
        auto innerOffset = j * stepCalcNum;
        auto pseOffset = batchIndex * sKVAlignVec + innerOffset;
        auto dstUbOffset = outerOffset + innerOffset;
        Add(mmResUb[dstUbOffset], mmResUb[dstUbOffset], pseUb[pseOffset], stepCalcNum, repeatTimes, repParams);
    }
    if (innerRemain > 0) {
        auto innerOffset = innerLoop * stepCalcNum;
        auto pseOffset = batchIndex * sKVAlignVec + innerOffset;
        auto dstUbOffset = outerOffset + innerOffset;
        Add(mmResUb[dstUbOffset], mmResUb[dstUbOffset], pseUb[pseOffset], innerRemain, repeatTimes, repParams);
    }
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUngs1s2Bbn<T1, T2, IS_BF16>::ClcAttenMask(LocalTensor<T2>& mmResUb)
{
    LocalTensor<uint8_t> attenMaskUb = vecInQue2.DeQue<uint8_t>();
    LocalTensor<uint8_t> ubWorkspace = apiClcQue.AllocTensor<uint8_t>();

    T2 scalar;
    if constexpr (IsSameType<T2, float>::value) {
        uint32_t tmp = 0xFF7FFFFF;
        scalar = *((float*)&tmp);
    } else {
        uint16_t tmp = 0xFBFF;
        scalar = *((half*)&tmp);
    }
    SelectWithBytesMaskShapeInfo shapeInfo;
    shapeInfo.firstAxis = nIn * g * sQ;
    shapeInfo.srcLastAxis = sKVAlignVec;
    shapeInfo.maskLastAxis = sKVAlignByte;
    attenMaskUb.SetSize(shapeInfo.firstAxis * shapeInfo.maskLastAxis); // 需要设置，否则cpu跑的时候会报size不匹配的错误
    mmResUb.SetSize(shapeInfo.firstAxis * shapeInfo.srcLastAxis);
    SelectWithBytesMask(mmResUb, mmResUb, scalar, attenMaskUb, ubWorkspace, shapeInfo);

    apiClcQue.FreeTensor(ubWorkspace);
    vecInQue2.FreeTensor(attenMaskUb);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUngs1s2Bbn<T1, T2, IS_BF16>::ClcSoftMax(LocalTensor<T2>& softmaxResInner,
                                                                                      LocalTensor<T2>& reMatmulResInner,
                                                                                      LocalTensor<float>& maxInner,
                                                                                      LocalTensor<float>& sumInner)
{
    vecInQue1.DeQue<float>();
    LocalTensor <uint8_t> apiClcTensor = apiClcQue.AllocTensor<uint8_t>();
    apiClcTensor.SetSize(apiClcQueueSize);
    bool isBasicBlock = ((nIn * g * sQ) % 8 == 0) && (sKV % 64 == 0);
    if (isBasicBlock) {
        SimpleSoftMax<T2, true, true>(softmaxResInner, sumInner, maxInner, reMatmulResInner, apiClcTensor,
                                      ordTilingData_->softmaxTilingData);
    } else {
        SimpleSoftMax<T2, true, false>(softmaxResInner, sumInner, maxInner, reMatmulResInner, apiClcTensor,
                                       ordTilingData_->softmaxTilingData);
    }
    apiClcQue.FreeTensor(apiClcTensor);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUngs1s2Bbn<T1, T2, IS_BF16>::CopyInDrop(
    LocalTensor<uint8_t> &dst, const GlobalTensor<uint8_t> &src)
{
    dst.SetSize(maskInputNum);
    struct DataCopyParams copyParams;
    copyParams.blockCount = (uint16_t)(nIn * g * sQ);
    // DataCopyPad时需要填sKV实际长度，DataCopy时因为sKV是对齐的，也可以直接用sKV
    copyParams.blockLen = (uint16_t)sKV;
    copyParams.dstStride = 0;
    copyParams.srcStride = 0;
    DataCopyPadParams copyPadParams;
    copyPadParams.leftPadding = 0;
    copyPadParams.paddingValue = 0;
    if (sKV % 32 == 0) {
        copyPadParams.isPad = false;
        copyPadParams.rightPadding = 0;
        DataCopyPad(dst, src, copyParams, copyPadParams);
    } else {
        DataCopyPadParams copyPadParams;
        copyPadParams.isPad = true;
        copyPadParams.rightPadding = (sKVAlignByte - sKV);
        DataCopyPad(dst, src, copyParams, copyPadParams);
    }
    vecInQue2.EnQue(dst);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUngs1s2Bbn<T1, T2, IS_BF16>::ClcDrop(
    LocalTensor<T2> &dpResInner, LocalTensor<T2> &dpMatmmulResInner, LocalTensor<uint8_t> &dpMaskInner)
{
    vecInQue2.DeQue<uint8_t>();
    DropOutShapeInfo info;
    info.firstAxis = nIn * g * sQ;
    info.srcLastAxis = sKVAlignVec;
    info.maskLastAxis = sKVAlignByte;
    LocalTensor<uint8_t> apiClcTensor = apiClcQue.AllocTensor<uint8_t>();
    apiClcTensor.SetSize(ordTilingData_->splitCoreParams.apiClcQueueSize);
    DropOut<T2, true>(dpResInner, dpMatmmulResInner, dpMaskInner, apiClcTensor, keepProb, info);
    apiClcQue.FreeTensor(apiClcTensor);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUngs1s2Bbn<T1, T2, IS_BF16>::FrontCompute(
    const uint64_t& batchSqLoopOffset, const uint64_t& batchSkvLoopOffset,
    const uint64_t &dropMaskOffset)
{
    LocalTensor<T2> softmaxGradOutUb = vecClc2.Get<T2>(nIn * g * sQ * 32 / sizeof(T2)); // FlashSoftmaxGrad的计算结果
    LocalTensor<T2> frontResUb = vecClc1.Get<T2>(innerMatResNum); // Sub的计算结果
    LocalTensor<T2> &dpResUb = frontResUb; // Dropout计算结果 共用内存
    LocalTensor<T2> &dpMatmmulResUb = frontResUb; // BMM1计算结果 共用内存
    LocalTensor<uint8_t> dpMaskInner = vecInQue2.AllocTensor<uint8_t>(); // drop_mask的输入

    dpMaskInner.SetSize(innerMatResNum);
    SetFrontClcShape(softmaxGradOutUb, frontResUb, dpMaskInner);
    if (isDrop) {
        CopyInDrop(dpMaskInner, dropoutWorkspaceGm[dropMaskOffset]);
    }

    if (vecCastSize == 0) { // 不切D
        HandleSoftMaxGrad(batchSqLoopOffset, softmaxGradOutUb);
    } else {
        HandleSoftMaxGradSplitD(batchSqLoopOffset, softmaxGradOutUb);
    }

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
    mm1.template GetTensorC<false>(dpMatmmulResUb, false, true);
    mm1.End();

    if (isDrop) {
        ClcDrop(dpResUb, dpMatmmulResUb, dpMaskInner);
    }

    ClcSub(frontResUb, dpMatmmulResUb, softmaxGradOutUb);

    vecInQue2.FreeTensor(dpMaskInner);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUngs1s2Bbn<T1, T2, IS_BF16>::ClcMm31(const LocalTensor<T1>& tensorA,
                                                                                   const GlobalTensor<T1>& tensorB,
                                                                                   const GlobalTensor<T2>& tensorC)
{
    mm31.SetTensorA(tensorA, false);
    mm31.SetTensorB(tensorB, false);
    mm31.template IterateAll<true>(tensorC, false);
    mm31.End();
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUngs1s2Bbn<T1, T2, IS_BF16>::ClcMm32(
    const LocalTensor<T1>& tensorA, const GlobalTensor<T1>& tensorB, const GlobalTensor<T2>& tensorC)
{
    mm32.SetTensorA(tensorA, true);
    mm32.SetTensorB(tensorB, false);
    mm32.template IterateAll<true>(tensorC, false);
    mm32.End();
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUngs1s2Bbn<T1, T2, IS_BF16>::ClcMm4(
    const LocalTensor<T1>& tensorA, const GlobalTensor<T1>& tensorB, const GlobalTensor<T1>& tensorC)
{
    mm4.SetTensorA(tensorA, true);
    mm4.SetTensorB(tensorB, false);
    mm4.template IterateAll<true>(tensorC, false);
    mm4.End();
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUngs1s2Bbn<T1, T2, IS_BF16>::ClcMatmul(const uint64_t& batchSqLoopOffset,
                                                                                     const uint64_t& batchSkvLoopOffset,
                                                                                     LocalTensor<T2>& mulResInner,
                                                                                     LocalTensor<T2>& dvDropResInner)
{
    if constexpr (IS_BF16) {
        LocalTensor <T1> castedMulResPad = vecInQue1.AllocTensor<T1>();
        pipe_barrier(PIPE_V);
        castedMulResPad.SetSize(innerMatResNum);
        for (int64_t i = 0; i < sQ; ++i) {
            Cast(castedMulResPad[i * sKVAlign], mulResInner[i * sKVAlignVec], RoundMode::CAST_ROUND, sKVAlignVec);
        }
        pipe_barrier(PIPE_V);

        LocalTensor <T1> castedDvDropResPad = vecInQue1.AllocTensor<T1>();
        castedDvDropResPad.SetSize(innerMatResNum);
        for (uint64_t i = 0; i < sQ; ++i) {
            Cast(castedDvDropResPad[i * sKVAlign], dvDropResInner[i * sKVAlignVec], RoundMode::CAST_ROUND, sKVAlignVec);
        }
        pipe_barrier(PIPE_V);
        mm31.SetTail(sQ, -1, sKV);
        ClcMm31(castedMulResPad, keyGm[batchSkvLoopOffset], dqWorkspaceGm[batchSqLoopOffset]);
        mm32.SetTail(sKV, -1, sQ);
        ClcMm32(castedMulResPad, queryGm[batchSqLoopOffset], dkWorkspaceGm[batchSkvLoopOffset]);
        mm4.SetTail(sKV, -1, sQ);
        ClcMm4(castedDvDropResPad, dxGm[batchSqLoopOffset], dvGm[batchSkvLoopOffset]);
        vecInQue1.FreeTensor(castedMulResPad);
        vecInQue1.FreeTensor(castedDvDropResPad);
    } else {
        mm31.SetTail(sQ, -1, sKV);
        ClcMm31(mulResInner, keyGm[batchSkvLoopOffset], dqWorkspaceGm[batchSqLoopOffset]);
        mm32.SetTail(sKV, -1, sQ);
        ClcMm4(mulResInner, queryGm[batchSqLoopOffset], dkWorkspaceGm[batchSkvLoopOffset]);
        mm4.SetTail(sKV, -1, sQ);
        ClcMm4(dvDropResInner, dxGm[batchSqLoopOffset], dvGm[batchSkvLoopOffset]);
    }
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUngs1s2Bbn<T1, T2, IS_BF16>::CalcAttenMaskOffset(
    uint64_t& attenMaskOffset, const int64_t delta) {
    if (delta == 0) {
        attenMaskOffset = 0;
    } else if (delta < 0) {
        if (-delta > sQ) {
            attenMaskOffset = sQ;
        } else {
            attenMaskOffset = -delta;
        }
    } else {
        if (delta > sKV) {
            attenMaskOffset = sKV * attenMaskDimS2;
        } else {
            attenMaskOffset = delta * attenMaskDimS2;
        }
    }
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUngs1s2Bbn<T1, T2, IS_BF16>::CalcCausalAttenMaskOffset(
    uint64_t& attenMaskOffset, const int64_t delta) {
    CalcAttenMaskOffset(attenMaskOffset, delta);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUngs1s2Bbn<T1, T2, IS_BF16>::ReCompute(
    const uint64_t& batchSqLoopOffset, const uint64_t& batchSkvLoopOffset,
    const uint64_t& attenMaskOffset, const uint64_t& dropMaskOffset,
    const uint64_t& batchReduceOffset, const uint64_t& batchPseOffset)
{
    LocalTensor <T2> subResInner = vecClc1.Get<T2>(innerMatResNum);
    LocalTensor <T2> &mulResInner = subResInner;
    LocalTensor <T2> dvDropResInner = vecClc2.Get<T2>(innerMatResNum);

    LocalTensor <T2> &reMatmulResInner = dvDropResInner;
    LocalTensor <T2> &softmaxResInner = dvDropResInner;
    LocalTensor <T2> &attenMaskResInner = dvDropResInner;

    bool clcAttenMask = false;
    bool clsPse = CopyInPse(batchPseOffset);
    if (!clsPse) {
        clcAttenMask = CopyInAttenMask(attenMaskOffset);
    }

    mm2.template GetTensorC<false>(reMatmulResInner, false, true);
    mm2.End();

    if (clsPse) {
        ClcPse(reMatmulResInner);
        clcAttenMask = CopyInAttenMask(attenMaskOffset);
    }
    LocalTensor<float> maxInner = vecInQue1.AllocTensor<float>();
    LocalTensor<float> sumInner = vecInQue1.AllocTensor<float>();
    maxInner.SetSize(innerReduceNum);
    sumInner.SetSize(innerReduceNum);
    SetReClcShape(mulResInner, maxInner, sumInner, dvDropResInner);
    CopyInSoftMax(maxInner, softmaxMaxGm[batchReduceOffset], sumInner, softmaxSumGm[batchReduceOffset]);
    Muls(reMatmulResInner, reMatmulResInner, (T2) scaleValue, innerMatResNumVec);

    if (clcAttenMask) {
        ClcAttenMask(reMatmulResInner);
    }
    uint32_t tempInnerMatOutShape[2];
    tempInnerMatOutShape[0] = nIn * g * sQ;
    tempInnerMatOutShape[1] = sKVAlignVec;
    dvDropResInner.SetShapeInfo(ShapeInfo(2, tempInnerMatOutShape, DataFormat::ND));
    ClcSoftMax(softmaxResInner, attenMaskResInner, maxInner, sumInner);
    mulResInner.SetShapeInfo(ShapeInfo(2, tempInnerMatOutShape, DataFormat::ND));
    Mul(mulResInner, softmaxResInner, subResInner, innerMatResNumVec);
    if (isDrop) {
        LocalTensor<uint8_t> dpMask = vecInQue2.AllocTensor<uint8_t>();
        /* 当s和D不大的时候，这一步搬运可以省掉。 */
        CopyInDrop(dpMask, dropoutWorkspaceGm[dropMaskOffset]);
        ClcDrop(softmaxResInner, softmaxResInner, dpMask);
        vecInQue2.FreeTensor(dpMask);
    }
    vecInQue1.FreeTensor(maxInner);
    vecInQue1.FreeTensor(sumInner);
    ClcMatmul(batchSqLoopOffset, batchSkvLoopOffset, mulResInner, dvDropResInner);
}

// T1 INPUT_T, T2 CALC_T
template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradUngs1s2Bbn<T1, T2, IS_BF16>::Process()
{
    if (g_coreType == AIV && mBlockIdx >= usedCoreNum) {
        auto workLocal = vecInQue2.AllocTensor<int32_t>();
        SyncAll<false>(syncGlobal, workLocal);
        vecInQue2.FreeTensor(workLocal);
        return;
    }
    isDrop = false;
    if (keepProb < 1 && dropoutWorkspaceLen > 0) {
        isDrop = true;
    }

    uint64_t batchOffset = mBlockIdx * singleCoreBatchRange;
    uint64_t currentBatchRange =
        singleCoreBatchRange < (totalBatch - batchOffset) ? singleCoreBatchRange : (totalBatch - batchOffset);
    uint64_t matResNum = nIn * g * sQ * sKVAlignVec;
    mm1.SetWorkspace((__gm__ uint8_t *) workspaceGm[(syncLen + dropoutWorkspaceLen) / sizeof(T2) +
                                                    batchOffset * matResNum].GetPhyAddr(),
                                                    matResNum * vecCalcDTypeSize);
    mm2.SetWorkspace(
        (__gm__ uint8_t *) workspaceGm[(syncLen + mm1WorkspaceLen + dropoutWorkspaceLen) / sizeof(T2) +
                                       batchOffset * matResNum].GetPhyAddr(),
                                       matResNum * vecCalcDTypeSize);
    for (uint64_t batchIdx = 0; batchIdx < currentBatchRange; ++batchIdx) {
        uint64_t totalBatch = batchOffset + batchIdx;
        uint64_t bIdx = totalBatch / nOut;
        uint64_t nIdx = totalBatch % nOut;
        uint64_t batchSqLoopOffset = 0;
        uint64_t batchSkvLoopOffset = 0;

        uint64_t previousBatchCnt = (bIdx * nOut * nIn + nIdx * nIn) * g;
        uint64_t dropMaskOffset = previousBatchCnt * sQ * sKV;
        uint64_t attenMaskOffset = 0;
        if (hasAttenMask == 1) {
            uint64_t compressMode = ordTilingData_->opInfo.attenMaskCompressMode;
            // compressMode == 0: no atten_mask compress
            if (compressMode == 1) {
                CalcCausalAttenMaskOffset(attenMaskOffset, 0);
            } else if (compressMode == 2) {
                CalcCausalAttenMaskOffset(attenMaskOffset, sKV - sQ);
            } else if (attenMaskShapeType == 0) { // SS
                attenMaskOffset = 0;
            } else if (attenMaskShapeType == 1) { // B1SS
                attenMaskOffset = bIdx * sQ * sKV;
            } else {  // BNSS
                attenMaskOffset = bIdx * nIdx * sQ * sKV;
            }
        }
        uint64_t batchSoftmaxInputOffset = previousBatchCnt * sQ * 8;
        uint64_t batchPseOffset = previousBatchCnt * pseSq * sKV;
        if (inputLayout == 0) {  // BSH ->BSND
            batchSqLoopOffset = bIdx * sQ * nOut * nIn * g * d + nIdx * nIn * g * d;
            batchSkvLoopOffset = bIdx * sKV * nOut * nIn * d + nIdx * nIn * d;
        } else if (inputLayout == 1) {  // SBH即SBND
            batchSqLoopOffset = bIdx * nOut * nIn * g * d + nIdx * nIn * g * d;
            batchSkvLoopOffset = bIdx * nOut * nIn * d + nIdx * nIn * d;
        } else if (inputLayout == 2) {  // BNSD
            batchSqLoopOffset = bIdx * nOut * nIn * g * sQ * d + nIdx * nIn * g * sQ * d;
            batchSkvLoopOffset = bIdx * nOut * nIn * sKV * d + nIdx * nIn * sKV * d;
        } else {
            return;
        }

        nIn = nIn < (n - nIdx * nIn) ? nIn : (n - nIdx * nIn);
        auto nInGSq = nIn * g * sQ;
        innerMatResNum = nInGSq * sKVAlign;
        innerMatResNumVec = nInGSq * sKVAlignVec;
        maskInputNum = nInGSq * sKVAlignByte;
        innerReduceNum = nInGSq * 8;

        softmaxGradInputShape[0] = nIn * g * sQ;

        innerMatOutShape[0] = nInGSq;
        innerMatOutShape[1] = sKVAlign;
        innerReduceShape[0] = nInGSq;
        innerReduceShape[1] = 8;

        mm1.SetTail(sQ, sKV, d);
        mm1.SetTensorA(dxGm[batchSqLoopOffset]);
        mm1.SetTensorB(valueGm[batchSkvLoopOffset], true);
        mm1.template Iterate<false>();

        mm2.SetTail(sQ, sKV, d);
        mm2.SetTensorA(queryGm[batchSqLoopOffset]);
        mm2.SetTensorB(keyGm[batchSkvLoopOffset], true);
        mm2.template Iterate<false>();

        FrontCompute(batchSqLoopOffset, batchSkvLoopOffset, dropMaskOffset);
        ReCompute(batchSqLoopOffset, batchSkvLoopOffset, attenMaskOffset, dropMaskOffset,
                  batchSoftmaxInputOffset, batchPseOffset);
    }
    auto workLocal = vecInQue2.AllocTensor<int32_t>();
    SyncAll<false>(syncGlobal, workLocal);
    vecInQue2.FreeTensor(workLocal);
}
#endif
