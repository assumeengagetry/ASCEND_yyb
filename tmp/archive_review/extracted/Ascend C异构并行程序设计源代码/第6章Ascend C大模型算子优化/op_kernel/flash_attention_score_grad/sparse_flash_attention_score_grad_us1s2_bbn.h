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

#ifndef _SPARSE_FLASH_ATTENTION_SCORE_GRAD_US1S2_BBN_H_
#define _SPARSE_FLASH_ATTENTION_SCORE_GRAD_US1S2_BBN_H_

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

using namespace matmul;

/**
 * UB切分：以基本块的形式，切S1，S2
 * Block切分：BN足够大，BN开多核。
 * N等长，N1=N2，G=1
 * 特化模板特点：
     1、由于切分s1，s2，进行BMM3.1/3.2/4之前拿到的输入都是中间结果，需要开workspace，缓存中间结果，最终进行一次BMM3.1/3.2/4操作。
     2、BMM3.1/3.2/4 的结果通过BMM api的随路cast能力，完成输出fp16/fp32/bf16。
     3、workspace的申请比较大，需要申请2块coreNum*S1*S2的大小，分别存DropoutDoMask及Mul的结果。
     4、N等长，G=1，dv/dq/dk不需要随路reduce
*/
constexpr uint32_t BLOCK_BYTE = 32; // 32B one block
constexpr uint32_t REPEAT_BLOCK_NUMS = 8;
constexpr uint32_t REPEAT_BLOCK_BYTE = 256;
constexpr uint32_t SOFTMAX_LAST_AXIS_SIZE = 8;
constexpr uint32_t MAX_STRIDE = 65535;
constexpr MatmulConfig normalConfig = GetNormalConfig(true);

enum class LAYOUT {
    BSH = 0,
    SBH,
    BNSD,
    BSND
};

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
class SparseFlashAttentionScoreGradUS1S2BBN {
public:
    __aicore__ inline SparseFlashAttentionScoreGradUS1S2BBN(){};
    __aicore__ inline void Init(
        __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* dy,
        __gm__ uint8_t* pse_shift, __gm__ uint8_t* drop_mask, __gm__ uint8_t* padding_mask, __gm__ uint8_t* atten_mask,
        __gm__ uint8_t* softmax_max, __gm__ uint8_t* softmax_sum, __gm__ uint8_t* softmax_in,
        __gm__ uint8_t* attention_in, __gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv, __gm__ uint8_t* dpse,
        __gm__ uint8_t* workspace, const FlashAttentionScoreGradUs1s2BbnTilingData* __restrict tilingData, TPipe *pipe_in,
        uint64_t *bmm31Offset, uint64_t *bmm32Offset, uint64_t *bmm4Offset);
    __aicore__ inline void Process();
    __aicore__ inline void InitCalcParams();
    __aicore__ inline void ClearSparseBlockGm();
    __aicore__ inline void CalcSparseIdx(const uint32_t bIdx, const uint32_t s1Idx,
                                         const uint32_t s2Start, const uint32_t s2End,
                                         const uint32_t s1BaseSize, const uint32_t s2BaseSize,
                                         uint32_t &s2LoopStartIdx, uint32_t &s2LoopEndIdx);

    __aicore__ inline void CalcForwardBMMS1Offset(const uint32_t s1IdxC);
    __aicore__ inline void CalcForwardBmmEmitParams(const uint32_t bIdx, const uint32_t nextS1IdxC, uint64_t& sparseS2Offset, uint32_t& curS2);
    __aicore__ inline void CalcBNOffset(const uint32_t bnIdx);
    __aicore__ inline void CalcAttenMaskOffset(uint64_t& attenMaskOffset, const int64_t delta);
    __aicore__ inline void CalcCausalAttenMaskOffset(const int64_t delta);
    __aicore__ inline void CalcS1OffsetLoopParams(const uint32_t s1Idx);
    __aicore__ inline void CalcS2OffsetLoopParams(const uint32_t s1Idx, const uint32_t s2Idx);

    __aicore__ inline bool CopyInPse();
    __aicore__ inline bool CopyInAttenMask();
    __aicore__ inline void CopyInSoftmaxMax();
    __aicore__ inline void CopyInSoftmaxSum();
    __aicore__ inline bool CopyInDropMask(LocalTensor<uint8_t>& dropMaskUb);
    __aicore__ inline void CopyInDyAttenIn(TQue<QuePosition::VECIN, 1>& vecInQue, const GlobalTensor<INPUT_T>& gm);
    __aicore__ inline void CopyInDyAttenInWithSplitD(LocalTensor<INPUT_T>& dyUb, LocalTensor<INPUT_T>& attenInUb,
                                                     uint32_t gmOffset, uint32_t rawSize);
    __aicore__ inline void CopyOutWorkspace(LocalTensor<INPUT_T>& vecOutUb, uint64_t workspaceOffset);
    __aicore__ inline void CopyOutWorkspaceWithCast(LocalTensor<CALC_T>& vecOutUb, uint64_t offset);

    __aicore__ inline void CalcAdd(LocalTensor<CALC_T>& bmm2ResUb, LocalTensor<CALC_T>& pseUb);
    __aicore__ inline void CalcPseAdd(LocalTensor<CALC_T>& bmm2ResUb);
    __aicore__ inline void CalcU8Mask(LocalTensor<CALC_T>& bmm2ResUb);
    __aicore__ inline void CalcMask(LocalTensor<CALC_T>& bmm2ResUb);
    __aicore__ inline void CalcSoftmax(LocalTensor<CALC_T>& bmm2ResUb);
    __aicore__ inline void CalcDropout(LocalTensor<CALC_T>& dst, LocalTensor<CALC_T>& src, LocalTensor<uint8_t>& dropMaskUb);
    __aicore__ inline void CalcSoftmaxGradWithSplitD(LocalTensor<CALC_T>& softmaxGradResUb);
    __aicore__ inline void CalcSoftmaxGrad(LocalTensor<CALC_T>& softmaxGradResUb, uint32_t dInnerSize, uint32_t dInnerAlignSize);
    __aicore__ inline void CalcSub(LocalTensor<CALC_T>& dst, LocalTensor<CALC_T>& src, LocalTensor<CALC_T>& softmaxGradResUb);
    __aicore__ inline void CalcMul(LocalTensor<CALC_T>& dst, LocalTensor<CALC_T>& src0, LocalTensor<CALC_T>& src1, uint64_t calcSize);

    __aicore__ inline void forwardBmmEmit(const uint32_t s2LoopStartIdxV, const uint32_t s2LoopEndIdxV);
    __aicore__ inline void forwardBmm2Emit(const uint32_t s1IdxC, const uint64_t sparseS2Offset, const uint32_t curS2);
    __aicore__ inline void forwardBmm1Emit(const uint32_t s1IdxC, const uint64_t sparseS2Offset, const uint32_t curS2);
    __aicore__ inline void SimpleSoftmaxComputeStage(LocalTensor<CALC_T>& bmm2ResUb, LocalTensor<uint8_t>& dropMaskUb);
    __aicore__ inline void SoftmaxGradComputeStage(LocalTensor<CALC_T> &softmaxGradResUb, LocalTensor<CALC_T>& bmm2ResUb,
                                                   LocalTensor<uint8_t>& dropMaskUb);

    __aicore__ inline void LastBmmStage();
    __aicore__ inline void LastBmmAsyncStage(uint32_t bnIdx);

    using biasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    // bmm1 bmm2
    using aType1 = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false>;
    using bType1 = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, true>;
    using cType1 = MatmulType<TPosition::VECCALC, CubeFormat::ND_ALIGN, CALC_T>;
    // bmm31
    using aType2 = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false>;
    using bType2 = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false>;
    using cType2 = MatmulType<TPosition::GM, CubeFormat::ND, CALC_T>;
    // bmm32 bmm4
    using aType3 = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, true>;
    using bType3 = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false>;
    using cType3 = MatmulType<TPosition::GM, CubeFormat::ND, CALC_T>;

    Matmul<aType1, bType1, cType1, biasType, normalConfig> bmm1;
    Matmul<aType1, bType1, cType1, biasType, normalConfig> bmm2;
    Matmul<aType2, bType2, cType2, biasType, normalConfig> bmm3;
    // bmm32和bmm4复用同一个bmm4对象，担心bmm对象过多，对性能影响非常大
    Matmul<aType3, bType3, cType3, biasType, normalConfig> bmm4;

protected:
    const FlashAttentionScoreGradUs1s2BbnTilingData* __restrict ordTilingData;
    TPipe *pipe;

    TQue<QuePosition::VECIN, 1> vecInQue1;
    TQue<QuePosition::VECIN, 1> vecInQue2;
    TQue<QuePosition::VECOUT, 1> vecOutQue;
    TQue<QuePosition::VECIN, 1> softmaxQue1;
    TQue<QuePosition::VECIN, 1> softmaxQue2;
    TQue<QuePosition::VECIN, 1> bmm1ResQue;
    TQue<QuePosition::VECIN, 1> bmm2ResQue;
    TBuf<> castBuff;
    TBuf<> softmaxGradResBuf;
    GlobalTensor<uint8_t> attenMaskU8Gm;
    GlobalTensor<INPUT_T> attenMaskGm;
    GlobalTensor<INPUT_T> queryGm, keyGm, valueGm, dyGm, pseGm, attenInGm, paddingMaskGm;
    GlobalTensor<INPUT_T> dqGm, dkGm, dvGm;
    GlobalTensor<float> softmaxMaxGm, softmaxSumGm;
    GlobalTensor<uint8_t> dropMaskGm;
    GlobalTensor<INPUT_T> workspaceGm;
    GlobalTensor<int32_t> syncGlobal;
    GlobalTensor<uint8_t> dropoutWorkspaceGm;
    GlobalTensor<CALC_T> postOpWorkspaceDqGm;
    GlobalTensor<CALC_T> postOpWorkspaceDvGm;
    GlobalTensor<CALC_T> postOpWorkspaceDkGm;
    CALC_T mulsCalcValue = -10000.0;

    uint64_t mBlockIdx;
    bool doSoftmaxGrad = true;
    bool isDrop = false;
    bool hasPse = false;
    LAYOUT layOutType;
    bool notSplitGradD = true;

    uint32_t inputTypeNum;
    uint32_t blockInputTypeNum;
    uint32_t blockCalcTypeNum;
    uint32_t usedCoreNum;
    uint32_t physicalCoreNum;
    uint32_t s2AlignSize;
    uint64_t spmBufOffset;
    uint64_t spmWorkspaceSize;
    uint64_t singleCoreSpmSize;
    uint32_t syncDropoutLen;
    uint32_t dropoutWorkspaceLen;
    uint32_t blockSplitBnRange;
    uint32_t normalCoreNums;
    uint32_t s1RangeV;
    uint32_t s2RangeV;
    uint32_t s1InnerSize;
    uint32_t s2InnerSize;
    uint32_t tailS1InnerSize;
    uint32_t tailS2InnerSize;
    uint32_t s1RangeC;
    uint32_t s2RangeC;
    uint32_t cubeBaseS1;
    uint32_t cubeBaseS2;
    uint32_t tailCubeBaseS1;
    uint32_t tailCubeBaseS2;
    uint32_t s1CVRatio;
    uint32_t s2CVRatio;

    uint32_t softmaxSize;
    uint32_t mmPadSrcGap = 0;
    uint32_t mmPadDstGap = 0;
    uint32_t hasPseInput;
    uint32_t hasAttenMaskInput;
    uint32_t attenMaskDimS2;

    // offset
    uint64_t beforeBatchReduceNums;
    uint64_t beforeBlockSplitBnNums;
    uint64_t bnLoopS1Offset;
    uint64_t bnLoopS2Offset;
    uint64_t bnLoopSoftmaxOffset;
    uint64_t bnLoopPseOffset;
    uint64_t bnLoopMaskOffset;
    uint64_t bnLoopDropMaskOffset;
    uint64_t s1LoopOffset;
    uint64_t s2LoopOffset;
    uint64_t pseOffset;
    uint64_t attenMaskOffset;
    uint64_t s1LoopSoftmaxOffset;
    uint64_t dropMaskOffset;
    uint64_t copyOut1Offset;
    uint64_t copyOut2Offset;
    uint64_t forwardBmmS1LoopOffset;

    uint32_t actualS1InnerSize;
    uint32_t actualS2InnerSize;
    uint32_t actualS1InnerAlignSize;
    uint32_t actualS2InnerAlignSize;
    uint32_t actualDAlignSize;
    uint32_t actualCubeBaseS1;
    uint64_t actualS1S2InnerSize;
    uint64_t actualS1S2AlignInnerSize;
    uint64_t actualS1DSize;
    uint64_t actualS1DAlignSize;
    uint32_t s1S2ShapeArray[2];
    uint32_t s1S2ShapeAlignArray[2];
};

/**
输入：
    query:       (B, N, S1, D)
    key:         (B, N, S2, D)
    value:       (B, N, S2, D)
    dy:          (B, N, S1, D)
    atten_in:    (B, N, S1, D)
    drop_mask:   (B, N, S1, S2)
    softmax_max: (B, N, S1, 8)
    pse:         (B, N, S1, S2) / (B, N, 1, S2)
    atten_mask:  (B, 1, S1, S2) / (1, 1, S1, S2)
输出：
    dq:          (B, N, S1, D)
    dk:          (B, N, S2, D)
    dv:          (B, N, S2, D)
    dpse:        (B, N, S1, S2) / (B, N, 1, S2)
*/
#if __CCE_KT_TEST__
template<typename T>
inline void Print(const GlobalTensor<T> global, uint32_t len)
{
    std::ostringstream os_;
    os_.str("");
    uint32_t printLen = len;
    uint32_t blockNum = ONE_BLK_SIZE / sizeof(T);
    uint32_t rowNum = printLen / blockNum;
    uint32_t residualNum = printLen % blockNum;
    const int32_t width = 4;
    for (uint32_t i = 0; i < rowNum; i++) {
        os_ << std::setw(width) << std::setfill('0') << i * blockNum << " : ";
        for (uint32_t j = 0; j < blockNum; j++) {
            if constexpr (IsSameType<T, float>::value) {
              os_ << global.GetValue(i * blockNum + j) << " ";
            } else {
              os_ << global.GetValue(i * blockNum + j).ToFloat() << " ";
            }
        }
        os_ << std::endl;
    }
    if (residualNum != 0) {
        os_ << std::setw(width) << std::setfill('0') << rowNum * blockNum << " : ";
        for (uint32_t i = 0; i < residualNum; i++) {
            if constexpr (IsSameType<T, float>::value) {
              os_ << global.GetValue(rowNum * blockNum + i) << " ";
            } else {
              os_ << global.GetValue(rowNum * blockNum + i).ToFloat() << " ";
            }
        }
        os_ << std::endl;
    }
    std::cout << os_.str();
}

template <typename T1>
inline void Print(const LocalTensor<T1> global, uint32_t len1, uint32_t printLen)
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
#endif  // __CCE_KT_TEST__

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline bool SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::CopyInPse() {
    if (hasPseInput == 0) {
        return false;
    }

    DataCopyParams copyParams;
    copyParams.blockCount = actualS1InnerSize;
    if (ordTilingData->baseParams.pseShapeType == 1) {  // BN1S
        copyParams.blockCount = 1;
    }
    copyParams.blockLen = actualS2InnerSize * inputTypeNum;
    copyParams.srcStride = (ordTilingData->baseParams.Skv - actualS2InnerSize) * inputTypeNum;
    copyParams.dstStride = 0;

    DataCopyPadParams padParams;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.paddingValue = 0;
    padParams.rightPadding = actualS2InnerAlignSize - actualS2InnerSize;

    LocalTensor<INPUT_T> pseUb = vecInQue1.template AllocTensor<INPUT_T>();
    pseUb.SetSize(copyParams.blockCount * actualS2InnerAlignSize);
    uint32_t shapeArray[] = {copyParams.blockCount, actualS2InnerAlignSize};
    pseUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    DataCopyPad(pseUb, pseGm[pseOffset], copyParams, padParams);
    vecInQue1.EnQue(pseUb);

    return true;
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline bool SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::CopyInAttenMask() {
    if (hasAttenMaskInput == 0) {
        return false;
    }

    DataCopyParams copyParams;
    copyParams.blockCount = actualS1InnerSize;
    copyParams.blockLen = actualS2InnerSize * inputTypeNum;
    copyParams.srcStride = (attenMaskDimS2 - actualS2InnerSize) * inputTypeNum;
    copyParams.dstStride = 0;

    DataCopyPadParams padParams;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.paddingValue = 0;
    padParams.rightPadding = actualS2InnerAlignSize - actualS2InnerSize;

    if (ordTilingData->baseParams.attenMaskType == 0) {  // 输入数据类型与 qkv 一致
        LocalTensor<INPUT_T> attenMaskUb = vecInQue2.template AllocTensor<INPUT_T>();
        DataCopyPad(attenMaskUb, attenMaskGm[attenMaskOffset], copyParams, padParams);
        vecInQue2.EnQue(attenMaskUb);
    } else {  // bool输入
        copyParams.blockLen = actualS2InnerSize * sizeof(uint8_t);
        copyParams.srcStride = (attenMaskDimS2 - actualS2InnerSize) * sizeof(uint8_t);

        LocalTensor<uint8_t> attenMaskUb = vecInQue2.template AllocTensor<uint8_t>();
        DataCopyPad(attenMaskUb, attenMaskU8Gm[attenMaskOffset], copyParams, padParams);
        vecInQue2.EnQue(attenMaskUb);
    }

    return true;
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::CopyInSoftmaxMax() {
    LocalTensor<float> softmaxMaxUb = softmaxQue1.template AllocTensor<float>();
    DataCopy(softmaxMaxUb, softmaxMaxGm[s1LoopSoftmaxOffset], softmaxSize);
    softmaxQue1.EnQue(softmaxMaxUb);
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::CopyInSoftmaxSum() {
    LocalTensor<float> softmaxSumUb = softmaxQue2.template AllocTensor<float>();
    DataCopy(softmaxSumUb, softmaxSumGm[s1LoopSoftmaxOffset], softmaxSize);
    softmaxQue2.EnQue(softmaxSumUb);
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline bool SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::CopyInDropMask(
    LocalTensor<uint8_t>& dropMaskUb) {
    if (dropMaskGm.GetPhyAddr() == nullptr) {
        return false;
    }
    DataCopyParams copyParams;
    copyParams.blockCount = actualS1InnerSize;
    copyParams.blockLen = actualS2InnerSize;
    copyParams.srcStride = ordTilingData->baseParams.Skv - copyParams.blockLen;
    copyParams.dstStride = 0;

    DataCopyPadParams padParams;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = (actualS2InnerSize + BLOCK_BYTE - 1) / BLOCK_BYTE * BLOCK_BYTE - actualS2InnerSize;
    padParams.paddingValue = 0;
    DataCopyPad(dropMaskUb, dropoutWorkspaceGm[dropMaskOffset], copyParams, padParams);
    vecInQue1.EnQue(dropMaskUb);

    return true;
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::CopyInDyAttenIn(
    TQue<QuePosition::VECIN, 1>& vecInQue, const GlobalTensor<INPUT_T>& gm) {
    DataCopyParams copyParams;
    copyParams.blockCount = actualS1InnerSize;
    copyParams.blockLen = ordTilingData->baseParams.D * inputTypeNum;
    copyParams.dstStride = 0;

    uint32_t srcStride = 0;
    uint32_t loopOffset = 0;
    switch (layOutType) {
        case LAYOUT::BSH:
        case LAYOUT::BSND:
            loopOffset = ordTilingData->baseParams.H;
            srcStride = (ordTilingData->baseParams.H - ordTilingData->baseParams.D) * inputTypeNum;  // 两块之间的gap
            break;
        case LAYOUT::SBH:
            loopOffset = ordTilingData->baseParams.B * ordTilingData->baseParams.H;
            srcStride = (ordTilingData->baseParams.B * ordTilingData->baseParams.H - ordTilingData->baseParams.D) *
                        inputTypeNum;
            break;
        case LAYOUT::BNSD:
            loopOffset = ordTilingData->baseParams.D;
            srcStride = 0;
            break;
    }

    LocalTensor<INPUT_T> localUb = vecInQue.template AllocTensor<INPUT_T>();
    if (ordTilingData->baseParams.D % blockInputTypeNum == 0) {
        copyParams.srcStride = srcStride / BLOCK_BYTE; // B 转 block，block间地址步长
        copyParams.blockLen = copyParams.blockLen / BLOCK_BYTE;
        if (copyParams.srcStride <= MAX_STRIDE) {
            DataCopy(localUb, gm[s1LoopOffset], copyParams);
        } else {
            copyParams.blockCount = 1;
            copyParams.srcStride = 0;
            uint64_t gmOffset = s1LoopOffset;
            for (uint32_t i = 0; i < actualS1InnerSize; ++i) {
                DataCopy(localUb[i * actualDAlignSize], gm[gmOffset], copyParams);
                gmOffset = gmOffset + loopOffset;
            }
        }
    } else {
        DataCopyPadParams padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.paddingValue = 0;
        padParams.rightPadding = actualDAlignSize - ordTilingData->baseParams.D;
        if (srcStride <= MAX_STRIDE) {
            copyParams.srcStride = srcStride;
            DataCopyPad(localUb, gm[s1LoopOffset], copyParams, padParams);
        } else {
            copyParams.blockCount = 1;
            copyParams.srcStride = 0;
            uint64_t gmOffset = s1LoopOffset;
            for (uint32_t i = 0; i < actualS1InnerSize; ++i) {
                DataCopyPad(localUb[i * actualDAlignSize], gm[gmOffset], copyParams, padParams);
                gmOffset = gmOffset + loopOffset;
            }
        }
    }

    vecInQue.EnQue(localUb);
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::CopyInDyAttenInWithSplitD(
    LocalTensor<INPUT_T>& dyUb, LocalTensor<INPUT_T>& attenInUb, uint32_t gmOffset, uint32_t rawSize) {
    uint32_t rawSizeAlign = (rawSize + blockInputTypeNum - 1) / blockInputTypeNum * blockInputTypeNum;
    uint32_t srcStride = 0;
    uint32_t loopOffset = 0;
    switch (layOutType) {
        case LAYOUT::BSH:
        case LAYOUT::BSND:
            loopOffset = ordTilingData->baseParams.H;
            srcStride = (ordTilingData->baseParams.H - rawSize) * inputTypeNum;
            break;
        case LAYOUT::SBH:
            loopOffset = ordTilingData->baseParams.B * ordTilingData->baseParams.H;
            srcStride = (ordTilingData->baseParams.B * ordTilingData->baseParams.H - rawSize) * inputTypeNum;
            break;
        case LAYOUT::BNSD:
            loopOffset = ordTilingData->baseParams.D;
            srcStride = (ordTilingData->baseParams.D - rawSize) * inputTypeNum;
            break;
    }

    DataCopyPadParams padParams;
    DataCopyParams intriParams;
    intriParams.dstStride = 0;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.paddingValue = 0;

    if (srcStride <= MAX_STRIDE) {
        intriParams.blockCount = actualS1InnerSize;
        intriParams.blockLen = rawSize * inputTypeNum;
        intriParams.srcStride = srcStride;
        padParams.rightPadding = rawSizeAlign - rawSize;
        DataCopyPad(dyUb, dyGm[gmOffset], intriParams, padParams);
        DataCopyPad(attenInUb, attenInGm[gmOffset], intriParams, padParams);
    } else {
        intriParams.blockCount = 1;
        intriParams.blockLen = rawSize * inputTypeNum;
        intriParams.srcStride = 0;
        padParams.rightPadding = rawSizeAlign - rawSize;
        for (uint32_t i = 0; i < actualS1InnerSize; i++) {
            DataCopyPad(dyUb[i * rawSizeAlign], dyGm[gmOffset], intriParams, padParams);
            DataCopyPad(attenInUb[i * rawSizeAlign], attenInGm[gmOffset], intriParams, padParams);
            gmOffset = gmOffset + loopOffset;
      }
    }
    vecInQue1.EnQue(attenInUb);
    vecInQue2.EnQue(dyUb);
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::CopyOutWorkspace(
    LocalTensor<INPUT_T>& vecOutUb, uint64_t offset) {
    // 这里的实现方案是，带pad搬到workspace，并参与Matmul计算，最终搬出时，搬出实际shape。由于UB都是带pad计算的，因此这里不用特殊处理pad
    DataCopyParams copyParams;
    copyParams.blockCount = actualS1InnerSize;
    copyParams.blockLen = actualS2InnerAlignSize * sizeof(INPUT_T) / BLOCK_BYTE;
    copyParams.srcStride = 0;
    copyParams.dstStride = (s2AlignSize - actualS2InnerAlignSize) * sizeof(INPUT_T) / BLOCK_BYTE;
    DataCopy(workspaceGm[offset], vecOutUb, copyParams);
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::CopyOutWorkspaceWithCast(
    LocalTensor<CALC_T>& src, uint64_t offset) {
    LocalTensor<INPUT_T> vecOutUb1 = vecOutQue.template AllocTensor<INPUT_T>();
    vecOutUb1.SetSize(actualS1S2AlignInnerSize);
    vecOutUb1.SetShapeInfo(ShapeInfo(2, s1S2ShapeAlignArray, 2, s1S2ShapeArray, DataFormat::ND));
    Cast(vecOutUb1, src, RoundMode::CAST_ROUND, actualS1S2AlignInnerSize);

    vecOutQue.EnQue(vecOutUb1);
    vecOutUb1 = vecOutQue.template DeQue<INPUT_T>();

    DataCopyParams copyParams;
    copyParams.blockCount = actualS1InnerSize;
    copyParams.blockLen = actualS2InnerAlignSize * sizeof(INPUT_T) / BLOCK_BYTE;
    copyParams.srcStride = 0;
    copyParams.dstStride = (s2AlignSize - actualS2InnerAlignSize) * sizeof(INPUT_T) / BLOCK_BYTE;
    DataCopy(workspaceGm[offset], vecOutUb1, copyParams);
    vecOutQue.FreeTensor(vecOutUb1);
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::CalcAdd(
    LocalTensor<CALC_T>& bmm2ResUb, LocalTensor<CALC_T>& pseUb) {
    if (ordTilingData->baseParams.pseShapeType == 0) {  // BNSS
        Add(bmm2ResUb, bmm2ResUb, pseUb, actualS1S2AlignInnerSize);
    } else {  // BN1S
        uint32_t stepCalcNum = REPEAT_BLOCK_BYTE / sizeof(CALC_T);
        /* Total data number of single step should be smaller than 256bytes.
         * If larger, we need to do add multiple times. */
        stepCalcNum = actualS2InnerAlignSize > stepCalcNum ? stepCalcNum : actualS2InnerAlignSize;
        uint32_t innerLoop = actualS2InnerAlignSize / stepCalcNum;
        uint32_t innerRemain = actualS2InnerAlignSize % stepCalcNum;

        BinaryRepeatParams repParams;
        repParams.src0BlkStride = 1;
        repParams.src0RepStride = actualS2InnerAlignSize / blockCalcTypeNum;
        repParams.src1BlkStride = 1;
        repParams.src1RepStride = 0;
        repParams.dstRepStride = repParams.src0RepStride;
        repParams.blockNumber = repParams.src0RepStride;
        // S1方向小于256，不用起for循环
        for (uint32_t loop = 0; loop < innerLoop; loop++) {
            uint32_t offset = loop * stepCalcNum;
            Add(bmm2ResUb[offset], bmm2ResUb[offset], pseUb[offset], stepCalcNum,
                actualS1InnerSize, repParams);
        }
        if (innerRemain > 0) {
            uint32_t offset = innerLoop * stepCalcNum;
            Add(bmm2ResUb[offset], bmm2ResUb[offset], pseUb[offset], innerRemain,
                actualS1InnerSize, repParams);
        }
    }
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::CalcPseAdd(
    LocalTensor<CALC_T>& bmm2ResUb) {
    // pse
    if (hasPse) {
        LocalTensor<INPUT_T> pseUb = vecInQue1.template DeQue<INPUT_T>();
        if constexpr (HIGH_PRECISE) {
            LocalTensor<CALC_T> castedPseUb = vecOutQue.template AllocTensor<CALC_T>();
            if (ordTilingData->baseParams.pseShapeType == 1) {  // BN1S
                s1S2ShapeAlignArray[0] = 1;
            }
            castedPseUb.SetSize(actualS1S2AlignInnerSize);
            castedPseUb.SetShapeInfo(ShapeInfo(2, s1S2ShapeAlignArray, DataFormat::ND));
            Cast(castedPseUb, pseUb, RoundMode::CAST_NONE, actualS1S2AlignInnerSize);
            pipe_barrier(PIPE_V);
            CalcAdd(bmm2ResUb, castedPseUb);
            vecOutQue.FreeTensor(castedPseUb);
      } else {
          CalcAdd(bmm2ResUb, pseUb);
      }
      vecInQue1.FreeTensor(pseUb);
      pipe_barrier(PIPE_V);
    }
    Muls(bmm2ResUb, bmm2ResUb, (CALC_T)ordTilingData->baseParams.scaleValue, actualS1S2AlignInnerSize);
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::CalcU8Mask(
    LocalTensor<CALC_T>& bmm2ResUb) {
    LocalTensor<uint8_t> attenMaskUb = vecInQue2.template DeQue<uint8_t>();
    LocalTensor<uint8_t> tmpUbBuffer = vecOutQue.AllocTensor<uint8_t>();

    attenMaskUb.SetSize(actualS1S2AlignInnerSize);
    attenMaskUb.SetShapeInfo(ShapeInfo(2, s1S2ShapeAlignArray, DataFormat::ND));

    CALC_T scalar;
    if constexpr (IsSameType<CALC_T, float>::value) {
        uint32_t tmp = 0xFF7FFFFF;
        scalar = *((float*)&tmp);
    } else {
        uint16_t tmp = 0xFBFF;
        scalar = *((half*)&tmp);
    }
    SelectWithBytesMaskShapeInfo info;
    info.firstAxis = actualS1InnerSize;
    info.srcLastAxis = actualS2InnerAlignSize;
    info.maskLastAxis = (actualS2InnerAlignSize + BLOCK_BYTE - 1) / BLOCK_BYTE * BLOCK_BYTE;
    SelectWithBytesMask(bmm2ResUb, bmm2ResUb, scalar, attenMaskUb, tmpUbBuffer, info);
    vecInQue2.FreeTensor(attenMaskUb);
    vecOutQue.FreeTensor(tmpUbBuffer);
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::CalcMask(
    LocalTensor<CALC_T>& bmm2ResUb) {
    LocalTensor<INPUT_T> attenMaskUb = vecInQue2.template DeQue<INPUT_T>();

    attenMaskUb.SetSize(actualS1S2AlignInnerSize);
    attenMaskUb.SetShapeInfo(ShapeInfo(2, s1S2ShapeAlignArray, DataFormat::ND));

    if constexpr (HIGH_PRECISE) {
        LocalTensor<CALC_T> castedAttenMaskUb = vecOutQue.template AllocTensor<CALC_T>();
        castedAttenMaskUb.SetSize(actualS1S2AlignInnerSize);
        castedAttenMaskUb.SetShapeInfo(ShapeInfo(2, s1S2ShapeAlignArray, DataFormat::ND));
        Cast(castedAttenMaskUb, attenMaskUb, RoundMode::CAST_NONE, actualS1S2AlignInnerSize);
        pipe_barrier(PIPE_V);
        Muls(castedAttenMaskUb, castedAttenMaskUb, mulsCalcValue, actualS1S2AlignInnerSize); // -10000
        pipe_barrier(PIPE_V);
        Add(bmm2ResUb, castedAttenMaskUb, bmm2ResUb, actualS1S2AlignInnerSize);
        vecOutQue.FreeTensor(castedAttenMaskUb);
    } else {
        Muls(attenMaskUb, attenMaskUb, mulsCalcValue, actualS1S2AlignInnerSize); // -10000
        pipe_barrier(PIPE_V);
        Add(bmm2ResUb, attenMaskUb, bmm2ResUb, actualS1S2AlignInnerSize);
    }
    vecInQue2.FreeTensor(attenMaskUb);
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::CalcSoftmax(
    LocalTensor<CALC_T>& bmm2ResUb) {
    LocalTensor<float> softmaxSumUb = softmaxQue2.template DeQue<float>();
    softmaxSumUb.SetSize(softmaxSize);
    uint32_t softmaxShape[] = {actualS1InnerSize, 8};
    softmaxSumUb.SetShapeInfo(ShapeInfo(2, softmaxShape, DataFormat::ND));

    LocalTensor<float> softmaxMaxUb = softmaxQue1.template DeQue<float>();
    softmaxMaxUb.SetSize(softmaxSize);
    softmaxMaxUb.SetShapeInfo(ShapeInfo(2, softmaxShape, DataFormat::ND));

    // softmax接口，在S1 8对齐和S2 64对齐时，可以使能基本块，优化性能
    bool isBasicBlock = (actualS1InnerSize % 8 == 0) && (actualS2InnerSize % 64 == 0);
    LocalTensor<uint8_t> helpTensor = castBuff.template Get<uint8_t>();
    if (likely(isBasicBlock)) {
        SimpleSoftMax<CALC_T, true, true>(bmm2ResUb, softmaxSumUb, softmaxMaxUb, bmm2ResUb, helpTensor,
                                          ordTilingData->softmaxTilingData);
    } else {
        SimpleSoftMax<CALC_T, true, false>(bmm2ResUb, softmaxSumUb, softmaxMaxUb, bmm2ResUb, helpTensor,
                                           ordTilingData->softmaxTilingData);
    }
    softmaxQue2.FreeTensor(softmaxSumUb);
    softmaxQue1.FreeTensor(softmaxMaxUb);
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::CalcDropout(
    LocalTensor<CALC_T>& dst, LocalTensor<CALC_T>& src, LocalTensor<uint8_t>& dropMaskUb) {
    uint32_t dropMaskS2InnerAlignSize = (actualS2InnerSize + 32 - 1) / 32 * 32; // uint8 模式
    dropMaskUb.SetSize(actualS1InnerSize * dropMaskS2InnerAlignSize);

    DropOutShapeInfo info;
    info.firstAxis = actualS1InnerSize;
    info.srcLastAxis = actualS2InnerAlignSize;
    info.maskLastAxis = dropMaskS2InnerAlignSize;

    LocalTensor<uint8_t> helpTensor = vecOutQue.template AllocTensor<uint8_t>();
    vecOutQue.EnQue(helpTensor);
    helpTensor = vecOutQue.template DeQue<uint8_t>();

    DropOut<CALC_T, true>(dst, src, dropMaskUb, helpTensor,
                          ordTilingData->baseParams.keepProb, info);
    vecOutQue.FreeTensor(helpTensor);
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::CalcSoftmaxGradWithSplitD(
    LocalTensor<CALC_T>& softmaxGradResUb) {
    uint64_t tmpResSize = actualS1InnerSize * blockCalcTypeNum;
    // 清空Res，用于累加
    Duplicate<CALC_T>(softmaxGradResUb, 0, tmpResSize);
    // 中间结果buffer
    LocalTensor<CALC_T> tmpResAddUb = softmaxQue2.template AllocTensor<CALC_T>();
    tmpResAddUb.SetSize(tmpResSize);
    uint32_t softmaxShape[] = {actualS1InnerSize, blockCalcTypeNum};
    tmpResAddUb.SetShapeInfo(ShapeInfo(2, softmaxShape, DataFormat::ND));

    uint32_t loopTimes = ordTilingData->baseParams.D / s2InnerSize;
    uint32_t remainTail = ordTilingData->baseParams.D % s2InnerSize;
    for (uint32_t loop = 0; loop < loopTimes; loop++) {
        uint32_t gmOffset = s1LoopOffset + loop * s2InnerSize;
        LocalTensor<INPUT_T> attenInUb = vecInQue1.template AllocTensor<INPUT_T>();
        LocalTensor<INPUT_T> dyUb = vecInQue2.template AllocTensor<INPUT_T>();
        CopyInDyAttenInWithSplitD(dyUb, attenInUb, gmOffset, s2InnerSize);
        uint32_t s2InnerAlignSize = (s2InnerSize + blockInputTypeNum - 1) / blockInputTypeNum * blockInputTypeNum;
        CalcSoftmaxGrad(tmpResAddUb, s2InnerSize, s2InnerAlignSize);
        pipe_barrier(PIPE_V);
        Add(softmaxGradResUb, softmaxGradResUb, tmpResAddUb, tmpResSize);
    }
    if (remainTail) {
        uint32_t gmOffset = s1LoopOffset + loopTimes * s2InnerSize;
        uint32_t dAlignSize = (remainTail + blockInputTypeNum - 1) / blockInputTypeNum * blockInputTypeNum;
        LocalTensor<INPUT_T> attenInUb = vecInQue1.template AllocTensor<INPUT_T>();
        LocalTensor<INPUT_T> dyUb = vecInQue2.template AllocTensor<INPUT_T>();
        CopyInDyAttenInWithSplitD(dyUb, attenInUb, gmOffset, remainTail);
        CalcSoftmaxGrad(tmpResAddUb, remainTail, dAlignSize);
        pipe_barrier(PIPE_V);
        Add(softmaxGradResUb, softmaxGradResUb, tmpResAddUb, tmpResSize);
    }
    softmaxQue2.FreeTensor(tmpResAddUb);
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::CalcSoftmaxGrad(
    LocalTensor<CALC_T>& softmaxGradResUb, uint32_t dInnerSize, uint32_t dInnerAlignSize) {
    uint64_t actualSize = actualS1InnerSize * dInnerAlignSize;
    bool isBasicBlock = (actualS1InnerSize % 8 == 0) && (dInnerAlignSize % 64 == 0);
    uint32_t shapeArray[] = {actualS1InnerSize, dInnerAlignSize};
    // trans attenInUb to cast dtype
    LocalTensor<INPUT_T> attenInUb = vecInQue1.template DeQue<INPUT_T>();
    LocalTensor<INPUT_T> dyUb = vecInQue2.template DeQue<INPUT_T>();
    dyUb.SetSize(actualSize);
    dyUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    attenInUb.SetSize(actualSize);
    attenInUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    if constexpr (HIGH_PRECISE) {
        LocalTensor<CALC_T> castedDyUb = vecOutQue.template AllocTensor<CALC_T>();
        castedDyUb.SetSize(actualSize);
        castedDyUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        Cast(castedDyUb, dyUb, RoundMode::CAST_NONE, actualSize);

        LocalTensor<CALC_T> castedattenInUb = castBuff.template Get<CALC_T>(actualSize);
        castedattenInUb.SetSize(actualSize);
        castedattenInUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        Cast(castedattenInUb, attenInUb, RoundMode::CAST_NONE, actualSize);
        pipe_barrier(PIPE_V);

        vecInQue2.FreeTensor(dyUb);
        vecInQue1.FreeTensor(attenInUb);

        LocalTensor<uint8_t> tempBuf = vecInQue1.template AllocTensor<uint8_t>();
        if (likely(isBasicBlock)) {
            SoftmaxGradFront<CALC_T, true>(softmaxGradResUb, castedattenInUb, castedDyUb, tempBuf,
                                           ordTilingData->softmaxGradTilingData);
        } else {
            SoftmaxGradFront<CALC_T, false>(softmaxGradResUb, castedattenInUb, castedDyUb, tempBuf,
                                            ordTilingData->softmaxGradTilingData);
        }
        vecOutQue.FreeTensor(castedDyUb);
        vecInQue1.FreeTensor(tempBuf);
    } else {
        LocalTensor<uint8_t> tempBuf = castBuff.template Get<uint8_t>();
        if (likely(isBasicBlock)) {
            SoftmaxGradFront<CALC_T, true>(softmaxGradResUb, attenInUb, dyUb, tempBuf,
                                           ordTilingData->softmaxGradTilingData);
        } else {
            SoftmaxGradFront<CALC_T, false>(softmaxGradResUb, attenInUb, dyUb, tempBuf,
                                            ordTilingData->softmaxGradTilingData);
        }
        vecInQue2.FreeTensor(dyUb);
        vecInQue1.FreeTensor(attenInUb);
    }
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::CalcSub(
    LocalTensor<CALC_T>& dst, LocalTensor<CALC_T>& src, LocalTensor<CALC_T>& softmaxGradResUb) {
    uint32_t s2BlockNums = actualS2InnerAlignSize / blockCalcTypeNum;
    uint32_t s2LoopTimes = s2BlockNums / REPEAT_BLOCK_NUMS;
    uint32_t remainS2BlockNums = s2BlockNums % REPEAT_BLOCK_NUMS;
    uint32_t maskNums = REPEAT_BLOCK_NUMS * blockCalcTypeNum;

    BinaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 0;
    repeatParams.dstRepStride = s2BlockNums;
    repeatParams.src0RepStride = s2BlockNums;
    repeatParams.src1RepStride = 1;
    // 按列方向循环，按行方向repeat，对128*64的基本块更友好
    for (int32_t idx = 0; idx < s2LoopTimes; idx++) {
        uint64_t src0Offset = idx * maskNums;
        Sub(dst[src0Offset], src[src0Offset], softmaxGradResUb, maskNums, actualS1InnerSize, repeatParams);
    }
    // 尾块，需要掩码
    if (remainS2BlockNums) {
        uint64_t src0Offset = s2LoopTimes * maskNums;
        Sub(dst[src0Offset], src[src0Offset], softmaxGradResUb,
            remainS2BlockNums * blockCalcTypeNum, actualS1InnerSize, repeatParams);
    }
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::CalcMul(
    LocalTensor<CALC_T>& dst, LocalTensor<CALC_T>& src0, LocalTensor<CALC_T>& src1, uint64_t calcSize) {
    Mul(dst, src0, src1, calcSize);
}
// end base类方法

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::Init(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* dy,
    __gm__ uint8_t* pse_shift, __gm__ uint8_t* drop_mask, __gm__ uint8_t* padding_mask, __gm__ uint8_t* atten_mask,
    __gm__ uint8_t* softmax_max, __gm__ uint8_t* softmax_sum, __gm__ uint8_t* softmax_in, __gm__ uint8_t* attention_in,
    __gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv, __gm__ uint8_t* dpse,
    __gm__ uint8_t* workspace, const FlashAttentionScoreGradUs1s2BbnTilingData* __restrict tilingData, TPipe *pipe_in,
    uint64_t *bmm31Offset, uint64_t *bmm32Offset, uint64_t *bmm4Offset) {
    // init Gm
    queryGm.SetGlobalBuffer((__gm__ INPUT_T *)query);
    keyGm.SetGlobalBuffer((__gm__ INPUT_T *)key);
    valueGm.SetGlobalBuffer((__gm__ INPUT_T *)value);
    dyGm.SetGlobalBuffer((__gm__ INPUT_T *)dy);
    pseGm.SetGlobalBuffer((__gm__ INPUT_T *)pse_shift);
    dropMaskGm.SetGlobalBuffer((__gm__ uint8_t *)drop_mask);
    paddingMaskGm.SetGlobalBuffer((__gm__ INPUT_T *)padding_mask);
    attenMaskU8Gm.SetGlobalBuffer((__gm__ uint8_t *)atten_mask);
    attenMaskGm.SetGlobalBuffer((__gm__ INPUT_T *)atten_mask);
    attenInGm.SetGlobalBuffer((__gm__ INPUT_T *)attention_in);
    softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmax_max);
    softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmax_sum);
    dqGm.SetGlobalBuffer((__gm__ INPUT_T *)dq);
    dkGm.SetGlobalBuffer((__gm__ INPUT_T *)dk);
    dvGm.SetGlobalBuffer((__gm__ INPUT_T *)dv);
    workspaceGm.SetGlobalBuffer((__gm__ INPUT_T *)workspace);

    mBlockIdx = GetBlockIdx();
    ordTilingData = tilingData;
    pipe = pipe_in;

    // 需要根据Tiling传入这个字段
    notSplitGradD = ordTilingData->splitCoreParams.notSplitGradD;

    hasPseInput = ordTilingData->baseParams.hasPseInput;
    hasAttenMaskInput = ordTilingData->baseParams.hasAttenMaskInput;
    attenMaskDimS2 = ordTilingData->baseParams.attenMaskS2Size;

    inputTypeNum = sizeof(INPUT_T);
    blockInputTypeNum = BLOCK_BYTE / inputTypeNum;
    blockCalcTypeNum = BLOCK_BYTE / sizeof(CALC_T);
    physicalCoreNum = GetBlockNum() * GetTaskRation();
    usedCoreNum = ordTilingData->baseParams.coreNum;
    syncDropoutLen = ordTilingData->baseParams.syncDropoutLen;
    dropoutWorkspaceLen = ordTilingData->dropoutCastParams.dropoutWorkspaceLen;
    dropoutWorkspaceGm.SetGlobalBuffer((__gm__ uint8_t*)workspace + 800 * sizeof(INPUT_T) + syncDropoutLen);

    // 当D > s2_inner时，需要切D，按照S2Inner切D
    pipe->InitBuffer(vecInQue1, 1, ordTilingData->splitCoreParams.innerMatResSize);  // s1InnerSize * s2InnerSize * sizeof(Input_t)
    pipe->InitBuffer(vecInQue2, 1, ordTilingData->splitCoreParams.innerMatResSize);  // s1InnerSize * s2InnerSize * sizeof(Input_t)
    // 这块size的名字统一，在tiling中根据数据类型进行计算bufferSize
    if constexpr (HIGH_PRECISE) {
        pipe->InitBuffer(vecOutQue, 1, ordTilingData->splitCoreParams.innerMatResSize * 2);  // s1InnerSize * s2InnerSize * sizeof(Calc_t)
        pipe->InitBuffer(bmm2ResQue, 1, ordTilingData->splitCoreParams.innerMatResSize * 2);  // s1InnerSize * s2InnerSize * sizeof(Calc_t)
        pipe->InitBuffer(bmm1ResQue, 1, ordTilingData->splitCoreParams.innerMatResSize * 2);  // s1InnerSize * s2InnerSize * sizeof(Calc_t)
        pipe->InitBuffer(castBuff, ordTilingData->splitCoreParams.innerMatResSize * 2);  // s1InnerSize * s2InnerSize * sizeof(Calc_t)
    } else {
        pipe->InitBuffer(vecOutQue, 1, ordTilingData->splitCoreParams.innerMatResSize);  // s1InnerSize * s2InnerSize * sizeof(Calc_t)
        pipe->InitBuffer(bmm2ResQue, 1, ordTilingData->splitCoreParams.innerMatResSize);  // s1InnerSize * s2InnerSize * sizeof(Calc_t)
        pipe->InitBuffer(bmm1ResQue, 1, ordTilingData->splitCoreParams.innerMatResSize);  // s1InnerSize * s2InnerSize * sizeof(Calc_t)
    }

    pipe->InitBuffer(softmaxQue1, 1, ordTilingData->splitCoreParams.s1InnerSize * BLOCK_BYTE);
    pipe->InitBuffer(softmaxQue2, 1, ordTilingData->splitCoreParams.s1InnerSize * BLOCK_BYTE);
    pipe->InitBuffer(softmaxGradResBuf, ordTilingData->splitCoreParams.s1InnerSize * BLOCK_BYTE);

    // workspace1 & workspace2 spm: totalSize=S1*S2*coreNum*2
    s2AlignSize = (ordTilingData->baseParams.Skv + blockInputTypeNum - 1) / blockInputTypeNum * blockInputTypeNum;
    spmWorkspaceSize = ordTilingData->baseParams.S * s2AlignSize;
    singleCoreSpmSize = spmWorkspaceSize * 2;

    uint64_t s2AlignSizeByBaseN = (ordTilingData->splitCoreParams.cubeBaseS2 +
                                   ordTilingData->splitCoreParams.s2InnerSize - 1) /
                                  ordTilingData->splitCoreParams.s2InnerSize *
                                  ordTilingData->splitCoreParams.s2InnerSize; // 对齐到BASEN的整数倍
    uint64_t bmm12WorkspaceSize = ordTilingData->splitCoreParams.cubeBaseS1 * s2AlignSizeByBaseN;

    // workspaceGm申请的时候是按照INPUT_T类型申请的，但是bmm1/2的workspaceGm是CALC_T类型，这里需要进行换算
    uint64_t bmm12AdjustSize = bmm12WorkspaceSize * sizeof(CALC_T) / sizeof(INPUT_T);
    uint64_t blockWorkspaceSize = singleCoreSpmSize + bmm12AdjustSize * 2;
    // w1的起始位置 800是多核同步预留的，该模板不需要多核同步，但这里为了保持一致，暂未删除
    spmBufOffset = 800 + (syncDropoutLen + dropoutWorkspaceLen) / sizeof(INPUT_T) + mBlockIdx * blockWorkspaceSize;
    // bmm2的起始位置
    uint64_t buffOffset = spmBufOffset + singleCoreSpmSize;
    bmm2.SetWorkspace((__gm__ uint8_t *)workspaceGm[buffOffset].GetPhyAddr(), bmm12WorkspaceSize * sizeof(CALC_T));
    bmm1.SetWorkspace((__gm__ uint8_t *)workspaceGm[buffOffset + bmm12AdjustSize].GetPhyAddr(),
        bmm12WorkspaceSize * sizeof(CALC_T));

    uint64_t postGmOffset = (800 + (syncDropoutLen + dropoutWorkspaceLen) / sizeof(INPUT_T) + physicalCoreNum * blockWorkspaceSize) * sizeof(INPUT_T) / sizeof(CALC_T);

    uint64_t dqWorkspaceLen = ordTilingData->postOpTilingParams.dqWorkspaceLen;
    uint64_t dkWorkspaceLen = ordTilingData->postOpTilingParams.dkWorkspaceLen;
    uint64_t dvWorkspaceLen = ordTilingData->postOpTilingParams.dvWorkspaceLen;

    *bmm31Offset = postGmOffset;
    *bmm32Offset = *bmm31Offset + dqWorkspaceLen / sizeof(CALC_T);
    *bmm4Offset = *bmm32Offset + dkWorkspaceLen / sizeof(CALC_T);

    postOpWorkspaceDqGm.SetGlobalBuffer(((__gm__ CALC_T*)workspace + *bmm31Offset));
    postOpWorkspaceDkGm.SetGlobalBuffer(((__gm__ CALC_T*)workspace + *bmm32Offset));
    postOpWorkspaceDvGm.SetGlobalBuffer(((__gm__ CALC_T*)workspace + *bmm4Offset));

    syncGlobal.SetGlobalBuffer((__gm__ int32_t*)workspace);
    InitOutput<int32_t>(syncGlobal[GetBlockIdx() * 8], 8, 0);
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::InitCalcParams() {
    if (ordTilingData->baseParams.keepProb < 1) {
        isDrop = true;
    }
    // tiling需要传入S1,S2方向的切块数量、s1、s2InnerSize
    normalCoreNums = ordTilingData->baseParams.formerCoreNum;
    blockSplitBnRange = ordTilingData->splitCoreParams.splitedBatchRange;
    s1RangeV = ordTilingData->splitCoreParams.s1Range;
    s2RangeV = ordTilingData->splitCoreParams.s2Range;
    s1InnerSize = ordTilingData->splitCoreParams.s1InnerSize;
    s2InnerSize = ordTilingData->splitCoreParams.s2InnerSize;
    tailS1InnerSize = ordTilingData->splitCoreParams.tailS1InnerSize;
    tailS2InnerSize = ordTilingData->splitCoreParams.tailS2InnerSize;
    s1RangeC = ordTilingData->splitCoreParams.s1RangeC;
    s2RangeC = ordTilingData->splitCoreParams.s2RangeC;
    cubeBaseS1 = ordTilingData->splitCoreParams.cubeBaseS1;
    cubeBaseS2 = ordTilingData->splitCoreParams.cubeBaseS2;
    tailCubeBaseS1 = ordTilingData->splitCoreParams.tailCubeBaseS1;
    tailCubeBaseS2 = ordTilingData->splitCoreParams.tailCubeBaseS2;
    s1CVRatio = ordTilingData->splitCoreParams.s1CVRatio;
    s2CVRatio = ordTilingData->splitCoreParams.s2CVRatio;
    softmaxSize = ordTilingData->splitCoreParams.innerReduceNum;
    actualDAlignSize = (ordTilingData->baseParams.D + blockInputTypeNum - 1) / blockInputTypeNum * blockInputTypeNum;
    if (ordTilingData->baseParams.layoutType == 0) {
        layOutType = LAYOUT::BSH;
    } else if (ordTilingData->baseParams.layoutType == 1) {
        layOutType = LAYOUT::SBH;
    } else if (ordTilingData->baseParams.layoutType == 2) {
        layOutType = LAYOUT::BNSD;
    } else if (ordTilingData->baseParams.layoutType == 3) {
        layOutType = LAYOUT::BSND;
    }
    beforeBatchReduceNums = mBlockIdx * ordTilingData->splitCoreParams.splitedCoreBatchReduceNum;
    beforeBlockSplitBnNums = mBlockIdx * blockSplitBnRange;
    // tail core
    if (mBlockIdx >= normalCoreNums) {
        blockSplitBnRange = ordTilingData->splitCoreParams.tailSplitedBatchRange;
        beforeBatchReduceNums = normalCoreNums * ordTilingData->splitCoreParams.splitedCoreBatchReduceNum +
            (mBlockIdx - normalCoreNums) * ordTilingData->splitCoreParams.tailSplitedCoreBatchReduceNum;
        beforeBlockSplitBnNums = normalCoreNums * ordTilingData->splitCoreParams.splitedBatchRange +
            (mBlockIdx - normalCoreNums) * ordTilingData->splitCoreParams.tailSplitedBatchRange;
    }
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::CalcBNOffset(const uint32_t bnIdx) {
    uint32_t bIdx = (beforeBlockSplitBnNums + bnIdx) / ordTilingData->baseParams.N;
    uint32_t nIdx = (beforeBlockSplitBnNums + bnIdx) % ordTilingData->baseParams.N;
    switch (layOutType) {
        case LAYOUT::BSH:
        case LAYOUT::BSND:
            bnLoopS1Offset = bIdx * ordTilingData->baseParams.S * ordTilingData->baseParams.H +
                             nIdx * ordTilingData->baseParams.D;
            bnLoopS2Offset = bIdx * ordTilingData->baseParams.Skv * ordTilingData->baseParams.H +
                             nIdx * ordTilingData->baseParams.D;
            break;
        case LAYOUT::SBH:
            bnLoopS1Offset = bIdx * ordTilingData->baseParams.H +
                             nIdx * ordTilingData->baseParams.D;
            bnLoopS2Offset = bnLoopS1Offset;
            break;
        case LAYOUT::BNSD:
            bnLoopS1Offset = bIdx * ordTilingData->baseParams.S * ordTilingData->baseParams.H +
                             nIdx * ordTilingData->baseParams.S * ordTilingData->baseParams.D;
            bnLoopS2Offset = bIdx * ordTilingData->baseParams.Skv * ordTilingData->baseParams.H +
                             nIdx * ordTilingData->baseParams.Skv * ordTilingData->baseParams.D;
            break;
    }
    bnLoopSoftmaxOffset = beforeBatchReduceNums + bnIdx * ordTilingData->baseParams.allReduceNum;
    // BNS1S2
    bnLoopDropMaskOffset = bIdx * ordTilingData->baseParams.N * ordTilingData->baseParams.S * ordTilingData->baseParams.Skv +
        nIdx * ordTilingData->baseParams.S * ordTilingData->baseParams.Skv;

    if (ordTilingData->baseParams.pseShapeType == 1) {
        // pse BN1S2场景offset
        bnLoopPseOffset = bIdx * ordTilingData->baseParams.N * ordTilingData->baseParams.Skv +
                          nIdx * ordTilingData->baseParams.Skv;
    } else {
        // pse BNS1S2场景offset
        bnLoopPseOffset = bnLoopDropMaskOffset;
    }

    if (ordTilingData->baseParams.attenMaskShapeType == 0) {
        // mask按11S1S2场景
        bnLoopMaskOffset = 0;
    } else if (ordTilingData->baseParams.attenMaskShapeType == 1) {
        // mask按B1S1S2场景设置
        bnLoopMaskOffset = bIdx * ordTilingData->baseParams.S * ordTilingData->baseParams.Skv;
    } else {
        // mask按BNS1S2场景设置
        bnLoopMaskOffset = bnLoopDropMaskOffset;
    }
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::CalcAttenMaskOffset(
    uint64_t& attenMaskOffset, const int64_t delta) {
    if (delta == 0) {
        attenMaskOffset = 0;
    } else if (delta < 0) {
        if (-delta > s1InnerSize) {
            attenMaskOffset = s1InnerSize;
        } else {
            attenMaskOffset = -delta;
        }
    } else {
        if (delta > s2InnerSize) {
            attenMaskOffset = s2InnerSize * attenMaskDimS2;
        } else {
            attenMaskOffset = delta * attenMaskDimS2;
        }
    }
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::CalcCausalAttenMaskOffset(
    const int64_t delta) {
    CalcAttenMaskOffset(attenMaskOffset, delta);
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::CalcS1OffsetLoopParams(
    const uint32_t s1Idx) {
    s1LoopOffset = bnLoopS1Offset + s1Idx * s1InnerSize * ordTilingData->splitCoreParams.SOffset;
    s1LoopSoftmaxOffset = bnLoopSoftmaxOffset + s1Idx * (s1InnerSize * SOFTMAX_LAST_AXIS_SIZE);

    actualS1InnerAlignSize = s1InnerSize;
    // S1方向尾块处理
    if (unlikely(s1Idx == s1RangeV - 1)) {
        actualS1InnerSize = tailS1InnerSize;
        actualS1InnerAlignSize = (actualS1InnerSize + blockInputTypeNum - 1) / blockInputTypeNum;
    }
    actualS1DSize = actualS1InnerSize * ordTilingData->baseParams.D;
    actualS1DAlignSize = actualS1InnerSize * actualDAlignSize;
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::CalcForwardBMMS1Offset(
    const uint32_t s1IdxC) {
    // S1方向loop的offset，只有最后一块是尾块时，offset的计算不需要考虑尾块
    forwardBmmS1LoopOffset = bnLoopS1Offset + s1IdxC * cubeBaseS1 * ordTilingData->splitCoreParams.SOffset;

    // S1方向尾块处理
    if (unlikely(s1IdxC == s1RangeC - 1)) {
      actualCubeBaseS1 = tailCubeBaseS1;
    }
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::CalcSparseIdx(
        const uint32_t bIdx, const uint32_t s1Idx, const uint32_t s2Start, const uint32_t s2End, const uint32_t s1BaseSize,
        const uint32_t s2BaseSize, uint32_t &s2LoopStartIdx, uint32_t &s2LoopEndIdx) {
    if (ordTilingData->baseParams.sparseMode == 3 || ordTilingData->baseParams.sparseMode == 5) {
        // s2LoopStartIdx = 0 除bmm发射S2IdxC不=0的情况外，其余传入s2LoopStartIdx注意在外部置0.
        int32_t s2IgnoredEndLen = static_cast<int32_t>(ordTilingData->baseParams.Sq) -
                                  static_cast<uint32_t>(s1BaseSize) * (s1Idx + 1);
        int32_t s2EndLen = 0;
        if (static_cast<int32_t>(ordTilingData->baseParams.Skv) > s2IgnoredEndLen) {
            s2EndLen = static_cast<int32_t>(ordTilingData->baseParams.Skv) - s2IgnoredEndLen;
        } else {
            s2EndLen = 0;
        }
        if (ordTilingData->baseParams.sparseMode == 5) {
            int32_t prefix_n = static_cast<int32_t>(ordTilingData->baseParams.prefixN[bIdx]);
            s2EndLen = s2EndLen >= prefix_n ? s2EndLen : prefix_n;
        }
        s2LoopEndIdx = (static_cast<uint32_t>(s2EndLen) + s2BaseSize - 1) / s2BaseSize;
        s2LoopEndIdx = s2LoopEndIdx > s2End ? s2End : s2LoopEndIdx;
    } else {
        // S等长的sparse计算
        int32_t s1OuterSize = s1Idx * s1BaseSize;
        int32_t s2StartIdx = (s1OuterSize - (int32_t)(ordTilingData->baseParams.preTokens)) /
                            (int32_t)s2BaseSize;
        s2LoopStartIdx = s2StartIdx < (int32_t)s2Start ? s2Start : (uint32_t)s2StartIdx;

        int32_t s2EndIdx = (s1OuterSize + s1BaseSize + (int32_t)(ordTilingData->baseParams.nextTokens) +
                            (int32_t)s2BaseSize - 1) / (int32_t)s2BaseSize;
        s2LoopEndIdx = s2EndIdx > (int32_t)s2End ? s2End : (uint32_t)s2EndIdx;
    }
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::CalcS2OffsetLoopParams(
    const uint32_t s1Idx, const uint32_t s2Idx) {
    uint64_t compressMode = ordTilingData->baseParams.attenMaskCompressMode;
    if (compressMode == 1) {
        // causal s1==s2
        int64_t causal_delta = (int64_t)(s1Idx * s1InnerSize) - (int64_t)(s2Idx * s2InnerSize);
        CalcCausalAttenMaskOffset(causal_delta);
    } else if (compressMode == 2) {
        // causal s1!=s2
        int64_t causal_delta = (int64_t)(s1Idx * s1InnerSize) - (int64_t)(s2Idx * s2InnerSize);
        causal_delta = causal_delta - ordTilingData->baseParams.Sq + ordTilingData->baseParams.Skv;
        CalcCausalAttenMaskOffset(causal_delta);
    } else {
        attenMaskOffset = bnLoopMaskOffset +
                          s1Idx * (s1InnerSize * ordTilingData->baseParams.Skv) + s2Idx * s2InnerSize;
    }
    // dropmask: BNS1S2
    dropMaskOffset = bnLoopDropMaskOffset + s1Idx * (s1InnerSize * ordTilingData->baseParams.Skv) + s2Idx * s2InnerSize;
    if (ordTilingData->baseParams.pseShapeType == 0) {
        // pse-shape为(BNS1S2)的offset
        pseOffset = dropMaskOffset;
    } else {
        // pse-shape为(BN1S2)的offset
        pseOffset = bnLoopPseOffset + s2Idx * s2InnerSize;
    }
    // 对齐拷到workspace上
    uint64_t outOffset = s1Idx * (s1InnerSize * s2AlignSize) + s2Idx * s2InnerSize;
    copyOut1Offset = spmBufOffset + outOffset;
    copyOut2Offset = spmBufOffset + spmWorkspaceSize + outOffset;

    actualS2InnerAlignSize = actualS2InnerSize;
    // S1方向尾块处理
    if (unlikely(s2Idx == s2RangeV - 1)) {
        actualS2InnerSize = tailS2InnerSize;
        actualS2InnerAlignSize = (actualS2InnerSize + blockInputTypeNum - 1) / blockInputTypeNum * blockInputTypeNum;
        if (HIGH_PRECISE) {
          mmPadDstGap = actualS2InnerAlignSize - actualS2InnerSize >= blockCalcTypeNum ? 1 : 0;
        }
    }

    actualS1S2InnerSize = actualS1InnerSize * actualS2InnerSize;
    actualS1S2AlignInnerSize = actualS1InnerSize * actualS2InnerAlignSize;
    s1S2ShapeAlignArray[0] = actualS1InnerSize;
    s1S2ShapeAlignArray[1] = actualS2InnerAlignSize;
    s1S2ShapeArray[0] = actualS1InnerSize;
    s1S2ShapeArray[1] = actualS2InnerSize;
}


template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::CalcForwardBmmEmitParams(
    const uint32_t bnIdx, const uint32_t nextS1IdxC, uint64_t& sparseS2Offset, uint32_t& curS2) {
    uint32_t nextS2LoopStartIdxC = 0;
    uint32_t nextS2LoopEndIdxC = s2RangeC;
    if (ordTilingData->splitCoreParams.sparse == 1) {
        CalcSparseIdx(bnIdx, nextS1IdxC, nextS2LoopStartIdxC, nextS2LoopEndIdxC, cubeBaseS1, cubeBaseS2, nextS2LoopStartIdxC, nextS2LoopEndIdxC);
    }
    uint32_t nextEmitS2StartIdx = nextS2LoopStartIdxC * s2CVRatio;
    uint32_t nextEmitS2EndIdx = nextEmitS2StartIdx + s2CVRatio > s2RangeV ? s2RangeV : nextEmitS2StartIdx + s2CVRatio;
    if (ordTilingData->splitCoreParams.sparse == 1) {
        CalcSparseIdx(bnIdx, nextS1IdxC, nextEmitS2StartIdx, nextEmitS2EndIdx,
                        cubeBaseS1, s2InnerSize, nextEmitS2StartIdx, nextEmitS2EndIdx);
    }

    uint32_t allSkvAlignSize = s2RangeC * cubeBaseS2;
    // 计算真实curS2，按照整块进行计算，去头去尾，如果尾巴不去，则剪掉补齐的padSize
    curS2 = allSkvAlignSize - nextEmitS2StartIdx * s2InnerSize;
    if (nextEmitS2EndIdx == s2RangeV) {
        curS2 -= allSkvAlignSize - ordTilingData->baseParams.Skv;
    } else {
        curS2 = (nextEmitS2EndIdx - nextEmitS2StartIdx) * s2InnerSize;
    }

    sparseS2Offset = bnLoopS2Offset;
    sparseS2Offset += nextEmitS2StartIdx * s2InnerSize * ordTilingData->splitCoreParams.SOffset;

    // 更新actualCubeBaseS1，为下一次s1IdxC的提前发射使用
    if (nextS1IdxC == s1RangeC - 1) {
        actualCubeBaseS1 = tailCubeBaseS1;
    }
}


template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::forwardBmm2Emit(
    const uint32_t s1IdxC, const uint64_t sparseS2Offset, const uint32_t curS2) {
    uint32_t newS1LoopOffset = bnLoopS1Offset + s1IdxC * cubeBaseS1 * ordTilingData->splitCoreParams.SOffset;

    bmm2.SetTensorA(queryGm[newS1LoopOffset]);
    bmm2.SetTensorB(keyGm[sparseS2Offset], true);
    bmm2.SetTail(actualCubeBaseS1, curS2, -1);
    bmm2.template Iterate<false>();
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::forwardBmm1Emit(
    const uint32_t s1IdxC, const uint64_t sparseS2Offset, const uint32_t curS2) {
    uint32_t newS1LoopOffset = bnLoopS1Offset + s1IdxC * cubeBaseS1 * ordTilingData->splitCoreParams.SOffset;

    bmm1.SetTensorA(dyGm[newS1LoopOffset]);
    bmm1.SetTensorB(valueGm[sparseS2Offset], true);
    bmm1.SetTail(actualCubeBaseS1, curS2, -1);
    bmm1.template Iterate<false>();
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::forwardBmmEmit(
    const uint32_t s2LoopStartIdxV, const uint32_t s2LoopEndIdxV) {
    uint32_t allSkvAlignSize = s2RangeC * cubeBaseS2; // cubeBaseS2
    // 计算真实curS2，按照整块进行计算，去头去尾，如果尾巴不去，则剪掉补齐的padSize
    uint32_t curS2 = allSkvAlignSize - s2LoopStartIdxV * s2InnerSize;
    if (s2LoopEndIdxV == s2RangeV) {
        curS2 -= allSkvAlignSize - ordTilingData->baseParams.Skv;
    } else {
        curS2 = (s2LoopEndIdxV - s2LoopStartIdxV) * s2InnerSize;
    }

    uint64_t sparseS2Offset = bnLoopS2Offset;
    sparseS2Offset += s2LoopStartIdxV * s2InnerSize * ordTilingData->splitCoreParams.SOffset;

    // BMM2 BMM1 发射整行, 输入cubeBaseS1*D, D*s2, 输出cubeBaseS1*s2
    bmm2.SetTensorA(queryGm[forwardBmmS1LoopOffset]);
    bmm2.SetTensorB(keyGm[sparseS2Offset], true);
    bmm2.SetTail(actualCubeBaseS1, curS2, -1);
    bmm2.template Iterate<false>();

    bmm1.SetTensorA(dyGm[forwardBmmS1LoopOffset]);
    bmm1.SetTensorB(valueGm[sparseS2Offset], true);
    bmm1.SetTail(actualCubeBaseS1, curS2, -1);
    bmm1.template Iterate<false>();
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::SimpleSoftmaxComputeStage(
    LocalTensor<CALC_T>& bmm2ResUb, LocalTensor<uint8_t>& dropMaskUb) {
    /**
        *workspace1 处理过程：BMM2 -> pseAdd -> Mask -> Softmax -> Dropout
    */
    // bmm2计算的同时，启动pse、atten_mask的搬入
    hasPse = CopyInPse();
    bool hasMask = CopyInAttenMask();

    // bmm2ResUb tensor 申请一次，最后释放
    bmm2ResUb = bmm2ResQue.template AllocTensor<CALC_T>();
    bmm2ResUb.SetShapeInfo(ShapeInfo(2, s1S2ShapeAlignArray, 2, s1S2ShapeArray, DataFormat::ND));
    bmm2ResUb.SetSize(actualS1S2AlignInnerSize);
    bmm2.template GetTensorC<false, true>(bmm2ResUb, false, false, actualS1InnerSize, actualS2InnerSize,
                                          mmPadSrcGap, mmPadDstGap);

    CalcPseAdd(bmm2ResUb);
    pipe_barrier(PIPE_V);

    CopyInSoftmaxMax();
    CopyInSoftmaxSum();

    if (hasMask) {
        if (ordTilingData->baseParams.attenMaskType == 0) {  // 0 表示输入数据类型与 qkv 一致
            CalcMask(bmm2ResUb);
        } else {
            CalcU8Mask(bmm2ResUb);
        }
    }
    pipe_barrier(PIPE_V);

    dropMaskUb = vecInQue1.template AllocTensor<uint8_t>();
    if (isDrop) {
        CopyInDropMask(dropMaskUb);
    }
    CalcSoftmax(bmm2ResUb);
    pipe_barrier(PIPE_V);

    if constexpr (HIGH_PRECISE) {
        if (isDrop) {
            LocalTensor<CALC_T> castedDropOutDoMask = castBuff.template Get<CALC_T>(actualS1S2AlignInnerSize);
            castedDropOutDoMask.SetSize(actualS1S2AlignInnerSize);
            castedDropOutDoMask.SetShapeInfo(ShapeInfo(2, s1S2ShapeAlignArray, DataFormat::ND));
            // 持有dropMask不复用
            dropMaskUb = vecInQue1.template DeQue<uint8_t>();
            CalcDropout(castedDropOutDoMask, bmm2ResUb, dropMaskUb);
            pipe_barrier(PIPE_V);
            CopyOutWorkspaceWithCast(castedDropOutDoMask, copyOut1Offset);
        } else {
            CopyOutWorkspaceWithCast(bmm2ResUb, copyOut1Offset);
        }
    } else {
        if (isDrop) {
            LocalTensor<INPUT_T> vecOutUb1 = vecOutQue.template AllocTensor<INPUT_T>();
            vecOutUb1.SetSize(actualS1S2AlignInnerSize);
            vecOutUb1.SetShapeInfo(ShapeInfo(2, s1S2ShapeAlignArray, 2, s1S2ShapeArray, DataFormat::ND));
            // 持有dropMask不复用
            dropMaskUb = vecInQue1.template DeQue<uint8_t>();
            CalcDropout(vecOutUb1, bmm2ResUb, dropMaskUb);
            vecOutQue.EnQue(vecOutUb1);
            vecOutUb1 = vecOutQue.template DeQue<INPUT_T>();
            // 持有bmm2ResUb不复用
            CopyOutWorkspace(vecOutUb1, copyOut1Offset);
            vecOutQue.FreeTensor(vecOutUb1);
        } else {
            // 持有bmm2ResUb不复用
            CopyOutWorkspace(bmm2ResUb, copyOut1Offset);
        }
    }
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::SoftmaxGradComputeStage(
    LocalTensor<CALC_T> &softmaxGradResUb, LocalTensor<CALC_T>& bmm2ResUb, LocalTensor<uint8_t>& dropMaskUb) {
    /**
        *workspace2 处理过程：BMM1 -> dropout -> softmaxGrad -> Sub -> Mul
    */
    // bmm1ResUb tensor 申请一次，最后释放
    LocalTensor<CALC_T> bmm1ResUb = bmm1ResQue.template AllocTensor<CALC_T>();
    bmm1ResUb.SetShapeInfo(ShapeInfo(2, s1S2ShapeAlignArray, 2, s1S2ShapeArray, DataFormat::ND));
    bmm1ResUb.SetSize(actualS1S2AlignInnerSize);
    bmm1.template GetTensorC<false, true>(bmm1ResUb, false, false, actualS1InnerSize, actualS2InnerSize,
                                          mmPadSrcGap, mmPadDstGap);

    if (isDrop) {
        CalcDropout(bmm1ResUb, bmm1ResUb, dropMaskUb);
        pipe_barrier(PIPE_V);
    }
    vecInQue1.FreeTensor(dropMaskUb);

    if (doSoftmaxGrad) {
        if (notSplitGradD) {
            CopyInDyAttenIn(vecInQue1, attenInGm);
            CopyInDyAttenIn(vecInQue2, dyGm);
            CalcSoftmaxGrad(softmaxGradResUb, ordTilingData->baseParams.D, actualDAlignSize);
        } else {
            CalcSoftmaxGradWithSplitD(softmaxGradResUb);
        }
        doSoftmaxGrad = false;
    }

    pipe_barrier(PIPE_V);
    CalcSub(bmm1ResUb, bmm1ResUb, softmaxGradResUb);
    pipe_barrier(PIPE_V);

    if constexpr (HIGH_PRECISE) {
        LocalTensor<CALC_T> castedMulOutput = castBuff.template Get<CALC_T>(actualS1S2AlignInnerSize);
        castedMulOutput.SetSize(actualS1S2AlignInnerSize);
        castedMulOutput.SetShapeInfo(ShapeInfo(2, s1S2ShapeAlignArray, DataFormat::ND));
        CalcMul(castedMulOutput, bmm2ResUb, bmm1ResUb, actualS1S2AlignInnerSize);
        pipe_barrier(PIPE_V);
        CopyOutWorkspaceWithCast(castedMulOutput, copyOut2Offset);
    } else {
        LocalTensor<INPUT_T> vecOutUb2 = vecOutQue.template AllocTensor<INPUT_T>();
        vecOutUb2.SetSize(actualS1S2AlignInnerSize);
        vecOutUb2.SetShapeInfo(ShapeInfo(2, s1S2ShapeAlignArray, 2, s1S2ShapeArray, DataFormat::ND));
        CalcMul(vecOutUb2, bmm2ResUb, bmm1ResUb, actualS1S2AlignInnerSize);
        vecOutQue.EnQue(vecOutUb2);
        vecOutUb2 = vecOutQue.template DeQue<INPUT_T>();
        CopyOutWorkspace(vecOutUb2, copyOut2Offset);
        vecOutQue.FreeTensor(vecOutUb2);
    }
    bmm2ResQue.FreeTensor(bmm2ResUb);
    bmm1ResQue.FreeTensor(bmm1ResUb);
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::LastBmmStage() {
    // workspace1和workspace2的起始地址
    uint64_t w1Offset = spmBufOffset;
    uint64_t w2Offset = w1Offset + spmWorkspaceSize;
    // BMM4
    bmm4.SetTensorA(workspaceGm[w1Offset], true);
    bmm4.SetTensorB(dyGm[bnLoopS1Offset]);
    bmm4.SetTail(ordTilingData->baseParams.Skv, ordTilingData->baseParams.D, -1);
    if constexpr (HIGH_PRECISE) {
        bmm4.template IterateAll<true>(postOpWorkspaceDvGm[bnLoopS2Offset]);
    } else {
        bmm4.template IterateAll<true>(dvGm[bnLoopS2Offset]);
    }
    bmm4.End();

    // BMM3.1
    bmm3.SetTensorA(workspaceGm[w2Offset]);
    bmm3.SetTensorB(keyGm[bnLoopS2Offset]);
    bmm3.SetTail(ordTilingData->baseParams.S, ordTilingData->baseParams.D, -1);
    bmm3.template IterateAll<true>(postOpWorkspaceDqGm[bnLoopS1Offset]);
    bmm3.End();

    // BMM3.2
    bmm4.SetTensorA(workspaceGm[w2Offset], true);
    bmm4.SetTensorB(queryGm[bnLoopS1Offset]);
    bmm4.SetTail(ordTilingData->baseParams.Skv, ordTilingData->baseParams.D, -1);
    bmm4.template IterateAll<true>(postOpWorkspaceDkGm[bnLoopS2Offset]);
    bmm4.End();
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::LastBmmAsyncStage(
    uint32_t bnIdx) {
    bool sync = false; // 是否同步，最后一次设置为true，同步模式，其他设置为false，使用异步模式
    if (bnIdx == blockSplitBnRange - 1) {
        sync = true;
    }
    // workspace1和workspace2的起始地址
    uint64_t w1Offset = spmBufOffset;
    uint64_t w2Offset = w1Offset + spmWorkspaceSize;
    // BMM4
    bmm4.SetTensorA(workspaceGm[w1Offset], true);
    bmm4.SetTensorB(dyGm[bnLoopS1Offset]);
    bmm4.SetTail(ordTilingData->baseParams.Skv, ordTilingData->baseParams.D, -1);
    if constexpr (HIGH_PRECISE) {
        if (sync) {
            bmm4.template IterateAll<true>(postOpWorkspaceDvGm[bnLoopS2Offset]);
        } else {
            bmm4.template IterateAll<false>(postOpWorkspaceDvGm[bnLoopS2Offset]);
        }
    } else {
        if (sync) {
            bmm4.template IterateAll<true>(dvGm[bnLoopS2Offset]);
        } else {
            bmm4.template IterateAll<false>(dvGm[bnLoopS2Offset]);
        }
    }

    // BMM3.1
    bmm3.SetTensorA(workspaceGm[w2Offset]);
    bmm3.SetTensorB(keyGm[bnLoopS2Offset]);
    bmm3.SetTail(ordTilingData->baseParams.S, ordTilingData->baseParams.D, -1);
    if (sync) {
        bmm3.template IterateAll<true>(postOpWorkspaceDqGm[bnLoopS1Offset]);
    } else {
        bmm3.template IterateAll<false>(postOpWorkspaceDqGm[bnLoopS1Offset]);
    }

    // BMM3.2
    bmm4.SetTensorA(workspaceGm[w2Offset], true);
    bmm4.SetTensorB(queryGm[bnLoopS1Offset]);
    bmm4.SetTail(ordTilingData->baseParams.Skv, ordTilingData->baseParams.D, -1);
    if (sync) {
        bmm4.template IterateAll<true>(postOpWorkspaceDkGm[bnLoopS2Offset]);
    } else {
        bmm4.template IterateAll<false>(postOpWorkspaceDkGm[bnLoopS2Offset]);
    }
}

template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::ClearSparseBlockGm() {
    INPUT_T valueZero(0);
    LocalTensor<INPUT_T> vecOutUb = vecOutQue.template AllocTensor<INPUT_T>();
    vecOutUb.SetSize(actualS1S2AlignInnerSize);
    Duplicate<INPUT_T>(vecOutUb, valueZero, actualS1S2AlignInnerSize);
    vecOutQue.EnQue(vecOutUb);
    vecOutQue.DeQue<INPUT_T>();
    CopyOutWorkspace(vecOutUb, copyOut1Offset);
    CopyOutWorkspace(vecOutUb, copyOut2Offset);
    vecOutQue.FreeTensor(vecOutUb);
}


template <typename INPUT_T, typename CALC_T, const bool HIGH_PRECISE>
__aicore__ inline void SparseFlashAttentionScoreGradUS1S2BBN<INPUT_T, CALC_T, HIGH_PRECISE>::Process() {
    if (g_coreType == AIV && mBlockIdx >= usedCoreNum) {
        LocalTensor<int32_t> workLocal = vecInQue1.AllocTensor<int32_t>();
        SyncAll<false>(syncGlobal, workLocal);
        vecInQue1.FreeTensor(workLocal);
        return;
    }

    // init params
    InitCalcParams();

    // bn split for block
    for (uint32_t bnIdx = 0; bnIdx < blockSplitBnRange; bnIdx++) {
        uint32_t bIdx = (beforeBlockSplitBnNums + bnIdx) / ordTilingData->baseParams.N;
        // 计算BN方向的offset
        CalcBNOffset(bnIdx);

        // 更新actualCubeBaseS1
        actualCubeBaseS1 = cubeBaseS1;
        // cube tiling s1
        for (uint32_t s1IdxC = 0; s1IdxC < s1RangeC; s1IdxC++) {
            // 计算S1方向的cube tiling的offset和loop参数
            CalcForwardBMMS1Offset(s1IdxC);

            // sparse计算, 用于计算s2方向配比基本块后的实际loop次数
            uint32_t s2LoopStartIdxC = 0;
            uint32_t s2LoopEndIdxC = s2RangeC;
            if (ordTilingData->splitCoreParams.sparse == 1) {
                CalcSparseIdx(bIdx, s1IdxC, s2LoopStartIdxC, s2LoopEndIdxC, cubeBaseS1, cubeBaseS2, s2LoopStartIdxC, s2LoopEndIdxC);
            }

            // cube tiling s1

            for (uint32_t s2IdxC = 0; s2IdxC < s2RangeC; s2IdxC++) {
                // sparse场景下，s2方向按小粒度基本块发射，减少重复计算
                uint32_t emitS2StartIdx = s2IdxC * s2CVRatio;
                uint32_t emitS2EndIdx = emitS2StartIdx + s2CVRatio > s2RangeV ? s2RangeV : emitS2StartIdx + s2CVRatio;
                if (ordTilingData->splitCoreParams.sparse == 1) {
                    CalcSparseIdx(bIdx, s1IdxC, emitS2StartIdx, emitS2EndIdx, cubeBaseS1, s2InnerSize, emitS2StartIdx, emitS2EndIdx);
                }
                // 前向bmm1/2发射，发射actualCubeBaseS1*cubeBaseS2的大小。从第二行开始跳过第一块block，其余在此发射。
                if ((s1IdxC == 0 && s2IdxC >= s2LoopStartIdxC && s2IdxC < s2LoopEndIdxC) || (s2IdxC > s2LoopStartIdxC && s2IdxC < s2LoopEndIdxC)) {
                    // s1IdxC=0时，发射一次，其余的在s2for循环的末尾发射，减少下一次循环的数据等待时间，这里暂不考虑s2IdxC的提前发射
                    forwardBmmEmit(emitS2StartIdx, emitS2EndIdx);
                }

                // 更新actualS1InnerSize，尾块的loop会刷新
                actualS1InnerSize = s1InnerSize;
                // CV tiling s1
                uint32_t s1LoopEndIdxV = s1RangeV - s1IdxC * s1CVRatio;
                if (s1LoopEndIdxV > s1CVRatio) {
                    s1LoopEndIdxV = s1CVRatio;
                }
                for (uint32_t s1IdxV = 0; s1IdxV < s1LoopEndIdxV; s1IdxV++) {
                    uint32_t s1Idx = s1IdxC * s1CVRatio + s1IdxV;

                    // 计算s1/s2方向实际有效的vector base块idx，用于减少sparse场景下的vector的重复计算
                    uint32_t s2LoopStartIdx = 0;
                    uint32_t s2LoopEndIdx = s2RangeV;
                    if (ordTilingData->splitCoreParams.sparse == 1) {
                        CalcSparseIdx(bIdx, s1Idx, s2LoopStartIdx, s2LoopEndIdx, s1InnerSize, s2InnerSize, s2LoopStartIdx, s2LoopEndIdx);
                    }

                    // 计算S1方向的offset和loop参数
                    CalcS1OffsetLoopParams(s1Idx);

                    // 更新actualS2InnerSize，尾块的loop会刷新
                    actualS2InnerSize = s2InnerSize;
                    mmPadDstGap = 0;
                    // 一次s2循环中仅计算一次softmaxGrad，减少重复计算
                    doSoftmaxGrad = true;
                    LocalTensor<CALC_T> softmaxGradResUb = softmaxGradResBuf.template Get<CALC_T>(
                        actualS1InnerSize * blockCalcTypeNum);
                    uint32_t softmaxShape[] = {actualS1InnerSize, blockCalcTypeNum};
                    softmaxGradResUb.SetShapeInfo(ShapeInfo(2, softmaxShape, DataFormat::ND));
                    for (uint32_t s2IdxV = 0; s2IdxV < s2CVRatio; s2IdxV++) {
                        uint32_t s2Idx = s2IdxC * s2CVRatio + s2IdxV;
                        if (s2Idx >= s2RangeV) {
                            break;
                        }
                        // 计算Vector S2方向offset和loop参数
                        CalcS2OffsetLoopParams(s1Idx, s2Idx);

                        if (s2Idx < s2LoopStartIdx || s2Idx >= s2LoopEndIdx) {
                            bool skipGetBMMResult = s2Idx >= emitS2StartIdx && s2Idx < emitS2EndIdx &&
                                                    s2IdxC >= s2LoopStartIdxC && s2IdxC < s2LoopEndIdxC;
                            if (skipGetBMMResult) {
                                // 发射大块配比基本块，可能出现cube多算，vector不算的情况，为不影响其他小块的getC，这里需要跳过多余块
                                bmm2.template GetTensorC<false>(false, false);
                                bmm1.template GetTensorC<false>(false, false);
                            }
                            if (0 == bnIdx) {
                                ClearSparseBlockGm();
                            }
                            continue;
                        }

                        // 左路softmaxGrad计算流
                        LocalTensor<CALC_T> bmm2ResUb;
                        LocalTensor<uint8_t> dropMaskUb; // dropMask持有不释放，右路不重复搬运

                        SimpleSoftmaxComputeStage(bmm2ResUb, dropMaskUb);

                        // 提前发射下一次s2IdxC的BMM2
                        uint64_t nextSparseS2Offset = 0;
                        uint32_t nextCurS2 = 0;
                        // 在s1IdxC内部所有循环的最后一次，发射下一次s1IdxC；这里没有考虑s2IdxC的提前发射
                        bool doNextEmit = s1IdxC < s1RangeC - 1 && s2IdxC == s2LoopEndIdxC - 1 &&
                                          s1IdxV == s1LoopEndIdxV - 1 && s2Idx == emitS2EndIdx - 1;
                         if (doNextEmit) {
                            CalcForwardBmmEmitParams(bIdx, s1IdxC + 1, nextSparseS2Offset, nextCurS2);
                            forwardBmm2Emit(s1IdxC + 1, nextSparseS2Offset, nextCurS2);
                        }

                        // 右路softmaxGrad计算流
                        SoftmaxGradComputeStage(softmaxGradResUb, bmm2ResUb, dropMaskUb);

                        // 提前发射下一次s2IdxC的BMM1
                        if (doNextEmit) {
                            forwardBmm1Emit(s1IdxC + 1, nextSparseS2Offset, nextCurS2);
                        }
                    }
                }
            }
            bmm1.End();
            bmm2.End();
        }
        // 最后的bmm4 bmm3.1 bmm3.2的处理过程
        LastBmmAsyncStage(bnIdx);
    }
    // post scalar
    LocalTensor<int32_t> workLocal = vecInQue1.AllocTensor<int32_t>();
    SyncAll<false>(syncGlobal, workLocal);
    vecInQue1.FreeTensor(workLocal);
}
#endif
