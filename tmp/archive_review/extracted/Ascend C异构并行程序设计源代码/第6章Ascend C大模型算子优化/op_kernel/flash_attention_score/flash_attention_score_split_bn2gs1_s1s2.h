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

#ifndef FLASH_ATTENTION_SCORE_SPLIT_B_N2_G_S1_S2_H
#define FLASH_ATTENTION_SCORE_SPLIT_B_N2_G_S1_S2_H
#include "flash_attention_score_common.h"
#include "kernel_data_copy_transpose.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

using matmul::MatmulType;

// INPUT_T - means data type for input
// T       - means data type when calc
template <typename INPUT_T, typename T = INPUT_T, bool isBasicBlock = false>
class FlashAttentionScoreSplitBn2gs1s2S1s2 : public FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock> {
public:
    __aicore__ inline FlashAttentionScoreSplitBn2gs1s2S1s2() {};

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
        __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax,
        __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
        const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

    // define matmul
    using a1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T>;
    using b1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, true>;
    using bias1Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c1Type = MatmulType<TPosition::VECCALC, CubeFormat::ND, T>;
    matmul::Matmul<a1Type, b1Type, c1Type, bias1Type, CFG_EXCEED> bmm1;

    // define batchmatmul
    using a2Type = MatmulType<TPosition::VECCALC, CubeFormat::ND, INPUT_T>;
    using b2Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T>;
    using bias2Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c2Type = MatmulType<TPosition::VECCALC, CubeFormat::ND_ALIGN, T>;
    matmul::Matmul<a2Type, b2Type, c2Type, bias2Type, CFG_EXCEED> bmm2;

protected:
    __aicore__ inline void InitInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
        __gm__ uint8_t *pse, __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask,
        __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut,
        __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
        const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void SetTiling(const FlashAttentionScoreGeneralTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffer();
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void ComputeAxisIdx(int64_t multiCoreInnerIdx);
    __aicore__ inline void PreProcessBlock();
    __aicore__ inline void SetSoftmaxTensorShape();
    __aicore__ inline void ProcessBlock();
    __aicore__ inline void Bmm1SetTensorA();
    __aicore__ inline void ComputeBmm1S2Tail(int64_t s2oIdx);
    __aicore__ inline void SetBmm1TensorB(int64_t s2oIdx);
    __aicore__ inline void ProcessStage1(int64_t s2LoopLimit);
    __aicore__ inline void ComputeS2TailSize();
    __aicore__ inline void CopyInAndCastPse(LocalTensor<T> &castTensor);
    __aicore__ inline int64_t ComputePseOffset();
    __aicore__ inline void CopyInAttenMask();
    __aicore__ inline int64_t ComputeAttenMaskOffset();
    __aicore__ inline int64_t ComputeOffsetForNoCompress();
    __aicore__ inline void GetBmm1Result(LocalTensor<T> &bmm1ResUb);
    __aicore__ inline void ComputePse(LocalTensor<T> &pseUb, LocalTensor<T> &bmm1ResUb);
    __aicore__ inline void CastAndComputeAttenMask(LocalTensor<T> &castUb, LocalTensor<T> &bmm1ResUb);
    __aicore__ inline void CopyInDropMask();
    __aicore__ inline int64_t ComputeDropOffset();
    __aicore__ inline void SoftMaxCompute(LocalTensor<T> &srcTensor, int64_t s2LoopLimit);
    __aicore__ inline void ComputeDropMask(LocalTensor<T> &bmm1ResUb);
    __aicore__ inline void ProcessStage2();
    __aicore__ inline void Bmm2ResultMul(LocalTensor<T> &bmm2ResUb);
    __aicore__ inline void Bmm2SetTensor(int64_t s2Size);
    __aicore__ inline void Bmm2ResultDiv();
    __aicore__ inline void Bmm2DataCopyOut();
    __aicore__ inline void SoftmaxDataCopyOut();

    // 构建dataCopyTranspose参数
    CopyTransposeTiling dataCopyTiling;

    uint32_t s1BaseSize;
    uint32_t s2BaseSize;
    uint32_t dSize;
    uint32_t alignDSize;
    uint32_t padDSize;

    // sparse 用函数
    __aicore__ inline void GetS1LoopRange(int64_t &multiCoreInnerOffset, int64_t &multiCoreInnerLimit);
    __aicore__ inline int64_t GetS2LoopRange();

    // sparse 用参数
    int64_t s2oStartIdx;
    int64_t s2oEndIdx;
    int64_t s2LoopStart;

    // s1方向的尾块
    uint32_t s1RealSize;

    // s2方向的尾块，包含N:1配比
    int32_t s2RealSize;
    int32_t s2RealSplitN;

    // s2方向的尾块，不包含N:1的配比
    int32_t s2TailSize;

    // k和v矩阵在核间的offset
    int64_t kCoreOffset;

    int64_t qCoreOffset;

    int64_t stage2WorkspaceGm1Offset;
    int64_t stage2WorkspaceGm2Offset;

    int64_t commonTBuf1Offset;
    int64_t commonTBuf2Offset;

    // s2轴循环计数
    int64_t s2LoopCount;
    int64_t s2RealLoopCount; // sparse模式下实际执行到的s2个数

    // 资源分配
    TQue<QuePosition::VECOUT, 1> bmm2OutQueue;
    TBuf<> stage1TBuf;     // 存stage1中间结果
    TBuf<> stage2TBuf;     // 存stage2中间结果
    TBuf<> softmaxExpTBuf; // 存softmax exp中间结果
    TBuf<> commonTBuf;     // common的复用空间

    LocalTensor<T> softmaxExpUb;
    GlobalTensor<T> mmWorkspaceGm;
    GlobalTensor<INPUT_T> stage2WorkspaceGm;

    // 轴的乘积
    int64_t gS1o;
    int64_t n2GS1o;
    int64_t s1D;
    int64_t gS1D;
    int64_t n2GS1D;
    int64_t s2D;
    int64_t n2S2D;
    int64_t s1S2;
    int64_t gS1S2;
    int64_t n2GS1S2;
    int64_t gS1;
    int64_t n2GS1;
    int64_t gD;
    int64_t n2D;
    int64_t n2G;
    int64_t n2GD;
    int64_t gS2;

    // s2base*N之后的长度
    int64_t s2BaseNratioSize;
    int64_t s1BaseS2;

    int64_t s2BaseN2D;
    int64_t s2BaseBN2D;
    int64_t s1BaseN2GD;
    int64_t s1BaseBN2GD;
    int64_t s1BaseD;
    int64_t s1BaseBN2D;
    int64_t s2BaseNratioD;
    int64_t s2BaseNratioN2D;
    int64_t s2BaseNratioBN2D;
    int64_t bN2G;
    int64_t n2GS2;

    // layoutType
    int64_t layoutBsh = 1;
    int64_t layoutSbh = 2;
    int64_t layoutBnsd = 3;
};

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::Init(__gm__ uint8_t *query,
    __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse, __gm__ uint8_t *dropMask,
    __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
    __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe)
{
    this->InitInput(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
        attentionOut, workspace, tiling, tPipe); // gm设置

    this->InitBuffer();
    this->ComputeConstexpr();

    LocalTensor<T> apiTmpBuffer = this->commonTBuf.template Get<T>();
    DropOutBitModeInit(apiTmpBuffer);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::InitInput(__gm__ uint8_t *query,
    __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse, __gm__ uint8_t *dropMask,
    __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
    __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe)
{
    this->blockIdx = GetBlockIdx();
    this->pipe = tPipe;
    this->repeatMaxSize = this->repeatMaxBytes / sizeof(T);
    this->SetTiling(tiling);

    // init global buffer
    this->queryGm.SetGlobalBuffer((__gm__ INPUT_T *)query);
    this->keyGm.SetGlobalBuffer((__gm__ INPUT_T *)key);
    this->valueGm.SetGlobalBuffer((__gm__ INPUT_T *)value);
    this->pseGm.SetGlobalBuffer((__gm__ INPUT_T *)pse);
    this->dropMaskGm.SetGlobalBuffer((__gm__ uint8_t *)dropMask);
    this->attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
    this->softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
    this->softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);
    this->attentionOutGm.SetGlobalBuffer((__gm__ INPUT_T *)attentionOut);

    this->mmWorkspaceGm.SetGlobalBuffer((__gm__ T *)workspace);

    int64_t bmm1ResultWorkspaceSize = s1BaseSize * s2BaseSize;
    int64_t bmm1WorkspaceSize = bmm1ResultWorkspaceSize * this->tilingData->coreParams.nRatio;
    int64_t bmm2WorkspaceSize = s1BaseSize * dSize;
    int64_t mmWorkspaceBaseSize = bmm1WorkspaceSize + bmm2WorkspaceSize;
    int64_t workspaceOffset = GetBlockNum() * GetTaskRation() * mmWorkspaceBaseSize * sizeof(T);

    this->stage2WorkspaceGm.SetGlobalBuffer((__gm__ INPUT_T *)workspace + workspaceOffset);

    this->hasPse = this->pseGm.GetPhyAddr() != nullptr;
    this->hasAttenMask = this->attenMaskGmInt.GetPhyAddr() != nullptr;
    this->hasDropMask = this->dropMaskGm.GetPhyAddr() != nullptr;
    this->isInputTAttenMask = this->tilingData->inputParams.attenMaskDataType == 0;

    this->bmm1.SetWorkspace((__gm__ uint8_t *)this->mmWorkspaceGm[this->blockIdx * mmWorkspaceBaseSize].GetPhyAddr(),
        bmm1WorkspaceSize);

    this->bmm2.SetWorkspace(
        (__gm__ uint8_t *)this->mmWorkspaceGm[this->blockIdx * mmWorkspaceBaseSize + bmm1WorkspaceSize].GetPhyAddr(),
        bmm2WorkspaceSize);

    this->stage2WorkspaceGm1Offset = this->blockIdx * bmm1ResultWorkspaceSize * 2;
    this->stage2WorkspaceGm2Offset = this->blockIdx * bmm1ResultWorkspaceSize * 2 + bmm1ResultWorkspaceSize;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::SetTiling(
    const FlashAttentionScoreGeneralTilingData *__restrict tilingData)
{
    // copy base params
    this->tilingData = tilingData;
    this->s1BaseSize = this->tilingData->coreParams.s1BaseSize;
    this->s2BaseSize = this->tilingData->coreParams.s2BaseSize;
    this->dSize = this->tilingData->inputParams.dSize;
    this->alignDSize = this->CeilDiv(this->tilingData->inputParams.dSize, 16) * 16;
    this->padDSize = (this->alignDSize - dSize) * 4 / 32;
};

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::InitBuffer()
{
    this->pipe->InitBuffer(this->optionalInputQueue, 1, s1BaseSize * s2BaseSize * sizeof(INPUT_T));
    this->pipe->InitBuffer(this->bmm2OutQueue, 1, s1BaseSize * s2BaseSize * sizeof(INPUT_T));
    this->pipe->InitBuffer(this->softmaxOutQueue, 2, s1BaseSize * this->blockBytes);
    this->pipe->InitBuffer(this->stage1TBuf, s1BaseSize * s2BaseSize * sizeof(T));
    this->pipe->InitBuffer(this->stage2TBuf, s1BaseSize * s2BaseSize * sizeof(T));
    this->pipe->InitBuffer(this->softmaxExpTBuf, s1BaseSize * this->blockBytes);

    int64_t commonBufSize = s1BaseSize * s2BaseSize;
    this->pipe->InitBuffer(this->commonTBuf, commonBufSize * 2 * sizeof(T));
    this->commonTBuf1Offset = 0;
    this->commonTBuf2Offset = commonBufSize;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::ComputeConstexpr()
{
    // 计算轴的乘积
    this->s1D = this->tilingData->inputParams.s1Size * dSize;
    this->s2D = this->tilingData->inputParams.s2Size * dSize;
    this->gD = this->tilingData->inputParams.gSize * dSize;
    this->n2D = this->tilingData->inputParams.n2Size * dSize;
    this->s1S2 = this->tilingData->inputParams.s1Size * this->tilingData->inputParams.s2Size;
    this->gS1 = this->tilingData->inputParams.gSize * this->tilingData->inputParams.s1Size;
    this->n2G = this->tilingData->inputParams.n2Size * this->tilingData->inputParams.gSize;
    this->gS1o = this->tilingData->inputParams.gSize * this->tilingData->coreParams.s1OuterSize;

    this->n2GS1o = this->tilingData->inputParams.n2Size * this->gS1o;
    this->gS1D = this->tilingData->inputParams.gSize * this->s1D;
    this->n2S2D = this->tilingData->inputParams.n2Size * this->s2D;
    this->n2GD = this->tilingData->inputParams.n2Size * this->gD;
    this->gS1S2 = this->tilingData->inputParams.gSize * this->s1S2;
    this->n2GS1 = this->tilingData->inputParams.n2Size * this->gS1;

    this->n2GS1D = this->tilingData->inputParams.n2Size * this->gS1D;
    this->n2GS1S2 = this->tilingData->inputParams.n2Size * this->gS1S2;

    // 计算切分轴的乘积
    this->s2BaseN2D = s2BaseSize * this->n2D;
    this->s2BaseNratioSize = s2BaseSize * this->tilingData->coreParams.nRatio;
    this->s1BaseS2 = s1BaseSize * this->tilingData->inputParams.s2Size;

    if (this->tilingData->inputParams.layoutType == this->layoutBsh) {
        // BSH/BSNGD
        this->s1BaseN2GD = s1BaseSize * this->n2GD;
        this->s2BaseNratioN2D = this->s2BaseN2D * this->tilingData->coreParams.nRatio;
    } else if (this->tilingData->inputParams.layoutType == this->layoutSbh) {
        // SBH/SBNGD
        this->bN2G = this->tilingData->inputParams.bSize * this->n2G;
        this->s1BaseBN2GD = s1BaseSize * this->tilingData->inputParams.bSize * this->n2GD;
        this->s2BaseBN2D = this->tilingData->inputParams.bSize * this->s2BaseN2D;
        this->s2BaseNratioBN2D = this->s2BaseBN2D * this->tilingData->coreParams.nRatio;
    } else if (this->tilingData->inputParams.layoutType == this->layoutBnsd) {
        // BNSD
        this->s1BaseD = s1BaseSize * dSize;
        this->s2BaseNratioD = this->s2BaseNratioSize * dSize;
    }

    if (this->tilingData->inputParams.pseShapeType == this->pse1S2) {
        this->gS2 = this->tilingData->inputParams.gSize * this->tilingData->inputParams.s2Size;
        this->n2GS2 = this->tilingData->inputParams.n2Size * this->gS2;
    }

    dataCopyTiling.dstShapeB = this->tilingData->transposeTilingData.dstShapeB;
    dataCopyTiling.dstShapeN = this->tilingData->transposeTilingData.dstShapeN;
    dataCopyTiling.dstShapeS = this->tilingData->transposeTilingData.dstShapeS;
    dataCopyTiling.dstShapeHN = this->tilingData->transposeTilingData.dstShapeHN;
    dataCopyTiling.dstShapeH = this->tilingData->transposeTilingData.dstShapeH;

    dataCopyTiling.srcShapeB = this->tilingData->transposeTilingData.srcShapeB;
    dataCopyTiling.srcShapeN = this->tilingData->transposeTilingData.srcShapeN;
    dataCopyTiling.srcShapeHN = this->tilingData->transposeTilingData.srcShapeHN;
    dataCopyTiling.originalShapeNLen = this->tilingData->transposeTilingData.originalShapeNLen;
    dataCopyTiling.shapeSHValue = this->tilingData->transposeTilingData.shapeSHValue;
    dataCopyTiling.shapeNsValue = this->tilingData->transposeTilingData.shapeNsValue;
    dataCopyTiling.shapeNsnValue = this->tilingData->transposeTilingData.shapeNsnValue;
    dataCopyTiling.shapeBHValue = this->tilingData->transposeTilingData.shapeBHValue;
}

// sparse functions
template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::GetS1LoopRange(
    int64_t &multiCoreInnerOffset, int64_t &multiCoreInnerLimit)
{
    // 计算sparse场景下s1的循环范围
    if (this->tilingData->inputParams.sparseType > 0) {
        // sparse场景下负载均衡后每个核获取的结果
        multiCoreInnerOffset = this->tilingData->multiCoreParams.sparseStartIdx[this->blockIdx];
        if (likely((this->tilingData->multiCoreParams.coreNum - 1) > this->blockIdx)) {
            multiCoreInnerLimit = this->tilingData->multiCoreParams.sparseStartIdx[this->blockIdx + 1];
        } else {
            multiCoreInnerLimit = this->tilingData->multiCoreParams.totalSize;
        }
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline int64_t FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::GetS2LoopRange()
{
    // this->s2oEndIdx是S2循环的上限（循环变量最大到this->s2oEndIdx - 1），所以this->s2oEndIdx需要+1
    if (this->tilingData->inputParams.sparseType == 3 ||
        this->tilingData->inputParams.sparseType == 5) { // 3:下三角, 5: prefix
        int64_t s2IgnoredEndLen = this->tilingData->inputParams.s1Size -
                                  this->tilingData->coreParams.s1BaseSize * (this->s1oIdx + 1);
        s2IgnoredEndLen = this->Max((int64_t)0, s2IgnoredEndLen);
        int64_t s2EndLen = 0;
        if (this->tilingData->inputParams.s2Size > s2IgnoredEndLen) {
            s2EndLen = this->tilingData->inputParams.s2Size - s2IgnoredEndLen;
        } else {
            s2EndLen = this->tilingData->inputParams.s2Size;
        }

        if (this->tilingData->inputParams.sparseType == 5) {
            s2EndLen = this->Max(s2EndLen, this->tilingData->inputParams.prefixNList[this->boIdx]);
        }
        this->s2oEndIdx = this->CeilDiv(s2EndLen,
                                        this->tilingData->coreParams.s2BaseSize * this->tilingData->coreParams.nRatio);
        this->s2oEndIdx = this->Min(this->s2oEndIdx, this->tilingData->coreParams.s2OuterSize);
        this->s2oStartIdx = 0;
    } else if (this->tilingData->inputParams.sparseType == 4) {
        int64_t idx = this->Max(this->tilingData->coreParams.s1SparseValidSize,
                                this->s1oIdx * this->tilingData->coreParams.s1BaseSize);
        this->s2oStartIdx = (idx - this->tilingData->coreParams.s1SparseValidSize) /
                            (this->tilingData->coreParams.s2BaseSize * this->tilingData->coreParams.nRatio);
        // idx 表示s2方向的结束结果
        idx = this->tilingData->coreParams.s2SparseValidSize +
              (this->s1oIdx + 1) * this->tilingData->coreParams.s1BaseSize;

        idx = Ceil(idx, (this->tilingData->coreParams.s2BaseSize * this->tilingData->coreParams.nRatio));
        this->s2oEndIdx = this->Min(idx, this->tilingData->coreParams.s2OuterSize);

        int64_t s2IgnoredStartLen = this->tilingData->coreParams.s1BaseSize * this->s1oIdx -
                                    this->tilingData->inputParams.preTokens;
        s2IgnoredStartLen = this->Max(s2IgnoredStartLen, 0);
        this->s2oStartIdx = s2IgnoredStartLen /
                            (this->tilingData->coreParams.s2BaseSize * this->tilingData->coreParams.nRatio);
    } else {
        this->s2oStartIdx = 0;
        this->s2oEndIdx = this->tilingData->coreParams.s2OuterSize;
    }

    /* s2LoopLimit表示S2方向总的循环次数 */
    int64_t s2LoopLimit = this->Min(this->tilingData->inputParams.s2Size,
                                    this->s2oEndIdx * this->tilingData->coreParams.s2BaseSize *
                                    this->tilingData->coreParams.nRatio);
    if (this->tilingData->inputParams.sparseType == 3) { // 下三角 S1 = S2特殊场景
        s2LoopLimit = this->Min(s2LoopLimit, (this->s1oIdx + 1) * this->tilingData->coreParams.s1BaseSize + 1);
    }
    s2LoopLimit -= this->s2oStartIdx * this->tilingData->coreParams.s2BaseSize * this->tilingData->coreParams.nRatio;
    s2LoopLimit = this->CeilDiv(s2LoopLimit, this->tilingData->coreParams.s2BaseSize) - 1;
    return s2LoopLimit;
}


template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::Process()
{
    // 确定核内切分起点
    int64_t multiCoreInnerOffset = this->blockIdx * this->tilingData->multiCoreParams.splitFactorSize;
    int64_t multiCoreInnerLimit = multiCoreInnerOffset + this->tilingData->multiCoreParams.splitFactorSize;
    if (this->tilingData->multiCoreParams.totalSize < multiCoreInnerLimit) {
        multiCoreInnerLimit = this->tilingData->multiCoreParams.totalSize;
    }
    // 计算sparse场景下s1的循环范围
    this->GetS1LoopRange(multiCoreInnerOffset, multiCoreInnerLimit);
    for (int64_t multiCoreInnerIdx = multiCoreInnerOffset; multiCoreInnerIdx < multiCoreInnerLimit;
        multiCoreInnerIdx++) {
        this->ComputeAxisIdx(multiCoreInnerIdx);
        this->PreProcessBlock();
        this->ProcessBlock();
    }
};

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::ComputeAxisIdx(
    int64_t multiCoreInnerIdx)
{
    this->boIdx = multiCoreInnerIdx / this->n2GS1o;
    this->n2oIdx = multiCoreInnerIdx % this->n2GS1o / this->gS1o;
    this->goIdx = multiCoreInnerIdx % this->gS1o / this->tilingData->coreParams.s1OuterSize;
    this->s1oIdx = multiCoreInnerIdx % this->tilingData->coreParams.s1OuterSize;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::PreProcessBlock()
{
    // 计算s1方向的尾块
    this->s1RealSize = s1BaseSize;
    uint32_t s1Reamin = this->tilingData->inputParams.s1Size - this->s1oIdx * s1BaseSize;
    if (unlikely(s1Reamin > 0 && s1Reamin < this->s1RealSize)) {
        this->s1RealSize = s1Reamin;
    }

    this->SetSoftmaxTensorShape();
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::SetSoftmaxTensorShape()
{
    uint32_t maxSumShape[] = {static_cast<uint32_t>(s1RealSize), static_cast<uint32_t>(this->fp32BaseSize)};
    this->softmaxSumUb = this->softmaxOutQueue.template AllocTensor<float>();
    this->softmaxMaxUb = this->softmaxOutQueue.template AllocTensor<float>();

    this->softmaxMaxUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));
    this->softmaxSumUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));

    uint32_t expShape[] = {static_cast<uint32_t>(s1RealSize), static_cast<uint32_t>(this->blockBytes / sizeof(T))};
    softmaxExpUb = softmaxExpTBuf.Get<T>(s1RealSize * this->blockBytes / sizeof(T));
    softmaxExpUb.SetShapeInfo(ShapeInfo(2, expShape, DataFormat::ND));
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::ProcessBlock()
{
    // BN2GS1o开多核场景，b,n2,g对应的OuterSize等于实际shape， s1需要判断尾块
    this->Bmm1SetTensorA();

    // s2轴循环计数, 支持sparse和非sparse场景
    int64_t s2LoopLimit = this->GetS2LoopRange();
    this->s2LoopStart = this->s2oStartIdx * this->tilingData->coreParams.nRatio;
    this->s2LoopCount = this->s2LoopStart;
    this->s2RealLoopCount = 0;
    s2LoopLimit += this->s2LoopStart;

    for (int64_t s2oIdx = this->s2oStartIdx; s2oIdx < this->s2oEndIdx; s2oIdx++) {
        this->ComputeBmm1S2Tail(s2oIdx);
        this->SetBmm1TensorB(s2oIdx);

        this->bmm1.template Iterate<false>();

        // n:1配比计算
        for (int64_t splitNIdx = 0; splitNIdx < this->s2RealSplitN; splitNIdx++) {
            this->ProcessStage1(s2LoopLimit);

            this->ProcessStage2();
            this->s2LoopCount++;
            this->s2RealLoopCount++;
        }
        this->bmm1.End();
    }

    // 尾块计算需要使用实际s2
    this->Bmm2SetTensor(this->s2TailSize);
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    set_flag(PIPE_MTE3, PIPE_S, eventID);
    wait_flag(PIPE_MTE3, PIPE_S, eventID);

    this->bmm2.template Iterate<false>();
    LocalTensor<T> stage2BufTensor = this->commonTBuf.template Get<T>()[this->commonTBuf2Offset];

    this->bmm2.template GetTensorC<false, true>(stage2BufTensor, false, false, s1RealSize, dSize, 0, padDSize);
    this->bmm2.End();
    LocalTensor<T> bmm2ResUb = this->stage2TBuf.template Get<T>();
    if (unlikely(this->s2RealLoopCount == 1)) {
        DataCopy(bmm2ResUb, stage2BufTensor, s1RealSize * alignDSize);
    } else {
        Add(bmm2ResUb, bmm2ResUb, stage2BufTensor, this->s1RealSize * alignDSize);
    }
    Bmm2ResultDiv();
    Bmm2DataCopyOut();
    SoftmaxDataCopyOut();
};

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::Bmm1SetTensorA()
{
    // 计算gm上的offset
    int64_t bOffset = 0;

    // s1需要考虑inner轴的影响
    int64_t s1Offset = 0;

    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    if (this->tilingData->inputParams.layoutType == this->layoutBsh) {
        // BSH/BSNGD
        bOffset = this->boIdx * this->n2GS1D;
        s1Offset = this->s1oIdx * this->s1BaseN2GD;
        n2Offset = this->n2oIdx * this->gD;
        gOffset = this->goIdx * dSize;
    } else if (this->tilingData->inputParams.layoutType == this->layoutSbh) {
        // SBH/SBNGD
        s1Offset = this->s1oIdx * this->s1BaseBN2GD;
        bOffset = this->boIdx * this->n2GD;
        n2Offset = this->n2oIdx * this->gD;
        gOffset = this->goIdx * dSize;
    } else if (this->tilingData->inputParams.layoutType == this->layoutBnsd) {
        // bnsd
        bOffset = this->boIdx * this->n2GS1D;
        n2Offset = this->n2oIdx * this->gS1D;
        gOffset = this->goIdx * this->s1D;
        s1Offset = this->s1oIdx * this->s1BaseD;
    }
    this->qCoreOffset = bOffset + n2Offset + gOffset + s1Offset;
    this->bmm1.SetTensorA(this->queryGm[this->qCoreOffset]);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::ComputeBmm1S2Tail(int64_t s2oIdx)
{
    // 计算s2方向的N:1切分的尾块
    this->s2RealSplitN = this->tilingData->coreParams.nRatio;
    this->s2RealSize = this->s2BaseNratioSize;
    if ((s2oIdx + 1) * s2RealSize > this->tilingData->inputParams.s2Size) {
        this->s2RealSize = this->tilingData->inputParams.s2Size - s2oIdx * s2RealSize;
        this->s2RealSplitN = this->CeilDiv(s2RealSize, s2BaseSize);
    }
    return;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::SetBmm1TensorB(int64_t s2oIdx)
{
    // 计算gm上的offset
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;
    if (this->tilingData->inputParams.layoutType == this->layoutBsh) {
        // BSH/BSND
        bOffset = this->boIdx * this->n2S2D;
        s2Offset = s2oIdx * this->s2BaseNratioN2D;
        n2Offset = this->n2oIdx * dSize;
    } else if (this->tilingData->inputParams.layoutType == this->layoutSbh) {
        // SBH/SBND
        s2Offset = s2oIdx * this->s2BaseNratioBN2D;
        bOffset = this->boIdx * this->n2D;
        n2Offset = this->n2oIdx * dSize;
    } else if (this->tilingData->inputParams.layoutType == this->layoutBnsd) {
        // BNSD
        bOffset = this->boIdx * this->n2S2D;
        n2Offset = this->n2oIdx * this->s2D;
        s2Offset = s2oIdx * this->s2BaseNratioD;
    }
    this->kCoreOffset = bOffset + n2Offset + s2Offset;
    this->bmm1.SetTensorB(this->keyGm[this->kCoreOffset], true);
    this->bmm1.SetTail(this->s1RealSize, this->s2RealSize);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::ProcessStage1(
    int64_t s2LoopLimit)
{
    LocalTensor<T> stage1BufTensor = this->commonTBuf.template Get<T>()[this->commonTBuf1Offset];

    // 获取bmm1的计算结果
    LocalTensor<T> bmm1ResUb = this->stage1TBuf.template Get<T>();

    // 计算S2轴的尾块
    this->ComputeS2TailSize();

    this->CopyInAndCastPse(stage1BufTensor);
    this->CopyInAttenMask();

    this->GetBmm1Result(bmm1ResUb);

    this->ComputePse(stage1BufTensor, bmm1ResUb);

    if (likely(this->s2LoopCount > 0)) {
        // 此处计算上一次的tensor，必定是s2的基本块
        this->Bmm2SetTensor(this->s2BaseSize);
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        set_flag(PIPE_MTE3, PIPE_S, eventID);
        wait_flag(PIPE_MTE3, PIPE_S, eventID);

        this->bmm2.template Iterate<false>();
    }

    pipe_barrier(PIPE_V);
    Muls(bmm1ResUb, bmm1ResUb, static_cast<T>(this->tilingData->inputParams.scaleValue), s1RealSize * s2TailSize);

    this->CastAndComputeAttenMask(stage1BufTensor, bmm1ResUb);
    this->CopyInDropMask();
    this->SoftMaxCompute(bmm1ResUb, s2LoopLimit);

    this->ComputeDropMask(bmm1ResUb);

    pipe_barrier(PIPE_V);
    LocalTensor<INPUT_T> castBufTensor = this->commonTBuf.template Get<INPUT_T>()[this->commonTBuf1Offset];
    if constexpr (!IsSameType<T, INPUT_T>::value) {
        Cast(castBufTensor, bmm1ResUb, RoundMode::CAST_ROUND, s1RealSize * s2TailSize);
    } else {
        DataCopy(castBufTensor, bmm1ResUb, s1RealSize * s2TailSize);
    }
    pipe_barrier(PIPE_V);
    return;
}

// bmm1在GetTensorC的尾块需要处理在s2方向不包含n:1的尾块
template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::ComputeS2TailSize()
{
    this->s2TailSize = s2BaseSize;
    int64_t s2RemainSize = this->tilingData->inputParams.s2Size - s2LoopCount * s2BaseSize;
    if (unlikely(s2RemainSize > 0 && s2RemainSize < this->s2TailSize)) {
        this->s2TailSize = s2RemainSize;
    }
    return;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::CopyInAndCastPse(
    LocalTensor<T> &castTensor)
{
    if (!this->hasPse) {
        return;
    }
    int64_t offset = this->ComputePseOffset();
    LocalTensor<INPUT_T> pseUb = this->optionalInputQueue.template AllocTensor<INPUT_T>();
    int32_t s1Size = this->tilingData->inputParams.pseShapeType == this->pse1S2 ? 1 : s1RealSize;
    this->DataCopyIn(pseUb, this->pseGm, offset, s1Size, s2TailSize);

    pseUb = this->optionalInputQueue.template DeQue<INPUT_T>();

    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    set_flag(PIPE_MTE3, PIPE_V, eventID);
    wait_flag(PIPE_MTE3, PIPE_V, eventID);
    if constexpr (!IsSameType<INPUT_T, T>::value) {
        Cast(castTensor, pseUb, RoundMode::CAST_NONE, s1Size * s2TailSize);
    } else {
        DataCopy(castTensor, pseUb, s1Size * s2TailSize);
    }
    this->optionalInputQueue.FreeTensor(pseUb);
    return;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline int64_t FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::ComputePseOffset()
{
    if (!this->hasPse) {
        return 0;
    }
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    int64_t s1Offset = 0;
    int64_t s2Offset = 0;
    if (this->tilingData->inputParams.pseShapeType == this->pseS1S2) {
        // b, n2, g, s1, s2
        bOffset = this->boIdx * this->n2GS1S2;
        n2Offset = this->n2oIdx * this->gS1S2;
        gOffset = this->goIdx * this->s1S2;
        s1Offset = this->s1oIdx * this->s1BaseS2;
        s2Offset = this->s2LoopCount * s2BaseSize;
    } else if (this->tilingData->inputParams.pseShapeType == this->pse1S2) {
        // b, n2, g, 1, s2
        bOffset = this->boIdx * this->n2GS2;
        n2Offset = this->n2oIdx * this->gS2;
        gOffset = this->goIdx * this->tilingData->inputParams.s2Size;
        s1Offset = 0;
        s2Offset = this->s2LoopCount * s2BaseSize;
    }
    return bOffset + n2Offset + gOffset + s1Offset + s2Offset;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::CopyInAttenMask()
{
    if (!this->hasAttenMask) {
        return;
    }

    int64_t attenMaskOffset = this->ComputeAttenMaskOffset();
    this->AttenMaskCopyIn(attenMaskOffset, this->s1RealSize, this->s2TailSize);
    return;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline int64_t FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::ComputeAttenMaskOffset()
{
    if (this->tilingData->inputParams.attenMaskCompressMode == this->attenMaskNoCompress) {
        return this->ComputeOffsetForNoCompress();
    } else if (this->tilingData->inputParams.attenMaskCompressMode == this->attenMaskLeftUpCausalCompress) {
        int64_t delta = this->s1oIdx * this->s1BaseSize - this->s2LoopCount * this->s2BaseSize;
        return this->ComputeOffsetForCausal(delta, this->s1BaseSize, this->s2BaseSize,
                                            this->tilingData->inputParams.attenMaskS2Size);
    } else if (this->tilingData->inputParams.attenMaskCompressMode == this->attenMaskRightDownCausalCompress) {
        int64_t delta = this->s1oIdx * this->s1BaseSize - this->s2LoopCount * this->s2BaseSize +
                        this->tilingData->inputParams.s2Size - this->tilingData->inputParams.s1Size;
        return this->ComputeOffsetForCausal(delta, this->s1BaseSize, this->s2BaseSize,
                                            this->tilingData->inputParams.attenMaskS2Size);
    }
    return 0;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline int64_t FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::ComputeOffsetForNoCompress()
{
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    int64_t s1Offset = this->s1oIdx * this->s1BaseS2;
    int64_t s2Offset = this->s2LoopCount * s2BaseSize;
    if (this->tilingData->inputParams.attenMaskShapeType == this->attenMaskBN2GS1S2) {
        bOffset = this->boIdx * this->n2GS1S2;
        n2Offset = this->n2oIdx * this->gS1S2;
        gOffset = this->goIdx * this->s1S2;
    } else if (this->tilingData->inputParams.attenMaskShapeType == this->attenMaskBS1S2) {
        bOffset = this->boIdx * this->s1S2;
    }
    return bOffset + n2Offset + gOffset + s1Offset + s2Offset;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::GetBmm1Result(
    LocalTensor<T> &bmm1ResUb)
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    set_flag(PIPE_MTE3, PIPE_MTE2, eventID);
    wait_flag(PIPE_MTE3, PIPE_MTE2, eventID);

    uint32_t bmm1ResShape[] = {static_cast<uint32_t>(s1RealSize), static_cast<uint32_t>(s2TailSize)};
    bmm1ResUb.SetShapeInfo(ShapeInfo(2, bmm1ResShape, DataFormat::ND));
    this->bmm1.template GetTensorC<false>(bmm1ResUb, false, true);
    pipe_barrier(PIPE_V);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::ComputePse(LocalTensor<T> &pseUb,
    LocalTensor<T> &bmm1ResUb)
{
    if (!this->hasPse) {
        return;
    }

    int32_t computeSize =
        this->tilingData->inputParams.pseShapeType == this->pseS1S2 ? s1RealSize * s2TailSize : s2TailSize;
    pipe_barrier(PIPE_V);
    this->PseBroadcastAdd(s1RealSize, s2TailSize, computeSize, pseUb, bmm1ResUb);
    return;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::CastAndComputeAttenMask(
    LocalTensor<T> &castUb, LocalTensor<T> &bmm1ResUb)
{
    if (!this->hasAttenMask) {
        return;
    }

    LocalTensor<uint8_t> attenMaskUb = this->optionalInputQueue.template DeQue<uint8_t>();
    int32_t alignedS2Size = this->CeilDiv(s2TailSize, this->blockBytes) * this->blockBytes;
    uint32_t shapeArray[] = {static_cast<uint32_t>(s1RealSize), static_cast<uint32_t>(alignedS2Size)};
    attenMaskUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    attenMaskUb.SetSize(s1RealSize * alignedS2Size);
    bmm1ResUb.SetSize(s1RealSize * s2TailSize);
    LocalTensor<uint8_t> apiTmpBuffer = commonTBuf.template Get<uint8_t>();
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
    shapeInfo.firstAxis = s1RealSize;
    shapeInfo.srcLastAxis = s2TailSize;
    shapeInfo.maskLastAxis = alignedS2Size;
    SelectWithBytesMask(bmm1ResUb, bmm1ResUb, scalar, attenMaskUb, apiTmpBuffer, shapeInfo);
    this->optionalInputQueue.FreeTensor(attenMaskUb);
    return;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::CopyInDropMask()
{
    if (!this->hasDropMask) {
        return;
    }
    // softmax的计算结果需要执行dropout操作
    int64_t dropMaskOffset = this->ComputeDropOffset();
    this->DropoutCopyIn(dropMaskOffset, this->s1RealSize, this->s2TailSize);
    return;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline int64_t FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::ComputeDropOffset()
{
    int64_t bOffset = this->boIdx * this->n2GS1S2;
    int64_t n2Offset = this->n2oIdx * this->gS1S2;
    int64_t gOffset = this->goIdx * this->s1S2;

    int64_t s1Offset = this->s1oIdx * this->s1BaseS2;
    int64_t s2Offset = this->s2LoopCount * s2BaseSize;

    return bOffset + n2Offset + gOffset + s1Offset + s2Offset;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::SoftMaxCompute(
    LocalTensor<T> &srcTensor, int64_t s2LoopLimit)
{
    LocalTensor<uint8_t> apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();
    bool formerBlock = this->s1RealSize == this->tilingData->coreParams.s1BaseSize &&
        this->s2TailSize == this->tilingData->coreParams.s2BaseSize;
    pipe_barrier(PIPE_V);
    SoftMaxTiling tailCoreTiling;
    if (unlikely(this->s2RealLoopCount == 0)) {
        if (likely(formerBlock)) {
            SoftmaxFlashV2<T, false, true, true>(srcTensor, this->softmaxSumUb, this->softmaxMaxUb, srcTensor,
                this->softmaxExpUb, this->softmaxSumUb, this->softmaxMaxUb, apiTmpBuffer,
                this->tilingData->softmaxFlashTilingData);
        } else {
            SoftmaxFlashV2<T, false, true, false>(srcTensor, this->softmaxSumUb, this->softmaxMaxUb, srcTensor,
                this->softmaxExpUb, this->softmaxSumUb, this->softmaxMaxUb, apiTmpBuffer, tailCoreTiling);
        }
    } else {
        if (likely(formerBlock)) {
            SoftmaxFlashV2<T, true, true, true>(srcTensor, this->softmaxSumUb, this->softmaxMaxUb, srcTensor,
                this->softmaxExpUb, this->softmaxSumUb, this->softmaxMaxUb, apiTmpBuffer,
                this->tilingData->softmaxFlashTilingData);
        } else {
            SoftmaxFlashV2<T, true, true, false>(srcTensor, this->softmaxSumUb, this->softmaxMaxUb, srcTensor,
                this->softmaxExpUb, this->softmaxSumUb, this->softmaxMaxUb, apiTmpBuffer, tailCoreTiling);
        }
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::ComputeDropMask(
    LocalTensor<T> &bmm1ResUb)
{
    if (!this->hasDropMask) {
        return;
    }
    LocalTensor<uint8_t> apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();

    // softmax的计算结果需要执行dropout操作
    LocalTensor<uint8_t> dropMaskUb = this->optionalInputQueue.template DeQue<uint8_t>();
    pipe_barrier(PIPE_V);

    DropOutShapeInfo dropOutShapeInfo;
    dropOutShapeInfo.firstAxis = s1RealSize;
    dropOutShapeInfo.srcLastAxis = s2TailSize;

    dropOutShapeInfo.maskLastAxis = this->CeilDiv(s2TailSize / this->byteBitRatio, this->blockBytes) * this->blockBytes;
    DropOut(bmm1ResUb, bmm1ResUb, dropMaskUb, apiTmpBuffer, this->tilingData->inputParams.keepProb, dropOutShapeInfo);

    this->optionalInputQueue.FreeTensor(dropMaskUb);
    return;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::ProcessStage2()
{
    LocalTensor<INPUT_T> stage1BufTensor = this->commonTBuf.template Get<INPUT_T>()[this->commonTBuf1Offset];

    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    set_flag(PIPE_V, PIPE_MTE3, eventID);
    wait_flag(PIPE_V, PIPE_MTE3, eventID);
    if (this->s2RealLoopCount % 2 == 0) {
        DataCopy(this->stage2WorkspaceGm[this->stage2WorkspaceGm1Offset], stage1BufTensor, s1RealSize * s2TailSize);
    } else {
        DataCopy(this->stage2WorkspaceGm[this->stage2WorkspaceGm2Offset], stage1BufTensor, s1RealSize * s2TailSize);
    }

    if (unlikely(this->s2RealLoopCount == 0)) {
        return;
    }
    // 获取缓存bmm2的计算结果
    LocalTensor<T> bmm2ResUb = this->stage2TBuf.template Get<T>();
    LocalTensor<T> stage2BufTensor = this->commonTBuf.template Get<T>()[this->commonTBuf2Offset];

    this->bmm2.template GetTensorC<false, true>(stage2BufTensor, false, false, s1RealSize, dSize, 0, padDSize);
    this->bmm2.End();
    pipe_barrier(PIPE_ALL);
    if (unlikely(this->s2RealLoopCount == 1)) {
        DataCopy(bmm2ResUb, stage2BufTensor, s1RealSize * alignDSize);
    } else {
        Add(bmm2ResUb, bmm2ResUb, stage2BufTensor, this->s1RealSize * alignDSize);
    }
    this->Bmm2ResultMul(bmm2ResUb);
    return;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::Bmm2ResultMul(
    LocalTensor<T> &bmm2ResUb)
{
    pipe_barrier(PIPE_V);
    BinaryRepeatParams repeatParams;
    repeatParams.src0BlkStride = 0;
    repeatParams.src0RepStride = 1;
    repeatParams.src1RepStride = alignDSize / this->blockSize;
    repeatParams.dstRepStride = alignDSize / this->blockSize;

    // s1长度可能会超过255限制，修改成双重循环
    // 根据一次最多计算的byte数量，对bmm2Res分组mul
    int32_t loop = alignDSize / this->repeatMaxSize;
    int32_t remain = alignDSize % this->repeatMaxSize;
    for (int i = 0; i < loop; i++) {
        Mul(bmm2ResUb[i * this->repeatMaxSize], this->softmaxExpUb, bmm2ResUb[i * this->repeatMaxSize],
            this->repeatMaxSize, this->s1RealSize, repeatParams);
    }
    if (likely(remain)) {
        Mul(bmm2ResUb[loop * this->repeatMaxSize], this->softmaxExpUb, bmm2ResUb[loop * this->repeatMaxSize], remain,
            this->s1RealSize, repeatParams);
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::Bmm2SetTensor(int64_t s2Size)
{
    if (this->s2RealLoopCount % 2 == 1) {
        this->bmm2.SetTensorA(this->stage2WorkspaceGm[this->stage2WorkspaceGm1Offset]);
    } else {
        this->bmm2.SetTensorA(this->stage2WorkspaceGm[this->stage2WorkspaceGm2Offset]);
    }

    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;

    if (this->tilingData->inputParams.layoutType == this->layoutBsh) {
        // BSH/BSND
        bOffset = this->boIdx * this->n2S2D;
        s2Offset = (this->s2LoopCount - 1) * this->s2BaseN2D;
        n2Offset = this->n2oIdx * dSize;
    } else if (this->tilingData->inputParams.layoutType == this->layoutSbh) {
        // SBH/SBND
        s2Offset = (this->s2LoopCount - 1) * this->s2BaseBN2D;
        bOffset = this->boIdx * this->n2D;
        n2Offset = this->n2oIdx * dSize;
    } else if (this->tilingData->inputParams.layoutType == this->layoutBnsd) {
        // BNSD
        bOffset = this->boIdx * this->n2S2D;
        n2Offset = this->n2oIdx * this->s2D;
        s2Offset = (this->s2LoopCount - 1) * s2BaseSize * dSize;
    }

    int64_t vCoreOffset = bOffset + n2Offset + s2Offset;
    this->bmm2.SetTensorB(this->valueGm[vCoreOffset]);
    this->bmm2.SetTail(-1, -1, s2Size);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::Bmm2ResultDiv()
{
    LocalTensor<T> bmm2ResUb = this->stage2TBuf.template Get<T>();

    BinaryRepeatParams repeatParams;
    repeatParams.src0BlkStride = 1;
    repeatParams.src0RepStride = alignDSize / this->blockSize;
    repeatParams.src1BlkStride = 0;
    repeatParams.src1RepStride = 1;
    repeatParams.dstRepStride = alignDSize / this->blockSize;
    int32_t loop = alignDSize / this->repeatMaxSize;
    int32_t remain = alignDSize % this->repeatMaxSize;

    int32_t calcSize = this->softmaxSumUb.GetSize();
    // 用optionalInputQueue的queue
    pipe_barrier(PIPE_V);
    if constexpr (IsSameType<T, half>::value) {
        LocalTensor<float> commonBufTensor = this->commonTBuf.template Get<float>();
        Copy(commonBufTensor, this->softmaxSumUb, 64, calcSize / 64, { 2, 1, 16, 8 });
        Copy(commonBufTensor[8], this->softmaxSumUb, 64, calcSize / 64, { 2, 1, 16, 8 });
        LocalTensor<T> sumCastTensor = softmaxExpTBuf.Get<T>(this->tilingData->tensorSizeParams.softmaxExpUbSize);

        Cast(sumCastTensor, commonBufTensor, RoundMode::CAST_ROUND, 2 * this->softmaxSumUb.GetSize());
        for (int i = 0; i < loop; i++) {
            Div(bmm2ResUb[i * this->repeatMaxSize], bmm2ResUb[i * this->repeatMaxSize], sumCastTensor,
                this->repeatMaxSize, this->s1RealSize, repeatParams);
        }
        if (likely(remain)) {
            Div(bmm2ResUb[loop * this->repeatMaxSize], bmm2ResUb[loop * this->repeatMaxSize], sumCastTensor, remain,
                this->s1RealSize, repeatParams);
        }
    } else {
        for (int i = 0; i < loop; i++) {
            Div(bmm2ResUb[i * this->repeatMaxSize], bmm2ResUb[i * this->repeatMaxSize], this->softmaxSumUb,
                this->repeatMaxSize, this->s1RealSize, repeatParams);
        }
        if (likely(remain)) {
            Div(bmm2ResUb[loop * this->repeatMaxSize], bmm2ResUb[loop * this->repeatMaxSize], this->softmaxSumUb,
                remain, this->s1RealSize, repeatParams);
        }
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::Bmm2DataCopyOut()
{
    LocalTensor<T> bmm2ResUb = this->stage2TBuf.template Get<T>();
    TransposeParams transposeParams;
    transposeParams.bIndex = this->boIdx;

    // n2轴和g轴合并到一起
    transposeParams.nIndex = this->n2oIdx * this->tilingData->inputParams.gSize + this->goIdx;
    transposeParams.sIndex = this->s1oIdx * s1RealSize;
    transposeParams.hNIndex = 0;

    int32_t calcSize = bmm2ResUb.GetSize();
    LocalTensor<INPUT_T> attenOut = this->bmm2OutQueue.template AllocTensor<INPUT_T>();
    pipe_barrier(PIPE_V);
    if constexpr (!IsSameType<INPUT_T, T>::value) {
        LocalTensor<INPUT_T> castTensor = this->commonTBuf.template Get<INPUT_T>();
        Cast(attenOut, bmm2ResUb, RoundMode::CAST_ROUND, calcSize);
    } else {
        DataCopy(attenOut, bmm2ResUb, calcSize);
    }

    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    set_flag(PIPE_V, PIPE_MTE3, eventID);
    wait_flag(PIPE_V, PIPE_MTE3, eventID);

    this->bmm2OutQueue.EnQue(attenOut);
    attenOut = this->bmm2OutQueue.template DeQue<INPUT_T>();

    dataCopyTiling.srcShapeS = this->s1RealSize;

    DataCopyParams dataCopyParams;
    dataCopyParams.blockLen = dSize * sizeof(INPUT_T);
    dataCopyParams.srcStride = 0;
    int64_t dstStride = 0;
    if (this->tilingData->inputParams.layoutType == this->layoutBsh) {
        dstStride =
            (this->tilingData->inputParams.n2Size * this->tilingData->inputParams.gSize - 1) * dSize * sizeof(INPUT_T);
    } else if (this->tilingData->inputParams.layoutType == this->layoutSbh) {
        dstStride = (this->tilingData->inputParams.bSize * this->tilingData->inputParams.n2Size *
            this->tilingData->inputParams.gSize -
            1) *
            dSize * sizeof(INPUT_T);
    }

    // dataCopyParams.dstStride类型定义uint16_t，65535是其最大值
    if (likely(dstStride <= 65535)) {
        dataCopyParams.blockCount = s1RealSize;
        dataCopyParams.dstStride = static_cast<uint16_t>(dstStride);
        DataCopyPad(this->attentionOutGm[this->qCoreOffset], attenOut, dataCopyParams);
    } else {
        dataCopyParams.blockCount = 1;
        dataCopyParams.dstStride = 0;
        int64_t datacopyOffset = dSize;
        if (this->tilingData->inputParams.layoutType == this->layoutBsh) {
            datacopyOffset = this->n2GD;
        } else if (this->tilingData->inputParams.layoutType == this->layoutSbh) {
            datacopyOffset = this->bN2G * dSize;
        }

        for (uint32_t i = 0; i < s1RealSize; i++) {
            DataCopyPad(this->attentionOutGm[this->qCoreOffset + i * datacopyOffset], attenOut[i * alignDSize],
                        dataCopyParams);
        }
    }

    bmm2OutQueue.FreeTensor(attenOut);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2<INPUT_T, T, isBasicBlock>::SoftmaxDataCopyOut()
{
    int64_t bOffset = this->boIdx * this->n2GS1 * this->fp32BaseSize;
    int64_t n2Offset = this->n2oIdx * this->gS1 * this->fp32BaseSize;
    int64_t gOffset = this->goIdx * this->tilingData->inputParams.s1Size * this->fp32BaseSize;
    int64_t s1Offset = this->s1oIdx * s1BaseSize * this->fp32BaseSize;
    int64_t gmOffset = bOffset + n2Offset + gOffset + s1Offset;
    int64_t calculateSize = s1RealSize * this->fp32BaseSize;

    this->softmaxOutQueue.EnQue(this->softmaxMaxUb);
    this->softmaxOutQueue.EnQue(this->softmaxSumUb);
    LocalTensor<float> maxTensor = this->softmaxOutQueue.template DeQue<float>();
    LocalTensor<float> sumTensor = this->softmaxOutQueue.template DeQue<float>();
    DataCopy(this->softmaxMaxGm[gmOffset], maxTensor, calculateSize);
    DataCopy(this->softmaxSumGm[gmOffset], sumTensor, calculateSize);
    this->softmaxOutQueue.FreeTensor(this->softmaxMaxUb);
    this->softmaxOutQueue.FreeTensor(this->softmaxSumUb);
}

#endif // FLASH_ATTENTION_SCORE_SPLIT_B_N2_G_S1_S2_H
