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

#ifndef FLASH_ATTENTION_SCORE_SPLIT_B_N2_G_S1_D_DB_PERFORMANCE_H
#define FLASH_ATTENTION_SCORE_SPLIT_B_N2_G_S1_D_DB_PERFORMANCE_H
#include "flash_attention_score_common.h"
#include "kernel_data_copy_transpose.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

using matmul::MatmulType;

struct SplitS1dExtraInfo {
    int64_t s2StartIdx;
    int64_t s2EndIdx;
    int64_t s2LoopCount;
    int64_t s1oIdx;
    int64_t boIdx;
    int64_t n2oIdx;
    int64_t goIdx;
    int64_t taskId;
    int64_t s1RealSize;
    int64_t s2RealSize;
    int64_t vecS1BaseSize;
    int64_t vecS1TailSize;
    int64_t realSplitN;
    int64_t s2LoopLimit;
    int64_t multiCoreInnerIdx;
    int64_t softmaxCopyOutLimit;
    int64_t softmaxCopyOutSize;
};

// INPUT_T - means data type for input
// T       - means data type when calc
template <typename INPUT_T, typename T = INPUT_T, bool isBasicBlock = false>
class FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf : public FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock> {
public:
    __aicore__ inline FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf() {};

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
        __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax,
        __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
        const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

    // define matmul
    using a1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T>;
    using b1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, true>;
    using bias1Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c1Type = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    matmul::Matmul<a1Type, b1Type, c1Type, bias1Type, CFG_EXCEED> bmm1;

    // define batchmatmul
    using a2Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T>;
    using b2Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T>;
    using bias2Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c2Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T>;
    matmul::Matmul<a2Type, b2Type, c2Type, bias2Type, CFG_EXCEED> bmm2;

protected:
    __aicore__ inline void InitInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
        __gm__ uint8_t *pse, __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask,
        __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut,
        __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
        const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void WaitBmm1Result();
    __aicore__ inline void WaitBmm2Result();
    __aicore__ inline void IterateBmm2(SplitS1dExtraInfo &extraInfo);
    __aicore__ inline void SetExtraInfo(SplitS1dExtraInfo &extraInfo, int64_t taskId, int64_t s2LoopCount,
        int64_t s2LoopLimit, int64_t multiCoreInnerIdx);
    __aicore__ inline void SetTiling(const FlashAttentionScoreGeneralTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffer();
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void ComputeAxisIdx(int64_t multiCoreInnerIdx);
    __aicore__ inline void IterateBmm1(SplitS1dExtraInfo &extraInfo);
    __aicore__ inline void Bmm1SetTensorA(SplitS1dExtraInfo &extraInfo);
    __aicore__ inline void ComputeBmm1Tail(SplitS1dExtraInfo &extraInfo);
    __aicore__ inline void SetBmm1TensorB(SplitS1dExtraInfo &extraInfo);
    __aicore__ inline void ProcessVec1(SplitS1dExtraInfo &extraInfo);
    __aicore__ inline void CopyInAndCastPse(LocalTensor<T> &castTensor, SplitS1dExtraInfo &extraInfo, int64_t loopIdx);
    __aicore__ inline int64_t ComputePseOffset(SplitS1dExtraInfo &extraInfo, int64_t loopIdx);
    __aicore__ inline void CopyInAttenMask(SplitS1dExtraInfo &extraInfo, int64_t loopIdx);
    __aicore__ inline int64_t ComputeAttenMaskOffset(SplitS1dExtraInfo &extraInfo, int64_t loopIdx);
    __aicore__ inline int64_t ComputeOffsetForNoCompress(SplitS1dExtraInfo &extraInfo, int64_t loopIdx);
    __aicore__ inline void GetBmm1Result(SplitS1dExtraInfo &extraInfo, LocalTensor<T> &bmm1ResUb, int64_t loopIdx);
    __aicore__ inline void ComputePse(LocalTensor<T> &pseUb, LocalTensor<T> &bmm1ResUb, SplitS1dExtraInfo &extraInfo);
    __aicore__ inline void ComputeAttenMask(SplitS1dExtraInfo &extraInfo, LocalTensor<T> &bmm1ResUb);
    __aicore__ inline void CopyInDropMask(SplitS1dExtraInfo &extraInfo, int64_t loopIdx);
    __aicore__ inline int64_t ComputeDropOffset(SplitS1dExtraInfo &extraInfo, int64_t loopIdx);
    __aicore__ inline void SoftMaxCompute(SplitS1dExtraInfo &extraInfo, LocalTensor<T> &srcTensor, int64_t loopIdx);
    __aicore__ inline void ComputeDropMask(SplitS1dExtraInfo &extraInfo, LocalTensor<T> &bmm1ResUb);
    __aicore__ inline void Bmm2SetTensor();

    uint32_t s1BaseSize;
    uint32_t s2BaseSize;
    uint32_t dSize;

    // sparse 用函数
    __aicore__ inline void GetS1LoopRange(int64_t &multiCoreInnerOffset, int64_t &multiCoreInnerLimit);
    __aicore__ inline void GetS2LoopRange();

    // sparse 用参数
    int64_t s2StartIdx;
    int64_t s2EndIdx;
    int64_t s2LoopStart;

    // s1方向Vec计算长度
    uint32_t s1VecSize;

    // s2方向的尾块，包含N:1配比
    int32_t s2RealSize;

    // s2方向的尾块，不包含N:1的配比
    int32_t s2TailSize;

    // k和v矩阵在核间的offset
    int64_t kCoreOffset;
    int64_t qCoreOffset;
    int64_t commonTBuf1Offset;
    int64_t commonTBuf2Offset;

    // 资源分配
    TBuf<> stage1TBuf; // 存stage1中间结果
    TBuf<> softmaxMaxBuf;
    TBuf<> softmaxSumBuf;
    TBuf<> commonTBuf; // common的复用空间
    
    GlobalTensor<T> mm1ResPing;
    GlobalTensor<T> mm1ResPong;
    GlobalTensor<INPUT_T> stage1ResPing;
    GlobalTensor<INPUT_T> stage1ResPong;

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
    int64_t bN2D;
    int64_t n2G;
    int64_t n2GD;
    int64_t bN2GD;
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

    int64_t softmaxBufSize = 256;
};

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf<INPUT_T, T, isBasicBlock>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse, __gm__ uint8_t *dropMask,
    __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
    __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe)
{
    this->InitInput(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
        attentionOut, workspace, tiling, tPipe); // gm设置

    this->ComputeConstexpr();
    this->InitBuffer();
    LocalTensor<T> apiTmpBuffer = this->commonTBuf.template Get<T>();
    DropOutBitModeInit(apiTmpBuffer);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf<INPUT_T, T, isBasicBlock>::InitInput(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse, __gm__ uint8_t *dropMask,
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

    // 补齐到512， 统一按T处理
    int64_t mmNRatioOffset = this->CeilDiv(s1BaseSize * this->tilingData->inputParams.s2Size, 256) * 256 *
        sizeof(INPUT_T); // todo:需要传对齐后的s2Size // bmm占2块*2，stage1占1块*2

    int64_t totalOffset = mmNRatioOffset * 6;

    // bmm1Result， 占用mmNRatioOffset空间
    this->mm1ResPing.SetGlobalBuffer((__gm__ T *)(workspace + this->blockIdx * totalOffset));
    this->mm1ResPong.SetGlobalBuffer((__gm__ T *)(workspace + this->blockIdx * totalOffset + 2 * mmNRatioOffset));

    // stage1Result， 占用mmNRatioOffset空间
    this->stage1ResPing.SetGlobalBuffer(
        (__gm__ INPUT_T *)(workspace + this->blockIdx * totalOffset + 4 * mmNRatioOffset));
    this->stage1ResPong.SetGlobalBuffer(
        (__gm__ INPUT_T *)(workspace + this->blockIdx * totalOffset + 5 * mmNRatioOffset));

    this->hasPse = this->pseGm.GetPhyAddr() != nullptr;
    this->hasAttenMask = this->attenMaskGmInt.GetPhyAddr() != nullptr;
    this->hasDropMask = this->dropMaskGm.GetPhyAddr() != nullptr;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf<INPUT_T, T, isBasicBlock>::SetTiling(
    const FlashAttentionScoreGeneralTilingData *__restrict tilingData)
{
    // copy base params
    this->tilingData = tilingData;
    this->s1BaseSize = this->tilingData->coreParams.s1BaseSize * this->tilingData->coreParams.nRatio;
    this->s2BaseSize = this->tilingData->coreParams.s2BaseSize;
    this->dSize = this->tilingData->inputParams.dSize;
};

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf<INPUT_T, T, isBasicBlock>::InitBuffer()
{
    // todo: 8*1024需要根据buf数量重新计算
    this->pipe->InitBuffer(this->optionalInputQueue, 1, 8 * 1024 * sizeof(INPUT_T));
    // 64 128 4
    this->pipe->InitBuffer(this->stage1TBuf, 8 * 1024 * sizeof(T));

    // todo: 8*1024需要根据buf数量重新计算
    this->pipe->InitBuffer(this->commonTBuf, 64 * 128 * 2 * sizeof(T));

    // 改为256*4*8
    this->pipe->InitBuffer(this->softmaxMaxBuf, this->softmaxBufSize * this->blockBytes);
    this->pipe->InitBuffer(this->softmaxSumBuf, this->softmaxBufSize * this->blockBytes);
    this->commonTBuf1Offset = 0;
    // todo: 8*1024需要根据buf数量重新计算
    this->commonTBuf2Offset = 64 * 128;
}
template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf<INPUT_T, T, isBasicBlock>::ComputeConstexpr()
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

    this->bN2D = this->tilingData->inputParams.bSize * n2D;
    this->n2GS1o = this->tilingData->inputParams.n2Size * this->gS1o;
    this->gS1D = this->tilingData->inputParams.gSize * this->s1D;
    this->n2S2D = this->tilingData->inputParams.n2Size * this->s2D;
    this->n2GD = this->tilingData->inputParams.n2Size * this->gD;
    this->bN2GD = this->tilingData->inputParams.bSize * n2GD;
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
}

// sparse functions
template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf<INPUT_T, T, isBasicBlock>::GetS1LoopRange(
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
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf<INPUT_T, T, isBasicBlock>::GetS2LoopRange()
{
    // 计算S2的循环范围相关参数: 后续可 使用static_cast<uint32_t>优化scale性能
    if (this->tilingData->inputParams.sparseType == 3) { // 下三角
        this->s2StartIdx = 0;
        this->s2EndIdx = this->Min((this->s1oIdx + 1) * s1BaseSize, this->tilingData->inputParams.s2Size);
    } else if (this->tilingData->inputParams.sparseType == 4 or
        this->tilingData->inputParams.sparseType == 6) { // 对角线往外扩散场景, s1和s2可能不同
        this->s2StartIdx = this->Max(
            this->s1oIdx * this->tilingData->coreParams.s1BaseSize - this->tilingData->coreParams.s1SparseValidSize, 0);
        this->s2EndIdx = this->Min((this->s1oIdx + 1) * s1BaseSize + this->tilingData->coreParams.s2SparseValidSize,
            this->tilingData->inputParams.s2Size);
    } else if (this->tilingData->inputParams.sparseType == 5) {
        this->s2StartIdx = 0;
        this->s2EndIdx = this->Max((this->s1oIdx + 1) * s1BaseSize,
                                    this->tilingData->inputParams.prefixNList[this->boIdx]);
        this->s2EndIdx = (this->s2EndIdx + s1BaseSize - 1) / s1BaseSize * s1BaseSize;
        this->s2EndIdx = this->Min(this->s2EndIdx, this->tilingData->inputParams.s2Size);
    } else { // 其它场景, 如无attention mask
        this->s2StartIdx = 0;
        this->s2EndIdx = this->tilingData->inputParams.s2Size;
    }
    return;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf<INPUT_T, T, isBasicBlock>::Process()
{
    // 确定核内切分起点
    int64_t multiCoreInnerOffset = this->blockIdx * this->tilingData->multiCoreParams.splitFactorSize;
    int64_t multiCoreInnerLimit = multiCoreInnerOffset + this->tilingData->multiCoreParams.splitFactorSize;
    if (this->tilingData->multiCoreParams.totalSize < multiCoreInnerLimit) {
        multiCoreInnerLimit = this->tilingData->multiCoreParams.totalSize;
    }
    // 计算sparse场景下s1的循环范围
    this->GetS1LoopRange(multiCoreInnerOffset, multiCoreInnerLimit);
    SplitS1dExtraInfo extraInfo[3];
    int64_t taskId = 0;
    for (int64_t multiCoreInnerIdx = multiCoreInnerOffset; multiCoreInnerIdx < multiCoreInnerLimit;
        multiCoreInnerIdx++) {
        this->ComputeAxisIdx(multiCoreInnerIdx);

        // s2轴循环计数, 支持sparse和非sparse场景
        this->GetS2LoopRange();
        if (taskId > 0) {
            WaitBmm1Result();
        }
        this->SetExtraInfo(extraInfo[taskId % 3], taskId, 0, 0, multiCoreInnerIdx);
        this->IterateBmm1(extraInfo[taskId % 3]);

        if (taskId > 0) {
            this->ProcessVec1(extraInfo[(taskId + 2) % 3]);
            set_flag(PIPE_MTE3, PIPE_MTE2, 0);
        }

        if (taskId > 1) {
            WaitBmm2Result();
        }

        if (taskId > 0) {
            wait_flag(PIPE_MTE3, PIPE_MTE2, 0);
            this->IterateBmm2(extraInfo[(taskId + 2) % 3]);
        }
        taskId++;
    }
    if (taskId >= 1) {
        WaitBmm1Result();
        this->ProcessVec1(extraInfo[(taskId + 2) % 3]);
        set_flag(PIPE_MTE3, PIPE_MTE2, 0);
        if (taskId > 1) {
            WaitBmm2Result();
        }
        wait_flag(PIPE_MTE3, PIPE_MTE2, 0);
        this->IterateBmm2(extraInfo[(taskId + 2) % 3]);
    }
    taskId++;
    if (taskId >= 2) {
        WaitBmm2Result();
    }
};

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf<INPUT_T, T, isBasicBlock>::ComputeAxisIdx(
    int64_t multiCoreInnerIdx)
{
    // 计算轴的idx
    this->boIdx = multiCoreInnerIdx / this->n2GS1o;
    this->n2oIdx = multiCoreInnerIdx % this->n2GS1o / this->gS1o;
    this->goIdx = multiCoreInnerIdx % this->gS1o / this->tilingData->coreParams.s1OuterSize;
    this->s1oIdx = multiCoreInnerIdx % this->tilingData->coreParams.s1OuterSize;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf<INPUT_T, T, isBasicBlock>::WaitBmm1Result()
{
    this->bmm1.WaitIterateAll();
    this->bmm1.End();
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf<INPUT_T, T, isBasicBlock>::SetExtraInfo(
    SplitS1dExtraInfo &extraInfo, int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit, int64_t multiCoreInnerIdx)
{
    extraInfo.s2StartIdx = this->s2StartIdx;
    extraInfo.s2EndIdx = this->s2EndIdx;
    extraInfo.s2LoopCount = s2LoopCount;
    extraInfo.s1oIdx = this->s1oIdx;
    extraInfo.boIdx = this->boIdx;
    extraInfo.n2oIdx = this->n2oIdx;
    extraInfo.goIdx = this->goIdx;
    extraInfo.taskId = taskId;
    extraInfo.s2LoopLimit = s2LoopLimit;
    extraInfo.multiCoreInnerIdx = multiCoreInnerIdx;

    // todo s1尾块
    if (extraInfo.s1oIdx == this->tilingData->coreParams.s1OuterSize - 1) {
        extraInfo.s1RealSize = this->tilingData->coreParams.s1BaseTailSize;
    } else {
        extraInfo.s1RealSize = s1BaseSize;
    }

    this->ComputeBmm1Tail(extraInfo);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf<INPUT_T, T, isBasicBlock>::ComputeBmm1Tail(
    SplitS1dExtraInfo &extraInfo)
{
    extraInfo.s2RealSize = extraInfo.s2EndIdx - extraInfo.s2StartIdx;

    // todo: 8*1024需要根据buf数量重新计算,待考虑对齐场景
    extraInfo.vecS1BaseSize = this->Min(1024 / extraInfo.s2RealSize * 8, this->s1BaseSize);
    // vec计算循环次数
    extraInfo.realSplitN = this->CeilDiv(extraInfo.s1RealSize, extraInfo.vecS1BaseSize);

    // 放满softmaxBuf需要的循环次数
    extraInfo.softmaxCopyOutLimit = this->softmaxBufSize / extraInfo.vecS1BaseSize;
    // 放满softmaxBuf时的数据量
    extraInfo.softmaxCopyOutSize = extraInfo.softmaxCopyOutLimit * extraInfo.vecS1BaseSize;
    return;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf<INPUT_T, T, isBasicBlock>::IterateBmm1(
    SplitS1dExtraInfo &extraInfo)
{
    int64_t s1StrideSize = 0;
    int64_t s2StrideSize = 0;
    if (this->tilingData->inputParams.layoutType == this->layoutBsh) {
        // BSH/BSNGD
        s1StrideSize = n2GD;
        s2StrideSize = n2D;
    } else if (this->tilingData->inputParams.layoutType == this->layoutSbh) {
        // SBH/SBNGD
        s1StrideSize = bN2GD;
        s2StrideSize = bN2D;
    } else if (this->tilingData->inputParams.layoutType == this->layoutBnsd) {
        // bnsd
        s1StrideSize = dSize;
        s2StrideSize = dSize;
    }

    this->bmm1.SetOrgShape(this->tilingData->inputParams.s1Size, s2StrideSize, s1StrideSize, s2StrideSize,
        extraInfo.s2RealSize);

    this->Bmm1SetTensorA(extraInfo);
    this->SetBmm1TensorB(extraInfo);
    if (extraInfo.taskId % 2 == 0) {
        this->bmm1.template IterateAll<false>(this->mm1ResPing, false, false, true);
    } else {
        this->bmm1.template IterateAll<false>(this->mm1ResPong, false, false, true);
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf<INPUT_T, T, isBasicBlock>::Bmm1SetTensorA(
    SplitS1dExtraInfo &extraInfo)
{
    // 计算gm上的offset
    int64_t bOffset = 0;

    // s1需要考虑inner轴的影响
    int64_t s1Offset = 0;

    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    if (this->tilingData->inputParams.layoutType == this->layoutBsh) {
        // BSH/BSNGD
        bOffset = extraInfo.boIdx * this->n2GS1D;
        s1Offset = extraInfo.s1oIdx * this->s1BaseN2GD;
        n2Offset = extraInfo.n2oIdx * this->gD;
        gOffset = extraInfo.goIdx * dSize;
    } else if (this->tilingData->inputParams.layoutType == this->layoutSbh) {
        // SBH/SBNGD
        s1Offset = extraInfo.s1oIdx * this->s1BaseBN2GD;
        bOffset = extraInfo.boIdx * this->n2GD;
        n2Offset = extraInfo.n2oIdx * this->gD;
        gOffset = extraInfo.goIdx * dSize;
    } else if (this->tilingData->inputParams.layoutType == this->layoutBnsd) {
        // bnsd
        bOffset = extraInfo.boIdx * this->n2GS1D;
        n2Offset = extraInfo.n2oIdx * this->gS1D;
        gOffset = extraInfo.goIdx * this->s1D;
        s1Offset = extraInfo.s1oIdx * this->s1BaseD;
    }
    this->qCoreOffset = bOffset + n2Offset + gOffset + s1Offset;
    this->bmm1.SetTensorA(this->queryGm[this->qCoreOffset]);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf<INPUT_T, T, isBasicBlock>::SetBmm1TensorB(
    SplitS1dExtraInfo &extraInfo)
{
    // 计算gm上的offset
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;
    if (this->tilingData->inputParams.layoutType == this->layoutBsh) {
        // BSH/BSND
        bOffset = extraInfo.boIdx * this->n2S2D;
        s2Offset = extraInfo.s2StartIdx * this->n2D;
        n2Offset = extraInfo.n2oIdx * dSize;
    } else if (this->tilingData->inputParams.layoutType == this->layoutSbh) {
        // SBH/SBND
        s2Offset = extraInfo.s2StartIdx * this->bN2D;
        bOffset = extraInfo.boIdx * this->n2D;
        n2Offset = extraInfo.n2oIdx * dSize;
    } else if (this->tilingData->inputParams.layoutType == this->layoutBnsd) {
        // BNSD
        bOffset = extraInfo.boIdx * this->n2S2D;
        n2Offset = extraInfo.n2oIdx * this->s2D;
        s2Offset = extraInfo.s2StartIdx * dSize;
    }
    int64_t kCoreOffset = bOffset + n2Offset + s2Offset;
    this->bmm1.SetTensorB(this->keyGm[kCoreOffset], true);
    this->bmm1.SetTail(extraInfo.s1RealSize, extraInfo.s2RealSize, this->tilingData->inputParams.dSize);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf<INPUT_T, T, isBasicBlock>::ProcessVec1(
    SplitS1dExtraInfo &extraInfo)
{
    for (uint32_t loopIdx = 0; loopIdx < extraInfo.realSplitN; loopIdx++) {
        extraInfo.vecS1TailSize = extraInfo.vecS1BaseSize;
        if (loopIdx == extraInfo.realSplitN - 1) {
            extraInfo.vecS1TailSize = extraInfo.s1RealSize - loopIdx * extraInfo.vecS1BaseSize;
        }

        LocalTensor<T> stage1BufTensor = this->commonTBuf.template Get<T>()[commonTBuf1Offset];

        // 获取bmm1的计算结果
        LocalTensor<T> bmm1ResUb = this->stage1TBuf.template Get<T>();

        // todo 考虑s2方向尾块
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        set_flag(PIPE_MTE3, PIPE_V, eventID);
        wait_flag(PIPE_MTE3, PIPE_V, eventID);

        this->CopyInAndCastPse(stage1BufTensor, extraInfo, loopIdx);
        this->CopyInAttenMask(extraInfo, loopIdx);

        eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        set_flag(PIPE_V, PIPE_MTE2, eventID);
        wait_flag(PIPE_V, PIPE_MTE2, eventID);

        this->GetBmm1Result(extraInfo, bmm1ResUb, loopIdx);
        eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        set_flag(PIPE_MTE2, PIPE_V, eventID);
        wait_flag(PIPE_MTE2, PIPE_V, eventID);
        this->ComputePse(stage1BufTensor, bmm1ResUb, extraInfo);

        pipe_barrier(PIPE_V);
        Muls(bmm1ResUb, bmm1ResUb, static_cast<T>(this->tilingData->inputParams.scaleValue),
            extraInfo.vecS1TailSize * extraInfo.s2RealSize);

        this->ComputeAttenMask(extraInfo, bmm1ResUb);
        this->CopyInDropMask(extraInfo, loopIdx);
        this->SoftMaxCompute(extraInfo, bmm1ResUb, loopIdx);

        this->ComputeDropMask(extraInfo, bmm1ResUb);

        pipe_barrier(PIPE_V);
        LocalTensor<INPUT_T> castBufTensor = this->commonTBuf.template Get<INPUT_T>()[this->commonTBuf1Offset];
        if constexpr (!IsSameType<T, INPUT_T>::value) {
            Cast(castBufTensor, bmm1ResUb, RoundMode::CAST_ROUND, extraInfo.vecS1TailSize * extraInfo.s2RealSize);
        } else {
            DataCopy(castBufTensor, bmm1ResUb, extraInfo.vecS1TailSize * extraInfo.s2RealSize);
        }
        eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        set_flag(PIPE_V, PIPE_MTE3, eventID);
        wait_flag(PIPE_V, PIPE_MTE3, eventID);

        if (extraInfo.taskId % 2 == 0) {
            DataCopy(this->stage1ResPing[loopIdx * extraInfo.vecS1BaseSize * extraInfo.s2RealSize], castBufTensor,
                extraInfo.vecS1TailSize * extraInfo.s2RealSize);
        } else {
            DataCopy(this->stage1ResPong[loopIdx * extraInfo.vecS1BaseSize * extraInfo.s2RealSize], castBufTensor,
                extraInfo.vecS1TailSize * extraInfo.s2RealSize);
        }
    }
    return;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf<INPUT_T, T, isBasicBlock>::CopyInAndCastPse(
    LocalTensor<T> &castTensor, SplitS1dExtraInfo &extraInfo, int64_t loopIdx)
{
    if (!this->hasPse) {
        return;
    }
    int64_t offset = this->ComputePseOffset(extraInfo, loopIdx);
    LocalTensor<INPUT_T> pseUb = this->optionalInputQueue.template AllocTensor<INPUT_T>();
    int32_t s1Size = this->tilingData->inputParams.pseShapeType == this->pse1S2 ? 1 : extraInfo.vecS1TailSize;
    this->DataCopyIn(pseUb, this->pseGm, offset, s1Size, extraInfo.s2RealSize);

    pseUb = this->optionalInputQueue.template DeQue<INPUT_T>();

    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    set_flag(PIPE_MTE3, PIPE_V, eventID);
    wait_flag(PIPE_MTE3, PIPE_V, eventID);
    if constexpr (!IsSameType<INPUT_T, T>::value) {
        Cast(castTensor, pseUb, RoundMode::CAST_NONE, s1Size * extraInfo.s2RealSize);
    } else {
        DataCopy(castTensor, pseUb, s1Size * extraInfo.s2RealSize);
    }
    this->optionalInputQueue.FreeTensor(pseUb);
    return;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline int64_t FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf<INPUT_T, T, isBasicBlock>::ComputePseOffset(
    SplitS1dExtraInfo &extraInfo, int64_t loopIdx)
{
    if (!this->hasPse) {
        return 0;
    }
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s1Offset = 0;
    int64_t s2Offset = extraInfo.s2StartIdx;
    int64_t gOffset = 0;
    if (this->tilingData->inputParams.pseShapeType == this->pseS1S2) {
        // b, n2, g, s1, s2
        bOffset = extraInfo.boIdx * this->n2GS1S2;
        n2Offset = extraInfo.n2oIdx * this->gS1S2;
        gOffset = extraInfo.goIdx * this->s1S2;
        s1Offset = extraInfo.s1oIdx * this->s1BaseS2 +
            loopIdx * extraInfo.vecS1BaseSize * this->tilingData->inputParams.s2Size;
    } else if (this->tilingData->inputParams.pseShapeType == this->pse1S2) {
        // b, n2, g, 1, s2
        bOffset = extraInfo.boIdx * this->n2GS2;
        n2Offset = extraInfo.n2oIdx * this->gS2;
        gOffset = extraInfo.goIdx * this->tilingData->inputParams.s2Size;
        s1Offset = 0;
    }
    return bOffset + n2Offset + gOffset + s1Offset + s2Offset;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf<INPUT_T, T, isBasicBlock>::CopyInAttenMask(
    SplitS1dExtraInfo &extraInfo, int64_t loopIdx)
{
    if (!this->hasAttenMask) {
        return;
    }

    int64_t attenMaskOffset = this->ComputeAttenMaskOffset(extraInfo, loopIdx);
    this->AttenMaskCopyIn(attenMaskOffset, extraInfo.vecS1TailSize, extraInfo.s2RealSize);
    return;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline int64_t FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf<INPUT_T, T,
    isBasicBlock>::ComputeAttenMaskOffset(SplitS1dExtraInfo &extraInfo, int64_t loopIdx)
{
    if (this->tilingData->inputParams.attenMaskCompressMode == this->attenMaskNoCompress) {
        return this->ComputeOffsetForNoCompress(extraInfo, loopIdx);
    } else if (this->tilingData->inputParams.attenMaskCompressMode == this->attenMaskLeftUpCausalCompress) {
        int64_t delta = extraInfo.s1oIdx * s1BaseSize + loopIdx * extraInfo.vecS1BaseSize;
        return this->ComputeOffsetForCausal(delta, extraInfo.vecS1BaseSize, this->tilingData->inputParams.s2Size,
                                            this->tilingData->inputParams.attenMaskS2Size);
    } else if (this->tilingData->inputParams.attenMaskCompressMode == this->attenMaskRightDownCausalCompress) {
        int64_t delta = extraInfo.s1oIdx * s1BaseSize + loopIdx * extraInfo.vecS1BaseSize +
                        this->tilingData->inputParams.s2Size - this->tilingData->inputParams.s1Size;
        return this->ComputeOffsetForCausal(delta, extraInfo.vecS1BaseSize, this->tilingData->inputParams.s2Size,
            this->tilingData->inputParams.attenMaskS2Size);
    }
}


template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline int64_t FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf<INPUT_T, T,
    isBasicBlock>::ComputeOffsetForNoCompress(SplitS1dExtraInfo &extraInfo, int64_t loopIdx)
{
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    int64_t s1Offset =
        extraInfo.s1oIdx * this->s1BaseS2 + loopIdx * extraInfo.vecS1BaseSize * this->tilingData->inputParams.s2Size;
    int64_t s2Offset = extraInfo.s2StartIdx;
    if (this->tilingData->inputParams.attenMaskShapeType == this->attenMaskBN2GS1S2) {
        bOffset = extraInfo.boIdx * this->n2GS1S2;
        n2Offset = extraInfo.n2oIdx * this->gS1S2;
        gOffset = extraInfo.goIdx * this->s1S2;
    } else if (this->tilingData->inputParams.attenMaskShapeType == this->attenMaskBS1S2) {
        bOffset = extraInfo.boIdx * this->s1S2;
    }
    return bOffset + n2Offset + gOffset + s1Offset + s2Offset;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf<INPUT_T, T, isBasicBlock>::GetBmm1Result(
    SplitS1dExtraInfo &extraInfo, LocalTensor<T> &bmm1ResUb, int64_t loopIdx)
{
    if (extraInfo.taskId % 2 == 0) {
        DataCopy(bmm1ResUb, this->mm1ResPing[loopIdx * extraInfo.vecS1BaseSize * extraInfo.s2RealSize],
            extraInfo.vecS1TailSize * extraInfo.s2RealSize);
    } else {
        DataCopy(bmm1ResUb, this->mm1ResPong[loopIdx * extraInfo.vecS1BaseSize * extraInfo.s2RealSize],
            extraInfo.vecS1TailSize * extraInfo.s2RealSize);
    }
    uint32_t bmm1ResUbShape[] = {static_cast<uint32_t>(extraInfo.vecS1TailSize), static_cast<uint32_t>(extraInfo.s2RealSize)};
    bmm1ResUb.SetShapeInfo(ShapeInfo(2, bmm1ResUbShape, DataFormat::ND));
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf<INPUT_T, T, isBasicBlock>::ComputePse(
    LocalTensor<T> &pseUb, LocalTensor<T> &bmm1ResUb, SplitS1dExtraInfo &extraInfo)
{
    if (!this->hasPse) {
        return;
    }

    int32_t computeSize = this->tilingData->inputParams.pseShapeType == this->pseS1S2 ?
        extraInfo.vecS1TailSize * extraInfo.s2RealSize :
        extraInfo.s2RealSize;
    pipe_barrier(PIPE_V);
    this->PseBroadcastAdd(extraInfo.vecS1TailSize, extraInfo.s2RealSize, computeSize, pseUb, bmm1ResUb);
    return;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf<INPUT_T, T, isBasicBlock>::ComputeAttenMask(
    SplitS1dExtraInfo &extraInfo, LocalTensor<T> &bmm1ResUb)
{
    if (!this->hasAttenMask) {
        return;
    }

    LocalTensor<uint8_t> attenMaskUb = this->optionalInputQueue.template DeQue<uint8_t>();
    int32_t alignedS2Size = this->CeilDiv(extraInfo.s2RealSize, this->blockBytes) * this->blockBytes;
    uint32_t shapeArray[] = {static_cast<uint32_t>(extraInfo.vecS1TailSize), static_cast<uint32_t>(alignedS2Size)};
    attenMaskUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    attenMaskUb.SetSize(extraInfo.vecS1TailSize * alignedS2Size);
    bmm1ResUb.SetSize(extraInfo.vecS1TailSize * extraInfo.s2RealSize);
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
    shapeInfo.firstAxis = extraInfo.vecS1TailSize;
    shapeInfo.srcLastAxis = extraInfo.s2RealSize;
    shapeInfo.maskLastAxis = alignedS2Size;
    SelectWithBytesMask(bmm1ResUb, bmm1ResUb, scalar, attenMaskUb, apiTmpBuffer, shapeInfo);
    this->optionalInputQueue.FreeTensor(attenMaskUb);
    return;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf<INPUT_T, T, isBasicBlock>::CopyInDropMask(
    SplitS1dExtraInfo &extraInfo, int64_t loopIdx)
{
    if (!this->hasDropMask) {
        return;
    }
    // softmax的计算结果需要执行dropout操作
    int64_t dropMaskOffset = this->ComputeDropOffset(extraInfo, loopIdx);
    this->DropoutCopyIn(dropMaskOffset, extraInfo.vecS1TailSize, extraInfo.s2RealSize);
    return;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline int64_t FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf<INPUT_T, T, isBasicBlock>::ComputeDropOffset(
    SplitS1dExtraInfo &extraInfo, int64_t loopIdx)
{
    int64_t bOffset = extraInfo.boIdx * this->n2GS1S2;
    int64_t n2Offset = extraInfo.n2oIdx * this->gS1S2;
    int64_t gOffset = extraInfo.goIdx * this->s1S2;

    int64_t s1Offset =
        extraInfo.s1oIdx * this->s1BaseS2 + loopIdx * extraInfo.vecS1BaseSize * this->tilingData->inputParams.s2Size;
    int64_t s2Offset = extraInfo.s2StartIdx;

    return bOffset + n2Offset + gOffset + s1Offset + s2Offset;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf<INPUT_T, T, isBasicBlock>::SoftMaxCompute(
    SplitS1dExtraInfo &extraInfo, LocalTensor<T> &srcTensor, int64_t loopIdx)
{
    uint32_t maxSumShape[] = {static_cast<uint32_t>(extraInfo.vecS1TailSize), static_cast<uint32_t>(this->fp32BaseSize)};

    LocalTensor<float> sumUb = this->softmaxSumBuf.template Get<float>()[loopIdx % extraInfo.softmaxCopyOutLimit *
        extraInfo.vecS1BaseSize * this->fp32BaseSize];
    LocalTensor<float> maxUb = this->softmaxMaxBuf.template Get<float>()[loopIdx % extraInfo.softmaxCopyOutLimit *
        extraInfo.vecS1BaseSize * this->fp32BaseSize];
    sumUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));
    maxUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));

    LocalTensor<uint8_t> apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();
    pipe_barrier(PIPE_V);

    AscendC::SoftMax<T, true, isBasicBlock>(srcTensor, sumUb, maxUb, srcTensor, apiTmpBuffer,
        this->tilingData->softmaxTilingData);
    if (loopIdx == extraInfo.realSplitN - 1 || (loopIdx + 1) % extraInfo.softmaxCopyOutLimit == 0) {
        int64_t bOffset = extraInfo.boIdx * this->n2GS1 * this->fp32BaseSize;
        int64_t n2Offset = extraInfo.n2oIdx * this->gS1 * this->fp32BaseSize;
        int64_t gOffset = extraInfo.goIdx * this->tilingData->inputParams.s1Size * this->fp32BaseSize;
        int64_t s1Offset = extraInfo.s1oIdx * s1BaseSize * this->fp32BaseSize;
        int64_t vs1Offset = loopIdx / extraInfo.softmaxCopyOutLimit * this->softmaxBufSize * this->fp32BaseSize;
        int64_t gmOffset = bOffset + n2Offset + gOffset + s1Offset + vs1Offset;
        int64_t calculateSize = this->Min(extraInfo.softmaxCopyOutSize,
            extraInfo.s1RealSize - loopIdx / extraInfo.softmaxCopyOutLimit * extraInfo.softmaxCopyOutSize) *
            this->fp32BaseSize;

        LocalTensor<float> maxTensor = this->softmaxMaxBuf.template Get<float>();
        LocalTensor<float> sumTensor = this->softmaxSumBuf.template Get<float>();
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        set_flag(PIPE_V, PIPE_MTE3, eventID);
        wait_flag(PIPE_V, PIPE_MTE3, eventID);
        DataCopy(this->softmaxMaxGm[gmOffset], maxTensor, calculateSize);
        DataCopy(this->softmaxSumGm[gmOffset], sumTensor, calculateSize);
        eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        set_flag(PIPE_MTE3, PIPE_V, eventID);
        wait_flag(PIPE_MTE3, PIPE_V, eventID);
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf<INPUT_T, T, isBasicBlock>::ComputeDropMask(
    SplitS1dExtraInfo &extraInfo, LocalTensor<T> &bmm1ResUb)
{
    if (!this->hasDropMask) {
        return;
    }
    LocalTensor<uint8_t> apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();

    // softmax的计算结果需要执行dropout操作
    LocalTensor<uint8_t> dropMaskUb = this->optionalInputQueue.template DeQue<uint8_t>();
    pipe_barrier(PIPE_V);

    DropOutShapeInfo dropOutShapeInfo;
    dropOutShapeInfo.firstAxis = extraInfo.vecS1TailSize;
    dropOutShapeInfo.srcLastAxis = extraInfo.s2RealSize;

    dropOutShapeInfo.maskLastAxis =
        this->CeilDiv(extraInfo.s2RealSize / this->byteBitRatio, this->blockBytes) * this->blockBytes;
    DropOut(bmm1ResUb, bmm1ResUb, dropMaskUb, apiTmpBuffer, this->tilingData->inputParams.keepProb, dropOutShapeInfo);

    this->optionalInputQueue.FreeTensor(dropMaskUb);
    return;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf<INPUT_T, T, isBasicBlock>::WaitBmm2Result()
{
    this->bmm2.WaitIterateAll();
    this->bmm2.End();
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf<INPUT_T, T, isBasicBlock>::IterateBmm2(
    SplitS1dExtraInfo &extraInfo)
{
    // 计算gm上的offset
    int64_t qBOffset = 0;

    // s1需要考虑inner轴的影响
    int64_t qS1Offset = 0;
    int64_t qN2Offset = 0;
    int64_t qGOffset = 0;

    if (this->tilingData->inputParams.layoutType == this->layoutBsh) {
        // BSH/BSNGD
        qBOffset = extraInfo.boIdx * this->n2GS1D;
        qS1Offset = extraInfo.s1oIdx * this->s1BaseN2GD;
        qN2Offset = extraInfo.n2oIdx * this->gD;
        qGOffset = extraInfo.goIdx * dSize;
    } else if (this->tilingData->inputParams.layoutType == this->layoutSbh) {
        // SBH/SBNGD
        qS1Offset = extraInfo.s1oIdx * this->s1BaseBN2GD;
        qBOffset = extraInfo.boIdx * this->n2GD;
        qN2Offset = extraInfo.n2oIdx * this->gD;
        qGOffset = extraInfo.goIdx * dSize;
    } else if (this->tilingData->inputParams.layoutType == this->layoutBnsd) {
        // BNSD
        qBOffset = extraInfo.boIdx * this->n2GS1D;
        qN2Offset = extraInfo.n2oIdx * this->gS1D;
        qGOffset = extraInfo.goIdx * this->s1D;
        qS1Offset = extraInfo.s1oIdx * this->s1BaseD;
    }
    int64_t qCoreOffset = qBOffset + qN2Offset + qGOffset + qS1Offset;

    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;
    int64_t bStrideSize = 0;
    int64_t cStrideSize = 0;

    if (this->tilingData->inputParams.layoutType == this->layoutBsh) {
        // BSH/BSND
        bOffset = extraInfo.boIdx * this->n2S2D;
        s2Offset = extraInfo.s2StartIdx * this->n2D;
        n2Offset = extraInfo.n2oIdx * dSize;
        bStrideSize = this->n2D;
        cStrideSize = this->n2GD;
    } else if (this->tilingData->inputParams.layoutType == this->layoutSbh) {
        // SBH/SBND
        s2Offset = extraInfo.s2StartIdx * this->bN2D;
        bOffset = extraInfo.boIdx * this->n2D;
        n2Offset = extraInfo.n2oIdx * dSize;
        bStrideSize = this->bN2D;
        cStrideSize = this->bN2GD;
    } else if (this->tilingData->inputParams.layoutType == this->layoutBnsd) {
        // BNSD
        bOffset = extraInfo.boIdx * this->n2S2D;
        n2Offset = extraInfo.n2oIdx * this->s2D;
        s2Offset = extraInfo.s2StartIdx * dSize;
        bStrideSize = dSize;
        cStrideSize = dSize;
    }
    int64_t vCoreOffset = bOffset + n2Offset + s2Offset;
    this->bmm2.SetOrgShape(this->tilingData->inputParams.s1Size, bStrideSize, extraInfo.s2RealSize, bStrideSize,
        cStrideSize);
    if (extraInfo.taskId % 2 == 0) {
        this->bmm2.SetTensorA(this->stage1ResPing);
    } else {
        this->bmm2.SetTensorA(this->stage1ResPong);
    }

    this->bmm2.SetTensorB(this->valueGm[vCoreOffset]);
    this->bmm2.SetTail(extraInfo.s1RealSize, this->tilingData->inputParams.dSize, extraInfo.s2RealSize);
    this->bmm2.template IterateAll<false>(this->attentionOutGm[qCoreOffset], false, false, true);
}

#endif // FLASH_ATTENTION_SCORE_SPLIT_B_N2_G_S1_D_DB_PERFORMANCE_H