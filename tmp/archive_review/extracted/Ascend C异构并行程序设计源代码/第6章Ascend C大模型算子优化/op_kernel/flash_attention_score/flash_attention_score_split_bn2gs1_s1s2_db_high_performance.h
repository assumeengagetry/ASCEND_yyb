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

#ifndef FLASH_ATTENTION_SCORE_SPLIT_B_N2_G_S1_S2_DB_PERFORMANCE_H
#define FLASH_ATTENTION_SCORE_SPLIT_B_N2_G_S1_S2_DB_PERFORMANCE_H
#include "flash_attention_score_common.h"
#include "kernel_data_copy_transpose.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

using matmul::MatmulType;

struct SplitExtraInfo {
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
    int64_t vec2S1BaseSize;
    int64_t vec2S1TailSize;
    int64_t realSplitN;
    int64_t s2LoopLimit;
    int64_t multiCoreInnerIdx;
    int64_t qCoreOffset;
};

// INPUT_T - means data type for input
// T       - means data type when calc
template <typename INPUT_T, typename T = INPUT_T, bool isBasicBlock = false>
class FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf : public FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock> {
public:
    __aicore__ inline FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf() {};

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
    using c2Type = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    matmul::Matmul<a2Type, b2Type, c2Type, bias2Type, CFG_EXCEED> bmm2;

protected:
    __aicore__ inline void InitInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
        __gm__ uint8_t *pse, __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask,
        __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut,
        __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
        const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void WaitBmm1Result();
    __aicore__ inline void WaitBmm2Result();
    __aicore__ inline void DataCopyIn(LocalTensor<INPUT_T> &dstTensor, GlobalTensor<INPUT_T> &srcTensor, int64_t offset,
        int32_t s1Size, int32_t s2Size);
    __aicore__ inline void BoolCopyIn(LocalTensor<uint8_t> &dstTensor, GlobalTensor<uint8_t> &srcTensor, int64_t offset,
        int32_t s1Size, int32_t s2Size);
    __aicore__ inline void Bit2Int8CopyIn(LocalTensor<uint8_t> &dstTensor, GlobalTensor<uint8_t> &srcTensor,
        int64_t offset, int32_t s1Size, int32_t s2Size, int32_t batchSize);
    __aicore__ inline void IterateBmm2(SplitExtraInfo &extraInfo);
    __aicore__ inline void SetExtraInfo(SplitExtraInfo &extraInfo, int64_t taskId, int64_t s2LoopCount,
        int64_t s2LoopLimit, int64_t multiCoreInnerIdx);
    __aicore__ inline void SetTiling(const FlashAttentionScoreGeneralTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffer();
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void ComputeAxisIdx(int64_t multiCoreInnerIdx);
    __aicore__ inline void IterateBmm1(SplitExtraInfo &extraInfo);
    __aicore__ inline void Bmm1SetTensorA(SplitExtraInfo &extraInfo);
    __aicore__ inline void ComputeBmm1Tail(SplitExtraInfo &extraInfo);
    __aicore__ inline void SetBmm1TensorB(SplitExtraInfo &extraInfo);
    __aicore__ inline void ProcessVec1(SplitExtraInfo &extraInfo);
    __aicore__ inline void CopyInAndCastPse(LocalTensor<T> &castTensor, SplitExtraInfo &extraInfo, int64_t loopIdx);
    __aicore__ inline int64_t ComputePseOffset(SplitExtraInfo &extraInfo, int64_t loopIdx);
    __aicore__ inline void CopyInAttenMask(SplitExtraInfo &extraInfo, int64_t loopIdx);
    __aicore__ inline int64_t ComputeAttenMaskOffset(SplitExtraInfo &extraInfo, int64_t loopIdx);
    __aicore__ inline int64_t ComputeOffsetForNoCompress(SplitExtraInfo &extraInfo, int64_t loopIdx);
    __aicore__ inline void GetBmm1Result(SplitExtraInfo &extraInfo, LocalTensor<T> &bmm1ResUb, int64_t loopIdx);
    __aicore__ inline void ComputePse(LocalTensor<T> &pseUb, LocalTensor<T> &bmm1ResUb, SplitExtraInfo &extraInfo);
    __aicore__ inline void ComputeAttenMask(SplitExtraInfo &extraInfo, LocalTensor<T> &bmm1ResUb);
    __aicore__ inline void CopyInDropMask(SplitExtraInfo &extraInfo, int64_t loopIdx);
    __aicore__ inline int64_t ComputeDropOffset(SplitExtraInfo &extraInfo, int64_t loopIdx);
    __aicore__ inline void SoftMaxCompute(SplitExtraInfo &extraInfo, LocalTensor<T> &srcTensor, int64_t loopIdx);
    __aicore__ inline void ComputeDropMask(SplitExtraInfo &extraInfo, LocalTensor<T> &bmm1ResUb);
    __aicore__ inline void ProcessVec2(SplitExtraInfo &extraInfo);
    __aicore__ inline void Bmm2ResultMul(SplitExtraInfo &extraInfo, LocalTensor<T> &bmm2ResUb, int64_t s1oIdx);
    __aicore__ inline void Bmm2ResultDiv(SplitExtraInfo &extraInfo, int64_t s1oIdx);
    __aicore__ inline void Bmm2DataCopyOut(SplitExtraInfo &extraInfo, int64_t s1oIdx);
    __aicore__ inline void SoftmaxDataCopyOut(SplitExtraInfo &extraInfo);

    // 构建dataCopyTranspose参数
    CopyTransposeTiling dataCopyTiling;

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

    // 资源分配
    TBuf<> maskTBufPing;
    TBuf<> maskTBufPong;
    TBuf<> pseTBuf;
    TBuf<> stage1PingBuf;
    TBuf<> stage1PongBuf;
    TBuf<> stage2TBuf;
    TBuf<> softmaxSumPingBuf;
    TBuf<> softmaxSumPongBuf;
    TBuf<> softmaxExpPingBuf;
    TBuf<> softmaxExpPongBuf;
    TBuf<> softmaxMaxBuf;
    TBuf<> commonTBuf; // common的复用空间

    LocalTensor<T> softmaxExpUb;
    GlobalTensor<T> mm1ResPing;
    GlobalTensor<T> mm1ResPong;
    GlobalTensor<T> mm2ResPing;
    GlobalTensor<T> mm2ResPong;
    GlobalTensor<T> vec2ResPing;
    GlobalTensor<T> vec2ResPong;
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
};

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::Init(
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
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::InitInput(
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
    int64_t mm1ResultSize = s1BaseSize * s2BaseSize;
    int64_t mmNRatioOffset = this->CeilDiv(mm1ResultSize * this->tilingData->coreParams.nRatio, 128) * 128 * sizeof(T);
    int64_t mm2ResultSize = s1BaseSize * dSize;
    int64_t mm2Offset = this->CeilDiv(mm2ResultSize, 128) * 128 * 4;

    int64_t totalOffset = mmNRatioOffset * 2 + mm2Offset * 2;
    if (dSize >= 256) {
        totalOffset = mmNRatioOffset * 2 + mm2Offset * 4;
    }

    // bmm1Result， 占用mmNRatioOffset空间
    this->mm1ResPing.SetGlobalBuffer((__gm__ T *)(workspace + this->blockIdx * totalOffset));
    this->mm1ResPong.SetGlobalBuffer((__gm__ T *)(workspace + this->blockIdx * totalOffset + mmNRatioOffset));

    // stage1Result， 占用mmNRatioOffset空间
    this->stage1ResPing.SetGlobalBuffer((__gm__ INPUT_T *)(workspace + this->blockIdx * totalOffset));
    this->stage1ResPong.SetGlobalBuffer((__gm__ INPUT_T *)(workspace + this->blockIdx * totalOffset + mmNRatioOffset));

    // bmm2Result， 占用mmOffset空间
    this->mm2ResPing.SetGlobalBuffer((__gm__ T *)(workspace + this->blockIdx * totalOffset + mmNRatioOffset * 2));
    this->mm2ResPong.SetGlobalBuffer(
        (__gm__ T *)(workspace + this->blockIdx * totalOffset + mmNRatioOffset * 2 + mm2Offset));

    // vec2阶段。占用mmOffset空间， 仅在D轴大于256的情况下出现
    if (dSize >= 256) {
        this->vec2ResPing.SetGlobalBuffer(
            (__gm__ T *)(workspace + this->blockIdx * totalOffset + mmNRatioOffset * 2 + mm2Offset * 2));
        this->vec2ResPong.SetGlobalBuffer(
            (__gm__ T *)(workspace + this->blockIdx * totalOffset + mmNRatioOffset * 2 + mm2Offset * 3));
    }

    this->hasPse = this->pseGm.GetPhyAddr() != nullptr;
    this->hasAttenMask = this->attenMaskGmInt.GetPhyAddr() != nullptr;
    this->hasDropMask = this->dropMaskGm.GetPhyAddr() != nullptr;
    this->isInputTAttenMask = this->tilingData->inputParams.attenMaskDataType == 0;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::SetTiling(
    const FlashAttentionScoreGeneralTilingData *__restrict tilingData)
{
    // copy base params
    this->tilingData = tilingData;
    this->s1BaseSize = this->tilingData->coreParams.s1BaseSize;
    this->s2BaseSize = this->tilingData->coreParams.s2BaseSize;
    this->dSize = this->tilingData->inputParams.dSize;
};

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::InitBuffer()
{
    this->pipe->InitBuffer(this->maskTBufPing, 8 * 1024);              // 可以给attenmask 8k
    this->pipe->InitBuffer(this->maskTBufPong, 8 * 1024);              // 可以给dropoutmask 8k
    this->pipe->InitBuffer(this->pseTBuf, 8 * 1024 * sizeof(INPUT_T)); // pse 16k

    // 64 128 4
    this->pipe->InitBuffer(this->stage1PingBuf, 8 * 1024 * sizeof(T)); // t.a 32k
    this->pipe->InitBuffer(this->stage1PongBuf, 8 * 1024 * sizeof(T)); // i.a 32k
    this->pipe->InitBuffer(this->stage2TBuf, 64 * 128 * sizeof(T));    // t.c 32k

    // 64 128 4
    this->pipe->InitBuffer(this->commonTBuf, 64 * 128 * sizeof(T)); // t.b 32k

    // 128 8 5 4  10 20
    this->pipe->InitBuffer(this->softmaxSumPingBuf, s1BaseSize * this->blockBytes); // 2k
    this->pipe->InitBuffer(this->softmaxSumPongBuf, s1BaseSize * this->blockBytes); // 2k
    this->pipe->InitBuffer(this->softmaxMaxBuf, s1BaseSize * this->blockBytes);     // 2k
    this->pipe->InitBuffer(this->softmaxExpPingBuf, s1BaseSize * this->blockBytes); // 2k
    this->pipe->InitBuffer(this->softmaxExpPongBuf, s1BaseSize * this->blockBytes); // 2k
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::ComputeConstexpr()
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
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::GetS1LoopRange(
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
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::GetS2LoopRange()
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
        this->s2EndIdx = this->Max((this->s1oIdx + 1) * s1BaseSize, this->tilingData->inputParams.prefixNList[this->boIdx]);
        this->s2EndIdx = this->CeilDiv(this->s2EndIdx, s2BaseSize) * s2BaseSize;
        this->s2EndIdx = this->Min(this->s2EndIdx, this->tilingData->inputParams.s2Size);
    } else { // 其它场景, 如无attention mask
        this->s2StartIdx = 0;
        this->s2EndIdx = this->tilingData->inputParams.s2Size;
    }
    return;
}


template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::Process()
{
    // 确定核内切分起点
    int64_t multiCoreInnerOffset = this->blockIdx * this->tilingData->multiCoreParams.splitFactorSize;
    int64_t multiCoreInnerLimit = multiCoreInnerOffset + this->tilingData->multiCoreParams.splitFactorSize;
    if (this->tilingData->multiCoreParams.totalSize < multiCoreInnerLimit) {
        multiCoreInnerLimit = this->tilingData->multiCoreParams.totalSize;
    }
    // 计算sparse场景下s1的循环范围
    this->GetS1LoopRange(multiCoreInnerOffset, multiCoreInnerLimit);

    SplitExtraInfo extraInfo[3];
    int64_t taskId = 0;
    event_t mte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    for (int64_t multiCoreInnerIdx = multiCoreInnerOffset; multiCoreInnerIdx < multiCoreInnerLimit;
        multiCoreInnerIdx++) {
        this->ComputeAxisIdx(multiCoreInnerIdx);

        // s2轴循环计数, 支持sparse和非sparse场景
        this->GetS2LoopRange();
        int64_t s2LoopLimit = this->CeilDiv(this->s2EndIdx - this->s2StartIdx, s2BaseNratioSize) - 1;
        for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; s2LoopCount++) {
            if (taskId > 0) {
                // 对应extraInfo[(i+2)%3]
                WaitBmm1Result();
            }
            this->SetExtraInfo(extraInfo[taskId % 3], taskId, s2LoopCount, s2LoopLimit, multiCoreInnerIdx);

            this->IterateBmm1(extraInfo[taskId % 3]);

            if (taskId > 0) {
                this->ProcessVec1(extraInfo[(taskId + 2) % 3]);
                set_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3);
            }

            if (taskId > 1) {
                // 对应extraInfo[(i+1)%3]
                WaitBmm2Result();
            }

            if (taskId > 0) {
                wait_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3);
                this->IterateBmm2(extraInfo[(taskId + 2) % 3]);
            }

            if (taskId > 1) {
                this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
            }
            taskId++;
        }
    }
    if (taskId >= 1) {
        // 对应extraInfo[(i+2)%3]
        WaitBmm1Result();
        this->ProcessVec1(extraInfo[(taskId + 2) % 3]);
        set_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3);
        if (taskId > 1) {
            // 对应extraInfo[(i+1)%3]
            WaitBmm2Result();
        }
        wait_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3);
        this->IterateBmm2(extraInfo[(taskId + 2) % 3]);
        if (taskId > 1) {
            this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
        }
    }
    taskId++;
    if (taskId >= 2) {
        // 对应extraInfo[(i+1)%3]
        WaitBmm2Result();
        this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
    }
};

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::ComputeAxisIdx(
    int64_t multiCoreInnerIdx)
{
    // 计算轴的idx
    this->boIdx = multiCoreInnerIdx / this->n2GS1o;
    this->n2oIdx = multiCoreInnerIdx % this->n2GS1o / this->gS1o;
    this->goIdx = multiCoreInnerIdx % this->gS1o / this->tilingData->coreParams.s1OuterSize;
    this->s1oIdx = multiCoreInnerIdx % this->tilingData->coreParams.s1OuterSize;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::WaitBmm1Result()
{
    this->bmm1.WaitIterateAll();
    this->bmm1.End();
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::SetExtraInfo(
    SplitExtraInfo &extraInfo, int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit, int64_t multiCoreInnerIdx)
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

    extraInfo.s1RealSize = s1BaseSize;

    this->ComputeBmm1Tail(extraInfo);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::ComputeBmm1Tail(
    SplitExtraInfo &extraInfo)
{
    if (this->tilingData->inputParams.s1Size < (extraInfo.s1oIdx + 1) * this->s1BaseSize) {
        extraInfo.s1RealSize = this->tilingData->inputParams.s1Size - extraInfo.s1oIdx * this->s1BaseSize;
    }
    extraInfo.s2RealSize = this->s2BaseNratioSize;
    if (extraInfo.s2StartIdx + (extraInfo.s2LoopCount + 1) * extraInfo.s2RealSize > extraInfo.s2EndIdx) {
        extraInfo.s2RealSize = extraInfo.s2EndIdx - extraInfo.s2LoopCount * extraInfo.s2RealSize - extraInfo.s2StartIdx;
    }

    extraInfo.vecS1BaseSize = this->Min(s2BaseNratioSize / extraInfo.s2RealSize * 8, extraInfo.s1RealSize);
    extraInfo.realSplitN = this->CeilDiv(extraInfo.s1RealSize, extraInfo.vecS1BaseSize);

    if (dSize >= 256) {
        extraInfo.vec2S1BaseSize = 64 * 128 / dSize;
    } else {
        extraInfo.vec2S1BaseSize = extraInfo.s1RealSize;
    }
    
    return;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::IterateBmm1(
    SplitExtraInfo &extraInfo)
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
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::Bmm1SetTensorA(
    SplitExtraInfo &extraInfo)
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
    extraInfo.qCoreOffset = this->qCoreOffset;
    this->bmm1.SetTensorA(this->queryGm[extraInfo.qCoreOffset]);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::SetBmm1TensorB(
    SplitExtraInfo &extraInfo)
{
    // 计算gm上的offset
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;
    if (this->tilingData->inputParams.layoutType == this->layoutBsh) {
        // BSH/BSND
        bOffset = extraInfo.boIdx * this->n2S2D;
        s2Offset = extraInfo.s2StartIdx * this->n2D + extraInfo.s2LoopCount * this->s2BaseNratioN2D;
        n2Offset = extraInfo.n2oIdx * dSize;
    } else if (this->tilingData->inputParams.layoutType == this->layoutSbh) {
        // SBH/SBND
        s2Offset = extraInfo.s2StartIdx * this->bN2D + extraInfo.s2LoopCount * this->s2BaseNratioBN2D;
        bOffset = extraInfo.boIdx * this->n2D;
        n2Offset = extraInfo.n2oIdx * dSize;
    } else if (this->tilingData->inputParams.layoutType == this->layoutBnsd) {
        // BNSD
        bOffset = extraInfo.boIdx * this->n2S2D;
        n2Offset = extraInfo.n2oIdx * this->s2D;
        s2Offset = extraInfo.s2StartIdx * dSize + extraInfo.s2LoopCount * this->s2BaseNratioD;
    }
    int64_t kCoreOffset = bOffset + n2Offset + s2Offset;
    this->bmm1.SetTensorB(this->keyGm[kCoreOffset], true);
    this->bmm1.SetTail(extraInfo.s1RealSize, extraInfo.s2RealSize);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::ProcessVec1(
    SplitExtraInfo &extraInfo)
{
    LocalTensor<T> stage1PingTensor = this->stage1PingBuf.template Get<T>(); // t.a 32k
    LocalTensor<T> stage1PongTensor = this->stage1PongBuf.template Get<T>(); // i.a 32k

    for (uint32_t loopIdx = 0; loopIdx < extraInfo.realSplitN; loopIdx++) {
        extraInfo.vecS1TailSize = extraInfo.vecS1BaseSize;
        if (loopIdx == extraInfo.realSplitN - 1) {
            extraInfo.vecS1TailSize = extraInfo.s1RealSize - loopIdx * extraInfo.vecS1BaseSize;
        }
        // todo 考虑s2方向尾块
        if (loopIdx > 0) {
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
        }
        this->CopyInAndCastPse(stage1PingTensor, extraInfo, loopIdx);
        this->GetBmm1Result(extraInfo, stage1PongTensor, loopIdx);

        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        set_flag(PIPE_MTE2, PIPE_V, eventID);
        wait_flag(PIPE_MTE2, PIPE_V, eventID);

        this->ComputePse(stage1PingTensor, stage1PongTensor, extraInfo);

        this->CopyInAttenMask(extraInfo, loopIdx);
        pipe_barrier(PIPE_V);

        Muls(stage1PingTensor, stage1PongTensor, static_cast<T>(this->tilingData->inputParams.scaleValue),
            extraInfo.vecS1TailSize * extraInfo.s2RealSize);
        eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        set_flag(PIPE_MTE2, PIPE_V, eventID);
        wait_flag(PIPE_MTE2, PIPE_V, eventID);

        uint32_t softmaxSize = extraInfo.vecS1TailSize * extraInfo.s2RealSize;

        if (loopIdx > 0) {
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
        }
        this->ComputeAttenMask(extraInfo, stage1PingTensor);
        if (loopIdx < extraInfo.realSplitN - 1) {
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
        }

        if (loopIdx > 0) {
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
        }

        this->CopyInDropMask(extraInfo, loopIdx);
        this->SoftMaxCompute(extraInfo, stage1PingTensor, loopIdx);

        this->ComputeDropMask(extraInfo, stage1PingTensor);

        if (loopIdx < extraInfo.realSplitN - 1) {
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
        }

        pipe_barrier(PIPE_V);
        if constexpr (!IsSameType<T, INPUT_T>::value) {
            LocalTensor<INPUT_T> stage1CastTensor = this->commonTBuf.template Get<INPUT_T>();
            Cast(stage1CastTensor, stage1PingTensor, RoundMode::CAST_ROUND,
                extraInfo.vecS1TailSize * extraInfo.s2RealSize);

            eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            set_flag(PIPE_V, PIPE_MTE3, eventID);
            wait_flag(PIPE_V, PIPE_MTE3, eventID);

            if (extraInfo.taskId % 2 == 0) {
                DataCopy(this->stage1ResPing[loopIdx * extraInfo.vecS1BaseSize * extraInfo.s2RealSize],
                    stage1CastTensor, extraInfo.vecS1TailSize * extraInfo.s2RealSize);
            } else {
                DataCopy(this->stage1ResPong[loopIdx * extraInfo.vecS1BaseSize * extraInfo.s2RealSize],
                    stage1CastTensor, extraInfo.vecS1TailSize * extraInfo.s2RealSize);
            }
        }
        if (loopIdx < extraInfo.realSplitN - 1) {
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
        }
    }
    return;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::CopyInAndCastPse(
    LocalTensor<T> &castTensor, SplitExtraInfo &extraInfo, int64_t loopIdx)
{
    if (!this->hasPse) {
        return;
    }
    int64_t offset = this->ComputePseOffset(extraInfo, loopIdx);
    LocalTensor<INPUT_T> pseUb = this->pseTBuf.template Get<INPUT_T>();

    int32_t s1Size = this->tilingData->inputParams.pseShapeType == this->pse1S2 ? 1 : extraInfo.vecS1TailSize;
    this->DataCopyIn(pseUb, this->pseGm, offset, s1Size, extraInfo.s2RealSize);

    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    set_flag(PIPE_MTE2, PIPE_V, eventID);
    wait_flag(PIPE_MTE2, PIPE_V, eventID);
    if constexpr (!IsSameType<INPUT_T, T>::value) {
        Cast(castTensor, pseUb, RoundMode::CAST_NONE, s1Size * extraInfo.s2RealSize);
    } else {
        DataCopy(castTensor, pseUb, s1Size * extraInfo.s2RealSize);
    }
    return;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline int64_t FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::ComputePseOffset(
    SplitExtraInfo &extraInfo, int64_t loopIdx)
{
    if (!this->hasPse) {
        return 0;
    }
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s1Offset = 0;
    int64_t s2Offset = extraInfo.s2StartIdx + extraInfo.s2LoopCount * s2BaseNratioSize;
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
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::CopyInAttenMask(
    SplitExtraInfo &extraInfo, int64_t loopIdx)
{
    if (!this->hasAttenMask) {
        return;
    }
    LocalTensor<uint8_t> attenMaskUb = this->maskTBufPing.template Get<uint8_t>();
    int64_t attenMaskOffset = this->ComputeAttenMaskOffset(extraInfo, loopIdx);
    BoolCopyIn(attenMaskUb, this->attenMaskGmInt, attenMaskOffset, extraInfo.vecS1TailSize, extraInfo.s2RealSize);
    return;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline int64_t FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T,
    isBasicBlock>::ComputeAttenMaskOffset(SplitExtraInfo &extraInfo, int64_t loopIdx)
{
    if (this->tilingData->inputParams.attenMaskCompressMode == this->attenMaskNoCompress) {
        return this->ComputeOffsetForNoCompress(extraInfo, loopIdx);
    } else if (this->tilingData->inputParams.attenMaskCompressMode == this->attenMaskLeftUpCausalCompress) {
        int64_t delta = extraInfo.s1oIdx * s1BaseSize + loopIdx * extraInfo.vecS1BaseSize - extraInfo.s2StartIdx -
                        extraInfo.s2LoopCount * s2BaseNratioSize;
        return this->ComputeOffsetForCausal(delta, extraInfo.vecS1BaseSize, s2BaseNratioSize,
                                            this->tilingData->inputParams.attenMaskS2Size);
    } else if (this->tilingData->inputParams.attenMaskCompressMode == this->attenMaskRightDownCausalCompress) {
        int64_t delta = extraInfo.s1oIdx * s1BaseSize + loopIdx * extraInfo.vecS1BaseSize - extraInfo.s2StartIdx -
                        extraInfo.s2LoopCount * s2BaseNratioSize + this->tilingData->inputParams.s2Size -
                        this->tilingData->inputParams.s1Size;
        return this->ComputeOffsetForCausal(delta, extraInfo.vecS1BaseSize, s2BaseNratioSize,
            this->tilingData->inputParams.attenMaskS2Size);
    }
    return 0;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline int64_t FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::ComputeOffsetForNoCompress(SplitExtraInfo &extraInfo, int64_t loopIdx)
{
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;
    int64_t s1Offset =
        extraInfo.s1oIdx * this->s1BaseS2 + loopIdx * extraInfo.vecS1BaseSize * this->tilingData->inputParams.s2Size;
    int64_t s2Offset = extraInfo.s2StartIdx + extraInfo.s2LoopCount * s2BaseNratioSize;
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
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::GetBmm1Result(
    SplitExtraInfo &extraInfo, LocalTensor<T> &bmm1ResUb, int64_t loopIdx)
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
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::ComputePse(
    LocalTensor<T> &pseUb, LocalTensor<T> &bmm1ResUb, SplitExtraInfo &extraInfo)
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
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::ComputeAttenMask(
    SplitExtraInfo &extraInfo, LocalTensor<T> &bmm1ResUb)
{
    if (!this->hasAttenMask) {
        return;
    }

    LocalTensor<uint8_t> attenMaskUb = this->maskTBufPing.template Get<uint8_t>();
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
    return;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::CopyInDropMask(
    SplitExtraInfo &extraInfo, int64_t loopIdx)
{
    if (!this->hasDropMask) {
        return;
    }
    // softmax的计算结果需要执行dropout操作
    int64_t dropMaskOffset = this->ComputeDropOffset(extraInfo, loopIdx);
    LocalTensor<uint8_t> dropMaskUb = this->maskTBufPong.template Get<uint8_t>();
    // 仅支持对齐场景，非对齐使用boolCopyIn
    this->Bit2Int8CopyIn(dropMaskUb, this->dropMaskGm, dropMaskOffset, extraInfo.vecS1TailSize, extraInfo.s2RealSize,
        1);
    return;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline int64_t FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::ComputeDropOffset(
    SplitExtraInfo &extraInfo, int64_t loopIdx)
{
    int64_t bOffset = extraInfo.boIdx * this->n2GS1S2;
    int64_t n2Offset = extraInfo.n2oIdx * this->gS1S2;
    int64_t gOffset = extraInfo.goIdx * this->s1S2;

    int64_t s1Offset =
        extraInfo.s1oIdx * this->s1BaseS2 + loopIdx * extraInfo.vecS1BaseSize * this->tilingData->inputParams.s2Size;
    int64_t s2Offset = extraInfo.s2StartIdx + extraInfo.s2LoopCount * s2BaseNratioSize;

    return bOffset + n2Offset + gOffset + s1Offset + s2Offset;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::SoftMaxCompute(
    SplitExtraInfo &extraInfo, LocalTensor<T> &srcTensor, int64_t loopIdx)
{
    uint32_t bmm1ResUbShape[] = {static_cast<uint32_t>(extraInfo.vecS1TailSize), static_cast<uint32_t>(extraInfo.s2RealSize)};
    srcTensor.SetShapeInfo(ShapeInfo(2, bmm1ResUbShape, DataFormat::ND));

    uint32_t maxSumShape[] = {static_cast<uint32_t>(extraInfo.vecS1TailSize), static_cast<uint32_t>(this->fp32BaseSize)};
    LocalTensor<float> sumUb;
    if (extraInfo.multiCoreInnerIdx % 2 == 0) {
        sumUb = this->softmaxSumPingBuf.template Get<float>()[loopIdx * extraInfo.vecS1BaseSize * this->fp32BaseSize];
    } else {
        sumUb = this->softmaxSumPongBuf.template Get<float>()[loopIdx * extraInfo.vecS1BaseSize * this->fp32BaseSize];
    }
    LocalTensor<float> maxUb =
        this->softmaxMaxBuf.template Get<float>()[loopIdx * extraInfo.vecS1BaseSize * this->fp32BaseSize];

    sumUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));
    maxUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));

    uint32_t expShape[] = {static_cast<uint32_t>(extraInfo.vecS1TailSize), static_cast<uint32_t>(this->blockBytes / sizeof(T))};
    LocalTensor<T> expUb;
    if (extraInfo.taskId % 2 == 0) {
        expUb =
            this->softmaxExpPingBuf.template Get<T>()[loopIdx * extraInfo.vecS1BaseSize * this->blockBytes / sizeof(T)];
    } else {
        expUb =
            this->softmaxExpPongBuf.template Get<T>()[loopIdx * extraInfo.vecS1BaseSize * this->blockBytes / sizeof(T)];
    }
    expUb.SetShapeInfo(ShapeInfo(2, expShape, DataFormat::ND));

    LocalTensor<uint8_t> apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();
    pipe_barrier(PIPE_V);
    if (unlikely(extraInfo.s2LoopCount == 0)) {
        SoftmaxFlashV2<T, false, true, true>(srcTensor, sumUb, maxUb, srcTensor, expUb, sumUb, maxUb, apiTmpBuffer,
            this->tilingData->softmaxFlashTilingData);
    } else {
        SoftmaxFlashV2<T, true, true, true>(srcTensor, sumUb, maxUb, srcTensor, expUb, sumUb, maxUb, apiTmpBuffer,
            this->tilingData->softmaxFlashTilingData);
    }

    if (extraInfo.s2LoopCount == extraInfo.s2LoopLimit) {
        int64_t bOffset = extraInfo.boIdx * this->n2GS1 * this->fp32BaseSize;
        int64_t n2Offset = extraInfo.n2oIdx * this->gS1 * this->fp32BaseSize;
        int64_t gOffset = extraInfo.goIdx * this->tilingData->inputParams.s1Size * this->fp32BaseSize;
        int64_t s1Offset = extraInfo.s1oIdx * s1BaseSize * this->fp32BaseSize;
        int64_t gmOffset = bOffset + n2Offset + gOffset + s1Offset;
        int64_t calculateSize = extraInfo.s1RealSize * this->fp32BaseSize;
        LocalTensor<float> maxTensor = this->softmaxMaxBuf.template Get<float>();
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        set_flag(PIPE_V, PIPE_MTE3, eventID);
        wait_flag(PIPE_V, PIPE_MTE3, eventID);
        DataCopy(this->softmaxMaxGm[gmOffset], maxTensor, calculateSize);
        eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        set_flag(PIPE_MTE3, PIPE_V, eventID);
        wait_flag(PIPE_MTE3, PIPE_V, eventID);
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::ComputeDropMask(
    SplitExtraInfo &extraInfo, LocalTensor<T> &bmm1ResUb)
{
    if (!this->hasDropMask) {
        return;
    }
    LocalTensor<uint8_t> apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();

    // softmax的计算结果需要执行dropout操作
    LocalTensor<uint8_t> dropMaskUb = this->maskTBufPong.template Get<uint8_t>();
    pipe_barrier(PIPE_V);

    DropOutShapeInfo dropOutShapeInfo;
    dropOutShapeInfo.firstAxis = extraInfo.vecS1TailSize;
    dropOutShapeInfo.srcLastAxis = extraInfo.s2RealSize;

    dropOutShapeInfo.maskLastAxis =
        this->CeilDiv(extraInfo.s2RealSize / this->byteBitRatio, this->blockBytes) * this->blockBytes;
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    set_flag(PIPE_MTE2, PIPE_V, eventID);
    wait_flag(PIPE_MTE2, PIPE_V, eventID);
    DropOut(bmm1ResUb, bmm1ResUb, dropMaskUb, apiTmpBuffer, this->tilingData->inputParams.keepProb, dropOutShapeInfo);
    return;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::WaitBmm2Result()
{
    this->bmm2.WaitIterateAll();
    this->bmm2.End();
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::IterateBmm2(
    SplitExtraInfo &extraInfo)
{
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;
    int64_t bStrideSize = 0;

    if (this->tilingData->inputParams.layoutType == this->layoutBsh) {
        // BSH/BSND
        bOffset = extraInfo.boIdx * this->n2S2D;
        s2Offset = extraInfo.s2StartIdx * this->n2D + extraInfo.s2LoopCount * s2BaseNratioSize * this->n2D;
        n2Offset = extraInfo.n2oIdx * dSize;
        bStrideSize = this->n2D;
    } else if (this->tilingData->inputParams.layoutType == this->layoutSbh) {
        // SBH/SBND
        s2Offset = extraInfo.s2StartIdx * this->bN2D + extraInfo.s2LoopCount * s2BaseNratioSize * this->bN2D;
        bOffset = extraInfo.boIdx * this->n2D;
        n2Offset = extraInfo.n2oIdx * dSize;
        bStrideSize = this->bN2D;
    } else if (this->tilingData->inputParams.layoutType == this->layoutBnsd) {
        // BNSD
        bOffset = extraInfo.boIdx * this->n2S2D;
        n2Offset = extraInfo.n2oIdx * this->s2D;
        s2Offset = extraInfo.s2StartIdx * dSize + extraInfo.s2LoopCount * s2BaseNratioSize * dSize;
        bStrideSize = dSize;
    }

    int64_t vCoreOffset = bOffset + n2Offset + s2Offset;
    this->bmm2.SetOrgShape(this->tilingData->inputParams.s1Size, bStrideSize, extraInfo.s2RealSize, bStrideSize, dSize);
    if (extraInfo.taskId % 2 == 0) {
        this->bmm2.SetTensorA(this->stage1ResPing);
    } else {
        this->bmm2.SetTensorA(this->stage1ResPong);
    }

    this->bmm2.SetTensorB(this->valueGm[vCoreOffset]);
    this->bmm2.SetTail(-1, -1, extraInfo.s2RealSize);

    if (extraInfo.taskId % 2 == 0) {
        this->bmm2.template IterateAll<false>(this->mm2ResPing, false, false, true);
    } else {
        this->bmm2.template IterateAll<false>(this->mm2ResPong, false, false, true);
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::ProcessVec2(
    SplitExtraInfo &extraInfo)
{
    // 获取缓存bmm2的计算结果
    LocalTensor<T> bmm2ResUb = this->stage2TBuf.template Get<T>();
    LocalTensor<T> stage2BufTensor = this->commonTBuf.template Get<T>();
    int64_t vec2LoopLimit = extraInfo.s1RealSize / extraInfo.vec2S1BaseSize;

    event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t mte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    event_t mte3WaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    event_t mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    for (int64_t s1oIdx = 0; s1oIdx < vec2LoopLimit; s1oIdx++) {
        int64_t mm2ResCalcSize = extraInfo.vec2S1BaseSize * dSize;
        int64_t mm2ResOffset = s1oIdx * mm2ResCalcSize;
        set_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3);
        wait_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3);

        if (extraInfo.taskId % 2 == 0) {
            DataCopy(stage2BufTensor, this->mm2ResPing[mm2ResOffset], mm2ResCalcSize);
        } else {
            DataCopy(stage2BufTensor, this->mm2ResPong[mm2ResOffset], mm2ResCalcSize);
        }

        if (vec2LoopLimit > 1) {
            set_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3);
            wait_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3);
            if (extraInfo.multiCoreInnerIdx % 2 == 0) {
                DataCopy(bmm2ResUb, this->vec2ResPing[mm2ResOffset], mm2ResCalcSize);
            } else {
                DataCopy(bmm2ResUb, this->vec2ResPong[mm2ResOffset], mm2ResCalcSize);
            }
        }

        set_flag(PIPE_MTE2, PIPE_V, vWaitMte2);
        wait_flag(PIPE_MTE2, PIPE_V, vWaitMte2);
        if (unlikely(extraInfo.s2LoopCount == 0)) {
            DataCopy(bmm2ResUb, stage2BufTensor, mm2ResCalcSize);
        } else {
            this->Bmm2ResultMul(extraInfo, bmm2ResUb, s1oIdx);
            pipe_barrier(PIPE_V);
            Add(bmm2ResUb, bmm2ResUb, stage2BufTensor, mm2ResCalcSize);
        }

        if (extraInfo.s2LoopCount == extraInfo.s2LoopLimit) {
            Bmm2ResultDiv(extraInfo, s1oIdx);
            Bmm2DataCopyOut(extraInfo, s1oIdx);
            SoftmaxDataCopyOut(extraInfo);
        } else if (vec2LoopLimit > 1) {
            set_flag(PIPE_MTE2, PIPE_MTE3, mte3WaitMte2);
            wait_flag(PIPE_MTE2, PIPE_MTE3, mte3WaitMte2);
            set_flag(PIPE_V, PIPE_MTE3, mte3WaitV);
            wait_flag(PIPE_V, PIPE_MTE3, mte3WaitV);
            if (extraInfo.multiCoreInnerIdx % 2 == 0) {
                DataCopy(this->vec2ResPing[mm2ResOffset], bmm2ResUb, mm2ResCalcSize);
            } else {
                DataCopy(this->vec2ResPong[mm2ResOffset], bmm2ResUb, mm2ResCalcSize);
            }
            set_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3);
            wait_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3);
        }
    }
    return;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::Bmm2ResultMul(
    SplitExtraInfo &extraInfo, LocalTensor<T> &bmm2ResUb, int64_t s1oIdx)
{
    pipe_barrier(PIPE_V);
    LocalTensor<T> expUb;
    if (extraInfo.taskId % 2 == 0) {
        expUb = softmaxExpPingBuf.Get<T>();
    } else {
        expUb = softmaxExpPongBuf.Get<T>();
    }

    BinaryRepeatParams repeatParams;
    repeatParams.src0BlkStride = 0;
    repeatParams.src0RepStride = 1;
    repeatParams.src1RepStride = dSize / this->blockSize;
    repeatParams.dstRepStride = dSize / this->blockSize;

    // s1长度可能会超过255限制，修改成双重循环
    // 根据一次最多计算的byte数量，对bmm2Res分组mul
    int32_t loop = dSize / this->repeatMaxSize;
    int32_t remain = dSize % this->repeatMaxSize;
    for (int i = 0; i < loop; i++) {
        Mul(bmm2ResUb[i * this->repeatMaxSize], expUb[s1oIdx * extraInfo.vec2S1BaseSize * 8],
            bmm2ResUb[i * this->repeatMaxSize], this->repeatMaxSize, extraInfo.vec2S1BaseSize, repeatParams);
    }
    if (likely(remain)) {
        Mul(bmm2ResUb[loop * this->repeatMaxSize], expUb[s1oIdx * extraInfo.vec2S1BaseSize * 8],
            bmm2ResUb[loop * this->repeatMaxSize], remain, extraInfo.vec2S1BaseSize, repeatParams);
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::Bmm2ResultDiv(
    SplitExtraInfo &extraInfo, int64_t s1oIdx)
{
    LocalTensor<T> bmm2ResUb = this->stage2TBuf.template Get<T>();

    BinaryRepeatParams repeatParams;
    repeatParams.src0BlkStride = 1;
    repeatParams.src0RepStride = dSize / this->blockSize;
    repeatParams.src1BlkStride = 0;
    repeatParams.src1RepStride = 1;
    repeatParams.dstRepStride = dSize / this->blockSize;
    int32_t loop = dSize / this->repeatMaxSize;
    int32_t remain = dSize % this->repeatMaxSize;

    LocalTensor<float> sumUb;
    if (extraInfo.multiCoreInnerIdx % 2 == 0) {
        sumUb = softmaxSumPingBuf.Get<float>();
    } else {
        sumUb = softmaxSumPongBuf.Get<float>();
    }

    int32_t calcSize = sumUb.GetSize();
    // 用optionalInputQueue的queue
    pipe_barrier(PIPE_V);
    if constexpr (IsSameType<T, half>::value) {
        LocalTensor<float> commonBufTensor = this->commonTBuf.template Get<float>();
        Copy(commonBufTensor, sumUb, 64, calcSize / 64, { 2, 1, 16, 8 });
        Copy(commonBufTensor[8], sumUb, 64, calcSize / 64, { 2, 1, 16, 8 });
        LocalTensor<T> sumCastTensor = softmaxExpPingBuf.Get<T>(this->tilingData->tensorSizeParams.softmaxExpUbSize);

        Cast(sumCastTensor, commonBufTensor, RoundMode::CAST_ROUND, 2 * sumUb.GetSize());
        for (int i = 0; i < loop; i++) {
            Div(bmm2ResUb[i * this->repeatMaxSize], bmm2ResUb[i * this->repeatMaxSize], sumCastTensor,
                this->repeatMaxSize, extraInfo.vec2S1BaseSize, repeatParams);
        }
        if (likely(remain)) {
            Div(bmm2ResUb[loop * this->repeatMaxSize], bmm2ResUb[loop * this->repeatMaxSize], sumCastTensor, remain,
                extraInfo.vec2S1BaseSize, repeatParams);
        }
    } else {
        for (int i = 0; i < loop; i++) {
            Div(bmm2ResUb[i * this->repeatMaxSize], bmm2ResUb[i * this->repeatMaxSize],
                sumUb[s1oIdx * extraInfo.vec2S1BaseSize * 8], this->repeatMaxSize, extraInfo.vec2S1BaseSize,
                repeatParams);
        }
        if (likely(remain)) {
            Div(bmm2ResUb[loop * this->repeatMaxSize], bmm2ResUb[loop * this->repeatMaxSize],
                sumUb[s1oIdx * extraInfo.vec2S1BaseSize * 8], remain, extraInfo.vec2S1BaseSize, repeatParams);
        }
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::Bmm2DataCopyOut(
    SplitExtraInfo &extraInfo, int64_t s1oIdx)
{
    LocalTensor<T> bmm2ResUb = this->stage2TBuf.template Get<T>();
    LocalTensor<INPUT_T> attenOut = this->stage2TBuf.template Get<INPUT_T>();
    int32_t calcSize = bmm2ResUb.GetSize();
    pipe_barrier(PIPE_V);

    if constexpr (!IsSameType<INPUT_T, T>::value) {
        Cast(attenOut, bmm2ResUb, RoundMode::CAST_ROUND, calcSize);
    }

    event_t mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    set_flag(PIPE_V, PIPE_MTE3, mte3WaitV);
    wait_flag(PIPE_V, PIPE_MTE3, mte3WaitV);

    DataCopyParams dataCopyParams;
    dataCopyParams.blockLen = dSize * sizeof(INPUT_T);
    dataCopyParams.srcStride = 0;
    int64_t dstStride = 0;
    int64_t attenOutOffset = dSize;
    if (this->tilingData->inputParams.layoutType == this->layoutBsh) {
        attenOutOffset = this->n2D;
        dstStride =
            (this->tilingData->inputParams.n2Size * this->tilingData->inputParams.gSize - 1) * dSize * sizeof(INPUT_T);
    } else if (this->tilingData->inputParams.layoutType == this->layoutSbh) {
        attenOutOffset = this->bN2GD;
        dstStride = (this->tilingData->inputParams.bSize * this->tilingData->inputParams.n2Size *
            this->tilingData->inputParams.gSize -
            1) *
            dSize * sizeof(INPUT_T);
    }

    // dataCopyParams.dstStride类型定义uint16_t，65535是其最大值
    if (likely(dstStride <= 65535)) {
        dataCopyParams.blockCount = extraInfo.vec2S1BaseSize;
        dataCopyParams.dstStride = static_cast<uint16_t>(dstStride);
        DataCopyPad(this->attentionOutGm[extraInfo.qCoreOffset + s1oIdx * extraInfo.vec2S1BaseSize * attenOutOffset],
            attenOut, dataCopyParams);
    } else {
        dataCopyParams.blockCount = 1;
        dataCopyParams.dstStride = 0;
        int64_t datacopyOffset = dSize;
        if (this->tilingData->inputParams.layoutType == this->layoutBsh) {
            datacopyOffset = this->n2GD;
        } else if (this->tilingData->inputParams.layoutType == this->layoutSbh) {
            datacopyOffset = this->bN2G * dSize;
        }

        for (uint32_t i = 0; i < extraInfo.vec2S1BaseSize; i++) {
            DataCopyPad(this->attentionOutGm[extraInfo.qCoreOffset +
                s1oIdx * extraInfo.vec2S1BaseSize * attenOutOffset + i * datacopyOffset],
                attenOut[i * dSize], dataCopyParams);
        }
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::SoftmaxDataCopyOut(
    SplitExtraInfo &extraInfo)
{
    int64_t bOffset = extraInfo.boIdx * this->n2GS1 * this->fp32BaseSize;
    int64_t n2Offset = extraInfo.n2oIdx * this->gS1 * this->fp32BaseSize;
    int64_t gOffset = extraInfo.goIdx * this->tilingData->inputParams.s1Size * this->fp32BaseSize;
    int64_t s1Offset = extraInfo.s1oIdx * s1BaseSize * this->fp32BaseSize;
    int64_t gmOffset = bOffset + n2Offset + gOffset + s1Offset;
    int64_t calculateSize = extraInfo.s1RealSize * this->fp32BaseSize;

    LocalTensor<float> sumTensor;
    if (extraInfo.multiCoreInnerIdx % 2 == 0) {
        sumTensor = softmaxSumPingBuf.template Get<float>();
    } else {
        sumTensor = softmaxSumPongBuf.template Get<float>();
    }
    DataCopy(this->softmaxSumGm[gmOffset], sumTensor, calculateSize);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::DataCopyIn(
    LocalTensor<INPUT_T> &dstTensor, GlobalTensor<INPUT_T> &srcTensor, int64_t offset, int32_t s1Size, int32_t s2Size)
{
    int32_t dtypeSize = sizeof(INPUT_T);
    int32_t alignedS2Size = this->Align(s2Size);
    uint32_t shapeArray[] = {static_cast<uint32_t>(s1Size), static_cast<uint32_t>(alignedS2Size)};
    dstTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    dstTensor.SetSize(s1Size * alignedS2Size);
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = s1Size;
    dataCopyParams.blockLen = this->CeilDiv(s2Size * dtypeSize, this->blockBytes); // 单位32B
    dataCopyParams.dstStride = 0;                                                  // gap
    if (this->tilingData->inputParams.s2Size * dtypeSize % this->blockBytes == 0) {
        dataCopyParams.srcStride =
            (this->tilingData->inputParams.s2Size * dtypeSize - dataCopyParams.blockLen * this->blockBytes) /
            this->blockBytes; // srcGap
        DataCopy(dstTensor, srcTensor[offset], dataCopyParams);
    } else {
        dataCopyParams.blockLen = s2Size * dtypeSize; // 单位Byte
        dataCopyParams.srcStride = (this->tilingData->inputParams.s2Size * dtypeSize - dataCopyParams.blockLen);
        dataCopyParams.dstStride = (alignedS2Size - s2Size) * dtypeSize;
        DataCopyPadParams dataCopyPadParams;
        dataCopyPadParams.rightPadding = (alignedS2Size - s2Size) * dtypeSize;
        DataCopyPad(dstTensor, srcTensor[offset], dataCopyParams, dataCopyPadParams);
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::BoolCopyIn(
    LocalTensor<uint8_t> &dstTensor, GlobalTensor<uint8_t> &srcTensor, int64_t offset, int32_t s1Size, int32_t s2Size)
{
    int32_t alignedS2Size = this->CeilDiv(s2Size, this->blockBytes) * this->blockBytes;
    uint32_t shapeArray[] = {static_cast<uint32_t>(s1Size), static_cast<uint32_t>(alignedS2Size)};
    dstTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    dstTensor.SetSize(s1Size * alignedS2Size);
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = s1Size;
    dataCopyParams.dstStride = 0;
    if (s2Size % this->blockBytes == 0) {
        dataCopyParams.blockLen = alignedS2Size / this->blockBytes;
        dataCopyParams.srcStride = (this->tilingData->inputParams.attenMaskS2Size - alignedS2Size) / this->blockBytes;
        DataCopy(dstTensor, srcTensor[offset], dataCopyParams);
    } else {
        dataCopyParams.blockLen = s2Size;
        dataCopyParams.srcStride = this->tilingData->inputParams.attenMaskS2Size - s2Size;
        DataCopyPadParams dataCopyPadParams;
        DataCopyPad(dstTensor, srcTensor[offset], dataCopyParams, dataCopyPadParams);
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf<INPUT_T, T, isBasicBlock>::Bit2Int8CopyIn(
    LocalTensor<uint8_t> &dstTensor, GlobalTensor<uint8_t> &srcTensor, int64_t offset, int32_t s1Size, int32_t s2Size,
    int32_t batchSize)
{
    int32_t alignedS2Size = this->CeilDiv(s2Size, 32) * 32;
    uint32_t shapeArray[] = {static_cast<uint32_t>(batchSize * s1Size), static_cast<uint32_t>(alignedS2Size)};
    dstTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    dstTensor.SetSize(batchSize * s1Size * alignedS2Size);
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = batchSize * s1Size;
    dataCopyParams.blockLen = this->CeilDiv(s2Size / this->byteBitRatio, this->blockBytes);
    dataCopyParams.dstStride = 0; // gap
    if (this->tilingData->inputParams.s2Size / this->byteBitRatio % this->blockBytes == 0) {
        dataCopyParams.srcStride =
            (this->tilingData->inputParams.s2Size / this->byteBitRatio - dataCopyParams.blockLen * this->blockBytes) /
            this->blockBytes; // srcGap
        DataCopy(dstTensor, srcTensor[offset / this->byteBitRatio], dataCopyParams);
    } else {
        dataCopyParams.blockLen = s2Size / this->byteBitRatio;
        dataCopyParams.srcStride =
            (this->tilingData->inputParams.s2Size / this->byteBitRatio - dataCopyParams.blockLen);
        DataCopyPadParams dataCopyPadParams;
        DataCopyPad(dstTensor, srcTensor[offset / this->byteBitRatio], dataCopyParams, dataCopyPadParams);
    }
}

#endif // FLASH_ATTENTION_SCORE_SPLIT_B_N2_G_S1_S2_DB_PERFORMANCE_H
