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

#ifndef FLASH_ATTENTION_SCORE_SPLIT_B_N2_G_S1_D_H
#define FLASH_ATTENTION_SCORE_SPLIT_B_N2_G_S1_D_H

#include "flash_attention_score_common.h"
#include "kernel_data_copy_transpose.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"


// INPUT_T - means data type for input
// T       - means data type when calc
template <typename INPUT_T, typename T = INPUT_T, bool isBasicBlock = false>
class FlashAttentionScoreSplitBn2gs1S1d : public FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock> {
public:
    __aicore__ inline FlashAttentionScoreSplitBn2gs1S1d() {};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
        __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax,
        __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
        const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

    // define bmm1
    using a1Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T>;
    using b1Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, true>;
    using bias1Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c1Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND_ALIGN, T>;
    matmul::Matmul<a1Type, b1Type, c1Type, bias1Type> bmm1;

    // define bmm2
    using a2Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T>;
    using b2Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T>;
    using bias2Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c2Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T>;
    matmul::Matmul<a2Type, b2Type, c2Type, bias2Type> bmm2;

protected:
    // 主流程函数
    __aicore__ inline void SplitNGroups(int64_t splitBaseNum, int64_t multiCoreInnerLimit);
    __aicore__ inline void ProcessBlock(int64_t countBase);
    __aicore__ inline void InitSoftmaxTensor();
    __aicore__ inline void ProcessStage1(int64_t splitNIdx);
    // offset计算相关函数
    __aicore__ inline void ComputeAxisStrideSize();
    __aicore__ inline void ComputeIdx();
    __aicore__ inline void SetBmm1ResShape(AscendC::LocalTensor<T> &bmm1ResUb);
    __aicore__ inline int64_t ComputePseOffset();
    __aicore__ inline int64_t ComputeAttenMaskOffset();
    __aicore__ inline int64_t ComputeOffsetForNoCompress();
    __aicore__ inline int64_t ComputeDropOffset();
    __aicore__ inline int64_t ComputeSoftmaxOutOffset();
    __aicore__ inline void Bmm1SetTensorA();
    __aicore__ inline void Bmm1SetTensorB();
    __aicore__ inline void Bmm1SetTail(int64_t countBase);
    __aicore__ inline void Bmm2Compute(int64_t tmpS1oIdx);
    // 计算相关代码
    __aicore__ inline void SoftmaxCompute(AscendC::LocalTensor<T> &dstTensor, AscendC::LocalTensor<T> &srcTensor);
    // 搬运相关代码
    __aicore__ inline void CopyInBmm1Result(AscendC::LocalTensor<T> &bmm1ResUb, int64_t splitNIdx);

    // workspace
    AscendC::GlobalTensor<T> bmm1WorkspaceGm;
    AscendC::GlobalTensor<INPUT_T> stage1WorkspaceGm;

    // 用于计算各轴idx
    int64_t gS1o;
    int64_t n2GS1o;

    // 当前base块的idx
    int64_t multiCoreInnerIdx;

    // S1轴尾块，用于S1轴处理N:1的尾块处理
    int32_t s1TailSize;

    // 一个N:1组的S1轴长度总和
    int64_t s1NGroupRealSize;

    // S1轴base块实际长度
    int32_t s1RealSize;

    // 用于计算不同layout下的各轴stride
    // Q、bmm2Output
    int64_t qBStrideSize;
    int64_t qN2StrideSize;
    int64_t qGStrideSize;
    int64_t qS1oStrideSize;
    // K、V
    int64_t kvBoStrideSize;
    int64_t kvN2oStrideSize;
    // attenMask
    int64_t attenGStrideSizeType1;
    int64_t attenN2StrideSizeType1;
    int64_t attenBStrideSizeType1;
    int64_t attenBStrideSizeType2;
    int64_t attenS1oStrideSize;
    // dropMask
    int64_t dropBStrideSize;
    int64_t dropN2StrideSize;
    int64_t dropGStrideSize;
    int64_t dropS1oStrideSize;
    // pse
    int64_t pseBStrideSizeType1;
    int64_t pseN2StrideSizeType1;
    int64_t pseGStrideSizeType1;
    int64_t pseS1oStrideSizeType1;
    int64_t pseBStrideSizeType2;
    int64_t pseN2StrideSizeType2;
    int64_t pseGStrideSizeType2;
    // softmaxOutput
    int64_t softmaxBStrideSize;
    int64_t softmaxN2StrideSize;
    int64_t softmaxGStrideSize;
    int64_t softmaxS1oStrideSize;
    // bmm1 workspace
    int64_t bmm1WorkspaceStrideSize;
};

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1d<INPUT_T, T, isBasicBlock>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse, __gm__ uint8_t *dropMask,
    __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
    __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe)
{
    this->BaseInit(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
        attentionOut, workspace, tiling, tPipe);

    // 申请两块workspace空间，分别保存bmm1和stage1的计算结果
    this->bmm1WorkspaceStrideSize = this->tilingData->coreParams.s1BaseSize * this->tilingData->coreParams.s2BaseSize;
    int64_t bmm1WorkspaceSize = this->bmm1WorkspaceStrideSize * this->tilingData->coreParams.nRatio;
    int64_t workspaceOffset =
        this->blockIdx * (bmm1WorkspaceSize * (sizeof(T) + sizeof(INPUT_T)));

    this->bmm1WorkspaceGm.SetGlobalBuffer((__gm__ T *)(workspace + workspaceOffset));
    this->stage1WorkspaceGm.SetGlobalBuffer((__gm__ INPUT_T *)(workspace + workspaceOffset +
        bmm1WorkspaceSize * sizeof(T)));
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1d<INPUT_T, T, isBasicBlock>::Process()
{
    this->s1TailSize = this->tilingData->coreParams.s1BaseTailSize;

    // 计算各轴stride，只需计算一次
    this->ComputeAxisStrideSize();

    // 确定核内切分起点
    // 当前S1 base块的偏移量
    this->multiCoreInnerIdx = this->blockIdx * this->tilingData->multiCoreParams.splitFactorSize;
    int64_t multiCoreInnerLimit = this->Min(this->multiCoreInnerIdx + this->tilingData->multiCoreParams.splitFactorSize,
        static_cast<int64_t>(this->tilingData->multiCoreParams.totalSize));

    // 计算轴的stride
    // N轴base块单位个数——G*S1o
    this->gS1o = this->tilingData->inputParams.gSize * this->tilingData->coreParams.s1OuterSize;
    // B轴base块单位个数——N2*G*S1o
    this->n2GS1o = this->tilingData->inputParams.n2Size * this->gS1o;

    // 起始offset可能在S1块内部，此时需要将这个“不完整”的S1块单独处理
    // 起始进行N:1分组的base块个数
    int64_t firstSplitBaseNum = this->Min(this->tilingData->coreParams.s1OuterSize -
        this->multiCoreInnerIdx % this->tilingData->coreParams.s1OuterSize,
        static_cast<int64_t>(this->tilingData->multiCoreParams.splitFactorSize));

    // 为了不在for循环里重复设置K矩阵，将该操作提到循环外面
    this->boIdx = this->multiCoreInnerIdx / this->n2GS1o;
    this->n2oIdx = this->multiCoreInnerIdx % this->n2GS1o / this->gS1o;
    this->Bmm1SetTensorB();

    this->SplitNGroups(firstSplitBaseNum, multiCoreInnerLimit);

    // 起始块计算完成后，计算剩下整块，整块的起始地址是一个新的S1，但单核base块很多时，每次跨轴都进一个新的循环
    int64_t sOuterLoopTotalTimes =
        this->CeilDiv((this->tilingData->multiCoreParams.splitFactorSize - firstSplitBaseNum),
        this->tilingData->coreParams.s1OuterSize);
    for (int64_t sOuterLoopIdx = 0;
        sOuterLoopIdx < sOuterLoopTotalTimes && this->multiCoreInnerIdx < multiCoreInnerLimit; sOuterLoopIdx++) {
        // 当前进行N:1分组的base块个数
        int64_t splitBaseNum = this->Min(static_cast<int64_t>(this->tilingData->coreParams.s1OuterSize),
            this->tilingData->multiCoreParams.splitFactorSize - firstSplitBaseNum);
        firstSplitBaseNum += splitBaseNum;

        // 为了不在for循环里重复设置K矩阵，将该操作提到循环外面
        // 计算各轴idx
        this->ComputeIdx();
        // 当gIdx和s1oIdx都为0时，说明刚切换G轴，因此重新设置K矩阵
        if (this->goIdx == 0 && this->s1oIdx == 0) {
            this->Bmm1SetTensorB();
        }

        this->SplitNGroups(splitBaseNum, multiCoreInnerLimit);
    }
}

/*
 * 函数功能：对输入的base块进行N:1分组，默认输入的base块都在同一轴内，即属于同一组S1
 * 输入参数：
 * splitBaseNum: 用于进行N:1分组的base块个数
 * multiCoreInnerLimit: 最多计算base块个数
 */
template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1d<INPUT_T, T, isBasicBlock>::SplitNGroups(
    int64_t splitBaseNum, int64_t multiCoreInnerLimit)
{
    // 对数据块进行N:1分组，循环的次数即为分组数，分组数=ceil(splitBaseNum, N)
    int64_t splitTotalTimes = this->CeilDiv(splitBaseNum, this->tilingData->coreParams.nRatio);
    for (int64_t splitTimes = 0; splitTimes < splitTotalTimes && this->multiCoreInnerIdx < multiCoreInnerLimit;
        splitTimes++) {
        // 当前N:1组所计算的base块数量，最多不超过N
        int64_t countBase = this->Min(static_cast<int64_t>(this->tilingData->coreParams.nRatio),
            splitBaseNum - splitTimes * this->tilingData->coreParams.nRatio);
        // 计算各轴idx
        this->ComputeIdx();

        // 对当前N:1组进行attention计算，最多不超过N个base块
        this->ProcessBlock(countBase);

        // 更新idx，偏移至下个N:1组的首地址
        this->multiCoreInnerIdx += countBase;
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1d<INPUT_T, T, isBasicBlock>::ProcessBlock(
    int64_t countBase)
{
    // 计算Q矩阵offset
    this->Bmm1SetTensorA();
    this->Bmm1SetTail(countBase);
    this->bmm1.IterateAll(this->bmm1WorkspaceGm);

    int64_t tmpS1oIdx = this->s1oIdx;

    for (int64_t splitNIdx = 0; splitNIdx < countBase; splitNIdx++) {
        // 计算S1的realsize，用于后续vector和bmm2计算
        if (this->s1oIdx == this->tilingData->coreParams.s1OuterSize - 1) {
            this->s1RealSize = this->s1TailSize;
        } else {
            this->s1RealSize = this->tilingData->coreParams.s1BaseSize;
        }

        this->InitSoftmaxTensor();
        this->ProcessStage1(splitNIdx);

        // 每次循环处理一个S1 base块，因此处理下一块前索引需要+1
        this->s1oIdx += 1;
    }
    this->Bmm2Compute(tmpS1oIdx);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1d<INPUT_T, T, isBasicBlock>::InitSoftmaxTensor()
{
    // softmaxSumUb尾轴长度按32B对齐
    uint32_t maxSumShape[] = {static_cast<uint32_t>(this->s1RealSize), 8};
    // softmaxSumUb尾轴真实数据长度
    uint32_t maxSumOrgShape[] = {static_cast<uint32_t>(this->s1RealSize), 1};
    this->softmaxSumUb = this->softmaxOutQueue.template AllocTensor<float>();
    this->softmaxMaxUb = this->softmaxOutQueue.template AllocTensor<float>();
    this->softmaxMaxUb.SetShapeInfo(ShapeInfo(2, maxSumShape, 2, maxSumOrgShape, AscendC::DataFormat::ND));
    this->softmaxSumUb.SetShapeInfo(ShapeInfo(2, maxSumShape, 2, maxSumOrgShape, AscendC::DataFormat::ND));
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1d<INPUT_T, T, isBasicBlock>::ProcessStage1(
    int64_t splitNIdx)
{
    int64_t pseOffset = this->ComputePseOffset();
    this->PseCopyIn(pseOffset, this->s1RealSize, this->tilingData->inputParams.s2Size);

    // 从workspace获取bmm1的计算结果
    AscendC::LocalTensor<T> bmm1ResUb = this->bmm1ResQueue.template AllocTensor<T>();
    this->SetBmm1ResShape(bmm1ResUb);
    this->CopyInBmm1Result(bmm1ResUb, splitNIdx);

    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    set_flag(PIPE_MTE2, PIPE_V, eventID);
    wait_flag(PIPE_MTE2, PIPE_V, eventID);

    this->PseCompute(bmm1ResUb, this->s1RealSize, this->tilingData->inputParams.s2Size);

    int64_t attenMaskOffset = this->ComputeAttenMaskOffset();
    this->AttenMaskCopyIn(attenMaskOffset, this->s1RealSize, this->tilingData->inputParams.s2Size);

    this->AttenMaskCompute(bmm1ResUb, this->s1RealSize, this->tilingData->inputParams.s2Size);

    if (this->hasDropMask) {
        // softmax的计算结果需要执行dropout操作
        int64_t dropMaskOffset = this->ComputeDropOffset();
        this->DropoutCopyIn(dropMaskOffset, this->s1RealSize, this->tilingData->inputParams.s2Size);
        this->SoftmaxCompute(bmm1ResUb, bmm1ResUb);
        if constexpr (AscendC::IsSameType<INPUT_T, T>::value) {
            AscendC::LocalTensor<T> stage1ResUb = this->stage1ResQueue.template AllocTensor<T>();
            this->DropoutCompute(stage1ResUb, bmm1ResUb, this->s1RealSize, this->tilingData->inputParams.s2Size);
            this->stage1ResQueue.template EnQue(stage1ResUb);
        } else {
            AscendC::LocalTensor<INPUT_T> stage1ResUb = this->stage1ResQueue.template AllocTensor<INPUT_T>();
            this->DropoutCompute(bmm1ResUb, bmm1ResUb, this->s1RealSize, this->tilingData->inputParams.s2Size);
            this->Stage1Cast(stage1ResUb, bmm1ResUb);
            this->stage1ResQueue.template EnQue(stage1ResUb);
        }
    } else {
        if constexpr (AscendC::IsSameType<INPUT_T, T>::value) {
            AscendC::LocalTensor<T> stage1ResUb = this->stage1ResQueue.template AllocTensor<T>();
            this->SoftmaxCompute(stage1ResUb, bmm1ResUb);
            this->stage1ResQueue.template EnQue(stage1ResUb);
        } else {
            AscendC::LocalTensor<INPUT_T> stage1ResUb = this->stage1ResQueue.template AllocTensor<INPUT_T>();
            this->SoftmaxCompute(bmm1ResUb, bmm1ResUb);
            this->Stage1Cast(stage1ResUb, bmm1ResUb);
            this->stage1ResQueue.template EnQue(stage1ResUb);
        }
    }

    this->bmm1ResQueue.FreeTensor(bmm1ResUb);

    // 将stage1的计算结果搬运至workspace
    AscendC::LocalTensor<INPUT_T> stage1ResUb = this->stage1ResQueue.template DeQue<INPUT_T>();
    DataCopy(this->stage1WorkspaceGm[splitNIdx * this->bmm1WorkspaceStrideSize], stage1ResUb,
        this->s1RealSize * this->tilingData->coreParams.s2BaseSize);
    this->stage1ResQueue.FreeTensor(stage1ResUb);
    return;
}

// 计算各轴strideSize，用于与各轴idx相乘计算offset
template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1d<INPUT_T, T, isBasicBlock>::ComputeAxisStrideSize()
{
    // BSNGD:0; SBNGD:1; BNGSD:2
    // Q、KV、PSE、attenMASK、dropMask、softmaxOutput
    if (this->tilingData->inputParams.layoutType == 1) { // BSNGD
        // Q:
        // D
        this->qGStrideSize = this->tilingData->inputParams.dSize;
        // G*D
        this->qN2StrideSize = this->tilingData->inputParams.gSize * this->tilingData->inputParams.dSize;
        // S1_b*N2*G*D
        this->qS1oStrideSize =
            this->tilingData->coreParams.s1BaseSize * this->tilingData->inputParams.n2Size * this->qN2StrideSize;
        // S1*N2*G*D
        this->qBStrideSize =
            this->tilingData->inputParams.s1Size * this->tilingData->inputParams.n2Size * this->qN2StrideSize;

        // KV: G=1
        // 1*D
        this->kvN2oStrideSize = this->tilingData->inputParams.dSize;
        // S2*N2*1*D
        this->kvBoStrideSize = this->tilingData->inputParams.s2Size * this->tilingData->inputParams.n2Size *
            this->tilingData->inputParams.dSize;
    } else if (this->tilingData->inputParams.layoutType == 2) { // SBNGD
        // Q:
        // D
        this->qGStrideSize = this->tilingData->inputParams.dSize;
        // G*D
        this->qN2StrideSize = this->tilingData->inputParams.gSize * this->tilingData->inputParams.dSize;
        // N2*G*D
        this->qBStrideSize = this->tilingData->inputParams.n2Size * this->qN2StrideSize;
        // S1_b*B*N2*G*D
        this->qS1oStrideSize =
            this->tilingData->coreParams.s1BaseSize * this->tilingData->inputParams.bSize * this->qBStrideSize;

        // KV: G=1
        // 1*D
        this->kvN2oStrideSize = this->tilingData->inputParams.dSize;
        // N2*1*D
        this->kvBoStrideSize = this->tilingData->inputParams.n2Size * this->tilingData->inputParams.dSize;
    } else if (this->tilingData->inputParams.layoutType == 3) { // BNGSD
        // Q and Bmm2output:
        // S1_b*D
        this->qS1oStrideSize = this->tilingData->coreParams.s1BaseSize * this->tilingData->inputParams.dSize;
        // S1*D
        this->qGStrideSize = this->tilingData->inputParams.s1Size * this->tilingData->inputParams.dSize;
        // G*S1*D
        this->qN2StrideSize = this->tilingData->inputParams.gSize * this->qGStrideSize;
        // N2*G*S1*D
        this->qBStrideSize = this->tilingData->inputParams.n2Size * this->qN2StrideSize;

        // KV: G=1
        // 1*S2*D
        this->kvN2oStrideSize = this->tilingData->inputParams.s2Size * this->tilingData->inputParams.dSize;
        // N2*1*S2*D
        this->kvBoStrideSize = this->tilingData->inputParams.n2Size * this->kvN2oStrideSize;
    }

    // attenMask[B,N2,G,S1,S2]
    // S1*S2
    this->attenGStrideSizeType1 = this->tilingData->inputParams.s1Size * this->tilingData->inputParams.s2Size;
    // G*S1*S2
    this->attenN2StrideSizeType1 = this->tilingData->inputParams.gSize * this->attenGStrideSizeType1;
    // N2*G*S1*S2
    this->attenBStrideSizeType1 = this->tilingData->inputParams.n2Size * this->attenN2StrideSizeType1;
    // attenMask[B,1,1,S1,S2]
    // S1*S2
    this->attenBStrideSizeType2 = this->attenGStrideSizeType1;
    // attenMask[1,1,1,S1,S2]
    // S1_b*S2
    this->attenS1oStrideSize = this->tilingData->coreParams.s1BaseSize * this->tilingData->inputParams.s2Size;

    // dropMask
    // S1_b*S2
    this->dropS1oStrideSize = this->attenS1oStrideSize;
    // S1*S2
    this->dropGStrideSize = this->attenGStrideSizeType1;
    // G*S1*S2
    this->dropN2StrideSize = this->tilingData->inputParams.gSize * this->dropGStrideSize;
    // N2*G*S1*S2
    this->dropBStrideSize = this->tilingData->inputParams.n2Size * this->dropN2StrideSize;

    // pse[B,N2,G,S1,S2]
    // S1_b*S2
    this->pseS1oStrideSizeType1 = this->dropS1oStrideSize;
    // S1*S2
    this->pseGStrideSizeType1 = this->dropGStrideSize;
    // G*S1*S2
    this->pseN2StrideSizeType1 = this->dropN2StrideSize;
    // N2*G*S1*S2
    this->pseBStrideSizeType1 = this->dropBStrideSize;

    // pse[B,N2,G,1,S2]
    // 1*S2
    this->pseGStrideSizeType2 = this->tilingData->inputParams.s2Size;
    // G*1*S2
    this->pseN2StrideSizeType2 = this->tilingData->inputParams.gSize * this->tilingData->inputParams.s2Size;
    // N2*G*1*S2
    this->pseBStrideSizeType2 = this->tilingData->inputParams.n2Size * this->pseN2StrideSizeType2;

    // softmaxOutput[B,N2,G,S1,8]
    // S1_b*8
    this->softmaxS1oStrideSize = this->tilingData->coreParams.s1BaseSize * this->fp32BaseSize;
    // S1*8
    this->softmaxGStrideSize = this->tilingData->inputParams.s1Size * this->fp32BaseSize;
    // G*S1*8
    this->softmaxN2StrideSize = this->tilingData->inputParams.gSize * this->softmaxGStrideSize;
    // N2*G*S1*8
    this->softmaxBStrideSize = this->tilingData->inputParams.n2Size * this->softmaxN2StrideSize;
}

// 计算各轴index
template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1d<INPUT_T, T, isBasicBlock>::ComputeIdx()
{
    this->boIdx = this->multiCoreInnerIdx / this->n2GS1o;
    this->n2oIdx = this->multiCoreInnerIdx % this->n2GS1o / this->gS1o;
    this->goIdx = this->multiCoreInnerIdx % this->gS1o / this->tilingData->coreParams.s1OuterSize;
    this->s1oIdx = this->multiCoreInnerIdx % this->tilingData->coreParams.s1OuterSize;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1d<INPUT_T, T, isBasicBlock>::SetBmm1ResShape(
    AscendC::LocalTensor<T> &bmm1ResUb)
{
    // bmm1ResUb尾轴长度按32B对齐
    uint32_t bmm1ResShape[] = {static_cast<uint32_t>(this->s1RealSize),
        static_cast<uint32_t>(this->tilingData->coreParams.s2BaseSize)};
    // bmm1ResUb尾轴真实数据长度
    uint32_t bmm1OrgShape[] = {static_cast<uint32_t>(this->s1RealSize),
        static_cast<uint32_t>(this->tilingData->inputParams.s2Size)};
    bmm1ResUb.SetShapeInfo(ShapeInfo(2, bmm1ResShape, 2, bmm1OrgShape, AscendC::DataFormat::ND));
    return;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline int64_t FlashAttentionScoreSplitBn2gs1S1d<INPUT_T, T, isBasicBlock>::ComputePseOffset()
{
    // 默认shape为[B,N2,G,S1,S2]
    int64_t pseBOffset = this->boIdx * this->pseBStrideSizeType1;
    int64_t pseN2Offset = this->n2oIdx * this->pseN2StrideSizeType1;
    int64_t pseGOffset = this->goIdx * this->pseGStrideSizeType1;
    int64_t pseS1Offset = this->s1oIdx * this->pseS1oStrideSizeType1;

    // 当pse的shape为[B,N,G,1,S2]
    if (this->tilingData->inputParams.pseShapeType == 1) {
        pseBOffset = this->boIdx * this->pseBStrideSizeType2;
        pseN2Offset = this->n2oIdx * this->pseN2StrideSizeType2;
        pseGOffset = this->goIdx * this->pseGStrideSizeType2;
        pseS1Offset = 0;
    }
    return pseBOffset + pseN2Offset + pseGOffset + pseS1Offset;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline int64_t
FlashAttentionScoreSplitBn2gs1S1d<INPUT_T, T, isBasicBlock>::ComputeAttenMaskOffset()
{
    if (this->tilingData->inputParams.attenMaskCompressMode == this->attenMaskNoCompress) {
        return this->ComputeOffsetForNoCompress();
    } else if (this->tilingData->inputParams.attenMaskCompressMode == this->attenMaskLeftUpCausalCompress) {
        int64_t delta = this->s1oIdx * this->tilingData->coreParams.s1BaseSize;
        return this->ComputeOffsetForCausal(delta, this->tilingData->coreParams.s1BaseSize,
                                            this->tilingData->inputParams.s2Size,
                                            this->tilingData->inputParams.attenMaskS2Size);
    } else if (this->tilingData->inputParams.attenMaskCompressMode == this->attenMaskRightDownCausalCompress) {
        int64_t delta = this->s1oIdx * this->tilingData->coreParams.s1BaseSize + this->tilingData->inputParams.s2Size -
                        this->tilingData->inputParams.s1Size;
        return this->ComputeOffsetForCausal(delta, this->tilingData->coreParams.s1BaseSize,
                                            this->tilingData->inputParams.s2Size,
                                            this->tilingData->inputParams.attenMaskS2Size);
    }
    return 0;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline int64_t
FlashAttentionScoreSplitBn2gs1S1d<INPUT_T, T, isBasicBlock>::ComputeOffsetForNoCompress()
{
    // 默认shape为[B,N2,G,S1,S2]
    int64_t attenBOffset = this->boIdx * this->attenBStrideSizeType1;
    int64_t attenN2Offset = this->n2oIdx * this->attenN2StrideSizeType1;
    int64_t attenGOffset = this->goIdx * this->attenGStrideSizeType1;
    int64_t attens1Offset = this->s1oIdx * this->attenS1oStrideSize;

    // 如果shape为[1,1,1,S1,S2]
    if (this->tilingData->inputParams.attenMaskShapeType == 2) {
        attenBOffset = 0;
        attenN2Offset = 0;
        attenGOffset = 0;
    } else if (this->tilingData->inputParams.attenMaskShapeType == 1) {
        // 如果shape为[B,1,1,S1,S2]
        attenBOffset = this->boIdx * this->attenBStrideSizeType2;
        attenN2Offset = 0;
        attenGOffset = 0;
    }

    return attenBOffset + attenN2Offset + attenGOffset + attens1Offset;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline int64_t FlashAttentionScoreSplitBn2gs1S1d<INPUT_T, T, isBasicBlock>::ComputeDropOffset()
{
    // shape为[B,N2,G,S1,S2]
    int64_t dropBOffset = this->boIdx * this->dropBStrideSize;
    int64_t dropN2Offset = this->n2oIdx * this->dropN2StrideSize;
    int64_t dropGOffset = this->goIdx * this->dropGStrideSize;
    int64_t dropS1Offset = this->s1oIdx * this->dropS1oStrideSize;

    return dropBOffset + dropN2Offset + dropGOffset + dropS1Offset;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline int64_t
FlashAttentionScoreSplitBn2gs1S1d<INPUT_T, T, isBasicBlock>::ComputeSoftmaxOutOffset()
{
    // shape为[B,N2,G,S1,8]
    int64_t softmaxBOffset = this->boIdx * this->softmaxBStrideSize;
    int64_t softmaxN2Offset = this->n2oIdx * this->softmaxN2StrideSize;
    int64_t softmaxGOffset = this->goIdx * this->softmaxGStrideSize;
    int64_t softmaxS1Offset = this->s1oIdx * this->softmaxS1oStrideSize;

    return softmaxBOffset + softmaxN2Offset + softmaxGOffset + softmaxS1Offset;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1d<INPUT_T, T, isBasicBlock>::Bmm1SetTensorA()
{
    // 计算各轴offset
    int64_t qBOffset = this->boIdx * this->qBStrideSize;
    int64_t qN2Offset = this->n2oIdx * this->qN2StrideSize;
    int64_t qGOffset = this->goIdx * this->qGStrideSize;
    int64_t qS1Offset = this->s1oIdx * this->qS1oStrideSize;

    // 当前base块offset
    int64_t qCoreOffset = qBOffset + qN2Offset + qGOffset + qS1Offset;

    this->bmm1.SetTensorA(this->queryGm[qCoreOffset]);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1d<INPUT_T, T, isBasicBlock>::Bmm1SetTensorB()
{
    // 计算各轴offset
    int64_t kBOffset = this->boIdx * this->kvBoStrideSize;
    int64_t kN2Offset = this->n2oIdx * this->kvN2oStrideSize;

    // 当前base块offset
    int64_t kCoreOffset = kBOffset + kN2Offset;

    this->bmm1.SetTensorB(this->keyGm[kCoreOffset], true);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1d<INPUT_T, T, isBasicBlock>::Bmm1SetTail(
    int64_t countBase)
{
    // 整块场景
    this->s1NGroupRealSize = this->tilingData->coreParams.s1BaseSize * countBase;

    // 有尾块的场景，可能由多个S1_b和一个S1_t组成一个N:1
    if ((this->multiCoreInnerIdx + countBase) % this->tilingData->coreParams.s1OuterSize == 0) {
        this->s1NGroupRealSize = this->tilingData->coreParams.s1BaseSize * (countBase - 1) + this->s1TailSize;
    }

    this->bmm1.SetTail(this->s1NGroupRealSize, -1, -1);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1d<INPUT_T, T, isBasicBlock>::Bmm2Compute(
    int64_t tmpS1oIdx)
{
    int64_t vBOffset = this->boIdx * this->kvBoStrideSize;
    int64_t vN2Offset = this->n2oIdx * this->kvN2oStrideSize;
    int64_t vCoreOffset = vBOffset + vN2Offset;

    int64_t qBOffset = this->boIdx * this->qBStrideSize;
    int64_t qN2Offset = this->n2oIdx * this->qN2StrideSize;
    int64_t qGOffset = this->goIdx * this->qGStrideSize;
    int64_t qS1Offset = tmpS1oIdx * this->qS1oStrideSize;
    int64_t qCoreOffset = qBOffset + qN2Offset + qGOffset + qS1Offset;

    this->bmm2.SetTensorA(this->stage1WorkspaceGm);
    this->bmm2.SetTensorB(this->valueGm[vCoreOffset]);
    this->bmm2.SetTail(this->s1NGroupRealSize, this->tilingData->inputParams.dSize,
        this->tilingData->inputParams.s2Size);

    this->bmm2.IterateAll(this->attentionOutGm[qCoreOffset]);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1d<INPUT_T, T, isBasicBlock>::SoftmaxCompute(
    AscendC::LocalTensor<T> &dstTensor, AscendC::LocalTensor<T> &srcTensor)
{
    AscendC::LocalTensor<uint8_t> apiTmpBuffer = this->apiTBuf.template Get<uint8_t>();
    AscendC::SoftMax<T, true, isBasicBlock>(dstTensor, this->softmaxSumUb, this->softmaxMaxUb, srcTensor, apiTmpBuffer,
        this->tilingData->softmaxTilingData);
    int64_t softmaxGmOffset = this->ComputeSoftmaxOutOffset();
    this->SoftmaxCopyOut(softmaxGmOffset, this->s1RealSize * this->fp32BaseSize);
    this->softmaxOutQueue.FreeTensor(this->softmaxSumUb);
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1d<INPUT_T, T, isBasicBlock>::CopyInBmm1Result(
    AscendC::LocalTensor<T> &bmm1ResUb, int64_t splitNIdx)
{
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = this->s1RealSize;
    dataCopyParams.blockLen = this->tilingData->coreParams.s2BaseSize * sizeof(T) / this->blockBytes;
    dataCopyParams.dstStride = 0;
    dataCopyParams.srcStride = 0;
    DataCopy(bmm1ResUb, this->bmm1WorkspaceGm[splitNIdx * this->bmm1WorkspaceStrideSize], dataCopyParams);
}

#endif // FLASH_ATTENTION_SCORE_SPLIT_B_N2_G_S1_D_H