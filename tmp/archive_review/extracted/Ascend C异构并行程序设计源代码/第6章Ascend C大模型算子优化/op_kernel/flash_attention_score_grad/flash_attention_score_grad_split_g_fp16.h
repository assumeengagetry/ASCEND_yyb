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

#ifndef FLASH_ATTENTION_SCORE_GRAD_SPLIT_B_N_ND_ND_ND_H_
#define FLASH_ATTENTION_SCORE_GRAD_SPLIT_B_N_ND_ND_ND_H_

#include "kernel_operator.h"
#include "attention_score_grad_tiling.h"
#include "lib/matmul_intf.h"

using namespace matmul;

template <typename T1, typename T2, const bool IS_BF16 = false>
class FlashAttentionScoreGradBNG {
public:
    __aicore__ inline FlashAttentionScoreGradBNG(){};
    __aicore__ inline void Init(
        __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dx, __gm__ uint8_t *query,
        __gm__ uint8_t *pse_shift, __gm__ uint8_t *drop_mask, __gm__ uint8_t *atten_mask,
        __gm__ uint8_t *forward_res, __gm__ uint8_t *softmax_max, __gm__ uint8_t *softmax_sum,
        __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dpse,
        __gm__ uint8_t *workspace, FlashAttentionScoreGradTilingData *ordTilingData, TPipe *pipe_in);
    __aicore__ inline void Process();

    using aType1 = MatmulType<TPosition::GM, CubeFormat::ND, T1, false>;
    using bType1 = MatmulType<TPosition::GM, CubeFormat::ND, T1, true>;
    using cType1 = MatmulType<TPosition::VECCALC, CubeFormat::ND_ALIGN, T2>;
    using biasType1 = MatmulType<TPosition::GM, CubeFormat::ND, float>;

    using aType2 = MatmulType<TPosition::VECCALC, CubeFormat::ND, T1, true>;
    using bType2 = MatmulType<TPosition::GM, CubeFormat::ND, T1, true>;
    using cType2 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using biasType2 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    Matmul<aType1, bType1, cType1, biasType1> mm1;
    Matmul<aType1, bType1, cType1, biasType1> mm2;
    Matmul<aType2, bType2, cType2, biasType2> mm3;
    Matmul<aType2, bType2, cType2, biasType2> mm4;

protected:
    TQue<QuePosition::VECIN, 1> vecInQue1;
    TQue<QuePosition::VECIN, 1> vecInQue2;
    TBuf<> vecClc1;
    TBuf<> vecClc2;
    TQue<QuePosition::VECOUT, 1> apiClcQue;

    FlashAttentionScoreGradTilingData* ordTilingData;

    TPipe *pipe;

    GlobalTensor<uint8_t> attenMaskU8Gm;
    GlobalTensor<T1> keyGm, valueGm, dxGm, queryGm, attenMaskGm, forwardResGm, pseGm, dpseGm;
    GlobalTensor<float> dqGm, dkGm, dvGm;

    GlobalTensor<float> softmaxMaxGm, softmaxSumGm;
    GlobalTensor<T2> workspaceGm;
    GlobalTensor<uint8_t> dropMaskGm;
    GlobalTensor<int32_t> syncGlobal;

    uint64_t pseOffset = 0;
    uint64_t attenMaskOffset = 0;

    uint32_t innerMatInShape[2];
    uint32_t innerMatOutShape[2];
    uint32_t innerReduceShape[2];

    T2 mulsValue = -10000.0;

    uint64_t mSFlaForClcOffset;
    uint64_t mSOutForClcOffset;
    uint64_t mBlockIdx;

    int32_t splitedBatchRange;
    int32_t flashKLoopRange;
    int32_t s1OutRange;
    int32_t sFla;
    int32_t sOut;
    int32_t outMaskNum;
    int32_t outReduceNum;
    int32_t outMatResSize; // Before sOut&&sFla parameters, except outMatResSize(used in sFla Loops)

    int32_t sInnerRange;
    int32_t sInner;
    int32_t innerReduceNum;
    int32_t innerMatInputNum;
    int32_t innerMaskNum;
    int32_t subMask;
    int32_t subRange; // sOut parameters

    int32_t innerMatResNum;
    int32_t innerMatResSize;
    int32_t sFlaBlockNum;
    int32_t sFlaBlock;
    int32_t sSubSFlaStride; // sFla parameters

    int32_t sOutAlign;
    int32_t sFlaAlign;
    int32_t innerMatResNumAlign;
    int32_t innerMatResSizeAlign;

    bool isDrop;

    int32_t layoutSBH;

    __aicore__ inline void InitGm();

    __aicore__ inline void FrontCompute(const uint64_t &flashKIdx, const uint64_t &s1OutIdx,
                                        const uint64_t &flashKLoopOffset, const uint64_t &s1OutLoopOffset,
                                        const uint64_t &s1OutLoopMaskOffset, const uint64_t s1OutIdxEnd);

    __aicore__ inline void ReCompute(const uint64_t &flashKIdx, const uint64_t &s1OutIdx,
                                     const uint64_t &flashKLoopOffset, const uint64_t &s1OutLoopOffset,
                                     const uint64_t &s1OutLoopMaskOffset, const uint64_t &s1OutLoopReduceOffset);

    __aicore__ inline void ClcFrontOffset(uint64_t &dxOffset, uint64_t &maskOffset, uint64_t &spmOffset,
                                          const uint64_t &inIdx, const uint64_t &flashKIdx,
                                          const uint64_t &s1OutLoopOffset, const uint64_t &s1OutLoopMaskOffset);

    __aicore__ inline void CopyInSoftMaxGrad(LocalTensor<T1> &dxInner, const GlobalTensor<T1> &dxGmIn,
                                             LocalTensor<T1> &forwardResInner, const GlobalTensor<T1> &forwardResGmIn);

    __aicore__ inline void ClcSoftMaxGrad(LocalTensor<T2> &sftFrontResInner, LocalTensor<T1> &dxInner,
                                          LocalTensor<T1> &forwardResInner);

    __aicore__ inline void CopyInDrop(LocalTensor<uint8_t> &dst, const GlobalTensor<uint8_t> &src);

    __aicore__ inline void ClcDrop(LocalTensor<T2> &dst, LocalTensor<T2> &src1, LocalTensor<uint8_t> &src2);

    __aicore__ inline void ClcSub(LocalTensor<T2> &frontResInner, LocalTensor<T2> &dpResInner,
                                  LocalTensor<T2> &sftFrontResInner);

    __aicore__ inline void ClcReComputeOffset(uint64_t &queryOffset, uint64_t &softmaxMaxOffset, uint64_t &maskOffset,
                                              uint64_t &spmOffset, uint64_t &scmOffset1, uint64_t &scmOffset2,
                                              const uint64_t &s1OutLoopOffset, const uint64_t &s1OutLoopMaskOffset,
                                              const uint64_t &s1OutLoopReduceOffset, const uint64_t &flashKIdx,
                                              const uint64_t &inIdx);

    __aicore__ inline void CopyInSoftMax(LocalTensor<float> &maxInner, const GlobalTensor<float> &softmaxMaxGmIn,
                                         LocalTensor<float> &sumInner, const GlobalTensor<float> &softmaxSumGmIn);

    __aicore__ inline void ClcSoftMax(LocalTensor<T2> &softmaxResInner, LocalTensor<T2> &reMatmulResInner,
                                      LocalTensor<float> &maxInner, LocalTensor<float> &sumInner);

    __aicore__ inline void ClcMul(LocalTensor<T2> &frontResInner, LocalTensor<T2> &mulResInner,
                                  LocalTensor<T2> &softmaxResInner);

    __aicore__ inline void ClcReDrop(LocalTensor<T2> &dvDropResInner, LocalTensor<T2> &softmaxResInner,
                                     const GlobalTensor<uint8_t> &dropMaskGmIn);

    __aicore__ inline void MyMatMulClc(const LocalTensor<T1> &tensorA, const GlobalTensor<T1> &tensorB,
                                       const GlobalTensor<float> &tensorC, uint64_t isAtomicAddFlag = 0,
                                       bool transA = false, bool transB = false);

    __aicore__ inline void DkvMatMulClc(const LocalTensor<T1> &tensorA, const GlobalTensor<T1> &tensorB,
                                        const GlobalTensor<float> &tensorC, uint64_t isAtomicAddFlag = 0,
                                        bool transA = false, bool transB = false);

    __aicore__ inline void SetFrontClcShape(LocalTensor<T1> &dxInner, LocalTensor<T1> &forwardResInner,
                                            LocalTensor<T2> &sftFrontResInner, LocalTensor<T2> &frontResInner,
                                            LocalTensor<uint8_t> &dpMaskInner);

    __aicore__ inline void SetReClcShape(LocalTensor<T2> &mulResInner, LocalTensor<float> &maxInner,
                                         LocalTensor<float> &sumInner, LocalTensor<T2> &dvDropResInner);

    __aicore__ inline void DetermineLoopParams(
        const SplitCoreParams &coreParams, uint64_t flashKIdx, uint64_t s1OutIdx);

    __aicore__ inline bool CopyInPse(const uint64_t &s1OutLoopMaskOffset, const uint64_t &flashKIdx);

    __aicore__ inline void ClcPse(LocalTensor<T2> &mmResUb);

    __aicore__ inline void ClcPseBroadcast(LocalTensor<T2> &mmResUb, LocalTensor<T2> pseUb, uint32_t eleNum);

    __aicore__ inline void ClcPseBroadcastInner(
        uint32_t repeatTimes, uint32_t outerOffset, uint32_t innerLoop,
        uint32_t innerRemain, const LocalTensor<T2> &pseUb,
        const LocalTensor<T2> &mmResUb, uint32_t stepCalcNum, const BinaryRepeatParams &repParams);

    __aicore__ inline bool CopyInAttenMask(const uint64_t &flashKIdx);

    __aicore__ inline void ClcAttenMask(LocalTensor<T2> &mmResUb);

    inline void Print(const GlobalTensor<half> global, uint64_t len);

    __aicore__ inline bool isBatchTailCore();
};

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradBNG<T1, T2, IS_BF16>::Init(
    __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dx, __gm__ uint8_t *query,
    __gm__ uint8_t *pse_shift, __gm__ uint8_t *drop_mask, __gm__ uint8_t *atten_mask,
    __gm__ uint8_t *forward_res, __gm__ uint8_t *softmax_max, __gm__ uint8_t *softmax_sum,
    __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dpse,
    __gm__ uint8_t *workspace, FlashAttentionScoreGradTilingData *ordTilingData, TPipe *pipe_in)
{
    this->keyGm.SetGlobalBuffer((__gm__ T1 *)key);
    this->valueGm.SetGlobalBuffer((__gm__ T1 *)value);
    this->dxGm.SetGlobalBuffer((__gm__ T1 *)dx);
    this->queryGm.SetGlobalBuffer((__gm__ T1 *)query);
    this->pseGm.SetGlobalBuffer((__gm__ T1 *)pse_shift);
    this->dropMaskGm.SetGlobalBuffer((__gm__ uint8_t *)drop_mask);
    this->attenMaskGm.SetGlobalBuffer((__gm__ T1 *)atten_mask);
    this->attenMaskU8Gm.SetGlobalBuffer((__gm__ uint8_t *)atten_mask);
    this->forwardResGm.SetGlobalBuffer((__gm__ T1 *)forward_res);
    this->softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmax_max);
    this->softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmax_sum);
    this->dqGm.SetGlobalBuffer((__gm__ float *)dq);
    this->dkGm.SetGlobalBuffer((__gm__ float *)dk);
    this->dvGm.SetGlobalBuffer((__gm__ float *)dv);
    this->dpseGm.SetGlobalBuffer((__gm__ T1 *)dpse);
    this->workspaceGm.SetGlobalBuffer((__gm__ T2 *)workspace);
    this->syncGlobal.SetGlobalBuffer((__gm__ int32_t *)workspace, 100 * 8);
    InitOutput<int32_t>(this->syncGlobal[GetBlockIdx() * 8], 8, 0);

    this->mBlockIdx = GetBlockIdx();
    this->ordTilingData = ordTilingData;
    this->pipe = pipe_in;

    MIX_LOG("this->ordTilingData->baseParams.B %d", this->ordTilingData->baseParams.B);
    MIX_LOG("this->ordTilingData->splitCoreParams.outMatResSize %d", this->ordTilingData->splitCoreParams.outMatResSize);

    MIX_LOG("this->ordTilingData->mm1TilingData.baseM %d", this->ordTilingData->mm1TilingData.baseM);
    MIX_LOG("this->ordTilingData->mm1TilingData.baseN %d", this->ordTilingData->mm1TilingData.baseN);
    MIX_LOG("this->ordTilingData->mm1TilingData.baseK %d", this->ordTilingData->mm1TilingData.baseK);

    MIX_LOG("this->ordTilingData->mm3TilingData.baseM %d", this->ordTilingData->mm3TilingData.baseM);
    MIX_LOG("this->ordTilingData->mm3TilingData.baseN %d", this->ordTilingData->mm3TilingData.baseN);
    MIX_LOG("this->ordTilingData->mm3TilingData.baseK %d", this->ordTilingData->mm3TilingData.baseK);

    this->mm1.Init(&(this->ordTilingData->mm1TilingData));
    this->mm2.Init(&(this->ordTilingData->mm1TilingData));
    this->mm3.Init(&(this->ordTilingData->mm2TilingData));
    this->mm4.Init(&(this->ordTilingData->mm3TilingData));

    this->innerMatInShape[0] = this->ordTilingData->splitCoreParams.sInner;
    this->innerMatInShape[1] = this->ordTilingData->baseParams.D;
    this->innerMatOutShape[0] = this->ordTilingData->splitCoreParams.sInner;
    this->innerMatOutShape[1] = this->ordTilingData->splitCoreParams.sFlaAlign;
    this->innerReduceShape[0] = this->ordTilingData->splitCoreParams.sInner;
    this->innerReduceShape[1] = 32 / sizeof(T2);

    this->pipe->InitBuffer(this->vecInQue1, 2, this->ordTilingData->splitCoreParams.innerMatInputSize);
    this->pipe->InitBuffer(this->vecInQue2, 1, this->ordTilingData->splitCoreParams.innerMatResSizeAlign);
    this->pipe->InitBuffer(this->vecClc1, this->ordTilingData->splitCoreParams.innerMatResSizeAlign);
    this->pipe->InitBuffer(this->vecClc2, this->ordTilingData->splitCoreParams.innerMatResSizeAlign);
    this->pipe->InitBuffer(this->apiClcQue, 1, this->ordTilingData->baseParams.apiClcQueueSize);

    this->pipe->InitSpmBuffer(
        this->workspaceGm[800 + this->mBlockIdx * this->ordTilingData->splitCoreParams.innerMatResSizeAlign],
        this->ordTilingData->splitCoreParams.innerMatResSizeAlign);  // [s_out, s_fla]
    uint64_t buff_offset = 800 + GetBlockNum() * GetTaskRation() * this->ordTilingData->splitCoreParams.innerMatResSizeAlign;
    this->mm1.SetWorkspace(
        (__gm__ uint8_t *)this->workspaceGm[
            buff_offset + this->mBlockIdx * this->ordTilingData->splitCoreParams.innerMatResSizeAlign].GetPhyAddr(),
        this->ordTilingData->splitCoreParams.innerMatResSizeAlign);
    this->mm2.SetWorkspace(
        (__gm__ uint8_t *)this->workspaceGm[
            buff_offset + GetBlockNum() * GetTaskRation() * this->ordTilingData->splitCoreParams.innerMatResSizeAlign +
            this->mBlockIdx * this->ordTilingData->splitCoreParams.innerMatResSizeAlign].GetPhyAddr(),
        this->ordTilingData->splitCoreParams.innerMatResSizeAlign);

    DropOutBitModeInit();
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradBNG<T1, T2, IS_BF16>::Process()
{
    this->isDrop = false;
    if (this->ordTilingData->baseParams.keepProb < 1) {
        this->isDrop = true;
    }

    this->layoutSBH = this->ordTilingData->baseParams.layoutSBH;

    this->splitedBatchRange = this->ordTilingData->splitCoreParams.splitedBatchRange;
    this->flashKLoopRange = this->ordTilingData->splitCoreParams.flashKLoopRange;
    this->s1OutRange = this->ordTilingData->splitCoreParams.s1OutRange;
    this->sFla = this->ordTilingData->splitCoreParams.sFla;
    this->sOut = this->ordTilingData->splitCoreParams.sOut;
    this->outMaskNum = this->ordTilingData->splitCoreParams.outMaskNum;
    this->outReduceNum = this->ordTilingData->splitCoreParams.outReduceNum;
    this->outMatResSize = this->ordTilingData->splitCoreParams.outMatResSize;

    this->mSFlaForClcOffset = this->ordTilingData->splitCoreParams.sFla;
    this->mSOutForClcOffset = this->ordTilingData->splitCoreParams.sOut;

    if (isBatchTailCore()) {  // Tail Core Engagement
        this->splitedBatchRange = this->ordTilingData->tailSplitCoreParams.splitedBatchRange;
        this->flashKLoopRange = this->ordTilingData->tailSplitCoreParams.flashKLoopRange;
        this->s1OutRange = this->ordTilingData->tailSplitCoreParams.s1OutRange;

        this->sFla = this->ordTilingData->tailSplitCoreParams.sFla;
        this->sOut = this->ordTilingData->tailSplitCoreParams.sOut;
        this->outMaskNum = this->ordTilingData->tailSplitCoreParams.outMaskNum;
        this->outReduceNum = this->ordTilingData->tailSplitCoreParams.outReduceNum;
    }

    for (int32_t batchIdx = 0; batchIdx < this->splitedBatchRange; batchIdx++) {
        uint64_t bIdx = (this->mBlockIdx * this->splitedBatchRange + batchIdx) / this->ordTilingData->baseParams.N;
        uint64_t nIdx = (this->mBlockIdx * this->splitedBatchRange + batchIdx) % this->ordTilingData->baseParams.N;

        uint64_t batchLoopOffset = bIdx * this->ordTilingData->baseParams.S * this->ordTilingData->baseParams.H +
                                nIdx * this->ordTilingData->baseParams.D;
        uint64_t batchLoopOffset2 = bIdx * this->ordTilingData->baseParams.Skv * this->ordTilingData->baseParams.H +
                                nIdx * this->ordTilingData->baseParams.D;
        if (this->layoutSBH == 1) {
            batchLoopOffset = bIdx * this->ordTilingData->baseParams.H + nIdx * this->ordTilingData->baseParams.D;
            batchLoopOffset2 = batchLoopOffset;
        }
        uint64_t batchLoopMaskOffset =
            this->mBlockIdx * this->ordTilingData->splitCoreParams.splitedCoreBatchMaskNum +
            batchIdx * this->ordTilingData->baseParams.allMatResNum;
        uint64_t batchLoopReduceOffset =
            this->mBlockIdx * this->ordTilingData->splitCoreParams.splitedCoreBatchReduceNum +
            batchIdx * this->ordTilingData->baseParams.allReduceNum;

        for (int32_t flashKIdx = 0; flashKIdx < this->flashKLoopRange; flashKIdx++) {  // s/flashK = s_fla
            uint64_t flashKLoopOffset = batchLoopOffset2 + flashKIdx * this->sFla * this->ordTilingData->baseParams.H;
            if (this->layoutSBH == 1) {
                flashKLoopOffset = batchLoopOffset2 +
                    flashKIdx * this->sFla * this->ordTilingData->baseParams.B * this->ordTilingData->baseParams.H;
            }

            uint32_t idx = 0;
            uint64_t s1OutIdxEnd = this->s1OutRange - 1;
            for (int32_t s1OutIdx = 0; s1OutIdx < this->s1OutRange; s1OutIdx++) {  // s/s1OutRges = s_out
                uint64_t s1OutLoopOffset = batchLoopOffset + s1OutIdx * this->sOut * this->ordTilingData->baseParams.H;
                if (this->layoutSBH == 1) {
                    s1OutLoopOffset =
                        batchLoopOffset +
                        s1OutIdx * this->sOut * this->ordTilingData->baseParams.B * this->ordTilingData->baseParams.H;
                }

                uint64_t s1OutLoopMaskOffset = batchLoopMaskOffset + s1OutIdx * this->outMaskNum;
                uint64_t s1OutLoopReduceOffset = batchLoopReduceOffset + s1OutIdx * this->outReduceNum;

                this->pseOffset = bIdx * this->ordTilingData->baseParams.N * this->ordTilingData->baseParams.Skv +
                                  nIdx * this->ordTilingData->baseParams.Skv;  // 按 BN1S场景设置
                this->attenMaskOffset = s1OutIdx * this->outMaskNum;
                if (this->ordTilingData->baseParams.attenMaskShapeType == 1) {  // B1SS
                    this->attenMaskOffset +=
                        bIdx * this->ordTilingData->baseParams.S * this->ordTilingData->baseParams.Skv;
                }

                if (isBatchTailCore()) {
                    this->DetermineLoopParams(this->ordTilingData->tailSplitCoreParams, flashKIdx, s1OutIdx);
                } else {
                    this->DetermineLoopParams(this->ordTilingData->splitCoreParams, flashKIdx, s1OutIdx);
                }

                if (idx == 0) {
                    this->mm1.SetTail(this->sOut, this->sFla, this->ordTilingData->baseParams.D); // M N K
                    this->mm1.SetTensorA(this->dxGm[s1OutLoopOffset]);
                    this->mm1.SetTensorB(this->valueGm[flashKLoopOffset], true);
                    this->mm1.template Iterate<false>();  // [sInner, H/N] * [H/N, sFla]
                }
                this->mm2.SetTail(this->sOut, this->sFla, this->ordTilingData->baseParams.D); // M N K
                this->mm2.SetTensorA(this->queryGm[s1OutLoopOffset]);
                this->mm2.SetTensorB(this->keyGm[flashKLoopOffset], true);
                this->mm2.template Iterate<false>();  // [sInner, H/N] * [H/N, sFla]
                this->FrontCompute(flashKIdx, s1OutIdx, flashKLoopOffset,
                                   s1OutLoopOffset, s1OutLoopMaskOffset, s1OutIdxEnd);
                this->ReCompute(flashKIdx, s1OutIdx, flashKLoopOffset,
                                s1OutLoopOffset, s1OutLoopMaskOffset, s1OutLoopReduceOffset);
                idx++;
            }
        }
    }
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline bool FlashAttentionScoreGradBNG<T1, T2, IS_BF16>::isBatchTailCore()
{
    return this->mBlockIdx != 0 && this->mBlockIdx == this->ordTilingData->baseParams.coreNum - 1;
}


template <typename T1, typename T2, const bool IS_BF16>
inline void FlashAttentionScoreGradBNG<T1, T2, IS_BF16>::Print(const GlobalTensor<half> global, uint64_t len)
{
    std::ostringstream os_;
    os_.str("");
    uint64_t printLen = len; // std::min(len, global.GetSize());
    uint64_t blockNum = ONE_BLK_SIZE / sizeof(half);
    uint64_t rowNum = printLen / blockNum;
    uint64_t residualNum = printLen % blockNum;
    const uint64_t width = 4;
    for (uint64_t i = 0; i < rowNum; i++) {
        os_ << std::setw(width) << std::setfill('0') << i * blockNum << " : ";
        for (uint64_t j = 0; j < blockNum; j++) {
            os_ << global.GetValue(i * blockNum + j).ToFloat() << " ";
        }
        os_ << std::endl;
    }
    if (residualNum != 0) {
        os_ << std::setw(width) << std::setfill('0') << rowNum * blockNum << " : ";
        for (uint64_t i = 0; i < residualNum; i++) {
            os_ << global.GetValue(rowNum * blockNum + i).ToFloat() << " ";
        }
        os_ << std::endl;
    }
    std::cout << os_.str();
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradBNG<T1, T2, IS_BF16>::InitGm()
{
    uint64_t allSize =
        this->ordTilingData->baseParams.B * this->ordTilingData->baseParams.S * this->ordTilingData->baseParams.H;

    uint32_t perSize =
        (allSize + this->ordTilingData->baseParams.coreNum - 1) / this->ordTilingData->baseParams.coreNum;
    uint32_t initUsedCore = (allSize + perSize - 1) / perSize;
    uint32_t tailSize = allSize - perSize * (initUsedCore - 1);

    uint32_t initSize = perSize;
    if (mBlockIdx == initUsedCore - 1) {
        initSize = tailSize;
    }

    uint64_t allSizeKv =
        this->ordTilingData->baseParams.B * this->ordTilingData->baseParams.Skv * this->ordTilingData->baseParams.H;
    uint32_t perSizeKv =
        (allSizeKv + this->ordTilingData->baseParams.coreNum - 1) / this->ordTilingData->baseParams.coreNum;
    uint32_t initUsedCoreKv = (allSizeKv + perSizeKv - 1) / perSizeKv;
    uint32_t tailSizeKv = allSizeKv - perSizeKv * (initUsedCoreKv - 1);
    uint32_t initSizeKv = perSizeKv;
    if (mBlockIdx == initUsedCoreKv - 1) {
        initSizeKv = tailSizeKv;
    }

    uint32_t offset = mBlockIdx * perSize;
    uint32_t offsetKv = mBlockIdx * perSizeKv;

    if (mBlockIdx < initUsedCore) {
        if constexpr (IS_BF16) {
            InitOutput<float>(dqGm[offset], initSize, 0.0);
        } else {
            InitOutput<float>(dqGm[offset], initSize, 0);
        }
    }
    if (mBlockIdx < initUsedCoreKv) {
        if constexpr (IS_BF16) {
            InitOutput<float>(dkGm[offsetKv], initSizeKv, 0.0);
            InitOutput<float>(dvGm[offsetKv], initSizeKv, 0.0);
        } else {
            InitOutput<float>(dkGm[offsetKv], initSizeKv, 0);
            InitOutput<float>(dvGm[offsetKv], initSizeKv, 0);
        }
    }

    LocalTensor<int32_t> workLocal = vecInQue1.AllocTensor<int32_t>();
    SyncAll<false>(syncGlobal, workLocal);
    vecInQue1.FreeTensor(workLocal);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradBNG<T1, T2, IS_BF16>::FrontCompute(
    const uint64_t &flashKIdx, const uint64_t &s1OutIdx, const uint64_t &flashKLoopOffset,
    const uint64_t &s1OutLoopOffset, const uint64_t &s1OutLoopMaskOffset, const uint64_t s1OutIdxEnd)
{
    LocalTensor<T1> dxInner = vecInQue1.AllocTensor<T1>();         // [s_inner, H/N]
    LocalTensor<T1> forwardResInner = vecInQue1.AllocTensor<T1>(); // [s_inner, H/N]
    LocalTensor<T2> sftFrontResInner = vecClc2.Get<T2>(sInner * 32 / sizeof(T2));
    LocalTensor<T2> frontResInner = vecClc1.Get<T2>(innerMatResNumAlign);
    LocalTensor<T2> &dpResInner = frontResInner;                         // [s_inner, s_fla]
    LocalTensor<T2> &dpMatmmulResInner = frontResInner;                  // [s_inner, s_fla]
    LocalTensor<uint8_t> dpMaskInner = vecInQue2.AllocTensor<uint8_t>(); // [s_inner, s_fla]
    dxInner.SetSize(innerMatInputNum);
    forwardResInner.SetSize(innerMatInputNum);
    dpMaskInner.SetSize(innerMatResNumAlign);
    SetFrontClcShape(dxInner, forwardResInner, sftFrontResInner, frontResInner, dpMaskInner);

    CopyInSoftMaxGrad(dxInner, dxGm[s1OutLoopOffset], forwardResInner, forwardResGm[s1OutLoopOffset]);
    if (isDrop) {
        uint64_t maskOffset = s1OutLoopMaskOffset + flashKIdx * this->mSFlaForClcOffset;
        CopyInDrop(dpMaskInner, dropMaskGm[maskOffset / 8]);
    }

    ClcSoftMaxGrad(sftFrontResInner, dxInner, forwardResInner);

    MIX_LOG("sftFrontResInner");
    sftFrontResInner.Print();

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);

    mm1.template GetTensorC<false>(dpMatmmulResInner, false, false);
    mm1.End();

    MIX_LOG("dpMatmmulResInner");
    dpMatmmulResInner.Print();

    if (s1OutIdx < s1OutIdxEnd) {
        mm1.SetTail(sOut, sFla, this->ordTilingData->baseParams.D);
        if (this->layoutSBH == 1) {
            mm1.SetTensorA(
                dxGm[s1OutLoopOffset + sOut * this->ordTilingData->baseParams.B * this->ordTilingData->baseParams.H]);
        } else {
            mm1.SetTensorA(dxGm[s1OutLoopOffset + sOut * this->ordTilingData->baseParams.H]);
        }
        mm1.SetTensorB(valueGm[flashKLoopOffset], true);
        mm1.template Iterate<false>();
    }

    if (isDrop) {
        ClcDrop(dpResInner, dpMatmmulResInner, dpMaskInner);
    }

    MIX_LOG("isDrop %d", isDrop);

    ClcSub(frontResInner, dpMatmmulResInner, sftFrontResInner);

    MIX_LOG("frontResInner");
    frontResInner.Print();

    pipe->WriteSpmBuffer(frontResInner, innerMatResSizeAlign, 0); // all [s_out, s_fla]

    vecInQue1.FreeTensor(dxInner);
    vecInQue1.FreeTensor(forwardResInner);
    vecInQue2.FreeTensor(dpMaskInner);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradBNG<T1, T2, IS_BF16>::ReCompute(
    const uint64_t &flashKIdx, const uint64_t &s1OutIdx, const uint64_t &flashKLoopOffset,
    const uint64_t &s1OutLoopOffset, const uint64_t &s1OutLoopMaskOffset, const uint64_t &s1OutLoopReduceOffset)
{
    LocalTensor<T2> frontResInner = vecClc1.Get<T2>(innerMatResNumAlign);
    LocalTensor<T2> &mulResInner = frontResInner;
    LocalTensor<T2> dvDropResInner = vecClc2.Get<T2>(innerMatResNumAlign); // [s_inner, s_fla]

    LocalTensor<T2> &reMatmulResInner = dvDropResInner;  // [s_inner, s_fla]
    LocalTensor<T2> &softmaxResInner = dvDropResInner;   // [s_inner, s_fla]
    LocalTensor<T2> &attenMaskResInner = dvDropResInner; // [s_inner, s_fla]

    LocalTensor<float> maxInner = vecInQue1.AllocTensor<float>(); // [s_inner, 8]
    LocalTensor<float> sumInner = vecInQue1.AllocTensor<float>(); // [s_inner, 8]
    maxInner.SetSize(innerReduceNum);
    sumInner.SetSize(innerReduceNum);

    SetReClcShape(mulResInner, maxInner, sumInner, dvDropResInner);
    CopyInSoftMax(maxInner, softmaxMaxGm[s1OutLoopReduceOffset], sumInner, softmaxSumGm[s1OutLoopReduceOffset]);

    bool clcAttenMask = false;
    bool clsPse = CopyInPse(s1OutLoopMaskOffset, flashKIdx);
    if (!clsPse) {
        clcAttenMask = CopyInAttenMask(flashKIdx);
    }

    mm2.template GetTensorC<false>(reMatmulResInner, false, false);
    mm2.End();

    MIX_LOG("reMatmulResInner");
    reMatmulResInner.Print();

    if (clsPse) {
        ClcPse(reMatmulResInner);
        clcAttenMask = CopyInAttenMask(flashKIdx);
    }
    
    MIX_LOG("reMatmulResInner");
    reMatmulResInner.Print(256);

    Muls(reMatmulResInner, reMatmulResInner, (T2)(this->ordTilingData->baseParams.scaleValue), innerMatResNumAlign);

    if (clcAttenMask) {
        ClcAttenMask(reMatmulResInner);
    }

    MIX_LOG("attenMaskResInner");
    attenMaskResInner.Print(256);

    ClcSoftMax(softmaxResInner, attenMaskResInner, maxInner, sumInner);

    MIX_LOG("softmaxResInner");
    softmaxResInner.Print();

    ClcMul(frontResInner, mulResInner, softmaxResInner);

    MIX_LOG("frontResInner");
    frontResInner.Print();

    LocalTensor<T1> castedMulResInner;

    if (isDrop) {
        uint64_t maskOffset = s1OutLoopMaskOffset + flashKIdx * this->mSFlaForClcOffset;
        ClcReDrop(dvDropResInner, softmaxResInner, dropMaskGm[maskOffset / 8]);
    }
    LocalTensor<T1> castedDvDropResInner;

    this->mm3.SetTail(sOut, -1, sFla);
    this->MyMatMulClc(mulResInner, this->keyGm[flashKLoopOffset], this->dqGm[s1OutLoopOffset], true, false, false);
    
    this->mm4.SetTail(sFla, -1, sOut);
    this->DkvMatMulClc(mulResInner, this->queryGm[s1OutLoopOffset], this->dkGm[flashKLoopOffset], true, true, false);
    
    this->mm4.SetTail(sFla, -1, sOut);
    this->DkvMatMulClc(dvDropResInner, this->dxGm[s1OutLoopOffset], this->dvGm[flashKLoopOffset], true, true, false);

    vecInQue1.FreeTensor(maxInner);
    vecInQue1.FreeTensor(sumInner);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradBNG<T1, T2, IS_BF16>::ClcFrontOffset(
    uint64_t &dxOffset, uint64_t &maskOffset, uint64_t &spmOffset, const uint64_t &inIdx, const uint64_t &flashKIdx,
    const uint64_t &s1OutLoopOffset, const uint64_t &s1OutLoopMaskOffset)
{
    dxOffset = s1OutLoopOffset + inIdx * innerMatInputNum;
    maskOffset = s1OutLoopMaskOffset + flashKIdx * this->mSFlaForClcOffset + inIdx * innerMaskNum;
    spmOffset = inIdx * innerMatResNum;
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradBNG<T1, T2, IS_BF16>::CopyInSoftMaxGrad(
    LocalTensor<T1> &dxInner, const GlobalTensor<T1> &dxGmIn, LocalTensor<T1> &forwardResInner,
    const GlobalTensor<T1> &forwardResGmIn)
{
    if (layoutSBH == 1) {
        DataCopy(dxInner, dxGmIn,
                 {(uint16_t)sInner, (uint16_t)(this->ordTilingData->baseParams.D * sizeof(T1) / 32),
                  (uint16_t)((this->ordTilingData->baseParams.B * this->ordTilingData->baseParams.H -
                              this->ordTilingData->baseParams.D) *
                             sizeof(T1) / 32),
                  0});
    } else {
        DataCopy(dxInner, dxGmIn,
                 {(uint16_t)sInner, (uint16_t)(this->ordTilingData->baseParams.D * sizeof(T1) / 32),
                  (uint16_t)((this->ordTilingData->baseParams.H - this->ordTilingData->baseParams.D) * sizeof(T1) / 32),
                  0});
    }
    if (layoutSBH == 1) {
        DataCopy(forwardResInner, forwardResGmIn,
                 {(uint16_t)sInner, (uint16_t)(this->ordTilingData->baseParams.D * sizeof(T1) / 32),
                  (uint16_t)((this->ordTilingData->baseParams.B * this->ordTilingData->baseParams.H -
                              this->ordTilingData->baseParams.D) *
                             sizeof(T1) / 32),
                  0});
    } else {
        DataCopy(forwardResInner, forwardResGmIn,
                 {(uint16_t)sInner, (uint16_t)(this->ordTilingData->baseParams.D * sizeof(T1) / 32),
                  (uint16_t)((this->ordTilingData->baseParams.H - this->ordTilingData->baseParams.D) * sizeof(T1) / 32),
                  0});
    }

    vecInQue1.EnQue(forwardResInner);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradBNG<T1, T2, IS_BF16>::ClcSoftMaxGrad(LocalTensor<T2> &sftFrontResInner,
                                                                                    LocalTensor<T1> &dxInner,
                                                                                    LocalTensor<T1> &forwardResInner)
{
    vecInQue1.DeQue<T1>();
    bool isBasicBlock = (this->sInner % 8 ==0) && (this->ordTilingData->baseParams.D % 64 ==0);
    LocalTensor<uint8_t> apiClcTensor = apiClcQue.AllocTensor<uint8_t>();
    apiClcTensor.SetSize(this->ordTilingData->baseParams.apiClcQueueSize);
   
    if (isBasicBlock) {
        SoftmaxGradFront<T1, true>(sftFrontResInner, dxInner, forwardResInner,
                                   apiClcTensor,
                                   this->ordTilingData->softmaxGradTilingData);
    } else {
        SoftmaxGradFront<T1, false>(sftFrontResInner, dxInner, forwardResInner,
                                    apiClcTensor,
                                    this->ordTilingData->softmaxGradTilingData);
    }

    apiClcQue.FreeTensor(apiClcTensor);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradBNG<T1, T2, IS_BF16>::CopyInDrop(LocalTensor<uint8_t> &dst,
                                                                                const GlobalTensor<uint8_t> &src)
{
    struct DataCopyParams intriParams;
    intriParams.blockCount = (uint16_t)sInner;
    intriParams.blockLen = (uint16_t)((sFla / 8 + 31) / 32);
    intriParams.dstStride = 0;
    if ((this->ordTilingData->baseParams.Skv / 8 - intriParams.blockLen * 32) % 32 == 0) {
        intriParams.srcStride = (uint16_t)(this->ordTilingData->baseParams.Skv / 8 - intriParams.blockLen * 32) / 32;
        DataCopy(dst, src, intriParams);
    } else {
        intriParams.blockLen = (uint16_t)sFla / 8;
        intriParams.srcStride = (uint16_t)(this->ordTilingData->baseParams.Skv / 8 - intriParams.blockLen);
        DataCopyPadParams intriParams1;
        DataCopyPad(dst, src, intriParams, intriParams1);
    }
    vecInQue2.EnQue(dst);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradBNG<T1, T2, IS_BF16>::ClcDrop(LocalTensor<T2> &dst,
                                                                             LocalTensor<T2> &src1,
                                                                             LocalTensor<uint8_t> &src2)
{
    vecInQue2.DeQue<uint8_t>();
    DropOutShapeInfo info;
    info.firstAxis = sInner;
    info.srcLastAxis = sFla;
    info.maskLastAxis = (sFla / 8 + 31) / 32 * 32;
    LocalTensor<uint8_t> apiClcTensor = apiClcQue.AllocTensor<uint8_t>();
    apiClcTensor.SetSize(this->ordTilingData->baseParams.apiClcQueueSize);
    DropOut<T2, true>(dst, src1, src2, apiClcTensor, this->ordTilingData->baseParams.keepProb, info);
    apiClcQue.FreeTensor(apiClcTensor);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradBNG<T1, T2, IS_BF16>::ClcSub(LocalTensor<T2> &frontResInner,
                                                                            LocalTensor<T2> &dpResInner,
                                                                            LocalTensor<T2> &sftFrontResInner)
{
    // [m,n] - [m,16] -> [m,n] 按n轴的block数repeat，每个指令repeat算[m,16] - [m,16], subRange循环处理 超过mask情况
    MIX_LOG("subRange %d", subRange);
    MIX_LOG("subMask %d", subMask);
    MIX_LOG("sFlaBlockNum %d", sFlaBlockNum);
    MIX_LOG("sftFrontResInner size %d", sftFrontResInner.GetSize());
    MIX_LOG("dpResInner size %d", dpResInner.GetSize());
    MIX_LOG("frontResInner size  %d", frontResInner.GetSize());
    uint64_t srcOffset = 0;
    uint64_t dstOffset = 0;
    int32_t subIdx = 0;
    for (subIdx; subIdx < subRange - 1; subIdx++) {
        srcOffset = subIdx * subMask;  // subIdx * 16 * 8
        dstOffset = subIdx * sFlaAlign * 8; // subIdx * sFla * 8
        Sub(frontResInner[dstOffset], dpResInner[dstOffset], sftFrontResInner[srcOffset], subMask, sFlaBlockNum,
            {(uint8_t)(sFlaBlockNum), (uint8_t)(sFlaBlockNum), 1, 1, 1, 0});
    }
    srcOffset = subIdx * subMask;
    dstOffset = subIdx * sFlaAlign * 8;
    subMask = (sOut - subIdx * 8) * 32 / sizeof(T1);
    MIX_LOG("srcOffset %d", srcOffset);
    MIX_LOG("dstOffset %d", dstOffset);
    MIX_LOG("subMask %d", subMask);
    Sub(frontResInner[dstOffset], dpResInner[dstOffset], sftFrontResInner[srcOffset], subMask, sFlaBlockNum,
            {(uint8_t)(sFlaBlockNum), (uint8_t)(sFlaBlockNum), 1, 1, 1, 0});
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradBNG<T1, T2, IS_BF16>::ClcReComputeOffset(
    uint64_t &queryOffset, uint64_t &softmaxMaxOffset, uint64_t &maskOffset, uint64_t &spmOffset, uint64_t &scmOffset1,
    uint64_t &scmOffset2, const uint64_t &s1OutLoopOffset, const uint64_t &s1OutLoopMaskOffset,
    const uint64_t &s1OutLoopReduceOffset, const uint64_t &flashKIdx, const uint64_t &inIdx)
{
    queryOffset = s1OutLoopOffset + inIdx * innerMatInputNum;
    softmaxMaxOffset = s1OutLoopReduceOffset + inIdx * sInner * this->ordTilingData->baseParams.typeBlockNum;
    maskOffset = s1OutLoopMaskOffset + flashKIdx * this->mSFlaForClcOffset + inIdx * innerMaskNum;

    pseOffset += flashKIdx * this->mSFlaForClcOffset + inIdx * innerMaskNum;
    attenMaskOffset += flashKIdx * this->mSFlaForClcOffset + inIdx * innerMaskNum;
    scmOffset1 = inIdx * innerMatResNum;
    scmOffset2 = scmOffset1 + inIdx * innerMatResNum;
    spmOffset = scmOffset1;
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradBNG<T1, T2, IS_BF16>::CopyInSoftMax(
    LocalTensor<float> &maxInner, const GlobalTensor<float> &softmaxMaxGmIn, LocalTensor<float> &sumInner,
    const GlobalTensor<float> &softmaxSumGmIn)
{
    DataCopy(maxInner, softmaxMaxGmIn, innerReduceNum);
    DataCopy(sumInner, softmaxSumGmIn, innerReduceNum);
    vecInQue1.EnQue(sumInner);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradBNG<T1, T2, IS_BF16>::ClcSoftMax(LocalTensor<T2> &softmaxResInner,
                                                                                LocalTensor<T2> &reMatmulResInner,
                                                                                LocalTensor<float> &maxInner,
                                                                                LocalTensor<float> &sumInner)
{
    vecInQue1.DeQue<float>();
    LocalTensor<uint8_t> apiClcTensor = apiClcQue.AllocTensor<uint8_t>();
    apiClcTensor.SetSize(this->ordTilingData->baseParams.apiClcQueueSize);
    bool isBasicBlock = (this->sInner % 8 ==0) && (this->sFla % 64 ==0);
    if (isBasicBlock) {
        SimpleSoftMax<T2, true, true>(softmaxResInner, sumInner, maxInner, reMatmulResInner,
                                      apiClcTensor,
                                      this->ordTilingData->softmaxTilingData);
    } else {
        SimpleSoftMax<T2, true, false>(softmaxResInner, sumInner, maxInner, reMatmulResInner,
                                       apiClcTensor,
                                       this->ordTilingData->softmaxTilingData);
    }
    apiClcQue.FreeTensor(apiClcTensor);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradBNG<T1, T2, IS_BF16>::ClcMul(LocalTensor<T2> &frontResInner,
                                                                            LocalTensor<T2> &mulResInner,
                                                                            LocalTensor<T2> &softmaxResInner)
{
    pipe->ReadSpmBuffer(frontResInner, 0);

    Mul(mulResInner, softmaxResInner, frontResInner, innerMatResNumAlign);

    MIX_LOG("after mulResInner");

    Muls(mulResInner, mulResInner, (T2)(this->ordTilingData->baseParams.scaleValue), innerMatResNumAlign);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradBNG<T1, T2, IS_BF16>::ClcReDrop(
    LocalTensor<T2> &dvDropResInner, LocalTensor<T2> &softmaxResInner, const GlobalTensor<uint8_t> &dropMaskGmIn)
{
    LocalTensor<uint8_t> dvMaskInner = vecInQue2.AllocTensor<uint8_t>(); // [s_inner, s_fla]
    dvMaskInner.SetSize(innerMatResNum);
    struct DataCopyParams intriParams;
    intriParams.blockCount = (uint16_t)sInner;
    intriParams.blockLen = (uint16_t)((sFla / 8 + 31) / 32);
    intriParams.dstStride = 0;
    if ((this->ordTilingData->baseParams.S / 8 - intriParams.blockLen * 32) % 32 == 0) {
        intriParams.srcStride = (uint16_t)(this->ordTilingData->baseParams.S / 8 - intriParams.blockLen * 32) / 32;
        DataCopy(dvMaskInner, dropMaskGmIn, intriParams);
    } else {
        intriParams.blockLen = (uint16_t)sFla / 8;
        intriParams.srcStride = (uint16_t)(this->ordTilingData->baseParams.S / 8 - intriParams.blockLen);
        DataCopyPadParams intriParams1;
        DataCopyPad(dvMaskInner, dropMaskGmIn, intriParams, intriParams1);
    }
    vecInQue2.EnQue(dvMaskInner);

    vecInQue2.DeQue<uint8_t>();
    DropOutShapeInfo info;
    info.firstAxis = sInner;
    info.srcLastAxis = sFla;
    info.maskLastAxis = (sFla / 8 + 31) / 32 * 32;
    LocalTensor<uint8_t> apiClcTensor = apiClcQue.AllocTensor<uint8_t>();
    apiClcTensor.SetSize(this->ordTilingData->baseParams.apiClcQueueSize);
    DropOut<T2, true>(dvDropResInner, softmaxResInner, dvMaskInner, apiClcTensor,
                      this->ordTilingData->baseParams.keepProb, info);
    apiClcQue.FreeTensor(apiClcTensor);
    vecInQue2.FreeTensor(dvMaskInner);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradBNG<T1, T2, IS_BF16>::MyMatMulClc(const LocalTensor<T1> &tensorA,
                                                                                 const GlobalTensor<T1> &tensorB,
                                                                                 const GlobalTensor<float> &tensorC,
                                                                                 uint64_t isAtomicAddFlag, bool transA,
                                                                                 bool transB)
{
    mm3.SetTensorA(tensorA, transA);
    mm3.SetTensorB(tensorB, transB);
    mm3.template IterateAll<false>(tensorC, isAtomicAddFlag);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradBNG<T1, T2, IS_BF16>::DkvMatMulClc(
        const LocalTensor<T1> &tensorA,
        const GlobalTensor<T1> &tensorB,
        const GlobalTensor<float> &tensorC,
        uint64_t isAtomicAddFlag, bool transA,
        bool transB)
{
    mm4.SetTensorA(tensorA, transA);
    mm4.SetTensorB(tensorB, transB);
    mm4.template IterateAll<false>(tensorC, isAtomicAddFlag);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradBNG<T1, T2, IS_BF16>::SetFrontClcShape(LocalTensor<T1> &dxInner,
                                                                                      LocalTensor<T1> &forwardResInner,
                                                                                      LocalTensor<T2> &sftFrontResInner,
                                                                                      LocalTensor<T2> &frontResInner,
                                                                                      LocalTensor<uint8_t> &dpMaskInner)
{
    uint32_t innerFrontReClcShape[2] = {static_cast<uint32_t>(sInner), 32 / sizeof(T2)};
    dxInner.SetShapeInfo(ShapeInfo(2, innerMatInShape, DataFormat::ND));
    forwardResInner.SetShapeInfo(ShapeInfo(2, innerMatInShape, DataFormat::ND));
    sftFrontResInner.SetShapeInfo(ShapeInfo(2, innerFrontReClcShape, DataFormat::ND));
    frontResInner.SetShapeInfo(ShapeInfo(2, innerMatOutShape, DataFormat::ND));
    dpMaskInner.SetShapeInfo(ShapeInfo(2, innerMatOutShape, DataFormat::ND));
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradBNG<T1, T2, IS_BF16>::SetReClcShape(LocalTensor<T2> &mulResInner,
                                                                                   LocalTensor<float> &maxInner,
                                                                                   LocalTensor<float> &sumInner,
                                                                                   LocalTensor<T2> &dvDropResInner)
{
    mulResInner.SetShapeInfo(ShapeInfo(2, innerMatOutShape, DataFormat::ND));
    maxInner.SetShapeInfo(ShapeInfo(2, innerReduceShape, DataFormat::ND)); // [s_inner, 8]
    sumInner.SetShapeInfo(ShapeInfo(2, innerReduceShape, DataFormat::ND)); // [s_inner, 8]
    dvDropResInner.SetShapeInfo(ShapeInfo(2, innerMatOutShape, DataFormat::ND));
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void
FlashAttentionScoreGradBNG<T1, T2, IS_BF16>::DetermineLoopParams(
    const SplitCoreParams &coreParams, uint64_t flashKIdx, uint64_t s1OutIdx)
{
    bool isS1OutTail = (s1OutIdx == (uint64_t)this->s1OutRange - 1);

    this->sOut = coreParams.sOut;
    this->sFla = coreParams.sFla;
    this->outMatResSize = coreParams.outMatResSize;
    this->sInnerRange = coreParams.sInnerRange;
    this->sInner = coreParams.sInner;
    this->innerReduceNum = coreParams.innerReduceNum;
    this->innerMatResNum = coreParams.innerMatResNum;
    this->innerMatInputNum = coreParams.innerMatInputNum;
    this->innerMaskNum = coreParams.innerMaskNum;
    this->subMask = coreParams.subMask;
    this->subRange = coreParams.subRange;
    this->innerMatResSize = coreParams.innerMatResSize;
    this->sFlaBlockNum = coreParams.sFlaBlockNum;
    this->sFlaBlock = coreParams.sFlaBlock;
    this->sSubSFlaStride = coreParams.sSubSFlaStride;

    this->sOutAlign = coreParams.sOutAlign;
    this->sFlaAlign = coreParams.sFlaAlign;
    this->innerMatResNumAlign = coreParams.innerMatResNumAlign;
    this->innerMatResSizeAlign = coreParams.innerMatResSizeAlign;

    this->innerMatInShape[0] = this->sInner;
    this->innerMatInShape[1] = this->ordTilingData->baseParams.D;
    this->innerMatOutShape[0] = this->sInner;
    this->innerMatOutShape[1] = this->sFlaAlign;
    this->innerReduceShape[0] = this->sInner;
    this->innerReduceShape[1] = 32 / sizeof(float);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline bool FlashAttentionScoreGradBNG<T1, T2, IS_BF16>::CopyInPse(
    const uint64_t &s1OutLoopMaskOffset, const uint64_t &flashKIdx)
{
    if (pseGm.GetPhyAddr() == nullptr) {
        return false;
    }

    struct DataCopyParams dataCopyParams;
    struct DataCopyPadParams padParams;

    dataCopyParams.blockCount = this->sOut; // repeat time
    dataCopyParams.blockLen = (uint16_t)(sFla * sizeof(T1)); // 单位 Byte  可以不对齐
    dataCopyParams.srcStride = 0; // 单位block
    dataCopyParams.dstStride = 0; // 单位 Byte 可以不对其

    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;

    if (this->ordTilingData->baseParams.pseShapeType == 1) {  // BN1S
        dataCopyParams.blockCount = 1;
        // BN1S 初始 offset 已在外层模板处设置
    } else {
        pseOffset = s1OutLoopMaskOffset;
    }
    pseOffset += flashKIdx * this->mSFlaForClcOffset;

    LocalTensor<T1> pseUb = vecInQue2.AllocTensor<T1>();
    pseUb.SetSize(dataCopyParams.blockCount * sFlaAlign);
    DataCopyPad(pseUb, pseGm[pseOffset], dataCopyParams, padParams);
    vecInQue2.EnQue(pseUb);
    return true;
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradBNG<T1, T2, IS_BF16>::ClcPse(LocalTensor<T2> &mmResUb)
{
    LocalTensor<T1> pseUb = vecInQue2.DeQue<T1>();
    uint32_t eleNum = pseUb.GetSize();
    
    ClcPseBroadcast(mmResUb, pseUb, eleNum);
    
    vecInQue2.FreeTensor(pseUb);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradBNG<T1, T2, IS_BF16>::ClcPseBroadcast(
    LocalTensor<T2> &mmResUb, LocalTensor<T2> pseUb, uint32_t eleNum)
{
    if (this->ordTilingData->baseParams.pseShapeType == 0) {  // BNSS
        Add(mmResUb, mmResUb, pseUb, eleNum);
    } else {  // BN1S
        int32_t stepCalcNum = 256 / sizeof(T2);
        /* Total data number of single step should be smaller than 256bytes.
         * If larger, we need to do add multiple times. */
        stepCalcNum = sFla > stepCalcNum ? stepCalcNum : sFla;
        uint32_t innerLoop = sFla / stepCalcNum;
        uint32_t innerRemain = sFla % stepCalcNum;
        BinaryRepeatParams repParams;
        repParams.src0BlkStride = 1;
        repParams.src0RepStride = sFla / (32 / sizeof(T2));
        repParams.src1BlkStride = 1;
        repParams.src1RepStride = 0;
        repParams.dstRepStride = repParams.src0RepStride;
        repParams.blockNumber = repParams.src0RepStride;
        /* Total repeated times should be <= 255. If larger,
         * we need to do multiple inner loops. */
        uint32_t outerLoop = sOut / 255;  // singleProcessSOuterSize 不确定
        uint32_t outerRemain = sOut % 255;
        for (uint32_t i = 0; i < outerLoop; i++) {
            uint32_t outerOffset = (i >> 8) * sFla;
            ClcPseBroadcastInner(255, outerOffset, innerLoop, innerRemain,
                                 pseUb, mmResUb, stepCalcNum, repParams);
        }
        if (outerRemain > 0) {
            uint32_t outerOffset = (outerLoop >> 8) * sFla;
            ClcPseBroadcastInner(outerRemain, outerOffset, innerLoop, innerRemain,
                                 pseUb, mmResUb, stepCalcNum, repParams);
        }
    }
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradBNG<T1, T2, IS_BF16>::ClcPseBroadcastInner(
    uint32_t repeatTimes, uint32_t outerOffset, uint32_t innerLoop,
    uint32_t innerRemain, const LocalTensor<T2> &pseUb,
    const LocalTensor<T2> &mmResUb, uint32_t stepCalcNum, const BinaryRepeatParams &repParams)
{
    for (uint32_t j = 0; j < innerLoop; j++) {
        auto innerOffset = j * stepCalcNum;
        auto ubOffset = outerOffset + innerOffset;
        Add(mmResUb[ubOffset], mmResUb[ubOffset], pseUb[innerOffset], stepCalcNum,
            repeatTimes, repParams);
    }
    if (innerRemain > 0) {
        auto innerOffset = innerLoop * stepCalcNum;
        auto ubOffset = outerOffset + innerOffset;
        Add(mmResUb[ubOffset], mmResUb[ubOffset], pseUb[innerOffset], innerRemain,
            repeatTimes, repParams);
    }
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline bool FlashAttentionScoreGradBNG<T1, T2, IS_BF16>::CopyInAttenMask(const uint64_t &flashKIdx)
{
    if (attenMaskGm.GetPhyAddr() == nullptr) {
        return false;
    }
    MIX_LOG("attenMaskType: %d", this->ordTilingData->baseParams.attenMaskType);
    if (this->ordTilingData->baseParams.attenMaskType == 0) {  // 0 表示输入数据类型与 qkv 一致
        LocalTensor<T1> attenMaskUb = vecInQue2.AllocTensor<T1>();  // [s_inner, s_fla]
        attenMaskUb.SetSize(innerMatResNumAlign);
        attenMaskOffset += flashKIdx * this->mSFlaForClcOffset;
        struct DataCopyParams dataCopyParams;
        struct DataCopyPadParams padParams;

        dataCopyParams.blockCount = this->sOut; // repeat time
        dataCopyParams.blockLen = (uint16_t)(sFla * sizeof(T1)); // 单位 Byte  可以不对齐
        dataCopyParams.srcStride = 0; // 单位block
        dataCopyParams.dstStride = 0; // 单位 Byte 可以不对其

        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;

        DataCopyPad(attenMaskUb, attenMaskGm[attenMaskOffset], dataCopyParams, padParams);

        MIX_LOG("sFla %d", sFla);
        MIX_LOG("attenMaskUb");
        attenMaskUb.Print(256);

        vecInQue2.EnQue(attenMaskUb);
    } else {
        LocalTensor<uint8_t> attenMaskUb = vecInQue2.AllocTensor<uint8_t>();
        attenMaskUb.SetSize(innerMatResNumAlign);
        attenMaskOffset += flashKIdx * this->mSFlaForClcOffset;
        uint16_t blockLen = (uint16_t)sFlaBlock / sizeof(T1);
        uint16_t srcStride = (uint16_t)sSubSFlaStride / sizeof(T1);
        DataCopy(attenMaskUb, attenMaskU8Gm[attenMaskOffset], {(uint16_t)sInner, blockLen, srcStride, 0});
        vecInQue2.EnQue(attenMaskUb);
    }
    return true;
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradBNG<T1, T2, IS_BF16>::ClcAttenMask(LocalTensor<T2> &mmResUb)
{
    if (this->ordTilingData->baseParams.attenMaskType == 0) {  // 0 表示输入数据类型与 qkv 一致
        LocalTensor<T1> attenMaskUb = vecInQue2.DeQue<T1>();
        
        Muls(attenMaskUb, attenMaskUb, mulsValue, innerMatResNumAlign); // -10000

        Add(mmResUb, attenMaskUb, mmResUb, innerMatResNumAlign);
        
        vecInQue2.FreeTensor(attenMaskUb);
    } else {
        LocalTensor<uint8_t> attenMaskUb = vecInQue2.DeQue<uint8_t>();
        LocalTensor<T2> castedAttenMaskUb = vecClc1.Get<T2>(innerMatResNumAlign);
        castedAttenMaskUb.SetSize(innerMatResNumAlign);
        uint32_t shapeArray[] = {(uint32_t)sOut, (uint32_t)sFla};
        castedAttenMaskUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        if constexpr (IsSameType<T2, float>::value) {
            LocalTensor<half> apiClcTensor = apiClcQue.AllocTensor<half>();
            apiClcTensor.SetSize(innerMatResNumAlign);
            apiClcTensor.SetShapeInfo(ShapeInfo(2, innerMatOutShape, DataFormat::ND));
            Cast(apiClcTensor, attenMaskUb, RoundMode::CAST_NONE, innerMatResNumAlign);
            pipe_barrier(PIPE_V);
            Cast(castedAttenMaskUb, apiClcTensor, RoundMode::CAST_NONE, innerMatResNumAlign);
            apiClcQue.FreeTensor(apiClcTensor);
        } else {
            Cast(castedAttenMaskUb, attenMaskUb, RoundMode::CAST_NONE, innerMatResNumAlign);
        }
        pipe_barrier(PIPE_V);
        Muls(castedAttenMaskUb, castedAttenMaskUb, mulsValue, innerMatResNumAlign); // -10000
        Add(mmResUb, castedAttenMaskUb, mmResUb, innerMatResNumAlign);
        vecInQue2.FreeTensor(attenMaskUb);
    }
}


#endif