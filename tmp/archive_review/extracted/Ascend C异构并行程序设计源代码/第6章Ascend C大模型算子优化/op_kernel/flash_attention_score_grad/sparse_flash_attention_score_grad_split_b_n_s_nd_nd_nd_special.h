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

#ifndef _SPARSE_FLASH_ATTENTION_SCORE_GRAD_SPLIT_B_N_S_ND_ND_ND_SPECIAL_H_
#define _SPARSE_FLASH_ATTENTION_SCORE_GRAD_SPLIT_B_N_S_ND_ND_ND_SPECIAL_H_

#include "flash_attention_score_grad_base_kernel.h"
#include "kernel_operator_common_intf.h"


template <typename T1, typename T2, const bool IS_BF16 = false>
class SparseFlashAttentionScoreGradBNSNdNdNdSpecial : public FlashAttentionScoreGradBase<T1, T2, IS_BF16> {
public:
    __aicore__ inline SparseFlashAttentionScoreGradBNSNdNdNdSpecial(){};
    __aicore__ inline void Process();

    __aicore__ inline void Init(__gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dx, __gm__ uint8_t *query,
        __gm__ uint8_t *pse_shift, __gm__ uint8_t *drop_mask, __gm__ uint8_t *atten_mask, __gm__ uint8_t *forward_res,
        __gm__ uint8_t *softmax_max, __gm__ uint8_t *softmax_sum, __gm__ uint8_t *dq, __gm__ uint8_t *dk,
        __gm__ uint8_t *dv, __gm__ uint8_t *workspace,
        const FlashAttentionScoreGradTilingData *__restrict ordTilingData, TPipe *pipe_in);

    using aType1 = MatmulType<TPosition::GM, CubeFormat::ND, T1>;
    using bType1 = MatmulType<TPosition::GM, CubeFormat::ND, T1, true>;
    using cType1 = MatmulType<TPosition::VECCALC, CubeFormat::ND, T2>;
    using biasType1 = MatmulType<TPosition::GM, CubeFormat::ND, float>;

    using aType2 = MatmulType<TPosition::VECCALC, CubeFormat::ND, T1, true>;
    using bType2 = MatmulType<TPosition::GM, CubeFormat::ND, T1>;
    using cType2 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using biasType2 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    static constexpr MatmulConfig mm1Config{false, true, false, 64, 128, 128};
    static constexpr MatmulConfig mm2Config{false, true, false, 64, 128, 128};
    Matmul<aType1, bType1, cType1, biasType1, mm1Config> mm1;
    Matmul<aType1, bType1, cType1, biasType1, mm1Config> mm2;
    Matmul<aType2, bType2, cType2, biasType2, mm2Config> mm3;

    GlobalTensor<T1> mulResWorkSpace, dvDropResWorkSpace;

    bool isFirstTimes;
    bool isSetFlag;

protected:
    int32_t flashKIdxFactor;
    int32_t s1OutIdxFactor;

    __aicore__ inline bool isBatchTailCore();
    __aicore__ inline bool isSeqTailCore();
    __aicore__ inline void getSplitCoreIdx(uint64_t &splitBatchCoreIdx, uint64_t &splitSeqCoreIdx);
    
    __aicore__ inline void FrontCompute(const uint64_t &batchIdx, const uint64_t &flashKIdx, const uint64_t &s1OutIdx,
        const uint64_t &flashKLoopOffset, const uint64_t &s1OutLoopOffset, const uint64_t &s1OutLoopMaskOffset,
        const uint64_t &s1OutIdxStart, const uint64_t &s1OutIdxEnd, const uint64_t &sFlaIdxStart, const uint64_t &sFlaIdxEnd);

    __aicore__ inline void ReCompute(const uint64_t &batchIdx, const uint64_t &flashKIdx, const uint64_t &s1OutIdx,
        const uint64_t &flashKLoopOffset, const uint64_t &s1OutLoopOffset, const uint64_t &s1OutLoopMm3Offset,
        const uint64_t &s1OutLoopMaskOffset, const uint64_t &s1OutLoopReduceOffset, const uint64_t &s1OutIdxStart,
        const uint64_t &s1OutIdxEnd, const uint64_t &sFlaIdxStart, const uint64_t &sFlaIdxEnd);

    __aicore__ inline void MyMatMulClc(const GlobalTensor<T1> &tensorA, const GlobalTensor<T1> &tensorB,
        const GlobalTensor<float> &tensorC, uint64_t isAtomicAddFlag = 0, bool transA = false, bool transB = false,
        bool isLastTimes = false);
};

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void SparseFlashAttentionScoreGradBNSNdNdNdSpecial<T1, T2, IS_BF16>::MyMatMulClc(const GlobalTensor<T1> &tensorA,
    const GlobalTensor<T1> &tensorB, const GlobalTensor<float> &tensorC, uint64_t isAtomicAddFlag, bool transA,
    bool transB, bool isLastTimes)
{
    mm3.SetTensorA(tensorA, transA);
    mm3.SetTensorB(tensorB, transB);
    if (isLastTimes) {
        mm3.template IterateAll<true>(tensorC, isAtomicAddFlag);
    } else {
        mm3.template IterateAll<false>(tensorC, isAtomicAddFlag);
    }
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void SparseFlashAttentionScoreGradBNSNdNdNdSpecial<T1, T2, IS_BF16>::Init(__gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *dx, __gm__ uint8_t *query, __gm__ uint8_t *pse_shift, __gm__ uint8_t *drop_mask,
    __gm__ uint8_t *atten_mask, __gm__ uint8_t *forward_res, __gm__ uint8_t *softmax_max, __gm__ uint8_t *softmax_sum,
    __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *workspace,
    const FlashAttentionScoreGradTilingData *__restrict ordTilingData, TPipe *pipe_in)
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
    this->dqGm.SetGlobalBuffer((__gm__ T1 *)dq);
    this->dkGm.SetGlobalBuffer((__gm__ T1 *)dk);
    this->dvGm.SetGlobalBuffer((__gm__ T1 *)dv);
    this->workspaceGm.SetGlobalBuffer((__gm__ T2 *)workspace);
    this->syncGlobal.SetGlobalBuffer((__gm__ int32_t *)workspace, SYNC_GLOBAL_WORKSPACE_SIZE);
    InitOutput<int32_t>(this->syncGlobal[GetBlockIdx() * 8], 8, 0);

    this->mBlockIdx = GetBlockIdx();
    this->ordTilingData = ordTilingData;
    this->pipe = pipe_in;

    this->innerMatInShape[0] = this->ordTilingData->splitCoreParams.sInner;
    this->innerMatInShape[1] = this->ordTilingData->baseParams.D;
    this->innerMatOutShape[0] = this->ordTilingData->splitCoreParams.sInner;
    this->innerMatOutShape[1] = this->ordTilingData->splitCoreParams.sFla;
    this->innerReduceShape[0] = this->ordTilingData->splitCoreParams.sInner;
    this->innerReduceShape[1] = 32 / sizeof(T2);

    if constexpr (IS_BF16) {
        this->pipe->InitBuffer(this->vecInQue1, 2, this->ordTilingData->splitCoreParams.innerMatInputSize / 2);
        if (this->ordTilingData->baseParams.isHighPerformanceMode == 1) {
            this->pipe->InitBuffer(this->vecInQue2, 1, 0);
        }
        else {
            this->pipe->InitBuffer(this->vecInQue2, 1, this->ordTilingData->splitCoreParams.innerMatResSize / 2);
        }
        if (this->ordTilingData->baseParams.D <= this->ordTilingData->splitCoreParams.sFla) {
            this->pipe->InitBuffer(this->vecClc1, this->ordTilingData->splitCoreParams.innerMatResSize);
            this->pipe->InitBuffer(this->vecClc2, this->ordTilingData->splitCoreParams.innerMatResSize);
        }
        else {
            this->pipe->InitBuffer(this->vecClc1, this->ordTilingData->splitCoreParams.innerMatInputSize);
            this->pipe->InitBuffer(this->vecClc2, this->ordTilingData->splitCoreParams.innerMatInputSize);
        }
    } else {
        this->pipe->InitBuffer(this->vecInQue1, 2, this->ordTilingData->splitCoreParams.innerMatInputSize);
        this->pipe->InitBuffer(this->vecInQue2, 1, this->ordTilingData->splitCoreParams.innerMatResSize);
        this->pipe->InitBuffer(this->vecClc1, this->ordTilingData->splitCoreParams.innerMatResSize);
        this->pipe->InitBuffer(this->vecClc2, this->ordTilingData->splitCoreParams.innerMatResSize);
    }
    this->pipe->InitBuffer(this->apiClcQue, 1, this->ordTilingData->baseParams.apiClcQueueSize);

    this->pipe->InitSpmBuffer(this->workspaceGm[SYNC_GLOBAL_WORKSPACE_SIZE +
        this->mBlockIdx * this->ordTilingData->splitCoreParams.outMatResNum * FRONT_COMPUTE_TIMES + 
        this->mBlockIdx * this->ordTilingData->splitCoreParams.s1OutRange * this->ordTilingData->splitCoreParams.sOut * 32 / sizeof(T2)],
        this->ordTilingData->splitCoreParams.outMatResSize * FRONT_COMPUTE_TIMES + this->ordTilingData->splitCoreParams.s1OutRange * this->ordTilingData->splitCoreParams.sOut * 32);

    uint64_t buff_offset = SYNC_GLOBAL_WORKSPACE_SIZE +
        GetBlockNum() * GetTaskRation() * this->ordTilingData->splitCoreParams.outMatResNum * FRONT_COMPUTE_TIMES
        + GetBlockNum() * GetTaskRation() * this->ordTilingData->splitCoreParams.s1OutRange * this->ordTilingData->splitCoreParams.sOut * 32 / sizeof(T2);
    mm1.SetWorkspace((__gm__ uint8_t *)this->workspaceGm[buff_offset +
        this->mBlockIdx * CUBE_PRELOAD_BLOCK_NUM * this->ordTilingData->splitCoreParams.outMatResNum]
                         .GetPhyAddr(),
        CUBE_PRELOAD_BLOCK_NUM * this->ordTilingData->splitCoreParams.outMatResSize);

    buff_offset +=
        GetBlockNum() * GetTaskRation() * CUBE_PRELOAD_BLOCK_NUM * this->ordTilingData->splitCoreParams.outMatResNum;
    mm2.SetWorkspace((__gm__ uint8_t *)this->workspaceGm[buff_offset +
        this->mBlockIdx * CUBE_PRELOAD_BLOCK_NUM2 * this->ordTilingData->splitCoreParams.outMatResNum].GetPhyAddr(),
        CUBE_PRELOAD_BLOCK_NUM2 * this->ordTilingData->splitCoreParams.outMatResSize);

    buff_offset = ((buff_offset + GetBlockNum() * GetTaskRation() * CUBE_PRELOAD_BLOCK_NUM2 *
        this->ordTilingData->splitCoreParams.outMatResNum) * 4 + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;

    this->dqWorkSpace.SetGlobalBuffer((__gm__ float *)workspace + buff_offset / 4);

    buff_offset = (buff_offset + this->ordTilingData->castTiling.allNumQuery * 4 + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE *
        ADDR_ALIGN_SIZE;
    this->dkWorkSpace.SetGlobalBuffer((__gm__ float *)workspace + buff_offset / 4);

    buff_offset = (buff_offset + this->ordTilingData->castTiling.allNumKv * 4 + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE *
        ADDR_ALIGN_SIZE;
    this->dvWorkSpace.SetGlobalBuffer((__gm__ float *)workspace + buff_offset / 4);

    buff_offset = (buff_offset + this->ordTilingData->castTiling.allNumKv * 4 + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE *
        ADDR_ALIGN_SIZE;
    this->syncLastGlobal.SetGlobalBuffer((__gm__ int32_t *)workspace + buff_offset / 4, SYNC_GLOBAL_WORKSPACE_SIZE);

    InitOutput<int32_t>(this->syncLastGlobal[this->mBlockIdx * 8], 8, 0);

    this->InitAttenMulsValue();
    DropOutBitModeInit();

    uint64_t buff_offset1 = (buff_offset + 1*1024*1024 +
        this->mBlockIdx * CUBE_POSTLOAD_BLOCK_NUM * 2 * this->ordTilingData->splitCoreParams.outMatResNum * sizeof(T1) +
        ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    this->mulResWorkSpace.SetGlobalBuffer((__gm__ T1 *)workspace + buff_offset1 / sizeof(T1));

    buff_offset = (buff_offset + 1*1024*1024 + this->ordTilingData->baseParams.coreNum * CUBE_POSTLOAD_BLOCK_NUM * 2 *
        this->ordTilingData->splitCoreParams.outMatResNum * sizeof(T1) +
        this->mBlockIdx * CUBE_POSTLOAD_BLOCK_NUM * 2 * this->ordTilingData->splitCoreParams.outMatResNum * sizeof(T1) +
        ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    this->dvDropResWorkSpace.SetGlobalBuffer((__gm__ T1 *)workspace + buff_offset / sizeof(T1));

    this->isFirstTimes = true;
    this->isSetFlag = false;
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void SparseFlashAttentionScoreGradBNSNdNdNdSpecial<T1, T2, IS_BF16>::FrontCompute(const uint64_t &batchIdx,
    const uint64_t &flashKIdx, const uint64_t &s1OutIdx, const uint64_t &flashKLoopOffset, const uint64_t &s1OutLoopOffset,
    const uint64_t &s1OutLoopMaskOffset, const uint64_t &s1OutIdxStart, const uint64_t &s1OutIdxEnd,
    const uint64_t &sFlaIdxStart, const uint64_t &sFlaIdxEnd)
{
    if (!this->isFirstTimes) {
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
        this->isSetFlag = false;
    }
    LocalTensor<T1> dxInner = this->vecInQue1.template AllocTensor<T1>();         // [s_inner, H/N]
    LocalTensor<T1> forwardResInner = this->vecInQue1.template AllocTensor<T1>(); // [s_inner, H/N]
    LocalTensor<T2> sftFrontResInner = this->vecClc2.template Get<T2>(this->sInner * 32 / sizeof(T2));
    LocalTensor<T2> frontResInner = this->vecClc1.template Get<T2>(this->innerMatResNum);
    LocalTensor<T2> &dpResInner = frontResInner;                         // [s_inner, s_fla]
    LocalTensor<T2> &dpMatmmulResInner = frontResInner;                  // [s_inner, s_fla]
    LocalTensor<uint8_t> dpMaskInner = this->vecInQue2.template AllocTensor<uint8_t>(); // [s_inner, s_fla]

    if constexpr (!IS_BF16) {
        this->frontResInnerCache = frontResInner;
    }

    dxInner.SetSize(this->innerMatInputNum);
    forwardResInner.SetSize(this->innerMatInputNum);
    dpMaskInner.SetSize(this->innerMatResNum);
    this->SetFrontClcShape(dxInner, forwardResInner, sftFrontResInner, frontResInner, dpMaskInner);

    this->CopyInSoftMaxGrad(dxInner, this->dxGm[s1OutLoopOffset], forwardResInner, this->forwardResGm[s1OutLoopOffset]);
    if (this->isDrop) {
        uint64_t maskOffset = s1OutLoopMaskOffset + flashKIdx * this->mSFlaForClcOffset;
        this->CopyInDrop(dpMaskInner, this->dropMaskGm[maskOffset / 8]);
    }

    this->ClcSoftMaxGrad(sftFrontResInner, dxInner, forwardResInner);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);

    mm1.template GetTensorC<false>(dpMatmmulResInner, false, true);
    if (s1OutIdx < s1OutIdxEnd - 1 &&
        (s1OutIdx - s1OutIdxStart) % CUBE_PRELOAD_BLOCK_NUM == CUBE_PRELOAD_BLOCK_NUM - 1) {
        this->mm1.SetTail(this->sOut * this->curCubePreloadNum, this->sFla, this->ordTilingData->baseParams.D); // M N K
        if (s1OutIdxEnd - s1OutIdx <= CUBE_PRELOAD_BLOCK_NUM) {
            this->mm1.SetTail(this->sOut * (s1OutIdxEnd - 1 - s1OutIdx), this->sFla,
                this->ordTilingData->baseParams.D); // M N K
        }
        if (this->layoutSBH == 1) {
            mm1.SetTensorA(
                this->dxGm[s1OutLoopOffset + this->sOut * this->ordTilingData->baseParams.B * this->ordTilingData->baseParams.H]);
        } else {
            mm1.SetTensorA(this->dxGm[s1OutLoopOffset + this->sOut * this->ordTilingData->baseParams.H]);
        }
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID7);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID7);
        this->mm1.template Iterate<false>();
    }

    if (this->isDrop) {
        this->ClcDrop(dpResInner, dpMatmmulResInner, dpMaskInner);
    }

    if constexpr (IS_BF16) {
        this->ClcSub(frontResInner, dpMatmmulResInner, sftFrontResInner);
        this->pipe->WriteSpmBuffer(frontResInner, 0, 0); // all [s_out, s_fla]
    } else {
        this->ClcSub(this->frontResInnerCache, dpMatmmulResInner, sftFrontResInner);
    }

    this->vecInQue1.FreeTensor(dxInner);
    this->vecInQue1.FreeTensor(forwardResInner);
    this->vecInQue2.FreeTensor(dpMaskInner);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void SparseFlashAttentionScoreGradBNSNdNdNdSpecial<T1, T2, IS_BF16>::ReCompute(
    const uint64_t &batchIdx, const uint64_t &flashKIdx, const uint64_t &s1OutIdx, const uint64_t &flashKLoopOffset,
    const uint64_t &s1OutLoopOffset, const uint64_t &s1OutLoopMm3Offset, const uint64_t &s1OutLoopMaskOffset,
    const uint64_t &s1OutLoopReduceOffset, const uint64_t &s1OutIdxStart, const uint64_t &s1OutIdxEnd,
    const uint64_t &sFlaIdxStart, const uint64_t &sFlaIdxEnd)
{
    bool isLastTimes = false;
    if ((batchIdx == this->splitedBatchRange - 1) &&
        (flashKIdx == sFlaIdxEnd - 1) &&
        (s1OutIdx == s1OutIdxEnd - 1)) {
        isLastTimes = true;
    }

    LocalTensor<T2> frontResInner = this->vecClc1.template Get<T2>(this->innerMatResNum);
    LocalTensor<T2> &mulResInner = frontResInner;
    LocalTensor<T2> dvDropResInner = this->vecClc2.template Get<T2>(this->innerMatResNum); // [s_inner, s_fla]

    LocalTensor<T2> &reMatmulResInner = dvDropResInner;  // [s_inner, s_fla]
    LocalTensor<T2> &softmaxResInner = dvDropResInner;   // [s_inner, s_fla]
    LocalTensor<T2> &attenMaskResInner = dvDropResInner; // [s_inner, s_fla]

    LocalTensor<float> maxInner = this->vecInQue1.template AllocTensor<float>(); // [s_inner, 8]
    LocalTensor<float> sumInner = this->vecInQue1.template AllocTensor<float>(); // [s_inner, 8]
    maxInner.SetSize(this->innerReduceNum);
    sumInner.SetSize(this->innerReduceNum);

    this->SetReClcShape(mulResInner, maxInner, sumInner, dvDropResInner);
    this->CopyInSoftMax(maxInner, this->softmaxMaxGm[s1OutLoopReduceOffset], sumInner, this->softmaxSumGm[s1OutLoopReduceOffset]);

    bool clcAttenMask = false;
    bool clsPse = this->CopyInPse(s1OutLoopMaskOffset, flashKIdx);
    if (!clsPse) {
        clcAttenMask = this->CopyInAttenMask(flashKIdx);
    }

    mm2.template GetTensorC<false>(reMatmulResInner, false, true);
    if (s1OutIdx < s1OutIdxEnd - 1 && (s1OutIdx - s1OutIdxStart) % CUBE_PRELOAD_BLOCK_NUM2 == CUBE_PRELOAD_BLOCK_NUM2 - 1) {
        this->mm2.SetTail(this->sOut * this->curCubePreloadNum2, this->sFla, this->ordTilingData->baseParams.D);
        if (s1OutIdxEnd - s1OutIdx <= CUBE_PRELOAD_BLOCK_NUM2) {
            this->mm2.SetTail(this->sOut * (s1OutIdxEnd - 1 - s1OutIdx), this->sFla, this->ordTilingData->baseParams.D);
        }
        if (this->layoutSBH == 1) {
            mm2.SetTensorA(this->queryGm[s1OutLoopOffset + this->sOut * this->ordTilingData->baseParams.B * this->ordTilingData->baseParams.H]);
        } else {
            mm2.SetTensorA(this->queryGm[s1OutLoopOffset + this->sOut * this->ordTilingData->baseParams.H]);
        }
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID7);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID7);
        this->mm2.template Iterate<false>();                  // [sInner, H/N] * [H/N, sFla]
    }

    if (clsPse) {
        this->ClcPse(reMatmulResInner);
        clcAttenMask = this->CopyInAttenMask(flashKIdx);
    }
    Muls(reMatmulResInner, reMatmulResInner, (T2)(this->ordTilingData->baseParams.scaleValue), this->innerMatResNum);
    if (clcAttenMask) {
        this->ClcAttenMask(reMatmulResInner);
    }

    this->ClcSoftMax(softmaxResInner, attenMaskResInner, maxInner, sumInner);

    this->ClcMul(frontResInner, mulResInner, softmaxResInner);

    LocalTensor<T1> castedMulResInner;
    if constexpr (IS_BF16) {
        pipe_barrier(PIPE_V);
        castedMulResInner = this->vecClc1.template Get<T1>(this->innerMatResNum);
        castedMulResInner.SetShapeInfo(ShapeInfo(2, this->innerMatOutShape, DataFormat::ND));
        Cast(castedMulResInner, mulResInner, RoundMode::CAST_ROUND, this->innerMatResNum);
        pipe_barrier(PIPE_V);
    }

    if (this->isDrop) {
        uint64_t maskOffset = s1OutLoopMaskOffset + flashKIdx * this->mSFlaForClcOffset;
        this->ClcReDrop(dvDropResInner, softmaxResInner, this->dropMaskGm[maskOffset / 8]);
    }
    LocalTensor<T1> castedDvDropResInner;
    if constexpr (IS_BF16) {
        castedDvDropResInner = this->vecClc2.template Get<T1>(this->innerMatResNum);
        castedDvDropResInner.SetShapeInfo(ShapeInfo(2, this->innerMatOutShape, DataFormat::ND));
        Cast(castedDvDropResInner, dvDropResInner, RoundMode::CAST_ROUND, this->innerMatResNum);
        pipe_barrier(PIPE_V);
    }

    // move to workspace
    uint64_t mm3IdxOffset = (s1OutIdx  - s1OutIdxStart) % (CUBE_POSTLOAD_BLOCK_NUM * 2) * this->innerMatResNum;
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);

    if constexpr (IS_BF16) {
        DataCopy(this->mulResWorkSpace[mm3IdxOffset], castedMulResInner, this->innerMatResNum);
        DataCopy(this->dvDropResWorkSpace[mm3IdxOffset], castedDvDropResInner, this->innerMatResNum);
    } else {
        DataCopy(this->mulResWorkSpace[mm3IdxOffset], mulResInner, this->innerMatResNum);
        DataCopy(this->dvDropResWorkSpace[mm3IdxOffset], dvDropResInner, this->innerMatResNum);
    }

    if (!isLastTimes)
    {
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
        this->isSetFlag = true;
    }

    if (((s1OutIdx - s1OutIdxStart) % CUBE_POSTLOAD_BLOCK_NUM == CUBE_POSTLOAD_BLOCK_NUM - 1) ||
            s1OutIdx == s1OutIdxEnd - 1) {
        bool isTail = (s1OutIdx - s1OutIdxStart) % CUBE_POSTLOAD_BLOCK_NUM != CUBE_POSTLOAD_BLOCK_NUM - 1;
        uint32_t tailBlockNum = (s1OutIdxEnd - s1OutIdxStart) % CUBE_POSTLOAD_BLOCK_NUM;
        uint32_t mm3TensorAOffset = int((s1OutIdx - s1OutIdxStart) / CUBE_POSTLOAD_BLOCK_NUM) % 2 *
                                    this->innerMatResNum * CUBE_POSTLOAD_BLOCK_NUM;

        this->mm3.SetTail(this->sOut * CUBE_POSTLOAD_BLOCK_NUM, -1, this->sFla);
        if (isTail) {
            this->mm3.SetTail(this->sOut * tailBlockNum, -1, this->sFla); // M N K
        }
        this->MyMatMulClc(this->mulResWorkSpace[mm3TensorAOffset], this->keyGm[flashKLoopOffset],
                          this->dqWorkSpace[s1OutLoopMm3Offset], true, false, false, isLastTimes);

        this->mm3.SetTail(this->sFla, -1, this->sOut * CUBE_POSTLOAD_BLOCK_NUM);
        if (isTail) {
            this->mm3.SetTail(this->sFla, -1, this->sOut * tailBlockNum); // M N K
        }
        this->MyMatMulClc(this->mulResWorkSpace[mm3TensorAOffset], this->queryGm[s1OutLoopMm3Offset],
                         this->dkWorkSpace[flashKLoopOffset], true, true, false, isLastTimes);

        this->mm3.SetTail(this->sFla, -1, this->sOut * CUBE_POSTLOAD_BLOCK_NUM);
        if (isTail) {
            this->mm3.SetTail(this->sFla, -1, this->sOut * tailBlockNum); // M N K
        }
        this->MyMatMulClc(this->dvDropResWorkSpace[mm3TensorAOffset], this->dxGm[s1OutLoopMm3Offset],
                          this->dvWorkSpace[flashKLoopOffset], true, true, false, isLastTimes);
    }

    this->vecInQue1.FreeTensor(maxInner);
    this->vecInQue1.FreeTensor(sumInner);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void SparseFlashAttentionScoreGradBNSNdNdNdSpecial<T1, T2, IS_BF16>::Process()
{
    this->isDrop = false;
    this->isFirstTimes = true;

    if (this->ordTilingData->baseParams.keepProb < 1) {
        this->isDrop = true;
    }

    this->InitGm();

    this->layoutSBH = this->ordTilingData->baseParams.layoutSBH;

    uint64_t splitBatchCoreIdx = 0, splitSeqCoreIdx = 0;
    getSplitCoreIdx(splitBatchCoreIdx, splitSeqCoreIdx);

    this->mSFlaForClcOffset = this->ordTilingData->splitCoreParams.sFla;
    this->mSOutForClcOffset = this->ordTilingData->splitCoreParams.sOut;

    // Before sOut&&sFla parameters
    this->splitedBatchRange = this->ordTilingData->splitCoreParams.splitedBatchRange;
    this->flashKLoopRange = this->ordTilingData->splitCoreParams.flashKLoopRange;
    this->s1OutRange = this->ordTilingData->splitCoreParams.s1OutRange;
    this->sFla = this->ordTilingData->splitCoreParams.sFla;
    this->sOut = this->ordTilingData->splitCoreParams.sOut;
    this->outMaskNum = this->ordTilingData->splitCoreParams.outMaskNum;
    this->outReduceNum = this->ordTilingData->splitCoreParams.outReduceNum;
    this->outMatResSize = this->ordTilingData->splitCoreParams.outMatResSize;

    if (isBatchTailCore()) {
        this->splitedBatchRange = this->ordTilingData->tailSplitCoreParams.splitedBatchRange;
    }

    if (isSeqTailCore()) { // Tail Core Engagement
        this->flashKLoopRange = this->ordTilingData->tailSplitCoreParams.flashKLoopRange;
        this->s1OutRange = this->ordTilingData->tailSplitCoreParams.s1OutRange;

        this->sFla = this->ordTilingData->tailSplitCoreParams.sFla;
        this->sOut = this->ordTilingData->tailSplitCoreParams.sOut;
        this->outMaskNum = this->ordTilingData->tailSplitCoreParams.outMaskNum;
        this->outReduceNum = this->ordTilingData->tailSplitCoreParams.outReduceNum;
    }

    int32_t nextTokenBlocks = this->ordTilingData->baseParams.nextTokensBlocks,
            preTokenBlocks = this->ordTilingData->baseParams.preTokensBlocks;

    flashKIdxFactor = this->sFla > this->sOut ? this->sFla / this->sOut : 1;
    s1OutIdxFactor = this->sFla > this->sOut ? 1 : this->sOut / this->sFla;
    if (this->sFla != this->sOut) {
        if (s1OutIdxFactor == 1) {
            preTokenBlocks += 1;
        } else {
            nextTokenBlocks += 1;
        }
    }

    uint64_t s1OutIdxStartS1 = this->mBlockIdx * this->ordTilingData->splitCoreParams.s1OutRange;
    uint64_t s1OutIdxEndS1 = s1OutIdxStartS1 + this->s1OutRange;
    uint64_t sFlaIdxStart = max(((int32_t)(s1OutIdxStartS1 * this->sOut - this->ordTilingData->baseParams.preTokens)
                            + this->sFla - 1) / this->sFla - 1, (int32_t)0);
    uint64_t sFlaIdxEnd = (min((int32_t)(s1OutIdxEndS1 * this->sOut + this->ordTilingData->baseParams.nextTokens),
                            (int32_t)this->flashKLoopRange * this->sFla) + this->sFla - 1) / this->sFla;

    for (int32_t batchIdx = 0; batchIdx < this->splitedBatchRange; batchIdx++) {
        int32_t bIdx = (this->mBlockIdx / this->ordTilingData->baseParams.splitSeqFactor *
            this->ordTilingData->splitCoreParams.splitedBatchRange + batchIdx) /
            this->ordTilingData->baseParams.N;
        int32_t nIdx = (this->mBlockIdx / this->ordTilingData->baseParams.splitSeqFactor *
            this->ordTilingData->splitCoreParams.splitedBatchRange + batchIdx) %
            this->ordTilingData->baseParams.N;

        uint64_t batchLoopOffset = bIdx * this->ordTilingData->baseParams.S * this->ordTilingData->baseParams.H +
                    nIdx * this->ordTilingData->baseParams.D +
                    splitSeqCoreIdx * this->ordTilingData->splitCoreParams.splitedCoreSeqNum * this->ordTilingData->baseParams.H;

        if (this->layoutSBH == 1) {
            batchLoopOffset = bIdx * this->ordTilingData->baseParams.H + nIdx * this->ordTilingData->baseParams.D +
                               splitSeqCoreIdx * this->ordTilingData->splitCoreParams.splitedCoreSeqNum * this->ordTilingData->baseParams.B * this->ordTilingData->baseParams.H;
        }

        uint64_t batchLoopMaskOffset = splitBatchCoreIdx * this->ordTilingData->splitCoreParams.splitedCoreBatchMaskNum +
                                batchIdx * this->ordTilingData->baseParams.allMatResNum +
                                splitSeqCoreIdx * this->ordTilingData->splitCoreParams.splitAllMatResNum;
        uint64_t batchLoopReduceOffset = splitBatchCoreIdx * this->ordTilingData->splitCoreParams.splitedCoreBatchReduceNum +
                                batchIdx * this->ordTilingData->baseParams.allReduceNum +
                                splitSeqCoreIdx * this->ordTilingData->splitCoreParams.splitAllReduceNum;

        for (int32_t flashKIdx = 0; flashKIdx < this->flashKLoopRange; flashKIdx++) {
            uint64_t flashKLoopOffset = bIdx * this->ordTilingData->baseParams.S * this->ordTilingData->baseParams.H +
                                    nIdx * this->ordTilingData->baseParams.D +
                                    flashKIdx * this->sFla * this->ordTilingData->baseParams.H;
            if (this->layoutSBH == 1) {
                flashKLoopOffset = bIdx * this->ordTilingData->baseParams.H + nIdx * this->ordTilingData->baseParams.D +
                                   flashKIdx * this->sFla * this->ordTilingData->baseParams.B * this->ordTilingData->baseParams.H;
            }

            uint32_t idx = 0;
            int32_t s1OutIdxStart = min(this->s1OutRange, max((int32_t)0, (int32_t)(flashKIdx * flashKIdxFactor - nextTokenBlocks - (int32_t)splitSeqCoreIdx * this->ordTilingData->splitCoreParams.s1OutRange))) / s1OutIdxFactor;
            int32_t s1OutIdxEnd = max(0, min((int32_t)this->s1OutRange, (int32_t)(flashKIdx * flashKIdxFactor + preTokenBlocks - (int32_t)splitSeqCoreIdx * this->ordTilingData->splitCoreParams.s1OutRange + 1))) / s1OutIdxFactor;

            int32_t mLength = s1OutIdxEnd - s1OutIdxStart;
            if (mLength <= 0) {
                continue;
            }

            this->mm1.SetTensorB(this->valueGm[flashKLoopOffset], true);
            this->mm2.SetTensorB(this->keyGm[flashKLoopOffset], true);

            for (int32_t s1OutIdx = s1OutIdxStart; s1OutIdx < s1OutIdxEnd; s1OutIdx++) {
                uint64_t s1OutLoopOffset =
                    batchLoopOffset + s1OutIdx * this->sOut * this->ordTilingData->baseParams.H;
                uint64_t s1OutLoopMm3Offset = batchLoopOffset + ((s1OutIdx - s1OutIdxStart) / CUBE_POSTLOAD_BLOCK_NUM *
                    CUBE_POSTLOAD_BLOCK_NUM + s1OutIdxStart) * this->sOut * this->ordTilingData->baseParams.H;

                if (this->layoutSBH == 1) {
                    s1OutLoopOffset = batchLoopOffset + s1OutIdx * this->sOut * this->ordTilingData->baseParams.B * this->ordTilingData->baseParams.H;
                    s1OutLoopMm3Offset =
                        batchLoopOffset +
                        ((s1OutIdx - s1OutIdxStart) / CUBE_POSTLOAD_BLOCK_NUM * CUBE_POSTLOAD_BLOCK_NUM + s1OutIdxStart) *
                        this->sOut * this->ordTilingData->baseParams.B * this->ordTilingData->baseParams.H;
                }
                uint64_t s1OutLoopMaskOffset = batchLoopMaskOffset + s1OutIdx * this->outMaskNum;
                uint64_t s1OutLoopReduceOffset = batchLoopReduceOffset + s1OutIdx * this->outReduceNum;

                this->pseOffset = bIdx * this->ordTilingData->baseParams.N * this->ordTilingData->baseParams.S +
                                    nIdx * this->ordTilingData->baseParams.S;  // 按 BN1S场景设置
                this->attenMaskOffset = s1OutIdx * this->outMaskNum +
                                        splitSeqCoreIdx * this->ordTilingData->splitCoreParams.splitAllMatResNum;
                if (this->ordTilingData->baseParams.attenMaskShapeType == 1) {  // B1SS
                    this->attenMaskOffset +=
                        bIdx * this->ordTilingData->baseParams.S * this->ordTilingData->baseParams.S;
                }

                if (isSeqTailCore()) {
                    this->DetermineLoopParams(this->ordTilingData->tailSplitCoreParams, flashKIdx, s1OutIdx);
                } else {
                    this->DetermineLoopParams(this->ordTilingData->splitCoreParams, flashKIdx, s1OutIdx);
                }

                if (idx == 0) {
                    this->curCubePreloadNum = (s1OutIdxEnd - s1OutIdx >= CUBE_PRELOAD_BLOCK_NUM ? CUBE_PRELOAD_BLOCK_NUM : s1OutIdxEnd - s1OutIdx);
                    this->mm1.SetTail(this->sOut * this->curCubePreloadNum, this->sFla, this->ordTilingData->baseParams.D); // M N K
                    this->mm1.SetTensorA(this->dxGm[s1OutLoopOffset]);
                    this->mm1.template Iterate<false>();

                    this->curCubePreloadNum2 = (s1OutIdxEnd - s1OutIdx >= CUBE_PRELOAD_BLOCK_NUM2 ? CUBE_PRELOAD_BLOCK_NUM2 : s1OutIdxEnd - s1OutIdx);
                    this->mm2.SetTail(this->sOut * this->curCubePreloadNum2, this->sFla, this->ordTilingData->baseParams.D);
                    this->mm2.SetTensorA(this->queryGm[s1OutLoopOffset]);
                    this->mm2.template Iterate<false>();  // [sInner, H/N] * [H/N, sFla]
                }

                this->FrontCompute(batchIdx, flashKIdx, s1OutIdx,
                                   flashKLoopOffset, s1OutLoopOffset, s1OutLoopMaskOffset,
                                   s1OutIdxStart, s1OutIdxEnd, sFlaIdxStart, sFlaIdxEnd);
                this->ReCompute(batchIdx, flashKIdx, s1OutIdx, flashKLoopOffset, s1OutLoopOffset, s1OutLoopMm3Offset,
                                s1OutLoopMaskOffset, s1OutLoopReduceOffset,
                                s1OutIdxStart, s1OutIdxEnd, sFlaIdxStart, sFlaIdxEnd);
                idx++;
                this->isFirstTimes = false;
            }
        }
    }

    if (this->isSetFlag) {
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
    }
    
    // synaAll
    this->SyncAllLast();
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline bool SparseFlashAttentionScoreGradBNSNdNdNdSpecial<T1, T2, IS_BF16>::isBatchTailCore()
{
    return this->mBlockIdx != 0 &&
        (this->mBlockIdx / this->ordTilingData->baseParams.splitSeqFactor ==
        this->ordTilingData->baseParams.splitBatchFactor - 1);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline bool SparseFlashAttentionScoreGradBNSNdNdNdSpecial<T1, T2, IS_BF16>::isSeqTailCore() {
    return this->mBlockIdx != 0 && (this->mBlockIdx % this->ordTilingData->baseParams.splitSeqFactor == this->ordTilingData->baseParams.splitSeqFactor - 1);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void SparseFlashAttentionScoreGradBNSNdNdNdSpecial<T1, T2, IS_BF16>::getSplitCoreIdx(
    uint64_t &splitBatchCoreIdx, uint64_t &splitSeqCoreIdx)
{
    splitBatchCoreIdx = this->mBlockIdx / this->ordTilingData->baseParams.splitSeqFactor;
    splitSeqCoreIdx = this->mBlockIdx % this->ordTilingData->baseParams.splitSeqFactor;
}

#endif
