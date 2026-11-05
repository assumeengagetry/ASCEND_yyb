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

#ifndef _FLASH_ATTENTION_SCORE_GRAD_S1_S2_BASE_H_
#define _FLASH_ATTENTION_SCORE_GRAD_S1_S2_BASE_H_

#include "lib/matmul_intf.h"

constexpr uint32_t SYNC_GLOBAL_WORKSPACE_SIZE_S1_S2 = 800;
constexpr uint32_t ADDR_ALIGN_SIZE_S1_S2 = 512;

using namespace matmul;

template <typename T1, typename T2, const bool IS_BF16 = false> // T1 for data, T2 for vecClc
class FlashAttentionScoreGradS1S2Base {
public:
    __aicore__ inline FlashAttentionScoreGradS1S2Base(){};
    __aicore__ inline void Init(
        __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dx, __gm__ uint8_t *query,
        __gm__ uint8_t *pse_shift, __gm__ uint8_t *drop_mask, __gm__ uint8_t *atten_mask,
        __gm__ uint8_t *forward_res, __gm__ uint8_t *softmax_max, __gm__ uint8_t *softmax_sum,
        __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv,
        __gm__ uint8_t *workspace, const FlashAttentionScoreGradTilingData* __restrict ordTilingData, TPipe *pipe_in);
    __aicore__ inline void Process();

    using aType1 = MatmulType<TPosition::GM, CubeFormat::ND, T1>;
    using bType1 = MatmulType<TPosition::GM, CubeFormat::ND, T1, true>;
    using cType1 = MatmulType<TPosition::VECCALC, CubeFormat::ND, T2>;
    using biasType1 = MatmulType<TPosition::GM, CubeFormat::ND, float>;

    using aType2 = MatmulType<TPosition::VECCALC, CubeFormat::ND, T1, true>;
    using bType2 = MatmulType<TPosition::GM, CubeFormat::ND, T1>;
    using cType2 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using biasType2 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    Matmul<aType1, bType1, cType1, biasType1> mm1;
    Matmul<aType1, bType1, cType1, biasType1> mm2;
    Matmul<aType2, bType2, cType2, biasType2> mm3;
    Matmul<aType2, bType2, cType2, biasType2> mm4;

protected:
    /* define the que */
    TQue<QuePosition::VECIN, 1> vecInQue1;
    TQue<QuePosition::VECIN, 1> vecInQue2;
    TBuf<> vecClc1;
    TBuf<> vecClc2;
    TQue<QuePosition::VECOUT, 1> apiClcQue;

    const FlashAttentionScoreGradTilingData* __restrict ordTilingData;

    TPipe *pipe;

    GlobalTensor<uint8_t> attenMaskU8Gm;
    GlobalTensor<T1> keyGm, valueGm, dxGm, queryGm, attenMaskGm, forwardResGm, pseGm, dqGm, dkGm, dvGm;
    GlobalTensor<float> dqWorkSpace, dkWorkSpace, dvWorkSpace;

    GlobalTensor<float> softmaxMaxGm, softmaxSumGm;
    GlobalTensor<T2> workspaceGm;
    GlobalTensor<uint8_t> dropMaskGm;
    GlobalTensor<int32_t> syncGlobal, syncLastGlobal;

    constexpr static uint32_t block_count = 1;
    constexpr static uint32_t src_stride = 8;
    constexpr static uint32_t dst_stride = 8;
    constexpr static uint32_t block_number = 8;
    constexpr static uint32_t dst_blk_stride = 1;
    constexpr static uint32_t src0_blk_stride = 1;
    constexpr static uint32_t src1_blk_stride = 1;
    constexpr static uint32_t dst_rep_stride = 8;
    constexpr static uint32_t src0_rep_stride = 8;
    constexpr static uint32_t src1_rep_stride = 8;

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
    bool isDrop;

    int32_t layoutSBH;
    int32_t curCubePreloadNum = 0;
    int32_t curCubePreloadNum2 = 0;
    LocalTensor<T2> frontResInnerCache;

    __aicore__ inline void InitGm();
    __aicore__ inline void InitAttenMulsValue();

    __aicore__ inline void FrontCompute(const uint64_t &flashKIdx, const uint64_t &s1OutIdx,
                                        const uint64_t &flashKLoopOffset, const uint64_t &s1OutLoopOffset,
                                        const uint64_t &s1OutLoopMaskOffset, const uint64_t &s1OutIdxStart,
                                        const uint64_t &s1OutIdxEnd);

    __aicore__ inline void ReCompute(const uint64_t &batchIdx, const uint64_t &flashKIdx, const uint64_t &s1OutIdx,
                                     const uint64_t &flashKLoopOffset, const uint64_t &s1OutLoopOffset,
                                     const uint64_t &s1OutLoopMaskOffset, const uint64_t &s1OutLoopReduceOffset,
                                     const uint64_t &s1OutIdxStart, const uint64_t &s1OutIdxEnd);

    __aicore__ inline void ClcDqk(const uint64_t &batchIdx, const uint64_t &flashKLoopOffset,
                                  const uint64_t &s1OutLoopOffset);

    __aicore__ inline void ClcDv(const uint64_t &batchIdx, const uint64_t &flashKLoopOffset,
                                 const uint64_t &s1OutLoopOffset);

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
                                       bool transA = false, bool transB = false, bool isLastTimes = false);

    __aicore__ inline void DkvMatMulClc(const LocalTensor<T1> &tensorA, const GlobalTensor<T1> &tensorB,
                                        const GlobalTensor<float> &tensorC, uint64_t isAtomicAddFlag = 0,
                                        bool transA = false, bool transB = false, bool isLastTimes = false);

    __aicore__ inline void SetFrontClcShape(LocalTensor<T1> &dxInner, LocalTensor<T1> &forwardResInner,
                                            LocalTensor<T2> &sftFrontResInner, LocalTensor<T2> &frontResInner,
                                            LocalTensor<uint8_t> &dpMaskInner);

    __aicore__ inline void SetReClcShape(LocalTensor<T2> &mulResInner, LocalTensor<float> &maxInner,
                                         LocalTensor<float> &sumInner, LocalTensor<T2> &dvDropResInner);

    __aicore__ inline bool isBatchTailCore();

    __aicore__ inline void UpdateTailLoopParams();

    __aicore__ inline void UpdateMatShape();

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

    __aicore__ inline void SyncAllLast();
};

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::Init(
    __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *dx, __gm__ uint8_t *query,
    __gm__ uint8_t *pse_shift, __gm__ uint8_t *drop_mask, __gm__ uint8_t *atten_mask,
    __gm__ uint8_t *forward_res, __gm__ uint8_t *softmax_max, __gm__ uint8_t *softmax_sum,
    __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv,
    __gm__ uint8_t *workspace, const FlashAttentionScoreGradTilingData* __restrict ordTilingData, TPipe *pipe_in)
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
        this->pipe->InitBuffer(this->vecInQue2, 1, this->ordTilingData->splitCoreParams.innerMatResSize / 2);
        this->pipe->InitBuffer(this->vecClc1, this->ordTilingData->splitCoreParams.innerMatResSize);
        this->pipe->InitBuffer(this->vecClc2, this->ordTilingData->splitCoreParams.innerMatResSize);
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
    mm1.SetWorkspace((__gm__ uint8_t*)workspaceGm[buff_offset +
                mBlockIdx * CUBE_PRELOAD_BLOCK_NUM * this->ordTilingData->splitCoreParams.outMatResNum].GetPhyAddr(),
                CUBE_PRELOAD_BLOCK_NUM * this->ordTilingData->splitCoreParams.outMatResSize);

    buff_offset += GetBlockNum() * GetTaskRation() * CUBE_PRELOAD_BLOCK_NUM * this->ordTilingData->splitCoreParams.outMatResNum;
    mm2.SetWorkspace((__gm__ uint8_t*)workspaceGm[buff_offset +
        mBlockIdx * CUBE_PRELOAD_BLOCK_NUM2 * this->ordTilingData->splitCoreParams.outMatResNum].GetPhyAddr(),
        CUBE_PRELOAD_BLOCK_NUM2 * this->ordTilingData->splitCoreParams.outMatResSize);

    buff_offset = ((buff_offset + GetBlockNum() * GetTaskRation() * CUBE_PRELOAD_BLOCK_NUM2 *
        this->ordTilingData->splitCoreParams.outMatResNum) * 4 + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;

    this->dqWorkSpace.SetGlobalBuffer((__gm__ float *)workspace + buff_offset / 4);

    buff_offset = (buff_offset + this->ordTilingData->castTiling.allNumQuery * 4 + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    this->dkWorkSpace.SetGlobalBuffer((__gm__ float *)workspace + buff_offset / 4);

    buff_offset = (buff_offset + this->ordTilingData->castTiling.allNumKv * 4 + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    this->dvWorkSpace.SetGlobalBuffer((__gm__ float *)workspace + buff_offset / 4);

    buff_offset = (buff_offset + this->ordTilingData->castTiling.allNumKv * 4 + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    this->syncLastGlobal.SetGlobalBuffer((__gm__ int32_t *)workspace + buff_offset / 4, 100 * 8);

    InitOutput<int32_t>(this->syncLastGlobal[this->mBlockIdx * 8], 8, 0);

    this->InitAttenMulsValue();
    DropOutBitModeInit();
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::InitGm()
{
    uint32_t perSizeQuery = ordTilingData->castTiling.singleCoreNumQuery;
    uint32_t tailSizeQuery = ordTilingData->castTiling.tailCoreNumQuery;
    uint32_t perSizeKv = ordTilingData->castTiling.singleCoreNumKv;
    uint32_t tailSizeKv = ordTilingData->castTiling.tailCoreNumKv;

    uint32_t initSizeQuery = perSizeQuery;
    uint32_t initSizeKv = perSizeKv;

    if (mBlockIdx == ordTilingData->baseParams.coreNum - 1) {
        initSizeQuery = tailSizeQuery;
        initSizeKv = tailSizeKv;
    }

    uint32_t offsetQuery = mBlockIdx * perSizeQuery;
    uint32_t offsetKv = mBlockIdx * perSizeKv;

    InitOutput<float>(dqWorkSpace[offsetQuery], initSizeQuery, 0);
    InitOutput<float>(dkWorkSpace[offsetKv], initSizeKv, 0);
    InitOutput<float>(dvWorkSpace[offsetKv], initSizeKv, 0);

    LocalTensor<int32_t> workLocal = vecInQue1.AllocTensor<int32_t>();
    SyncAll<false>(syncGlobal, workLocal);
    vecInQue1.FreeTensor(workLocal);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::InitAttenMulsValue()
{
    if (this->ordTilingData->baseParams.attenMaskType == 0) {  // 0 表示输入数据类型与 qkv 一致
        if constexpr (IsSameType<T2, float>::value) {
            mulsValue = -2e38;
        } else {
            mulsValue = -10000.0;
        }
    } else {
        if constexpr (IsSameType<T2, float>::value) {
            uint32_t tmp = 0xFF7FFFFF;      // -INF + 1
            mulsValue = *((float*)&tmp);
        } else {
            uint16_t tmp = 0xFBFF;          // -INF + 1
            mulsValue = *((half*)&tmp);
        }
    }
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::FrontCompute(
    const uint64_t &flashKIdx, const uint64_t &s1OutIdx, const uint64_t &flashKLoopOffset,
    const uint64_t &s1OutLoopOffset, const uint64_t &s1OutLoopMaskOffset, const uint64_t &s1OutIdxStart, const uint64_t &s1OutIdxEnd)
{
    LocalTensor<T1> dxInner = vecInQue1.AllocTensor<T1>();         // [s_inner, H/N]
    LocalTensor<T1> forwardResInner = vecInQue1.AllocTensor<T1>(); // [s_inner, H/N]
    LocalTensor<T2> sftFrontResInner = vecClc2.Get<T2>(sInner * 32 / sizeof(T2));
    LocalTensor<T2> frontResInner = vecClc1.Get<T2>(innerMatResNum);
    LocalTensor<T2> &dpResInner = frontResInner;                         // [s_inner, s_fla]
    LocalTensor<T2> &dpMatmmulResInner = frontResInner;                  // [s_inner, s_fla]
    LocalTensor<uint8_t> dpMaskInner = vecInQue2.AllocTensor<uint8_t>(); // [s_inner, s_fla]

    if constexpr (!IS_BF16) {
        frontResInnerCache = frontResInner;
    }

    dxInner.SetSize(innerMatInputNum);
    forwardResInner.SetSize(innerMatInputNum);
    dpMaskInner.SetSize(innerMatResNum);
    SetFrontClcShape(dxInner, forwardResInner, sftFrontResInner, frontResInner, dpMaskInner);

    CopyInSoftMaxGrad(dxInner, dxGm[s1OutLoopOffset], forwardResInner, forwardResGm[s1OutLoopOffset]);
    if (isDrop) {
        uint64_t maskOffset = s1OutLoopMaskOffset + flashKIdx * this->mSFlaForClcOffset;
        CopyInDrop(dpMaskInner, dropMaskGm[maskOffset / 8]);
    }

    ClcSoftMaxGrad(sftFrontResInner, dxInner, forwardResInner);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);

    mm1.template GetTensorC<false>(dpMatmmulResInner, false, true);
    if (s1OutIdx < s1OutIdxEnd - 1 && (s1OutIdx - s1OutIdxStart) % CUBE_PRELOAD_BLOCK_NUM == CUBE_PRELOAD_BLOCK_NUM - 1) {
        this->mm1.SetTail(this->sOut * curCubePreloadNum, this->sFla, this->ordTilingData->baseParams.D); // M N K
        if (s1OutIdxEnd - s1OutIdx <= CUBE_PRELOAD_BLOCK_NUM) {
            this->mm1.SetTail(this->sOut * (s1OutIdxEnd - 1 - s1OutIdx), this->sFla, this->ordTilingData->baseParams.D); // M N K
        }
        if (this->layoutSBH == 1) {
            mm1.SetTensorA(dxGm[s1OutLoopOffset + sOut * this->ordTilingData->baseParams.B * this->ordTilingData->baseParams.H]);
        } else {
            mm1.SetTensorA(dxGm[s1OutLoopOffset + sOut * this->ordTilingData->baseParams.H]);
        }
        this->mm1.template Iterate<false>();
    }

    if (isDrop) {
        ClcDrop(dpResInner, dpMatmmulResInner, dpMaskInner);
    }

    if constexpr (IS_BF16) {
        ClcSub(frontResInner, dpMatmmulResInner, sftFrontResInner);
        pipe->WriteSpmBuffer(frontResInner, innerMatResSize, 0); // all [s_out, s_fla]
    }
    else {
        ClcSub(frontResInnerCache, dpMatmmulResInner, sftFrontResInner);
    }

    vecInQue1.FreeTensor(dxInner);
    vecInQue1.FreeTensor(forwardResInner);
    vecInQue2.FreeTensor(dpMaskInner);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::ReCompute(
    const uint64_t &batchIdx, const uint64_t &flashKIdx, const uint64_t &s1OutIdx, const uint64_t &flashKLoopOffset,
    const uint64_t &s1OutLoopOffset, const uint64_t &s1OutLoopMaskOffset, const uint64_t &s1OutLoopReduceOffset,
    const uint64_t &s1OutIdxStart, const uint64_t &s1OutIdxEnd)
{
    bool isLastTimes = false;
    if ((batchIdx == (uint64_t)this->splitedBatchRange - 1) &&
        (flashKIdx == (uint64_t)this->flashKLoopRange - 1) &&
        (s1OutIdx == (uint64_t)this->s1OutRange - 1))
    {
        isLastTimes = true;
    }

    LocalTensor<T2> frontResInner = vecClc1.Get<T2>(innerMatResNum);
    LocalTensor<T2> &mulResInner = frontResInner;
    LocalTensor<T2> dvDropResInner = vecClc2.Get<T2>(innerMatResNum); // [s_inner, s_fla]

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

    mm2.template GetTensorC<false>(reMatmulResInner, false, true);
    mm2.End();

    if (clsPse) {
        ClcPse(reMatmulResInner);
        clcAttenMask = CopyInAttenMask(flashKIdx);
    }
    Muls(reMatmulResInner, reMatmulResInner, (T2)(this->ordTilingData->baseParams.scaleValue), innerMatResNum);
    if (clcAttenMask) {
        ClcAttenMask(reMatmulResInner);
    }

    ClcSoftMax(softmaxResInner, attenMaskResInner, maxInner, sumInner);

    ClcMul(frontResInner, mulResInner, softmaxResInner);

    LocalTensor<T1> castedMulResInner;
    if constexpr (IS_BF16) {
        pipe_barrier(PIPE_V);
        castedMulResInner = vecClc1.Get<T1>(innerMatResNum);
        castedMulResInner.SetShapeInfo(ShapeInfo(2, innerMatOutShape, DataFormat::ND));
        Cast(castedMulResInner, mulResInner, RoundMode::CAST_ROUND, innerMatResNum);
        pipe_barrier(PIPE_V);
    }

    if (isDrop) {
        uint64_t maskOffset = s1OutLoopMaskOffset + flashKIdx * this->mSFlaForClcOffset;
        ClcReDrop(dvDropResInner, softmaxResInner, dropMaskGm[maskOffset / 8]);
    }
    LocalTensor<T1> castedDvDropResInner;
    if constexpr (IS_BF16) {
        castedDvDropResInner = vecClc2.Get<T1>(innerMatResNum);
        castedDvDropResInner.SetShapeInfo(ShapeInfo(2, innerMatOutShape, DataFormat::ND));
        Cast(castedDvDropResInner, dvDropResInner, RoundMode::CAST_ROUND, innerMatResNum);
        pipe_barrier(PIPE_V);
    }

    this->mm3.SetTail(sOut, -1, sFla);
    if constexpr (IS_BF16) {
        this->MyMatMulClc(castedMulResInner, this->keyGm[flashKLoopOffset], this->dqWorkSpace[s1OutLoopOffset], true, false,
                            false, isLastTimes);
    } else {
        this->MyMatMulClc(mulResInner, this->keyGm[flashKLoopOffset], this->dqWorkSpace[s1OutLoopOffset], true, false,
                            false, isLastTimes);
    }
    this->mm4.SetTail(sFla, -1, sOut);
    if constexpr (IS_BF16) {
        this->DkvMatMulClc(castedMulResInner, this->queryGm[s1OutLoopOffset], this->dkWorkSpace[flashKLoopOffset], true, true,
                            false, isLastTimes);
    } else {
        this->DkvMatMulClc(mulResInner, this->queryGm[s1OutLoopOffset], this->dkWorkSpace[flashKLoopOffset], true, true,
                            false, isLastTimes);
    }
    this->mm4.SetTail(sFla, -1, sOut);
    if constexpr (IS_BF16) {
        this->DkvMatMulClc(castedDvDropResInner, this->dxGm[s1OutLoopOffset], this->dvWorkSpace[flashKLoopOffset], true, true,
                            false, isLastTimes);
    } else {
        this->DkvMatMulClc(dvDropResInner, this->dxGm[s1OutLoopOffset], this->dvWorkSpace[flashKLoopOffset], true, true,
                            false, isLastTimes);
    }

    vecInQue1.FreeTensor(maxInner);
    vecInQue1.FreeTensor(sumInner);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::ClcDqk(const uint64_t &batchIdx,
                                                                                const uint64_t &flashKLoopOffset,
                                                                                const uint64_t &s1OutLoopOffset)
{
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::ClcDv(const uint64_t &batchIdx,
                                                                               const uint64_t &flashKLoopOffset,
                                                                               const uint64_t &s1OutLoopOffset)
{
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::ClcFrontOffset(
    uint64_t &dxOffset, uint64_t &maskOffset, uint64_t &spmOffset, const uint64_t &inIdx, const uint64_t &flashKIdx,
    const uint64_t &s1OutLoopOffset, const uint64_t &s1OutLoopMaskOffset)
{
    dxOffset = s1OutLoopOffset + inIdx * innerMatInputNum;
    maskOffset = s1OutLoopMaskOffset + flashKIdx * this->mSFlaForClcOffset + inIdx * innerMaskNum;
    spmOffset = inIdx * innerMatResNum;
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::CopyInSoftMaxGrad(
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
__aicore__ inline void FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::ClcSoftMaxGrad(
        LocalTensor<T2> &sftFrontResInner,
        LocalTensor<T1> &dxInner,
        LocalTensor<T1> &forwardResInner)
{
    vecInQue1.DeQue<T1>();
    bool isBasicBlock = (this->sInner % 8 ==0) && (this->ordTilingData->baseParams.D % 64 ==0);
    LocalTensor<uint8_t> apiClcTensor = apiClcQue.AllocTensor<uint8_t>();
    apiClcTensor.SetSize(this->ordTilingData->baseParams.apiClcQueueSize);
    if constexpr (IS_BF16) {
        pipe_barrier(PIPE_V);
        LocalTensor<T2> castedFrontResInner = vecClc2.Get<T2>(innerMatInputNum);
        LocalTensor<T2> castedDxInner = vecClc1.Get<T2>(innerMatInputNum);
        castedFrontResInner.SetShapeInfo(ShapeInfo(2, innerMatInShape, DataFormat::ND));
        castedDxInner.SetShapeInfo(ShapeInfo(2, innerMatInShape, DataFormat::ND));
        Cast(castedFrontResInner, forwardResInner, RoundMode::CAST_NONE, innerMatInputNum);
        Cast(castedDxInner, dxInner, RoundMode::CAST_NONE, innerMatInputNum);
        pipe_barrier(PIPE_V);
        if (isBasicBlock) {
            SoftmaxGradFront<T2, true>(sftFrontResInner, castedFrontResInner, castedDxInner,
                                       apiClcTensor,
                                       this->ordTilingData->softmaxGradTilingData);
        } else {
            SoftmaxGradFront<T2, false>(sftFrontResInner, castedFrontResInner, castedDxInner,
                                        apiClcTensor,
                                        this->ordTilingData->softmaxGradTilingData);
        }
    } else {
        if (isBasicBlock) {
            SoftmaxGradFront<T1, true>(sftFrontResInner, dxInner, forwardResInner,
                                       apiClcTensor,
                                       this->ordTilingData->softmaxGradTilingData);
        } else {
            SoftmaxGradFront<T1, false>(sftFrontResInner, dxInner, forwardResInner,
                                        apiClcTensor,
                                        this->ordTilingData->softmaxGradTilingData);
        }
    }
    apiClcQue.FreeTensor(apiClcTensor);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::CopyInDrop(LocalTensor<uint8_t> &dst,
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
__aicore__ inline void FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::ClcDrop(LocalTensor<T2> &dst,
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
    if (this->ordTilingData->baseParams.attenMaskType == 0) {
        DropOut<T2, true>(dst, src1, src2, apiClcTensor, this->ordTilingData->baseParams.keepProb, info);
    } else {
        DropOut(dst, src1, src2, apiClcTensor, this->ordTilingData->baseParams.keepProb, info);
    }
    apiClcQue.FreeTensor(apiClcTensor);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::ClcSub(LocalTensor<T2> &frontResInner,
                                                                                LocalTensor<T2> &dpResInner,
                                                                                LocalTensor<T2> &sftFrontResInner)
{
    // [m,n] - [m,16] -> [m,n] 按n轴的block数repeat，每个指令repeat算[m,16] - [m,16], subRange循环处理 超过mask情况
    for (int32_t subIdx = 0; subIdx < subRange; subIdx++) {
        uint64_t srcOffset = subIdx * subMask;  // subIdx * 16 * 8
        uint64_t dstOffset = subIdx * sFla * 8; // subIdx * sFla * 8
        // dst_blk_stride, src0_blk_stride, src1_blk_stride, dst_rep_stride, src0_rep_stride, src1_rep_stride
        Sub(frontResInner[dstOffset], dpResInner[dstOffset], sftFrontResInner[srcOffset], subMask, sFlaBlockNum,
            {(uint8_t)(sFlaBlockNum), (uint8_t)(sFlaBlockNum), 1, 1, 1, 0});
    }
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::ClcReComputeOffset(
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
__aicore__ inline void FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::CopyInSoftMax(
    LocalTensor<float> &maxInner, const GlobalTensor<float> &softmaxMaxGmIn, LocalTensor<float> &sumInner,
    const GlobalTensor<float> &softmaxSumGmIn)
{
    DataCopy(maxInner, softmaxMaxGmIn, innerReduceNum);
    DataCopy(sumInner, softmaxSumGmIn, innerReduceNum);
    vecInQue1.EnQue(sumInner);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::ClcSoftMax(LocalTensor<T2> &softmaxResInner,
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
__aicore__ inline void FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::ClcMul(LocalTensor<T2> &frontResInner,
                                                                                LocalTensor<T2> &mulResInner,
                                                                                LocalTensor<T2> &softmaxResInner)
{
    if constexpr (IS_BF16) {
        pipe->ReadSpmBuffer(frontResInner, 0);
        Mul(mulResInner, softmaxResInner, frontResInner, innerMatResNum);
    }
    else {
        Mul(mulResInner, softmaxResInner, frontResInnerCache, innerMatResNum);
    }
    if constexpr (IsSameType<T1, float>::value) {
        Muls(mulResInner, mulResInner, (T2)(this->ordTilingData->baseParams.scaleValue), innerMatResNum);
    }
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::ClcReDrop(
    LocalTensor<T2> &dvDropResInner, LocalTensor<T2> &softmaxResInner, const GlobalTensor<uint8_t> &dropMaskGmIn)
{
    LocalTensor<uint8_t> dvMaskInner = vecInQue2.AllocTensor<uint8_t>(); // [s_inner, s_fla]
    dvMaskInner.SetSize(innerMatResNum);
    struct DataCopyParams intriParams;
    intriParams.blockCount = (uint16_t)sInner;
    intriParams.blockLen = (uint16_t)((sFla / 8 + 31) / 32);
    intriParams.dstStride = 0;
    if ((this->ordTilingData->baseParams.Skv / 8 - intriParams.blockLen * 32) % 32 == 0) {
        intriParams.srcStride = (uint16_t)(this->ordTilingData->baseParams.Skv / 8 - intriParams.blockLen * 32) / 32;
        DataCopy(dvMaskInner, dropMaskGmIn, intriParams);
    } else {
        intriParams.blockLen = (uint16_t)sFla / 8;
        intriParams.srcStride = (uint16_t)(this->ordTilingData->baseParams.Skv / 8 - intriParams.blockLen);
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
    if (this->ordTilingData->baseParams.attenMaskType == 0) {
        DropOut<T2, true>(dvDropResInner, softmaxResInner, dvMaskInner, apiClcTensor,
                          this->ordTilingData->baseParams.keepProb, info);
    } else {
        DropOut(dvDropResInner, softmaxResInner, dvMaskInner, apiClcTensor,
                this->ordTilingData->baseParams.keepProb, info);
    }
    apiClcQue.FreeTensor(apiClcTensor);
    vecInQue2.FreeTensor(dvMaskInner);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::MyMatMulClc(
        const LocalTensor<T1> &tensorA,
        const GlobalTensor<T1> &tensorB,
        const GlobalTensor<float> &tensorC,
        uint64_t isAtomicAddFlag, bool transA,
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
__aicore__ inline void FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::DkvMatMulClc(
        const LocalTensor<T1> &tensorA,
        const GlobalTensor<T1> &tensorB,
        const GlobalTensor<float> &tensorC,
        uint64_t isAtomicAddFlag, bool transA,
        bool transB, bool isLastTimes)
{
    mm4.SetTensorA(tensorA, transA);
    mm4.SetTensorB(tensorB, transB);
    if (isLastTimes) {
        mm4.template IterateAll<true>(tensorC, isAtomicAddFlag);
    } else {
        mm4.template IterateAll<false>(tensorC, isAtomicAddFlag);
    }
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::SetFrontClcShape(
        LocalTensor<T1> &dxInner,
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
__aicore__ inline void FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::SetReClcShape(
        LocalTensor<T2> &mulResInner,
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
__aicore__ inline void FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::UpdateTailLoopParams()
{
    this->mLoopParams.innerMatResNum = this->mLoopParams.innerMatResTailNum;
    this->mLoopParams.outMatResNum = this->mLoopParams.outMatResTailNum;
    this->mLoopParams.sFlaMatInputNum = this->mLoopParams.sFlaMatInputTailNum;
    this->mLoopParams.innerMatResSize = this->mLoopParams.innerMatResTailSize;
    this->mLoopParams.outMatResSize = this->mLoopParams.outMatResTailSize;
    this->mLoopParams.sFlaMatInputSize = this->mLoopParams.sFlaMatInputTailSize;
    this->mLoopParams.sFlaBlock = this->mLoopParams.sFlaTailBlock;
    this->mLoopParams.sSubSFlaStride = this->mLoopParams.sSubSFlaTailStride;
    this->mLoopParams.sFlaBlockNum = this->mLoopParams.sFlaBlockTailNum;
    this->mLoopParams.innerMatResBlock = this->mLoopParams.innerMatResTailBlock;
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::UpdateMatShape()
{
    this->innerMatInShape[0] = this->mLoopParams.sInner;
    this->innerMatInShape[1] = this->ordTilingData->baseParams.D;
    this->innerMatOutShape[0] = this->mLoopParams.sInner;
    this->innerMatOutShape[1] = this->mSingleCoreParams.sFla;
    this->innerReduceShape[0] = this->mLoopParams.sInner;
    this->innerReduceShape[1] = 32 / sizeof(float);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void
FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::DetermineLoopParams(
    const SplitCoreParams &coreParams, uint64_t flashKIdx, uint64_t s1OutIdx)
{
    bool isFlaTail = (flashKIdx == (uint64_t)this->flashKLoopRange - 1);
    bool isS1OutTail = (s1OutIdx == (uint64_t)this->s1OutRange - 1);

    if (!isS1OutTail) {
        // sOut parameters
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
        if (isFlaTail) {
            // sFla parameters
            this->sFla = coreParams.sFlaTail;
            this->innerMatResSize = coreParams.innerMatResTailSize;
            this->outMatResSize = coreParams.outMatResTailSize;
            this->sFlaBlockNum = coreParams.sFlaBlockTailNum;
            this->sFlaBlock = coreParams.sFlaTailBlock;
            this->sSubSFlaStride = coreParams.sSubSFlaTailStride;
            this->innerMatResNum = coreParams.innerMatResTailNum;
        }
    } else {
        this->sOut = coreParams.sOutTail;
        this->sFla = coreParams.sFla;
        this->outMatResSize = coreParams.tailOutMatResSize;
        this->sInnerRange = coreParams.tailSInnerRange;
        this->sInner = coreParams.tailSInner;
        this->innerReduceNum = coreParams.tailInnerReduceNum;
        this->innerMatResNum = coreParams.tailInnerMatResNum;
        this->innerMatInputNum = coreParams.tailInnerMatInputNum;
        this->innerMaskNum = coreParams.tailInnerMaskNum;
        this->subMask = coreParams.tailSubMask;
        this->subRange = coreParams.tailSubRange;
        this->innerMatResSize = coreParams.tailInnerMatResSize;
        this->sFlaBlockNum = coreParams.tailSFlaBlockNum;
        this->sFlaBlock = coreParams.tailSFlaBlock;
        this->sSubSFlaStride = coreParams.tailSSubSFlaStride;
        if (isFlaTail) {
            this->sFla = coreParams.sFlaTail;
            this->innerMatResSize = coreParams.tailInnerMatResTailSize;
            this->outMatResSize = coreParams.tailOutMatResTailSize;
            this->sFlaBlockNum = coreParams.tailSFlaBlockTailNum;
            this->sFlaBlock = coreParams.tailSFlaTailBlock;
            this->sSubSFlaStride = coreParams.tailSSubSFlaTailStride;
            this->innerMatResNum = coreParams.tailInnerMatResTailNum;
        }
    }
    this->innerMatInShape[0] = this->sInner;
    this->innerMatInShape[1] = this->ordTilingData->baseParams.D;
    this->innerMatOutShape[0] = this->sInner;
    this->innerMatOutShape[1] = this->sFla;
    this->innerReduceShape[0] = this->sInner;
    this->innerReduceShape[1] = 32 / sizeof(float);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline bool FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::CopyInPse(
    const uint64_t &s1OutLoopMaskOffset, const uint64_t &flashKIdx)
{
    if (pseGm.GetPhyAddr() == nullptr) {
        return false;
    }

    DataCopyParams copyParams;
    copyParams.blockCount = (uint16_t)sOut;
    copyParams.blockLen = (uint16_t)sFlaBlock;
    copyParams.srcStride = (uint16_t)sSubSFlaStride;
    copyParams.dstStride = 0;
    if (this->ordTilingData->baseParams.pseShapeType == 1) {  // BN1S
        copyParams.blockCount = 1;
        // BN1S 初始 offset 已在外层模板处设置
    } else {
        pseOffset = s1OutLoopMaskOffset;
    }
    pseOffset += flashKIdx * this->mSFlaForClcOffset;

    LocalTensor<T1> pseUb = vecInQue2.AllocTensor<T1>();
    pseUb.SetSize(copyParams.blockCount * sFla);
    DataCopy(pseUb, pseGm[pseOffset], copyParams);
    vecInQue2.EnQue(pseUb);
    return true;
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::ClcPse(LocalTensor<T2> &mmResUb)
{
    LocalTensor<T1> pseUb = vecInQue2.DeQue<T1>();
    uint32_t eleNum = pseUb.GetSize();

    if constexpr (IS_BF16) {
        LocalTensor<T2> castedPseUb = vecClc1.Get<T2>(eleNum);
        castedPseUb.SetSize(eleNum);
        uint32_t shapeArray[] = {(uint32_t)sOut, (uint32_t)sFla};
        if (this->ordTilingData->baseParams.pseShapeType == 1) {  // BN1S
            shapeArray[0] = 1;
        }
        castedPseUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        Cast(castedPseUb, pseUb, RoundMode::CAST_NONE, eleNum);
        pipe_barrier(PIPE_V);
        ClcPseBroadcast(mmResUb, castedPseUb, eleNum);
    } else {
        ClcPseBroadcast(mmResUb, pseUb, eleNum);
    }
    vecInQue2.FreeTensor(pseUb);
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::ClcPseBroadcast(
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
__aicore__ inline void FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::ClcPseBroadcastInner(
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
__aicore__ inline bool FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::CopyInAttenMask(const uint64_t &flashKIdx)
{
    if (attenMaskGm.GetPhyAddr() == nullptr) {
        return false;
    }
    if (this->ordTilingData->baseParams.attenMaskType == 0) {  // 0 表示输入数据类型与 qkv 一致
        LocalTensor<T1> attenMaskUb = vecInQue2.AllocTensor<T1>();  // [s_inner, s_fla]
        attenMaskUb.SetSize(innerMatResNum);
        attenMaskOffset += flashKIdx * this->mSFlaForClcOffset;
        DataCopy(attenMaskUb, attenMaskGm[attenMaskOffset],
                 {(uint16_t)sInner, (uint16_t)sFlaBlock, (uint16_t)sSubSFlaStride, 0});
        vecInQue2.EnQue(attenMaskUb);
    } else {
        LocalTensor<uint8_t> attenMaskUb = vecInQue2.AllocTensor<uint8_t>();
        attenMaskOffset += flashKIdx * this->mSFlaForClcOffset;
        DataCopyParams intriParams;
        intriParams.blockCount = sInner;
        intriParams.blockLen = (sFla * sizeof(uint8_t) + 31) / 32;
        intriParams.dstStride = 0;

        if ((ordTilingData->baseParams.Skv - intriParams.blockLen * 32) % 32 == 0) {
            intriParams.srcStride = (ordTilingData->baseParams.Skv - sFla) / (32 / sizeof(uint8_t));
            DataCopy(attenMaskUb, attenMaskU8Gm[attenMaskOffset], intriParams);
        } else {
            intriParams.blockLen = sFla * sizeof(uint8_t);
            intriParams.srcStride = (ordTilingData->baseParams.Skv * sizeof(uint8_t) - intriParams.blockLen);
            DataCopyPadParams intriParams1;
            DataCopyPad(attenMaskUb, attenMaskU8Gm[attenMaskOffset], intriParams, intriParams1);
        }
        vecInQue2.EnQue(attenMaskUb);
    }
    return true;
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::ClcAttenMask(LocalTensor<T2> &mmResUb)
{
    if (this->ordTilingData->baseParams.attenMaskType == 0) {  // 0 表示输入数据类型与 qkv 一致
        LocalTensor<T1> attenMaskUb = vecInQue2.DeQue<T1>();
        if constexpr (IS_BF16) {
            LocalTensor<T2> castedAttenMaskUb = vecClc1.Get<T2>(innerMatResNum);
            Cast(castedAttenMaskUb, attenMaskUb, RoundMode::CAST_NONE, innerMatResNum);
            pipe_barrier(PIPE_V);
            Muls(castedAttenMaskUb, castedAttenMaskUb, mulsValue, innerMatResNum);
            pipe_barrier(PIPE_V);
            Add(mmResUb, castedAttenMaskUb, mmResUb, innerMatResNum);
        } else {
            Muls(attenMaskUb, attenMaskUb, mulsValue, innerMatResNum);
            pipe_barrier(PIPE_V);
            Add(mmResUb, attenMaskUb, mmResUb, innerMatResNum);
        }
        vecInQue2.FreeTensor(attenMaskUb);
    } else {
        LocalTensor<uint8_t> attenMaskUb = vecInQue2.DeQue<uint8_t>();
        LocalTensor<uint8_t> ubWorkspace = apiClcQue.AllocTensor<uint8_t>();

        SelectWithBytesMaskShapeInfo shapeInfo;
        shapeInfo.firstAxis = sInner;
        shapeInfo.srcLastAxis = sFla;
        shapeInfo.maskLastAxis = (sFla * sizeof(uint8_t) + 31) / 32 * 32 / sizeof(uint8_t);
        SelectWithBytesMask(mmResUb, mmResUb, mulsValue, attenMaskUb, ubWorkspace, shapeInfo);

        apiClcQue.FreeTensor(ubWorkspace);
        vecInQue2.FreeTensor(attenMaskUb);
    }
}

template <typename T1, typename T2, const bool IS_BF16>
__aicore__ inline void FlashAttentionScoreGradS1S2Base<T1, T2, IS_BF16>::SyncAllLast() {
    LocalTensor<int32_t> workLocal = vecInQue1.AllocTensor<int32_t>();
    SyncAll<false>(this->syncLastGlobal, workLocal);
    vecInQue1.FreeTensor(workLocal);
}
#endif
