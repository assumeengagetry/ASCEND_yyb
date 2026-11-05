/*
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

#ifndef FLASH_ATTENTION_SCORE_GRAD_H
#define FLASH_ATTENTION_SCORE_GRAD_H

#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
using namespace AscendC;
using namespace matmul;

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t DROP_DATA_SIZE = 8;

template <typename T, typename U, LayoutMode LAYOUT>
class NewFlashAttentionScoreGradBase {
public:
    __aicore__ inline NewFlashAttentionScoreGradBase() {};

    using a12Type = MatmulType<TPosition::GM, CubeFormat::ND, T, false, LAYOUT>;
    using b12Type = MatmulType<TPosition::GM, CubeFormat::ND, T, true, LAYOUT>;
    using c12Type = MatmulType<TPosition::VECCALC, CubeFormat::ND, U, false, LayoutMode::BNGS1S2>;
    using bias12Type = MatmulType<TPosition::GM, CubeFormat::ND, U, false, LayoutMode::BNGS1S2>;

    using a3Type = MatmulType<TPosition::VECCALC, CubeFormat::ND, T, false, LayoutMode::BNGS1S2>;
    using a4Type = MatmulType<TPosition::VECCALC, CubeFormat::ND, T, true, LayoutMode::BNGS1S2>;
    using b34Type = MatmulType<TPosition::GM, CubeFormat::ND, T, false, LAYOUT>;
    using c34Type = MatmulType<TPosition::GM, CubeFormat::ND, T, false, LAYOUT>;
    using bias34Type = MatmulType<TPosition::GM, CubeFormat::ND, T, false, LAYOUT>;

    using a3Type1 = MatmulType<TPosition::VECCALC, CubeFormat::ND, T, false>;
    using b34Type1 = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using c34Type1 = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using biasType = MatmulType<TPosition::GM, CubeFormat::ND, T>;

    matmul::Matmul<a12Type, b12Type, c12Type, bias12Type> m_bmm1;
    matmul::Matmul<a12Type, b12Type, c12Type, bias12Type> m_bmm2;
    matmul::Matmul<a3Type, b34Type, c34Type, bias34Type> m_bmm3;
    matmul::Matmul<a4Type, b34Type, c34Type, bias34Type> m_bmm4;

protected:
    __aicore__ inline void RunBmm1(uint64_t batchOffsetQ, uint64_t batchOffsetKV); // get from tiling
    __aicore__ inline void CopyInSoftmaxGrad(uint64_t &batchOffsetQ, uint32_t dSizeIdx);
    __aicore__ inline void RightPad(const LocalTensor<U> global, uint32_t dim1, uint32_t dim2,
        uint32_t padNum, U value);
    __aicore__ inline void CalcSoftmaxFront(LocalTensor<U> &softmaxTensorAll, uint32_t dSizeIdx);
    __aicore__ inline void CalcSoftmaxFrontFor1D(LocalTensor<U> &softmaxTensorAll);
    __aicore__ inline void CopyInDrop(uint64_t &batchOffsetDrop);
    __aicore__ inline void CalcDrop(LocalTensor<U> &src1Tensor);
    __aicore__ inline void SubClc(LocalTensor<U> &softmaxGradTensor);

    __aicore__ inline void CalcDP(uint64_t &batchOffsetQ, uint64_t &batchOffsetKV, uint64_t &batchOffsetDrop);

    __aicore__ inline void RunBmm2(uint64_t batchOffsetQ, uint64_t batchOffsetKV, LocalTensor<U> &calcDSTensor); // get from tiling

    __aicore__ inline bool CropInPse(uint64_t &batchOffsetPse);
    __aicore__ inline void AddPseClc(uint64_t &batchOffsetPse, uint64_t &batchOffsetKV, LocalTensor<U> &mat2addTensor);
    __aicore__ inline void PseBroadcastAdd(LocalTensor<U> &mat2addTensor, LocalTensor<U> pseTensor);
    __aicore__ inline void CopyInMask(uint64_t &batchOffsetMask);
    __aicore__ inline void CalcMask(LocalTensor<U> &pse2maskTensor);
    __aicore__ inline void CopyInSoftmax(uint64_t &batchOffsetSoftmax);
    __aicore__ inline void CalcSoftmax(LocalTensor<U> &mask2softmaxTensor);
    __aicore__ inline void CalcMul(LocalTensor<U> &softmaxTensor);
    __aicore__ inline void CalcDS(uint64_t &batchOffsetQ, uint64_t &batchOffsetKV, uint64_t &batchOffsetMask,
        uint64_t &batchOffsetSoftmax, uint64_t &batchOffsetPse);

    __aicore__ inline void CalcDQDK(uint64_t &batchOffsetQ, uint64_t &batchOffsetKV);
    __aicore__ inline void RunBmm31(uint64_t batchOffsetQ, uint64_t batchOffsetKV); // get from tiling
    __aicore__ inline void RunBmm32(uint64_t batchOffsetQ, uint64_t batchOffsetKV); // get from tiling
    __aicore__ inline void CalcDV(uint64_t &batchOffsetQ, uint64_t &batchOffsetKV, uint64_t &batchOffsetDrop);
    __aicore__ inline void RunBmm4(uint64_t batchOffsetQ, uint64_t batchOffsetKV);

protected:
    TQue<QuePosition::VECIN, 1> m_vecInQue1;
    TQue<QuePosition::VECIN, 1> m_vecInQue2;
    TQue<QuePosition::VECIN, 1> m_vecInQue3;
    TQue<QuePosition::VECIN, 1> m_vecInQue4;

    TQue<QuePosition::VECOUT, 1> m_vecOutQue;
    TQue<QuePosition::VECOUT, 1> m_apiClcQue;

    const FlashAttentionScoreGradUbngs1s2BbTilingData* __restrict m_ordTilingData;
    TPipe *m_pipe;

    GlobalTensor<T> m_queryGm;
    GlobalTensor<T> m_keyGm;
    GlobalTensor<T> m_valueGm;
    GlobalTensor<T> m_outGm;
    GlobalTensor<T> m_doutGm;
    GlobalTensor<T> m_attenMaskGm;
    GlobalTensor<T> m_pseGm;
    GlobalTensor<uint8_t> m_dropMaskGm;
    GlobalTensor<float> m_softmaxMaxGm;
    GlobalTensor<float> m_softmaxSumGm;

    GlobalTensor<T> m_dqGm;
    GlobalTensor<T> m_dkGm;
    GlobalTensor<T> m_dvGm;
    GlobalTensor<T> m_dpseGm; // temp

    GlobalTensor<U> m_workspaceGm;

    GlobalTensor<int32_t> m_syncGlobal;

    uint32_t m_blockIdx;

    uint32_t m_b;
    uint32_t m_nQ;
    uint32_t m_nKV;

    uint32_t m_sQ;
    uint32_t m_g;
    uint32_t m_gSQ;
    uint32_t m_sKV;
    uint32_t m_sKVAligned;

    uint32_t m_d;
    uint32_t m_dInner;
    uint32_t m_innerSize;
    uint32_t m_allInnerSize;
};

template<typename T, typename U, LayoutMode LAYOUT>
__aicore__ inline void NewFlashAttentionScoreGradBase<T, U, LAYOUT>::RunBmm1(uint64_t batchOffsetQ,
    uint64_t batchOffsetKV) // get from tiling
{
    LocalTensor<U> vDoutTensor = m_vecOutQue.AllocTensor<U>();
    vDoutTensor.SetSize(m_allInnerSize);
    uint32_t vDoutShape[3] { m_nKV, m_gSQ, m_sKV };
    vDoutTensor.SetShapeInfo(ShapeInfo(3, vDoutShape, DataFormat::ND));

    m_bmm1.SetTail(m_sQ, m_sKV, m_d);
    m_bmm1.SetTensorA(m_doutGm[batchOffsetQ]);
    m_bmm1.SetTensorB(m_valueGm[batchOffsetKV], true);
    m_bmm1.template IterateBatch<true>(vDoutTensor, m_nKV, m_nKV, false);

    m_vecOutQue.EnQue(vDoutTensor);
    m_bmm1.End();
}

template<typename T, typename U, LayoutMode LAYOUT>
__aicore__ inline void NewFlashAttentionScoreGradBase<T, U, LAYOUT>::RightPad(const LocalTensor<U> global,
    uint32_t dim1, uint32_t dim2, uint32_t padNum, U value)
{
    for(uint32_t i =0; i<dim1; i++ ){
        for(uint32_t j =dim2-padNum; j<dim2; j++ ){
            global.SetValue(i*dim2+j, value);
        }
    }
}

template<typename T, typename U, LayoutMode LAYOUT>
__aicore__ inline void NewFlashAttentionScoreGradBase<T, U, LAYOUT>::CopyInSoftmaxGrad(
  uint64_t &batchOffsetQ, uint32_t dSizeIdx)
{
    uint32_t clcDInner = m_ordTilingData->splitDInfo.clcDInner;

    uint32_t dSize = m_ordTilingData->splitDInfo.dSize;
    uint32_t dInnerTail = m_ordTilingData->splitDInfo.dInnerTail;
    uint32_t dInnerTailAlign = m_ordTilingData->splitDInfo.dInnerTailAlign;

    uint32_t dInner = (dSizeIdx == dSize - 1) ? dInnerTail : clcDInner;
    uint32_t dInnerAlign = (dSizeIdx == dSize - 1) ? dInnerTailAlign : clcDInner;

    LocalTensor<T> doutTensor = m_vecInQue1.AllocTensor<T>();
    uint32_t outShape[] = { m_nKV, m_gSQ, dInnerAlign };
    doutTensor.SetSize(m_nKV * m_gSQ * dInnerAlign);
    doutTensor.SetShapeInfo(ShapeInfo(3, outShape, DataFormat::ND));

    LocalTensor<T> outTensor = m_vecInQue2.AllocTensor<T>();
    outTensor.SetSize(m_nKV * m_gSQ * dInnerAlign);
    outTensor.SetShapeInfo(ShapeInfo(3, outShape, DataFormat::ND));

    DataCopyParams intriParams;
    uint16_t dOffset = (dSizeIdx == dSize - 1) ? dSizeIdx * clcDInner : m_d - dInner;

    if constexpr (LAYOUT == LayoutMode::BSNGD) {
        intriParams.blockCount = (uint16_t)(m_sQ);
        intriParams.blockLen = (uint16_t)(dInner * sizeof(T)); // TODO: add to tiling
        intriParams.srcStride = ((m_nQ -1) * m_d + dOffset) * sizeof(T);
        intriParams.dstStride = 0;

        for (uint32_t nIdx = 0; nIdx < m_nQ; ++nIdx) {
            DataCopyPad(doutTensor[nIdx * m_sQ * dInnerAlign],
                        m_doutGm[batchOffsetQ + nIdx * m_d + dSizeIdx * clcDInner], intriParams,
                        {true, 0, static_cast<uint8_t>((dInnerAlign - dInner) * sizeof(T)), 0});
            DataCopyPad(outTensor[nIdx * m_sQ * dInnerAlign],
                        m_outGm[batchOffsetQ + nIdx * m_d + dSizeIdx * clcDInner], intriParams,
                        {true, 0, static_cast<uint8_t>((dInnerAlign - dInner) * sizeof(T)), 0});
        }
    }

    if constexpr (LAYOUT == LayoutMode::SBNGD) {
        intriParams.blockCount = (uint16_t)(m_sQ);
        intriParams.blockLen = (uint16_t)(dInner * sizeof(T)); // TODO: add to tiling
        intriParams.srcStride = ((m_b * m_nQ - 1) * m_d + dOffset) * sizeof(T);
        intriParams.dstStride = 0;

        for (uint32_t nIdx = 0; nIdx < m_nQ; ++nIdx) {
            DataCopyPad(doutTensor[nIdx * m_sQ * dInnerAlign],
                        m_doutGm[batchOffsetQ + nIdx * m_d + dSizeIdx * clcDInner], intriParams,
                        {true, 0, static_cast<uint8_t>((dInnerAlign - dInner) * sizeof(T)), 0});
            DataCopyPad(outTensor[nIdx * m_sQ * dInnerAlign],
                        m_outGm[batchOffsetQ + nIdx * m_d + dSizeIdx * clcDInner], intriParams,
                        {true, 0, static_cast<uint8_t>((dInnerAlign - dInner) * sizeof(T)), 0});
        }
    }

    if constexpr (LAYOUT == LayoutMode::BNGS1S2) {
        intriParams.blockCount = (uint16_t)(m_sQ);
        intriParams.blockLen = (uint16_t)(dInner * sizeof(T)); // TODO: add to tiling
        intriParams.srcStride = dOffset * sizeof(T);
        intriParams.dstStride = 0;

        for (uint32_t nIdx = 0; nIdx < m_nQ; ++nIdx) {
            DataCopyPad(doutTensor[nIdx * m_sQ * dInnerAlign],
                        m_doutGm[batchOffsetQ + m_sQ * nIdx * m_d + dSizeIdx * clcDInner], intriParams,
                        {true, 0, static_cast<uint8_t>((dInnerAlign - dInner) * sizeof(T)), 0});
            DataCopyPad(outTensor[nIdx * m_sQ * dInnerAlign],
                        m_outGm[batchOffsetQ + m_sQ * nIdx * m_d + dSizeIdx * clcDInner], intriParams,
                        {true, 0, static_cast<uint8_t>((dInnerAlign - dInner) * sizeof(T)), 0});
        }
    }

    m_vecInQue1.EnQue<T>(doutTensor);
    m_vecInQue2.EnQue<T>(outTensor);
}

template<typename T, typename U, LayoutMode LAYOUT>
__aicore__ inline void NewFlashAttentionScoreGradBase<T, U, LAYOUT>::CalcSoftmaxFront(
    LocalTensor<U> &softmaxTensorAll, uint32_t dSizeIdx)
{
    uint32_t clcDInner = m_ordTilingData->splitDInfo.clcDInner;
    uint32_t dSize = m_ordTilingData->splitDInfo.dSize;
    uint32_t dInnerTail = m_ordTilingData->splitDInfo.dInnerTail;
    uint32_t dInnerTailAlign = m_ordTilingData->splitDInfo.dInnerTailAlign;

    uint32_t dInnerAlign = (dSizeIdx == dSize - 1) ? dInnerTailAlign : clcDInner;

    LocalTensor<T> doutTensor = m_vecInQue1.DeQue<T>();
    LocalTensor<T> outTensor = m_vecInQue2.DeQue<T>();
    LocalTensor<U> castDoutTensor;
    LocalTensor<U> castOutTensor;

    uint32_t outShape[] = { m_nKV * m_gSQ, dInnerAlign };
    outTensor.SetSize(m_nKV * m_gSQ * dInnerAlign);
    outTensor.SetShapeInfo(ShapeInfo(2, outShape, DataFormat::ND));

    doutTensor.SetSize(m_nKV * m_gSQ * dInnerAlign);
    doutTensor.SetShapeInfo(ShapeInfo(2, outShape, DataFormat::ND));

    if constexpr (!IsSameType<T, U>::value) {
        castDoutTensor = m_vecInQue4.AllocTensor<U>();
        castDoutTensor.SetSize(m_nKV * m_gSQ * dInnerAlign);

        Cast(castDoutTensor, doutTensor, RoundMode::CAST_NONE, m_nKV * m_gSQ * dInnerAlign);
        m_vecInQue1.FreeTensor(doutTensor);

        castOutTensor = m_vecInQue1.AllocTensor<U>();
        castOutTensor.SetSize(m_nKV * m_gSQ * dInnerAlign);
        castOutTensor.SetShapeInfo(ShapeInfo(2, outShape, DataFormat::ND));

        pipe_barrier(PIPE_V);
        Cast(castOutTensor, outTensor, RoundMode::CAST_NONE, m_nKV * m_gSQ * dInnerAlign);
        m_vecInQue2.FreeTensor(outTensor);
    }

    LocalTensor<uint8_t> apiClcTensor = m_apiClcQue.AllocTensor<uint8_t>();
    apiClcTensor.SetSize(m_ordTilingData->splitCoreParams.apiClcQueueSize);

    LocalTensor<U> softmaxTensor;
    if constexpr (!IsSameType<T, U>::value) {
        softmaxTensor = m_vecInQue2.AllocTensor<U>();
    } else {
        softmaxTensor = m_vecInQue4.AllocTensor<U>();
    }

    softmaxTensorAll.SetSize(m_nKV * m_gSQ * (32 / sizeof(U)));
    uint32_t outOffset = 0;
    uint32_t softmaxOffset = 0;
    pipe_barrier(PIPE_V);
    if constexpr (!IsSameType<T, U>::value) {
        SoftmaxGradFront<U, false>(softmaxTensor, castDoutTensor, castOutTensor,
                                   apiClcTensor,
                                   m_ordTilingData->softmaxGradTilingData);
    } else {
        SoftmaxGradFront<U, false>(softmaxTensor, doutTensor, outTensor,
                                   apiClcTensor,
                                   m_ordTilingData->softmaxGradTilingData);
    }

    m_apiClcQue.FreeTensor(apiClcTensor);

    pipe_barrier(PIPE_V);
    Add(softmaxTensorAll, softmaxTensor, softmaxTensorAll, m_nKV * m_gSQ * (32 / sizeof(U)));

    if constexpr (!IsSameType<T, U>::value) {
        m_vecInQue4.FreeTensor(castDoutTensor);
        m_vecInQue1.FreeTensor(castOutTensor);
        m_vecInQue2.FreeTensor(softmaxTensor);
    } else {
        m_vecInQue1.FreeTensor(doutTensor);
        m_vecInQue2.FreeTensor(outTensor);
        m_vecInQue4.FreeTensor(softmaxTensor);
    }
}

template<typename T, typename U, LayoutMode LAYOUT>
__aicore__ inline void NewFlashAttentionScoreGradBase<T, U, LAYOUT>::CalcSoftmaxFrontFor1D(
    LocalTensor<U> &softmaxTensorAll)
{
    uint32_t dInnerTailAlign = m_ordTilingData->splitDInfo.dInnerTailAlign;
    uint32_t dInnerAlign = dInnerTailAlign;

    LocalTensor<T> doutTensor = m_vecInQue1.DeQue<T>();
    LocalTensor<T> outTensor = m_vecInQue2.DeQue<T>();
    LocalTensor<U> castDoutTensor;
    LocalTensor<U> castOutTensor;

    uint32_t outShape[] = { m_nKV * m_gSQ, dInnerAlign };
    outTensor.SetSize(m_nKV * m_gSQ * dInnerAlign);
    outTensor.SetShapeInfo(ShapeInfo(2, outShape, DataFormat::ND));

    doutTensor.SetSize(m_nKV * m_gSQ * dInnerAlign);
    doutTensor.SetShapeInfo(ShapeInfo(2, outShape, DataFormat::ND));

    if constexpr (!IsSameType<T, U>::value) {
        castDoutTensor = m_vecInQue4.AllocTensor<U>();
        castDoutTensor.SetSize(m_nKV * m_gSQ * dInnerAlign);

        Cast(castDoutTensor, doutTensor, RoundMode::CAST_NONE, m_nKV * m_gSQ * dInnerAlign);
        m_vecInQue1.FreeTensor(doutTensor);

        castOutTensor = m_vecInQue1.AllocTensor<U>();
        castOutTensor.SetSize(m_nKV * m_gSQ * dInnerAlign);
        castOutTensor.SetShapeInfo(ShapeInfo(2, outShape, DataFormat::ND));

        pipe_barrier(PIPE_V);
        Cast(castOutTensor, outTensor, RoundMode::CAST_NONE, m_nKV * m_gSQ * dInnerAlign);
        m_vecInQue2.FreeTensor(outTensor);
    }

    LocalTensor<uint8_t> apiClcTensor = m_apiClcQue.AllocTensor<uint8_t>();
    apiClcTensor.SetSize(m_ordTilingData->splitCoreParams.apiClcQueueSize);

    softmaxTensorAll.SetSize(m_nKV * m_gSQ * (32 / sizeof(U)));
    uint32_t outOffset = 0;
    uint32_t softmaxOffset = 0;
    pipe_barrier(PIPE_V);
    if constexpr (!IsSameType<T, U>::value) {
        SoftmaxGradFront<U, false>(softmaxTensorAll, castDoutTensor, castOutTensor,
                                   apiClcTensor,
                                   m_ordTilingData->softmaxGradTilingData);
    } else {
        SoftmaxGradFront<U, false>(softmaxTensorAll, doutTensor, outTensor,
                                   apiClcTensor,
                                   m_ordTilingData->softmaxGradTilingData);
    }

    m_apiClcQue.FreeTensor(apiClcTensor);

    pipe_barrier(PIPE_V);
    if constexpr (!IsSameType<T, U>::value) {
        m_vecInQue4.FreeTensor(castDoutTensor);
        m_vecInQue1.FreeTensor(castOutTensor);
    } else {
        m_vecInQue1.FreeTensor(doutTensor);
        m_vecInQue2.FreeTensor(outTensor);
    }
}

template<typename T, typename U, LayoutMode LAYOUT>
__aicore__ inline void NewFlashAttentionScoreGradBase<T, U, LAYOUT>::CopyInDrop(uint64_t &batchOffsetMask)
{
    LocalTensor<uint8_t> dropMaskTensor = m_vecInQue1.AllocTensor<uint8_t>();
    dropMaskTensor.SetSize(m_ordTilingData->offsetInfo.maskInSize);
    uint32_t dropMaskShape[] = { m_nKV, m_gSQ, m_sKVAligned };
    dropMaskTensor.SetShapeInfo(ShapeInfo(3, dropMaskShape, DataFormat::ND));

    DataCopyParams intriParams;
    intriParams.blockCount = (uint16_t)(m_nKV * m_gSQ);
    intriParams.blockLen = (uint16_t)(m_sKV * DROP_DATA_SIZE);
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    DataCopyPadParams intriParamsPad { m_sKV != m_sKVAligned,
                                       0,
                                       static_cast<uint8_t>((m_sKVAligned - m_sKV) * DROP_DATA_SIZE),
                                       0 };
    DataCopyPad(dropMaskTensor, m_dropMaskGm[batchOffsetMask], intriParams, intriParamsPad);
    m_vecInQue1.EnQue(dropMaskTensor);
}

template<typename T, typename U, LayoutMode LAYOUT>
__aicore__ inline void NewFlashAttentionScoreGradBase<T, U, LAYOUT>::CalcDrop(LocalTensor<U> &src1Tensor)
{
    LocalTensor<uint8_t> dropMaskTensor = m_vecInQue1.DeQue<uint8_t>();
    LocalTensor<U> dstTensor = m_vecInQue4.AllocTensor<U>();

    LocalTensor<uint8_t> apiClcTensor = m_apiClcQue.AllocTensor<uint8_t>();
    apiClcTensor.SetSize(m_ordTilingData->splitCoreParams.apiClcQueueSize);

    DropOutShapeInfo info;
    info.firstAxis = m_nKV * m_gSQ; // TODO
    info.srcLastAxis = m_sKV;
    info.maskLastAxis = m_sKV; // (sKV / 16 + 31) / 32 * 32; // why 16?

    pipe_barrier(PIPE_V);

    DropOut<U, true>(dstTensor, src1Tensor, dropMaskTensor,
        apiClcTensor, m_ordTilingData->opInfo.keepProb, info);

    m_vecInQue4.EnQue<U>(dstTensor);

    m_apiClcQue.FreeTensor(apiClcTensor);
    m_vecInQue1.FreeTensor(dropMaskTensor);
}

template<typename T, typename U, LayoutMode LAYOUT>
__aicore__ inline void NewFlashAttentionScoreGradBase<T, U, LAYOUT>::SubClc(LocalTensor<U> &softmaxGradTensor)
{
    LocalTensor<U> dropOutTensor = m_vecInQue4.DeQue<U>();

    LocalTensor<U> dropOut2subTensor = m_vecOutQue.AllocTensor<U>();
    uint32_t subOffset = 0;
    uint32_t softmaxOffset = 0;

    uint16_t blockSize = BLOCK_SIZE / sizeof(U);
    uint8_t repeatTime = static_cast<uint8_t>(m_ordTilingData->singleCoreParams.sKVAlignBlockNum);
    uint32_t subRange = m_ordTilingData->singleCoreParams.subRange;
    uint32_t subMask = m_ordTilingData->singleCoreParams.subMask;
    uint32_t subMaskTail = m_ordTilingData->singleCoreParams.subMaskTail;

    for (uint32_t nIdx = 0; nIdx < m_nQ; ++nIdx) {
        for (uint32_t idx = 0; idx < subRange; ++idx) {
            uint32_t mask = subMask;
            if (idx == subRange - 1) {
                mask = subMaskTail;
            }
            Sub(dropOut2subTensor[subOffset + idx * mask * repeatTime],
                dropOutTensor[subOffset + idx * mask * repeatTime],
                softmaxGradTensor[softmaxOffset + mask * idx],
                mask, repeatTime, {repeatTime, repeatTime, 1, 1, 1, 0});
        }

        subOffset += m_innerSize;
        softmaxOffset += m_sQ * blockSize;
    }

    m_vecOutQue.EnQue<U>(dropOut2subTensor);
    m_vecInQue4.FreeTensor(dropOutTensor);
}

template<typename T, typename U, LayoutMode LAYOUT>
__aicore__ inline void NewFlashAttentionScoreGradBase<T, U, LAYOUT>::CalcDP(
  uint64_t &batchOffsetQ, uint64_t &batchOffsetKV, uint64_t &batchOffsetDrop)
{
    RunBmm1(batchOffsetQ, batchOffsetKV);
    uint32_t softmaxSize = m_gSQ * (32 / sizeof(U));
    LocalTensor<U> softmaxTensorAll = m_vecInQue3.AllocTensor<U>();

    Duplicate<U>(softmaxTensorAll, 0.0, m_nKV * softmaxSize); // 需要初始化为0！

    uint32_t dSize = m_ordTilingData->splitDInfo.dSize;
    // 切D处理
    if (dSize == 1) {
        CopyInSoftmaxGrad(batchOffsetQ, 0);
        CalcSoftmaxFrontFor1D(softmaxTensorAll);
    } else {
        for (uint32_t dSizeIdx = 0; dSizeIdx < dSize; dSizeIdx++) {
            CopyInSoftmaxGrad(batchOffsetQ, dSizeIdx);
            CalcSoftmaxFront(softmaxTensorAll, dSizeIdx);
        }
    }

    CopyInDrop(batchOffsetDrop);
    LocalTensor<U> vDoutTensor = m_vecOutQue.DeQue<U>();

    CalcDrop(vDoutTensor);
    m_vecOutQue.FreeTensor(vDoutTensor);

    SubClc(softmaxTensorAll); // TODO
    m_vecInQue3.FreeTensor(softmaxTensorAll);
}

template<typename T, typename U, LayoutMode LAYOUT>
__aicore__ inline void NewFlashAttentionScoreGradBase<T, U, LAYOUT>::RunBmm2(uint64_t batchOffsetQ,
    uint64_t batchOffsetKV, LocalTensor<U> &calcDSTensor)
{
    uint32_t vDoutShape[3] = {m_nKV, m_gSQ, m_sKV};
    calcDSTensor.SetShapeInfo(ShapeInfo(3, vDoutShape, DataFormat::ND));

    m_bmm2.SetTail(m_sQ, m_sKV, m_d);
    m_bmm2.SetTensorA(m_queryGm[batchOffsetQ]);
    m_bmm2.SetTensorB(m_keyGm[batchOffsetKV], true);
    m_bmm2.template IterateBatch<true>(calcDSTensor, m_nKV, m_nKV, false);

    m_bmm2.End();
}

template<typename T, typename U, LayoutMode LAYOUT>
__aicore__ inline bool NewFlashAttentionScoreGradBase<T, U, LAYOUT>::CropInPse(uint64_t &batchOffsetPse)
{
    if (m_pseGm.GetPhyAddr() == nullptr) {
        return false;
    }
    LocalTensor<T> pseTensor = m_vecInQue1.AllocTensor<T>();
    uint32_t pseShape[3] = {m_nQ, m_sQ, m_sKV};
    if (m_ordTilingData->opInfo.pseShapeType == 1) {  // BN1S
        pseShape[1] = BLOCK_SIZE / sizeof(U);
    }

    DataCopyParams intriParams;
    intriParams.blockCount = (uint16_t)(pseShape[1]);
    intriParams.blockLen = (uint16_t)(m_sKV * sizeof(U));
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    DataCopyPadParams intriParamsPad { m_sKV != m_sKVAligned,
                                       0,
                                       static_cast<uint8_t>((m_sKVAligned - m_sKV) * sizeof(U)),
                                       0 };

    pseTensor.SetShapeInfo(ShapeInfo(3, pseShape, DataFormat::ND));
    DataCopy(pseTensor, m_pseGm[batchOffsetPse], m_allInnerSize);

    m_vecInQue1.EnQue(pseTensor);
    return true;
}

template<typename T, typename U, LayoutMode LAYOUT>
__aicore__ inline void NewFlashAttentionScoreGradBase<T, U, LAYOUT>::AddPseClc(uint64_t &batchOffsetPse,
    uint64_t &batchOffsetKV, LocalTensor<U> &mat2addTensor)
{
    auto pseTensor = m_vecInQue1.DeQue<T>();
    if constexpr (!IsSameType<T, U>::value) {
        LocalTensor<U> castPseTensor = m_vecInQue2.AllocTensor<U>();
        castPseTensor.SetSize(m_ordTilingData->offsetInfo.pseInSize); // pse input size

        uint32_t castPseShape[3] = {m_nQ, m_sQ, m_sKV};
        castPseTensor.SetShapeInfo(ShapeInfo(3, castPseShape, DataFormat::ND));

        Cast(castPseTensor, pseTensor, RoundMode::CAST_NONE, m_ordTilingData->offsetInfo.pseInSize);
        m_vecInQue1.FreeTensor(pseTensor);
        PseBroadcastAdd(mat2addTensor, castPseTensor);
        m_vecInQue2.FreeTensor(castPseTensor);
    } else {
        PseBroadcastAdd(mat2addTensor, pseTensor);
        m_vecInQue1.FreeTensor(pseTensor);
    }
    pipe_barrier(PIPE_V);
    Muls(mat2addTensor, mat2addTensor, (U)(m_ordTilingData->opInfo.scaleValue), m_allInnerSize);
}

template <typename T, typename U, LayoutMode LAYOUT>
__aicore__ inline void NewFlashAttentionScoreGradBase<T, U, LAYOUT>::PseBroadcastAdd(
    LocalTensor<U> &mat2addTensor, LocalTensor<U> pseTensor)
{
    pipe_barrier(PIPE_V);
    if (m_ordTilingData->opInfo.pseShapeType == 0) {  // BNSS
        Add(mat2addTensor, mat2addTensor, pseTensor, m_ordTilingData->offsetInfo.pseInSize); // TODO;
    } else {  // BN_11S
        uint32_t innerOffset = 0;
        uint32_t pseOffset = 0;

        // TODO: sKV > stepCalcNum tail process
        uint32_t stepCalcNum = 256 / sizeof(U);
        stepCalcNum = m_sKV > stepCalcNum ? stepCalcNum : m_sKV;

        BinaryRepeatParams repParams;
        repParams.src0BlkStride = 1;
        repParams.src0RepStride = m_sKV / (BLOCK_SIZE / sizeof(U));
        repParams.src1BlkStride = 1;
        repParams.src1RepStride = 0;
        repParams.dstRepStride = repParams.src0RepStride;
        repParams.blockNumber = repParams.src0RepStride;
        for (uint32_t nIdx = 0; nIdx < m_nQ; ++nIdx) {
          Add(mat2addTensor[innerOffset], mat2addTensor[innerOffset], pseTensor[pseOffset], stepCalcNum,
            m_sQ, repParams);
          innerOffset += m_innerSize;
          pseOffset += m_ordTilingData->opInfo.innerPseSize;
        }
    }
}

template<typename T, typename U, LayoutMode LAYOUT>
__aicore__ inline void NewFlashAttentionScoreGradBase<T, U, LAYOUT>::CopyInMask(uint64_t &batchOffsetMask)
{
    LocalTensor<T> maskTensor = m_vecInQue1.AllocTensor<T>();
    maskTensor.SetSize(m_ordTilingData->offsetInfo.maskInSize); // pse input size
    uint32_t maskShape[] = {m_nKV, m_gSQ, m_sKV};
    maskTensor.SetShapeInfo(ShapeInfo(3, maskShape, DataFormat::ND));

    struct DataCopyParams intriParams;
    intriParams.blockCount = (uint16_t)(m_nKV * m_gSQ);
    intriParams.blockLen = (uint16_t)(m_sKV * sizeof(T));
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    DataCopyPadParams intriParamsPad { m_sKV != m_sKVAligned,
                                       0,
                                       static_cast<uint8_t>((m_sKVAligned - m_sKV) * sizeof(T)),
                                       0 };
    DataCopyPad(maskTensor, m_attenMaskGm[batchOffsetMask], intriParams, intriParamsPad);
    m_vecInQue1.EnQue(maskTensor);
}

template<typename T, typename U, LayoutMode LAYOUT>
__aicore__ inline void NewFlashAttentionScoreGradBase<T, U, LAYOUT>::CalcMask(LocalTensor<U> &pse2maskTensor)
{
    LocalTensor<T> maskTensor = m_vecInQue1.DeQue<T>();
    LocalTensor<U> castMaskTensor;
    if constexpr (!IsSameType<T, U>::value) {
        pipe_barrier(PIPE_V);
        castMaskTensor = m_vecInQue4.AllocTensor<U>();
        castMaskTensor.SetSize(m_ordTilingData->offsetInfo.maskInSize);
        Cast(castMaskTensor, maskTensor, RoundMode::CAST_NONE, m_ordTilingData->offsetInfo.maskInSize);
        m_vecInQue1.FreeTensor(maskTensor);
    }

    if (m_ordTilingData->opInfo.attenMaskType == 0) {  // 0 表示输入数据类型与 qkv 一致
        if constexpr (!IsSameType<T, U>::value) {
            pipe_barrier(PIPE_V);
            Muls(castMaskTensor, castMaskTensor, (U)(-10000), m_ordTilingData->offsetInfo.maskInSize);
            pipe_barrier(PIPE_V);
            for (uint32_t nIdx = 0; nIdx < m_nQ; ++nIdx) {
                Add(pse2maskTensor[nIdx * m_innerSize], castMaskTensor,
                    pse2maskTensor[nIdx * m_innerSize], m_innerSize);
            }
            m_vecInQue4.FreeTensor(castMaskTensor);
        } else {
            Muls(maskTensor, maskTensor, (T)(-10000), m_allInnerSize); // -10000
            for (uint32_t nIdx = 0; nIdx < m_nQ; ++nIdx) {
                Add(pse2maskTensor[nIdx * m_innerSize], maskTensor, pse2maskTensor[nIdx * m_innerSize], m_innerSize);
            }
            m_vecInQue1.FreeTensor(maskTensor);
        }
    }
}

template<typename T, typename U, LayoutMode LAYOUT>
__aicore__ inline void NewFlashAttentionScoreGradBase<T, U, LAYOUT>::CopyInSoftmax(uint64_t &batchOffsetSoftmax)
{
    LocalTensor<float> softmaxMaxTensor = m_vecInQue2.AllocTensor<float>();
    softmaxMaxTensor.SetSize(m_ordTilingData->offsetInfo.maskInSize);
    uint32_t maskShape[] = {m_nKV, m_gSQ, 8};
    softmaxMaxTensor.SetShapeInfo(ShapeInfo(3, maskShape, DataFormat::ND));

    DataCopyParams intriParams;
    intriParams.blockCount = (uint16_t)(1);
    intriParams.blockLen = (uint16_t)(m_ordTilingData->offsetInfo.softmaxSize);
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    DataCopy(softmaxMaxTensor, m_softmaxMaxGm[batchOffsetSoftmax], m_ordTilingData->offsetInfo.softmaxSize);

    m_vecInQue2.EnQue(softmaxMaxTensor);

    LocalTensor<float> softmaxSumTensor = m_vecInQue1.AllocTensor<float>();
    softmaxSumTensor.SetSize(m_ordTilingData->offsetInfo.softmaxSize);
    softmaxSumTensor.SetShapeInfo(ShapeInfo(3, maskShape, DataFormat::ND));

    DataCopy(softmaxSumTensor, m_softmaxSumGm[batchOffsetSoftmax], m_ordTilingData->offsetInfo.softmaxSize);

    m_vecInQue1.EnQue(softmaxSumTensor);
}

template<typename T, typename U, LayoutMode LAYOUT>
__aicore__ inline void NewFlashAttentionScoreGradBase<T, U, LAYOUT>::CalcSoftmax(LocalTensor<U> &mask2softmaxTensor)
{
    LocalTensor<uint8_t> apiClcTensor = m_apiClcQue.AllocTensor<uint8_t>();
    apiClcTensor.SetSize(m_ordTilingData->splitCoreParams.apiClcQueueSize);
    LocalTensor<float> softmaxMaxTensor = m_vecInQue2.DeQue<float>();
    LocalTensor<float> softmaxSumTensor = m_vecInQue1.DeQue<float>();

    LocalTensor<float> castTensor = m_vecInQue4.AllocTensor<float>();
    castTensor.SetSize(m_allInnerSize);

    uint32_t innerOffset = 0;
    uint32_t softmaxOffset = 0;
    uint32_t outShape1[] = {m_gSQ, m_sKV};
    pipe_barrier(PIPE_V);
    for (uint32_t nIdx = 0; nIdx < m_nKV; ++nIdx) {
        LocalTensor<U> outTensor1 = mask2softmaxTensor[innerOffset];
        outTensor1.SetShapeInfo(ShapeInfo(2, outShape1, DataFormat::ND));
        SimpleSoftMax<U, true, false>(outTensor1, softmaxSumTensor[softmaxOffset], softmaxMaxTensor[softmaxOffset],
			              outTensor1, apiClcTensor, m_ordTilingData->softmaxTilingData);
        innerOffset += m_gSQ * m_sKV;
        softmaxOffset += m_gSQ * 8; // TODO number padding
    }

    m_apiClcQue.FreeTensor(apiClcTensor);
    m_vecInQue2.FreeTensor(softmaxMaxTensor);
    m_vecInQue1.FreeTensor(softmaxSumTensor);
    m_vecInQue4.FreeTensor(castTensor);
}

template<typename T, typename U, LayoutMode LAYOUT>
__aicore__ inline void NewFlashAttentionScoreGradBase<T, U, LAYOUT>::CalcMul(LocalTensor<U> &softmaxTensor)
{
    LocalTensor<U> sub2MulTensor = m_vecOutQue.DeQue<U>();

    pipe_barrier(PIPE_V);
    Mul(sub2MulTensor, softmaxTensor, sub2MulTensor, m_allInnerSize);

    if constexpr (!IsSameType<T, U>::value) {
        LocalTensor<T> castSub2MulTensor = m_vecInQue2.AllocTensor<T>();
        castSub2MulTensor.SetSize(sub2MulTensor.GetSize());
        uint32_t castPseShape[3] = {m_nQ,
                                    m_sQ,
                                    m_sKV};

        castSub2MulTensor.SetShapeInfo(ShapeInfo(3, castPseShape, DataFormat::ND));
        pipe_barrier(PIPE_V);
        Cast(castSub2MulTensor, sub2MulTensor, RoundMode::CAST_ROUND, sub2MulTensor.GetSize());
        m_vecOutQue.FreeTensor(sub2MulTensor);
        m_vecInQue2.EnQue(castSub2MulTensor);
        return;
    }

    m_vecOutQue.EnQue(sub2MulTensor);
}

template<typename T, typename U, LayoutMode LAYOUT>
__aicore__ inline void NewFlashAttentionScoreGradBase<T, U, LAYOUT>::CalcDS(uint64_t &batchOffsetQ,
    uint64_t &batchOffsetKV, uint64_t &batchOffsetMask, uint64_t &batchOffsetSoftmax, uint64_t &batchOffsetPse)
{
    LocalTensor<U> calcDSTensor = m_vecInQue3.AllocTensor<U>();
    RunBmm2(batchOffsetQ, batchOffsetKV, calcDSTensor);

    CropInPse(batchOffsetPse);
    AddPseClc(batchOffsetPse, batchOffsetKV, calcDSTensor);

    CopyInMask(batchOffsetMask);
    CalcMask(calcDSTensor);

    CopyInSoftmax(batchOffsetSoftmax);
    CalcSoftmax(calcDSTensor);
    CalcMul(calcDSTensor);

    m_vecInQue3.EnQue<U>(calcDSTensor);
}

template<typename T, typename U, LayoutMode LAYOUT>
__aicore__ inline void NewFlashAttentionScoreGradBase<T, U, LAYOUT>::RunBmm4(uint64_t batchOffsetQ,
    uint64_t batchOffsetKV) // get from tiling
{
    LocalTensor<U> drop2BmmTensor = m_vecInQue4.DeQue<U>();

    LocalTensor<T> castDrop2BmmTensor = m_vecInQue1.AllocTensor<T>();

    if constexpr (!IsSameType<T, U>::value) {
        Cast(castDrop2BmmTensor, drop2BmmTensor, RoundMode::CAST_ROUND, m_allInnerSize);
    }

    m_bmm4.SetTail(m_sKV, -1, m_gSQ);
    if constexpr (!IsSameType<T, U>::value) {
        m_bmm4.SetTensorA(castDrop2BmmTensor, true);
    } else {
        m_bmm4.SetTensorA(drop2BmmTensor, true);
    }
    m_bmm4.SetTensorB(m_doutGm[batchOffsetQ]);
    m_bmm4.template IterateBatch<true>(m_dvGm[batchOffsetKV], m_nKV, m_nKV, false);

    m_vecInQue1.FreeTensor(castDrop2BmmTensor);
    m_vecInQue4.FreeTensor(drop2BmmTensor);

    m_bmm4.End();
}

template<typename T, typename U, LayoutMode LAYOUT>
__aicore__ inline void NewFlashAttentionScoreGradBase<T, U, LAYOUT>::CalcDV(uint64_t &batchOffsetQ,
    uint64_t &batchOffsetKV, uint64_t &batchOffsetDrop)
{
    LocalTensor<U> dSTensor = m_vecInQue3.DeQue<U>();
    CopyInDrop(batchOffsetDrop);
    CalcDrop(dSTensor);
    RunBmm4(batchOffsetQ, batchOffsetKV);
    m_vecInQue3.FreeTensor(dSTensor);
}

template<typename T, typename U, LayoutMode LAYOUT>
__aicore__ inline void NewFlashAttentionScoreGradBase<T, U, LAYOUT>::RunBmm31(uint64_t batchOffsetQ,
    uint64_t batchOffsetKV) // get from tiling
{
    LocalTensor<T> mul2BmmTensor;
    if constexpr (!IsSameType<T, U>::value) {
        mul2BmmTensor = m_vecInQue2.DeQue<T>();
    } else {
        mul2BmmTensor = m_vecOutQue.DeQue<T>();
    }

    m_bmm4.SetTail(m_sKV, -1, m_gSQ);
    m_bmm4.SetTensorA(mul2BmmTensor, true);
    m_bmm4.SetTensorB(m_queryGm[batchOffsetQ]);
    m_bmm4.template IterateBatch<true>(m_dkGm[batchOffsetKV], m_nKV, m_nKV, false);

    if constexpr (!IsSameType<T, U>::value) {
        m_vecInQue2.EnQue(mul2BmmTensor);
    } else {
        m_vecOutQue.EnQue(mul2BmmTensor);
    }
    m_bmm4.End();
}


template<typename T, typename U, LayoutMode LAYOUT>
__aicore__ inline void NewFlashAttentionScoreGradBase<T, U, LAYOUT>::RunBmm32(uint64_t batchOffsetQ,
    uint64_t batchOffsetKV)
{
    LocalTensor<T> mul2BmmTensor;
    if constexpr (!IsSameType<T, U>::value) {
        mul2BmmTensor = m_vecInQue2.DeQue<T>();
    } else {
        mul2BmmTensor = m_vecOutQue.DeQue<T>();
    }

    m_bmm3.SetTail(m_sQ, -1, m_sKV);
    m_bmm3.SetTensorA(mul2BmmTensor, false);
    m_bmm3.SetTensorB(m_keyGm[batchOffsetKV]);
    m_bmm3.template IterateBatch<true>(m_dqGm[batchOffsetQ], m_nKV, m_nKV, false);

    if constexpr (!IsSameType<T, U>::value) {
        m_vecInQue2.FreeTensor(mul2BmmTensor);
    } else {
        m_vecOutQue.FreeTensor(mul2BmmTensor);
    }
    m_bmm3.End();
}

template<typename T, typename U, LayoutMode LAYOUT>
__aicore__ inline void NewFlashAttentionScoreGradBase<T, U, LAYOUT>::CalcDQDK(uint64_t &batchOffsetQ,
    uint64_t &batchOffsetKV)
{
    RunBmm31(batchOffsetQ, batchOffsetKV);
    RunBmm32(batchOffsetQ, batchOffsetKV);
    MIX_LOG("end");
}
#endif

