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

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

#include "flash_attention_score_common.h"

using AscendC::BinaryRepeatParams;
using AscendC::DataCopyPadParams;
using AscendC::DataCopyParams;
using AscendC::GetBlockIdx;
using AscendC::GetUserWorkspace;
using AscendC::GlobalTensor;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::QuePosition;
using AscendC::RoundMode;
using AscendC::SetSysWorkspace;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TQue;
using matmul::MatmulType;

#if __CCE_KT_TEST__
#include <sys/types.h>
#include <unistd.h>

#include <sstream>
#include <string>

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

constexpr int DEBUG_T_WIDTH = 10;
constexpr int DEBUG_T_PRECISION = 6;
constexpr int DEBUG_HALF_WIDTH = 14;
constexpr int DEBUG_HALF_PRECISION = 4;

template <typename T>
std::string DoPrintData(const LocalTensor<T> &tensor, size_t count, size_t stride, size_t elementsPerRow,
                        const std::string &block_id, const std::string &core_type)
{
    auto data = tensor.GetPhyAddr();
    std::ostringstream oss;
    for (size_t localCount = 0; localCount < count; ++localCount) {
        if (localCount == 0) {
            oss << "[" << block_id << "][" << core_type << "] ";
        }
        oss << std::setw(DEBUG_T_WIDTH) << std::setprecision(DEBUG_T_PRECISION) << data[localCount * stride] << " ";
        if (localCount % elementsPerRow == elementsPerRow - 1) {
            oss << std::endl << "[" << block_id << "][" << core_type << "] ";
        }
    }
    return oss.str();
}

template <>
std::string DoPrintData(const LocalTensor<uint8_t> &tensor, size_t count, size_t stride, size_t elementsPerRow,
                        const std::string &block_id, const std::string &core_type)
{
    auto data = tensor.GetPhyAddr();
    std::ostringstream oss;
    for (size_t localCount = 0; localCount < count; ++localCount) {
        if (localCount == 0) {
            oss << "[" << block_id << "][" << core_type << "] ";
        }
        oss << std::setw(DEBUG_T_WIDTH) << std::setprecision(DEBUG_T_PRECISION) << (int)data[localCount * stride]
            << " ";
        if (localCount % elementsPerRow == elementsPerRow - 1) {
            oss << std::endl << "[" << block_id << "][" << core_type << "] ";
        }
    }
    return oss.str();
}

template <>
std::string DoPrintData(const LocalTensor<half> &tensor, size_t count, size_t stride, size_t elementsPerRow,
                        const std::string &block_id, const std::string &core_type)
{
    auto data = tensor.GetPhyAddr();
    std::ostringstream oss;
    for (size_t localCount = 0; localCount < count; ++localCount) {
        if (localCount == 0) {
            oss << "[" << block_id << "][" << core_type << "] ";
        }
        oss << std::setw(DEBUG_HALF_WIDTH) << std::setw(DEBUG_HALF_PRECISION) << data[localCount * stride].ToFloat()
            << " ";
        if ((localCount % elementsPerRow == elementsPerRow - 1) && (localCount != count - 1)) {
            oss << std::endl << "[" << block_id << "][" << core_type << "] ";
        }
    }
    return oss.str();
}

template <typename T>
std::string DoPrintData(const GlobalTensor<T> &tensor, size_t count, size_t stride, size_t elementsPerRow,
                        const std::string &block_id, const std::string &core_type)
{
    auto data = tensor.GetPhyAddr();
    std::ostringstream oss;
    for (size_t localCount = 0; localCount < count; ++localCount) {
        if (localCount == 0) {
            oss << "[" << block_id << "][" << core_type << "] ";
        }
        oss << std::setw(DEBUG_T_WIDTH) << std::setprecision(DEBUG_T_PRECISION) << data[localCount * stride] << " ";
        if ((localCount % elementsPerRow == elementsPerRow - 1) && (localCount != count - 1)) {
            oss << std::endl << "[" << block_id << "][" << core_type << "] ";
        }
    }
    return oss.str();
}

template <>
std::string DoPrintData(const GlobalTensor<half> &tensor, size_t count, size_t stride, size_t elementsPerRow,
                        const std::string &block_id, const std::string &core_type)
{
    auto data = tensor.GetPhyAddr();
    std::ostringstream oss;
    for (size_t localCount = 0; localCount < count; ++localCount) {
        if (localCount == 0) {
            oss << "[" << block_id << "][" << core_type << "] ";
        }
        oss << std::setw(DEBUG_HALF_WIDTH) << std::setw(DEBUG_HALF_PRECISION) << data[localCount * stride].ToFloat()
            << " ";
        if ((localCount % elementsPerRow == elementsPerRow - 1) && (localCount != count - 1)) {
            oss << std::endl << "[" << block_id << "][" << core_type << "] ";
        }
    }
    return oss.str();
}

#define PRINT_DATA(data, count, stride, elementsPerRow, format, ...)                                                 \
    do {                                                                                                             \
        std::string core_type = "";                                                                                  \
        std::string block_id = "Block_";                                                                             \
        if (g_coreType == AscendC::AIC_TYPE) {                                                                       \
            core_type = "AIC_";                                                                                      \
        } else if (g_coreType == AscendC::AIV_TYPE) {                                                                \
            core_type = "AIV_";                                                                                      \
        } else {                                                                                                     \
            core_type = "MIX_";                                                                                      \
        }                                                                                                            \
        core_type += std::to_string(sub_block_idx);                                                                  \
        block_id += std::to_string(block_idx);                                                                       \
        printf("[%s][%s][%s:%d][%s][%ld] " format "\n%s\n", block_id.c_str(), core_type.c_str(), FILENAME, __LINE__, \
               __FUNCTION__, (long)getpid(), ##__VA_ARGS__,                                                          \
               DoPrintData(data, count, stride, elementsPerRow, block_id, core_type).c_str());                       \
    } while (0)

#ifndef ASCENDC_OP_UTEST
#define SHORT_MIX_LOG(format, ...)                                                 \
    do {                                                                                                             \
        std::string core_type = "";                                                                                  \
        std::string block_id = "Block_";                                                                             \
        if (g_coreType == AscendC::AIC_TYPE) {                                                                       \
            core_type = "AIC_";                                                                                      \
        } else if (g_coreType == AscendC::AIV_TYPE) {                                                                \
            core_type = "AIV_";                                                                                      \
        } else {                                                                                                     \
            core_type = "MIX_";                                                                                      \
        }                                                                                                            \
        core_type += std::to_string(sub_block_idx);                                                                  \
        block_id += std::to_string(block_idx);                                                                       \
        printf("[%s][%s][%s:%d][%s][%ld] " format "\n", block_id.c_str(), core_type.c_str(), FILENAME, __LINE__,     \
               __FUNCTION__, (long)getpid(), ##__VA_ARGS__);                                                         \
    } while (0)
#else
#define SHORT_MIX_LOG(format, ...)
#endif

#else

#define SHORT_MIX_LOG(format, ...)
#define PRINT_DATA(format, ...)

#endif

constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t NUM_BITS_IN_BYTE = 8;
constexpr int32_t SOFTMAX_MAX_SUM_BUFFER_NUM = 2;
constexpr uint8_t LAYOUT_BSND = 1;
constexpr uint8_t LAYOUT_SBH = 2;
constexpr uint8_t LAYOUT_BNSD = 3;

__aicore__ inline bool MaxSumTilingFuncHelper(const uint32_t workLocalSize, const LastAxisShapeND &ndInfo,
                                              MaxSumTiling &tiling, const uint32_t elementNumPerBlk,
                                              const uint32_t splitBufferCount, const uint32_t reduceBufferCount)
{
    tiling.srcM = ndInfo.m;
    tiling.srcK = ndInfo.k;
    tiling.srcSize = ndInfo.m * ndInfo.k;

    tiling.outMaxM = ndInfo.m;
    tiling.outMaxK = elementNumPerBlk;
    tiling.outMaxSize = ndInfo.m * elementNumPerBlk;

    tiling.reduceM = workLocalSize / (elementNumPerBlk * reduceBufferCount + ndInfo.k * splitBufferCount);
    constexpr uint32_t fp32ElementNumPerBlk = BLOCK_BYTES / sizeof(float);
    if (tiling.reduceM < ndInfo.m && tiling.reduceM > fp32ElementNumPerBlk) {
        tiling.reduceM = tiling.reduceM / fp32ElementNumPerBlk * fp32ElementNumPerBlk;
    }
    tiling.reduceM = tiling.reduceM < ndInfo.m ? tiling.reduceM : ndInfo.m;
    tiling.reduceK = elementNumPerBlk;
    tiling.reduceSize = tiling.reduceM * tiling.reduceK;

    tiling.splitM = tiling.reduceM;
    tiling.splitK = ndInfo.k;
    tiling.splitSize = tiling.splitM * tiling.splitK;

#if __CCE_KT_TEST__ == 1
    assert(tiling.reduceM != 0 && "tiling.reduceM must > 0");
#endif

    tiling.rangeM = ndInfo.m / tiling.reduceM;
    tiling.tailM = ndInfo.m % tiling.reduceM;

    tiling.tailSplitSize = tiling.tailM * ndInfo.k;
    tiling.tailReduceSize = tiling.tailM * elementNumPerBlk;

    return true;
}

__aicore__ inline bool MaxSumTilingFunc(const uint32_t workLocalSize, const LastAxisShapeND &ndInfo,
                                        MaxSumTiling &tiling, const uint32_t elementNumPerBlk)
{
    return MaxSumTilingFuncHelper(workLocalSize, ndInfo, tiling, elementNumPerBlk, 1, 1);
}

__aicore__ inline bool MaxSumNeedCastTilingFunc(const uint32_t workLocalSize, const LastAxisShapeND &ndInfo,
                                                MaxSumTiling &tiling, const uint32_t elementNumPerBlk)
{
  return MaxSumTilingFuncHelper(workLocalSize, ndInfo, tiling, elementNumPerBlk, 2, 1);  // 2: splitBuffer count
}

__aicore__ inline bool MaxSumUpdateTilingFunc(const uint32_t workLocalSize, const LastAxisShapeND &ndInfo,
                                              MaxSumTiling &tiling, const uint32_t elementNumPerBlk)
{
  return MaxSumTilingFuncHelper(workLocalSize, ndInfo, tiling, elementNumPerBlk, 1, 3);  // 3: reduceBuffer count
}

__aicore__ inline bool MaxSumUpdateNeedCastTilingFunc(const uint32_t workLocalSize, const LastAxisShapeND &ndInfo,
                                                      MaxSumTiling &tiling, const uint32_t elementNumPerBlk)
{
  return MaxSumTilingFuncHelper(workLocalSize, ndInfo, tiling, elementNumPerBlk, 2, 3);  // 2: splitBuffer count
}

__aicore__ inline bool CustomSoftmaxTilingFunc(const uint32_t workLocalSize, const LastAxisShapeND &ndInfo,
                                               MaxSumTiling &tiling, const uint32_t elementNumPerBlk)
{
  return MaxSumTilingFuncHelper(workLocalSize, ndInfo, tiling, elementNumPerBlk, 2, 0);  // 2:splitBuffer count
}

namespace math {
    template <typename T>
    __aicore__ inline T Min(T a, T b)
    {
        return a > b ? b : a;
    }

    template <typename T>
    __aicore__ inline T Max(T a, T b)
    {
        return a > b ? a : b;
    }

    template <typename T>
    __aicore__ inline T Ceil(T a, T b)
    {
        return (a + b - 1) / b;
    }

    template <typename T>
    __aicore__ inline T Align(T a, T b)
    {
        return Ceil(a, b) * b;
    }
}

template <typename T, int32_t depth>
__aicore__ inline void CopyOutToGm(GlobalTensor<T> dst, TQue<QuePosition::VECOUT, depth> &outputQue, uint32_t size)
{
    auto src = outputQue.template DeQue<T>();
    DataCopy(dst, src, size);
    outputQue.FreeTensor(src);
}

template <typename T, int32_t depth>
__aicore__ inline void CopyOutToGm(GlobalTensor<T> dst, TQue<QuePosition::VECOUT, depth> &outputQue,
                                   const DataCopyParams &params)
{
    auto src = outputQue.template DeQue<T>();
    DataCopy(dst, src, params);
    outputQue.FreeTensor(src);
}

template <typename T, int32_t depth>
__aicore__ inline void CopyOutToWorkspace(GlobalTensor<T> dst, TQue<QuePosition::VECOUT, depth> &outputQue,
                                          uint32_t size)
{
    CopyOutToGm(dst, outputQue, size);
}

template <typename T, int32_t depth>
__aicore__ inline void CopyOutToWorkspace(GlobalTensor<T> dst, TQue<QuePosition::VECOUT, depth> &outputQue,
                                          const DataCopyParams &params)
{
    CopyOutToGm(dst, outputQue, params);
}

template <typename T>
__aicore__ inline void CopyInFromWorkspace(TQue<QuePosition::VECIN, 1> &inputQue, GlobalTensor<T> src, uint32_t size)
{
    auto dst = inputQue.AllocTensor<T>();
    DataCopy(dst, src, size);
    inputQue.EnQue(dst);
}

template <typename T>
__aicore__ inline void CopyInFromWorkspace(TQue<QuePosition::VECIN, 1> &inputQue, GlobalTensor<T> src,
                                           const DataCopyParams &params)
{
    auto dst = inputQue.AllocTensor<T>();
    DataCopy(dst, src, params);
    inputQue.EnQue(dst);
    PRINT_DATA(dst, 0, 1, 8, "[Data] stage2 bmm2 copy from wksp");
}

template <typename T>
__aicore__ inline void MaxSumUpdateInplaceComputeHelper(
    const LocalTensor<T> &src, const LocalTensor<float> &prevSum, const LocalTensor<float> &prevMax,
    const LocalTensor<float> &workLocal, const LastAxisShapeND &originalSrcShape, const uint32_t &offset1,
    const uint32_t &offset2, const uint32_t &splitSize, const uint32_t &reduceSize, const ReduceLastND &reduceParam,
    const BroadCastLastND &brcParam)
{
    if constexpr (sizeof(T) == sizeof(half)) {
        const LocalTensor<float> &tmpBuffer0 = workLocal;
        const LocalTensor<float> &tmpBuffer1 = workLocal[splitSize];
        const LocalTensor<float> &tmpBuffer2 = workLocal[splitSize + splitSize];
        const LocalTensor<float> &tmpBuffer3 = workLocal[splitSize + splitSize + reduceSize];
        const LocalTensor<float> &tmpBuffer4 = workLocal[splitSize + splitSize + reduceSize + reduceSize];

        const LocalTensor<float> &inSumTmp = prevSum[offset2];

        Cast(tmpBuffer0, src[offset1], RoundMode::CAST_NONE, splitSize);
        pipe_barrier(PIPE_V);
        DataCopy(tmpBuffer2, prevMax[offset2], reduceSize);
        pipe_barrier(PIPE_V);

        // currMax = max(x)
        ReduceMaxLastNDImpl(tmpBuffer4, tmpBuffer0, tmpBuffer3, reduceParam);
        pipe_barrier(PIPE_V);
        // tillMax = max(currMax, prevMax) (s1,1)
        Max(prevMax[offset2], tmpBuffer2, tmpBuffer4, reduceSize);
        pipe_barrier(PIPE_V);
        Sub(tmpBuffer2, tmpBuffer2, prevMax[offset2], reduceSize);
        pipe_barrier(PIPE_V);
        Exp(tmpBuffer2, tmpBuffer2, reduceSize);
        pipe_barrier(PIPE_V);
        Mul(tmpBuffer2, tmpBuffer2, inSumTmp, reduceSize);
        pipe_barrier(PIPE_V);
        // max(currMax, prevMax) (s1,s2)
        BroadCastLastImpl(tmpBuffer1, prevMax[offset2], brcParam);
        pipe_barrier(PIPE_V);
        // x - tillMax
        Sub(tmpBuffer1, tmpBuffer0, tmpBuffer1, splitSize);
        pipe_barrier(PIPE_V);
        // exp(x - tillMax)
        Exp(tmpBuffer1, tmpBuffer1, splitSize);
        pipe_barrier(PIPE_V);
        ReduceSumLastNDImpl(tmpBuffer3, tmpBuffer1, tmpBuffer4, reduceParam);
        pipe_barrier(PIPE_V);
        Add(inSumTmp, tmpBuffer2, tmpBuffer3, reduceSize);
        pipe_barrier(PIPE_V);
    } else {
        const LocalTensor<float> &tmpBuffer0 = workLocal;
        const LocalTensor<float> &tmpBuffer1 = workLocal[splitSize];
        const LocalTensor<float> &tmpBuffer2 = workLocal[splitSize + reduceSize];
        const LocalTensor<float> &tmpBuffer3 = workLocal[splitSize + reduceSize + reduceSize];

        const LocalTensor<float> &srcBuffer = src[offset1];
        const LocalTensor<float> &inSumTmp = prevSum[offset2];

        DataCopy(tmpBuffer1, prevMax[offset2], reduceSize);
        pipe_barrier(PIPE_V);

        // currMax = max(x)
        ReduceMaxLastNDImpl(tmpBuffer3, srcBuffer, tmpBuffer2, reduceParam);
        pipe_barrier(PIPE_V);
        // tillMax = max(currMax, prevMax) (s1,1)
        Max(prevMax[offset2], tmpBuffer1, tmpBuffer3, reduceSize);
        pipe_barrier(PIPE_V);
        Sub(tmpBuffer1, tmpBuffer1, prevMax[offset2], reduceSize);
        pipe_barrier(PIPE_V);
        Exp(tmpBuffer1, tmpBuffer1, reduceSize);
        pipe_barrier(PIPE_V);
        Mul(tmpBuffer1, tmpBuffer1, inSumTmp, reduceSize);
        pipe_barrier(PIPE_V);
        // max(currMax, prevMax) (s1,s2)
        BroadCastLastImpl(tmpBuffer0, prevMax[offset2], brcParam);
        pipe_barrier(PIPE_V);
        // x - tillMax
        Sub(tmpBuffer0, srcBuffer, tmpBuffer0, splitSize);
        pipe_barrier(PIPE_V);
        // exp(x - tillMax)
        Exp(tmpBuffer0, tmpBuffer0, splitSize);
        pipe_barrier(PIPE_V);
        ReduceSumLastNDImpl(tmpBuffer2, tmpBuffer0, tmpBuffer3, reduceParam);
        pipe_barrier(PIPE_V);
        Add(inSumTmp, tmpBuffer1, tmpBuffer2, reduceSize);
        pipe_barrier(PIPE_V);
    }
}

template <typename T>
__aicore__ inline void MaxSumUpdateInplaceComputeImpl(LocalTensor<float> &workLocal, const MaxSumTiling &tiling,
                                                      LocalTensor<float> &prevMax, LocalTensor<float> &prevSum,
                                                      LocalTensor<T> &src, const LastAxisShapeND &originalSrcShape)
{
    uint32_t offset1 = 0;
    uint32_t offset2 = 0;

    ReduceLastND reduceParam = {tiling.splitM, originalSrcShape.k, tiling.splitM,
                                tiling.splitK, tiling.reduceM,     tiling.reduceK};
    BroadCastLastND brcParam = {tiling.splitM, tiling.splitK, tiling.reduceM, tiling.reduceK};

    for (uint32_t i = 0; i < tiling.rangeM; i++) {
        offset2 = i * tiling.reduceSize;
        offset1 = i * tiling.splitSize;
        MaxSumUpdateInplaceComputeHelper<T>(src, prevSum, prevMax, workLocal, originalSrcShape, offset1, offset2,
                                            tiling.splitSize, tiling.reduceSize, reduceParam, brcParam);
    }
    if (tiling.tailM != 0) {
        reduceParam.originalSrcM = tiling.tailM;
        reduceParam.srcM = tiling.tailM;
        reduceParam.dstM = tiling.tailM;
        brcParam.dstM = tiling.tailM;
        brcParam.srcM = tiling.tailM;

        offset2 = tiling.rangeM * tiling.reduceSize;
        offset1 = tiling.rangeM * tiling.splitSize;
        MaxSumUpdateInplaceComputeHelper<T>(src, prevSum, prevMax, workLocal, originalSrcShape, offset1, offset2,
                                            tiling.tailSplitSize, tiling.tailReduceSize, reduceParam, brcParam);
    }
}

template <typename T>
__aicore__ inline void MaxSumUpdateInplaceCompute(LocalTensor<T> &src, LocalTensor<float> &prevMax,
                                                  LocalTensor<float> &prevSum, LocalTensor<float> &workLocal,
                                                  const MaxSumTiling &tiling)
{
    ShapeInfo srcShape = src.GetShapeInfo();
    LastAxisShapeND originalSrcShape = GetLastAxisOriginShapeND(srcShape);

    if (originalSrcShape.m != tiling.srcM || originalSrcShape.k != tiling.srcK) {
        MaxSumTiling newTiling = tiling;
        if (IsSameType<T, float>::value) {
            MaxSumUpdateTilingFunc(workLocal.GetSize(), originalSrcShape, newTiling, BLOCK_BYTES / sizeof(float));
        } else {
            MaxSumUpdateNeedCastTilingFunc(workLocal.GetSize(), originalSrcShape, newTiling,
                                           BLOCK_BYTES / sizeof(float));
        }
        MaxSumUpdateInplaceComputeImpl(workLocal, newTiling, prevMax, prevSum, src, originalSrcShape);
    } else {
        MaxSumUpdateInplaceComputeImpl(workLocal, tiling, prevMax, prevSum, src, originalSrcShape);
    }
}

template <typename T>
__aicore__ inline void CustomSoftmaxComputeImplHelper(
    const LocalTensor<T> &src, const LocalTensor<T> &dst, LocalTensor<float> &softmaxMaxUb,
    LocalTensor<float> &softmaxSumUb, const LocalTensor<float> &workLocal, const uint32_t &offset1,
    const uint32_t &offset2, const uint32_t &splitSize, const BroadCastLastND &brcParam,
    const LocalTensor<float> &tmpBuffer0, const LocalTensor<float> &tmpBuffer1)
{
    if constexpr (sizeof(T) == sizeof(half)) {
        Cast(tmpBuffer1, src[offset1], RoundMode::CAST_NONE, splitSize);
        pipe_barrier(PIPE_V);

        BroadCastLastImpl(tmpBuffer0, softmaxMaxUb[offset2], brcParam);
        pipe_barrier(PIPE_V);
        Sub(tmpBuffer0, tmpBuffer1, tmpBuffer0, splitSize);
        pipe_barrier(PIPE_V);
        Exp(tmpBuffer0, tmpBuffer0, splitSize);
        pipe_barrier(PIPE_V);
        BroadCastLastImpl(tmpBuffer1, softmaxSumUb[offset2], brcParam);
        pipe_barrier(PIPE_V);
        Div(tmpBuffer0, tmpBuffer0, tmpBuffer1, splitSize);
        pipe_barrier(PIPE_V);

        Cast(dst[offset1], tmpBuffer0, RoundMode::CAST_ROUND, splitSize);
        pipe_barrier(PIPE_V);
    } else {
        BroadCastLastImpl(tmpBuffer0, softmaxMaxUb[offset2], brcParam);
        pipe_barrier(PIPE_V);
        Sub(tmpBuffer0, src[offset1], tmpBuffer0, splitSize);
        pipe_barrier(PIPE_V);
        Exp(tmpBuffer0, tmpBuffer0, splitSize);
        pipe_barrier(PIPE_V);
        BroadCastLastImpl(tmpBuffer1, softmaxSumUb[offset2], brcParam);
        pipe_barrier(PIPE_V);
        Div(dst[offset1], tmpBuffer0, tmpBuffer1, splitSize);
        pipe_barrier(PIPE_V);
    }
}

template <typename T>
__aicore__ inline void CustomSoftmaxComputeImpl(LocalTensor<float> &workLocal, const MaxSumTiling &tiling,
                                                LocalTensor<float> &softmaxMaxUb, LocalTensor<float> &softmaxSumUb,
                                                LocalTensor<T> &src, LocalTensor<T> &dst)
{
    const LocalTensor<float> &tmpBuffer0 = workLocal;
    const LocalTensor<float> &tmpBuffer1 = workLocal[tiling.splitSize];

    BroadCastLastND brcParam = {tiling.splitM, tiling.splitK, tiling.reduceM, tiling.reduceK};

    uint32_t offset1 = 0;
    uint32_t offset2 = 0;

    for (uint32_t i = 0; i < tiling.rangeM; i++) {
        offset2 = i * tiling.reduceSize;
        offset1 = i * tiling.splitSize;

        CustomSoftmaxComputeImplHelper(src, dst, softmaxMaxUb, softmaxSumUb, workLocal, offset1, offset2,
                                       tiling.splitSize, brcParam, tmpBuffer0, tmpBuffer1);
    }
    if (tiling.tailM != 0) {
        brcParam.dstM = tiling.tailM;
        brcParam.srcM = tiling.tailM;

        offset2 = tiling.rangeM * tiling.reduceSize;
        offset1 = tiling.rangeM * tiling.splitSize;

        CustomSoftmaxComputeImplHelper(src, dst, softmaxMaxUb, softmaxSumUb, workLocal, offset1, offset2,
                                       tiling.tailSplitSize, brcParam, tmpBuffer0, tmpBuffer1);
    }
}

template <typename T>
__aicore__ inline void CustomSoftmaxCompute(LocalTensor<float> &softmaxMaxUb, LocalTensor<float> &softmaxSumUb,
                                            TQue<QuePosition::VECIN, 1> &inputQue, LocalTensor<T> &dst,
                                            LocalTensor<float> &workLocal, const MaxSumTiling &tiling,
                                            const ShapeInfo &srcShape)
{
    LocalTensor<T> src = inputQue.DeQue<T>();
    LastAxisShapeND originalSrcShape = GetLastAxisOriginShapeND(srcShape);

    if (originalSrcShape.m != tiling.srcM || originalSrcShape.k != tiling.srcK) {
        MaxSumTiling newTiling = tiling;
        CustomSoftmaxTilingFunc(workLocal.GetSize(), originalSrcShape, newTiling, BLOCK_BYTES / sizeof(float));
        CustomSoftmaxComputeImpl(workLocal, newTiling, softmaxMaxUb, softmaxSumUb, src, dst);
    } else {
        CustomSoftmaxComputeImpl(workLocal, tiling, softmaxMaxUb, softmaxSumUb, src, dst);
    }

    inputQue.FreeTensor(src);
}

template <typename INPUT_T, typename PROCESS_T = INPUT_T>
class FlashAttentionScoreSplitBn2gs1S1s2d
{
   public:
    __aicore__ inline FlashAttentionScoreSplitBn2gs1S1s2d(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
                                __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask,
                                __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut,
                                __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe);

    __aicore__ inline void Process();
    __aicore__ inline void ProcessStage1OnS2(uint32_t actualS1BaseSize, uint64_t offsetQ, uint64_t pseOffsetWithoutS2,
                                             uint64_t kOffsetWithoutS2D, uint64_t attenMaskOffsetWithoutS2,
                                             uint64_t idxS1base);
    __aicore__ inline void ProcessStage2OnS2(uint32_t actualS1BaseSize, int dropmaskOffsetWithoutS2,
                                             int maxSumOffsetWithoutS2D);
    __aicore__ inline void ProcessStage3OnS2(uint64_t kvOffsetWithoutS2D, uint64_t qOffsetWithoutD,
                                             uint32_t actualS1BaseSize);

    __aicore__ inline void PseCopyIn(uint64_t offset, uint32_t s1Size, uint32_t s2Size,
                                     TQue<QuePosition::VECIN, 1> &inputQue);
    __aicore__ inline void PseComputeInplace(LocalTensor<PROCESS_T> &res, uint32_t s1Size, uint32_t s2Size,
                                             TQue<QuePosition::VECIN, 1> &inputQue);
    __aicore__ inline void DropMaskCopyIn(uint64_t offset, uint32_t s1Size, uint32_t s2Size,
                                          TQue<QuePosition::VECIN, 1> &inputQue);
    __aicore__ inline void DropoutComputeInplace(LocalTensor<PROCESS_T> &res, uint32_t s1Size, uint32_t s2Size,
                                                 TQue<QuePosition::VECIN, 1> &inputQue, LocalTensor<uint8_t> &apiBuf);
    __aicore__ inline void MaskCopyIn(TQue<QuePosition::VECIN, 1> &inputQue, uint64_t offset, uint32_t s1Size,
                                      uint32_t s2Size);
    __aicore__ inline void MaskComputeInplace(TQue<QuePosition::VECIN, 1> &inputQue, LocalTensor<PROCESS_T> &src,
                                              uint32_t s1Size, uint32_t s2Size);

    // 基础API
    __aicore__ inline void InnerBroadcastAdd(uint32_t repeatTimes, uint32_t outerOffset, uint32_t innerLoop,
                                             uint32_t innerRemain, const LocalTensor<PROCESS_T> &src1,
                                             const LocalTensor<PROCESS_T> &src0, uint32_t stepCalcNum,
                                             const BinaryRepeatParams &repParams);
    __aicore__ inline void BroadcastAddOnSrc0(uint32_t s1Size, uint32_t s2Size, const LocalTensor<PROCESS_T> &src1,
                                              const LocalTensor<PROCESS_T> &src0);
    __aicore__ inline void ComputePseRepParams(uint32_t innerSize, BinaryRepeatParams &repParam);

    /* Offset */
    __aicore__ inline void OffsetWithoutS2D(uint64_t idxBCore, uint64_t idxN2Core, uint64_t idxGCore,
                                            uint64_t idxS1Base, uint64_t &offsetQ, uint64_t &offsetKV,
                                            uint64_t &offsetDropmask, uint64_t &offsetMaxSum, uint64_t &offsetAttenMask,
                                            uint64_t &offsetPse);
    __aicore__ inline void CalcOffsetWithoutDInStage2(uint64_t idxS2Outer, uint64_t dropmaskOffsetWithoutS2,
                                                      uint64_t &dropmaskOffset);
    __aicore__ inline uint64_t ComputeOffsetForCausal(const int64_t &delta, const uint64_t &s1BaseSize,
                                                      const uint64_t &s2BaseSize, const uint32_t &attenMaskS2Size);
    __aicore__ inline void CalcOffsetInStage1(uint64_t idxS1Base, uint64_t idxS2Base, uint64_t pseOffsetWithoutS2,
                                              uint64_t attenMaskOffsetWithoutS2, uint64_t &pseOffset,
                                              uint64_t &attenMaskOffset);
    __aicore__ inline void CalcOffsetWithoutS2nratioD(uint64_t idxS2Outer, uint64_t kOffsetWithoutS2D,
                                                      uint64_t &kOffsetWithoutS2nratioD);
    __aicore__ inline void CalcRangeOuter();
    __aicore__ inline void CalcRangeS2Outer(uint32_t idxS1Base, uint32_t idxBCore);
    __aicore__ inline void UpdateOffsetV(uint64_t &kvOffsetWithoutS2D);

    /* resources */
    using a1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T>;
    using b1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, true>;
    using bias1Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c1Type = MatmulType<TPosition::VECCALC, CubeFormat::ND_ALIGN, PROCESS_T>;
    matmul::Matmul<a1Type, b1Type, c1Type, bias1Type, CFG_EXCEED> bmm1;

    using a2Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T>;
    using b2Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T>;
    using bias2Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c2Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T>;
    matmul::Matmul<a2Type, b2Type, c2Type, bias2Type, CFG_EXCEED> bmm2;

    TQue<QuePosition::VECIN, 1> inputQue;

    TBuf<> tmpBuf;
    LocalTensor<INPUT_T> mm1Res;
    LocalTensor<INPUT_T> pseCastRes;
    LocalTensor<bool> attenMaskApiBuf;
    LocalTensor<float> softmaxMaxUb;
    LocalTensor<float> softmaxSumUb;
    LocalTensor<float> maxSumApiBuf;
    LocalTensor<float> customSoftmaxApiBuf;
    LocalTensor<uint8_t> dropoutApiBuf;
    LocalTensor<PROCESS_T> bmm1ProccessTInStage2;

    TQue<QuePosition::VECOUT, 1> outputQue;
    TQue<QuePosition::VECOUT, SOFTMAX_MAX_SUM_BUFFER_NUM> outputSoftmaxQue;

    /* member */
    TPipe *pipe;
    int64_t currBlockIdx;
    const FlashAttentionScoreGeneralTilingData *__restrict t;

    int64_t startOuter;
    int64_t endOuter;
    int64_t startS2Outer;
    int64_t endS2Outer;

    const InputParams *inputParams;
    const CoreParams *coreParams;
    const MultiCoreParams *multiCoreParams;
    const TensorSizeParams *tensorSizeParams;

    int64_t idxBCore;
    int64_t idxN2Core;
    int64_t idxGCore;
    int64_t idxS1Base;

    int64_t numS2BlocksInputT;
    int64_t numS2BlocksProcessT;
    int64_t numS2Base;
    int64_t actualS2Size;

    int64_t queryOffsetWithoutS1;
    int64_t keyOffsetWithoutS1;
    int64_t valueCoreOffset;

    GlobalTensor<INPUT_T> queryGm;
    GlobalTensor<INPUT_T> keyGm;
    GlobalTensor<INPUT_T> valueGm;
    GlobalTensor<INPUT_T> pseGm;
    GlobalTensor<uint8_t> dropMaskGm;
    GlobalTensor<bool> attenMaskGmBool;
    GlobalTensor<INPUT_T> attenMaskGmInputT;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    GlobalTensor<INPUT_T> attentionOutGm;
    GlobalTensor<PROCESS_T> workspaceNRatio;
    GlobalTensor<PROCESS_T> workspaceGmProcessT;
    GlobalTensor<INPUT_T> workspaceGmInputT;

    uint32_t workspaceGmProcessTBaseOffsetBytes = 0;
    uint32_t workspaceGmInputTBaseOffsetBytes = 0;

    bool hasPse = false;
    bool hasDropout = false;
    bool hasAttenMask = false;

    // sparse
    bool isSparse = false;
    int64_t minS1S2 = 0;

    // reduce scalar
    uint32_t nS2BaseSize = 0;
    int64_t n1Size = 0;

    uint32_t attenMaskNoCompress = 0;
    uint32_t attenMaskLeftUpCausalCompress = 1;
    uint32_t attenMaskRightDownCausalCompress = 2;
};

template <typename INPUT_T, typename PROCESS_T>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1s2d<INPUT_T, PROCESS_T>::ComputePseRepParams(
    uint32_t innerSize, BinaryRepeatParams &repParam)
{
    repParam.src0BlkStride = 1;
    repParam.src0RepStride = innerSize / (BLOCK_BYTES / sizeof(PROCESS_T));
    repParam.src1BlkStride = 1;
    repParam.src1RepStride = 0;
    repParam.dstRepStride = repParam.src0RepStride;
    repParam.blockNumber = repParam.src0RepStride;
}

template <typename INPUT_T, typename PROCESS_T>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1s2d<INPUT_T, PROCESS_T>::InnerBroadcastAdd(
    uint32_t repeatTimes, uint32_t outerOffset, uint32_t innerLoop, uint32_t innerRemain,
    const LocalTensor<PROCESS_T> &src1, const LocalTensor<PROCESS_T> &src0, uint32_t stepCalcNum,
    const BinaryRepeatParams &repParams)
{
    for (uint32_t j = 0; j < innerLoop; j++) {
        auto innerOffset = j * stepCalcNum;
        auto ubOffset = outerOffset + innerOffset;
        Add(src0[ubOffset], src0[ubOffset], src1[innerOffset], stepCalcNum, repeatTimes, repParams);
    }
    if (innerRemain > 0) {
        auto innerOffset = innerLoop * stepCalcNum;
        auto ubOffset = outerOffset + innerOffset;
        Add(src0[ubOffset], src0[ubOffset], src1[innerOffset], innerRemain, repeatTimes, repParams);
    }
}

template <typename INPUT_T, typename PROCESS_T>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1s2d<INPUT_T, PROCESS_T>::BroadcastAddOnSrc0(
    uint32_t s1Size, uint32_t s2Size, const LocalTensor<PROCESS_T> &src1, const LocalTensor<PROCESS_T> &src0)
{
    if (inputParams->pseShapeType == 0) {
        Add(src0, src0, src1, s1Size * s2Size);
    } else {
        uint32_t stepCalcNum = 256 / sizeof(PROCESS_T);
        /* Total data number of single step should be smaller than 256bytes.
         * If larger, we need to do add multiple times. */
        stepCalcNum = s2Size > stepCalcNum ? stepCalcNum : s2Size;
        uint32_t innerLoop = s2Size / stepCalcNum;
        uint32_t innerRemain = s2Size % stepCalcNum;
        BinaryRepeatParams repParams;
        ComputePseRepParams(s2Size, repParams);
        /* Total repeated times should be <= 255. If larger, we need to do multiple inner loops. */
        uint32_t outerLoop = s1Size / 255;
        uint32_t outerRemain = s1Size % 255;
        for (uint32_t i = 0; i < outerLoop; i++) {
            uint32_t outerOffset = (i >> 8) * s2Size;
            InnerBroadcastAdd(255, outerOffset, innerLoop, innerRemain, src1, src0, stepCalcNum, repParams);
        }
        if (outerRemain > 0) {
            uint32_t outerOffset = (outerLoop >> 8) * s2Size;
            InnerBroadcastAdd(outerRemain, outerOffset, innerLoop, innerRemain, src1, src0, stepCalcNum, repParams);
        }
    }
}

template <typename INPUT_T, typename PROCESS_T>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1s2d<INPUT_T, PROCESS_T>::DropMaskCopyIn(
    uint64_t offset, uint32_t s1Size, uint32_t s2Size, TQue<QuePosition::VECIN, 1> &inputQue)
{
    if (!hasDropout) {
        SHORT_MIX_LOG("[Status] no dropMask");
        return;
    }
    SHORT_MIX_LOG("[Status] has dropMask");
    uint32_t s2SizeAligned = math::Align(s2Size, (uint32_t)(BLOCK_BYTES / sizeof(uint8_t)));
    LocalTensor<uint8_t> dropMaskUb = inputQue.AllocTensor<uint8_t>();
    uint32_t shapeArray[] = {s1Size, s2SizeAligned};
    dropMaskUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    dropMaskUb.SetSize(s1Size * s2SizeAligned);
    DataCopyParams intriParams;
    intriParams.blockCount = s1Size;
    intriParams.blockLen = math::Ceil(s2SizeAligned / NUM_BITS_IN_BYTE, BLOCK_BYTES);
    intriParams.dstStride = 0;
    if (inputParams->s2Size / NUM_BITS_IN_BYTE % BLOCK_BYTES == 0) {
        intriParams.srcStride =
            (inputParams->s2Size / NUM_BITS_IN_BYTE - intriParams.blockLen * BLOCK_BYTES) / BLOCK_BYTES;
        DataCopy(dropMaskUb, dropMaskGm[offset / NUM_BITS_IN_BYTE], intriParams);
    } else {
        intriParams.blockLen = s2Size / NUM_BITS_IN_BYTE;
        intriParams.srcStride = (inputParams->s2Size / NUM_BITS_IN_BYTE - intriParams.blockLen);
        DataCopyPadParams intriParams1;
        DataCopyPad(dropMaskUb, dropMaskGm[offset / NUM_BITS_IN_BYTE], intriParams, intriParams1);
    }
    PRINT_DATA(dropMaskUb, 0, 1, 8, "[Data][IN] stage2 dropMask");
    inputQue.EnQue(dropMaskUb);
}

template <typename INPUT_T, typename PROCESS_T>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1s2d<INPUT_T, PROCESS_T>::DropoutComputeInplace(
    LocalTensor<PROCESS_T> &src, uint32_t s1Size, uint32_t s2Size, TQue<QuePosition::VECIN, 1> &inputQue,
    LocalTensor<uint8_t> &apiBuf)
{
    if (!hasDropout) {
        SHORT_MIX_LOG("[Status] no dropout");
        return;
    }
    SHORT_MIX_LOG("[Status] has dropout");

    LocalTensor<uint8_t> dropMaskUb = inputQue.DeQue<uint8_t>();
    DropOutShapeInfo dropOutShapeInfo;
    dropOutShapeInfo.firstAxis = s1Size;
    dropOutShapeInfo.srcLastAxis = math::Align(s2Size, (uint32_t)(BLOCK_BYTES / sizeof(PROCESS_T)));
    dropOutShapeInfo.maskLastAxis = math::Align(s2Size / NUM_BITS_IN_BYTE, BLOCK_BYTES);
    if (hasAttenMask && inputParams->attenMaskDataType == 0) {
        // INPUT_T
        SHORT_MIX_LOG("[ERROR] not impl");
    } else {
        // uint8_t
        DropOut(src, src, dropMaskUb, apiBuf, inputParams->keepProb, dropOutShapeInfo);
    }
    inputQue.FreeTensor(dropMaskUb);
}

template <typename T>
__aicore__ inline void MaxSumComputeHelper(const LocalTensor<T> &src, LocalTensor<float> &dstMax,
                                           LocalTensor<float> &dstSum, const LocalTensor<float> &workLocal,
                                           const LastAxisShapeND &originalSrcShape, const uint32_t &offset1,
                                           const uint32_t &offset2, const uint32_t &splitSize,
                                           const uint32_t &reduceSize, const ReduceLastND &reduceParam,
                                           const BroadCastLastND &brcParam)
{
    if constexpr (sizeof(T) == sizeof(half)) {
        const LocalTensor<float> &tmpBuffer0 = workLocal;
        const LocalTensor<float> &tmpBuffer1 = workLocal[splitSize];
        const LocalTensor<float> &tmpBuffer2 = workLocal[splitSize + splitSize];

        Cast(tmpBuffer0, src[offset1], RoundMode::CAST_NONE, splitSize);
        pipe_barrier(PIPE_V);
        ReduceMaxLastNDImpl(dstMax[offset2], tmpBuffer0, tmpBuffer2, reduceParam);
        pipe_barrier(PIPE_V);
        BroadCastLastImpl(tmpBuffer1, dstMax[offset2], brcParam);
        pipe_barrier(PIPE_V);
        Sub(tmpBuffer1, tmpBuffer0, tmpBuffer1, splitSize);  // x - max
        pipe_barrier(PIPE_V);
        Exp(tmpBuffer1, tmpBuffer1, splitSize);  // exp(x - max)
        pipe_barrier(PIPE_V);
        ReduceSumLastNDImpl(dstSum[offset2], tmpBuffer1, tmpBuffer2, reduceParam);
        pipe_barrier(PIPE_V);
    } else {
        const LocalTensor<float> &tmpBuffer1 = workLocal;
        const LocalTensor<float> &tmpBuffer2 = workLocal[splitSize];

        ReduceMaxLastNDImpl(dstMax[offset2], src[offset1], tmpBuffer2, reduceParam);
        pipe_barrier(PIPE_V);
        BroadCastLastImpl(tmpBuffer1, dstMax[offset2], brcParam);
        pipe_barrier(PIPE_V);
        Sub(tmpBuffer1, src[offset1], tmpBuffer1, splitSize);  // x - max
        pipe_barrier(PIPE_V);
        Exp(tmpBuffer1, tmpBuffer1, splitSize);  // exp(x - max)
        pipe_barrier(PIPE_V);
        ReduceSumLastNDImpl(dstSum[offset2], tmpBuffer1, tmpBuffer2, reduceParam);
        pipe_barrier(PIPE_V);
    }
}

template <typename T>
__aicore__ inline void MaxSumComputeImpl(LocalTensor<float> &workLocal, const MaxSumTiling &tiling,
                                         LocalTensor<float> &dstMax, LocalTensor<float> &dstSum, LocalTensor<T> &src,
                                         const LastAxisShapeND &originalSrcShape)
{
    uint32_t offset1 = 0;
    uint32_t offset2 = 0;

    ReduceLastND reduceParam = {tiling.splitM, originalSrcShape.k, tiling.splitM,
                                tiling.splitK, tiling.reduceM,     tiling.reduceK};
    BroadCastLastND brcParam = {tiling.splitM, tiling.splitK, tiling.reduceM, tiling.reduceK};

    for (uint32_t i = 0; i < tiling.rangeM; i++) {
        offset2 = i * tiling.reduceSize;
        offset1 = i * tiling.splitSize;
        MaxSumComputeHelper<T>(src, dstMax, dstSum, workLocal, originalSrcShape, offset1, offset2, tiling.splitSize,
                               tiling.reduceSize, reduceParam, brcParam);
    }
    if (tiling.tailM != 0) {
        reduceParam.originalSrcM = tiling.tailM;
        reduceParam.srcM = tiling.tailM;
        reduceParam.dstM = tiling.tailM;
        brcParam.dstM = tiling.tailM;
        brcParam.srcM = tiling.tailM;

        offset2 = tiling.rangeM * tiling.reduceSize;
        offset1 = tiling.rangeM * tiling.splitSize;
        MaxSumComputeHelper<T>(src, dstMax, dstSum, workLocal, originalSrcShape, offset1, offset2, tiling.tailSplitSize,
                               tiling.tailReduceSize, reduceParam, brcParam);
    }
}

template <typename T>
__aicore__ inline void MaxSumCompute(LocalTensor<T> &src, LocalTensor<float> &dstMax, LocalTensor<float> &dstSum,
                                     LocalTensor<float> &workLocal, const MaxSumTiling &tiling)
{
    LastAxisShapeND originalSrcShape = GetLastAxisOriginShapeND(src.GetShapeInfo());

    if (originalSrcShape.m != tiling.srcM || originalSrcShape.k != tiling.srcK) {
        MaxSumTiling newTiling = tiling;
        if (IsSameType<T, float>::value) {
            MaxSumTilingFunc(workLocal.GetSize(), originalSrcShape, newTiling, BLOCK_BYTES / sizeof(float));
        } else {
            MaxSumNeedCastTilingFunc(workLocal.GetSize(), originalSrcShape, newTiling, BLOCK_BYTES / sizeof(float));
        }
        MaxSumComputeImpl(workLocal, newTiling, dstMax, dstSum, src, originalSrcShape);
    } else {
        MaxSumComputeImpl(workLocal, tiling, dstMax, dstSum, src, originalSrcShape);
    }
}

template <typename INPUT_T, typename PROCESS_T>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1s2d<INPUT_T, PROCESS_T>::PseCopyIn(
    uint64_t offset, uint32_t s1Size, uint32_t s2Size, TQue<QuePosition::VECIN, 1> &inputQue)
{
    if (!hasPse) {
        SHORT_MIX_LOG("[Status] no pse");
        return;
    }
    SHORT_MIX_LOG("[Status] has pse");
    // B, N2, G, S1, S2
    // B, N2, G,  1, S2
    LocalTensor<INPUT_T> pseUb = inputQue.AllocTensor<INPUT_T>();
    DataCopyParams intriParams;
    intriParams.blockCount = s1Size;
    if (inputParams->pseShapeType == 1) {
        intriParams.blockCount = 1;
    }
    intriParams.blockLen = s2Size * sizeof(INPUT_T);
    intriParams.srcStride = inputParams->s2Size * sizeof(INPUT_T) - intriParams.blockLen;
    intriParams.dstStride = 0;

    DataCopyPadParams padParams;
    if (s2Size % (BLOCK_BYTES / sizeof(INPUT_T)) != 0) {
        padParams.isPad = true;
        padParams.rightPadding = BLOCK_BYTES / sizeof(INPUT_T) - (s2Size % (BLOCK_BYTES / sizeof(INPUT_T)));
    }

    SHORT_MIX_LOG(
        "[GmOffset] pse offset %d blockCount %d blockLen %d srcStride %d dstStride %d isPad %d rightPadding %d", offset,
        intriParams.blockCount, intriParams.blockLen, intriParams.srcStride, intriParams.dstStride, padParams.isPad,
        padParams.rightPadding);
    DataCopyPad(pseUb, pseGm[offset], intriParams, padParams);
    inputQue.EnQue(pseUb);
}

template <typename INPUT_T, typename PROCESS_T>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1s2d<INPUT_T, PROCESS_T>::PseComputeInplace(
    LocalTensor<PROCESS_T> &src, uint32_t s1Size, uint32_t s2Size, TQue<QuePosition::VECIN, 1> &inputQue)
{
    if (!hasPse) {
        SHORT_MIX_LOG("[Status] no pse");
        return;
    }
    SHORT_MIX_LOG("[Status] has pse");
    LocalTensor<INPUT_T> pseUb = inputQue.DeQue<INPUT_T>();
    if constexpr (!IsSameType<INPUT_T, PROCESS_T>::value) {
        LocalTensor<PROCESS_T> tmpPseCastTensor =
            tmpBuf.GetWithOffset<PROCESS_T>(tensorSizeParams->pseUbSize, tensorSizeParams->pseTbufOffsetBytes);
        Cast(tmpPseCastTensor, pseUb, RoundMode::CAST_NONE, s1Size * s2Size);
        BroadcastAddOnSrc0(s1Size, s2Size, tmpPseCastTensor, src);
    } else {
        BroadcastAddOnSrc0(s1Size, s2Size, pseUb, src);
    }
    inputQue.FreeTensor(pseUb);
}

template <typename INPUT_T, typename PROCESS_T>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1s2d<INPUT_T, PROCESS_T>::MaskCopyIn(
    TQue<QuePosition::VECIN, 1> &inputQue, uint64_t offset, uint32_t s1Size, uint32_t s2Size)
{
    if (!hasAttenMask) {
        SHORT_MIX_LOG("[Status] no atten mask");
        return;
    }
    SHORT_MIX_LOG("[Status] has atten mask");

    if (inputParams->attenMaskDataType == 0) {
        // INPUT_T
        SHORT_MIX_LOG("[ERROR] not impl");
    } else {
        // bool
        LocalTensor<bool> attenMaskUb = inputQue.AllocTensor<bool>();
        uint32_t alignedS2Size = math::Align(s2Size, BLOCK_BYTES);
        uint32_t shapeArray[] = {s1Size, alignedS2Size};
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = s1Size;
        dataCopyParams.blockLen = s2Size;
        dataCopyParams.srcStride = inputParams->attenMaskS2Size - s2Size;
        dataCopyParams.dstStride = 0;
        DataCopyPadParams dataCopyPadParams;
        if (s2Size != alignedS2Size) {
            dataCopyPadParams.isPad = true;
            dataCopyPadParams.rightPadding = alignedS2Size - s2Size;
            dataCopyPadParams.paddingValue = 1;
        }
        DataCopyPad(attenMaskUb, attenMaskGmBool[offset], dataCopyParams, dataCopyPadParams);
        inputQue.EnQue(attenMaskUb);
    }
}

template <typename INPUT_T, typename PROCESS_T>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1s2d<INPUT_T, PROCESS_T>::MaskComputeInplace(
    TQue<QuePosition::VECIN, 1> &inputQue, LocalTensor<PROCESS_T> &src, uint32_t s1Size, uint32_t s2Size)
{
    if (!hasAttenMask) {
        SHORT_MIX_LOG("[Status] no atten mask");
        return;
    }
    SHORT_MIX_LOG("[Status] has atten mask");

    if (inputParams->attenMaskDataType == 0) {
        SHORT_MIX_LOG("[ERROR] not impl");
    } else {
        // bool
        LocalTensor<bool> attenMaskUb = inputQue.template DeQue<bool>();
        uint32_t alignedS2Size = math::Align(s2Size, BLOCK_BYTES);
        uint32_t shapeArray[] = {s1Size, alignedS2Size};
        attenMaskUb.SetSize(s1Size * alignedS2Size);
        src.SetSize(s1Size * alignedS2Size);

        SelectWithBytesMaskShapeInfo shapeInfo;
        shapeInfo.firstAxis = s1Size;
        shapeInfo.srcLastAxis = alignedS2Size;
        shapeInfo.maskLastAxis = alignedS2Size;
        if constexpr (IsSameType<PROCESS_T, float>::value) {
            uint32_t tmp = 0xFF7FFFFF;
            SelectWithBytesMask(src, src, *((float *)&tmp), attenMaskUb, attenMaskApiBuf, shapeInfo);
        } else {
            uint16_t tmp = 0xFBFF;
            SelectWithBytesMask(src, src, *((half *)&tmp), attenMaskUb, attenMaskApiBuf, shapeInfo);
        }

        inputQue.FreeTensor(attenMaskUb);
    }
}

template <typename INPUT_T, typename PROCESS_T>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1s2d<INPUT_T, PROCESS_T>::CalcRangeOuter() {
    if (inputParams->sparseType > 0) {
        // sparse场景下负载均衡后每个核获取的结果
        startOuter = multiCoreParams->sparseStartIdx[currBlockIdx];
        if (likely((multiCoreParams->coreNum - 1) > currBlockIdx)) {
            endOuter = multiCoreParams->sparseStartIdx[currBlockIdx + 1];
        } else {
            endOuter = multiCoreParams->totalSize;
        }
    } else {
        startOuter = currBlockIdx * multiCoreParams->splitFactorSize;
        endOuter = math::Min(multiCoreParams->totalSize, startOuter + multiCoreParams->splitFactorSize);
    }
    SHORT_MIX_LOG("[DEBUG]current : startOuter %ld endOuter %ld", startOuter, endOuter);
    SHORT_MIX_LOG("[DEBUG]original: startOuter %ld endOuter %ld", currBlockIdx * multiCoreParams->splitFactorSize,
            math::Min(multiCoreParams->totalSize, startOuter + multiCoreParams->splitFactorSize));
}

template <typename INPUT_T, typename PROCESS_T>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1s2d<INPUT_T, PROCESS_T>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse, __gm__ uint8_t *dropMask,
    __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
    __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const FlashAttentionScoreGeneralTilingData *__restrict tiling, TPipe *tPipe)
{
    currBlockIdx = GetBlockIdx();
    t = tiling;
    inputParams = &(t->inputParams);
    coreParams = &(t->coreParams);
    multiCoreParams = &(t->multiCoreParams);
    tensorSizeParams = &(t->tensorSizeParams);

    queryGm.SetGlobalBuffer((__gm__ INPUT_T *)query);
    keyGm.SetGlobalBuffer((__gm__ INPUT_T *)key);
    valueGm.SetGlobalBuffer((__gm__ INPUT_T *)value);
    pseGm.SetGlobalBuffer((__gm__ INPUT_T *)pse);
    dropMaskGm.SetGlobalBuffer((__gm__ uint8_t *)dropMask);
    // workspace
    // mm1 n比1 * coreNum
    // stage1 mm1 copy out / stage copy out (float16/float32) * coreNum
    // stage2 copy out (bfloat16/float16 高精度) * coreNum
    workspaceNRatio.SetGlobalBuffer((__gm__ PROCESS_T *)workspace);
    int64_t workspaceGmProcessTBytes =
        coreParams->s1BaseSize * math::Align((int64_t)(inputParams->s2Size * sizeof(PROCESS_T)), (int64_t)BLOCK_BYTES);
    workspaceGmProcessTBaseOffsetBytes = (uint32_t)(currBlockIdx * workspaceGmProcessTBytes);
    workspaceGmProcessT.SetGlobalBuffer((__gm__ PROCESS_T *)(workspace + tensorSizeParams->wkspSection1OffsetBytes +
                                                             workspaceGmProcessTBaseOffsetBytes));

    if constexpr (!IsSameType<INPUT_T, PROCESS_T>::value) {
        int64_t workspaceGmInputTBytes =
            coreParams->s1BaseSize *
            math::Align((int64_t)(inputParams->s2Size * sizeof(INPUT_T)), (int64_t)BLOCK_BYTES);
        workspaceGmInputTBaseOffsetBytes = (uint32_t)(currBlockIdx * workspaceGmInputTBytes);
        workspaceGmInputT.SetGlobalBuffer((__gm__ INPUT_T *)(workspace + tensorSizeParams->wkspSection2OffsetBytes +
                                                             workspaceGmInputTBaseOffsetBytes));
    }

    softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmaxMax);
    softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmaxSum);
    attentionOutGm.SetGlobalBuffer((__gm__ INPUT_T *)attentionOut);

    /* 分配buffer空间 */
    pipe = tPipe;
    SHORT_MIX_LOG("[DEBUG] tmpBuf %d inputQue %d outputQue %d outputSoftmaxQue %d", tensorSizeParams->tmpBufBytes,
            tensorSizeParams->inputQueBytes, tensorSizeParams->outputQueBytes,
            tensorSizeParams->softmaxMaxUbSize * sizeof(float) * 2);
    pipe->InitBuffer(tmpBuf, tensorSizeParams->tmpBufBytes);
    pipe->InitBuffer(inputQue, 1, tensorSizeParams->inputQueBytes);
    pipe->InitBuffer(outputQue, 1, tensorSizeParams->outputQueBytes);
    pipe->InitBuffer(outputSoftmaxQue, 2, tensorSizeParams->softmaxMaxUbSize * sizeof(float));

    auto workspaceOffsetSizePerBlock = currBlockIdx * coreParams->nRatio * tensorSizeParams->bmm1ResUbSize;
    bmm1.SetWorkspace((__gm__ uint8_t *)workspaceNRatio[workspaceOffsetSizePerBlock].GetPhyAddr(),
                      tensorSizeParams->bmm1ResUbSize * coreParams->nRatio * sizeof(PROCESS_T));
    SHORT_MIX_LOG("[DEBUG] bmm1.SetWrokspace offset %d size %d*%d", workspaceOffsetSizePerBlock,
            tensorSizeParams->bmm1ResUbSize * coreParams->nRatio * sizeof(PROCESS_T));

    maxSumApiBuf = tmpBuf.GetWithOffset<float>(tensorSizeParams->maxSumApiSize, tensorSizeParams->maxSumApiOffsetBytes);
    customSoftmaxApiBuf = tmpBuf.GetWithOffset<float>(tensorSizeParams->customSoftmaxApiSize,
                                                      tensorSizeParams->customSoftmaxApiOffsetBytes);
    dropoutApiBuf =
        tmpBuf.GetWithOffset<uint8_t>(tensorSizeParams->dropoutApiSize, tensorSizeParams->dropoutApiOffsetBytes);
    DropOutBitModeInit(dropoutApiBuf);

    if constexpr (!IsSameType<INPUT_T, PROCESS_T>::value) {
        bmm1ProccessTInStage2 = tmpBuf.GetWithOffset<PROCESS_T>(tensorSizeParams->bmm1ProcessTInStage2Size,
                                                                tensorSizeParams->bmm1ProcessTInStage2OffsetBytes);
    }

    numS2BlocksProcessT = math::Ceil((int64_t)(inputParams->s2Size * sizeof(PROCESS_T)), (int64_t)BLOCK_BYTES);
    numS2BlocksInputT = math::Ceil((int64_t)(inputParams->s2Size * sizeof(INPUT_T)), (int64_t)BLOCK_BYTES);
    numS2Base = math::Ceil(inputParams->s2Size, (int64_t)coreParams->s2BaseSize);

    hasPse = tensorSizeParams->pseUbSize > 0;
    hasDropout = tensorSizeParams->dropMaskUbSize > 0;
    hasAttenMask = tensorSizeParams->attenMaskUbSize > 0;

    if (hasAttenMask) {
        attenMaskApiBuf = tmpBuf.GetWithOffset<bool>(tensorSizeParams->attenMaskApiSize,
                                                     tensorSizeParams->attenMaskApiOffsetBytes);
        if (inputParams->attenMaskDataType == 0) {
            // INPUT_T
            SHORT_MIX_LOG("[ERROR] not impl");
        } else {
            // bool
            attenMaskGmBool.SetGlobalBuffer((__gm__ bool *)attenMask);
        }
    }

    // sparse
    minS1S2 = math::Min(inputParams->s1Size, inputParams->s2Size);
    actualS2Size = inputParams->s2Size;
    isSparse = inputParams->sparseType == 3 || inputParams->sparseType == 4 ||
        inputParams->sparseType == 5;

    // reduce scalar
    nS2BaseSize = coreParams->nRatio * coreParams->s2BaseSize;
    n1Size = inputParams->n2Size * inputParams->gSize;
}

template <typename INPUT_T, typename PROCESS_T>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1s2d<INPUT_T, PROCESS_T>::CalcRangeS2Outer(uint32_t idxS1Base,
                                                                                                 uint32_t idxBCore)
{
// NOTE 只支持S1等于S2
    // 计算S2的循环范围相关参数: 后续可 使用static_cast<uint32_t>优化scale性能
    if (inputParams->sparseType == 3 || inputParams->sparseType == 5) {
        actualS2Size = math::Min((int64_t)((idxS1Base + 1) * coreParams->s1BaseSize), minS1S2);
        if (t->inputParams.sparseType == 5) {
            actualS2Size = math::Max(actualS2Size, t->inputParams.prefixNList[idxBCore]);
            actualS2Size = math::Align(actualS2Size, (int64_t)t->coreParams.s1BaseSize);
            actualS2Size = math::Min(actualS2Size, t->inputParams.s2Size);
        }
        startS2Outer = 0;
        endS2Outer = math::Ceil(actualS2Size, (int64_t)coreParams->s2BaseSize * coreParams->nRatio);
        numS2Base = math::Ceil(actualS2Size, (int64_t)coreParams->s2BaseSize);
    } else if (inputParams->sparseType == 4) {
        actualS2Size = t->coreParams.s1BaseSize * (idxS1Base + 1) + t->inputParams.nextTokens;
        actualS2Size = math::Align(actualS2Size, (int64_t)t->coreParams.s1BaseSize);
        actualS2Size = math::Min(actualS2Size, t->inputParams.s2Size);
        endS2Outer = Ceil(actualS2Size, t->coreParams.s2BaseSize * t->coreParams.nRatio);
        // band场景需要额外计算起始的index
        int64_t s2IgnoredStartLen = t->coreParams.s1BaseSize * idxS1Base -
                                    t->inputParams.preTokens;
        s2IgnoredStartLen = math::Max(s2IgnoredStartLen, static_cast<int64_t>(0));
        startS2Outer = s2IgnoredStartLen / (t->coreParams.s2BaseSize * t->coreParams.nRatio);
        numS2Base = math::Ceil(actualS2Size, (int64_t)coreParams->s2BaseSize);
    } else {  // 其它场景, 如无attention mask
        startS2Outer = 0;
        endS2Outer = coreParams->s2OuterSize;
    }
    SHORT_MIX_LOG("[DEBUG]current : startS2Outer %ld endS2Outer %ld numS2Base %ld actualS2Size %ld", startS2Outer,
                  endS2Outer, numS2Base, actualS2Size);
    SHORT_MIX_LOG("[DEBUG]original: startS2Outer %ld endS2Outer %ld numS2Base %ld actualS2Size %ld", 0,
                  coreParams->s2OuterSize, math::Ceil(inputParams->s2Size, (int64_t)coreParams->s2BaseSize),
                  inputParams->s2Size);
}

template <typename INPUT_T, typename PROCESS_T>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1s2d<INPUT_T, PROCESS_T>::ProcessStage1OnS2(
    uint32_t actualS1BaseSize, uint64_t offsetQ, uint64_t pseOffsetWithoutS2, uint64_t kOffsetWithoutS2D,
    uint64_t attenMaskOffsetWithoutS2, uint64_t idxS1base)
{
    uint64_t kOffsetWithoutS2nratioD = 0;
    uint64_t pseOffset = 0;
    uint64_t attenMaskOffset = 0;

    bool needUpdateMaxSum = false;
    for (uint64_t idxS2Outer = startS2Outer; idxS2Outer < endS2Outer; ++idxS2Outer) {
        auto startS2Base = idxS2Outer * coreParams->nRatio;
        auto endS2Base = math::Min(startS2Base + coreParams->nRatio, (uint64_t)numS2Base);

        CalcOffsetWithoutS2nratioD(idxS2Outer, kOffsetWithoutS2D, kOffsetWithoutS2nratioD);
        // B,N2,G,S1,D
        bmm1.SetTensorA(queryGm[offsetQ]);
        PRINT_DATA(queryGm[offsetQ], 0, 1, 8, "[Data] bmm1 tensor a");
        // B,N2,1,S2,D
        bmm1.SetTensorB(keyGm[kOffsetWithoutS2nratioD], true);
        PRINT_DATA(keyGm[kOffsetWithoutS2nratioD], 0, 1, 8, "[Data] bmm1 tensor b");
        bmm1.SetSingleShape(actualS1BaseSize,
                            math::Min(coreParams->s2BaseSize * (endS2Base - startS2Base),
                                      actualS2Size - coreParams->s2BaseSize * startS2Base),
                            -1);
        SHORT_MIX_LOG("[GmOffset] bmm1 AOffset %d BOffset %d SetSingleShape (%d,%d)", offsetQ, kOffsetWithoutS2nratioD,
                      actualS1BaseSize,
                      math::Min(coreParams->s2BaseSize * (endS2Base - startS2Base),
                                actualS2Size - coreParams->s2BaseSize * startS2Base));

        for (uint64_t idxS2Base = startS2Base; idxS2Base < endS2Base; ++idxS2Base) {
            CalcOffsetInStage1(idxS1Base, idxS2Base, pseOffsetWithoutS2, attenMaskOffsetWithoutS2, pseOffset, attenMaskOffset);
            bmm1.template Iterate<true>();

            uint32_t actualS2BaseSize = (uint32_t)math::Min((uint64_t)coreParams->s2BaseSize,
                                                            actualS2Size - idxS2Base * coreParams->s2BaseSize);
            uint32_t actualS2BaseSizeAligned =
                math::Align(actualS2BaseSize, (uint32_t)(BLOCK_BYTES / sizeof(PROCESS_T)));
            uint32_t shapeArrayMm1ResUb[] = {actualS1BaseSize, actualS2BaseSizeAligned};
            SHORT_MIX_LOG("[Index] stage1 s2 index %d range [%d, %d)", idxS2Base, idxS2Base * coreParams->s2BaseSize,
                    idxS2Base * coreParams->s2BaseSize + actualS2BaseSize);

            LocalTensor<PROCESS_T> mm1ResUb = outputQue.AllocTensor<PROCESS_T>();
            bmm1.template GetTensorC<true, true>(mm1ResUb, false, true, actualS1BaseSize, actualS2BaseSize);
            PRINT_DATA(mm1ResUb, 0, 1, 8, "[Data] bmm1 tensor c [%d, %d]", actualS1BaseSize, actualS2BaseSize);

            PseCopyIn(pseOffset, actualS1BaseSize, actualS2BaseSize, inputQue);
            PseComputeInplace(mm1ResUb, actualS1BaseSize, actualS2BaseSize, inputQue);
            PRINT_DATA(mm1ResUb, 0, 1, 8, "[Data] pse result");

            Muls(mm1ResUb, mm1ResUb, static_cast<PROCESS_T>(inputParams->scaleValue),
                 actualS1BaseSize * actualS2BaseSizeAligned);
            pipe_barrier(PIPE_V);
            PRINT_DATA(mm1ResUb, 0, 1, 8, "[Data] muls result");

            SHORT_MIX_LOG("[GmOffset] attenMask Offset %d Size %d*%d", attenMaskOffset, actualS1BaseSize,
                          actualS2BaseSize);
            MaskCopyIn(inputQue, attenMaskOffset, actualS1BaseSize, actualS2BaseSize);
            MaskComputeInplace(inputQue, mm1ResUb, actualS1BaseSize, actualS2BaseSize);
            PRINT_DATA(mm1ResUb, 0, 1, 8, "[Data] AttenMask result");

            mm1ResUb.SetShapeInfo(ShapeInfo(2, shapeArrayMm1ResUb, DataFormat::ND));
            if (unlikely(!needUpdateMaxSum)) {
                MaxSumCompute(mm1ResUb, softmaxMaxUb, softmaxSumUb, maxSumApiBuf, t->maxSumTiling);
                PRINT_DATA(softmaxMaxUb, 0, 1, 8, "[Data] stage1 max");
                PRINT_DATA(softmaxSumUb, 0, 1, 8, "[Data] stage1 sum");
                needUpdateMaxSum = true;
            } else {
                MaxSumUpdateInplaceCompute(mm1ResUb, softmaxMaxUb, softmaxSumUb, maxSumApiBuf, t->maxSumUpdateTiling);
                PRINT_DATA(softmaxMaxUb, 0, 1, 8, "[Data] stage1 max updated");
                PRINT_DATA(softmaxSumUb, 0, 1, 8, "[Data] stage1 sum updated");
            }

            outputQue.EnQue<PROCESS_T>(mm1ResUb);
            DataCopyParams intriParams;
            intriParams.blockCount = actualS1BaseSize;
            intriParams.blockLen = math::Ceil((uint16_t)(actualS2BaseSize * sizeof(PROCESS_T)), (uint16_t)BLOCK_BYTES);
            intriParams.srcStride = 0;
            intriParams.dstStride = numS2BlocksProcessT - intriParams.blockLen;
#if __CCE_KT_TEST__
            assert(math::Ceil((int64_t)(actualS2BaseSize * sizeof(PROCESS_T)), (int64_t)BLOCK_BYTES) <= 65535 &&
                   "blockLen exceed 65535");
            assert(numS2BlocksProcessT - intriParams.blockLen <= 65535 && "dstStride exceed 65535");
#endif
            SHORT_MIX_LOG(
                "[GmOffset] stage1 copy out to wksp idx %d offset %d+%d blockCount %d blockLen %d srcStride %d "
                "dstStride %d",
                idxS2Base, workspaceGmProcessTBaseOffsetBytes / sizeof(PROCESS_T), idxS2Base * coreParams->s2BaseSize,
                intriParams.blockCount, intriParams.blockLen, intriParams.srcStride, intriParams.dstStride);
            CopyOutToWorkspace<PROCESS_T>(workspaceGmProcessT[idxS2Base * coreParams->s2BaseSize], outputQue,
                                          intriParams);
            PRINT_DATA(workspaceGmProcessT[idxS2Base * coreParams->s2BaseSize], 0, 1, 8,
                       "[Data] stage1 copy out to wksp");
        }
    }
    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(pipe->FetchEventID(HardEvent::MTE3_MTE2));
    set_flag(PIPE_MTE3, PIPE_MTE2, eventIDMTE3ToMTE2);
    wait_flag(PIPE_MTE3, PIPE_MTE2, eventIDMTE3ToMTE2);
}

template <typename INPUT_T, typename PROCESS_T>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1s2d<INPUT_T, PROCESS_T>::ProcessStage2OnS2(
    uint32_t actualS1BaseSize, int dropmaskOffsetWithoutS2, int maxSumOffsetWithoutS2D)
{
    uint64_t dropmaskOffset;
    for (uint64_t idxS2Base = startS2Outer * coreParams->nRatio; idxS2Base < numS2Base; idxS2Base++) {
        uint32_t actualS2BaseSize =
            math::Min((uint32_t)coreParams->s2BaseSize, (uint32_t)(actualS2Size - idxS2Base * coreParams->s2BaseSize));
        SHORT_MIX_LOG("[Index] stage2 s2 index %d range [%d, %d)", idxS2Base, idxS2Base * coreParams->s2BaseSize,
                      idxS2Base * coreParams->s2BaseSize + actualS2BaseSize);
        uint32_t actualS2BaseSizeAligned =
            math::Align(actualS2BaseSize, (uint32_t)(BLOCK_BYTES / sizeof(PROCESS_T)));
        uint32_t shapeArrayMm1ResUb[] = {actualS1BaseSize, actualS2BaseSizeAligned};

        CalcOffsetWithoutDInStage2(idxS2Base, dropmaskOffsetWithoutS2, dropmaskOffset);
        DataCopyParams inIntriParams;
        inIntriParams.blockCount = actualS1BaseSize;
        inIntriParams.blockLen = math::Ceil((uint16_t)(actualS2BaseSize * sizeof(PROCESS_T)), (uint16_t)BLOCK_BYTES);
        inIntriParams.srcStride = numS2BlocksProcessT - inIntriParams.blockLen;
        inIntriParams.dstStride = 0;
#if __CCE_KT_TEST__
        assert(math::Ceil((int64_t)(actualS2BaseSize * sizeof(PROCESS_T)), (int64_t)BLOCK_BYTES) <= 65535 &&
               "blockLen exceed 65535");
        assert(numS2BlocksProcessT - inIntriParams.blockLen <= 65535 && "srcStride exceed 65535");
#endif
        SHORT_MIX_LOG(
            "[GmOffset] stage2 copy in from wksp index %d offset %d+%d blockCount %d blockLen %d srcStride %d "
            "dstStride %d",
            idxS2Base, workspaceGmProcessTBaseOffsetBytes / sizeof(PROCESS_T), idxS2Base * coreParams->s2BaseSize,
            inIntriParams.blockCount, inIntriParams.blockLen, inIntriParams.srcStride, inIntriParams.dstStride);
        CopyInFromWorkspace(inputQue, workspaceGmProcessT[idxS2Base * coreParams->s2BaseSize], inIntriParams);

        LocalTensor<PROCESS_T> dst;
        if constexpr (IsSameType<INPUT_T, PROCESS_T>::value) {
            dst = outputQue.AllocTensor<PROCESS_T>();
        } else {
            dst = bmm1ProccessTInStage2;
        }

        CustomSoftmaxCompute<PROCESS_T>(softmaxMaxUb, softmaxSumUb, inputQue, dst, customSoftmaxApiBuf,
                                        t->customSoftmaxTiling, ShapeInfo(2, shapeArrayMm1ResUb, DataFormat::ND));
        PRINT_DATA(dst, 0, 1, 8, "[Data] stage2 custom softmax result");

        SHORT_MIX_LOG("[GmOffset] dropMask Offset %d Size %d*%d", dropmaskOffset, actualS1BaseSize, actualS2BaseSize);
        DropMaskCopyIn(dropmaskOffset, actualS1BaseSize, actualS2BaseSize, inputQue);
        DropoutComputeInplace(dst, actualS1BaseSize, actualS2BaseSize, inputQue, dropoutApiBuf);
        PRINT_DATA(dst, 0, 1, 8, "[Data] stage2 dropout result");

        if constexpr (IsSameType<INPUT_T, PROCESS_T>::value) {
            outputQue.EnQue<PROCESS_T>(dst);
            DataCopyParams intriParams;
            intriParams.blockCount = actualS1BaseSize;
            intriParams.blockLen = math::Ceil((uint16_t)(actualS2BaseSize * sizeof(PROCESS_T)), (uint16_t)BLOCK_BYTES);
            intriParams.srcStride = 0;
            intriParams.dstStride = numS2BlocksProcessT - intriParams.blockLen;
#if __CCE_KT_TEST__
            assert(math::Ceil((int64_t)(actualS2BaseSize * sizeof(PROCESS_T)), (int64_t)BLOCK_BYTES) <= 65535 &&
                   "blockLen exceed 65535");
            assert(numS2BlocksProcessT - intriParams.blockLen <= 65535 && "dstStride exceed 65535");
#endif
            SHORT_MIX_LOG(
                "[GmOffset] stage2 copy out to wksp ProcessT idx %d offset %d+%d blockCount %d blockLen %d srcStride "
                "%d dstStride %d",
                idxS2Base, workspaceGmProcessTBaseOffsetBytes / sizeof(PROCESS_T), idxS2Base * coreParams->s2BaseSize,
                intriParams.blockCount, intriParams.blockLen, intriParams.srcStride, intriParams.dstStride);
            CopyOutToWorkspace<PROCESS_T>(workspaceGmProcessT[idxS2Base * coreParams->s2BaseSize], outputQue,
                                          intriParams);
        } else {
            LocalTensor<INPUT_T> out = outputQue.AllocTensor<INPUT_T>();
            Cast(out, dst, RoundMode::CAST_ROUND, actualS1BaseSize * actualS2BaseSizeAligned);
            pipe_barrier(PIPE_V);

            outputQue.EnQue<INPUT_T>(out);
            DataCopyParams intriParams;
            intriParams.blockCount = actualS1BaseSize;
            intriParams.blockLen = math::Ceil((uint16_t)(actualS2BaseSize * sizeof(INPUT_T)), (uint16_t)BLOCK_BYTES);
            intriParams.srcStride = 0;
            intriParams.dstStride = numS2BlocksInputT - intriParams.blockLen;
#if __CCE_KT_TEST__
            assert(math::Ceil((int64_t)(actualS2BaseSize * sizeof(INPUT_T)), (int64_t)BLOCK_BYTES) <= 65535 &&
                   "blockLen exceed 65535");
            assert(numS2BlocksInputT - intriParams.blockLen <= 65535 && "dstStride exceed 65535");
#endif
            SHORT_MIX_LOG(
                "[GmOffset] stage2 copy out to wksp InputT idx %d offset %d+%d blockCount %d blockLen %d srcStride %d "
                "dstStride %d",
                idxS2Base, workspaceGmInputTBaseOffsetBytes / sizeof(INPUT_T), idxS2Base * coreParams->s2BaseSize,
                intriParams.blockCount, intriParams.blockLen, intriParams.srcStride, intriParams.dstStride);
            CopyOutToWorkspace<INPUT_T>(workspaceGmInputT[idxS2Base * coreParams->s2BaseSize], outputQue, intriParams);
        }

        PRINT_DATA(dst, 0, 1, 16, "[Data] stage2 copy to wksp");
    }

    outputSoftmaxQue.EnQue(softmaxMaxUb);
    outputSoftmaxQue.EnQue(softmaxSumUb);
    // b,n1,g,s1,8
    CopyOutToGm<float>(softmaxMaxGm[maxSumOffsetWithoutS2D], outputSoftmaxQue, actualS1BaseSize * 8);
    // b,n1,g,s1,8
    CopyOutToGm<float>(softmaxSumGm[maxSumOffsetWithoutS2D], outputSoftmaxQue, actualS1BaseSize * 8);
}

template <typename INPUT_T, typename PROCESS_T>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1s2d<INPUT_T, PROCESS_T>::UpdateOffsetV(
    uint64_t &kvOffsetWithoutS2D)
{
    if (isSparse) {
        if (inputParams->layoutType == LAYOUT_BNSD) {
            // B,N2,1,(S2),(D)
            kvOffsetWithoutS2D += startS2Outer * nS2BaseSize * inputParams->dSize;
        } else if (inputParams->layoutType == LAYOUT_BSND) {
            // B,(S2),N2,1,(D)
            kvOffsetWithoutS2D +=
                startS2Outer * nS2BaseSize * inputParams->n2Size * inputParams->dSize;
        } else if (inputParams->layoutType == LAYOUT_SBH) {
            // (S2),B,N2,1,(D)
            kvOffsetWithoutS2D += startS2Outer * nS2BaseSize * inputParams->bSize *
                                  inputParams->n2Size * inputParams->dSize;
        }
    }
}

template <typename INPUT_T, typename PROCESS_T>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1s2d<INPUT_T, PROCESS_T>::ProcessStage3OnS2(
    uint64_t kvOffsetWithoutS2D, uint64_t qOffsetWithoutD, uint32_t actualS1BaseSize)
{
    if constexpr (IsSameType<INPUT_T, PROCESS_T>::value) {
        PRINT_DATA(workspaceGmProcessT, 0, 1, 8, "[Data] bmm2 tensor a");
        SHORT_MIX_LOG("[GmOffset] bmm2 tensor a ProcessT offset %d+%d",
                      workspaceGmProcessTBaseOffsetBytes / sizeof(PROCESS_T),
                      startS2Outer * nS2BaseSize);
        bmm2.SetTensorA(workspaceGmProcessT[startS2Outer * nS2BaseSize]);
    } else {
        SHORT_MIX_LOG("[GmOffset] bmm2 tensor a InputT offset %d+%d",
                      workspaceGmInputTBaseOffsetBytes / sizeof(INPUT_T),
                      startS2Outer * nS2BaseSize);
        bmm2.SetTensorA(workspaceGmInputT[startS2Outer * nS2BaseSize]);
    }

    UpdateOffsetV(kvOffsetWithoutS2D);

    // (s2, D)
    PRINT_DATA(valueGm[kvOffsetWithoutS2D], 0, 64, 8, "[Data] bmm2 tensor b");
    SHORT_MIX_LOG("[GmOffset] bmm2 tensor b InputT offset %d", kvOffsetWithoutS2D);
    bmm2.SetTensorB(valueGm[kvOffsetWithoutS2D]);
    bmm2.SetSingleShape(actualS1BaseSize, inputParams->dSize, actualS2Size - startS2Outer * nS2BaseSize);
    // (b,n1,s1,d)
    SHORT_MIX_LOG("[GmOffset] bmm2 AOffsetSize %d BOffsetSize %d COffsetSize %d SetSingleShape(m,n,k) (%d,%d,%d)",
                  startS2Outer * nS2BaseSize,
                  kvOffsetWithoutS2D, qOffsetWithoutD, actualS1BaseSize, inputParams->dSize,
                  actualS2Size - startS2Outer * coreParams->s2BaseSize);
    bmm2.template IterateAll<false>(attentionOutGm[qOffsetWithoutD], false);
    PRINT_DATA(attentionOutGm[qOffsetWithoutD], 0, 1, 8, "[Data] bmm2 tensor c");
}

template <typename INPUT_T, typename PROCESS_T>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1s2d<INPUT_T, PROCESS_T>::OffsetWithoutS2D(
    uint64_t idxBCore, uint64_t idxN2Core, uint64_t idxGCore, uint64_t idxS1Base, uint64_t &offsetQ, uint64_t &offsetKV,
    uint64_t &offsetDropmask, uint64_t &offsetMaxSum, uint64_t &offsetAttenMask, uint64_t &offsetPse)
{
    // B,N2,G,S1,8
    offsetMaxSum = idxBCore * n1Size * inputParams->s1Size * 8 +
                   idxN2Core * inputParams->gSize * inputParams->s1Size * 8 + idxGCore * inputParams->s1Size * 8 +
                   idxS1Base * coreParams->s1BaseSize * 8;

    if (inputParams->layoutType == LAYOUT_BNSD) {
        // B,N2,1,(S2),(D)
        offsetKV = idxBCore * inputParams->n2Size * 1 * inputParams->s2Size * inputParams->dSize +
                   idxN2Core * 1 * inputParams->s2Size * inputParams->dSize;
        // B,N2,G,S1,(D)
        offsetQ = idxBCore * n1Size * inputParams->s1Size * inputParams->dSize +
                  idxN2Core * inputParams->gSize * inputParams->s1Size * inputParams->dSize +
                  idxGCore * inputParams->s1Size * inputParams->dSize +
                  idxS1Base * coreParams->s1BaseSize * inputParams->dSize;
    } else if (inputParams->layoutType == LAYOUT_BSND) {
        // B,(S2),N2,1,(D)
        offsetKV = idxBCore * inputParams->s2Size * inputParams->n2Size * 1 * inputParams->dSize +
                   idxN2Core * 1 * inputParams->dSize;
        // B,S1,N2,G,(D)
        offsetQ = idxBCore * inputParams->s1Size * n1Size * inputParams->dSize +
                  idxS1Base * coreParams->s1BaseSize * n1Size * inputParams->dSize +
                  idxN2Core * inputParams->gSize * inputParams->dSize + idxGCore * inputParams->dSize;
    } else if (inputParams->layoutType == LAYOUT_SBH) {
        // (S2),B,N2,1,(D)
        offsetKV = idxBCore * inputParams->n2Size * 1 * inputParams->dSize + idxN2Core * 1 * inputParams->dSize;
        // S1,B,N2,G,(D)
        offsetQ = idxS1Base * coreParams->s1BaseSize * inputParams->bSize * n1Size *
                      inputParams->dSize +
                  idxBCore * n1Size * inputParams->dSize +
                  idxN2Core * inputParams->gSize * inputParams->dSize + idxGCore * inputParams->dSize;
    }

    if (hasAttenMask) {
        if (inputParams->attenMaskShapeType == 0) {
            // B,N2,G,S1,(S2)
            offsetAttenMask =
                idxBCore * n1Size * inputParams->s1Size * inputParams->s2Size +
                idxN2Core * inputParams->gSize * inputParams->s1Size * inputParams->s2Size +
                idxGCore * inputParams->s1Size * inputParams->s2Size +
                idxS1Base * coreParams->s1BaseSize * inputParams->s2Size;
        } else if (inputParams->attenMaskShapeType == 1) {
            // B,1,1,S1,(S2)
            offsetAttenMask =
                idxBCore * inputParams->s1Size * inputParams->s2Size +
                idxS1Base * coreParams->s1BaseSize * inputParams->s2Size;
        } else if (inputParams->attenMaskShapeType == 2) {
            // 1,1,1,S1,(S2)
            offsetAttenMask = idxS1Base * coreParams->s1BaseSize * inputParams->s2Size;
        }
    }

    if (hasPse) {
        if (inputParams->pseShapeType == 0) {
            // B,N2,G,S1,(S2)
            offsetPse =
                idxBCore * n1Size * inputParams->s1Size * inputParams->s2Size +
                idxN2Core * inputParams->gSize * inputParams->s1Size * inputParams->s2Size +
                idxGCore * inputParams->s1Size * inputParams->s2Size +
                idxS1Base * coreParams->s1BaseSize * inputParams->s2Size;
        } else if (inputParams->pseShapeType == 1) {
            // B,N2,G,1,(S2)
            offsetPse = idxBCore * n1Size * 1 * inputParams->s2Size +
                        idxN2Core * inputParams->gSize * 1 * inputParams->s2Size + idxGCore * 1 * inputParams->s2Size;
        }
    }

    if (hasDropout) {
        // B,N2,G,S1,(S2)
        offsetDropmask =
            idxBCore * n1Size * inputParams->s1Size * inputParams->s2Size +
            idxN2Core * inputParams->gSize * inputParams->s1Size * inputParams->s2Size +
            idxGCore * inputParams->s1Size * inputParams->s2Size +
            idxS1Base * coreParams->s1BaseSize * inputParams->s2Size;
    }
}

template <typename INPUT_T, typename PROCESS_T>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1s2d<INPUT_T, PROCESS_T>::CalcOffsetWithoutS2nratioD(
    uint64_t idxS2Outer, uint64_t kOffsetWithoutS2D, uint64_t &kOffsetWithoutS2nratioD)
{
    if (inputParams->layoutType == LAYOUT_BNSD) {
        // (B,N2,1,S2,D)
        kOffsetWithoutS2nratioD = kOffsetWithoutS2D + idxS2Outer * nS2BaseSize * inputParams->dSize;
    } else if (inputParams->layoutType == LAYOUT_BSND) {
        // (B,S2,N2,1,D)
        kOffsetWithoutS2nratioD =
            kOffsetWithoutS2D + idxS2Outer * nS2BaseSize * inputParams->n2Size * 1 * inputParams->dSize;
    } else if (inputParams->layoutType == LAYOUT_SBH) {
        // (S2,B,N2,1,D)
        kOffsetWithoutS2nratioD = kOffsetWithoutS2D + idxS2Outer * nS2BaseSize * inputParams->bSize *
                                                          inputParams->n2Size * 1 * inputParams->dSize;
    }
}

template <typename INPUT_T, typename PROCESS_T>
__aicore__ inline uint64_t FlashAttentionScoreSplitBn2gs1S1s2d<INPUT_T, PROCESS_T>::ComputeOffsetForCausal(
    const int64_t &delta, const uint64_t &s1BaseSize, const uint64_t &s2BaseSize, const uint32_t &attenMaskS2Size)
{
    if (delta <= 0) {
        return math::Min(-1 * delta, static_cast<int64_t>(s1BaseSize));
    } else {
        return math::Min(delta, static_cast<int64_t>(s2BaseSize)) * attenMaskS2Size;
    }
}

template <typename INPUT_T, typename PROCESS_T>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1s2d<INPUT_T, PROCESS_T>::CalcOffsetInStage1(uint64_t idxS1Base,
    uint64_t idxS2Base, uint64_t pseOffsetWithoutS2, uint64_t attenMaskOffsetWithoutS2, uint64_t &pseOffset,
    uint64_t &attenMaskOffset)
{
    if (hasPse) {
        // B,N2,G,S1,S2 or B,N2,G,1,S2
        pseOffset = pseOffsetWithoutS2 + idxS2Base * coreParams->s2BaseSize;
    }

    if (hasAttenMask) {
        if (inputParams->attenMaskCompressMode == attenMaskNoCompress) {
            // B,N2,G,S1,S2 or B,1,1,S1,S2 or 1,1,1,S1,S2
            attenMaskOffset = attenMaskOffsetWithoutS2 + idxS2Base * coreParams->s2BaseSize;
        } else if (inputParams->attenMaskCompressMode == attenMaskLeftUpCausalCompress) {
            int64_t delta = idxS1Base * coreParams->s1BaseSize - idxS2Base * coreParams->s2BaseSize;
            attenMaskOffset = ComputeOffsetForCausal(delta, coreParams->s1BaseSize, coreParams->s2BaseSize,
                                                     inputParams->attenMaskS2Size);
        } else if (inputParams->attenMaskCompressMode == attenMaskRightDownCausalCompress) {
            uint64_t delta = idxS1Base * coreParams->s1BaseSize - idxS2Base * coreParams->s2BaseSize +
                             inputParams->s2Size - inputParams->s1Size;
            attenMaskOffset = ComputeOffsetForCausal(delta, coreParams->s1BaseSize, coreParams->s2BaseSize,
                                                     inputParams->attenMaskS2Size);
        }
    }
}

template <typename INPUT_T, typename PROCESS_T>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1s2d<INPUT_T, PROCESS_T>::CalcOffsetWithoutDInStage2(
    uint64_t idxS2Outer, uint64_t dropmaskOffsetWithoutS2, uint64_t &dropmaskOffset)
{
    // B,N2,G,S1,S2
    dropmaskOffset = dropmaskOffsetWithoutS2 + idxS2Outer * coreParams->s2BaseSize;
}

template <typename INPUT_T, typename PROCESS_T>
__aicore__ inline void FlashAttentionScoreSplitBn2gs1S1s2d<INPUT_T, PROCESS_T>::Process()
{
    // P = QK
    // result = PV
    // Q (B,N2,G,S1,D)
    uint64_t qOffsetWithoutD;
    // KV (B,N2,1,S2,D)
    uint64_t kvOffsetWithoutS2D;
    uint64_t dropmaskOffsetWithoutS2;
    uint64_t maxSumOffsetWithoutS2D;
    uint64_t attenMaskOffsetWithoutS2;
    uint64_t pseOffsetWithoutS2;

    CalcRangeOuter();
    for (uint64_t idxOuter = startOuter; idxOuter < endOuter; ++idxOuter) {
        idxBCore = idxOuter / (n1Size * coreParams->s1OuterSize);
        idxN2Core = idxOuter / (inputParams->gSize * coreParams->s1OuterSize) % inputParams->n2Size;
        idxGCore = idxOuter / (coreParams->s1OuterSize) % inputParams->gSize;
        idxS1Base = idxOuter % coreParams->s1OuterSize;
        softmaxMaxUb = outputSoftmaxQue.AllocTensor<float>();
        softmaxSumUb = outputSoftmaxQue.AllocTensor<float>();

        uint32_t actualS1BaseSize =
            math::Min(coreParams->s1BaseSize, (int32_t)(inputParams->s1Size - idxS1Base * coreParams->s1BaseSize));
        SHORT_MIX_LOG("[Index] b %d n2 %d g %d s1 [%d, %d)", idxBCore, idxN2Core, idxGCore,
                idxS1Base * coreParams->s1BaseSize, idxS1Base * coreParams->s1BaseSize + actualS1BaseSize);

        OffsetWithoutS2D(idxBCore, idxN2Core, idxGCore, idxS1Base, qOffsetWithoutD, kvOffsetWithoutS2D,
                         dropmaskOffsetWithoutS2, maxSumOffsetWithoutS2D, attenMaskOffsetWithoutS2, pseOffsetWithoutS2);
        CalcRangeS2Outer(idxS1Base, idxBCore);

        ProcessStage1OnS2(actualS1BaseSize, qOffsetWithoutD, pseOffsetWithoutS2, kvOffsetWithoutS2D,
                          attenMaskOffsetWithoutS2, idxS1Base);
        ProcessStage2OnS2(actualS1BaseSize, dropmaskOffsetWithoutS2, maxSumOffsetWithoutS2D);
        ProcessStage3OnS2(kvOffsetWithoutS2D, qOffsetWithoutD, actualS1BaseSize);
    }
}
