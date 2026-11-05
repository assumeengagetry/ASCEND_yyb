/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _FLASH_ATTENTION_SCORE_GRAD_NGS1S2_BN_DROPOUT_H_
#define _FLASH_ATTENTION_SCORE_GRAD_NGS1S2_BN_DROPOUT_H_
#include "kernel_operator.h"
using GradTilingDataUngs1s2Bbn = FlashAttentionScoreGradTilingDataUngs1s2Bbn;

class FAGNgs1s2Bn2Dropout {
public:
    __aicore__ inline FAGNgs1s2Bn2Dropout(){};
    __aicore__ inline void Init(GM_ADDR drop_mask, GM_ADDR workspace, const GradTilingDataUngs1s2Bbn* __restrict ordTilingData, TPipe *pipe_in);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void DropoutProcess(const GlobalTensor<uint8_t> &dstGm, const GlobalTensor<uint8_t> &srcGm, LocalTensor<half> &src0Tensor);

    constexpr static uint32_t BLOCK = 32;
    constexpr static uint32_t BUFFER_NUM = 2;
    constexpr static uint32_t B8_BLOCK_NUM = BLOCK / sizeof(uint8_t);
    constexpr static uint32_t HELP_LEN = 256;
    constexpr static uint32_t BIT8 = 8;
    constexpr static uint32_t B16_VECTOR_MASK = 128;
    constexpr static uint32_t B32_BLOCK_NUM = BLOCK / sizeof(int32_t);
    constexpr static uint64_t VALUE_ZERO = 0;

    GlobalTensor<uint8_t> dropmaskGm;
    GlobalTensor<uint8_t> dropoutWorkspaceGm;
    GlobalTensor<int32_t> syncDropoutGlobal;

    TPipe *pipe;
    int64_t blockIdx;
    const GradTilingDataUngs1s2Bbn* __restrict tilingData;

    TQue<QuePosition::VECIN, BUFFER_NUM> helpQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> castQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue;

    uint64_t usedCoreNum;
    uint64_t singleUBProcessNum;
    uint64_t tailUBProcessNum;
    uint64_t ubLoop;

    // Shape
    uint32_t dimB;
    uint32_t dimN2;
    uint32_t dimS1;
    uint32_t dimS2;
    uint32_t dimG;
    uint32_t dimD;

    half padValue{1.0};
};

__aicore__ inline void FAGNgs1s2Bn2Dropout::Init(GM_ADDR drop_mask, GM_ADDR workspace,
    const GradTilingDataUngs1s2Bbn* __restrict ordTilingData, TPipe *pipe_in) {
    tilingData = ordTilingData;
    if (tilingData->dropoutCastParams.dropoutWorkspaceLen == 0) {
        return;
    }
    blockIdx = GetBlockIdx();
    pipe = pipe_in;

    // helpTensor：固定大小256(X2)
    // inputTensor：输入mask逐bit生效, 假设加载128 uint8
    // castTensor：满足vsel逐bit生成fp16Tensor, 需加载 128 * 8 fp16, input:cast ~= 1/16
    // outputTensor: fp16转为uint8，output:cast ~= 1/2
    // 所以整个UB虽然划分了4块Tensor，但是可以看做只有2个存活节点，1个是castTensor，1个是inputTensor + helpTensor + outputTensor
    pipe->InitBuffer(helpQue, BUFFER_NUM, HELP_LEN);
    pipe->InitBuffer(inputQue, BUFFER_NUM, tilingData->dropoutCastParams.inputBufferLen);
    pipe->InitBuffer(castQue, BUFFER_NUM, tilingData->dropoutCastParams.castBufferLen);
    pipe->InitBuffer(outQue, BUFFER_NUM, tilingData->dropoutCastParams.outputBufferLen);

    auto syncDropoutLen = 3200;
    auto beginAddr = (syncDropoutLen) / sizeof(uint8_t);
    dropoutWorkspaceGm.SetGlobalBuffer((__gm__ uint8_t*)workspace + beginAddr);
    dropmaskGm.SetGlobalBuffer((__gm__ uint8_t *)drop_mask);

    usedCoreNum = tilingData->splitCoreParams.usedCoreNum;
    singleUBProcessNum = tilingData->dropoutCastParams.singleUBProcessNum;

    dimB = tilingData->opInfo.b;
    dimN2 = tilingData->opInfo.n;
    dimS1 = tilingData->opInfo.sQ;
    dimS2 = tilingData->opInfo.sKV;
    dimG = tilingData->opInfo.g;
    dimD = tilingData->opInfo.d;
    // 软同步初始化
    syncDropoutGlobal.SetGlobalBuffer((__gm__ int32_t*)workspace, syncDropoutLen / sizeof(int32_t));
    InitOutput<int32_t>(syncDropoutGlobal[blockIdx * B32_BLOCK_NUM], B32_BLOCK_NUM, VALUE_ZERO);
}

__aicore__ inline void FAGNgs1s2Bn2Dropout::Process() {
    if (tilingData->dropoutCastParams.dropoutWorkspaceLen == 0) {
        return;
    }
    /*
    split data by core:
    core0: sizeByCore, core1: sizeByCore, core2: sizeByCore, ..., coreX: sizeByCore, tailCore: sizeByTailCore
    split coreX by ub:
    loop0: singleUBProcessNum, loop1: singleUBProcessNum, ..., loopX: singleUBProcessNum, tailLoop: tailUBProcessNumX
    split tailCore by ub:
    loop0: singleUBProcessNum, loop1: singleUBProcessNum, ..., loopM: singleUBProcessNum, tailLoop: tailUBProcessNumY
    */
    // outputSize and outputSize align by 32B
    uint64_t dpSize = dimB * dimN2 * dimG * dimS1 * dimS2;
    dpSize = (dpSize + B8_BLOCK_NUM - 1) / B8_BLOCK_NUM * B8_BLOCK_NUM;

    // size of processed data by each core which must be aligned by 32B.
    // size of processed data by each core which must be divided by 8bit.
    // before contains after.
    uint64_t sizeByCore = (dpSize + usedCoreNum - 1) / usedCoreNum;
    sizeByCore = (sizeByCore + B8_BLOCK_NUM - 1) / B8_BLOCK_NUM * B8_BLOCK_NUM;
    uint64_t coreNum = (dpSize + sizeByCore - 1) / sizeByCore;
    uint64_t sizeByTailCore = dpSize - (coreNum - 1) * sizeByCore;
    sizeByTailCore = (sizeByTailCore + B8_BLOCK_NUM - 1) / B8_BLOCK_NUM * B8_BLOCK_NUM;

    if (blockIdx < coreNum) {
        // calc params about UB
        // singleUBProcessNum and tailUBProcessNum must be divided by 8bit
        // vsel(half) wanna mask is 128, so singleUBProcessNum must be divided by 128
        // due to sizeByTailCore divided by 32, tailUBProcessNum must be divided by 32
        if (blockIdx == coreNum - 1) {
            ubLoop = (sizeByTailCore + singleUBProcessNum - 1) / singleUBProcessNum;
            tailUBProcessNum = sizeByTailCore - (ubLoop - 1) * singleUBProcessNum;
        } else {
            ubLoop = (sizeByCore + singleUBProcessNum - 1) / singleUBProcessNum;
            tailUBProcessNum = sizeByCore - (ubLoop - 1) * singleUBProcessNum;
        }

        // malloc tensor filled by 1.0
        auto helpTensor = helpQue.AllocTensor<half>();
        Duplicate<half>(helpTensor, padValue, HELP_LEN / sizeof(half));
        pipe_barrier(PIPE_V);

        // do process
        // assure outputAddr and inputAddr
        uint64_t outputAddr = blockIdx * sizeByCore;
        uint64_t inputAddr = blockIdx * sizeByCore / BIT8;
        DropoutProcess(dropoutWorkspaceGm[outputAddr], dropmaskGm[inputAddr], helpTensor);
        helpQue.FreeTensor(helpTensor);
    }
    // cast
    auto workLocalDropout = castQue.AllocTensor<int32_t>();
    SyncAll<false>(syncDropoutGlobal, workLocalDropout);
    castQue.FreeTensor(workLocalDropout);
}

__aicore__ inline void FAGNgs1s2Bn2Dropout::DropoutProcess(const GlobalTensor<uint8_t> &dstGm, const GlobalTensor<uint8_t> &srcGm, LocalTensor<half> &src0Tensor) {
    // params
    DataCopyPadParams padParams;
    DataCopyParams mte2Params;
    mte2Params.blockCount = 1;
    mte2Params.blockLen = singleUBProcessNum / BIT8;
    mte2Params.srcStride = 0;
    mte2Params.dstStride = 0;

    DataCopyParams mte3Params;
    mte3Params.blockCount = 1;
    mte3Params.blockLen = singleUBProcessNum;
    mte3Params.srcStride = 0;
    mte3Params.dstStride = 0;

    BinaryRepeatParams repParams;
    repParams.src0BlkStride = 1;
    repParams.src0RepStride = 0;
    repParams.src1BlkStride = 0;
    repParams.src1RepStride = 0;
    repParams.dstBlkStride = 1;
    repParams.dstRepStride = BIT8;

    /*
    split data by core:
    core0: sizeByCore, core1: sizeByCore, core2: sizeByCore, ..., coreX: sizeByCore, tailCore: sizeByTailCore
    split coreX by ub:
    loop0: singleUBProcessNum, loop1: singleUBProcessNum, ..., loopX: singleUBProcessNum, tailLoop: tailUBProcessNumX
    split tailCore by ub:
    loop0: singleUBProcessNum, loop1: singleUBProcessNum, ..., loopM: singleUBProcessNum, tailLoop: tailUBProcessNumY
    */

    // process
    uint64_t processNum = singleUBProcessNum;
    uint8_t repeat = processNum / B16_VECTOR_MASK; // singleUBProcessNum can be divided by 128.
    for (uint64_t idx = 0; idx < ubLoop; ++idx) {
        // allocTensor
        uint64_t outputOffset = idx * processNum;
        uint64_t inputOffset = idx * processNum / BIT8;
        if (idx == ubLoop - 1) {
            // 尾块是32的倍数，但不是128的倍数，但是由于Tensor足够大，Select就当他是128的倍数处理
            // 最后搬出的时候，依旧只搬出实际大小
            processNum = tailUBProcessNum;
            repeat = (processNum + B16_VECTOR_MASK - 1) / B16_VECTOR_MASK;
            mte2Params.blockLen = tailUBProcessNum / BIT8;
            mte3Params.blockLen = tailUBProcessNum;
        }
        // copyIn
        auto castTensor = castQue.AllocTensor<half>();
        auto inputTensor = inputQue.AllocTensor<uint8_t>();
        DataCopyPad(inputTensor, srcGm[inputOffset], mte2Params, padParams);
        inputQue.EnQue(inputTensor);
        inputQue.DeQue<uint8_t>();

        // select
        Select(castTensor, inputTensor, src0Tensor, (half)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, B16_VECTOR_MASK, repeat, repParams);
        inputQue.FreeTensor(inputTensor);

        // cast
        auto outputTensor = outQue.AllocTensor<uint8_t>();
        Cast(outputTensor, castTensor, RoundMode::CAST_ROUND, processNum);
        castQue.FreeTensor(castTensor);

        // copyOut
        outQue.EnQue(outputTensor);
        outQue.DeQue<uint8_t>();
        DataCopyPad(dstGm[outputOffset], outputTensor, mte3Params);
        outQue.FreeTensor(outputTensor);
    }
}
#endif