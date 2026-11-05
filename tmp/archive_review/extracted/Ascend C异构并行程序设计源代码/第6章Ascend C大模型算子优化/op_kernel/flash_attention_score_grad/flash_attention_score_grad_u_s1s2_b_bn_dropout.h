/**

Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef _FLASH_ATTENTION_SCORE_GRAD_U_S1S2_B_BN_DROPOUT_H_
#define _FLASH_ATTENTION_SCORE_GRAD_U_S1S2_B_BN_DROPOUT_H_
#include "kernel_operator.h"

using GradTilingData = FlashAttentionScoreGradUs1s2BbnTilingData;

class FAGUs1s2BbnDropout {
 public:
  __aicore__ inline FAGUs1s2BbnDropout(){};
  __aicore__ inline void Init(GM_ADDR drop_mask, GM_ADDR workspace, const GradTilingData* __restrict ordTilingData,
                              TPipe *pipe_in);
  __aicore__ inline void Process();

 protected:
  __aicore__ inline void DropoutProcess(const GlobalTensor<uint8_t> &dstGm, const GlobalTensor<uint8_t> &srcGm,
                                        LocalTensor<half> &src0Tensor);

  constexpr static uint32_t BLOCK = 32;
  constexpr static uint32_t BUFFER_NUM = 1;
  constexpr static uint32_t B8_BLOCK_NUM = BLOCK / sizeof(uint8_t);
  constexpr static uint32_t HELP_LEN = 256;
  constexpr static uint32_t BIT8 = 8;
  constexpr static uint32_t B16_VECTOR_MASK = 128;
  constexpr static uint32_t B32_BLOCK_NUM = BLOCK / sizeof(int32_t);
  constexpr static uint64_t VALUE_ZERO = 0;
  constexpr static uint64_t NUMBER_8 = 8;
  
  GlobalTensor<uint8_t> dropmaskGm;
  GlobalTensor<uint8_t> dropoutWorkspaceGm;
  GlobalTensor<int32_t> syncDropoutGlobal;

  TPipe *pipe;
  int64_t blockIdx;
  const GradTilingData* __restrict tilingData;

  TQue<QuePosition::VECIN, 1> helpQue;
  TQue<QuePosition::VECIN, BUFFER_NUM> inputQue;
  TQue<QuePosition::VECIN, BUFFER_NUM> castQue;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQue;

  uint64_t usedCoreNum;
  uint64_t singleUBProcessNum;
  uint64_t tailUBProcessNum;
  uint64_t ubLoop;

  // Shape
  uint32_t dimB;
  uint32_t dimN;
  uint32_t dimS1;
  uint32_t dimS2;
  DataCopyPadParams padParams;
  BinaryRepeatParams repParams;
  DataCopyParams mainMTE2Params;
  DataCopyParams tailMTE2Params;
  DataCopyParams mainMTE3Params;
  DataCopyParams tailMTE3Params;
  half padValue{1.0};
};

__aicore__ inline void FAGUs1s2BbnDropout::Init(GM_ADDR drop_mask, GM_ADDR workspace,
                                                const GradTilingData* __restrict ordTilingData, TPipe *pipe_in) {
  blockIdx = GetBlockIdx();
  tilingData = ordTilingData;
  pipe = pipe_in;

  // helpTensor：固定大小256(X2)
  // inputTensor：输入mask逐bit生效, 假设加载128 uint8
  // castTensor：满足vsel逐bit生成fp16Tensor, 需加载 128 * 8 fp16, input:cast ~= 1/16
  // outputTensor: fp16转为uint8，output:cast ~= 1/2
  // 所以整个UB虽然划分了4块Tensor，但是可以看做只有2个存活节点，1个是castTensor，1个是inputTensor + helpTensor + outputTensor
  pipe->InitBuffer(helpQue, 1, HELP_LEN);
  pipe->InitBuffer(inputQue, BUFFER_NUM, tilingData->dropoutCastParams.inputBufferLen);
  pipe->InitBuffer(castQue, BUFFER_NUM, tilingData->dropoutCastParams.castBufferLen);
  pipe->InitBuffer(outQue, BUFFER_NUM, tilingData->dropoutCastParams.outputBufferLen);

  auto syncDropoutLen = tilingData->baseParams.syncDropoutLen;
  dropoutWorkspaceGm.SetGlobalBuffer((__gm__ uint8_t*)workspace + 800 * 2 + syncDropoutLen);
  dropmaskGm.SetGlobalBuffer((__gm__ uint8_t *)drop_mask);

  usedCoreNum = tilingData->baseParams.coreNum;
  singleUBProcessNum = tilingData->dropoutCastParams.singleUBProcessNum;

  dimB = tilingData->baseParams.B;
  dimN = tilingData->baseParams.N;
  dimS1 = tilingData->baseParams.S;
  dimS2 = tilingData->baseParams.Skv;

  syncDropoutGlobal.SetGlobalBuffer((__gm__ int32_t*)workspace + 800 * 2 / sizeof(int32_t),
                                    syncDropoutLen / sizeof(int32_t));
  InitOutput<int32_t>(syncDropoutGlobal[blockIdx * B32_BLOCK_NUM], B32_BLOCK_NUM, VALUE_ZERO);
}

__aicore__ inline void FAGUs1s2BbnDropout::Process() {
  /*
  split data by core:
  core0: sizeByCore, core1: sizeByCore, core2: sizeByCore, ..., coreX: sizeByCore, tailCore: sizeByTailCore
  split coreX by ub:
  loop0: singleUBProcessNum, loop1: singleUBProcessNum, ..., loopX: singleUBProcessNum, tailLoop: tailUBProcessNumX
  split tailCore by ub:
  loop0: singleUBProcessNum, loop1: singleUBProcessNum, ..., loopM: singleUBProcessNum, tailLoop: tailUBProcessNumY
  */
// outputSize and outputSize align by 32B
  uint64_t dpSize = dimB * dimN * dimS1 * dimS2;
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

__aicore__ inline void FAGUs1s2BbnDropout::DropoutProcess(const GlobalTensor<uint8_t> &dstGm,
                                                          const GlobalTensor<uint8_t> &srcGm,
                                                          LocalTensor<half> &src0Tensor) {
  // params
  repParams.src0BlkStride = 1;
  repParams.src0RepStride = 0;
  repParams.src1BlkStride = 0;
  repParams.src1RepStride = 0;
  repParams.dstBlkStride = 1;
  repParams.dstRepStride = NUMBER_8;

  uint8_t main_repeat = singleUBProcessNum / B16_VECTOR_MASK;
  uint8_t tail_repeat = (tailUBProcessNum + B16_VECTOR_MASK - 1) / B16_VECTOR_MASK;

  mainMTE2Params.blockCount = 1;
  mainMTE2Params.blockLen = singleUBProcessNum / BIT8;
  mainMTE2Params.srcStride = 0;
  mainMTE2Params.dstStride = 0;

  tailMTE2Params.blockCount = 1;
  tailMTE2Params.blockLen = tailUBProcessNum / BIT8;
  tailMTE2Params.srcStride = 0;
  tailMTE2Params.dstStride = 0;

  mainMTE3Params.blockCount = 1;
  mainMTE3Params.blockLen = singleUBProcessNum;
  mainMTE3Params.srcStride = 0;
  mainMTE3Params.dstStride = 0;

  tailMTE3Params.blockCount = 1;
  tailMTE3Params.blockLen = tailUBProcessNum;
  tailMTE3Params.srcStride = 0;
  tailMTE3Params.dstStride = 0;

  // process
  for (uint64_t idx = 0; idx < ubLoop; idx++) {
    // allocTensor
    uint64_t outputOffset = idx * singleUBProcessNum;
    uint64_t inputOffset = idx * singleUBProcessNum / BIT8;
    bool is_last = idx == ubLoop - 1;

    // copyIn
    auto inputTensor = inputQue.AllocTensor<uint8_t>();
    DataCopyPad(inputTensor, srcGm[inputOffset], is_last ? tailMTE2Params : mainMTE2Params, padParams);
    inputQue.EnQue(inputTensor);
    inputQue.DeQue<uint8_t>();

    // select
    auto castTensor = castQue.AllocTensor<half>();
    Select(castTensor, inputTensor, src0Tensor, (half)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE,
           B16_VECTOR_MASK, is_last ? tail_repeat : main_repeat, repParams);
    pipe_barrier(PIPE_V);
    inputQue.FreeTensor(inputTensor);

    // cast
    auto outputTensor = outQue.AllocTensor<uint8_t>();
    Cast(outputTensor, castTensor, RoundMode::CAST_ROUND, is_last ? tailUBProcessNum : singleUBProcessNum);
    castQue.FreeTensor(castTensor);

    // copyOut
    outQue.EnQue(outputTensor);
    outQue.DeQue<uint8_t>();
    DataCopyPad(dstGm[outputOffset], outputTensor, is_last ? tailMTE3Params : mainMTE3Params);
    outQue.FreeTensor(outputTensor);
  }
}
#endif