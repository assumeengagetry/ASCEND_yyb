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

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FLASHATTENTIONSCOREGRAD_BNGS1S2_B_DEF_TILING_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FLASHATTENTIONSCOREGRAD_BNGS1S2_B_DEF_TILING_H_
#include <cstdint>
#include <vector>

#include "register/tilingdata_base.h"
#include "../../flash_attention_score_grad_tiling_common.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FlashAttentionScoreGradUbngs1s2BbBaseParams)
TILING_DATA_FIELD_DEF(int64_t, b);
TILING_DATA_FIELD_DEF(int64_t, nQ);
TILING_DATA_FIELD_DEF(int64_t, nKV);
TILING_DATA_FIELD_DEF(int64_t, g); // key & value对应的G
TILING_DATA_FIELD_DEF(int64_t, sQ);
TILING_DATA_FIELD_DEF(int64_t, sQAlign); // 将sQ对齐到32bytes之后的元素个数
TILING_DATA_FIELD_DEF(int64_t, gSQ);
TILING_DATA_FIELD_DEF(int64_t, sKV);
TILING_DATA_FIELD_DEF(int64_t, sKVAlign); // 将sKV对齐到32bytes之后的元素个数
TILING_DATA_FIELD_DEF(int64_t, sKVAlignSize); // sKv轴的size对齐到32Byte之后的结果
TILING_DATA_FIELD_DEF(int64_t, sKVAlignSizeVec); // sKv轴的size对齐到32Byte之后的结果,使用vector计算的datatype
TILING_DATA_FIELD_DEF(int64_t, hQ); // n * g * d
TILING_DATA_FIELD_DEF(int64_t, hKV); // n * 1 * d
TILING_DATA_FIELD_DEF(int64_t, d);
TILING_DATA_FIELD_DEF(int64_t, dAlign);
TILING_DATA_FIELD_DEF(int64_t, dInner);
TILING_DATA_FIELD_DEF(int64_t, innerSize);        // S*S
TILING_DATA_FIELD_DEF(int64_t, allInnerSize);      // n*S*s1*dtype
TILING_DATA_FIELD_DEF(int64_t, allInnerDataSize);      // n*S*s1*dtype
TILING_DATA_FIELD_DEF(int64_t, innerPseSize);        // S*S
TILING_DATA_FIELD_DEF(int64_t, innerPadSize);        // S*16  or S*8

TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(float, keepProb);
TILING_DATA_FIELD_DEF(uint32_t, preTokens);
TILING_DATA_FIELD_DEF(uint32_t, nextTokens);
TILING_DATA_FIELD_DEF(uint32_t, headNum); // query对应n, 也就是通常意义上的N2
TILING_DATA_FIELD_DEF(uint32_t, inputLayout); // layout格式，有BSH，SBH，BSND等
TILING_DATA_FIELD_DEF(uint32_t, preTokensBlocks);
TILING_DATA_FIELD_DEF(uint32_t, nextTokensBlocks);

TILING_DATA_FIELD_DEF(uint32_t, inputDType); // query, key, value, dx, attention_in, pse输入的datatype
TILING_DATA_FIELD_DEF(uint32_t, inputDTypeSize); // query, key, value, dx, attention_in, pse输入的datatype
TILING_DATA_FIELD_DEF(uint32_t, vecCalcDTypeSize); // 内部vector计算的数据大小，fp32和bf16作为输入的情况下，vector使用fp32计算
TILING_DATA_FIELD_DEF(uint32_t, pseSq); // 等于sQ或者1
TILING_DATA_FIELD_DEF(uint32_t, attenMaskShapeType); // 0: [b,1,sQ,sKV]或者1: [1,1,sQ,sKV]
TILING_DATA_FIELD_DEF(uint32_t, attenMaskType);
TILING_DATA_FIELD_DEF(uint32_t, elementPerBlock); // 针对qkv、attention in、dx输入每个block（32Bytes）对应的元素个数
TILING_DATA_FIELD_DEF(uint32_t, pseShapeType);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGradUbngs1s2BbBaseParamsOp, FlashAttentionScoreGradUbngs1s2BbBaseParams)

BEGIN_TILING_DATA_DEF(SplitBSplitCoreParams)
TILING_DATA_FIELD_DEF(int64_t, totalBatch); // 总的batch数，等于bOut
TILING_DATA_FIELD_DEF(int64_t, bOut); // 用于分核的外层b轴
TILING_DATA_FIELD_DEF(int64_t, apiClcQueueSize); // softmax、dropout、softmaxgrad这三个高阶api计算可以用到的最大的ubsize
TILING_DATA_FIELD_DEF(int64_t, mm1ResSize); // matmul结果的size = sQ * Skv * inputDTypeSize
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum); // 实际使用的vector核数
TILING_DATA_FIELD_DEF(uint32_t, reserved);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SplitBSplitCoreParamsOp, SplitBSplitCoreParams)

BEGIN_TILING_DATA_DEF(SplitBSingleCoreParams)
TILING_DATA_FIELD_DEF(int64_t, bIn); // 用于给单核内凑数据搬运量的内层b轴
TILING_DATA_FIELD_DEF(int64_t, bInTail); // b对bIn做了拆分之后，不能整除时剩余的尾巴b_inner的大小
TILING_DATA_FIELD_DEF(uint32_t, singleCoreBatchRange); // 单个核需要处理的batch数，这里的batch是指B * Nout
// 最后一个核需要处理的batch数，可能会小于singleCoreBatchRange
// 当totalBatch不能整除核数的时候，最后一个核处理的batch会变少。
TILING_DATA_FIELD_DEF(uint32_t, singleCoreBatchRangeTail);
TILING_DATA_FIELD_DEF(int64_t, softMaxWorkspaceSize);
TILING_DATA_FIELD_DEF(int64_t, softMaxGradWorkspaceSize);
TILING_DATA_FIELD_DEF(int64_t, splitedDLen); // TODO d轴的切分，注意对齐到block
TILING_DATA_FIELD_DEF(int64_t, dRange);

/* Sub计算用到的参数 */
TILING_DATA_FIELD_DEF(int64_t, subRange); // 用sQ / 8 向上取整，最后可能有尾块
// 表示单条vector指令算多少个数，256 / sizeof(vector计算的数据类型size)
TILING_DATA_FIELD_DEF(int64_t, subMask);
// 表示尾块单条指令算多少个数,(sQ % 8) * (32 / sizeof(vector计算的数据类型size))
TILING_DATA_FIELD_DEF(int64_t, subMaskTail);
// skVAlign占据多少个block = sKv * sizeof(vector计算的数据类型size) / 32
TILING_DATA_FIELD_DEF(int64_t, sKVAlignBlockNum);
// 所有核上单个matmul的L0C的计算结果需要用到的workspace大小， 等于 b * n * g * sQ * sKVAlign * sizeof(float)
TILING_DATA_FIELD_DEF(int64_t, workspaceCntPerMm);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SplitBSingleCoreParamsOp, SplitBSingleCoreParams)

BEGIN_TILING_DATA_DEF(FlashAttentionScoreGradNewOffsetParams)
TILING_DATA_FIELD_DEF(uint32_t, matInSizeQ);
TILING_DATA_FIELD_DEF(uint32_t, matInSizeKV);
TILING_DATA_FIELD_DEF(uint32_t, softmaxSize);
TILING_DATA_FIELD_DEF(uint32_t, pseInSize);
TILING_DATA_FIELD_DEF(uint32_t, maskInSize);
TILING_DATA_FIELD_DEF(uint32_t, dropInSize);
TILING_DATA_FIELD_DEF(uint32_t, batchOffsetQ);
TILING_DATA_FIELD_DEF(uint32_t, batchOffsetKV);
TILING_DATA_FIELD_DEF(uint32_t, batchOffsetSoftmax);
TILING_DATA_FIELD_DEF(uint32_t, batchOffsetPse);       // H/N
TILING_DATA_FIELD_DEF(uint32_t, batchOffsetMask);       // H/N
TILING_DATA_FIELD_DEF(uint32_t, batchOffsetDrop);        // 32/typeSize
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGradNewOffsetParamsOp, FlashAttentionScoreGradNewOffsetParams)

BEGIN_TILING_DATA_DEF(SplitDParams)
TILING_DATA_FIELD_DEF(int64_t, clcDInner); // 非尾块的每段d长度长度
TILING_DATA_FIELD_DEF(int64_t, dSize); // 总长d，按照clcDInner切分，可以分几段
TILING_DATA_FIELD_DEF(int64_t, dInnerTail); // 总长d，按照clcDInner切分，尾块的dtail长度;
TILING_DATA_FIELD_DEF(int64_t, dInnerTailAlign); // 尾块的dtail长度进行32字节补齐后尾段的d长度
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SplitDParamsOp, SplitDParams)

BEGIN_TILING_DATA_DEF(FlashAttentionScoreGradUbngs1s2BbTilingData)
TILING_DATA_FIELD_DEF_STRUCT(FlashAttentionScoreGradUbngs1s2BbBaseParams, opInfo);
TILING_DATA_FIELD_DEF_STRUCT(SplitBSplitCoreParams, splitCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(SplitBSingleCoreParams, singleCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(FlashAttentionScoreGradNewOffsetParams, offsetInfo);
TILING_DATA_FIELD_DEF_STRUCT(SplitDParams, splitDInfo);

TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1AndMm2TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm31TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm32AndMm4TilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxGradTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000103099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000100099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000102099, FlashAttentionScoreGradUbngs1s2BbTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000101099, FlashAttentionScoreGradUbngs1s2BbTilingData)
}
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_FLASHATTENTIONSCOREGRAD_BNGS1S2_B_DEF_TILING_H_