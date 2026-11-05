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

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FLASHATTENTIONSCOREGRAD_TILING_US1S2_BBN_DEF_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FLASHATTENTIONSCOREGRAD_TILING_US1S2_BBN_DEF_H_
#include <cstdint>
#include <vector>
#include "register/tilingdata_base.h"
#include "../../flash_attention_score_grad_tiling_common.h"
namespace optiling {

BEGIN_TILING_DATA_DEF(FlashAttentionScoreGradUs1s2BbnBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, formerCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, B);
TILING_DATA_FIELD_DEF(uint32_t, N);
TILING_DATA_FIELD_DEF(uint32_t, S);
TILING_DATA_FIELD_DEF(uint32_t, Sq);
TILING_DATA_FIELD_DEF(uint32_t, Skv);
TILING_DATA_FIELD_DEF(uint32_t, H);
TILING_DATA_FIELD_DEF(uint32_t, D); // H/N
TILING_DATA_FIELD_DEF(uint32_t, batch); // BN
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(float, keepProb);
TILING_DATA_FIELD_DEF(uint32_t, preTokens);
TILING_DATA_FIELD_DEF(uint32_t, nextTokens);
TILING_DATA_FIELD_DEF(uint32_t, allReduceNum); // S16 or S*8
TILING_DATA_FIELD_DEF(uint32_t, layoutType);
TILING_DATA_FIELD_DEF(uint32_t, pseShapeType);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskType);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskShapeType);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskCompressMode);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskS2Size);
TILING_DATA_FIELD_DEF(uint32_t, syncDropoutLen);
TILING_DATA_FIELD_DEF(uint32_t, hasPseInput);
TILING_DATA_FIELD_DEF(uint32_t, hasAttenMaskInput);
TILING_DATA_FIELD_DEF(uint32_t, sparseMode);
TILING_DATA_FIELD_DEF(uint32_t, reserved);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, prefixN);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGradUs1s2BbnBaseParamsOp, FlashAttentionScoreGradUs1s2BbnBaseParams)

BEGIN_TILING_DATA_DEF(Us1s2BbnSplitCoreParams)
TILING_DATA_FIELD_DEF(uint32_t, splitedBatchRange); // batch/splitBatchFactor
TILING_DATA_FIELD_DEF(uint32_t, splitedCoreBatchReduceNum); // splitedBatchRange * S * 16 or 8
TILING_DATA_FIELD_DEF(uint32_t, s2Range);
TILING_DATA_FIELD_DEF(uint32_t, s1Range);
TILING_DATA_FIELD_DEF(uint32_t, s1InnerSize);
TILING_DATA_FIELD_DEF(uint32_t, s2InnerSize);
TILING_DATA_FIELD_DEF(uint32_t, tailS1InnerSize);
TILING_DATA_FIELD_DEF(uint32_t, tailS2InnerSize);
TILING_DATA_FIELD_DEF(uint32_t, s1RangeC);
TILING_DATA_FIELD_DEF(uint32_t, s2RangeC);
TILING_DATA_FIELD_DEF(uint32_t, cubeBaseS1);
TILING_DATA_FIELD_DEF(uint32_t, cubeBaseS2);
TILING_DATA_FIELD_DEF(uint32_t, tailCubeBaseS1);
TILING_DATA_FIELD_DEF(uint32_t, tailCubeBaseS2);
TILING_DATA_FIELD_DEF(uint32_t, s1CVRatio);
TILING_DATA_FIELD_DEF(uint32_t, s2CVRatio);
TILING_DATA_FIELD_DEF(uint32_t, innerReduceNum);
TILING_DATA_FIELD_DEF(uint32_t, innerMatResSize);
TILING_DATA_FIELD_DEF(uint32_t, tailSplitedBatchRange);
TILING_DATA_FIELD_DEF(uint32_t, tailSplitedCoreBatchReduceNum);
TILING_DATA_FIELD_DEF(uint32_t, sparse);
TILING_DATA_FIELD_DEF(uint32_t, notSplitGradD);
TILING_DATA_FIELD_DEF(uint32_t, SOffset);
TILING_DATA_FIELD_DEF(uint32_t, placeholder);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Us1s2BbnSplitCoreParamsOp, Us1s2BbnSplitCoreParams)

BEGIN_TILING_DATA_DEF(DropoutCastParams)
TILING_DATA_FIELD_DEF(uint32_t, castBufferLen);
TILING_DATA_FIELD_DEF(uint32_t, outputBufferLen);
TILING_DATA_FIELD_DEF(uint32_t, inputBufferLen);
TILING_DATA_FIELD_DEF(uint32_t, singleUBProcessNum);
TILING_DATA_FIELD_DEF(uint32_t, dropoutWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, placeholder);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(DropoutCastParamsOp, DropoutCastParams)

BEGIN_TILING_DATA_DEF(PostOpTilingParams)
TILING_DATA_FIELD_DEF(uint32_t, workspaceOffset);
TILING_DATA_FIELD_DEF(uint32_t, dqWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, dkWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, dvWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, inputBufferLen);
TILING_DATA_FIELD_DEF(uint32_t, outputBufferLen);
TILING_DATA_FIELD_DEF(uint32_t, singleUBProcessNum);
TILING_DATA_FIELD_DEF(uint32_t, dqSingleCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, dkvSingleCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, dqTailCoreLoop);
TILING_DATA_FIELD_DEF(uint32_t, kvTailCoreLoop);
TILING_DATA_FIELD_DEF(uint32_t, dqSingleCoreLoop);
TILING_DATA_FIELD_DEF(uint32_t, kvSingleCoreLoop);
TILING_DATA_FIELD_DEF(uint32_t, dqTailCoreLastLoopNum);
TILING_DATA_FIELD_DEF(uint32_t, kvTailCoreLastLoopNum);
TILING_DATA_FIELD_DEF(uint32_t, dqLastLoopNum);
TILING_DATA_FIELD_DEF(uint32_t, kvLastLoopNum);
TILING_DATA_FIELD_DEF(uint32_t, placeHolder);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PostOpTilingParamsOp, PostOpTilingParams)

BEGIN_TILING_DATA_DEF(FlashAttentionScoreGradUs1s2BbnTilingData)
TILING_DATA_FIELD_DEF_STRUCT(FlashAttentionScoreGradUs1s2BbnBaseParams, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(Us1s2BbnSplitCoreParams, splitCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(DropoutCastParams, dropoutCastParams);
TILING_DATA_FIELD_DEF_STRUCT(PostOpTilingParams, postOpTilingParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm2TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm3TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm4TilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxGradTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000113234, FlashAttentionScoreGradUs1s2BbnTilingData)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000112234, FlashAttentionScoreGradUs1s2BbnTilingData)
}

#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_FLASHATTENTIONSCOREGRAD_TILING_US1S2_BBN_DEF_H_
