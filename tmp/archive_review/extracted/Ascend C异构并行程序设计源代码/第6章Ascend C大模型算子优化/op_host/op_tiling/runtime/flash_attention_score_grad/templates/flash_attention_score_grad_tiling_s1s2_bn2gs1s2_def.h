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

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FLASHATTENTIONSCOREGRAD_NEW_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FLASHATTENTIONSCOREGRAD_NEW_H_
#include <cstdint>
#include <vector>
#include "register/tilingdata_base.h"
#include "../../flash_attention_score_grad_tiling_common.h"
namespace optiling {

/////////////////////////////////////////////////////////////////////////
// S1S2_BNGS1S2
/////////////////////////////////////////////////////////////////////////
BEGIN_TILING_DATA_DEF(FlashAttentionScoreGradS1S2BNGS1S2BaseParams)
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, b);
TILING_DATA_FIELD_DEF(uint32_t, n2);
TILING_DATA_FIELD_DEF(uint32_t, g);
TILING_DATA_FIELD_DEF(uint32_t, s1);
TILING_DATA_FIELD_DEF(uint32_t, s2);
TILING_DATA_FIELD_DEF(uint32_t, h);
TILING_DATA_FIELD_DEF(uint32_t, d);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(float, keepProb);
TILING_DATA_FIELD_DEF(uint32_t, layout);
TILING_DATA_FIELD_DEF(uint32_t, pseOptional);
TILING_DATA_FIELD_DEF(uint32_t, pseShapeType);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskOptional);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskDtype);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskShapeType);
TILING_DATA_FIELD_DEF(uint32_t, s1Token);  // pre_tokens
TILING_DATA_FIELD_DEF(uint32_t, s2Token);  // next_tokens
TILING_DATA_FIELD_DEF(uint32_t, sparseMode);
TILING_DATA_FIELD_DEF(uint32_t, isSparse);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, prefixN);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskCompressMode);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskS2Size);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGradS1S2BNGS1S2BaseParamsOp, FlashAttentionScoreGradS1S2BNGS1S2BaseParams)

BEGIN_TILING_DATA_DEF(FlashAttentionScoreGradS1S2BNGS1S2SplitCoreParams)
TILING_DATA_FIELD_DEF(uint32_t, s1Align);
TILING_DATA_FIELD_DEF(uint32_t, s1Outer);
TILING_DATA_FIELD_DEF(uint32_t, s1Inner);
TILING_DATA_FIELD_DEF(uint32_t, s1Tail);
TILING_DATA_FIELD_DEF(uint32_t, s2Align);
TILING_DATA_FIELD_DEF(uint32_t, s2Outer);
TILING_DATA_FIELD_DEF(uint32_t, s2CvRatio);
TILING_DATA_FIELD_DEF(uint32_t, s2Inner);
TILING_DATA_FIELD_DEF(uint32_t, s2Tail);
TILING_DATA_FIELD_DEF(uint32_t, baseMN);
TILING_DATA_FIELD_DEF(uint32_t, mm1CalRatio);
TILING_DATA_FIELD_DEF(uint32_t, sfmgdOuter);
TILING_DATA_FIELD_DEF(uint32_t, sfmgdFactor);
TILING_DATA_FIELD_DEF(uint32_t, sfmgdTail);
TILING_DATA_FIELD_DEF(uint32_t, blockOuter);
TILING_DATA_FIELD_DEF(uint32_t, tmpBufferSize);
END_TILING_DATA_DEF;
// 固定写法不能换行，会失败
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGradS1S2BNGS1S2SplitCoreParamsOp, FlashAttentionScoreGradS1S2BNGS1S2SplitCoreParams)

BEGIN_TILING_DATA_DEF(BlockNumListParams)
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, blockStarts);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, blockEnds);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(BlockNumListParamsOp, BlockNumListParams)

// dq/dv/dk vdup and dropMask bit2bool
BEGIN_TILING_DATA_DEF(PreParams)
TILING_DATA_FIELD_DEF(uint32_t, qPreBlockFactor);
TILING_DATA_FIELD_DEF(uint32_t, qPreBlockTotal);
TILING_DATA_FIELD_DEF(uint32_t, qPreBlockTail);
TILING_DATA_FIELD_DEF(uint32_t, kvPreBlockFactor);
TILING_DATA_FIELD_DEF(uint32_t, kvPreBlockTotal);
TILING_DATA_FIELD_DEF(uint32_t, kvPreBlockTail);
TILING_DATA_FIELD_DEF(uint32_t, maskPreBlockFactor);
TILING_DATA_FIELD_DEF(uint32_t, maskPreBlockTotal);
TILING_DATA_FIELD_DEF(uint32_t, maskPreTailNum);
TILING_DATA_FIELD_DEF(uint32_t, maskPreBaseNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PreParamsOp, PreParams)

// dq/dv/dk cast fp322T1
BEGIN_TILING_DATA_DEF(PostParams)
TILING_DATA_FIELD_DEF(uint32_t, postUbBaseSize);
TILING_DATA_FIELD_DEF(uint32_t, UbBaseSize);
TILING_DATA_FIELD_DEF(uint32_t, qPostBlockFactor);
TILING_DATA_FIELD_DEF(uint32_t, qPostBlockTotal);
TILING_DATA_FIELD_DEF(uint32_t, qPostBaseNum);
TILING_DATA_FIELD_DEF(uint32_t, qPostTailNum);
TILING_DATA_FIELD_DEF(uint32_t, kvPostBlockFactor);
TILING_DATA_FIELD_DEF(uint32_t, kvPostBlockTotal);
TILING_DATA_FIELD_DEF(uint32_t, kvPostBaseNum);
TILING_DATA_FIELD_DEF(uint32_t, kvPostTailNum);
TILING_DATA_FIELD_DEF(uint64_t, qSizeAlign);
TILING_DATA_FIELD_DEF(uint64_t, kvSizeAlign);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PostParamsOp, PostParams)

BEGIN_TILING_DATA_DEF(FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2)
TILING_DATA_FIELD_DEF_STRUCT(FlashAttentionScoreGradS1S2BNGS1S2BaseParams, s1s2BNGS1S2BaseParams);
TILING_DATA_FIELD_DEF_STRUCT(FlashAttentionScoreGradS1S2BNGS1S2SplitCoreParams, s1s2BNGS1S2SplitCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm2TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm3TilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxGradTilingData);
TILING_DATA_FIELD_DEF_STRUCT(BlockNumListParams, s1s2BNGS1S2BlockNumList);
TILING_DATA_FIELD_DEF_STRUCT(PreParams, preTilingData);
TILING_DATA_FIELD_DEF_STRUCT(PostParams, postTilingData);
END_TILING_DATA_DEF;
// 固定写法不能换行，会失败
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000001434, FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000002434, FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000003434, FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2)
}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_FLASHATTENTIONSCOREGRAD_NEW_H_
