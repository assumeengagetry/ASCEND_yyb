/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */
#pragma once
#include <cstdint>
#include <vector>
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FlashAttentionScoreGradBNGGs1s2BaseParams)
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, formerCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, remainCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, B);
TILING_DATA_FIELD_DEF(uint32_t, N1);
TILING_DATA_FIELD_DEF(uint32_t, N2);
TILING_DATA_FIELD_DEF(uint32_t, G);
TILING_DATA_FIELD_DEF(uint32_t, S1);
TILING_DATA_FIELD_DEF(uint32_t, S2);
TILING_DATA_FIELD_DEF(uint32_t, H);
TILING_DATA_FIELD_DEF(uint32_t, D);
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(float, keepProb);
TILING_DATA_FIELD_DEF(uint32_t, preTokens);
TILING_DATA_FIELD_DEF(uint32_t, nextTokens);
TILING_DATA_FIELD_DEF(uint32_t, typeBlockNum);
TILING_DATA_FIELD_DEF(uint32_t, layout);
TILING_DATA_FIELD_DEF(uint32_t, pseOptional);
TILING_DATA_FIELD_DEF(uint32_t, pseShapeType);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskDtype);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskShapeType);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGradBNGGs1s2BaseParamsOp, FlashAttentionScoreGradBNGGs1s2BaseParams)

BEGIN_TILING_DATA_DEF(FlashAttentionScoreGradBNGGs1s2SplitCoreParams)
TILING_DATA_FIELD_DEF(uint32_t, batchRange);
TILING_DATA_FIELD_DEF(uint32_t, splitedBatchRange);
TILING_DATA_FIELD_DEF(uint32_t, splitedRemainBatchRange);
TILING_DATA_FIELD_DEF(uint32_t, gOut);
TILING_DATA_FIELD_DEF(uint32_t, gInner);
TILING_DATA_FIELD_DEF(uint32_t, gTail);
TILING_DATA_FIELD_DEF(uint32_t, innerMatResSize);
TILING_DATA_FIELD_DEF(uint32_t, innerMatInputSize);
END_TILING_DATA_DEF;
// op name must be unique, and REGISTER_TILING_DATA_CLASS() must be in one line
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGradBNGGs1s2SplitCoreParamsOp, FlashAttentionScoreGradBNGGs1s2SplitCoreParams)

BEGIN_TILING_DATA_DEF(FlashAttentionScoreGradTilingDataBNGgs1s2)
TILING_DATA_FIELD_DEF_STRUCT(FlashAttentionScoreGradBNGGs1s2BaseParams, gBNBaseParams);
TILING_DATA_FIELD_DEF_STRUCT(FlashAttentionScoreGradBNGGs1s2SplitCoreParams, gBNSplitCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm2TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm3TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm4TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1TilingDataTailCore);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm2TilingDataTailCore);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxGradTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000001012, FlashAttentionScoreGradTilingDataBNGgs1s2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000002012, FlashAttentionScoreGradTilingDataBNGgs1s2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000003012, FlashAttentionScoreGradTilingDataBNGgs1s2)

}  // namespace optiling