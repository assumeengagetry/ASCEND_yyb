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

BEGIN_TILING_DATA_DEF(FlashAttentionScoreGradUgBbnBaseParams)
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
TILING_DATA_FIELD_DEF(uint32_t, D);       // H/N
TILING_DATA_FIELD_DEF(uint32_t, batch);   // B*N
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(float, keepProb);
TILING_DATA_FIELD_DEF(uint32_t, preTokens);
TILING_DATA_FIELD_DEF(uint32_t, nextTokens);
TILING_DATA_FIELD_DEF(uint32_t, typeBlockNum);        // 32/typeSize
TILING_DATA_FIELD_DEF(uint32_t, layout);
TILING_DATA_FIELD_DEF(uint32_t, pseOptional);
TILING_DATA_FIELD_DEF(uint32_t, pseShapeType);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskDtype);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskShapeType);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGradUgBbnBaseParamsOp, FlashAttentionScoreGradUgBbnBaseParams)

BEGIN_TILING_DATA_DEF(FlashAttentionScoreGradUgBbnSplitCoreParams)
TILING_DATA_FIELD_DEF(uint32_t, batchRange);
TILING_DATA_FIELD_DEF(uint32_t, splitedBatchRange);
TILING_DATA_FIELD_DEF(uint32_t, splitedRemainBatchRange);
TILING_DATA_FIELD_DEF(uint32_t, gOut);
TILING_DATA_FIELD_DEF(uint32_t, gInner);
TILING_DATA_FIELD_DEF(uint32_t, gTail);
TILING_DATA_FIELD_DEF(uint32_t, innerMatResSize);
TILING_DATA_FIELD_DEF(uint32_t, innerMatInputSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGradUgBbnSplitCoreParamsOp, FlashAttentionScoreGradUgBbnSplitCoreParams)

BEGIN_TILING_DATA_DEF(FlashAttentionScoreGradTilingDataUgBbn)
TILING_DATA_FIELD_DEF_STRUCT(FlashAttentionScoreGradUgBbnBaseParams, gBNBaseParams);
TILING_DATA_FIELD_DEF_STRUCT(FlashAttentionScoreGradUgBbnSplitCoreParams, gBNSplitCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm2TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm3TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm4TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1TilingDataTailCore);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm2TilingDataTailCore);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxGradTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000000129, FlashAttentionScoreGradTilingDataUgBbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000001129, FlashAttentionScoreGradTilingDataUgBbn)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000002129, FlashAttentionScoreGradTilingDataUgBbn)

}  // namespace optiling