/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FLASHATTENTIONSCOREGRAD_TILING_S1S2_BN2_DEF_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FLASHATTENTIONSCOREGRAD_TILING_S1S2_BN2_DEF_H_
#include <cstdint>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(FlashAttentionScoreGradBaseParamsUs1s2Bbn2)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, formerCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, remainCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, formerCoreProcessNNum);
TILING_DATA_FIELD_DEF(uint32_t, B);
TILING_DATA_FIELD_DEF(uint32_t, N2);
TILING_DATA_FIELD_DEF(uint32_t, S1);
TILING_DATA_FIELD_DEF(uint32_t, S2);
TILING_DATA_FIELD_DEF(uint32_t, G);
TILING_DATA_FIELD_DEF(uint32_t, D);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(float, keepProb);
TILING_DATA_FIELD_DEF(uint32_t, layout);
TILING_DATA_FIELD_DEF(uint32_t, preTokens);
TILING_DATA_FIELD_DEF(uint32_t, nextTokens);
TILING_DATA_FIELD_DEF(uint32_t, isSparse);
TILING_DATA_FIELD_DEF(uint32_t, maskDataType);
TILING_DATA_FIELD_DEF(uint32_t, maskShapeType);
TILING_DATA_FIELD_DEF(uint32_t, pseShapeType);
TILING_DATA_FIELD_DEF(uint32_t, existPse);
TILING_DATA_FIELD_DEF(uint32_t, existAttenMask);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskCompressMode);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskS2Size);
TILING_DATA_FIELD_DEF(uint32_t, inputBufferLen);
TILING_DATA_FIELD_DEF(uint32_t, helpBufferLen);
TILING_DATA_FIELD_DEF(uint32_t, syncDropoutLen);
TILING_DATA_FIELD_DEF(uint32_t, syncAtomicLen);
TILING_DATA_FIELD_DEF(uint32_t, syncCastLen);
TILING_DATA_FIELD_DEF(uint32_t, mm1WorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, mm2WorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, mm4InputWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, mm3InputWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, sparseMode);
TILING_DATA_FIELD_DEF(uint32_t, reserved);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, prefixN);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGradBaseParamsUs1s2Bbn2Op, FlashAttentionScoreGradBaseParamsUs1s2Bbn2)

BEGIN_TILING_DATA_DEF(SplitCoreParamsUs1s2Bbn2)
TILING_DATA_FIELD_DEF(uint32_t, baseM);
TILING_DATA_FIELD_DEF(uint32_t, baseN);
TILING_DATA_FIELD_DEF(uint32_t, singleN);
TILING_DATA_FIELD_DEF(uint32_t, baseMmm);
TILING_DATA_FIELD_DEF(uint32_t, baseNmm);
TILING_DATA_FIELD_DEF(uint32_t, singleMmm);
TILING_DATA_FIELD_DEF(uint32_t, singleNmm);
TILING_DATA_FIELD_DEF(uint32_t, s1Outer);
TILING_DATA_FIELD_DEF(uint32_t, s2OuterOuter);
TILING_DATA_FIELD_DEF(uint32_t, dInner);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SplitCoreParamsUs1s2Bbn2Op, SplitCoreParamsUs1s2Bbn2)

BEGIN_TILING_DATA_DEF(CastParams)
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
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(CastParamsOp, CastParams)

BEGIN_TILING_DATA_DEF(DropoutCastParamsUs1s2Bbn2)
TILING_DATA_FIELD_DEF(uint32_t, castBufferLen);
TILING_DATA_FIELD_DEF(uint32_t, outputBufferLen);
TILING_DATA_FIELD_DEF(uint32_t, inputBufferLen);
TILING_DATA_FIELD_DEF(uint32_t, singleUBProcessNum);
TILING_DATA_FIELD_DEF(uint32_t, dropoutWorkspaceLen);
TILING_DATA_FIELD_DEF(uint32_t, placeholder);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(DropoutCastParamsUs1s2Bbn2Op, DropoutCastParamsUs1s2Bbn2)

BEGIN_TILING_DATA_DEF(FlashAttentionScoreGradTilingDataUs1s2Bbn2)
TILING_DATA_FIELD_DEF_STRUCT(FlashAttentionScoreGradBaseParamsUs1s2Bbn2, opInfo);
TILING_DATA_FIELD_DEF_STRUCT(SplitCoreParamsUs1s2Bbn2, splitCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(CastParams, castParams);
TILING_DATA_FIELD_DEF_STRUCT(DropoutCastParamsUs1s2Bbn2, dropoutCastParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm31TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm4TilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxGradTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000020134, FlashAttentionScoreGradTilingDataUs1s2Bbn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000021134, FlashAttentionScoreGradTilingDataUs1s2Bbn2)
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGrad_10000000000000022134, FlashAttentionScoreGradTilingDataUs1s2Bbn2)

}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_FLASHATTENTIONSCOREGRAD_TILING_S1S2_BN2_DEF_H_
