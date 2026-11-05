/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FLASHATTENTIONSCOREGRAD_BNGS1S2_B_TILING_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FLASHATTENTIONSCOREGRAD_BNGS1S2_B_TILING_H_

#pragma once
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "flash_attention_score_grad_tiling_bngs1s2_b_def.h"
#include "../../base/tiling_base.h"
#include "../../data_copy_transpose_tiling.h"
#include "../../flash_attention_score_grad_tiling.h"

namespace optiling {
struct Ubngs1s2BbTempParams { // 频繁使用的中间态临时变量
    uint32_t dataType;
    uint32_t precisionMode {1};
    uint32_t layout;
};

class FlashAttentionScoreGradUbngs1s2BbTiling : public TilingBaseClass {
public:
    explicit FlashAttentionScoreGradUbngs1s2BbTiling(gert::TilingContext* context_) : TilingBaseClass(context_) {}
    ~FlashAttentionScoreGradUbngs1s2BbTiling() override {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    FlashAttentionScoreGradUbngs1s2BbTilingData td_;
    Ubngs1s2BbTempParams basicParams;
    uint32_t GetApiTmpBufferSize(int64_t bIn, int64_t sKV);
    bool CheckArgsLegal(int64_t bIn);
    ge::graphStatus DoCoresSplitTiling();
    ge::graphStatus DoInCoreTiling();
    ge::graphStatus SetMm1AndMm2Tiling(matmul_tiling::MatmulApiTiling& mm1AndMm2, int32_t l1SizeRemain, int64_t nIn,
        TCubeTiling& mm1AndMm2Tiling);
    ge::graphStatus SetMm31Tiling(matmul_tiling::MatmulApiTiling& mm31, int32_t l1SizeRemain, int64_t nIn,
        TCubeTiling& mm31Tiling);
    ge::graphStatus SetMm32AndMm4Tiling(matmul_tiling::MatmulApiTiling& mm32AndMm4, int32_t l1SizeRemain, int64_t nIn,
        TCubeTiling& mm32AndMm4Tiling);
};

}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_FLASHATTENTIONSCOREGRAD_BNGS1S2_B_TILING_H_
