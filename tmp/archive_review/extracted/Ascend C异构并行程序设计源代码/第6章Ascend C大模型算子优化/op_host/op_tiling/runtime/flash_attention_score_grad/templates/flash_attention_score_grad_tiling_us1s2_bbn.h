/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#pragma once
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "../../base/tiling_base.h"
#include "../../data_copy_transpose_tiling.h"
#include "../../flash_attention_score_grad_tiling.h"

namespace optiling {

struct TmpBaseInfoParams { // 频繁使用的基础参数
    uint32_t coreNum;
    uint32_t b;
    uint32_t n;
    uint32_t s;
    uint32_t sq;
    uint32_t skv;
    uint32_t d;
    uint32_t g;
    uint32_t h;
    uint32_t dataTypeSize;
    uint32_t blockNum;
    uint32_t mask;
    uint32_t splitedCoreSeqNum;
    uint32_t usedUBSize;
    uint32_t layoutType;
    uint32_t attenMaskCompressMode;
    uint32_t attenMaskS1Size = 0;
    uint32_t attenMaskS2Size = 0;
    bool isSparse;
    bool isHighPrecison = false; // 高精度模式
    uint32_t dataType;
    ge::DataType dType;
    uint64_t l2CacheSize;
    uint32_t cubeTilingBaseM;
    uint32_t cubeTilingBaseN;
    SparseMode sparseMode = NO_MASK;
};


class FlashAttentionScoreGradTilingUs1s2Bbn : public TilingBaseClass {
public:
    explicit FlashAttentionScoreGradTilingUs1s2Bbn(gert::TilingContext* context_) : TilingBaseClass(context_) {
    }
    ~FlashAttentionScoreGradTilingUs1s2Bbn() {
    }

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
    ge::graphStatus GetShapeInfo();
    ge::graphStatus DoOpTilingPre();
    void GetBasicSplitBlockSize(std::pair<uint32_t, uint32_t> &split);
    bool CheckArgsLegal(uint32_t s1InnerSize, uint32_t s2InnerSize);
    bool JudgeNeedChangeCVRatio();
    void BisectionForBestSplit(uint32_t &s1InnerSize, uint32_t &s2InnerSize);
    ge::graphStatus SplitCoreTilingBN();
    ge::graphStatus SingleCoreTiling();
    ge::graphStatus DropoutTiling();
    ge::graphStatus DoPostScalarOpTiling();
    ge::graphStatus CheckAttenMaskShape();
    ge::graphStatus SetMatmulTilingV2(matmul_tiling::MatmulApiTiling& mm1,
        const uint32_t& s1InnerSize, const uint32_t& s2InnerSize, const uint32_t& l1SizeRemain,
        TCubeTiling& mm1Tiling);
    ge::graphStatus SetBMM31MatmulTilingV2(matmul_tiling::MatmulApiTiling& mm3,
        const uint32_t& s1InnerSize, const uint32_t& s2InnerSize, const uint32_t& l1SizeRemain,
        TCubeTiling& mm3Tiling);
    ge::graphStatus SetBMM4MatmulTilingV2(matmul_tiling::MatmulApiTiling& mm4,
        const uint32_t& s1InnerSize, const uint32_t& s2InnerSize, const uint32_t& l1SizeRemain,
        TCubeTiling& mm4Tiling);
    uint32_t Us1s2BbnCeilCommon(uint32_t num1, uint32_t num2);
    void PrintBaseParams();
    ge::graphStatus SetAttenMaskTilingInfo();
    void SetSparseParams();
    void CalcSOffset();

    FlashAttentionScoreGradUs1s2BbnTilingData tilingData;
    TmpBaseInfoParams mBaseParams;
};


}  // namespace optiling
