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
#include "flash_attention_score_grad_tiling_s1s2_bn2_def.h"
#include "../../flash_attention_score_grad_tiling_common.h"

namespace optiling {

struct TempParamsUs1s2Bbn2 {
  uint32_t dataTypeSize;
  uint32_t mask;
  uint32_t queryType;
  uint32_t branch;
  uint32_t calcMode;
};

class FlashAttentionScoreGradTilingUs1s2Bbn2 : public TilingBaseClass {
 public:
  explicit FlashAttentionScoreGradTilingUs1s2Bbn2(gert::TilingContext* context_) : TilingBaseClass(context_) {
  }
  ~FlashAttentionScoreGradTilingUs1s2Bbn2() {
  }

  FlashAttentionScoreGradTilingDataUs1s2Bbn2 td_;

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
  ge::graphStatus GetLayoutInfo();
  ge::graphStatus GetBaseShapeInfo();
  ge::graphStatus GetAttenMaskInfo();
  ge::graphStatus CheckAttenMaskShape();
  ge::graphStatus DoCastTiling();
  ge::graphStatus DropoutTiling();
  ge::graphStatus SetBmm1TilingData(uint32_t sOut, uint32_t sFla, uint32_t l1SizeRemain);
  ge::graphStatus SetBmm31TilingData(uint32_t sOut, uint32_t sFla, uint32_t l1SizeRemain);
  ge::graphStatus SetBmm4TilingData(uint32_t sOut, uint32_t sFla, uint32_t l1SizeRemain);
  void DecideBranch();
  void DecideBaseMND();
  void SetSparseParams();

  TempParamsUs1s2Bbn2 tmpData_;
  uint32_t baseM;
  uint32_t baseN;
  uint32_t singleN;
  uint32_t baseD;

  uint32_t baseMmm;
  uint32_t baseNmm;
  uint32_t singleMmm;
  uint32_t singleNmm;
  bool needSplitD;
  uint32_t attenMaskCompressMode;
  uint32_t attenMaskS1Size = 0;
  uint32_t attenMaskS2Size = 0;
  SparseMode sparseMode = NO_MASK;
};

}  // namespace optiling
