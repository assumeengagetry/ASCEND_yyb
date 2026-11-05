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

class FlashAttentionScoreGradTilingUs1s2Bb : public TilingBaseClass {
public:
  explicit FlashAttentionScoreGradTilingUs1s2Bb(gert::TilingContext* context_) : TilingBaseClass(context_) {
  }
  ~FlashAttentionScoreGradTilingUs1s2Bb() {
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
};

}  // namespace optiling
