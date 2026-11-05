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
#include "flash_attention_score_grad_tiling_bng_gs1s2_def.h"
#include "../../base/tiling_base.h"
#include "../../data_copy_transpose_tiling.h"

namespace optiling {
struct BaseInfoParams {
  // 频繁使用的基础参数
  uint32_t coreNum;
  uint32_t ubSize;
  uint32_t l1Size;
  uint32_t l0aSize;
  uint32_t l0bSize;
  uint32_t l0cSize;

  uint32_t b;
  uint32_t n1;
  uint32_t n2;
  uint32_t s1;
  uint32_t s2;
  uint32_t g;
  uint32_t d;
  uint32_t h;
  uint32_t blockNum;
  uint32_t mask;

  uint32_t tilingKey;
  uint32_t usedUBSize;

  uint32_t layout;

  bool isSparse;
  bool isFlash;
  uint32_t dataType;
  uint32_t dataTypeSize;
  uint32_t calTypeSize;
  ge::DataType dType;

  // 基本块信息
  uint32_t gInner;
};

class FlashAttentionScoreGradTilingBNGgs1s2 : public TilingBaseClass {
  constexpr static int64_t BLOCK_SIZE = 32;
  constexpr static int64_t DATA_SIZE_FP32 = 4;
  constexpr static int64_t DATA_SIZE_FP16 = 2;
  constexpr static int64_t BLOCK_COUNT_FP32 = BLOCK_SIZE / DATA_SIZE_FP32;
  constexpr static int64_t BLOCK_COUNT_FP16 = BLOCK_SIZE / DATA_SIZE_FP16;
  constexpr static int64_t VMASK_FP16 = 128;
  constexpr static int64_t VMASK_FP32 = 64;
  constexpr static int64_t DIM_INDEX_0 = 0;
  constexpr static int64_t DIM_INDEX_1 = 1;
  constexpr static int64_t DIM_INDEX_2 = 2;
  constexpr static int64_t DIM_INDEX_3 = 3;
  constexpr static int64_t DIM_INDEX_4 = 4;
  constexpr static int64_t LAYOUT_INDEX_0 = 0;
  constexpr static int64_t LAYOUT_INDEX_1 = 1;
  constexpr static int64_t LAYOUT_INDEX_2 = 2;
  constexpr static int64_t LAYOUT_INDEX_3 = 3;
  constexpr static int64_t BSH = 0;
  constexpr static int64_t SBH = 1;
  constexpr static int64_t BNSD = 2;
  constexpr static int64_t BSND = 3;

 public:
  explicit FlashAttentionScoreGradTilingBNGgs1s2(gert::TilingContext* context_) : TilingBaseClass(context_) {
  }

  ~FlashAttentionScoreGradTilingBNGgs1s2() override {
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
  FlashAttentionScoreGradTilingDataBNGgs1s2 tilingData;
  BaseInfoParams mBaseInfoParams;
  bool CheckGInnerLegal(uint32_t gInner);
  uint32_t CalcGInner();
  ge::graphStatus SetBaseTilingData();
  ge::graphStatus SetCoresSplitTilingData();
  ge::graphStatus SetUbSplitTilingData();
  void SetMatmulTiling1(matmul_tiling::MatmulApiTiling& mm, TCubeTiling& mmTiling);
  void SetMatmulTiling2(matmul_tiling::MatmulApiTiling& mm, TCubeTiling& mmTiling);
  void SetMatmulTiling3(matmul_tiling::MatmulApiTiling& mm, TCubeTiling& mmTiling);
};
}  // namespace optiling