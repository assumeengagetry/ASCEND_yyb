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
#include "flash_attention_score_grad_tiling_s1s2_bn2gs1s2_def.h"

namespace optiling {

constexpr uint32_t CORE_LIST_NUM = 50;
constexpr uint32_t ARRAY_LENGTH = 3;
struct FuzzyBaseInfoParams {  // 频繁使用的基础参数
  uint64_t coreNum;
  uint64_t ubSize;
  uint64_t l1Size;
  uint64_t l0aSize;
  uint64_t l0cSize;

  uint32_t b;
  uint32_t s1;
  uint32_t s1Align;
  uint32_t s1Outer;
  uint32_t s1Inner;
  uint32_t s1Tail;
  uint32_t n2;
  uint32_t s2;
  uint32_t s2Align;
  uint32_t s2Outer;
  uint32_t s2CvRatio = 1;
  uint32_t mm1CalRatio = 1;
  uint32_t cvS2Inner;
  uint32_t s2Inner;
  uint32_t s2Tail;
  uint32_t g;
  uint32_t d;
  uint32_t sfmgdOuter;
  uint32_t sfmgdInner;
  uint32_t sfmgdTail;

  uint32_t queryType;
  uint32_t pseOptional;
  uint32_t pseShapeType;
  uint32_t attenMaskOptional;
  uint32_t attenMaskShapeType;
  uint32_t attenMaskDtype;
  uint32_t attenMaskCompressMode;
  uint32_t attenMaskS1Size = 0;
  uint32_t attenMaskS2Size = 0;
  uint32_t layoutType;
  float scaleValue;
  float keepProb;

  uint32_t dataTypeSize;
  uint32_t dataBlockNum;
  uint32_t calTypeSize;
  uint32_t calBlockNum;
  uint32_t blockNum;
  uint32_t s1Token;
  uint32_t s2Token;
  uint32_t blockOuter;
  uint32_t blockFactor;

  uint32_t qSize;
  uint32_t kvSize;
  uint64_t dropMaskSize;

  uint32_t baseMN;
  uint32_t blockStarts[CORE_LIST_NUM];
  uint32_t blockEnds[CORE_LIST_NUM];
  uint32_t sparseMode;
  uint32_t prefixN[BATCH_MAX_SIZE];

  bool isSparse;
  bool isBf16;

  uint32_t tmpBufferSize = 0;

  TilingDataType mode;
};

class FlashAttentionScoreGradTilingUs1s2Bs2 : public TilingBaseClass {
 public:
  explicit FlashAttentionScoreGradTilingUs1s2Bs2(gert::TilingContext* context_) : TilingBaseClass(context_) {
  }
  ~FlashAttentionScoreGradTilingUs1s2Bs2() {
  }
  FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2 tilingData;

 protected:
  bool IsCapable() override;
  ge::graphStatus GetPlatformInfo() override;
  ge::graphStatus GetShapeAttrsInfo() override;
  ge::graphStatus DoOpTiling() override;
  ge::graphStatus DoLibApiTiling() override;
  uint64_t GetTilingKey() const override;
  ge::graphStatus GetWorkspaceSize() override;
  ge::graphStatus PostTiling() override;

  void DoSplit();
  void DoSparse();
  bool CheckFuzzyArgsLegal(uint32_t s1Inner, uint32_t s2Inner);
  std::tuple<uint32_t, uint32_t, uint32_t> FuzzyForBestSplit();
  void SetMatmulTilingBufferInfo(TCubeTiling& mmTiling);
  ge::graphStatus GetSparseBlockInfo();
  void DoPreTiling();
  void DoPostTiling();
  void DetermineMode();
  ge::graphStatus CheckAttenMaskShape();
  ge::graphStatus SaveToTilingData();
  ge::graphStatus GetSparsePrefixBlockInfo();
  bool CheckPrefixNExist(const uint32_t bIdx, const uint32_t prefixN,
                         std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &S2ValidIdx);
  void SplitBlockInS2(const uint32_t curTotalBlock, const uint32_t blockFactor, uint32_t &coreNum,
                      uint32_t blockStarts[], uint32_t blockEnds[], uint32_t &tmepBlock);
  void GetCommS1S2OuterInfo(const uint32_t s1, const uint32_t s2, uint32_t(*parseInfo)[ARRAY_LENGTH]);
  void GetParseS1S2OuterInfo(uint32_t(*parseInfo)[ARRAY_LENGTH]);
  void SetSparseParams();
  void GetCommS1S2OuterInfo(const uint32_t prefixN, std::vector<std::pair<uint32_t, uint32_t>> &S2ValidIdx);

 private:
  FuzzyBaseInfoParams fBaseParams;
};

}  // namespace optiling
