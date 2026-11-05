/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#pragma once
#include "exe_graph/runtime/tiling_context.h"
#include "op_log.h"
#include <vector>
namespace optiling {

constexpr int HIGH_PRECISION = 0;
constexpr int HIGH_PERFORMANCE = 1;

const std::vector<gert::Shape> OLD_TEMPLATE_SUPPORTED_SHAPE = {
  // 5d shape [N,S1,S2,D,B]
  {5, 9216, 9216, 64, 24},
  {10, 2304, 2304, 64, 24},
  {5, 9216, 80, 64, 24},
};

// 检查是否是8月份的算子 CASE
//   True: 使用老的算子逻辑
//   False: 使用新的泛化算子逻辑
bool IsFlashAttentionScoreTilingGradOldCases(gert::TilingContext* context) {
    const gert::StorageShape* queryShape = context->GetInputShape(0);
    const gert::StorageShape* keyShape = context->GetInputShape(1);
    const char* inputLayout = context->GetAttrs()->GetAttrPointer<char>(5);
    int32_t n = *context->GetAttrs()->GetAttrPointer<int32_t>(4);  // 4 is N index

    // 1. get input shape and attr info
    int64_t b;
    int64_t s1;
    int64_t h1;
    int64_t s2;
    int64_t h2;

    if (strcmp(inputLayout, "BSH") == 0) {
        b  = queryShape->GetStorageShape().GetDim(0);
        s1 = queryShape->GetStorageShape().GetDim(1);
        h1 = queryShape->GetStorageShape().GetDim(2);
        s2 = keyShape->GetStorageShape().GetDim(1);
        h2 = keyShape->GetStorageShape().GetDim(2);
    } else if (strcmp(inputLayout, "SBH") == 0) {
        b  = queryShape->GetStorageShape().GetDim(1);
        s1 = queryShape->GetStorageShape().GetDim(0);
        h1 = queryShape->GetStorageShape().GetDim(2);
        s2 = keyShape->GetStorageShape().GetDim(0);
        h2 = keyShape->GetStorageShape().GetDim(2);
    } else {
        b  = queryShape->GetStorageShape().GetDim(0);
        s1 = queryShape->GetStorageShape().GetDim(2);
        int64_t d = queryShape->GetStorageShape().GetDim(3);
        s2 = keyShape->GetStorageShape().GetDim(2);
        if ((s1 == s2 && b == 24 && n == 10 && s1 == 2304 && d == 64) ||
            (s1 == s2 && b == 24 && n == 5 && s1 == 9216 && d == 64)) {
            auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
            int64_t coreNum = ascendcPlatform.GetCoreNumAiv();
            if (coreNum == 48) {
                return false;
            }
            return true;
        }
        return false;
    }

    // 2. check input shape
    if (h1 != h2) {
        OP_LOGD(context->GetNodeName(), "Q and K's shape H is unequal.");
        return false;
    }
    if (n==0) {
        OP_LOGD(context->GetNodeName(), "Unsupported shape, please check op_tiling/runtime/flash_attention_score_grad/white_list.h .");
        return false;
    }

    auto queryType = context->GetInputDesc(0)->GetDataType();
    const int* innerPrecisePtr = context->GetAttrs()->GetAttrPointer<int>(6);
    int innerPrecise = (innerPrecisePtr == nullptr) ? HIGH_PERFORMANCE : *innerPrecisePtr;
    if (queryType == ge::DT_FLOAT16 && innerPrecise == HIGH_PERFORMANCE) {
        // 如果是高性能场景则直接返回true,直接走白名单
        OP_LOGD(context->GetNodeName(), "fp16 high performance route old template.");
        return true;
    }

    gert::Shape input_shape_4d = {n, s1, s2, h1/n};
    gert::Shape input_shape_5d = {n, s1, s2, h1/n, b};
    bool unsupported_4d = std::find(OLD_TEMPLATE_SUPPORTED_SHAPE.begin(), OLD_TEMPLATE_SUPPORTED_SHAPE.end(), input_shape_4d) == OLD_TEMPLATE_SUPPORTED_SHAPE.end();
    bool unsupported_5d = std::find(OLD_TEMPLATE_SUPPORTED_SHAPE.begin(), OLD_TEMPLATE_SUPPORTED_SHAPE.end(), input_shape_5d) == OLD_TEMPLATE_SUPPORTED_SHAPE.end();
    if (unsupported_4d && unsupported_5d) {
        OP_LOGD(context->GetNodeName(), "Unsupported shape, please check white_list.");
        OP_LOGD(context->GetNodeName(), "current n, s1, s2, d, b is [%ld], [%ld], [%ld], [%ld], [%ld]", n, s1, s2, h1/n, b);
        return false;
    }
    return true;
}
} // namespace optiling
