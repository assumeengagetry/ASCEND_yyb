/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#pragma once
#include <memory>
#include <sstream>
#include "graph/ge_error_codes.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_ascendc.h"
#include "op_log.h"

namespace optiling {

struct AiCoreParams {
  uint64_t ubSize;
  uint64_t blockDim;
  uint64_t l1Size;
  uint64_t l0aSize;
  uint64_t l0bSize;
  uint64_t l0cSize;
};

class TilingBaseClass {
public:
  explicit TilingBaseClass(gert::TilingContext* context) : context_(context) {
  }
  virtual ~TilingBaseClass() {};

  // Tiling执行框架
  //     1、GRAPH_SUCCESS: 成功，并且不需要继续执行后续Tiling类的实现
  //     2、GRAPH_FAILED: 失败，中止整个Tiling流程
  //     3、GRAPH_PARAM_INVALID: 本类不支持，需要继续往下执行其他Tiling类的实现
  ge::graphStatus DoTiling() {
    auto ret = GetShapeAttrsInfo();
    if (ret != ge::GRAPH_SUCCESS) {
      return ret;
    }
    ret = GetPlatformInfo();
    if (ret != ge::GRAPH_SUCCESS) {
      return ret;
    }
    if (!IsCapable()) {
      return ge::GRAPH_PARAM_INVALID;
    }
    ret = DoOpTiling();
    if (ret != ge::GRAPH_SUCCESS) {
      return ret;
    }
    ret = DoLibApiTiling();
    if (ret != ge::GRAPH_SUCCESS) {
      return ret;
    }
    ret = GetWorkspaceSize();
    if (ret != ge::GRAPH_SUCCESS) {
      return ret;
    }
    ret = PostTiling();
    if (ret != ge::GRAPH_SUCCESS) {
      return ret;
    }
    context_->SetTilingKey(GetTilingKey());
    DumpTilingInfo();
    return ge::GRAPH_SUCCESS;
  }

  // 更新 context
  virtual void Reset(gert::TilingContext* context) {
    context_ = context;
  }

protected:
  virtual bool IsCapable() = 0;
  // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
  virtual ge::graphStatus GetPlatformInfo() = 0;
  // 2、获取INPUT/OUTPUT/ATTR信息
  virtual ge::graphStatus GetShapeAttrsInfo() = 0;
  // 3、计算数据切分TilingData
  virtual ge::graphStatus DoOpTiling() = 0;
  // 4、计算高阶API的TilingData
  virtual ge::graphStatus DoLibApiTiling() = 0;
  // 5、计算TilingKey
  virtual uint64_t GetTilingKey() const = 0;
  // 6、计算Workspace 大小
  virtual ge::graphStatus GetWorkspaceSize() = 0;
  // 7、保存Tiling数据
  virtual ge::graphStatus PostTiling() = 0;
  // 8、Dump Tiling数据
  virtual void DumpTilingInfo() {
    int32_t enable = CheckLogLevel(static_cast<int32_t>(OP), DLOG_DEBUG);
    if (enable != 1) {
      return;
    }
    auto buf = (uint32_t*)context_->GetRawTilingData()->GetData();
    auto bufLen = context_->GetRawTilingData()->GetDataSize();
    std::ostringstream oss;
    oss << "Start to dump tiling info. tilingkey:" << GetTilingKey() << ", tiling data size:" << bufLen << ", content:";
    for (size_t i = 0; i < bufLen / sizeof(uint32_t); i++) {
      oss << *(buf + i) << ",";
    }
    OP_LOGD(context_->GetNodeType(), "%s", oss.str().c_str());
  }

protected:
  gert::TilingContext* context_;
  std::unique_ptr<platform_ascendc::PlatformAscendC> ascendcPlatform_{nullptr};
  uint32_t blockDim_{0};
  uint32_t workspaceSize_{0};
  uint64_t tilingKey_{0};
  AiCoreParams aicoreParams_;
};

}  // namespace optiling
