/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#pragma once
#include <map>
#include <string>
#include <memory>

#include "tiling_base.h"
#include "error_util.h"
#include "error_log.h"
#include "exe_graph/runtime/tiling_context.h"

namespace optiling {

template <typename T>
std::unique_ptr<TilingBaseClass> TILING_CLASS(gert::TilingContext* context) {
  return std::unique_ptr<T>(new (std::nothrow) T(context));
};

using TilingClassCase = std::unique_ptr<TilingBaseClass> (*)(gert::TilingContext*);

class TilingCases {
public:
  TilingCases(const std::string& op_type):op_type_(op_type) {
  }

  template <typename T>
  void AddTiling(int32_t priority) {
    OP_TILING_CHECK(cases_.find(priority) != cases_.end(),
              VECTOR_INNER_ERR_REPORT_TILIING(op_type_, "There are duplicate registrations."),
              return);
    cases_[priority] = TILING_CLASS<T>;
    OP_TILING_CHECK(cases_[priority] == nullptr,
              VECTOR_INNER_ERR_REPORT_TILIING(op_type_, "Register op tiling func failed, please check the class name."),
              return);
  }

  const std::map<int32_t, TilingClassCase>& GetTilingCases() {
    return cases_;
  }

private:
  std::map<int32_t, TilingClassCase> cases_;
  const std::string op_type_;
};

class TilingRegistry {
public:
  TilingRegistry() {
  }
  static TilingRegistry& GetInstance() {
    static TilingRegistry registry_impl_;
    return registry_impl_;
  }

  std::shared_ptr<TilingCases> RegisterOp(const std::string& op_type) {
    if (registy_map_.find(op_type) == registy_map_.end()) {
      registy_map_[op_type] = std::shared_ptr<TilingCases>(new (std::nothrow) TilingCases(op_type));
    }
    OP_TILING_CHECK(registy_map_[op_type] == nullptr,
              VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Register tiling func failed, please check the class name."),
              return nullptr);
    return registy_map_[op_type];
  }

  ge::graphStatus DoTilingImpl(gert::TilingContext* context) {
    const char* op_type = context->GetNodeType();
    auto tilingTemplateRegistryMap = GetTilingTemplates(op_type);
    for (auto it = tilingTemplateRegistryMap.begin(); it != tilingTemplateRegistryMap.end(); ++it) {
      auto tilingTemplate = it->second(context);
      if (tilingTemplate != nullptr) {
        ge::graphStatus status = tilingTemplate->DoTiling();
        if (status != ge::GRAPH_PARAM_INVALID) {
          OP_LOGD(op_type, "Do fag general op tiling success priority=%d", it->first);
          return status;
        }
        OP_LOGD(op_type, "Ignore fag general op tiling priority=%d", it->first);
      }
    }
    VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Do op tiling failed, no valid template is found.");
    return ge::GRAPH_FAILED;
  }

  const std::map<int32_t, TilingClassCase>& GetTilingTemplates(const std::string& op_type) {
    OP_TILING_CHECK(registy_map_.find(op_type) == registy_map_.end(),
              VECTOR_INNER_ERR_REPORT_TILIING(op_type, "Get op tiling func failed, please check the op name."),
              return empty_tiling_case_);
    return registy_map_[op_type]->GetTilingCases();
  }
private:
  std::map<std::string, std::shared_ptr<TilingCases>> registy_map_;
  const std::map<int32_t, TilingClassCase> empty_tiling_case_;
};

class Register {
public:
  explicit Register(const std::string& op_type) : op_type_(op_type) {
  }

  template <typename T>
  Register& tiling(int32_t priority) {
    auto tilingCases = TilingRegistry::GetInstance().RegisterOp(op_type_);
    OP_TILING_CHECK(tilingCases == nullptr,
              VECTOR_INNER_ERR_REPORT_TILIING(op_type_, "Register op tiling failed, please the op name."),
              return *this);
    tilingCases->AddTiling<T>(priority);
    return *this;
  }

private:
  const std::string op_type_;
};

// op_type: 算子名称， class_name: 注册的 tiling 类,
// priority: tiling 类的优先级, 越小表示优先级越高, 即被选中的概率越大
#define REGISTER_TILING_TEMPLATE(op_type, class_name, priority) \
  static Register VAR_UNUSED##op_type_##class_name##priority_register = Register(op_type).tiling<class_name>(priority)

}  // namespace optiling
