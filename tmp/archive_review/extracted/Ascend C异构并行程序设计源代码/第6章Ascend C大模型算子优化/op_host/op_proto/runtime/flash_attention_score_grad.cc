/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "runtime_util.h"
#include "op_util.h"
#include "op_const.h"

using namespace ge;
namespace ops {

ge::graphStatus InferShape4FlashAttentionScoreGrad(gert::InferShapeContext* context) {
  OP_LOGI(context->GetNodeName(), "Enter FlashAttentionScoreGrad runtime infershape impl.");
  const gert::Shape* query_shape = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, query_shape);
  const gert::Shape* key_shape = context->GetInputShape(1);
  OPS_CHECK_NULL_WITH_CONTEXT(context, key_shape);
  const gert::Shape* value_shape = context->GetInputShape(2);
  OPS_CHECK_NULL_WITH_CONTEXT(context, value_shape);

  auto attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

  auto head_num = attrs->GetInt(4);  // N1
  const char* input_layout = attrs->GetAttrPointer<char>(5);
  std::string input_layout_str = std::string(input_layout);
  for (auto& c : input_layout_str) {
    c = toupper(c);
  }
  if (input_layout_str != "BSH" && input_layout_str != "SBH" && input_layout_str != "BSND" &&
      input_layout_str != "BNSD") {
    OP_LOGE(context->GetNodeName(), "The input_layout should be BSH/SBH/BSND/BNSD(case-insensitive), but got %s.",
            input_layout_str.c_str());
    return GRAPH_FAILED;
  }

  int64_t b = 0;
  int64_t s_q = 0;
  int64_t s_kv = 0;
  if (input_layout_str == "BSH" || input_layout_str == "BSND") {
    b = query_shape->GetDim(0);
    s_q = query_shape->GetDim(1);
    s_kv = key_shape->GetDim(1);
  } else if (input_layout_str == "SBH") {
    b = query_shape->GetDim(1);
    s_q = query_shape->GetDim(0);
    s_kv = key_shape->GetDim(0);
  } else {
    // BNSD
    b = query_shape->GetDim(0);
    s_q = query_shape->GetDim(2);
    s_kv = key_shape->GetDim(2);
  }
  OP_LOGI(context->GetNodeName(), "B=%ld, N=%ld, Sq=%ld, Skv=%ld, input_layout=%s", b, *head_num, s_q, s_kv,
          input_layout_str.c_str());

  gert::Shape* dq_shape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, dq_shape);
  *dq_shape = *query_shape;

  gert::Shape* dk_shape = context->GetOutputShape(1);
  OPS_CHECK_NULL_WITH_CONTEXT(context, dk_shape);
  *dk_shape = *key_shape;

  gert::Shape* dv_shape = context->GetOutputShape(2);
  OPS_CHECK_NULL_WITH_CONTEXT(context, dv_shape);
  *dv_shape = *value_shape;

  // dpse output
  gert::Shape* dpse_shape = context->GetOutputShape(3);
  OPS_CHECK_NULL_WITH_CONTEXT(context, dpse_shape);

  const gert::Shape* pse_shape = context->GetOptionalInputShape(4);
  if (pse_shape != nullptr) {
    OP_LOGD(context->GetNodeName(), "pse_shift is not nullptr");
    *dpse_shape = *pse_shape;
  } else {
    OP_LOGD(context->GetNodeName(), "pse_shift is nullptr");
    dpse_shape->SetDimNum(4);
    dpse_shape->SetDim(0, 0);
    dpse_shape->SetDim(1, 0);
    dpse_shape->SetDim(2, 0);
    dpse_shape->SetDim(3, 0);
  }

  return GRAPH_SUCCESS;
}

ge::graphStatus InferDataType4FlashAttentionScoreGrad(gert::InferDataTypeContext* context) {
  OP_LOGI(context->GetNodeName(), "Enter FlashAttentionScoreGrad infer data type impl.");
  auto dtype = context->GetInputDataType(0);
  // dq, outidx:0
  context->SetOutputDataType(0, dtype);
  // dk, outidx:1
  context->SetOutputDataType(1, dtype);
  // dv, outidx:2
  context->SetOutputDataType(2, dtype);
  // dpse, outidx:3
  context->SetOutputDataType(3, dtype);
  return GRAPH_SUCCESS;
}

IMPL_OP(FlashAttentionScoreGrad)
    .InferShape(InferShape4FlashAttentionScoreGrad)
    .InferDataType(InferDataType4FlashAttentionScoreGrad);

}  // namespace ops
