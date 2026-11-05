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

#include "graph/utils/type_utils.h"
#include "op_const.h"
#include "op_util.h"
#include "runtime_util.h"

using namespace ge;
namespace ops {

constexpr int FLA_SOFTMAXMAX_F32_DIM0SHAPE = 8;

ge::graphStatus InferShapeFlashAttentionScore(gert::InferShapeContext* context) {
  OP_LOGI(context->GetNodeName(), "Enter FlashAttentionScore runtime infershape impl.");

  const gert::Shape* query_shape = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, query_shape);
  auto attrs = context->GetAttrs();
  const auto* query_tensor = context->GetInputTensor(0);

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

  int64_t shapeB = 1;
  int64_t shapeS = 1;

  if (input_layout_str == "BSH" || input_layout_str == "BSND") {
    shapeB = query_shape->GetDim(0);
    shapeS = query_shape->GetDim(1);
  } else if (input_layout_str == "SBH") {
    shapeB = query_shape->GetDim(1);
    shapeS = query_shape->GetDim(0);
  } else {
    // BNSD
    shapeB = query_shape->GetDim(0);
    // BNSD中S的shape
    shapeS = query_shape->GetDim(2);
  }

  auto headNum = attrs->GetInt(4);
  OP_LOGI(context->GetNodeName(), "B=%ld, N=%ld, S=%ld, input_layout=%s, dtype=%s", shapeB, *headNum, shapeS,
          input_layout_str.c_str(), ge::TypeUtils::DataTypeToSerialString(query_tensor->GetDataType()).c_str());

  // softmaxMax, fp32: (B, N, S, 8)
  gert::Shape* softmaxMaxShape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, softmaxMaxShape);
  // 0, 1, 2, 3, 4 : dim idx
  softmaxMaxShape->SetDimNum(4);
  softmaxMaxShape->SetDim(0, shapeB);
  softmaxMaxShape->SetDim(1, *headNum);
  softmaxMaxShape->SetDim(2, shapeS);
  softmaxMaxShape->SetDim(3, FLA_SOFTMAXMAX_F32_DIM0SHAPE);

  // softmaxSum, shape same as softmaxMax
  gert::Shape* softmaxSumShape = context->GetOutputShape(1);
  OPS_CHECK_NULL_WITH_CONTEXT(context, softmaxSumShape);
  *softmaxSumShape = *softmaxMaxShape;

  // softmaxOut, shape: (B, N, S, S)
  gert::Shape* softmaxOutShape = context->GetOutputShape(2);
  OPS_CHECK_NULL_WITH_CONTEXT(context, softmaxOutShape);
  // 0, 1, 2, 3, 4 : dim idx
  softmaxOutShape->SetDimNum(4);
  softmaxOutShape->SetDim(0, 0);
  softmaxOutShape->SetDim(1, 0);
  softmaxOutShape->SetDim(2, 0);
  softmaxOutShape->SetDim(3, 0);

  gert::Shape* attentionOutShape = context->GetOutputShape(3);
  OPS_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);
  *attentionOutShape = *query_shape;

  return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeFlashAttentionScore(gert::InferDataTypeContext* context) {
  auto dtype = context->GetInputDataType(0);
  // softmax_max, outidx:0
  context->SetOutputDataType(0, DT_FLOAT);
  // softmax_sum, outidx:1
  context->SetOutputDataType(1, DT_FLOAT);
  // softmax_out, outidx:2
  context->SetOutputDataType(2, dtype);
  // attention_out, outidx:3
  context->SetOutputDataType(3, dtype);
  return GRAPH_SUCCESS;
}

IMPL_OP(FlashAttentionScore).InferShape(InferShapeFlashAttentionScore).InferDataType(InferDataTypeFlashAttentionScore);

}  // namespace ops
