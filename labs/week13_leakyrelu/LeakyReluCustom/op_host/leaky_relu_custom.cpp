#include "leaky_relu_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cmath>
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
 const auto *shape=context->GetInputShape(0);
 const auto *attrs=context->GetAttrs();
 if (!shape || !attrs) return ge::GRAPH_FAILED;
 const float *a=attrs->GetFloat(0);
 const int64_t n=shape->GetStorageShape().GetShapeSize();
 if (!a || !std::isfinite(*a) || *a==0.0f || n<=0) return ge::GRAPH_FAILED;
 constexpr uint32_t tile=2048;
 const uint64_t tiles=(n+tile-1)/tile;
 LeakyReluCustomTilingData data;
 data.set_totalLength(n);data.set_tileLength(tile);data.set_negativeSlope(*a);
 context->SetBlockDim(tiles<8?tiles:8);
 data.SaveToBuffer(context->GetRawTilingData()->GetData(),context->GetRawTilingData()->GetCapacity());
 context->GetRawTilingData()->SetDataSize(data.GetDataSize());
 context->GetWorkspaceSizes(1)[0]=0;
 return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* c){*c->GetOutputShape(0)=*c->GetInputShape(0);return GRAPH_SUCCESS;}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* c){c->SetOutputDataType(0,c->GetInputDataType(0));return GRAPH_SUCCESS;}
}
namespace ops {
class LeakyReluCustom:public OpDef {
public:
 explicit LeakyReluCustom(const char* name):OpDef(name){
  this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  this->Attr("negative_slope").AttrType(OPTIONAL).Float(0.1);
  this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
  this->AICore().SetTiling(optiling::TilingFunc);this->AICore().AddConfig("ascend910b");
 }
};
OP_ADD(LeakyReluCustom);
}
