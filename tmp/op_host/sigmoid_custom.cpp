#include "sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t elementCount = inputShape->GetStorageShape().GetShapeSize();
    if (elementCount <= 0) {
        return ge::GRAPH_FAILED;
    }

    // Each tile fits in UB for both supported dtypes. All tile starts are
    // 32-byte aligned; DataCopyPad handles the final, possibly short tile.
    constexpr uint32_t tileLength = 2048;
    const uint64_t totalLength = static_cast<uint64_t>(elementCount);
    const uint64_t tileCount = (totalLength + tileLength - 1) / tileLength;
    const uint32_t blockCount = static_cast<uint32_t>(tileCount < 8 ? tileCount : 8);

    SigmoidCustomTilingData tiling;
    tiling.set_totalLength(totalLength);
    tiling.set_tileLength(tileLength);
    context->SetBlockDim(blockCount);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                       context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    workspaceSizes[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr) {
        return GRAPH_FAILED;
    }
    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class SigmoidCustom : public OpDef {
public:
    explicit SigmoidCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SigmoidCustom);
}
