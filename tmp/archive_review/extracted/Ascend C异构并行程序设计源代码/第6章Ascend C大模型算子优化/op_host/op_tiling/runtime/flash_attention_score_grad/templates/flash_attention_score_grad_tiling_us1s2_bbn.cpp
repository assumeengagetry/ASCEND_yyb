/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */
#include "../../base/tiling_templates_registry.h"
#include "../../base/tiling_type.h"
#include "../../base/tiling_base.h"
#include "tiling/tiling_api.h"
#include "../../data_copy_transpose_tiling.h"
#include "platform/platform_info.h"
#include "op_tiling_util.h"
#include <cstdint>
#include <map>
#include <functional>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cstdlib>
#include <dlfcn.h>
#include <unistd.h>
#include <cstdio>
#include <cmath>
#include "flash_attention_score_grad_tiling_us1s2_bbn_def.h"
#include "flash_attention_score_grad_tiling_us1s2_bbn.h"

using namespace ge;
using namespace AscendC;

namespace optiling {

#define CHECK_ZERO(num) \
    do {                                              \
        if ((num) == 0) {                             \
            return ge::GRAPH_FAILED;                  \
        }                                             \
    } while (0)

constexpr uint32_t DIM_COUNT_NUM_3 = 3;
constexpr uint32_t DIM_COUNT_NUM_4 = 4;
constexpr uint32_t ALIGN_SIZE_16 = 16;
constexpr uint32_t INNER_MATRES_NUM = 5;
constexpr uint32_t NUM2 = 2;
constexpr double L2_CACHE_SIZE_PERCENT = 0.75;
constexpr uint32_t WORKSPACE_ALIGNSIZE = 512;
constexpr uint32_t IS_SPARSE = 1;
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t SIXTY_KBYTES = 60 * 1024;
constexpr uint32_t THIRTY_KBYTES = 30 * 1024;

constexpr uint32_t S1_S2_RATIO = 8;
constexpr uint32_t MAX_STRIDE_LIMIT = 65535;
constexpr int64_t ATTEN_MASK_TYPE_11SS_DIM_NUM = 2;

enum class LayoutType {
    BSH = 0,
    SBH,
    BNSD,
    BSND
};

const std::map<uint32_t, int64_t> BEST_BASIC_BLOCK_SIZE {
    {static_cast<uint32_t>(ge::DT_FLOAT16), 16384},
    {static_cast<uint32_t>(ge::DT_BF16), 8192},
    {static_cast<uint32_t>(ge::DT_FLOAT), 4096}
};

enum class SplitDtypeEnum {
    FLOAT16 = 0,
    BFLOAT16 = 1,
    FLOAT32 = 2,
    FLOAT16_PRECISION = 3,
};

// 不同类型设置初始切分大小
const std::map<SplitDtypeEnum, std::pair<uint32_t, uint32_t>> BEST_BASIC_SPLIT_BLOCK_SIZE {
    {SplitDtypeEnum::FLOAT16, std::make_pair(128, 128)},
    {SplitDtypeEnum::BFLOAT16, std::make_pair(128, 64)},
    {SplitDtypeEnum::FLOAT32, std::make_pair(64, 64)},
    {SplitDtypeEnum::FLOAT16_PRECISION, std::make_pair(128, 64)}
};

inline uint32_t Align(const uint32_t n, const uint32_t alignSize)
{
    return (n + alignSize - 1) & (~(alignSize - 1));
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bbn::SetAttenMaskTilingInfo() {
    auto attenMaskDesc = context_->GetOptionalInputDesc(ATTEN_MASK);
    if (attenMaskDesc != nullptr) {
        auto attenMaskDtype = attenMaskDesc->GetDataType();
        auto queryType = context_->GetInputDesc(QUERY)->GetDataType();
        if (attenMaskDtype == queryType) {
            tilingData.baseParams.set_attenMaskType(ATTEN_MASK_TYPE_SAME);
        } else {
            tilingData.baseParams.set_attenMaskType(ATTEN_MASK_TYPE_U8_BOOL);
        }
    }
    mBaseParams.attenMaskCompressMode = NO_COMPRESS_MODE;
    auto attenMaskShape = context_->GetOptionalInputShape(ATTEN_MASK);
    if (attenMaskShape != nullptr && attenMaskShape->GetStorageShape().GetShapeSize() != 0) {
        tilingData.baseParams.set_hasAttenMaskInput(1);
        auto storageShape = attenMaskShape->GetStorageShape();
        auto maskShapeDims = storageShape.GetDimNum();
        if (maskShapeDims == ATTEN_MASK_TYPE_11SS_DIM_NUM) {
            tilingData.baseParams.set_attenMaskShapeType(ATTEN_MASK_SHAPE_TYPE_SS);
        } else if (maskShapeDims == DIM_COUNT_NUM_4) {
            auto dim0 = storageShape.GetDim(DIM_0);
            auto dim1 = storageShape.GetDim(DIM_1);
            if (dim0 == 1 && dim1 == 1) {
                OP_LOGD(context_->GetNodeName(),
                        "FlashAttentionScoreGradTilingUs1s2Bbn attenMaskShape 11SS");
                tilingData.baseParams.set_attenMaskShapeType(ATTEN_MASK_SHAPE_TYPE_SS);
            } else if (dim0 == mBaseParams.b && dim1 == 1) {
                OP_LOGD(context_->GetNodeName(),
                        "FlashAttentionScoreGradTilingUs1s2Bbn attenMaskShape B1SS");
                tilingData.baseParams.set_attenMaskShapeType(ATTEN_MASK_SHAPE_TYPE_B1SS);
            } else if (dim0 == mBaseParams.b && dim1 == mBaseParams.n) {
                OP_LOGD(context_->GetNodeName(),
                        "FlashAttentionScoreGradTilingUs1s2Bbn attenMaskShape BNSS");
                tilingData.baseParams.set_attenMaskShapeType(ATTEN_MASK_SHAPE_TYPE_BNSS);
            } else {
                OP_LOGE(context_->GetNodeName(),
                        "FlashAttentionScoreGradTilingUs1s2Bbn not support attenMaskShape dim0: %ld, dim1: %ld", dim0, dim1);
                return ge::GRAPH_FAILED;
            }
        } else {
            OP_LOGE(context_->GetNodeName(),
                    "FlashAttentionScoreGradTilingUs1s2Bbn not support attenMaskShape dim num: %zu", maskShapeDims);
            return ge::GRAPH_FAILED;
        }
        mBaseParams.attenMaskS2Size = storageShape.GetDim(maskShapeDims - LAST_AXIS_IDX);
        mBaseParams.attenMaskS1Size = storageShape.GetDim(maskShapeDims - SEC_LAST_AXIS_IDX);
        if (mBaseParams.sparseMode == LEFT_UP_CAUSAL) {
            mBaseParams.attenMaskCompressMode = LEFT_UP_CAUSAL_MODE;
        } else if (mBaseParams.sparseMode == RIGHT_DOWN_CAUSAL) {
            mBaseParams.attenMaskCompressMode = RIGHT_DOWN_CAUSAL_MODE;
        }
    }
    tilingData.baseParams.set_attenMaskS2Size(mBaseParams.attenMaskS2Size);
    tilingData.baseParams.set_attenMaskCompressMode(mBaseParams.attenMaskCompressMode);
    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreGradTilingUs1s2Bbn::SetSparseParams()
{
    tilingData.splitCoreParams.set_sparse(0);
    tilingData.baseParams.set_sparseMode(mBaseParams.sparseMode);
    auto attenMaskShape = context_->GetOptionalInputShape(ATTEN_MASK);
    auto attenMaskDesc = context_->GetOptionalInputDesc(ATTEN_MASK);
    if (mBaseParams.sq != mBaseParams.skv || mBaseParams.sparseMode == ALL_MASK || attenMaskShape == nullptr ||
        attenMaskDesc == nullptr) {
        OP_LOGI(context_->GetNodeName(),
            "FAG Us1s2Bbn s1[%u] and s2[%u] are not equal, sparseMode[%d] is all_mask or attenmask is nullptr," \
            "not support sparse mode!", mBaseParams.sq, mBaseParams.skv, mBaseParams.sparseMode);
        return;
    }

    uint32_t preTokens = *context_->GetAttrs()->GetAttrPointer<int>(PRE_TOKENS); // 2
    uint32_t nextTokens = *context_->GetAttrs()->GetAttrPointer<int>(NEXT_TOKENS); // 3
    tilingData.baseParams.set_preTokens(preTokens);
    tilingData.baseParams.set_nextTokens(nextTokens);
    OP_LOGD(context_->GetNodeName(), "preTokens: %u, nextTokens: %u, sparseMode: %d.",
            preTokens, nextTokens, mBaseParams.sparseMode);

    if (mBaseParams.sparseMode == NO_MASK) {
        if (mBaseParams.sq <= preTokens && nextTokens == 0) { // 原causal 场景
            tilingData.splitCoreParams.set_sparse(1);
        } else if (mBaseParams.sq > preTokens || mBaseParams.skv > nextTokens) { // 原band 场景
            tilingData.splitCoreParams.set_sparse(1);
        } else {
            OP_LOGI(context_->GetNodeName(),
                    "FAG Us1s2Bbn sparseMode[%d] with s1[%u] s2[%u] preToken[%u] nextToken[%u] is not support!",
                    mBaseParams.sparseMode, mBaseParams.sq, mBaseParams.skv, preTokens, nextTokens);
        }
        return;
    }

    if (mBaseParams.sparseMode != RIGHT_DOWN_CAUSAL && mBaseParams.sparseMode != BAND &&
        mBaseParams.sparseMode != PREFIX) {
        OP_LOGW(context_->GetNodeName(), "FAG Us1s2Bbn sparseMode[%d] is not support!", mBaseParams.sparseMode);
        return;
    }

    if (mBaseParams.sparseMode == BAND) {
        if (mBaseParams.sq <= preTokens && mBaseParams.skv <= preTokens) {
            OP_LOGW(context_->GetNodeName(),
                    "FAG Us1s2Bbn sparseMode is band, but preToken[%u] or nextToken[%u] is invalid with s1[%u] s2[%u]!",
                    preTokens, nextTokens, mBaseParams.sq, mBaseParams.skv);
            tilingData.splitCoreParams.set_sparse(0);
            return;
        }
    }

    if (mBaseParams.sparseMode == PREFIX) {
        auto prefixNTensor = context_->GetOptionalInputTensor(PREFIX_N);
        if (prefixNTensor != nullptr) {
            auto &prefixShape = prefixNTensor->GetShape().GetStorageShape();
            if (prefixShape.GetDimNum() != 1 || prefixShape.GetDim(0) != mBaseParams.b) {
                OP_LOGW(context_->GetNodeName(),
                    "FAG Us1s2Bbn sparseMode is prefix, but prefixshape size[%zu] or value is invalid!",
                    prefixShape.GetDimNum());
                return;
            }
            std::vector<int64_t> prefixN;
            const int64_t* value = prefixNTensor->GetData<int64_t>();
            const size_t shapeSize = prefixNTensor->GetShapeSize();
            for (size_t i = 0; i < shapeSize; i++) {
                prefixN.push_back(value[i]);
            }

            if (prefixN.size() == mBaseParams.b && prefixN.size() < BATCH_MAX_SIZE &&
                (std::find_if(prefixN.begin(), prefixN.end(),
                            [=] (uint32_t i) { return i >= mBaseParams.skv;}) == prefixN.end())) {
                uint32_t tempPrefixN[BATCH_MAX_SIZE];
                std::copy(prefixN.begin(), prefixN.end(), tempPrefixN);
                tilingData.baseParams.set_prefixN(tempPrefixN);
            } else {
                OP_LOGW(context_->GetNodeName(),
                    "FAG Us1s2Bbn sparseMode is prefix, but prefixN size[%zu] or value is invalid!", prefixN.size());
                return;
            }
        } else {
            OP_LOGW(context_->GetNodeName(), "FAG Us1s2Bbn sparseMode is prefix, but prefixN tensor is null!");
            return;
        }
    }
    tilingData.splitCoreParams.set_sparse(1);
    return;
}

// 1、获取INPUT/OUTPUT/ATTR信息
ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bbn::GetShapeAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "Get shape attrs informations.");
    /* 1. 获取属性信息 */
    tilingData.baseParams.set_scaleValue(*(context_->GetAttrs()->GetAttrPointer<float>(SCALE_VALUE))); // 0
    tilingData.baseParams.set_keepProb(*(context_->GetAttrs()->GetAttrPointer<float>(KEEP_PROB))); // 1
    OP_LOGD(context_->GetNodeName(), "ScaleValue: %u, keepProb: %u.",
            tilingData.baseParams.get_scaleValue(), tilingData.baseParams.get_keepProb());
    mBaseParams.isHighPrecison = true;

    /* 2. 获取shape和轴信息 */
    if (GetShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_PARAM_INVALID;
    }
    tilingData.baseParams.set_B(mBaseParams.b);
    tilingData.baseParams.set_N(mBaseParams.n);
    tilingData.baseParams.set_S(mBaseParams.s);
    tilingData.baseParams.set_D(mBaseParams.d);
    tilingData.baseParams.set_H(mBaseParams.h);
    tilingData.baseParams.set_batch(mBaseParams.b * mBaseParams.n);
    tilingData.baseParams.set_layoutType(mBaseParams.layoutType);
    tilingData.baseParams.set_Sq(mBaseParams.sq);
    tilingData.baseParams.set_Skv(mBaseParams.skv);

    /* 3. 获取data type信息 */
    auto queryType = context_->GetInputDesc(QUERY)->GetDataType(); // 0
    mBaseParams.dType = queryType;
    mBaseParams.dataTypeSize = DEFAULT_DATA_TYPE_SIZE; // init date type bf16 is 4
    mBaseParams.mask = DEFAULT_MASK;  // 64 is init mask
    mBaseParams.dataType = 0;

    if (queryType == ge::DT_FLOAT16) {
        mBaseParams.dataTypeSize = FP16_DATA_TYPE_SIZE;  // 2 is fp16
        mBaseParams.mask = FP16_MASK;  // fp16 init mask is 128
        mBaseParams.dataType = FP16;
    } else if (queryType == ge::DT_BF16) {
        mBaseParams.dataTypeSize = BF16_DATA_TYPE_SIZE;  // 2 is bf16
        mBaseParams.mask = BF16_MASK;  // bf16 init mask is 64
        mBaseParams.dataType = BF16;  // 2 is bf16
    }

    /* 4. 获取其他输入shape信息 */
    auto pseShape = context_->GetOptionalInputShape(PSE_SHIFT); // 4
    if (pseShape != nullptr && pseShape->GetStorageShape().GetShapeSize() != 0) {
        tilingData.baseParams.set_hasPseInput(1);
        auto dim2 = pseShape->GetStorageShape().GetDim(DIM_2);
        if (dim2 != gert::Shape::kInvalidDimValue && dim2 == 1) {
            tilingData.baseParams.set_pseShapeType(PSE_SHAPE_TYPE_BN1S);
        } else {
            tilingData.baseParams.set_pseShapeType(PSE_SHAPE_TYPE_BNSS);
        }
    }

    // 新增SPARSE_MODE属性，上库兼容处理
    auto attrs = context_->GetAttrs();
    if (attrs->GetAttrNum() > static_cast<size_t>(SPARSE_MODE)) {
        mBaseParams.sparseMode = SparseMode(*(attrs->GetAttrPointer<int>(SPARSE_MODE))); // 7
    }

    if (SetAttenMaskTilingInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    tilingData.baseParams.set_allReduceNum(mBaseParams.s * BLOCK_BYTES / sizeof(float));
    mBaseParams.blockNum = BYTE_BLOCK / mBaseParams.dataTypeSize;

    SetSparseParams();
    OP_LOGD(context_->GetNodeName(), "FAG Us1s2Bbn sparse %s.",
            tilingData.splitCoreParams.get_sparse() == 0 ? "disable" : "enable");

    return ge::GRAPH_SUCCESS;
}

// 2、是否满足执行条件
bool FlashAttentionScoreGradTilingUs1s2Bbn::IsCapable()
{
    OP_LOGD(context_->GetNodeName(), "Check is capable.");
    if (mBaseParams.g != 1) {
        OP_LOGI(context_->GetNodeName(), "G(%u) is not 1.", mBaseParams.g);
        return false;
    }
    if (mBaseParams.b * mBaseParams.n < mBaseParams.coreNum) {
        OP_LOGI(context_->GetNodeName(), "Block is smaller than core num: b(%u), n(%u), coreNum(%u).",
                mBaseParams.b, mBaseParams.n, mBaseParams.coreNum);
        return false;
    }

    int64_t baseBlockSize;
    auto iter = BEST_BASIC_BLOCK_SIZE.find(mBaseParams.dType);
    if (iter == BEST_BASIC_BLOCK_SIZE.end()) {
        OP_LOGI(context_->GetNodeName(), "Dtype(%u) is not supported.",
                static_cast<uint32_t>(mBaseParams.dType));
        return false;
    }

    baseBlockSize = iter->second;
    if (mBaseParams.dType == ge::DT_FLOAT16 && mBaseParams.isHighPrecison) {
        baseBlockSize = baseBlockSize / 2;
    }

    int64_t gSqSkvSize = mBaseParams.sq * Align(mBaseParams.skv, ALIGN_SIZE_16);
    if (gSqSkvSize < baseBlockSize) {
        OP_LOGI(context_->GetNodeName(), "SqSkvSize(%u) is smaller than baseBlockSize(%u).",
                gSqSkvSize, baseBlockSize);
        return false;
    }

    if (static_cast<double>(gSqSkvSize * aicoreParams_.blockDim * mBaseParams.dataTypeSize) >=
        static_cast<double>(mBaseParams.l2CacheSize) * L2_CACHE_SIZE_PERCENT) {
        OP_LOGI(context_->GetNodeName(), "L2cache size is not enough: SqSkvBlockSize(%f), l2CacheSizePercent(%f).",
                static_cast<double>(gSqSkvSize * aicoreParams_.blockDim * mBaseParams.dataTypeSize),
                static_cast<double>(mBaseParams.l2CacheSize) * L2_CACHE_SIZE_PERCENT);
        return false;
    }

    OP_LOGD(context_->GetNodeName(), "IsCapable check ok.");
    return true;
}


// 3、平台信息比如CoreNum、UB/L1/L0C资源大小
ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bbn::GetPlatformInfo()
{
    OP_LOGD(context_->GetNodeName(), "Get platform informations.");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    aicoreParams_.blockDim = coreNum;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, aicoreParams_.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, aicoreParams_.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, aicoreParams_.l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, aicoreParams_.l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, aicoreParams_.l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, mBaseParams.l2CacheSize);
    mBaseParams.coreNum = coreNum;
    tilingData.baseParams.set_coreNum(coreNum);

    // 打印下所有的参数
    PrintBaseParams();
    return ge::GRAPH_SUCCESS;
}

// 4、计算数据切分TilingData
ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bbn::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "Do op tiling.");
    // 切分s1InnerSize、s2InnerSize，设置到tilingData
    auto ret = DoOpTilingPre();
    if (ret != ge::GRAPH_SUCCESS) {
        return GRAPH_PARAM_INVALID;
    }

    // 核切BN
    ret = SplitCoreTilingBN();
    if (ret != ge::GRAPH_SUCCESS) {
        return GRAPH_PARAM_INVALID;
    }

    // 单核tiling
    ret = SingleCoreTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return GRAPH_PARAM_INVALID;
    }

    // dropout tiling
    ret = DropoutTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return GRAPH_PARAM_INVALID;
    }

    ret = DoPostScalarOpTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return GRAPH_PARAM_INVALID;
    }

    if (CheckAttenMaskShape() != SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 5、计算高阶API的TilingData
ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bbn::DoLibApiTiling()
{
    OP_LOGD(context_->GetNodeName(), "Do lib api tiling.");
    // mm tiling
    uint32_t s1InnerSize = tilingData.splitCoreParams.get_s1InnerSize();
    uint32_t s2InnerSize = tilingData.splitCoreParams.get_s2InnerSize();

    uint32_t scmSize = 0;
    uint32_t l1SizeRemain = aicoreParams_.l1Size - scmSize;
    OP_LOGD(context_->GetNodeName(), "ScmSize is %u, l1SizeRemain is %u.", scmSize, l1SizeRemain);

    matmul_tiling::MatmulApiTiling mm1;
    matmul_tiling::MatmulApiTiling mm2;
    matmul_tiling::MatmulApiTiling mm3;
    matmul_tiling::MatmulApiTiling mm4;
    // BMM1/2
    ge::graphStatus ret = SetMatmulTilingV2(mm1, s1InnerSize, s2InnerSize, l1SizeRemain, tilingData.mm1TilingData);
    if (ret != ge::GRAPH_SUCCESS) {
        return GRAPH_PARAM_INVALID;
    }
    // BMM3.1
    ret = SetBMM31MatmulTilingV2(mm3, s1InnerSize, s2InnerSize, l1SizeRemain, tilingData.mm3TilingData);
    if (ret != ge::GRAPH_SUCCESS) {
        return GRAPH_PARAM_INVALID;
    }
    // BMM3.2/BMM4
    ret = SetBMM4MatmulTilingV2(mm4, s1InnerSize, s2InnerSize, l1SizeRemain, tilingData.mm4TilingData);
    if (ret != ge::GRAPH_SUCCESS) {
        return GRAPH_PARAM_INVALID;
    }

    // bf16和高精度模式，需要加上castBuff
    uint32_t softmaxApiDataTypeSize = mBaseParams.dataTypeSize;
    if (mBaseParams.dataType == BF16 || mBaseParams.isHighPrecison) {
        softmaxApiDataTypeSize = softmaxApiDataTypeSize * 2; // float
    }

    uint32_t apiTmpSize = aicoreParams_.ubSize - mBaseParams.usedUBSize;
    auto softmaxShape = Shape({s1InnerSize, s2InnerSize});
    int64_t softmaxTmpSize = GetSoftMaxMinTmpSize(softmaxShape, sizeof(float), true);
    int64_t softmaxGradTmpSize = GetSoftMaxGradMinTmpSize(softmaxShape, mBaseParams.dataTypeSize, true, true);

    OP_LOGD(context_->GetNodeName(), "ApiTmpSize is %u, softmaxTmpSize is %ld, softmaxGradTmpSize is %ld.",
            apiTmpSize, softmaxTmpSize, softmaxGradTmpSize);
    SoftMaxTilingFunc(softmaxShape, sizeof(float), s1InnerSize * s2InnerSize * sizeof(float),
                      tilingData.softmaxTilingData);
    SoftMaxGradTilingFunc(softmaxShape, softmaxApiDataTypeSize,
                          s1InnerSize * s2InnerSize * sizeof(float) / NUM2, tilingData.softmaxGradTilingData, true);
    return ge::GRAPH_SUCCESS;
}

// 6、计算Workspace 大小
ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bbn::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

// 7、保存Tiling数据
ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bbn::PostTiling()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    auto aicNum = ascendcPlatform.GetCoreNumAic();
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    context_->SetBlockDim(ascendcPlatform.CalcTschBlockDim(tilingData.baseParams.get_coreNum(), aicNum, aivNum));
    OP_LOGD(context_->GetNodeName(), "AicNum is %u, aivNum is %u.", aicNum, aivNum);

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    uint32_t inputTypeNum = BYTE_BLOCK / mBaseParams.dataTypeSize;
    uint32_t skvAlignSize = (tilingData.baseParams.get_Skv() + inputTypeNum - 1) / inputTypeNum * inputTypeNum;
    uint32_t s2AlignSizeByBaseN = (tilingData.splitCoreParams.get_cubeBaseS2() +
                                   tilingData.splitCoreParams.get_s2InnerSize() - 1) /
                                  tilingData.splitCoreParams.get_s2InnerSize() *
                                  tilingData.splitCoreParams.get_s2InnerSize();
    uint32_t bmm12WorkspaceSize = tilingData.splitCoreParams.get_cubeBaseS1() * s2AlignSizeByBaseN;
    if (mBaseParams.dataType == BF16 || mBaseParams.isHighPrecison) {
        bmm12WorkspaceSize *= 2;
    }
    uint32_t syncDropoutLen = 3200;
    tilingData.baseParams.set_syncDropoutLen(syncDropoutLen);
    uint32_t dropoutWorkspaceLen = tilingData.dropoutCastParams.get_dropoutWorkspaceLen();

    // post Cast Scalar
    uint32_t dqPostWorkspaceLen = tilingData.postOpTilingParams.get_dqWorkspaceLen();
    uint32_t dkPostWorkspaceLen = tilingData.postOpTilingParams.get_dkWorkspaceLen();
    uint32_t dvPostWorkspaceLen = tilingData.postOpTilingParams.get_dvWorkspaceLen();
    uint64_t totalPostOpWorkspaceSize = dqPostWorkspaceLen + dkPostWorkspaceLen + dvPostWorkspaceLen;

    uint64_t workspaceSize = 16 * 1024 * 1024 + 800 * mBaseParams.dataTypeSize +
                             (bmm12WorkspaceSize +
                             tilingData.baseParams.get_Sq() * skvAlignSize) * 2 * aivNum * mBaseParams.dataTypeSize +
                             syncDropoutLen + dropoutWorkspaceLen + totalPostOpWorkspaceSize;
    workspaces[0] = Align(workspaceSize, WORKSPACE_ALIGNSIZE);

    OP_LOGD(context_->GetNodeName(), "InputTypeNum: %u, skvAlignSize: %u, bmm12WorkspaceSize: %u, workspaceSize: %lu.",
            inputTypeNum, skvAlignSize, bmm12WorkspaceSize, workspaceSize);
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

// 8、设置tiling key
uint64_t FlashAttentionScoreGradTilingUs1s2Bbn::GetTilingKey() const
{
    uint64_t tilingKey = 0;
    // 模板特性要求生成tilingkey这里的入参必须为静态的，不能动态设置入参
    if (mBaseParams.dType == ge::DT_FLOAT16) {
        if (mBaseParams.isHighPrecison) {
            tilingKey = TILINGKEY(S2, S1, G, FLOAT16_PRECISION, SBND, NONE); // 10000000000000113234
        } else {
            tilingKey = TILINGKEY(S2, S1, G, FLOAT16, SBND, NONE); // 10000000000000110234
        }
    } else if (mBaseParams.dType == ge::DT_BF16) {
        tilingKey = TILINGKEY(S2, S1, G, BFLOAT16, SBND, NONE); // 10000000000000112234
    } else {
        tilingKey = TILINGKEY(S2, S1, G, FLOAT32, SBND, NONE); // 10000000000000111234
    }

    OP_LOGD(context_->GetNodeName(), "TilingKey is %lu.", tilingKey);
    return tilingKey;
}

// 内部私有处理函数
ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bbn::GetShapeInfo()
{
    mBaseParams.n = *context_->GetAttrs()->GetAttrPointer<int>(HEAD_NUM); // 4 is N index
    CHECK_ZERO(mBaseParams.n);
    const char* inputLayout = context_->GetAttrs()->GetAttrPointer<char>(INPUT_LAYOUT); // 5
    const gert::Shape& queryShape = context_->GetInputShape(QUERY)->GetStorageShape();
    const gert::Shape& keyShape = context_->GetInputShape(KEY)->GetStorageShape();

    uint32_t aix;
    if (strcmp(inputLayout, BSND_STR) == 0 || strcmp(inputLayout, BNSD_STR) == 0) {
        if (queryShape.GetDimNum() != DIM_COUNT_NUM_4 || keyShape.GetDimNum() != DIM_COUNT_NUM_4) {
            VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "Shape size is not 4[%zu, %zu].",
                                            queryShape.GetDimNum(), keyShape.GetDimNum());
            return ge::GRAPH_FAILED;
        }
        if (strcmp(inputLayout, BSND_STR) == 0) {
            aix = static_cast<uint32_t>(InputLayout::BSND);
            mBaseParams.layoutType = static_cast<uint32_t>(LayoutType::BSND);
        } else {
            aix = static_cast<uint32_t>(InputLayout::BNSD);
            mBaseParams.layoutType = static_cast<uint32_t>(LayoutType::BNSD);
        }
        mBaseParams.d = queryShape.GetDim(LAYOUT_TO_AXIS[aix][AXIS4_D]);
        mBaseParams.h = queryShape.GetDim(LAYOUT_TO_AXIS[aix][AXIS4_N]) * mBaseParams.d; // h = n * d
        uint32_t hkv = keyShape.GetDim(LAYOUT_TO_AXIS[aix][AXIS4_N]) *
                       keyShape.GetDim(LAYOUT_TO_AXIS[aix][AXIS4_D]);
        CHECK_ZERO(hkv);
        mBaseParams.g = mBaseParams.h / hkv; // g = hq / jkv
    } else {
        if (queryShape.GetDimNum() != DIM_COUNT_NUM_3 || keyShape.GetDimNum() != DIM_COUNT_NUM_3) {
            VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "Shape size is not 3[%zu, %zu].",
                                            queryShape.GetDimNum(), keyShape.GetDimNum());
            return ge::GRAPH_FAILED;
        }
        if (strcmp(inputLayout, BSH_STR) == 0) {
            aix = static_cast<uint32_t>(InputLayout::BSH);
            mBaseParams.layoutType = static_cast<uint32_t>(LayoutType::BSH);
        } else {
            aix = static_cast<uint32_t>(InputLayout::SBH);
            mBaseParams.layoutType = static_cast<uint32_t>(LayoutType::SBH);
        }
        mBaseParams.h = queryShape.GetDim(LAYOUT_TO_AXIS[aix][H]);
        mBaseParams.d = mBaseParams.h / mBaseParams.n;
        uint32_t hkv = keyShape.GetDim(LAYOUT_TO_AXIS[aix][H]);
        CHECK_ZERO(hkv);
        mBaseParams.g = mBaseParams.h / hkv;
    }

    mBaseParams.b = queryShape.GetDim(LAYOUT_TO_AXIS[aix][B]);
    mBaseParams.s = queryShape.GetDim(LAYOUT_TO_AXIS[aix][S]);
    mBaseParams.sq = mBaseParams.s;
    mBaseParams.skv = keyShape.GetDim(LAYOUT_TO_AXIS[aix][S]);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bbn::DoOpTilingPre()
{
    // 根据shape、硬件size限制（参考CheckArgsLegal函数）、切分s1InnerSize、s2InnerSize 并设置；
    uint32_t s1InnerSize = 0;
    uint32_t s2InnerSize = 0;
    BisectionForBestSplit(s1InnerSize, s2InnerSize);
    if (s1InnerSize == 0 || s2InnerSize == 0) {
        OP_LOGI(context_->GetNodeName(), "Split not success, s1InnerSize(%u), s2InnerSize(%u).",
                s1InnerSize, s2InnerSize);
        return ge::GRAPH_FAILED;
    }

    tilingData.splitCoreParams.set_s1InnerSize(s1InnerSize);
    tilingData.splitCoreParams.set_s2InnerSize(s2InnerSize);
    OP_LOGD(context_->GetNodeName(), "Split success, s1InnerSize(%u), s2InnerSize(%u).",
            s1InnerSize, s2InnerSize);
    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreGradTilingUs1s2Bbn::GetBasicSplitBlockSize(std::pair<uint32_t, uint32_t> &split)
{
    SplitDtypeEnum splitDtype;
    if (mBaseParams.dType == ge::DT_FLOAT16) {
        if (mBaseParams.isHighPrecison) {
            splitDtype = SplitDtypeEnum::FLOAT16_PRECISION;
        } else {
            splitDtype = SplitDtypeEnum::FLOAT16;
        }
    } else if (mBaseParams.dType == ge::DT_BF16) {
        splitDtype = SplitDtypeEnum::BFLOAT16;
    } else {
        splitDtype = SplitDtypeEnum::FLOAT32;
    }
    auto iter = BEST_BASIC_SPLIT_BLOCK_SIZE.find(splitDtype);
    if (iter != BEST_BASIC_SPLIT_BLOCK_SIZE.end()) {
        split = iter->second;
    }
}

bool FlashAttentionScoreGradTilingUs1s2Bbn::CheckArgsLegal(uint32_t s1InnerSize, uint32_t s2InnerSize)
{
    int64_t ubBufferSize;
    if (mBaseParams.dataType == BF16 || mBaseParams.isHighPrecison) {
        // 5块float buffer
        ubBufferSize = s1InnerSize * s2InnerSize * mBaseParams.dataTypeSize * NUM2 * INNER_MATRES_NUM +
                       s1InnerSize * BYTE_BLOCK * NUM2; // 2块softmax fp32 buffer
    } else {
        ubBufferSize = s1InnerSize * s2InnerSize * mBaseParams.dataTypeSize * INNER_MATRES_NUM +
                       s1InnerSize * BYTE_BLOCK * NUM2;
    }

    auto softmaxShape = Shape({s1InnerSize, s2InnerSize});
    int64_t softmaxTmpSize = GetSoftMaxMinTmpSize(softmaxShape, sizeof(float), true);
    int64_t softmaxGradTmpSize = GetSoftMaxGradMinTmpSize(softmaxShape, mBaseParams.dataTypeSize, true, true);
    uint64_t usedUbSize = static_cast<uint64_t>(ubBufferSize + std::max(softmaxTmpSize, softmaxGradTmpSize));

    OP_LOGD(context_->GetNodeName(), "UbBufferSize: %ld, softmaxSize: %ld, softmaxGradSize: %ld, usedUbSize: %lu.",
            ubBufferSize, softmaxTmpSize, softmaxGradTmpSize, usedUbSize);

    if (usedUbSize <= aicoreParams_.ubSize) {
        mBaseParams.usedUBSize = static_cast<uint32_t>(ubBufferSize);
        OP_LOGD(context_->GetNodeName(), "Set mBaseParams.usedUBSize %u.", mBaseParams.usedUBSize);
        return true;
    }
    return false;
}

void FlashAttentionScoreGradTilingUs1s2Bbn::BisectionForBestSplit(uint32_t &s1InnerSize, uint32_t &s2InnerSize)
{
    std::pair<uint32_t, uint32_t> defaultSplit; // s1InnerSize, s2InnerSize
    GetBasicSplitBlockSize(defaultSplit);

    s1InnerSize = std::min(mBaseParams.sq, defaultSplit.first);
    s2InnerSize = std::min(mBaseParams.skv, defaultSplit.second);

    // s1InnerSize和s2InnerSize对齐16向上取整
    s1InnerSize = Align(s1InnerSize, ALIGN_SIZE_16);
    s2InnerSize = Align(s2InnerSize, ALIGN_SIZE_16);
    while (true) {
        if (CheckArgsLegal(s1InnerSize, s2InnerSize)) {
            return;
        }
        if (s1InnerSize  < defaultSplit.first) {
            s1InnerSize = s1InnerSize - ALIGN_SIZE_16;
        } else if (s2InnerSize  < defaultSplit.second) {
            s2InnerSize = s2InnerSize - ALIGN_SIZE_16;
        } else {
            // 不应该出现的场景，default的split不能满足ub大小，做个保护处理
            s1InnerSize = s1InnerSize - ALIGN_SIZE_16;
            s2InnerSize = s2InnerSize - ALIGN_SIZE_16;
            OP_LOGD(context_->GetNodeName(), "S1InnerSize and s2InnerSize are both larger than default values.");
        }
    }
    return;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bbn::SplitCoreTilingBN()
{
    // 按照B * N分核，填充参数tailSplitCoreParams和splitCoreParams
    FlashAttentionScoreGradUs1s2BbnBaseParams& baseParams = tilingData.baseParams;
    uint32_t batch = baseParams.get_batch();
    uint32_t seq = baseParams.get_S();
    uint32_t usedCoreNum;
    int64_t remainCoreNum = 1;
    uint32_t batchRange;
    uint32_t remainBatchRange;

    if (batch >= mBaseParams.coreNum) { // use all cores
        batchRange = Us1s2BbnCeilCommon(batch, mBaseParams.coreNum); // ceil
        remainBatchRange = batchRange - 1;
        remainCoreNum = batchRange * mBaseParams.coreNum - batch;
        usedCoreNum = mBaseParams.coreNum;
    } else { // average use the core
        batchRange = Us1s2BbnCeilCommon(batch, mBaseParams.coreNum); // ceil
        usedCoreNum = Us1s2BbnCeilCommon(batch, batchRange);
        remainBatchRange = batch - batchRange * (usedCoreNum - 1);
        mBaseParams.coreNum = usedCoreNum;
    }
    baseParams.set_coreNum(usedCoreNum);
    baseParams.set_formerCoreNum(usedCoreNum - remainCoreNum);
    Us1s2BbnSplitCoreParams& splitCoreParams = tilingData.splitCoreParams;
    splitCoreParams.set_splitedBatchRange(batchRange); // 每个核处理总大小
    mBaseParams.splitedCoreSeqNum = seq;
    splitCoreParams.set_splitedCoreBatchReduceNum(batchRange * seq * BLOCK_BYTES / sizeof(float)); // 32 reserve
    splitCoreParams.set_tailSplitedBatchRange(remainBatchRange); // 每个核处理总大小
    splitCoreParams.set_tailSplitedCoreBatchReduceNum(remainBatchRange * seq * BLOCK_BYTES / sizeof(float));
    OP_LOGD(context_->GetNodeName(), "Set params: usedCoreNum(%u), formerCoreNum(%u), splitedBatchRange(%u),",
            usedCoreNum, baseParams.get_formerCoreNum(), batchRange);
    OP_LOGD(context_->GetNodeName(), "Set params: mBaseParams.splitedCoreSeqNum(%u), splitedCoreBatchReduceNum(%u)," \
            "tailSplitedBatchRange(%u), tailSplitedCoreBatchReduceNum(%u).",
            seq, splitCoreParams.get_splitedCoreBatchReduceNum(), splitCoreParams.get_tailSplitedBatchRange(),
            splitCoreParams.get_tailSplitedCoreBatchReduceNum());
    /** 待考虑有空核的情况 */
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bbn::DropoutTiling() {
    // 120KB FP16Tensor, 60KB U8Tensor, 8KB MaskTensor, 512B HelpTensor which less than UB(192KB).
    // singleUBProcessNum: UB最大处理FP16数据大小，需保证能被128整除.
    uint32_t castBufferLen = SIXTY_KBYTES;
    uint32_t outputBufferLen = THIRTY_KBYTES;
    uint32_t inputBufferLen = 4 * 1024;
    uint32_t singleUBProcessNum = castBufferLen / 2;
    uint32_t dropoutWorkspaceLen =
        tilingData.baseParams.get_B() * tilingData.baseParams.get_N() *
        tilingData.baseParams.get_S() * tilingData.baseParams.get_Skv();
    dropoutWorkspaceLen = (dropoutWorkspaceLen + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES; // 32B align
    if (tilingData.baseParams.get_keepProb() == 1) {
        dropoutWorkspaceLen = 0;
    }
    tilingData.dropoutCastParams.set_dropoutWorkspaceLen(dropoutWorkspaceLen);
    tilingData.dropoutCastParams.set_castBufferLen(castBufferLen);
    tilingData.dropoutCastParams.set_outputBufferLen(outputBufferLen);
    tilingData.dropoutCastParams.set_inputBufferLen(inputBufferLen);
    tilingData.dropoutCastParams.set_singleUBProcessNum(singleUBProcessNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bbn::DoPostScalarOpTiling() {
    const uint32_t BLOCK = BLOCK_BYTES;
    const uint32_t calcTypeByteLen = 4;
    const uint32_t outTypeByteLen = 2;
    uint32_t INPUT_BUFFER_LEN = SIXTY_KBYTES;
    uint32_t OUTPUT_BUFFER_LEN = THIRTY_KBYTES;

    uint32_t baseOutNum = OUTPUT_BUFFER_LEN / outTypeByteLen;
    uint32_t used_core_num = tilingData.baseParams.get_coreNum();

    // query
    uint32_t allNumQuery =
        tilingData.baseParams.get_B() * tilingData.baseParams.get_N() * tilingData.baseParams.get_Sq() * tilingData.baseParams.get_D();
    uint32_t singleCoreNumQuery = CeilCommon(allNumQuery / 16, used_core_num) * 16;
    uint32_t tailCoreNumQuery = allNumQuery - singleCoreNumQuery * (used_core_num - 1);

    uint32_t singleCoreLoopTimesQuery = CeilCommon(singleCoreNumQuery / 16, baseOutNum / 16);
    uint32_t lastLoopNumQuery = singleCoreNumQuery - (singleCoreLoopTimesQuery - 1) * baseOutNum;

    uint32_t tailCoreLoopTimesQuery = CeilCommon(tailCoreNumQuery / 16, baseOutNum / 16);
    uint32_t tailCoreLastLoopNumQuery = tailCoreNumQuery - (tailCoreLoopTimesQuery - 1) * baseOutNum;

    // K V
    uint32_t allNumKv = tilingData.baseParams.get_B() * tilingData.baseParams.get_N() * tilingData.baseParams.get_Skv() * tilingData.baseParams.get_D();
    uint32_t singleCoreNumKv = CeilCommon(allNumKv / 16, used_core_num) * 16;
    uint32_t tailCoreNumKv = allNumKv - singleCoreNumKv * (used_core_num - 1);

    uint32_t singleCoreLoopTimesKv = CeilCommon(singleCoreNumKv / 16, baseOutNum / 16);
    uint32_t lastLoopNumKv = singleCoreNumKv - (singleCoreLoopTimesKv - 1) * baseOutNum;

    uint32_t tailCoreLoopTimesKv = CeilCommon(tailCoreNumKv / 16, baseOutNum / 16);
    uint32_t tailCoreLastLoopNumKv = tailCoreNumKv - (tailCoreLoopTimesKv - 1) * baseOutNum;

    tilingData.postOpTilingParams.set_dqWorkspaceLen(Align(allNumQuery * calcTypeByteLen, BLOCK));
    tilingData.postOpTilingParams.set_dkWorkspaceLen(Align(allNumKv * calcTypeByteLen, BLOCK));
    tilingData.postOpTilingParams.set_dvWorkspaceLen(Align(allNumKv * calcTypeByteLen, BLOCK));
    tilingData.postOpTilingParams.set_inputBufferLen(INPUT_BUFFER_LEN);
    tilingData.postOpTilingParams.set_outputBufferLen(OUTPUT_BUFFER_LEN);
    tilingData.postOpTilingParams.set_singleUBProcessNum(baseOutNum);
    tilingData.postOpTilingParams.set_dqSingleCoreNum(singleCoreNumQuery);
    tilingData.postOpTilingParams.set_dkvSingleCoreNum(singleCoreNumKv);
    tilingData.postOpTilingParams.set_dqSingleCoreLoop(singleCoreLoopTimesQuery);
    tilingData.postOpTilingParams.set_kvSingleCoreLoop(singleCoreLoopTimesKv);
    tilingData.postOpTilingParams.set_dqTailCoreLoop(tailCoreLoopTimesQuery);
    tilingData.postOpTilingParams.set_kvTailCoreLoop(tailCoreLoopTimesKv);

    tilingData.postOpTilingParams.set_dqTailCoreLastLoopNum(tailCoreLastLoopNumQuery);
    tilingData.postOpTilingParams.set_kvTailCoreLastLoopNum(tailCoreLastLoopNumKv);
    tilingData.postOpTilingParams.set_dqLastLoopNum(lastLoopNumQuery);
    tilingData.postOpTilingParams.set_kvLastLoopNum(lastLoopNumKv);

    OP_LOGD(context_->GetNodeName(), "post ScaleAndMuls Op tiling success");

    return ge::GRAPH_SUCCESS;
}

bool FlashAttentionScoreGradTilingUs1s2Bbn::JudgeNeedChangeCVRatio() {
     // SBH 场景下，BH大于65535 不改变配比，等将来matmul优化以后去掉
    if (mBaseParams.layoutType == static_cast<uint32_t>(LayoutType::SBH) &&
        mBaseParams.b * mBaseParams.h > MAX_STRIDE_LIMIT) {
        return false;
    }
    return true;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bbn::CheckAttenMaskShape()
{
    // check atten_mask shape when enable atten_mask_compress
    if (mBaseParams.attenMaskCompressMode != NO_COMPRESS_MODE) {
        if (mBaseParams.attenMaskS1Size != mBaseParams.attenMaskS2Size) {
            OP_LOGE("Atten Mask Compress", "atten mask shape is not square.");
            return ge::GRAPH_FAILED;
        }
        if (mBaseParams.attenMaskS2Size < std::max(tilingData.splitCoreParams.get_s1InnerSize(),
                                                   tilingData.splitCoreParams.get_s2InnerSize()) * MULT_BASE) {
            OP_LOGE("Atten Mask Compress", "atten mask shape is small, try setting it to [2048, 2048].");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreGradTilingUs1s2Bbn::CalcSOffset() {
    uint32_t SOffset = 0;
    if (mBaseParams.layoutType == static_cast<uint32_t>(LayoutType::BSH) ||
        mBaseParams.layoutType == static_cast<uint32_t>(LayoutType::BSND)) {
        SOffset = tilingData.baseParams.get_H();
    } else if (mBaseParams.layoutType == static_cast<uint32_t>(LayoutType::SBH)) {
        SOffset = tilingData.baseParams.get_B() * tilingData.baseParams.get_H();
    } else if (mBaseParams.layoutType == static_cast<uint32_t>(LayoutType::BNSD)) {
        SOffset = tilingData.baseParams.get_D();
    }
    tilingData.splitCoreParams.set_SOffset(SOffset);
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bbn::SingleCoreTiling()
{
    // 填singleCoreParams的参数
    Us1s2BbnSplitCoreParams& splitCoreParams = tilingData.splitCoreParams;
    // 通过每个核的s1InnerSize、s2InnerSize计算其他相关参数
    uint32_t s1InnerSize = splitCoreParams.get_s1InnerSize();
    uint32_t s2InnerSize = splitCoreParams.get_s2InnerSize();

    uint32_t s1InnerSizeDiv = Us1s2BbnCeilCommon(mBaseParams.splitedCoreSeqNum, s1InnerSize);
    uint32_t s1InnerSizeTail = mBaseParams.splitedCoreSeqNum - s1InnerSize * (s1InnerSizeDiv - 1);
    uint32_t s2InnerSizeDiv = Us1s2BbnCeilCommon(mBaseParams.skv, s2InnerSize);
    uint32_t s2InnerSizeTail = mBaseParams.skv - s2InnerSize * (s2InnerSizeDiv - 1);

    splitCoreParams.set_s1Range(s1InnerSizeDiv);
    splitCoreParams.set_s2Range(s2InnerSizeDiv);
    splitCoreParams.set_tailS1InnerSize(s1InnerSizeTail);
    splitCoreParams.set_tailS2InnerSize(s2InnerSizeTail);

    uint32_t s2CVRatio = Us1s2BbnCeilCommon(mBaseParams.skv, s2InnerSize);
    uint32_t s1CVRatio = 1;
    if (JudgeNeedChangeCVRatio()) {
        if (s2CVRatio > S1_S2_RATIO) {
            s2CVRatio = S1_S2_RATIO;
        }
        s1CVRatio = S1_S2_RATIO / s2CVRatio; // s2CVRatio must not equal 0
    }
    s1CVRatio = s1CVRatio * s1InnerSize > mBaseParams.sq ?
                mBaseParams.sq / s1InnerSize : s1CVRatio; // s1InnerSize must not equal 0
    s1CVRatio = s1CVRatio > 0 ? s1CVRatio : 1;
    s2CVRatio = s2CVRatio > 0 ? s2CVRatio : 1;

    uint32_t cubeBaseS1 = s1CVRatio * s1InnerSize;
    uint32_t cubeBaseS2 = std::min(s2CVRatio * s2InnerSize, mBaseParams.skv);
    uint32_t cubeMDiv = Us1s2BbnCeilCommon(mBaseParams.splitedCoreSeqNum, cubeBaseS1);
    uint32_t cubeNDiv = Us1s2BbnCeilCommon(mBaseParams.skv, cubeBaseS2);
    uint32_t cubeMTailSize = mBaseParams.splitedCoreSeqNum - cubeBaseS1 * (cubeMDiv - 1);
    uint32_t cubeNTailSize = mBaseParams.skv - cubeBaseS2 * (cubeNDiv - 1);
    splitCoreParams.set_s1RangeC(cubeMDiv);
    splitCoreParams.set_s2RangeC(cubeNDiv);
    splitCoreParams.set_cubeBaseS1(cubeBaseS1);
    splitCoreParams.set_cubeBaseS2(cubeBaseS2);
    splitCoreParams.set_tailCubeBaseS1(cubeMTailSize);
    splitCoreParams.set_tailCubeBaseS2(cubeNTailSize);
    splitCoreParams.set_s1CVRatio(s1CVRatio);
    splitCoreParams.set_s2CVRatio(s2CVRatio);
    mBaseParams.cubeTilingBaseM = cubeBaseS1;
    mBaseParams.cubeTilingBaseN = cubeBaseS2;

    splitCoreParams.set_innerReduceNum(s1InnerSize * BLOCK_BYTES / sizeof(float));
    splitCoreParams.set_innerMatResSize(s1InnerSize * s2InnerSize * mBaseParams.dataTypeSize);
    // 是否切D
    tilingData.splitCoreParams.set_notSplitGradD(1);
    if (mBaseParams.d > s2InnerSize) {
        tilingData.splitCoreParams.set_notSplitGradD(0);
    }
    // 根据layout计算S轴的offset
    CalcSOffset();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bbn::SetMatmulTilingV2(matmul_tiling::MatmulApiTiling& mm1,
    const uint32_t& s1InnerSize, const uint32_t& s2InnerSize, const uint32_t& l1SizeRemain, TCubeTiling& mm1Tiling)
{
    matmul_tiling::DataType matmul_dtype = matmul_tiling::DataType::DT_FLOAT16;
    matmul_tiling::DataType l0c_matmul_dtype = matmul_tiling::DataType::DT_FLOAT16;
    if (mBaseParams.dType == ge::DT_FLOAT) {
        matmul_dtype = matmul_tiling::DataType::DT_FLOAT;
        l0c_matmul_dtype = matmul_tiling::DataType::DT_FLOAT;
    } else if (mBaseParams.dType == ge::DT_BF16) {
        matmul_dtype = matmul_tiling::DataType::DT_BF16;
        l0c_matmul_dtype = matmul_tiling::DataType::DT_FLOAT;
    }
    if (mBaseParams.isHighPrecison) {
        l0c_matmul_dtype = matmul_tiling::DataType::DT_FLOAT;
    }
    mm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype, false);
    mm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype, true);
    mm1.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND_ALIGN, l0c_matmul_dtype);

    mm1.SetShape(mBaseParams.cubeTilingBaseM, mBaseParams.cubeTilingBaseN, mBaseParams.d);
    if (mBaseParams.layoutType == static_cast<uint32_t>(LayoutType::BSH) ||
        mBaseParams.layoutType == static_cast<uint32_t>(LayoutType::BSND)) {
        mm1.SetOrgShape(mBaseParams.sq, mBaseParams.skv, mBaseParams.d * mBaseParams.n);
    } else if (mBaseParams.layoutType == static_cast<uint32_t>(LayoutType::SBH)) {
        mm1.SetOrgShape(mBaseParams.sq, mBaseParams.skv, mBaseParams.b * mBaseParams.d * mBaseParams.n);
    } else if (mBaseParams.layoutType == static_cast<uint32_t>(LayoutType::BNSD)) {
        mm1.SetOrgShape(mBaseParams.sq, mBaseParams.skv, mBaseParams.d);
    }
    mm1.SetBias(false);
    mm1.SetBufferSpace(l1SizeRemain, aicoreParams_.l0cSize);
    mm1.SetFixSplit(s1InnerSize, s2InnerSize, -1);
    mm1.SetTraverse(matmul_tiling::MatrixTraverse::FIRSTN);
    bool res = mm1.GetTiling(mm1Tiling) == -1;
    if (res) {
        OP_LOGI(context_->GetNodeName(), "Res is -1.");
        return ge::GRAPH_FAILED;
    }
    mm1Tiling.set_stepM(1);
    mm1Tiling.set_stepN(1);
    mm1Tiling.set_shareMode(0);
    mm1Tiling.set_shareL1Size(l1SizeRemain);
    mm1Tiling.set_shareL0CSize(aicoreParams_.l0cSize);
    OP_LOGD(context_->GetNodeName(), "Mm1Tiling params: shareL1Size(%u), shareL0CSize(%lu).",
            l1SizeRemain, aicoreParams_.l0cSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bbn::SetBMM31MatmulTilingV2(matmul_tiling::MatmulApiTiling& mm3,
    const uint32_t& s1InnerSize, const uint32_t& s2InnerSize, const uint32_t& l1SizeRemain, TCubeTiling& mm3Tiling)
{
    matmul_tiling::DataType matmul_dtype = matmul_tiling::DataType::DT_FLOAT16;
    matmul_tiling::DataType matmul_result_dtype = matmul_tiling::DataType::DT_FLOAT16;
    if (mBaseParams.dType == ge::DT_FLOAT) {
        matmul_dtype = matmul_tiling::DataType::DT_FLOAT;
        matmul_result_dtype = matmul_tiling::DataType::DT_FLOAT;
    } else if (mBaseParams.dType == ge::DT_BF16) {
        matmul_dtype = matmul_tiling::DataType::DT_BF16;
        matmul_result_dtype = matmul_tiling::DataType::DT_FLOAT;
    }
    if (mBaseParams.isHighPrecison) {
        matmul_result_dtype = matmul_tiling::DataType::DT_FLOAT;
    }
    mm3.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype, false);
    mm3.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype, false);
    mm3.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_result_dtype);
    uint32_t skvAlignSize = (mBaseParams.skv + mBaseParams.blockNum - 1) / mBaseParams.blockNum * mBaseParams.blockNum;
    mm3.SetShape(mBaseParams.sq, mBaseParams.d, mBaseParams.skv);
    if (mBaseParams.layoutType == static_cast<uint32_t>(LayoutType::BSH) ||
        mBaseParams.layoutType == static_cast<uint32_t>(LayoutType::BSND)) {
        mm3.SetOrgShape(mBaseParams.sq, mBaseParams.d * mBaseParams.n, skvAlignSize, mBaseParams.skv);
    } else if (mBaseParams.layoutType == static_cast<uint32_t>(LayoutType::SBH)) {
        mm3.SetOrgShape(mBaseParams.sq, mBaseParams.b * mBaseParams.d * mBaseParams.n, skvAlignSize, mBaseParams.skv);
    } else if (mBaseParams.layoutType == static_cast<uint32_t>(LayoutType::BNSD)) {
        mm3.SetOrgShape(mBaseParams.sq, mBaseParams.d, skvAlignSize, mBaseParams.skv);
    }
    mm3.SetBias(false);
    mm3.SetBufferSpace(l1SizeRemain, aicoreParams_.l0cSize);
    // api自己做tiling
    mm3.SetFixSplit(-1, -1, -1);
    bool res = mm3.GetTiling(mm3Tiling) == -1;
    if (res) {
        OP_LOGI(context_->GetNodeName(), "Res is -1.");
        return ge::GRAPH_FAILED;
    }

    mm3Tiling.set_shareMode(0);
    mm3Tiling.set_shareL1Size(l1SizeRemain);
    mm3Tiling.set_shareL0CSize(aicoreParams_.l0cSize);
    OP_LOGD(context_->GetNodeName(), "Mm31Tiling params: shareL1Size(%u), shareL0CSize(%lu).",
            l1SizeRemain, aicoreParams_.l0cSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bbn::SetBMM4MatmulTilingV2(matmul_tiling::MatmulApiTiling& mm4,
    const uint32_t& s1InnerSize, const uint32_t& s2InnerSize, const uint32_t& l1SizeRemain, TCubeTiling& mm4Tiling)
{
    matmul_tiling::DataType matmul_dtype = matmul_tiling::DataType::DT_FLOAT16;
    matmul_tiling::DataType matmul_result_dtype = matmul_tiling::DataType::DT_FLOAT16;
    if (mBaseParams.dType == ge::DT_FLOAT) {
        matmul_dtype = matmul_tiling::DataType::DT_FLOAT;
        matmul_result_dtype = matmul_tiling::DataType::DT_FLOAT;
    } else if (mBaseParams.dType == ge::DT_BF16) {
        matmul_dtype = matmul_tiling::DataType::DT_BF16;
        matmul_result_dtype = matmul_tiling::DataType::DT_FLOAT;
    }
    if (mBaseParams.isHighPrecison) {
        matmul_result_dtype = matmul_tiling::DataType::DT_FLOAT;
    }
    mm4.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype, true);
    mm4.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_dtype, false);
    mm4.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_result_dtype);

    uint32_t skvAlignSize = (mBaseParams.skv + mBaseParams.blockNum - 1) / mBaseParams.blockNum * mBaseParams.blockNum;
    mm4.SetShape(mBaseParams.skv, mBaseParams.d, mBaseParams.sq);
    if (mBaseParams.layoutType == static_cast<uint32_t>(LayoutType::BSH) ||
        mBaseParams.layoutType == static_cast<uint32_t>(LayoutType::BSND)) {
        // workspace带pad
        mm4.SetOrgShape(skvAlignSize, mBaseParams.d * mBaseParams.n, mBaseParams.sq);
    } else if (mBaseParams.layoutType == static_cast<uint32_t>(LayoutType::SBH)) {
        mm4.SetOrgShape(skvAlignSize, mBaseParams.b * mBaseParams.d * mBaseParams.n, mBaseParams.sq);
    } else if (mBaseParams.layoutType == static_cast<uint32_t>(LayoutType::BNSD)) {
        mm4.SetOrgShape(skvAlignSize, mBaseParams.d, mBaseParams.sq);
    }
    mm4.SetBias(false);
    mm4.SetBufferSpace(l1SizeRemain, aicoreParams_.l0cSize);
    mm4.SetFixSplit(-1, -1, -1);
    bool res = mm4.GetTiling(mm4Tiling) == -1;
    if (res) {
        OP_LOGI(context_->GetNodeName(), "Res is -1.");
        return ge::GRAPH_FAILED;
    }

    mm4Tiling.set_shareMode(0);
    mm4Tiling.set_shareL1Size(l1SizeRemain);
    mm4Tiling.set_shareL0CSize(aicoreParams_.l0cSize);
    OP_LOGD(context_->GetNodeName(), "Mm4Tiling params: shareL1Size(%u), shareL0CSize(%lu).",
            l1SizeRemain, aicoreParams_.l0cSize);
    return ge::GRAPH_SUCCESS;
}

uint32_t FlashAttentionScoreGradTilingUs1s2Bbn::Us1s2BbnCeilCommon(uint32_t num1, uint32_t num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

void FlashAttentionScoreGradTilingUs1s2Bbn::PrintBaseParams()
{
    // mBaseParams infos
    OP_LOGD(context_->GetNodeName(), "mBaseParams: corenum(%u), b(%u), n(%u), s(%u), sq(%u), skv(%u), d(%u), g(%u)",
            mBaseParams.coreNum, mBaseParams.b, mBaseParams.n, mBaseParams.s, mBaseParams.sq, mBaseParams.skv,
            mBaseParams.d, mBaseParams.g);
    OP_LOGD(context_->GetNodeName(), "mBaseParams: h(%u), dataTypeSize(%u), blockNum(%u), mask(%u)",
            mBaseParams.h, mBaseParams.dataTypeSize, mBaseParams.blockNum, mBaseParams.mask);
    OP_LOGD(context_->GetNodeName(), "mBaseParams: splitedCoreSeqNum(%u), usedUBSize(%u), layoutType(%u), isSparse(%d)",
            mBaseParams.splitedCoreSeqNum, mBaseParams.usedUBSize, mBaseParams.layoutType,
            tilingData.splitCoreParams.get_sparse());
    OP_LOGD(context_->GetNodeName(), "mBaseParams: isHighPrecison(%d), dataType(%u), l2CacheSize(%llu).",
            mBaseParams.isHighPrecison, static_cast<uint32_t>(mBaseParams.dType), mBaseParams.l2CacheSize);

    // aicoreParams_ infos
    OP_LOGD(context_->GetNodeName(), "aicoreParams: blockDim(%llu), ubSize(%llu), l1Size(%llu).",
            aicoreParams_.blockDim, aicoreParams_.ubSize, aicoreParams_.l1Size);
    OP_LOGD(context_->GetNodeName(), "aicoreParams: l0aSize(%llu), l0bSize(%llu), l0cSize(%llu).",
            aicoreParams_.l0aSize, aicoreParams_.l0bSize, aicoreParams_.l0cSize);
}

REGISTER_TILING_TEMPLATE("FlashAttentionScoreGrad", FlashAttentionScoreGradTilingUs1s2Bbn, 14000);
}  // namespace optiling
