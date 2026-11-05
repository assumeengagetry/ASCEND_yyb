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

#include <queue>
#include <math.h>
#include <set>
#include <cfloat>
#include "op_log.h"
#include "error_log.h"
#include "error_util.h"
#include "op_tiling_util.h"
#include "runtime2_util.h"
#include "graph/utils/op_desc_utils.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "data_copy_transpose_tiling.h"
#include "flash_attention_score_tiling.h"
#include "external/exe_graph/runtime/shape.h"
#include "base/tiling_templates_registry.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t FRACTAL_NUM = 16;
constexpr uint32_t MAX_S_INNER = 512;
constexpr uint32_t NUM2 = 2;
constexpr uint32_t SIZE_2 = 2;
constexpr uint32_t SIZE_4 = 4;
constexpr uint32_t BASE_SIZE_128 = 128;
constexpr uint32_t BASE_SIZE_64 = 64;
constexpr uint32_t SOFTMAX_BASE_M = 8;
constexpr uint32_t SOFTMAX_BASE_N = 64;

constexpr int HIGH_PRECISION = 0;
constexpr int HIGH_PERFORMANCE = 1;

constexpr uint32_t KEY_NON_FLASH_F16 = 100;
constexpr uint32_t KEY_NON_FLASH_F32 = 101;
constexpr uint32_t REMAIN_SIZE = 100 * 1024 * 1024;
const int64_t DIMS_3 = 3;
const int64_t DIMS_4 = 4;
constexpr size_t PSE_INPUT_INDEX = 3;
constexpr size_t PSE_DIM_2 = 2;
constexpr uint32_t PSE_SHAPE_TYPE_B_N_S_S = 0;
constexpr uint32_t PSE_SHAPE_TYPE_B_N_1_S = 1;
constexpr uint32_t PSE_SHAPE_TYPE_NONE = 3;
constexpr uint32_t ATTE_SHAPE_TYPE_NONE = 3;

constexpr size_t ATTENTION_MASK_INPUT_INDEX = 6;
constexpr size_t QUERY_INPUT_INDEX = 0;
constexpr size_t KEY_INPUT_INDEX = 1;
constexpr size_t VALUE_INPUT_INDEX = 2;
constexpr size_t SOFTMAXMAX_OUPUT_INDEX = 0;
constexpr size_t SOFTMAXSUM_OUPUT_INDEX = 1;
constexpr size_t ATTENTIONOUT_OUPUT_INDEX = 3;
constexpr size_t INPUTLAYOUT_ATTRS_INDEX = 5;
constexpr size_t MIN_COPY_UINT_SIZE = 32;

class SupportedShape {
public:
    SupportedShape(uint32_t nIn, uint32_t qSIn, uint32_t kSIn, uint32_t hDivNIn, int32_t batchNum = -1)
    {
        n = nIn;
        qS = qSIn;
        kS = kSIn;
        hDivN = hDivNIn;
        batch = batchNum;
    }
    bool operator < (const SupportedShape &right) const
    {
        if (batch == -1 || right.batch == -1) {
            return memcmp(this, &right, sizeof(SupportedShape) - sizeof(batch)) < 0;
        } else {
            return memcmp(this, &right, sizeof(SupportedShape)) < 0;
        }
    }

private:
    uint32_t n;
    uint32_t qS;
    uint32_t kS;
    uint32_t hDivN;
    int32_t batch;
};

const std::set<SupportedShape> SpecialShape13WhiteList = {
  // 13B
};

const std::set<SupportedShape> SpecialShape175or65WhiteList = {
  // 175B 65B
};

const std::set<SupportedShape> SDShapeWhiteList = {
  // SD
};

const std::set<SupportedShape> SupportShapeWhiteList = {
    // n, s1, s2, d, batch
    SupportedShape(4, 8192, 8192, 80, 2), // GPT3 2.6B SBH performance degradation
};

uint32_t Gcd(uint32_t a, uint32_t b)
{
    if (b == 0) {
        return 0;
    }
    if (a % b == 0) {
        return b;
    } else {
        return Gcd(b, a % b);
    }
}

uint32_t Ceil(uint32_t num1, uint32_t num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

uint32_t Align16(uint32_t num)
{
    return (num + FRACTAL_NUM - 1) / FRACTAL_NUM * FRACTAL_NUM;
}

struct FlashAttentionScoreCompileInfo {};

class FlashAttentionScoreEmptyInputTiling {
public:
    FlashAttentionScoreTilingData tilingData;

    void FlashAttentionScoreSetEmptyInputTilingData(gert::TilingContext *context,
        FlashAttentionScoreTilingData &tilingData);
    void GetTilingKeyAttentionScore4EmptyInput(uint32_t &tilingKey, gert::TilingContext *context);
};

class FlashAttentionScoreTiling {
public:
    FlashAttentionScoreTilingData tilingData;
    ge::graphStatus RunBigKernelTiling(gert::TilingContext *context);
    int64_t ubSizeRemain;
    int64_t ubSizeRemainTail;

protected:
    bool isNTail;
    bool isPipeLine;
    bool isRec;
    int innerPrecise = HIGH_PERFORMANCE;
    uint32_t MAT_L0B_FACTOR = 2;
    uint64_t tBufRemain;
    uint64_t tBufRemainTail;
    bool isSparse;
    uint32_t calcTypeSize = 2;
    uint32_t inputTypeSize = 2;
    matmul_tiling::DataType matmulDype;
    matmul_tiling::DataType matmulOutDtype;
    bool IsPipeline();
    bool IsHeadNumTail(FlashAttentionScoreTilingData &tilingData, uint32_t coreNum);
    void GetCoreSeqNumList(uint32_t splitSeqFactor, uint32_t sBaseNum);
    void AllocCoreSeqNumList();
    ge::graphStatus RunFlashAttentionScoreTiling(gert::TilingContext *context,
        FlashAttentionScoreTilingData &tilingData, uint32_t coreNum, int64_t ubSize, int64_t l1Size, int64_t l0CSize);
    ge::graphStatus RunSparseFlashAttentionScoreTiling(gert::TilingContext *context,
        FlashAttentionScoreTilingData &tilingData, uint32_t coreNum, int64_t ubSize, int64_t l1Size, int64_t l0CSize);
    ge::graphStatus GetSeqBaseAndNewCoreNum(uint32_t coreNum, uint32_t b, uint32_t n, uint32_t s, uint32_t &seqBaseSize,
        uint32_t &NewCoreNum);
    bool ifUseAllCoreNum(uint32_t coreNum, uint32_t B, uint32_t N, uint32_t S);
    bool IsSparseAttentionScoreOri();

    ge::graphStatus TilingGetTilingKeyAttentionScoreAscendC(uint32_t &tilingKey, gert::TilingContext *context,
        uint32_t coreNum, bool isTransposeOut);
    ge::graphStatus GetHeadNumSplitParam(FlashAttentionScoreTilingData &tilingData, uint32_t coreNumRemain);
    ge::graphStatus FlashAttentionScoreSplitBN(gert::TilingContext *context, FlashAttentionScoreTilingData &tilingData,
        uint32_t coreNum);
    ge::graphStatus FlashAttentionScoreSplitBNS(gert::TilingContext *context, FlashAttentionScoreTilingData &tilingData,
        uint32_t coreNum);
    bool isRecTiling(bool isBNSplit, FlashAttentionScoreTilingData &tilingData);
    ge::graphStatus FlashAttentionScoreSingleCoreTiling(gert::TilingContext *context,
        FlashAttentionScoreTilingData &tilingData);
    ge::graphStatus SparseFlashAttentionScoreSingleCoreTiling(gert::TilingContext *context,
        FlashAttentionScoreTilingData &tilingData);
    ge::graphStatus FlashAttentionScoreApiGetTmpBufferSize(FlashAttentionScoreTilingData &tilingData,
        uint32_t &softmaxTmpSize, uint32_t &softmaxFlashTmpSize, uint32_t &dropoutTmpSize);
    ge::graphStatus GetSeqSplitList(FlashAttentionScoreTilingData &tilingData, uint32_t seqSplit,
        std::queue<uint32_t> &sQueue);
    bool FlashAttentionScoreCheckArgsLegal(FlashAttentionScoreTilingData &tilingData, int64_t ubSize, int64_t l1Size,
        int64_t l0CSize, bool &updateDiv, bool &updateDivTail);
    ge::graphStatus FlashAttentionScoreApiTiling(FlashAttentionScoreTilingData &tilingData);
    ge::graphStatus FlashAttentionScoreGetTensorSize(FlashAttentionScoreTilingData &tilingData);
    ge::graphStatus FlashAttentionScoreSetTilingData(gert::TilingContext *context,
        FlashAttentionScoreTilingData &tilingData);
};

bool FlashAttentionScoreTiling::IsSparseAttentionScoreOri()
{
    if (tilingData.baseParams.get_preTockens() >= tilingData.baseParams.get_seqSizeK() &&
        tilingData.baseParams.get_nextTockens() >= tilingData.baseParams.get_seqSizeK()) {
        return false;
    }
    return true;
}

bool FlashAttentionScoreTiling::IsPipeline()
{
    return true;
}

ge::graphStatus FlashAttentionScoreTiling::GetSeqBaseAndNewCoreNum(uint32_t coreNum, uint32_t b, uint32_t n, uint32_t s,
    uint32_t &seqBaseSize, uint32_t &NewCoreNum)
{
    constexpr uint32_t MinSeqBaseSize = 16;
    constexpr uint32_t seqBaseSizeDefault = 128;

    seqBaseSize = seqBaseSizeDefault;
    NewCoreNum = Ceil(b * n * s, seqBaseSize);
    while (seqBaseSize > MinSeqBaseSize) {
        if (NewCoreNum >= coreNum) {
            break;
        }
        seqBaseSize = seqBaseSize >> 1;
        NewCoreNum = Ceil(b * n * s, seqBaseSize);
    }
    return ge::GRAPH_SUCCESS;
}

bool FlashAttentionScoreTiling::ifUseAllCoreNum(uint32_t coreNum, uint32_t B, uint32_t N, uint32_t S)
{
    if (B * N >= coreNum && Gcd(B, coreNum) != 0 && coreNum / Gcd(B, coreNum) <= N) {
        return false;
    } else {
        if (B < coreNum && coreNum % B == 0) {
            coreNum = coreNum / B;
        }
        if (N < coreNum && coreNum % N == 0) {
            coreNum = coreNum / N;
        }
        // 32 : threshold
        if (coreNum != 0 && S / coreNum < 32) {
            return true;
        }
        return false;
    }
}

ge::graphStatus FlashAttentionScoreTiling::TilingGetTilingKeyAttentionScoreAscendC(uint32_t &tilingKey,
    gert::TilingContext *context, uint32_t coreNum, bool isTransposeOut)
{
    const gert::StorageShape *queryShape = context->GetInputShape(0);
    auto kernelType = context->GetInputDesc(0)->GetDataType();
    bool trans = isTransposeOut;
    uint32_t B = 0;
    if (tilingData.baseParams.get_transType() == 0) {
        B = queryShape->GetStorageShape().GetDim(0);
    } else if (tilingData.baseParams.get_transType() == 1) {
        B = queryShape->GetStorageShape().GetDim(1);
    }
    const uint32_t *tmp = context->GetAttrs()->GetAttrPointer<uint32_t>(4);
    uint32_t N = *tmp;
    if (B * N >= coreNum && Gcd(B, coreNum) != 0 && coreNum / Gcd(B, coreNum) <= N) {
        if (kernelType == ge::DT_FLOAT16) {
            tilingKey = 0; // 0: DT_FLOAT16
        } else {
            tilingKey = 4; // 4: others
        }
        if (trans && kernelType == ge::DT_FLOAT16) {
            tilingKey = 2; // 2: trans, DT_FLOAT16, BN spilt
        } else if (trans && kernelType == ge::DT_FLOAT) {
            tilingKey = 6; // 6:trans, DT_FLOAT, BN spilt
        } else if (trans && kernelType == ge::DT_BF16) {
            tilingKey = 8; // 8:trans, DT_BF16, BN spilt
        }
    } else {
        if (kernelType == ge::DT_FLOAT16) {
            tilingKey = 1; // 1: DT_FLOAT16
        } else {
            tilingKey = 5; // 5: others
        }
        if (trans && kernelType == ge::DT_FLOAT16) {
            tilingKey = 3; // 3: trans, DT_FLOAT16, BNS spilt
        } else if (trans && kernelType == ge::DT_FLOAT) {
            tilingKey = 7; // 7:trans, DT_FLOAT, BNS spilt
        } else if (trans && kernelType == ge::DT_BF16) {
            tilingKey = 9; // 9:trans, DT_BF16, BNS spilt
        }
    }

    if (isSparse) {
        // 16 : sparse offset
        tilingKey = tilingKey + 16;
    }
    if (innerPrecise == HIGH_PRECISION) {
        uint32_t highPrecisionTilingKey = 1000;
        tilingKey += highPrecisionTilingKey;
    }

    auto pseShape = context->GetOptionalInputShape(PSE_INPUT_INDEX);
    if (pseShape != nullptr) {
        auto dim2 = pseShape->GetStorageShape().GetDim(PSE_DIM_2);
        if (dim2 != gert::Shape::kInvalidDimValue && dim2 == 1) {
            tilingData.baseParams.set_pseShapeType(PSE_SHAPE_TYPE_B_N_1_S);
        } else {
            tilingData.baseParams.set_pseShapeType(PSE_SHAPE_TYPE_B_N_S_S);
        }
    } else {
        tilingData.baseParams.set_pseShapeType(PSE_SHAPE_TYPE_NONE);
    }
    return ge::GRAPH_SUCCESS;
};

ge::graphStatus FlashAttentionScoreTiling::GetHeadNumSplitParam(FlashAttentionScoreTilingData &tilingData,
    uint32_t coreNumRemain)
{
    uint32_t n = coreNumRemain;
    for (int i = n; i >= 1; i--) {
        if (coreNumRemain % static_cast<uint32_t>(i) == 0 &&
            tilingData.baseParams.get_headNumSize() >= static_cast<uint32_t>(i)) {
            tilingData.attentionScoreCoreParams.set_n1(i);
            return ge::GRAPH_SUCCESS;
        }
    }
    return ge::GRAPH_FAILED;
}

bool FlashAttentionScoreTiling::isRecTiling(bool isBNSplit, FlashAttentionScoreTilingData &tilingData)
{
    if (isBNSplit && tilingData.baseParams.get_seqSizeQ() != tilingData.baseParams.get_seqSizeK()) {
        return true;
    }
    if (calcTypeSize == SIZE_4) {
        if (tilingData.baseParams.get_attenMaskTypeIdx() == ATTE_SHAPE_TYPE_NONE && fabs(tilingData.baseParams.get_keepProb() - 1) < FLT_EPSILON) {
            tilingData.baseParams.set_seqBaseSize(BASE_SIZE_128);
        } else {
            tilingData.baseParams.set_seqBaseSize(BASE_SIZE_64);
        }
        return false;
    }
    // 128 : basic1 seq length
    if (tilingData.baseParams.get_seqSizeQ() / 128 > tilingData.attentionScoreCoreParams.get_s1()) {
        tilingData.baseParams.set_seqBaseSize(128);
        // 64 : basic2 seq length
    } else if (tilingData.baseParams.get_seqSizeQ() / 64 > tilingData.attentionScoreCoreParams.get_s1()) {
        tilingData.baseParams.set_seqBaseSize(64);
    } else {
        tilingData.baseParams.set_seqBaseSize(16); // 16 : basic3 seq length
        return true;
    }
    return tilingData.baseParams.get_seqSizeQ() != tilingData.baseParams.get_seqSizeK();
}

ge::graphStatus FlashAttentionScoreTiling::FlashAttentionScoreSplitBN(gert::TilingContext *context,
    FlashAttentionScoreTilingData &tilingData, uint32_t coreNum)
{
    AttentionScoreCoreParams *params = &tilingData.attentionScoreCoreParams;
    params->set_coreNum(coreNum);
    params->set_b1(std::min(tilingData.baseParams.get_batchSize(), coreNum));
    if (tilingData.baseParams.get_batchSize() < coreNum && coreNum % tilingData.baseParams.get_batchSize() != 0) {
        params->set_b1(Gcd(coreNum, tilingData.baseParams.get_batchSize()));
    }
    if (tilingData.baseParams.get_batchSize() > coreNum && tilingData.baseParams.get_batchSize() % coreNum != 0) {
        params->set_b1(Gcd(tilingData.baseParams.get_batchSize(), coreNum));
    }
    params->set_singleCoreBatchSize(tilingData.baseParams.get_batchSize() / params->get_b1());

    params->set_loopBatchNum(params->get_singleCoreBatchSize());
    params->set_n1(coreNum / params->get_b1());
    params->set_singleCoreHeadNumSizeTail(tilingData.baseParams.get_headNumSize() / params->get_n1());

    uint32_t remain = tilingData.baseParams.get_headNumSize() % params->get_n1();
    if (remain > 0) {
        params->set_formerNum(remain);
        params->set_tailNum(params->get_n1() - params->get_formerNum());
        params->set_singleCoreHeadNumSize(params->get_singleCoreHeadNumSizeTail() + 1);
    } else {
        params->set_formerNum(params->get_n1());
        params->set_tailNum(0);
        params->set_singleCoreHeadNumSize(params->get_singleCoreHeadNumSizeTail());
    }
    params->set_loopHeadNum(params->get_singleCoreHeadNumSize());
    params->set_loopHeadNumTail(params->get_singleCoreHeadNumSizeTail());
    params->set_singleCoreSeqSize(tilingData.baseParams.get_seqSizeQ());
    params->set_singleCoreSeqSizeTail(tilingData.baseParams.get_seqSizeQ());
    params->set_singleCoreHeadSize(tilingData.baseParams.get_headSize());
    params->set_s1(1);
    isRec = isRecTiling(true, tilingData);
    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreTiling::AllocCoreSeqNumList()
{
    AttentionScoreCoreParams *params = &tilingData.attentionScoreCoreParams;
    uint32_t preTockens = tilingData.baseParams.get_preTockens();
    if (preTockens > tilingData.baseParams.get_seqSizeK()) {
        preTockens = tilingData.baseParams.get_seqSizeK();
    }
    uint32_t seqBaseRange = Ceil(tilingData.baseParams.get_seqSizeK(), tilingData.baseParams.get_seqBaseSize());
    // Base length of the right-angled edge of the triangle
    if (tilingData.baseParams.get_seqBaseSize() == 0) {
        return;
    }
    uint32_t triangleSeqBaseNum = preTockens / tilingData.baseParams.get_seqBaseSize();
    // Base lengh of the upper side of the parallelogram
    uint32_t parallelogramBaseWise = triangleSeqBaseNum;
    // Total number of base blocks for triangles
    uint32_t triangleTotleBaseNum = (1 + triangleSeqBaseNum) * (triangleSeqBaseNum / 2);
    // Total number of base rows in the calculation area
    uint32_t TotleSeqBaseNum = (seqBaseRange - triangleSeqBaseNum) + Ceil(triangleTotleBaseNum, parallelogramBaseWise);
    // Number of larger base blocks after core division
    uint32_t bigBaseNum = TotleSeqBaseNum % params->get_s1();
    // Number of smaller base blocks after core division
    uint32_t littleBaseNum = params->get_s1() - bigBaseNum;
    // Number of rows of smaller base blocks after core division
    uint32_t littleSeqBaseSize = TotleSeqBaseNum / params->get_s1();
    // Number of rows in a larger base block after core division
    uint32_t bigSeqBaseSize = littleSeqBaseSize + 1;
    // Total number of larger base blocks after core division
    uint32_t bigSeqBaseTotleNum = bigSeqBaseSize * parallelogramBaseWise;
    // Total number of smaller base blocks after core division
    uint32_t littleSeqBaseTotleNum = littleSeqBaseSize * parallelogramBaseWise;

    if (params->get_s1() >= seqBaseRange) {
        for (uint32_t i = 0; i < seqBaseRange; i++) {
            params->get_seqList()[i] = 1;
        }
        for (uint32_t i = 0; i < params->get_b1(); i++) {
            for (uint32_t j = 0; j < params->get_n1(); j++) {
                for (uint32_t k = 0; k < params->get_s1(); k++) {
                    params->get_seqList()[k + j * params->get_s1() + i * params->get_n1() * params->get_s1()] = params->get_seqList()[k];
                }
            }
        }
        return;
    }

    // Accumulate data in each row from top to botton to approximate the number of base rows in a large block.
    uint32_t SeqIdx = seqBaseRange - 1;
    uint32_t listIdx = params->get_s1() - 1;
    for (; listIdx >= 0; listIdx--) {
        if ((littleBaseNum > 0) && (triangleSeqBaseNum >= 1) &&
            ((SeqIdx - littleSeqBaseSize) >= (triangleSeqBaseNum - 1))) {
            // Preferentially allocate smaller blocks.
            params->get_seqList()[listIdx] = littleSeqBaseTotleNum / parallelogramBaseWise;
            littleBaseNum--;
            SeqIdx -= littleSeqBaseSize;
        } else if ((bigBaseNum > 0) && (triangleSeqBaseNum >= 1) &&
            ((SeqIdx - bigSeqBaseSize) >= (triangleSeqBaseNum - 1))) {
            // Allocate larger blocks after smaller blocks are allocated.
            params->get_seqList()[listIdx] = bigSeqBaseTotleNum / parallelogramBaseWise;
            bigBaseNum--;
            SeqIdx -= bigSeqBaseSize;
        } else {
            break;
        }
    }
    // Finally, allocate the protion with triangles.
    GetCoreSeqNumList(listIdx + 1, SeqIdx + 1);
    for (uint32_t i = 0; i < params->get_b1(); i++) {
        for (uint32_t j = 0; j < params->get_n1(); j++) {
            for (uint32_t k = 0; k < params->get_s1(); k++) {
                params->get_seqList()[k + j * params->get_s1() + i * params->get_n1() * params->get_s1()] = params->get_seqList()[k];
            }
        }
    }
}

void FlashAttentionScoreTiling::GetCoreSeqNumList(uint32_t splitSeqFactor, uint32_t sBaseNum)
{
    if (splitSeqFactor == 0) {
        return;
    }
    AttentionScoreCoreParams *params = &tilingData.attentionScoreCoreParams;
    const uint32_t maxCoreNum = 48;
    uint32_t seqIdxList[maxCoreNum] = {0};
    uint32_t sum = sBaseNum * (sBaseNum + 1) / 2;
    float perNum = float(sum) / splitSeqFactor;
    uint32_t index = 1;
    uint32_t fixFlag = 0;
    for (uint32_t i = 0; i < splitSeqFactor; i++) {
        // Sum of [index - x] = perNum, find x
        // x(x+1)/2 = index(index - 1)/2 = perNum, find x
        float p = perNum + static_cast<float>(index) * (static_cast<float>(index) + 1) / 2;
        float x = (-1 + sqrt(1 + 4 * 2 * p)) / 2;

        // Rounding
        seqIdxList[i] = (uint32_t)round(x);
        index = seqIdxList[i];

        if (seqIdxList[i] > sBaseNum) {
            seqIdxList[i] = sBaseNum;
        }
    }
    while (seqIdxList[splitSeqFactor - 1] < sBaseNum) {
        // The allocation is incorrect. Adjust the previous core calculation amount.
        index = 1;
        fixFlag++;
        for (uint32_t i = 0; i < splitSeqFactor; i++) {
            // Sum of [index - x] = perNum, find x
            // x(x+1)/2 = index(index - 1)/2 = perNum, find x
            float p = perNum + static_cast<float>(index) * (static_cast<float>(index) + 1) / 2;
            float x = (-1 + sqrt(1 + 4 * 2 * p)) / 2;

            if (i < fixFlag) {
                seqIdxList[i] = (uint32_t)ceil(x);
            } else {
                seqIdxList[i] = (uint32_t)round(x);
            }
            index = seqIdxList[i];

            if (seqIdxList[i] > sBaseNum) {
                seqIdxList[i] = sBaseNum;
            }
        }
    }
    params->get_seqList()[0] = seqIdxList[0];
    int i = splitSeqFactor - 1; // Array has 48 elements.
    while (i > 0) {
        if (seqIdxList[i] == 0) {
            --i;
            continue;
        }
        params->get_seqList()[i] = seqIdxList[i] - seqIdxList[i - 1];
        --i;
    }
}

ge::graphStatus FlashAttentionScoreTiling::FlashAttentionScoreSplitBNS(gert::TilingContext *context,
    FlashAttentionScoreTilingData &tilingData, uint32_t coreNum)
{
    AttentionScoreCoreParams *params = &tilingData.attentionScoreCoreParams;
    params->set_coreNum(coreNum);
    params->set_b1(tilingData.baseParams.get_batchSize());
    if (tilingData.baseParams.get_batchSize() < coreNum && coreNum % tilingData.baseParams.get_batchSize() != 0) {
        params->set_b1(Gcd(coreNum, tilingData.baseParams.get_batchSize()));
    }
    if (tilingData.baseParams.get_batchSize() > coreNum && tilingData.baseParams.get_batchSize() % coreNum != 0) {
        params->set_b1(Gcd(tilingData.baseParams.get_batchSize(), coreNum));
    }
    if (params->get_b1() == 0) {
        return ge::GRAPH_PARAM_INVALID;
    }
    params->set_singleCoreBatchSize(tilingData.baseParams.get_batchSize() / params->get_b1());
    params->set_loopBatchNum(params->get_singleCoreBatchSize());

    if (isSparse) {
        params->set_n1(Gcd(coreNum / params->get_b1(), tilingData.baseParams.get_headNumSize()));
        if (params->get_n1() == 0) {
            return ge::GRAPH_PARAM_INVALID;
        }
        params->set_singleCoreHeadNumSize(tilingData.baseParams.get_headNumSize() / params->get_n1());
        params->set_singleCoreHeadNumSizeTail(tilingData.baseParams.get_headNumSize() % params->get_n1());
        params->set_s1(coreNum / params->get_b1() / params->get_n1());
        if (params->get_s1() == 0) {
            return ge::GRAPH_PARAM_INVALID;
        }
        params->set_formerNum(tilingData.baseParams.get_seqBaseRange() % params->get_s1());
        params->set_tailNum(params->get_s1() - params->get_formerNum());

        params->set_singleCoreSeqSizeTail(tilingData.baseParams.get_seqBaseRange() / params->get_s1() *
            tilingData.baseParams.get_seqBaseSize());
        if (params->get_formerNum() == 0) {
            params->set_singleCoreSeqSize(params->get_singleCoreSeqSizeTail());
        } else {
            params->set_singleCoreSeqSize((tilingData.baseParams.get_seqSizeAlign() -
                params->get_tailNum() * params->get_singleCoreSeqSizeTail()) /
                params->get_formerNum());
        }
        params->set_loopHeadNum(params->get_singleCoreHeadNumSize());
    } else {
        params->set_n1(Gcd(coreNum / params->get_b1(), tilingData.baseParams.get_headNumSize()));
        if (params->get_n1() == 0) {
            return ge::GRAPH_PARAM_INVALID;
        }
        params->set_singleCoreHeadNumSizeTail(tilingData.baseParams.get_headNumSize() / params->get_n1());
        if (isNTail) {
            GetHeadNumSplitParam(tilingData, coreNum / params->get_b1());
            params->set_singleCoreHeadNumSizeTail(tilingData.baseParams.get_headNumSize() / params->get_n1());
            params->set_singleCoreHeadNumSize(tilingData.baseParams.get_headNumSize() / params->get_n1());
            uint32_t headNumRemain = tilingData.baseParams.get_headNumSize() % params->get_n1();
            params->set_formerHeadNum(headNumRemain);
            params->set_tailHeadNum(params->get_n1() - params->get_formerHeadNum());
            params->set_singleCoreHeadNumSize(params->get_singleCoreHeadNumSizeTail() + 1);
        } else {
            params->set_formerHeadNum(params->get_n1());
            params->set_tailHeadNum(0);
            params->set_singleCoreHeadNumSize(params->get_singleCoreHeadNumSizeTail());
        }
        params->set_loopHeadNum(params->get_singleCoreHeadNumSize());
        params->set_loopHeadNumTail(params->get_singleCoreHeadNumSizeTail());
        params->set_s1(coreNum / params->get_b1() / params->get_n1());
        if (params->get_s1() == 0) {
            return ge::GRAPH_PARAM_INVALID;
        }
        isRec = isRecTiling(false, tilingData);
        params->set_singleCoreSeqSizeTail(tilingData.baseParams.get_seqSizeQ() /
            tilingData.baseParams.get_seqBaseSize() / params->get_s1() * tilingData.baseParams.get_seqBaseSize());

        uint32_t remain =
            tilingData.baseParams.get_seqSizeQ() / tilingData.baseParams.get_seqBaseSize() % params->get_s1();
        if (remain > 0) {
            params->set_formerNum(remain);
            params->set_tailNum(params->get_s1() - params->get_formerNum());
            params->set_singleCoreSeqSize(params->get_singleCoreSeqSizeTail() +
                tilingData.baseParams.get_seqBaseSize());
        } else {
            params->set_formerNum(params->get_s1());
            params->set_tailNum(0);
            params->set_singleCoreSeqSize(params->get_singleCoreSeqSizeTail());
        }
    }
    auto attentionMask = context->GetOptionalInputShape(ATTENTION_MASK_INPUT_INDEX);
    if (attentionMask != nullptr) {
        AllocCoreSeqNumList();
        // 48 : max support core num
        for (int i = 0; i < 48; i++) {
            params->get_eachCoreSeqSize()[i] = params->get_seqList()[i] * tilingData.baseParams.get_seqBaseSize();
        }
    }
    params->set_singleCoreHeadSize(tilingData.baseParams.get_headSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreTiling::FlashAttentionScoreSingleCoreTiling(gert::TilingContext *context,
    FlashAttentionScoreTilingData &tilingData)
{
    AttentionScoreSingleCoreParams *params = &tilingData.attentionScoreSingleCoreParams;
    params->set_sOuterLoopTimes(
        Ceil(tilingData.attentionScoreCoreParams.get_singleCoreSeqSize(), params->get_singleProcessSOuterSize()));
    auto attentionMask = context->GetOptionalInputShape(ATTENTION_MASK_INPUT_INDEX);
    if (attentionMask != nullptr) {
        // 48 : max support core num
        for (int i = 0; i < 48; i++) {
            params->get_sOuterLoopTimesList()[i] = Ceil(tilingData.attentionScoreCoreParams.get_eachCoreSeqSize()[i],
                params->get_singleProcessSOuterSize());
        }
    }
    params->set_sInnerLoopTimes(Ceil(tilingData.baseParams.get_seqSizeK(), params->get_singleProcessSInnerSize()));
    if (params->get_sInnerLoopTimes() >= 1) {
        params->set_singleProcessSInnerSizeTail(tilingData.baseParams.get_seqSizeK() -
            (params->get_sInnerLoopTimes() - 1) * params->get_singleProcessSInnerSize());
    }
    params->set_tailCoreSOuterLoopTimes(Ceil(tilingData.attentionScoreCoreParams.get_singleCoreSeqSizeTail(),
        params->get_tailCoreSingleProcessSOuterSize()));
    params->set_tailCoreSInnerLoopTimes(
        Ceil(tilingData.baseParams.get_seqSizeK(), params->get_tailCoreSingleProcessSInnerSize()));
    if (params->get_tailCoreSInnerLoopTimes() >= 1) {
        params->set_tailCoreSingleProcessSInnerSizeTail(tilingData.baseParams.get_seqSizeK() -
            (params->get_tailCoreSInnerLoopTimes() - 1) * params->get_tailCoreSingleProcessSInnerSize());
    }
    params->set_singleProcessBatchSize(1);
    params->set_singleProcessHeadNumSize(1);
    params->set_singleProcessHeadSize(tilingData.baseParams.get_headSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreTiling::SparseFlashAttentionScoreSingleCoreTiling(gert::TilingContext *context,
    FlashAttentionScoreTilingData &tilingData)
{
    AttentionScoreSingleCoreParams *params = &tilingData.attentionScoreSingleCoreParams;
    params->set_sOuterLoopTimes(
        Ceil(tilingData.attentionScoreCoreParams.get_singleCoreSeqSize(), params->get_singleProcessSOuterSize()));
    auto attentionMask = context->GetOptionalInputShape(ATTENTION_MASK_INPUT_INDEX);
    if (attentionMask != nullptr) {
        // 48 : max support core num
        for (int i = 0; i < 48; i++) {
            params->get_sOuterLoopTimesList()[i] = Ceil(tilingData.attentionScoreCoreParams.get_eachCoreSeqSize()[i],
                params->get_singleProcessSOuterSize());
        }
    }
    if (params->get_sOuterLoopTimes() >= 1) {
        params->set_singleProcessSOuterSizeTail(tilingData.attentionScoreCoreParams.get_singleCoreSeqSize() -
            (params->get_sOuterLoopTimes() - 1) * params->get_singleProcessSOuterSize());
    }
    params->set_sInnerLoopTimes(Ceil(tilingData.baseParams.get_seqSizeK(), params->get_singleProcessSInnerSize()));
    if (params->get_sInnerLoopTimes() >= 1) {
        params->set_singleProcessSInnerSizeTail(tilingData.baseParams.get_seqSizeK() -
            (params->get_sInnerLoopTimes() - 1) * params->get_singleProcessSInnerSize());
    }
    params->set_tailCoreSOuterLoopTimes(Ceil(tilingData.attentionScoreCoreParams.get_singleCoreSeqSizeTail(),
        params->get_tailCoreSingleProcessSOuterSize()));
    if (params->get_tailCoreSOuterLoopTimes() >= 1) {
        params->set_tailCoreSingleProcessSOuterSizeTail(
            tilingData.attentionScoreCoreParams.get_singleCoreSeqSizeTail() -
            (params->get_tailCoreSOuterLoopTimes() - 1) * params->get_tailCoreSingleProcessSOuterSize());
    }
    params->set_lastCoreSOuterLoopTimes(Ceil(tilingData.attentionScoreCoreParams.get_singleCoreSeqSizeTail(),
        params->get_tailCoreSingleProcessSOuterSize()));
    params->set_lastCoreSingleProcessSOuterSizeTail(tilingData.baseParams.get_seqBaseSize() -
        tilingData.baseParams.get_seqBasePadValue());

    params->set_tailCoreSInnerLoopTimes(
        Ceil(tilingData.baseParams.get_seqSizeK(), params->get_tailCoreSingleProcessSInnerSize()));
    if (params->get_tailCoreSInnerLoopTimes() >= 1) {
        params->set_tailCoreSingleProcessSInnerSizeTail(tilingData.baseParams.get_seqSizeK() -
            (params->get_tailCoreSInnerLoopTimes() - 1) * params->get_tailCoreSingleProcessSInnerSize());
    }

    params->set_singleProcessBatchSize(1);
    params->set_singleProcessHeadNumSize(1);
    params->set_singleProcessHeadSize(tilingData.baseParams.get_headSize());

    return ge::GRAPH_SUCCESS;
}

bool FlashAttentionScoreTiling::FlashAttentionScoreCheckArgsLegal(FlashAttentionScoreTilingData &tilingData,
    int64_t ubSize, int64_t l1Size, int64_t l0CSize, bool &updateDiv, bool &updateDivTail)
{
    bool res = true;
    MAT_L0B_FACTOR = Ceil(tilingData.baseParams.get_seqSizeK(),
        tilingData.attentionScoreSingleCoreParams.get_singleProcessSInnerSize());

    auto tmpShape = Shape({ tilingData.attentionScoreSingleCoreParams.get_singleProcessSOuterSize(),
        tilingData.attentionScoreSingleCoreParams.get_singleProcessSInnerSize() }); // [S,s]
    auto tmpShapeTail = Shape({ tilingData.attentionScoreSingleCoreParams.get_tailCoreSingleProcessSOuterSize(),
        tilingData.attentionScoreSingleCoreParams.get_tailCoreSingleProcessSInnerSize() });

    int64_t softmaxFlashTmpSize = GetSoftMaxFlashV2MaxTmpSize(tmpShape, calcTypeSize, sizeof(float), true, true);
    int64_t dropoutTmpSize = GetDropOutMinTmpSize(tmpShape, calcTypeSize, true);
    int64_t softmaxFlashTmpSizeTail =
        GetSoftMaxFlashV2MaxTmpSize(tmpShapeTail, calcTypeSize, sizeof(float), true, true);
    int64_t dropoutTmpSizeTail = GetDropOutMinTmpSize(tmpShapeTail, calcTypeSize, true);

    if (tilingData.baseParams.get_attenMaskTypeIdx() == ATTE_SHAPE_TYPE_NONE && fabs(tilingData.baseParams.get_keepProb() - 1) < FLT_EPSILON) {
        tilingData.attentionScoreSingleCoreTensorSize.set_attenMaskUbSize(0);
        tilingData.attentionScoreSingleCoreTensorSize.set_tailCoreAttenMaskUbSize(0);
    } else {
        tilingData.attentionScoreSingleCoreTensorSize.set_attenMaskUbSize(
            tilingData.attentionScoreSingleCoreParams.get_singleProcessSOuterSize() *
            tilingData.attentionScoreSingleCoreParams.get_singleProcessSInnerSize());
        tilingData.attentionScoreSingleCoreTensorSize.set_tailCoreAttenMaskUbSize(
            tilingData.attentionScoreSingleCoreParams.get_tailCoreSingleProcessSOuterSize() *
            tilingData.attentionScoreSingleCoreParams.get_tailCoreSingleProcessSInnerSize());
    }

    if (tilingData.baseParams.get_pseShapeType() == PSE_SHAPE_TYPE_B_N_1_S) {
        tilingData.attentionScoreSingleCoreTensorSize.set_pseUbSize(
            tilingData.attentionScoreOffestStrideParams.get_typeByteNum() *
            tilingData.attentionScoreSingleCoreParams.get_singleProcessSInnerSize());
    } else {
        tilingData.attentionScoreSingleCoreTensorSize.set_pseUbSize(
            tilingData.attentionScoreSingleCoreParams.get_singleProcessSOuterSize() *
            tilingData.attentionScoreSingleCoreParams.get_singleProcessSInnerSize());
    }
    tilingData.attentionScoreSingleCoreTensorSize.set_mmResUbSize(
        tilingData.attentionScoreSingleCoreParams.get_singleProcessSOuterSize() *
        tilingData.attentionScoreSingleCoreParams.get_singleProcessSInnerSize());
    tilingData.attentionScoreSingleCoreTensorSize.set_softmaxMaxSize(
        tilingData.attentionScoreSingleCoreParams.get_singleProcessSOuterSize() * (BYTE_BLOCK / sizeof(float)));
    tilingData.attentionScoreSingleCoreTensorSize.set_softmaxSumSize(
        tilingData.attentionScoreSingleCoreTensorSize.get_softmaxMaxSize());
    tilingData.attentionScoreSingleCoreTensorSize.set_softmaxExpSize(
        tilingData.attentionScoreSingleCoreParams.get_singleProcessSOuterSize() *
        tilingData.attentionScoreOffestStrideParams.get_typeByteNum());
    tilingData.attentionScoreSingleCoreTensorSize.set_bmm2ResUbSize(
        tilingData.attentionScoreSingleCoreParams.get_singleProcessSOuterSize() * tilingData.baseParams.get_headSize());
    tilingData.attentionScoreSingleCoreTensorSize.set_tmpMMResBmm2PreUbSize(
        tilingData.attentionScoreSingleCoreTensorSize.get_mmResUbSize() +
        tilingData.attentionScoreSingleCoreTensorSize.get_bmm2ResUbSize());
    if (isPipeLine) {
        tilingData.attentionScoreSingleCoreTensorSize.set_tmpSoftmaxBmm2UbSize(
            tilingData.attentionScoreSingleCoreTensorSize.get_bmm2ResUbSize() +
            tilingData.attentionScoreSingleCoreTensorSize.get_softmaxExpSize()); // 3:softmax size
    } else {
        tilingData.attentionScoreSingleCoreTensorSize.set_tmpSoftmaxBmm2UbSize(
            std::max(tilingData.attentionScoreSingleCoreTensorSize.get_bmm2ResUbSize(),
            tilingData.attentionScoreSingleCoreTensorSize.get_softmaxExpSize())); // 3:softmax size
    }
    tilingData.attentionScoreSingleCoreTensorSize.set_spmTmpSize(
        tilingData.attentionScoreSingleCoreTensorSize.get_bmm2ResUbSize() +
        tilingData.attentionScoreSingleCoreTensorSize.get_softmaxExpSize() * 2); // 2:exp max
    tilingData.attentionScoreSingleCoreTensorSize.set_scmTmpSize(
        tilingData.attentionScoreSingleCoreTensorSize.get_mmResUbSize());

    tilingData.attentionScoreSingleCoreTensorSize.set_tailCoreMMResUbSize(
        tilingData.attentionScoreSingleCoreParams.get_tailCoreSingleProcessSOuterSize() *
        tilingData.attentionScoreSingleCoreParams.get_tailCoreSingleProcessSInnerSize());
    tilingData.attentionScoreSingleCoreTensorSize.set_tailCoreSoftmaxMaxSize(
        tilingData.attentionScoreSingleCoreParams.get_tailCoreSingleProcessSOuterSize() * (BYTE_BLOCK / sizeof(float)));
    tilingData.attentionScoreSingleCoreTensorSize.set_tailCoreSoftmaxSumSize(
        tilingData.attentionScoreSingleCoreTensorSize.get_tailCoreSoftmaxMaxSize());
    tilingData.attentionScoreSingleCoreTensorSize.set_tailCoreSoftmaxExpSize(
        tilingData.attentionScoreSingleCoreParams.get_tailCoreSingleProcessSOuterSize() *
        tilingData.attentionScoreOffestStrideParams.get_typeByteNum());
    tilingData.attentionScoreSingleCoreTensorSize.set_tailCoreBmm2ResUbSize(
        tilingData.attentionScoreSingleCoreParams.get_tailCoreSingleProcessSOuterSize() *
        tilingData.baseParams.get_headSize());
    tilingData.attentionScoreSingleCoreTensorSize.set_tailCoreTmpMMResBmm2PreUbSize(
        tilingData.attentionScoreSingleCoreTensorSize.get_tailCoreAttenMaskUbSize() +
        tilingData.attentionScoreSingleCoreTensorSize.get_tailCoreBmm2ResUbSize());
    if (isPipeLine) {
        tilingData.attentionScoreSingleCoreTensorSize.set_tailCoreTmpSoftmaxBmm2UbSize(
            tilingData.attentionScoreSingleCoreTensorSize.get_tailCoreBmm2ResUbSize() +
            tilingData.attentionScoreSingleCoreTensorSize.get_tailCoreSoftmaxExpSize()); // 3:softmax size
    } else {
        tilingData.attentionScoreSingleCoreTensorSize.set_tailCoreTmpSoftmaxBmm2UbSize(
            std::max(tilingData.attentionScoreSingleCoreTensorSize.get_tailCoreBmm2ResUbSize(),
            tilingData.attentionScoreSingleCoreTensorSize.get_tailCoreSoftmaxExpSize())); // 3:softmax size
    }

    tilingData.attentionScoreSingleCoreTensorSize.set_tailCoreSpmTmpSize(
        tilingData.attentionScoreSingleCoreTensorSize.get_tailCoreBmm2ResUbSize() +
        tilingData.attentionScoreSingleCoreTensorSize.get_tailCoreSoftmaxExpSize() * 2); // 2:exp max
    tilingData.attentionScoreSingleCoreTensorSize.set_tailCoreScmTmpSize(
        tilingData.attentionScoreSingleCoreTensorSize.get_tailCoreAttenMaskUbSize());

    int64_t apiTmpSize = std::max(dropoutTmpSize, softmaxFlashTmpSize);
    int64_t apiTmpSizeTail = std::max(dropoutTmpSizeTail, softmaxFlashTmpSizeTail);

    // 2:exp max
    int64_t queueBufferSizeByte = (tilingData.attentionScoreSingleCoreTensorSize.get_attenMaskUbSize() +
        2 * tilingData.attentionScoreSingleCoreTensorSize.get_softmaxExpSize()) *
        calcTypeSize;
    // 2:exp max
    int64_t queueBufferSizeTailByte = (tilingData.attentionScoreSingleCoreTensorSize.get_tailCoreAttenMaskUbSize() +
        2 * tilingData.attentionScoreSingleCoreTensorSize.get_tailCoreSoftmaxExpSize()) *
        calcTypeSize;
    if (inputTypeSize == 2 && calcTypeSize == 4) { // DT_BF16, 2: bf16 bytes, 4: calc float dtype size
        int64_t castTensorSize =
            std::max(tilingData.attentionScoreSingleCoreTensorSize.get_attenMaskUbSize() * calcTypeSize,
            tilingData.attentionScoreSingleCoreTensorSize.get_bmm2ResUbSize() * inputTypeSize);
        int64_t castTensorSizeTail =
            std::max(tilingData.attentionScoreSingleCoreTensorSize.get_tailCoreAttenMaskUbSize() * calcTypeSize,
            tilingData.attentionScoreSingleCoreTensorSize.get_tailCoreBmm2ResUbSize() * inputTypeSize);
        castTensorSize = std::max(
            (int64_t)tilingData.attentionScoreSingleCoreTensorSize.get_mmResUbSize() * inputTypeSize, castTensorSize);
        castTensorSizeTail =
            std::max((int64_t)tilingData.attentionScoreSingleCoreTensorSize.get_tailCoreMMResUbSize() * inputTypeSize,
            castTensorSizeTail);
        // 2:exp max
        queueBufferSizeByte = tilingData.attentionScoreSingleCoreTensorSize.get_attenMaskUbSize() * inputTypeSize +
            2 * tilingData.attentionScoreSingleCoreTensorSize.get_softmaxExpSize() * calcTypeSize;
        apiTmpSize = std::max(castTensorSize, apiTmpSize);
        // 2:exp max
        queueBufferSizeTailByte =
            tilingData.attentionScoreSingleCoreTensorSize.get_tailCoreAttenMaskUbSize() * inputTypeSize +
            2 * tilingData.attentionScoreSingleCoreTensorSize.get_tailCoreSoftmaxMaxSize() * calcTypeSize;
        apiTmpSizeTail = std::max(castTensorSizeTail, apiTmpSizeTail);
    }
    // Bufsize calc
    // mm1resQue + mm2resQue
    int64_t mmInSize = tilingData.attentionScoreSingleCoreTensorSize.get_mmResUbSize() +
        tilingData.attentionScoreSingleCoreTensorSize.get_bmm2ResUbSize();
    int64_t stageLen1 = mmInSize * calcTypeSize;
    // softmax
    int64_t stageLen2 = apiTmpSize + (mmInSize + tilingData.attentionScoreSingleCoreTensorSize.get_softmaxExpSize() +
        tilingData.attentionScoreSingleCoreTensorSize.get_bmm2ResUbSize()) *
        calcTypeSize;
    // update 2 * bmm1
    int64_t stageLen3 = (mmInSize + tilingData.attentionScoreSingleCoreTensorSize.get_bmm2ResUbSize()) * calcTypeSize;
    int64_t tmpBufLen = std::max(std::max(stageLen1, stageLen2), stageLen3);

    int64_t mmInSizeTail = tilingData.attentionScoreSingleCoreTensorSize.get_tailCoreMMResUbSize() +
        tilingData.attentionScoreSingleCoreTensorSize.get_tailCoreBmm2ResUbSize();
    int64_t stageLen1Tail = mmInSizeTail * calcTypeSize;
    int64_t stageLen2Tail =
        apiTmpSizeTail + (mmInSizeTail + tilingData.attentionScoreSingleCoreTensorSize.get_tailCoreSoftmaxExpSize() +
        tilingData.attentionScoreSingleCoreTensorSize.get_tailCoreBmm2ResUbSize()) *
        calcTypeSize;
    int64_t stageLen3Tail =
        (mmInSizeTail + tilingData.attentionScoreSingleCoreTensorSize.get_tailCoreBmm2ResUbSize()) * calcTypeSize;
    int64_t tmpBufLenTail = std::max(std::max(stageLen1Tail, stageLen2Tail), stageLen3Tail);
    tilingData.attentionScoreSingleCoreTensorSize.set_mmResInUbSize(mmInSize);
    tilingData.attentionScoreSingleCoreTensorSize.set_mmResInUbSizeTail(mmInSizeTail);

    ubSizeRemain = ubSize - queueBufferSizeByte - tmpBufLen;
    ubSizeRemainTail = ubSize - queueBufferSizeTailByte - tmpBufLenTail;

    tBufRemain = ubSizeRemain + apiTmpSize;
    tBufRemainTail = ubSizeRemainTail + apiTmpSizeTail;

    if (ubSizeRemain < 0) {
        updateDiv = true;
        res = false;
    }
    if (ubSizeRemainTail < 0) {
        updateDivTail = true;
        res = false;
    }
    if (!res) {
        return res;
    }

    int32_t l1SizeRemain = l1Size - calcTypeSize * (tilingData.attentionScoreSingleCoreTensorSize.get_spmTmpSize() +
        tilingData.attentionScoreSingleCoreTensorSize.get_scmTmpSize());
    int32_t l1SizeRemainTail =
        l1Size - calcTypeSize * (tilingData.attentionScoreSingleCoreTensorSize.get_tailCoreSpmTmpSize() +
        tilingData.attentionScoreSingleCoreTensorSize.get_tailCoreScmTmpSize());
    if (l1SizeRemain < 0) {
        updateDiv = true;
        res = false;
    }
    if (l1SizeRemainTail < 0) {
        updateDivTail = true;
        res = false;
    }
    if (!res) {
        return res;
    }

    matmul_tiling::MatmulApiTiling bmm1;
    matmul_tiling::MatmulApiTiling bmm2;
    matmul_tiling::MatmulApiTiling bmm1TailCore;
    matmul_tiling::MatmulApiTiling bmm2TailCore;

    bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDype, false);
    bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDype, true);
    bmm1.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, matmulOutDtype);
    bmm1.SetShape(tilingData.attentionScoreSingleCoreParams.get_singleProcessSOuterSize(),
        tilingData.attentionScoreSingleCoreParams.get_singleProcessSInnerSize() * MAT_L0B_FACTOR,
        tilingData.baseParams.get_headSize());
    bmm1.SetOrgShape(tilingData.baseParams.get_seqSizeQ(), tilingData.baseParams.get_seqSizeK(),
        tilingData.attentionScoreOffestStrideParams.get_matmulHead());
    bmm1.SetBias(false);
    // 2: space half
    bmm1.SetBufferSpace(l1SizeRemain, l0CSize);
    bmm1.SetFixSplit(tilingData.attentionScoreSingleCoreParams.get_singleProcessSOuterSize(),
        tilingData.attentionScoreSingleCoreParams.get_singleProcessSInnerSize());
    if (bmm1.GetTiling(tilingData.bmm1TilingData) == -1) {
        updateDiv = true;
        res = false;
    }
    tilingData.bmm1TilingData.set_depthB1(2); // 2: space half
    tilingData.bmm1TilingData.set_shareMode(0);
    tilingData.bmm1TilingData.set_shareL1Size(l1SizeRemain);
    tilingData.bmm1TilingData.set_shareL0CSize(l0CSize);
    tilingData.bmm1TilingData.set_shareUbSize(0);

    bmm2.SetAType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, matmulDype, false);
    bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDype, false);
    bmm2.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, matmulOutDtype);
    bmm2.SetShape(tilingData.attentionScoreSingleCoreParams.get_singleProcessSOuterSize(),
        tilingData.baseParams.get_headSize(), tilingData.attentionScoreSingleCoreParams.get_singleProcessSInnerSize());
    bmm2.SetOrgShape(tilingData.baseParams.get_seqSizeQ(), tilingData.attentionScoreOffestStrideParams.get_matmulHead(),
        tilingData.baseParams.get_seqSizeV());
    bmm2.SetBias(false);
    // 2: space half
    bmm2.SetBufferSpace(l1SizeRemain, l0CSize);
    bmm2.SetFixSplit(tilingData.attentionScoreSingleCoreParams.get_singleProcessSOuterSize(),
        tilingData.baseParams.get_headSize());
    if (bmm2.GetTiling(tilingData.bmm2TilingData) == -1) {
        updateDiv = true;
        res = false;
    }
    tilingData.bmm2TilingData.set_shareMode(0);
    tilingData.bmm2TilingData.set_shareL1Size(l1SizeRemain);
    tilingData.bmm2TilingData.set_shareL0CSize(l0CSize);
    tilingData.bmm2TilingData.set_shareUbSize(0);

    uint32_t sOuterSizeTail;
    if (isSparse) {
        sOuterSizeTail = tilingData.attentionScoreSingleCoreParams.get_lastCoreSingleProcessSOuterSizeTail();
    } else {
        sOuterSizeTail = tilingData.attentionScoreSingleCoreParams.get_tailCoreSingleProcessSOuterSize();
    }

    bmm1TailCore.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDype, false);
    bmm1TailCore.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDype, true);
    bmm1TailCore.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, matmulOutDtype);
    bmm1TailCore.SetShape(sOuterSizeTail,
        tilingData.attentionScoreSingleCoreParams.get_tailCoreSingleProcessSInnerSize() * MAT_L0B_FACTOR,
        tilingData.baseParams.get_headSize());
    bmm1TailCore.SetOrgShape(tilingData.baseParams.get_seqSizeQ(), tilingData.baseParams.get_seqSizeK(),
        tilingData.attentionScoreOffestStrideParams.get_matmulHead());
    bmm1TailCore.SetBias(false);
    // 2: space half
    bmm1TailCore.SetBufferSpace(l1SizeRemain, l0CSize);
    bmm1TailCore.SetFixSplit(sOuterSizeTail,
        tilingData.attentionScoreSingleCoreParams.get_tailCoreSingleProcessSInnerSize());
    if (bmm1TailCore.GetTiling(tilingData.bmm1TilingDataTailCore) == -1) {
        updateDivTail = true;
        res = false;
    }
    tilingData.bmm1TilingDataTailCore.set_depthB1(2); // 2: space half
    tilingData.bmm1TilingDataTailCore.set_shareMode(0);
    tilingData.bmm1TilingDataTailCore.set_shareL1Size(l1SizeRemain);
    tilingData.bmm1TilingDataTailCore.set_shareL0CSize(l0CSize);
    tilingData.bmm1TilingDataTailCore.set_shareUbSize(0);
    bmm2TailCore.SetAType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, matmulDype, false);
    bmm2TailCore.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDype, false);
    bmm2TailCore.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, matmulOutDtype);
    bmm2TailCore.SetShape(sOuterSizeTail, tilingData.baseParams.get_headSize(),
        tilingData.attentionScoreSingleCoreParams.get_tailCoreSingleProcessSInnerSize());
    bmm2TailCore.SetOrgShape(tilingData.baseParams.get_seqSizeQ(),
        tilingData.attentionScoreOffestStrideParams.get_matmulHead(), tilingData.baseParams.get_seqSizeK());
    bmm2TailCore.SetBias(false);
    // 2: space half
    bmm2TailCore.SetBufferSpace(l1SizeRemain, l0CSize);
    bmm2TailCore.SetFixSplit(sOuterSizeTail, tilingData.baseParams.get_headSize());
    if (bmm2TailCore.GetTiling(tilingData.bmm2TilingDataTailCore) == -1) {
        updateDivTail = true;
        res = false;
    }
    tilingData.bmm2TilingDataTailCore.set_shareMode(0);
    tilingData.bmm2TilingDataTailCore.set_shareL1Size(l1SizeRemain);
    tilingData.bmm2TilingDataTailCore.set_shareL0CSize(l0CSize);
    tilingData.bmm2TilingDataTailCore.set_shareUbSize(0);
    return res;
}

ge::graphStatus FlashAttentionScoreTiling::FlashAttentionScoreApiTiling(FlashAttentionScoreTilingData &tilingData)
{
    uint32_t sOuterSizeTail;
    if (isSparse) {
        sOuterSizeTail = tilingData.attentionScoreSingleCoreParams.get_lastCoreSingleProcessSOuterSizeTail();
    } else {
        sOuterSizeTail = tilingData.attentionScoreSingleCoreParams.get_tailCoreSingleProcessSOuterSize();
    }

    auto softmaxShape = Shape({ tilingData.attentionScoreSingleCoreParams.get_singleProcessSOuterSize(),
        tilingData.attentionScoreSingleCoreParams.get_singleProcessSInnerSize() });
    auto softmaxShapeTail = Shape({ tilingData.attentionScoreSingleCoreParams.get_tailCoreSingleProcessSOuterSize(),
        tilingData.attentionScoreSingleCoreParams.get_tailCoreSingleProcessSInnerSize() });

    bool isBasicBlock;
    bool isBasicBlockTail;
    if (tilingData.attentionScoreSingleCoreParams.get_singleProcessSOuterSize() % SOFTMAX_BASE_M == 0 &&
        tilingData.attentionScoreSingleCoreParams.get_singleProcessSInnerSize() % SOFTMAX_BASE_N == 0) {
        isBasicBlock = true;
    } else {
        isBasicBlock = false;
    }

    if (tilingData.attentionScoreSingleCoreParams.get_tailCoreSingleProcessSOuterSize() % SOFTMAX_BASE_M == 0 &&
        tilingData.attentionScoreSingleCoreParams.get_tailCoreSingleProcessSInnerSize() % SOFTMAX_BASE_N == 0) {
        isBasicBlockTail = true;
    } else {
        isBasicBlockTail = false;
    }

    tilingData.attentionScoreSingleCoreTensorSize.set_apiTmpUbSize(tBufRemain);
    tilingData.attentionScoreSingleCoreTensorSize.set_apiTmpUbSizeTail(tBufRemainTail);
    SoftMaxTilingFunc(softmaxShape, sizeof(float), tBufRemain, tilingData.softmaxTilingData);
    SoftMaxFlashV2TilingFunc(softmaxShape, calcTypeSize, sizeof(float), tBufRemain, tilingData.softmaxFlashTilingData,
        true, isBasicBlock);
    SoftMaxTilingFunc(softmaxShapeTail, sizeof(float), tBufRemainTail, tilingData.softmaxTilingDataTailCore);
    SoftMaxFlashV2TilingFunc(softmaxShapeTail, calcTypeSize, sizeof(float), tBufRemainTail,
        tilingData.softmaxFlashTilingDataTailCore, true, isBasicBlockTail);

    tilingData.attentionScoreSingleCoreTensorSize.set_softmaxTmpBuffer(tBufRemain);
    tilingData.attentionScoreSingleCoreTensorSize.set_dropOutTmpBuffer(tBufRemain);

    auto transposeSrcShape = Shape({ tilingData.attentionScoreSingleCoreParams.get_singleProcessBatchSize(),
        tilingData.attentionScoreSingleCoreParams.get_singleProcessHeadNumSize(),
        tilingData.attentionScoreSingleCoreParams.get_singleProcessSOuterSize(),
        tilingData.attentionScoreSingleCoreParams.get_singleProcessHeadSize() });
    auto transposeDstShape = Shape({ tilingData.baseParams.get_batchSize(), tilingData.baseParams.get_headNumSize(),
        tilingData.baseParams.get_seqSizeQ(),
        tilingData.baseParams.get_headSize() * tilingData.baseParams.get_headNumSize() });
    auto transposeSrcShapeTail = Shape({ tilingData.attentionScoreSingleCoreParams.get_singleProcessBatchSize(),
        tilingData.attentionScoreSingleCoreParams.get_singleProcessHeadNumSize(), sOuterSizeTail,
        tilingData.attentionScoreSingleCoreParams.get_singleProcessHeadSize() });
    GetDataCopyTransposeTiling(transposeDstShape, transposeSrcShape, inputTypeSize, tilingData.transposeTilingData);
    GetDataCopyTransposeTiling(transposeDstShape, transposeSrcShapeTail, inputTypeSize,
        tilingData.transposeTilingDataTailCore);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreTiling::FlashAttentionScoreGetTensorSize(FlashAttentionScoreTilingData &tilingData)
{
    tilingData.attentionScoreSingleCoreTensorSize.set_maskSize(
        tilingData.attentionScoreSingleCoreTensorSize.get_mmResUbSize());
    tilingData.attentionScoreSingleCoreTensorSize.set_mmResUbSizeTailLoop(
        tilingData.attentionScoreSingleCoreParams.get_singleProcessSOuterSize() *
        tilingData.attentionScoreSingleCoreParams.get_singleProcessSInnerSizeTail());
    tilingData.attentionScoreSingleCoreTensorSize.set_attenMaskUbSizeTailLoop(
        tilingData.attentionScoreSingleCoreTensorSize.get_mmResUbSizeTailLoop());
    tilingData.attentionScoreSingleCoreTensorSize.set_maskSizeTailLoop(
        tilingData.attentionScoreSingleCoreTensorSize.get_mmResUbSizeTailLoop());

    tilingData.attentionScoreSingleCoreTensorSize.set_tailCoreMaskSize(
        tilingData.attentionScoreSingleCoreParams.get_tailCoreSingleProcessSOuterSize() *
        tilingData.attentionScoreSingleCoreParams.get_tailCoreSingleProcessSInnerSize());
    tilingData.attentionScoreSingleCoreTensorSize.set_tailCoreMMResUbSizeTailLoop(
        tilingData.attentionScoreSingleCoreParams.get_tailCoreSingleProcessSOuterSize() *
        tilingData.attentionScoreSingleCoreParams.get_tailCoreSingleProcessSInnerSizeTail());
    tilingData.attentionScoreSingleCoreTensorSize.set_tailCoreAttenMaskUbSizeTailLoop(
        tilingData.attentionScoreSingleCoreTensorSize.get_tailCoreMMResUbSizeTailLoop());
    tilingData.attentionScoreSingleCoreTensorSize.set_tailCoreMaskSizeTailLoop(
        tilingData.attentionScoreSingleCoreTensorSize.get_tailCoreMMResUbSizeTailLoop());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreTiling::FlashAttentionScoreSetTilingData(gert::TilingContext *context,
    FlashAttentionScoreTilingData &tilingData)
{
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreTiling::GetSeqSplitList(FlashAttentionScoreTilingData &tilingData, uint32_t seqSplit,
    std::queue<uint32_t> &sQueue)
{
    uint32_t baseSize = tilingData.baseParams.get_seqBaseSize() == 128 ? 64 : 16;
    for (uint32_t i = 2; i <= seqSplit; i++) {
        if (seqSplit % i == 0 && (seqSplit / i) % baseSize == 0) {
            sQueue.push(seqSplit / i);
        }
    }
    return ge::GRAPH_SUCCESS;
}

bool FlashAttentionScoreTiling::IsHeadNumTail(FlashAttentionScoreTilingData &tilingData, uint32_t coreNum)
{
    uint32_t b1;
    uint32_t n1;
    uint32_t s1;
    b1 = tilingData.baseParams.get_batchSize();
    if (tilingData.baseParams.get_batchSize() < coreNum && coreNum % tilingData.baseParams.get_batchSize() != 0) {
        b1 = Gcd(coreNum, tilingData.baseParams.get_batchSize());
    }
    if (tilingData.baseParams.get_batchSize() > coreNum && coreNum != 0 &&
        tilingData.baseParams.get_batchSize() % coreNum != 0) {
        b1 = Gcd(tilingData.baseParams.get_batchSize(), coreNum);
    }
    if (b1 == 0) {
        return false;
    }
    n1 = Gcd(coreNum / b1, tilingData.baseParams.get_headNumSize());
    if (n1 == 0) {
        return false;
    }
    s1 = coreNum / b1 / n1;
    // 128 : basic seq length
    if (s1 <= tilingData.baseParams.get_seqSizeQ() / 128) {
        return false;
    }
    // 128 : basic seq length
    if (tilingData.baseParams.get_seqSizeQ() % 128 == 0) {
        return true;
    }
    if (tilingData.baseParams.get_headNumSize() % n1 == 0 && n1 != 1) {
        return false;
    }
    return true;
}

ge::graphStatus FlashAttentionScoreTiling::RunFlashAttentionScoreTiling(gert::TilingContext *context,
    FlashAttentionScoreTilingData &tilingData, uint32_t coreNum, int64_t ubSize, int64_t l1Size, int64_t l0CSize)
{
    std::queue<uint32_t> sOuterQueue;
    std::queue<uint32_t> sInnerQueue;
    std::queue<uint32_t> sOuterQueueTail;
    std::queue<uint32_t> sInnerQueueTail;
    // Souter pretail
    GetSeqSplitList(tilingData, tilingData.attentionScoreCoreParams.get_singleCoreSeqSize(), sOuterQueue);
    GetSeqSplitList(tilingData, tilingData.attentionScoreCoreParams.get_singleCoreSeqSizeTail(), sOuterQueueTail);
    GetSeqSplitList(tilingData, tilingData.baseParams.get_seqSizeK(), sInnerQueue);
    GetSeqSplitList(tilingData, tilingData.baseParams.get_seqSizeK(), sInnerQueueTail);
    tilingData.attentionScoreSingleCoreParams.set_singleProcessSOuterSize(
        tilingData.attentionScoreCoreParams.get_singleCoreSeqSize());
    tilingData.attentionScoreSingleCoreParams.set_tailCoreSingleProcessSOuterSize(
        tilingData.attentionScoreCoreParams.get_singleCoreSeqSizeTail());

    // Sinner pretail
    if (!isRec) {
        tilingData.attentionScoreSingleCoreParams.set_singleProcessSOuterSize(tilingData.baseParams.get_seqBaseSize());
        tilingData.attentionScoreSingleCoreParams.set_tailCoreSingleProcessSOuterSize(
            tilingData.baseParams.get_seqBaseSize());
    }
    tilingData.attentionScoreSingleCoreParams.set_singleProcessSInnerSize(tilingData.baseParams.get_seqSizeK());
    tilingData.attentionScoreSingleCoreParams.set_tailCoreSingleProcessSInnerSize(tilingData.baseParams.get_seqSizeK());

    while (true) {
        bool updateDiv = false;
        bool updateDivTail = false;
        FlashAttentionScoreSingleCoreTiling(context, tilingData);
        if (FlashAttentionScoreCheckArgsLegal(tilingData, ubSize, l1Size, l0CSize, updateDiv, updateDivTail)) {
            break;
        }
        if (updateDiv) {
            // 16 : souter threshold
            if (tilingData.attentionScoreSingleCoreParams.get_singleProcessSOuterSize() <= 16 || !isRec ||
                sOuterQueue.empty()) {
                if (!sInnerQueue.empty()) {
                    uint32_t a = sInnerQueue.front();
                    sInnerQueue.pop();
                    tilingData.attentionScoreSingleCoreParams.set_singleProcessSInnerSize(a);
                } else {
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "sInnerQueue is empty");
                    return ge::GRAPH_FAILED;
                }
            } else {
                uint32_t b = sOuterQueue.front();
                sOuterQueue.pop();
                tilingData.attentionScoreSingleCoreParams.set_singleProcessSOuterSize(b);
            }
        }
        if (updateDivTail) {
            // 16 : souter threshold
            if (tilingData.attentionScoreSingleCoreParams.get_tailCoreSingleProcessSOuterSize() <= 16 || !isRec ||
                sOuterQueueTail.empty()) {
                if (!sInnerQueueTail.empty()) {
                    uint32_t a = sInnerQueueTail.front();
                    sInnerQueueTail.pop();
                    tilingData.attentionScoreSingleCoreParams.set_tailCoreSingleProcessSInnerSize(a);
                } else {
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "sInnerQueueTail is empty");
                    return ge::GRAPH_FAILED;
                }
            } else {
                uint32_t b = sOuterQueueTail.front();
                sOuterQueueTail.pop();
                tilingData.attentionScoreSingleCoreParams.set_tailCoreSingleProcessSOuterSize(b);
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreTiling::RunSparseFlashAttentionScoreTiling(gert::TilingContext *context,
    FlashAttentionScoreTilingData &tilingData, uint32_t coreNum, int64_t ubSize, int64_t l1Size, int64_t l0CSize)
{
    std::queue<uint32_t> sOuterQueue;

    GetSeqSplitList(tilingData, tilingData.baseParams.get_seqBaseSize(), sOuterQueue);

    tilingData.attentionScoreSingleCoreParams.set_singleProcessSOuterSize(tilingData.baseParams.get_seqBaseSize());
    tilingData.attentionScoreSingleCoreParams.set_singleProcessSOuterSizeTail(tilingData.baseParams.get_seqBaseSize());
    tilingData.attentionScoreSingleCoreParams.set_tailCoreSingleProcessSOuterSize(
        tilingData.baseParams.get_seqBaseSize());
    tilingData.attentionScoreSingleCoreParams.set_tailCoreSingleProcessSOuterSizeTail(
        tilingData.baseParams.get_seqBaseSize());

    tilingData.attentionScoreSingleCoreParams.set_lastCoreSingleProcessSOuterSize(
        tilingData.baseParams.get_seqBaseSize());
    tilingData.attentionScoreSingleCoreParams.set_lastCoreSingleProcessSOuterSizeTail(
        tilingData.baseParams.get_seqBaseSize() - tilingData.baseParams.get_seqBasePadValue());

    tilingData.attentionScoreSingleCoreParams.set_singleProcessSInnerSize(tilingData.baseParams.get_seqBaseSize());
    tilingData.attentionScoreSingleCoreParams.set_tailCoreSingleProcessSInnerSize(
        tilingData.baseParams.get_seqBaseSize());

    while (true) {
        bool updateDiv = false;
        bool updateDivTail = false;

        SparseFlashAttentionScoreSingleCoreTiling(context, tilingData);
        if (FlashAttentionScoreCheckArgsLegal(tilingData, ubSize, l1Size, l0CSize, updateDiv, updateDivTail)) {
            break;
        }
        if (updateDiv || updateDivTail) {
            if (sOuterQueue.empty() ||
                (tilingData.attentionScoreSingleCoreParams.get_singleProcessSOuterSize() <= 16)) { // 16: seq threshold
                break;
            } else {
                if (tilingData.attentionScoreSingleCoreParams.get_singleProcessSOuterSize() >=
                    tilingData.attentionScoreSingleCoreParams.get_singleProcessSInnerSize()) {
                    uint32_t singleProcessSOuterSize =
                        tilingData.attentionScoreSingleCoreParams.get_singleProcessSOuterSize() / 2;

                    tilingData.baseParams.set_seqBaseSize(singleProcessSOuterSize);
                    tilingData.baseParams.set_seqBaseRange(
                        Ceil(tilingData.baseParams.get_seqSizeK(), tilingData.baseParams.get_seqBaseSize()));
                    tilingData.baseParams.set_seqSizeAlign(tilingData.baseParams.get_seqBaseRange() *
                        tilingData.baseParams.get_seqBaseSize());
                    tilingData.baseParams.set_seqBasePadValue(tilingData.baseParams.get_seqSizeAlign() -
                        tilingData.baseParams.get_seqSizeK());

                    tilingData.attentionScoreSingleCoreParams.set_singleProcessSOuterSize(singleProcessSOuterSize);
                    tilingData.attentionScoreSingleCoreParams.set_singleProcessSOuterSizeTail(singleProcessSOuterSize);
                    tilingData.attentionScoreSingleCoreParams.set_tailCoreSingleProcessSOuterSize(
                        singleProcessSOuterSize);

                    tilingData.attentionScoreSingleCoreParams.set_lastCoreSingleProcessSOuterSizeTail(
                        tilingData.baseParams.get_seqBaseSize() - tilingData.baseParams.get_seqBasePadValue());
                } else {
                    uint32_t singleProcessSInnerSize =
                        tilingData.attentionScoreSingleCoreParams.get_singleProcessSInnerSize() / 2;
                    tilingData.attentionScoreSingleCoreParams.set_singleProcessSInnerSize(singleProcessSInnerSize);
                    tilingData.attentionScoreSingleCoreParams.set_tailCoreSingleProcessSInnerSize(
                        singleProcessSInnerSize);
                }
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreTiling::RunBigKernelTiling(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    uint64_t ubSizePlatForm = 0;
    uint64_t l1SizePlatForm = 0;
    uint64_t l0CSizePlatForm = 0;
    uint32_t coreNum = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    if (platformInfo != nullptr) {
        coreNum = ascendcPlatform.GetCoreNumAiv();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1SizePlatForm);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSizePlatForm);
    }
    int64_t ubSize = (int64_t)ubSizePlatForm;
    int64_t l1Size = (int64_t)l1SizePlatForm;
    int64_t l0CSize = (int64_t)l0CSizePlatForm;

    auto attrs = context->GetAttrs();
    const float *keepProb = attrs->GetAttrPointer<float>(1);
    const float *scaleValue = attrs->GetAttrPointer<float>(0);
    bool isTransposeOut = true;
    const uint32_t *preTockens = attrs->GetAttrPointer<uint32_t>(2);
    const uint32_t *nextTockens = attrs->GetAttrPointer<uint32_t>(3);
    const char *inputLayout = attrs->GetAttrPointer<char>(5);
    const int *innerPrecisePtr = attrs->GetAttrPointer<int>(6);
    if (innerPrecisePtr != nullptr) {
        innerPrecise = *innerPrecisePtr;
    }
    OP_LOGD(context->GetNodeName(), "inner_precise is %d", innerPrecise);

    calcTypeSize = 4; // 4: default dtype size
    matmulDype = matmul_tiling::DataType::DT_FLOAT;
    matmulOutDtype = matmul_tiling::DataType::DT_FLOAT;
    ge::DataType attentionScoreType = context->GetInputDesc(0)->GetDataType();
    inputTypeSize = 4;
    if (attentionScoreType != ge::DT_FLOAT16) {
        innerPrecise = HIGH_PERFORMANCE;
    }
    if (attentionScoreType == ge::DT_FLOAT16) { // attention_score type: fp16
        if (innerPrecise == HIGH_PRECISION) {
            calcTypeSize = 4;
            matmulDype = matmul_tiling::DataType::DT_FLOAT16;
            matmulOutDtype = matmul_tiling::DataType::DT_FLOAT;
            inputTypeSize = 2;
        } else {
            calcTypeSize = 2;
            matmulDype = matmul_tiling::DataType::DT_FLOAT16;
            matmulOutDtype = matmul_tiling::DataType::DT_FLOAT16;
            inputTypeSize = 2;
        }
    }
    if (attentionScoreType == ge::DT_BF16) { // attention_score type: bf16
        inputTypeSize = 2;                   // 2: bf16, 2bytes
        matmulDype = matmul_tiling::DataType::DT_BF16;
        matmulOutDtype = matmul_tiling::DataType::DT_FLOAT;
    }

    if (context->GetOptionalInputShape(ATTENTION_MASK_INPUT_INDEX) != nullptr) {
        ge::DataType attenMaskType = context->GetOptionalInputDesc(ATTENTION_MASK_INPUT_INDEX)->GetDataType();
        uint32_t attenMaskTypeIdx; // 0:fp, 1:int8, 2:int1
        attenMaskTypeIdx = (attenMaskType == ge::DT_BOOL) ? 1 : 0;
        tilingData.baseParams.set_attenMaskTypeIdx(attenMaskTypeIdx);
    } else {
        // default
        tilingData.baseParams.set_attenMaskTypeIdx(ATTE_SHAPE_TYPE_NONE);
    }

    tilingData.attentionScoreOffestStrideParams.set_typeByteNum(BYTE_BLOCK / calcTypeSize);
    // get shape
    const gert::StorageShape *attenShape;
    const gert::StorageShape *queryShape = context->GetInputShape(0);
    const gert::StorageShape *keyShape = context->GetInputShape(1);
    const gert::StorageShape *valueShape = context->GetInputShape(2);
    uint32_t tilingKey;

    // get dims
    uint32_t b = 0;
    uint32_t h = 0;
    uint32_t vh = 0;
    uint32_t qS = 0;
    uint32_t kS = 0;
    uint32_t vS = 0;
    if (inputLayout[0] == 'B' && inputLayout[1] == 'S' && inputLayout[2] == 'H') {
        b = queryShape->GetStorageShape().GetDim(0);
        qS = queryShape->GetStorageShape().GetDim(1); // query s
        kS = keyShape->GetStorageShape().GetDim(1);   // key s
        vS = valueShape->GetStorageShape().GetDim(1); // value s
        h = queryShape->GetStorageShape().GetDim(2);
        vh = valueShape->GetStorageShape().GetDim(2);
        tilingData.baseParams.set_transType(0);
        tilingData.attentionScoreOffestStrideParams.set_matmulHead(h);
    } else if (inputLayout[0] == 'S' && inputLayout[1] == 'B' && inputLayout[2] == 'H') {
        qS = queryShape->GetStorageShape().GetDim(0); // query s
        kS = keyShape->GetStorageShape().GetDim(0);   // key s
        vS = valueShape->GetStorageShape().GetDim(0); // value s
        b = queryShape->GetStorageShape().GetDim(1);
        h = queryShape->GetStorageShape().GetDim(2);
        vh = valueShape->GetStorageShape().GetDim(2);
        tilingData.baseParams.set_transType(1);
        tilingData.attentionScoreOffestStrideParams.set_matmulHead(b * h);
    } else {
        OP_LOGW(context->GetNodeName(), "invalid input_layout[%s].", inputLayout);
        return ge::GRAPH_PARAM_INVALID;
    }

    if (h != vh) {
        OP_LOGW(context->GetNodeName(), "Q and K's shape H is unequal.");
        return ge::GRAPH_PARAM_INVALID;
    }

    const uint32_t *tmp = attrs->GetAttrPointer<uint32_t>(4);
    uint32_t n = *tmp;
    uint32_t hDivN = h / n;

    attenShape = context->GetOptionalInputShape(ATTENTION_MASK_INPUT_INDEX);
    if (attenShape != nullptr) {
        tilingData.baseParams.set_attenBatchSize(1);
        tilingData.baseParams.set_attenHeadNum(1);
        if (attenShape->GetStorageShape().GetDim(0) == b) {
            tilingData.baseParams.set_attenBatchSize(b);
        } else if (attenShape->GetStorageShape().GetDim(1) == n) {
            tilingData.baseParams.set_attenHeadNum(n);
        }
    }

    // the general only support the case not in whitelist and not high perfermance mode
    SupportedShape curShape(n, qS, kS, hDivN, b);
    if (SupportShapeWhiteList.find(curShape) == SupportShapeWhiteList.end() &&
        (attentionScoreType != ge::DT_FLOAT16 || innerPrecise != HIGH_PERFORMANCE)) {
        OP_LOGW(context->GetNodeName(),
                "Unsupported shape: b, n, s1, s2, d is [%u],[%u],[%u],[%u],[%u]", b, n, qS, kS, hDivN);
        return ge::GRAPH_PARAM_INVALID;
    }

    tilingData.baseParams.set_batchSize(b);
    tilingData.baseParams.set_headNumSize(n);
    tilingData.baseParams.set_seqSizeQ(qS);
    tilingData.baseParams.set_seqSizeK(kS);
    tilingData.baseParams.set_seqSizeV(vS);
    tilingData.baseParams.set_headSize(hDivN);
    tilingData.baseParams.set_keepProb(*keepProb);
    tilingData.baseParams.set_scaleValue(*scaleValue);
    tilingData.baseParams.set_preTockens(*preTockens);
    tilingData.baseParams.set_nextTockens(*nextTockens);
    isNTail = IsHeadNumTail(tilingData, coreNum);
    isSparse = IsSparseAttentionScoreOri();
    if (isSparse) {
        while (ifUseAllCoreNum(coreNum, b, n, qS)) {
            coreNum = coreNum / 2; // 2 : half
        }
        uint32_t NewCoreNum = coreNum;
        uint32_t seqBaseSize = 128;
        GetSeqBaseAndNewCoreNum(coreNum, b, n, qS, seqBaseSize, NewCoreNum);
        if (coreNum > NewCoreNum) {
            coreNum = NewCoreNum;
        }
        tilingData.baseParams.set_seqBaseSize(seqBaseSize);
        tilingData.baseParams.set_seqBaseRange(
            Ceil(tilingData.baseParams.get_seqSizeK(), tilingData.baseParams.get_seqBaseSize()));
        tilingData.baseParams.set_seqSizeAlign(tilingData.baseParams.get_seqBaseRange() *
            tilingData.baseParams.get_seqBaseSize());
        tilingData.baseParams.set_seqBasePadValue(tilingData.baseParams.get_seqSizeAlign() -
            tilingData.baseParams.get_seqSizeK());
    }
    isPipeLine = IsPipeline();
    if (isPipeLine) {
        MAT_L0B_FACTOR = Ceil(tilingData.baseParams.get_seqSizeK(), tilingData.baseParams.get_seqBaseSize());
    }
    TilingGetTilingKeyAttentionScoreAscendC(tilingKey, context, coreNum, isTransposeOut);
    context->SetTilingKey(tilingKey);

    SupportedShape shape(n, qS, kS, hDivN, b);
    if (!isSparse && innerPrecise == HIGH_PRECISION && SDShapeWhiteList.find(shape) != SDShapeWhiteList.end()) {
        if (attenShape != nullptr){
            return ge::GRAPH_PARAM_INVALID;
        }
        OP_LOGD(context->GetNodeName(), "SD shape: n, s1, s2, d is [%u],[%u],[%u],[%u]", n, qS, kS, hDivN);

        auto resultCode = TilingRegistry::GetInstance().DoTilingImpl(context);
        return resultCode;
    }
    if (isSparse && attentionScoreType == ge::DT_BF16 && (SpecialShape175or65WhiteList.find(shape) != SpecialShape175or65WhiteList.end()||
        SpecialShape13WhiteList.find(shape) != SpecialShape13WhiteList.end())) {
        OP_LOGD(context->GetNodeName(), "175B 65B shape: n, s1, s2, d is [%u],[%u],[%u],[%u]", n, qS, kS, hDivN);
        tilingKey = 1011; // 1011: DT_BF16, sparse, 175B 65B, BNS split
    }

    // compute tilingdata
    switch (tilingKey) {
        case 0:
        case 2:    // DT_FLOAT16
        case 1002: // in:DT_FLOAT16, out:DT_FLOAT
        case 6:    // DT_FLOAT
        case 8:    // DT_BF16
        // sparse +16
        case 16:   // 16: DT_FLOAT16, sparse, no trans, BN split
        case 18:   // 18: DT_FLOAT16, sparse, BN split
        case 1016: // 1016: in:DT_FLOAT16, out:DT_FLOAT, sparse, no trans, BN split
        case 1018: // 1018: in:DT_FLOAT16, out:DT_FLOAT, sparse, BN split
        case 22:   // 22: DT_FLOAT, sparse, BN split
        case 24:   // 24: DT_BF16, sparse, BN split
        case 1010: // 1010: DT_FLOAT16, sparse, split BN
            FlashAttentionScoreSplitBN(context, tilingData, coreNum);
            break;
        case 1:
        case 3:    // DT_FLOAT16
        case 1003: // in:DT_FLOAT16, out:DT_FLOAT
        case 7:    // DT_FLOAT
        case 9:    // DT_BF16
        // sparse +16
        case 17:   // 17: DT_FLOAT16, sparse, no trans, BNS split
        case 19:   // 19: DT_FLOAT16, sparse, BNS split
        case 1017: // 1017: in:DT_FLOAT16, out:DT_FLOAT,  sparse, no trans, BNS split
        case 1019: // 1019: in:DT_FLOAT16, out:DT_FLOAT,  BNS split
        case 23:   // 23: DT_FLOAT, sparse, BNS split
        case 25:   // 25: DT_BF16, sparse, BNS split
        case 1011: // 1011: DT_BF16, sparse, 175B 65B, BNS split
            FlashAttentionScoreSplitBNS(context, tilingData, coreNum);
            break;
        default:
            return ge::GRAPH_FAILED;
    }

    if (isSparse) {
        RunSparseFlashAttentionScoreTiling(context, tilingData, coreNum, ubSize, l1Size, l0CSize);
    } else {
        RunFlashAttentionScoreTiling(context, tilingData, coreNum, ubSize, l1Size, l0CSize);
    }

    FlashAttentionScoreApiTiling(tilingData);
    FlashAttentionScoreGetTensorSize(tilingData);
    auto aicNum = ascendcPlatform.GetCoreNumAic();
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum));

    FlashAttentionScoreSetTilingData(context, tilingData);
    size_t *workspaces = context->GetWorkspaceSizes(1);
    auto mmResUbSize = tilingData.attentionScoreSingleCoreTensorSize.get_mmResUbSize();
    auto bmm2ResUbSize = tilingData.attentionScoreSingleCoreTensorSize.get_bmm2ResUbSize();
    workspaces[0] = tilingData.attentionScoreSingleCoreTensorSize.get_spmTmpSize() *
        tilingData.attentionScoreCoreParams.get_coreNum() * calcTypeSize +
        // 100 1024 1024 : fixed offset spaces
        100 * 1024 * 1024 + 
        (mmResUbSize * MAT_L0B_FACTOR + bmm2ResUbSize) * tilingData.attentionScoreCoreParams.get_coreNum() *
        calcTypeSize;
    // update the tilingKey with softmax tiling
    if (IsBasicBlockInSoftMax(tilingData.softmaxFlashTilingData, calcTypeSize)) {
        // 50 : tilingKey shift value
        context->SetTilingKey(tilingKey + 50);
    }
    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreEmptyInputTiling::GetTilingKeyAttentionScore4EmptyInput(uint32_t &tilingKey,
    gert::TilingContext *context)
{
    auto kernelType = context->GetInputDesc(KEY_INPUT_INDEX)->GetDataType();
    if (kernelType == ge::DT_FLOAT16) {
        tilingKey = 90;
    } else if (kernelType == ge::DT_FLOAT) {
        tilingKey = 92;
    } else {
        tilingKey = 94;
    }
}

void FlashAttentionScoreEmptyInputTiling::FlashAttentionScoreSetEmptyInputTilingData(gert::TilingContext *context,
    FlashAttentionScoreTilingData &tilingData)
{
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

ge::graphStatus CheckParams(const gert::TilingContext *context)
{
    if (context->GetInputShape(QUERY_INPUT_INDEX) != nullptr && context->GetInputShape(KEY_INPUT_INDEX) != nullptr &&
        context->GetInputShape(VALUE_INPUT_INDEX) != nullptr && context->GetAttrs() != nullptr) {
        auto &queryShape = context->GetInputShape(QUERY_INPUT_INDEX)->GetStorageShape();
        auto &keyShape = context->GetInputShape(KEY_INPUT_INDEX)->GetStorageShape();
        auto &valueShape = context->GetInputShape(VALUE_INPUT_INDEX)->GetStorageShape();
        const char *inputLayout = context->GetAttrs()->GetAttrPointer<char>(INPUTLAYOUT_ATTRS_INDEX);
        if (strlen(inputLayout) == 3) { // 3: BSH or SBH
            OP_TILING_CHECK((keyShape != valueShape),
                VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "key or value shape is invalid"),
                return ge::GRAPH_FAILED);
            if (inputLayout[0] == 'B') {
                // layout is BSH
                OP_TILING_CHECK((queryShape.GetDim(0) != keyShape.GetDim(0)),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "query or key shape is invalid"),
                    return ge::GRAPH_FAILED);
            } else {
                // layout is SBH
                OP_TILING_CHECK((queryShape.GetDim(1) != keyShape.GetDim(1)),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "query or key shape is invalid"),
                    return ge::GRAPH_FAILED);
            }
        } else if (strlen(inputLayout) == 4) { // 4: layout is BNSD or BSND
            OP_TILING_CHECK((keyShape != valueShape),
                VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "key or value shape is invalid"),
                return ge::GRAPH_FAILED);
            OP_TILING_CHECK((queryShape.GetDim(0) != keyShape.GetDim(0)),
                VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "query or key shape is invalid"),
                return ge::GRAPH_FAILED);
            OP_TILING_CHECK((queryShape.GetDim(3) != keyShape.GetDim(3)),
                VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "query or key shape is invalid"),
                return ge::GRAPH_FAILED);
        } else {
            OP_LOGW(context->GetNodeName(), "invalid input_layout[%s].", inputLayout);
            return ge::GRAPH_FAILED;
        }
        return ge::SUCCESS;
    }
    OP_LOGW(context->GetNodeName(), "fail to get shape or attr from context");
    return ge::GRAPH_FAILED;
}

bool IsEmptyInput(gert::TilingContext *context)
{
    uint64_t attentionOutShapeSize =
        (uint64_t)context->GetOutputShape(ATTENTIONOUT_OUPUT_INDEX)->GetStorageShape().GetShapeSize();
    uint64_t queryShapeSize = (uint64_t)context->GetInputShape(QUERY_INPUT_INDEX)->GetStorageShape().GetShapeSize();
    uint64_t keyShapeSize = (uint64_t)context->GetInputShape(KEY_INPUT_INDEX)->GetStorageShape().GetShapeSize();
    uint64_t softmaxSumShapeSize =
        (uint64_t)context->GetOutputShape(SOFTMAXSUM_OUPUT_INDEX)->GetStorageShape().GetShapeSize();
    if ((queryShapeSize == 0 || keyShapeSize == 0) && (attentionOutShapeSize != 0 || softmaxSumShapeSize != 0)) {
        /* 以 MIN_COPY_UINT_SIZE 为 32Byte说明, blocks为数据的块数, blocks与coreNum存在三种关系:
          (1) blocks % coreNum == 0
             主核数量为coreNum,主核处理块数为blocks / coreNum, 最后一个核处理非32Byte对齐的数据, 尾核数量为0
          (2) blocks % coreNum != 0
              (2.1) blocks < coreNum
                    主核数量为blocks,主核处理块数为1, 最后一个核处理非32Byte对齐的数据, 尾核数量为0
              (2.2) blocks > coreNum
                    主核数量为blocks % coreNum, 尾核数量为coreNum - (blocks % coreNum), 尾核处理块数为blocks / coreNum
                    主核处理块数为blocks / coreNum + 1,最后一个尾核处理非对齐场景
        (2.2)情况如下:
        |-------------主核块-----------------|------------尾核块-----------|非对齐块|
        |                                   |                             |       |
        |                                   |                             |       |
        |--------n*(blocks/coreNum+1)-------|-----m*(blocks/coreNum)------|<32Byte|
        */
        auto kernelType = context->GetInputDesc(KEY_INPUT_INDEX)->GetDataType();
        FlashAttentionScoreEmptyInputTiling emptyInputTiling;
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
        OP_TILING_CHECK((coreNum <= 0),
            VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "platform info is invalid, coreNum=%u", coreNum),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK((kernelType != ge::DT_FLOAT16 && kernelType != ge::DT_FLOAT && kernelType != ge::DT_BF16),
            VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "kernelType is invalid, kernelType is %u",
            kernelType),
            return ge::GRAPH_FAILED);
        uint32_t attentionOutFormerNum;          // attentionOut的主核
        uint32_t attentionOutTailNum;            // attentionOut的尾核
        uint32_t softmaxMaxFormerNum;            // softmaxMax 和 softmaxSum的主核
        uint32_t softmaxMaxTailNum;              // softmaxMax 和 softmaxSum的尾核
        uint64_t attentionOutSingleCoreDataSize; // attentionOut的每个主核处理的数据个数
        uint64_t attentionOutTailCoreDataSize;   // attentionOut的每个尾核处理的数据个数
        uint64_t softmaxMaxSingleCoreDataSize;
        uint64_t softmaxMaxTailCoreDataSize;
        uint64_t attentionOutLastCoreDataSize = 0; // 最后一个核应该处理的数据量
        uint64_t attentionOutLastCoreIndex = 0;    // 最后一个核起始地址
        uint32_t tilingKey = 0;
        uint64_t attentionOutBlockSize = 0;
        uint64_t softmaxSumBlockSize = 0;

        // 计算 MIN_COPY_UINT_SIZE 块数
        attentionOutBlockSize = Ceil(attentionOutShapeSize * ge::GetSizeByDataType(kernelType), MIN_COPY_UINT_SIZE);
        // softmaxSum 和 softmaxMax 输出为 fp32
        softmaxSumBlockSize = Ceil(softmaxSumShapeSize * ge::GetSizeByDataType(ge::DT_FLOAT), MIN_COPY_UINT_SIZE);

        if (attentionOutShapeSize != 0) {
            if (attentionOutBlockSize % coreNum == 0) {
                attentionOutTailCoreDataSize = 0;
                attentionOutFormerNum = coreNum;
                attentionOutTailNum = 0;
                attentionOutSingleCoreDataSize =
                    attentionOutBlockSize / coreNum * MIN_COPY_UINT_SIZE / ge::GetSizeByDataType(kernelType);
                attentionOutLastCoreDataSize = attentionOutSingleCoreDataSize -
                    (attentionOutBlockSize * MIN_COPY_UINT_SIZE / ge::GetSizeByDataType(kernelType) -
                    attentionOutShapeSize);
                attentionOutLastCoreIndex = (attentionOutFormerNum - 1) * attentionOutSingleCoreDataSize;
            } else {
                attentionOutTailCoreDataSize =
                    attentionOutBlockSize / coreNum * MIN_COPY_UINT_SIZE / ge::GetSizeByDataType(kernelType);
                attentionOutSingleCoreDataSize =
                    attentionOutTailCoreDataSize + MIN_COPY_UINT_SIZE / ge::GetSizeByDataType(kernelType);
                if (attentionOutBlockSize > coreNum) {
                    attentionOutFormerNum = attentionOutBlockSize % coreNum;
                    attentionOutTailNum = coreNum - attentionOutFormerNum;
                    attentionOutLastCoreIndex = attentionOutFormerNum * attentionOutSingleCoreDataSize +
                        (attentionOutTailNum - 1) * attentionOutTailCoreDataSize;
                    attentionOutLastCoreDataSize =
                        attentionOutTailCoreDataSize - (attentionOutSingleCoreDataSize * attentionOutFormerNum +
                        attentionOutTailCoreDataSize * attentionOutTailNum - attentionOutShapeSize);
                } else {
                    attentionOutFormerNum = attentionOutBlockSize;
                    attentionOutTailNum = 0;
                    attentionOutLastCoreIndex = (attentionOutFormerNum - 1) * attentionOutSingleCoreDataSize;
                    attentionOutLastCoreDataSize = attentionOutSingleCoreDataSize -
                        (attentionOutFormerNum * attentionOutSingleCoreDataSize - attentionOutShapeSize);
                }
            }
        } else {
            attentionOutFormerNum = 0;
            attentionOutTailNum = 0;
            attentionOutSingleCoreDataSize = 0;
            attentionOutTailCoreDataSize = 0;
            attentionOutLastCoreDataSize = 0;
            attentionOutLastCoreIndex = 0;
        }

        if (softmaxSumBlockSize % coreNum == 0) {
            softmaxMaxSingleCoreDataSize =
                softmaxSumBlockSize / coreNum * MIN_COPY_UINT_SIZE / ge::GetSizeByDataType(ge::DT_FLOAT);
            softmaxMaxTailCoreDataSize = 0;
            softmaxMaxFormerNum = coreNum;
            softmaxMaxTailNum = 0;
        } else {
            if (softmaxSumBlockSize > coreNum) {
                softmaxMaxFormerNum = softmaxSumBlockSize % coreNum;
                softmaxMaxTailNum = coreNum - softmaxMaxFormerNum;
            } else {
                softmaxMaxFormerNum = softmaxSumBlockSize;
                softmaxMaxTailNum = 0;
            }
            softmaxMaxTailCoreDataSize =
                softmaxSumBlockSize / coreNum * MIN_COPY_UINT_SIZE / ge::GetSizeByDataType(ge::DT_FLOAT);
            softmaxMaxSingleCoreDataSize =
                softmaxMaxTailCoreDataSize + MIN_COPY_UINT_SIZE / ge::GetSizeByDataType(ge::DT_FLOAT);
        }

        emptyInputTiling.tilingData.emptyInputTilingData.set_coreNum(coreNum);
        emptyInputTiling.tilingData.emptyInputTilingData.set_attentionOutFormerNum(attentionOutFormerNum);
        emptyInputTiling.tilingData.emptyInputTilingData.set_attentionOutTailNum(attentionOutTailNum);
        emptyInputTiling.tilingData.emptyInputTilingData.set_softmaxMaxFormerNum(softmaxMaxFormerNum);
        emptyInputTiling.tilingData.emptyInputTilingData.set_softmaxMaxTailNum(softmaxMaxTailNum);
        emptyInputTiling.tilingData.emptyInputTilingData.set_attentionOutSingleCoreDataSize(
            attentionOutSingleCoreDataSize);
        emptyInputTiling.tilingData.emptyInputTilingData.set_attentionOutTailCoreDataSize(attentionOutTailCoreDataSize);
        emptyInputTiling.tilingData.emptyInputTilingData.set_softmaxMaxSingleCoreDataSize(softmaxMaxSingleCoreDataSize);
        emptyInputTiling.tilingData.emptyInputTilingData.set_softmaxMaxTailCoreDataSize(softmaxMaxTailCoreDataSize);
        emptyInputTiling.tilingData.emptyInputTilingData.set_attentionOutLastCoreDataSize(attentionOutLastCoreDataSize);
        emptyInputTiling.tilingData.emptyInputTilingData.set_attentionOutLastCoreIndex(attentionOutLastCoreIndex);
        emptyInputTiling.FlashAttentionScoreSetEmptyInputTilingData(context, emptyInputTiling.tilingData);
        emptyInputTiling.GetTilingKeyAttentionScore4EmptyInput(tilingKey, context);
        context->SetTilingKey(tilingKey);
        auto aivNum = ascendcPlatform.GetCoreNumAiv();
        uint32_t aivActualNum =
            std::max((attentionOutFormerNum + attentionOutTailNum), (softmaxMaxFormerNum + softmaxMaxTailNum));
        context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aivActualNum, 0, aivNum));
        size_t *workspaces = context->GetWorkspaceSizes(1);
        // workspace上预留100M
        workspaces[0] = 100 * 1024 * 1024;

        return true;
    }
    return false;
}

ge::graphStatus TilingFlashAttentionScore(gert::TilingContext *context)
{
    if (CheckParams(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (IsEmptyInput(context)) {
        return ge::GRAPH_SUCCESS;
    } else {
        FlashAttentionScoreTiling flash_tiling;
        auto resultCode = flash_tiling.RunBigKernelTiling(context);
        if (resultCode == ge::GRAPH_PARAM_INVALID) {
            resultCode = TilingRegistry::GetInstance().DoTilingImpl(context);
        }
        return resultCode;
    }
}

ge::graphStatus TilingPrepareForFlashAttentionScore(gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(FlashAttentionScore)
    .Tiling(TilingFlashAttentionScore)
    .TilingParse<FlashAttentionScoreCompileInfo>(TilingPrepareForFlashAttentionScore); // 向框架注册入口函数
} // namespace optiling
