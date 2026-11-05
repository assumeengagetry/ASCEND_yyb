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

#include <map>
#include <sstream>
#include <algorithm>
#include <numeric>

#include "base/tiling_base.h"
#include "base/tiling_templates_registry.h"
#include "base/tiling_type.h"
#include "cube_tiling_runtime.h"
#include "data_copy_transpose_tiling.h"
#include "error_log.h"
#include "exe_graph/runtime/tiling_context.h"
#include "external/exe_graph/runtime/shape.h"
#include "flash_attention_score_tiling.h"
#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "op_tiling_util.h"
#include "tiling/tiling_api.h"

namespace optiling {
namespace FA {
const uint32_t BYTE_BLOCK = 32;
const int64_t FRACTAL_NUM = 16L;

const int64_t BYTE_BIT_NUM = 8UL;
const size_t PSE_INPUT_INDEX = 3UL;
const size_t DROP_MASK_INPUT_INDEX = 4UL;
const size_t ATTENTION_MASK_INPUT_INDEX = 6UL;
const size_t PREFIX_INPUT_INDEX = 7UL;
const size_t ATTEN_OUT_INDEX = 3UL;
const size_t ATTENTION_MASK_DIM_NUM = 4UL;
const int64_t BMM_SOFTMAX_RATIO = 4L;
const size_t DATA_COPY_TRANPOSE_DIM_NUM = 4L;
const int64_t MAX_AIV_NUM = 48L;
const int64_t DROP_MASK_ALIGN_UNIT = 256L;  // input bits, and align to 32B in UB
const int64_t HIGH_PERF_BUFFER_NUM = 6L;
const int64_t HIGH_PERF_SUPPORT_D_LIMIT = 128L;
const int64_t HIGH_PERF_SUPPORT_S2_BASIC = 128L;
const int64_t HIGH_PERF_API_BUFFER_MULTIPLE = 2L;
const int64_t HIGH_PERF_WORKSPACE_MULTIPLE = 3L;

constexpr size_t WORK_SPACE_RESERVE_SIZE = 16 * 1024 * 1024;
const int64_t MAX_PREFIX_N_SIZE = 24L;

enum LayoutType : uint8_t {
    None = 0,
    LAYOUT_BSH = 1,
    LAYOUT_BSND = 1,
    LAYOUT_SBH = 2,
    LAYOUT_BNSD = 3,
};

enum AttenMaskShapeType : uint8_t {
    ATTEN_B_N2_G_S1_S2 = 0,
    ATTEN_B_1_1_S1_S2 = 1,
    ATTEN_1_1_1_S1_S2 = 2,
};

enum PseShapeType : uint8_t {
    PSE_B_N2_G_S1_S2 = 0,
    PSE_B_N2_G_1_S2 = 1,
};

enum SparseMode : uint8_t  {
    NO_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
    PREFIX,
    GLOBAL,
    DILATED,
    BLOCK_LOCAL
};

enum AttenMaskCompressMode : uint8_t  {
    NO_COMPRESS_MODE = 0,
    LEFT_UP_CAUSAL_MODE,
    RIGHT_DOWN_CAUSAL_MODE
};

template <typename T>
static T AlignUp(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

template <typename T>
static T AlignDown(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    return num1 / num2 * num2;
}

template <typename T>
static T CeilDivision(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

template <typename T>
static T CalcTailSize(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    T mod = num1 % num2;
    return mod != 0 ? mod : num2;
}

inline std::vector<uint32_t> GetLastAxisShapeND(const ge::Shape &srcShape)
{
    std::vector<uint32_t> ret;
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    uint32_t calculateSize = 1;
    for (uint32_t i = 0; i < shapeDims.size(); i++) {
        calculateSize *= shapeDims[i];
    }

    uint32_t srcK = shapeDims[shapeDims.size() - 1];
    uint32_t srcM = calculateSize / srcK;
    ret = {srcM, srcK};
    return ret;
}

uint32_t CustomSoftmaxHelper(const ge::Shape srcShape, const uint32_t dataTypeSize, const uint32_t splitBufferCount,
                             const uint32_t reduceBufferCount)
{
    // convert srcShape to 2-dims (m,k)
    std::vector<uint32_t> retVec = GetLastAxisShapeND(srcShape);
    if (retVec.size() <= 1 || dataTypeSize == 0) {
        return 0;
    }
    uint32_t srcK = retVec[1];
    uint32_t elementNumPerBlk = 32 / dataTypeSize;
    uint32_t needSize = elementNumPerBlk * reduceBufferCount + srcK * splitBufferCount;
    return needSize * sizeof(float);
}

uint32_t GetMaxSumMinTmpSize(const ge::Shape srcShape, const uint32_t dataTypeSize)
{
    static uint32_t splitBufferCount = 2;
    static uint32_t reduceBufferCount = 1;

    if (dataTypeSize == sizeof(float)) {
        splitBufferCount = 1;
    } else {
        splitBufferCount = 2;  // 高精度模式时，splitBuffer数量为2
    }
    return CustomSoftmaxHelper(srcShape, dataTypeSize, splitBufferCount, reduceBufferCount);
}

uint32_t GetMaxSumUpdateMinTmpSize(const ge::Shape srcShape, const uint32_t dataTypeSize)
{
    static uint32_t splitBufferCount = 2;
    static uint32_t reduceBufferCount = 3;

    if (dataTypeSize == sizeof(float)) {
        splitBufferCount = 1;
    } else {
        splitBufferCount = 2;  // 高精度模式时，splitBuffer数量为2
    }
    return CustomSoftmaxHelper(srcShape, dataTypeSize, splitBufferCount, reduceBufferCount);
}

uint32_t GetCustomSoftmaxMinTmpSize(const ge::Shape srcShape, const uint32_t dataTypeSize)
{
    static uint32_t splitBufferCount = 2;
    static uint32_t reduceBufferCount = 0;
    return CustomSoftmaxHelper(srcShape, dataTypeSize, splitBufferCount, reduceBufferCount);
}

bool MaxSumTilingFuncHelper(const ge::Shape srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize,
                            const uint32_t splitBufferCount, const uint32_t reduceBufferCount, MaxSumTiling &tiling)
{
    std::vector<uint32_t> retVec = GetLastAxisShapeND(srcShape);
    if (retVec.size() <= 1 || dataTypeSize == 0) {
        return false;
    }
    uint32_t elementNumPerBlk = BYTE_BLOCK / dataTypeSize;
    uint32_t workLocalSize = localWorkSpaceSize / sizeof(float);
    uint32_t srcK = retVec[1];
    uint32_t srcM = retVec[0];
    uint32_t baseM = std::min(workLocalSize / (elementNumPerBlk * reduceBufferCount + srcK * splitBufferCount), srcM);
    constexpr uint32_t fp32ElementNumPerBlk = BYTE_BLOCK / sizeof(float);
    if (baseM > fp32ElementNumPerBlk && srcM % fp32ElementNumPerBlk == 0) {
        baseM = baseM / fp32ElementNumPerBlk * fp32ElementNumPerBlk;
        while (srcM % baseM != 0) {
            baseM -= fp32ElementNumPerBlk;
        }
    }

    tiling.set_srcM(srcM);
    tiling.set_srcK(srcK);
    tiling.set_srcSize(srcM * srcK);

    tiling.set_outMaxM(srcM);
    tiling.set_outMaxK(elementNumPerBlk);
    tiling.set_outMaxSize(srcM * elementNumPerBlk);

    tiling.set_splitM(baseM);
    tiling.set_splitK(srcK);
    tiling.set_splitSize(baseM * srcK);

    tiling.set_reduceM(baseM);
    tiling.set_reduceK(elementNumPerBlk);
    tiling.set_reduceSize(baseM * elementNumPerBlk);

    uint32_t range = srcM / baseM;
    uint32_t tail = srcM % baseM;
    tiling.set_rangeM(range);
    tiling.set_tailM(tail);

    tiling.set_tailSplitSize(tail * srcK);
    tiling.set_tailReduceSize(tail * elementNumPerBlk);

    return true;
}

bool MaxSumTilingFunc(const ge::Shape srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize,
                      MaxSumTiling &tiling)
{
    static uint32_t splitBufferCount = 2;
    static uint32_t reduceBufferCount = 1;

    // 数据类型大小为4时，表示高性能模式
    if (dataTypeSize == 4) {
        splitBufferCount = 1;
    } else {
        splitBufferCount = 2;  // 高精度模式时，splitBuffer数量为2
    }
    return MaxSumTilingFuncHelper(srcShape, sizeof(float), localWorkSpaceSize, splitBufferCount, reduceBufferCount,
                                  tiling);
}

bool MaxSumUpdateTilingFunc(const ge::Shape srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize,
                            MaxSumTiling &tiling)
{
    static uint32_t splitBufferCount = 2;
    static uint32_t reduceBufferCount = 3;

    // 高性能数据类型大小为4
    if (dataTypeSize == 4) {
        splitBufferCount = 1;
    } else {
        splitBufferCount = 2;  // 高精度模式时，splitBuffer数量为2
    }
    return MaxSumTilingFuncHelper(srcShape, sizeof(float), localWorkSpaceSize, splitBufferCount, reduceBufferCount,
                                  tiling);
}

bool CustomSoftmaxTilingFunc(const ge::Shape srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize,
                             MaxSumTiling &tiling)
{
    static uint32_t splitBufferCount = 2;
    static uint32_t reduceBufferCount = 0;
    return MaxSumTilingFuncHelper(srcShape, sizeof(float), localWorkSpaceSize, splitBufferCount, reduceBufferCount,
                                  tiling);
}

class TilingKey {
public:
    TilingKey() : splitS1(0), splitS2(0), splitD(0), dtype(0), layoutType(0), sparseType(0), reserved(0) {}

    void Reset()
    {
        splitS1 = 0;
        splitS2 = 0;
        splitD = 0;
        dtype = 0;
        layoutType = 0;
        sparseType = 0;
        reserved = 0;
    }

    uint32_t GetRawTilingKey() const { return *(reinterpret_cast<const uint32_t *>(this)); }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << " splitS1: " << splitS1 << " splitS2: " << splitS2 << " splitD: " << splitD;
        return ss.str();
    }

    uint32_t splitS1 : 1;
    uint32_t splitS2 : 1;
    uint32_t splitD : 1;
    uint32_t dtype : 2;
    uint32_t layoutType : 2;
    uint32_t sparseType : 2;
    uint32_t reserved : 23;  // to fullfil 32 bit, if add new template bit then decrease this number
};

inline bool operator==(const TilingKey &left, const TilingKey &right)
{
    return left.GetRawTilingKey() == right.GetRawTilingKey();
}

using TemplateType = TilingKey;

class BufferNum {
public:
    // sum and max always use fp32, shape is (S1, 1), inner axis align 32B.
    size_t bufferS1S2Num;  // unit: input dtype
    size_t bufferS1DNum;
    size_t bufferExpNum;  // unit: input dtype, shape: [S1, 1], inner axis align 32B.
};

class FlashAttentionScoreTilingBase : public TilingBaseClass {
public:
    explicit FlashAttentionScoreTilingBase(gert::TilingContext *context) : TilingBaseClass(context) { Reset(); }
    ~FlashAttentionScoreTilingBase() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override { return true; }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    virtual void GetSparseInfo();
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override = 0;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

    virtual void GetBufferNum(BufferNum &bufferNum) const = 0;

    void Reset();

    virtual int64_t GetNRatio();

    virtual int64_t GetMinS1BasicBlock() const { return std::min(64L, alignedS1); }

    virtual bool IsTemplateMatched() const { return expectTemplate == actualTemplate; }

    ge::graphStatus CheckContext();
    virtual bool AnalyzeDtype();
    bool AnalyzeAttrs();
    bool AnalyzeLayout();
    bool Analyze3DimLayout(const gert::Shape &queryShape, const gert::Shape &keyShape, size_t layoutLen);
    bool Analyze4DimLayout(const gert::Shape &queryShape, const gert::Shape &keyShape, size_t layoutLen);
    bool AnalyzeOptionalInput();
    bool MatchTemplate();
    virtual void CalcS1S2BasicBlock(const BufferNum &bufferNum);
    virtual void CalcDBasicBlock(const BufferNum &bufferNum);
    int64_t CalcMaxS1BasicBlockSize(int64_t actualD, const BufferNum &bufferNum) const;
    int64_t CalcMaxS2BasicBlockSize(const BufferNum &bufferNum, int64_t tmpS1BasicBlock) const;
    virtual bool CalcUBSize(const BufferNum &bufferNum, int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                            int64_t tmpDBasicBlock, int64_t batch = 1);
    virtual int64_t CalcMinApiTmpSize(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t batch);
    bool IsBasicBlockInSoftMax(const ge::Shape &shape) const;
    virtual bool SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock,
                                    int64_t batch, matmul_tiling::MatmulApiTiling &bmm1);
    virtual bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock,
                                    int64_t batch, matmul_tiling::MatmulApiTiling &bmm2) = 0;
    bool SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch = 1);
    virtual void SetCoreParams();
    virtual void SetMultiBatchCoreParams();
    void SetMultiCoreParams();
    virtual void SetSoftMaxTiling();
    void SetDataCopyTransposeTiling();
    virtual void SetTensorSizeParams();
    virtual void SetSparseParams();
    bool InitSparseValidArray(std::vector<int64_t> &sparseValidArray, int64_t bIdx);
    bool SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray,
                           MultiCoreParams &multiCoreParams);
    void SetPrefixSparseStartIdx(const std::vector<std::vector<int64_t>> &sparseValidArray,
                                 MultiCoreParams &multiCoreParams);
    void PrintSparseMaxMinLoadPerCore(const std::vector<int64_t> &sparseValidArray, int64_t *sparseStartIdx,
                                      int32_t validAivNum, int64_t avgLoadSize);
    bool PartitionSparseData(const std::vector<int64_t> &sparseRollingArray,
                             int64_t sparseRollingArraySum,
                             int64_t sparseArraySize,
                             int64_t loadMaxEachCore,
                             std::vector<int64_t> &partitionResult);
    SparseEnum GetPrefixNList();

    uint32_t aivNum;
    int64_t apiMaxUBSize = 0;

    matmul_tiling::DataType bmmDtype = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::DataType bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::DataType bmm2OutDtype = matmul_tiling::DataType::DT_FLOAT;

    ge::DataType inputDtype;
    int64_t inputDtypeBytes;
    int64_t calcTypeSize;

    bool isHighPercision;  // fp16 high percision mode

    DtypeEnum tilingKeyDType;
    LayoutEnum tilingKeyLayout;

    int64_t bSize;
    int64_t gSize;
    int64_t dSize;
    int64_t n1Size;
    int64_t n2Size;
    int64_t s1Size;
    int64_t s2Size;
    int64_t s1StrideSize;  // query Shape S inner axes, for bmm1
    int64_t s2StrideSize;  // key Shape S inner axes, for bmm1
    int64_t preTokens;
    int64_t nextTokens;
    int64_t sparseMode;
    float keepProb;
    float scaleValue;
    uint32_t attenMaskCompressMode;
    uint8_t pseExistFlag;
    uint8_t attenMaskExistFlag;
    uint8_t dropMaskExistFlag;

    int64_t alignedS1;
    int64_t alignedS2;
    int64_t alignedD;

    int64_t s1BasicBlock;
    int64_t s2BasicBlock;
    int64_t dBasicBlock;
    int64_t batchBasic;
    int64_t nRatio;

    int64_t minUsedUBSize;

    const char *templateName = "base";
    const char *opName;
    const char *inputLayout;
    TemplateType expectTemplate;
    TemplateType actualTemplate;

    FlashAttentionScoreGeneralTilingData tilingData;
};

int64_t FlashAttentionScoreTilingBase::GetNRatio()
{
    return BMM_SOFTMAX_RATIO;
}

ge::graphStatus FlashAttentionScoreTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opName, "fail to get platform info"),
                    return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    aivNum = ascendcPlatform.GetCoreNumAiv();
    aicoreParams_.blockDim = 0;
    uint64_t ubSizePlatForm;
    uint64_t l1SizePlatForm;
    uint64_t l0CSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1SizePlatForm);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSizePlatForm);
    aicoreParams_.ubSize = ubSizePlatForm;
    aicoreParams_.l1Size = l1SizePlatForm;
    aicoreParams_.l0cSize = l0CSizePlatForm;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreTilingBase::CheckContext()
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    size_t idx = 0;
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx++);
    auto keepProbPtr = attrs->GetAttrPointer<float>(idx++);
    auto preTokensPtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto nextTokensPtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto n1SizePtr = attrs->GetAttrPointer<uint32_t>(idx++);
    auto inputLayoutPtr = attrs->GetAttrPointer<char>(idx++);

    OPS_CHECK_NULL_WITH_CONTEXT(context_, scaleValuePtr);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, keepProbPtr);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, preTokensPtr);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, nextTokensPtr);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, n1SizePtr);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputLayoutPtr);

    auto queryShape = context_->GetInputShape(0);
    auto queryDesc = context_->GetInputDesc(0);
    auto keyShape = context_->GetInputShape(1);
    auto attenOutShape = context_->GetOutputShape(ATTEN_OUT_INDEX);

    OPS_CHECK_NULL_WITH_CONTEXT(context_, queryShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, queryDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, keyShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attenOutShape);

    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
    OP_TILING_CHECK(
        context_->GetRawTilingData()->GetCapacity() < tilingData.GetDataSize(),
        VECTOR_INNER_ERR_REPORT_TILIING(opName, "context tiling data capacity %zu < actual tiling data size %zu",
                                        context_->GetRawTilingData()->GetCapacity(), tilingData.GetDataSize()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreTilingBase::GetSparseInfo() {
    auto &inputParams = tilingData.inputParams;

    SparseEnum sparseType = SparseEnum::ALL;
    OP_LOGD(opName, "check sparse feature: preTokens[%ld], nextTokens[%ld], s1[%ld], s2[%ld], attenMaskExistFlag[%d]",
            preTokens, nextTokens, s1Size, s2Size, attenMaskExistFlag);
    if (attenMaskExistFlag == 1 && s1Size == s2Size) {
        // 2: if nextTokens is 0 and preToken bigger than s1, or no algin to 64 sparse impl it as causal
        if ((preTokens >= s1Size || (preTokens % 64L) != 0) && nextTokens == 0) {
            sparseType = SparseEnum::CAUSAL;
            preTokens = s1Size;  // if sparse type is causal, template always need preTokens equal to s1Size
        } else if (preTokens > 0 && preTokens < s1Size - 1 && nextTokens > 0 && nextTokens < s2Size - 1) {
            sparseType = SparseEnum::BAND;
        } else if (preTokens > 0 && preTokens < s1Size - 1 && nextTokens == 0) {
            sparseType = SparseEnum::BANDX1;
        }

        if (sparseMode == static_cast<int64_t>(SparseEnum::CAUSAL)) {
            sparseType = SparseEnum::CAUSAL;
        } else if (sparseMode == static_cast<int64_t>(SparseEnum::PREFIX)) {
            sparseType = GetPrefixNList();
        }
    }

    inputParams.set_attenMaskCompressMode(attenMaskCompressMode);
    inputParams.set_sparseType(static_cast<uint8_t>(sparseType));
    inputParams.set_preTokens(preTokens);
    inputParams.set_nextTokens(nextTokens);
}

ge::graphStatus FlashAttentionScoreTilingBase::GetShapeAttrsInfo()
{
    opName = context_->GetNodeName();
    OP_LOG_FULL(DLOG_DEBUG, opName, "TilingContext: %s", optiling::DebugTilingContext(context_).c_str());
    OP_TILING_CHECK(CheckContext() != ge::GRAPH_SUCCESS, VECTOR_INNER_ERR_REPORT_TILIING(opName, "invalid context"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(!AnalyzeAttrs() || !AnalyzeDtype() || !AnalyzeLayout() || !AnalyzeOptionalInput(),
                    VECTOR_INNER_ERR_REPORT_TILIING(opName, "fail to analyze context info"), return ge::GRAPH_FAILED);

    alignedS1 = AlignUp(s1Size, FRACTAL_NUM);
    alignedS2 = AlignUp(s2Size, FRACTAL_NUM);
    alignedD = AlignUp(dSize, FRACTAL_NUM);

    auto &inputParams = tilingData.inputParams;
    inputParams.set_bSize(bSize);
    inputParams.set_n2Size(n2Size);
    inputParams.set_gSize(gSize);
    inputParams.set_s1Size(s1Size);
    inputParams.set_s2Size(s2Size);
    inputParams.set_dSize(dSize);
    inputParams.set_keepProb(keepProb);
    inputParams.set_scaleValue(scaleValue);

    OP_LOGD(opName, "input params: bn2gs1s2d[%ld, %ld, %ld, %ld, %ld, %ld], keepProb[%f], scaleValue[%f]", bSize,
            n2Size, gSize, s1Size, s2Size, dSize, keepProb, scaleValue);

    GetSparseInfo();

    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreTilingBase::Reset()
{
    tilingData.SetDataPtr(context_->GetRawTilingData()->GetData());
    apiMaxUBSize = 0;

    bmmDtype = matmul_tiling::DataType::DT_FLOAT;
    bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
    bmm2OutDtype = matmul_tiling::DataType::DT_FLOAT;

    inputDtype = ge::DT_FLOAT16;
    inputDtypeBytes = ge::GetSizeByDataType(inputDtype);
    calcTypeSize = inputDtypeBytes;

    tilingKeyDType = DtypeEnum::FLOAT16;
    tilingKeyLayout = LayoutEnum::BNSD;

    bSize = 0;
    gSize = 0;
    dSize = 0;
    n1Size = 0;
    n2Size = 0;
    s1Size = 0;
    s2Size = 0;
    s1StrideSize = 0;
    s2StrideSize = 0;
    preTokens = std::numeric_limits<int32_t>::max();
    nextTokens = std::numeric_limits<int32_t>::max();
    sparseMode = static_cast<int64_t>(NO_MASK);
    keepProb = 1.0f;
    scaleValue = 1.0f;
    pseExistFlag = 0;
    attenMaskCompressMode = static_cast<uint32_t>(NO_COMPRESS_MODE);
    attenMaskExistFlag = 0;
    dropMaskExistFlag = 0;
    isHighPercision = false;

    alignedS1 = 0;
    alignedS2 = 0;
    alignedD = 0;

    s1BasicBlock = std::numeric_limits<int64_t>::max();
    s2BasicBlock = std::numeric_limits<int64_t>::max();
    dBasicBlock = std::numeric_limits<int64_t>::max();
    nRatio = GetNRatio();

    minUsedUBSize = 0;
    batchBasic = 1;

    opName = nullptr;
    inputLayout = nullptr;

    actualTemplate.Reset();
}

bool FlashAttentionScoreTilingBase::AnalyzeDtype()
{
    inputDtype = context_->GetInputDesc(0)->GetDataType();
    inputDtypeBytes = ge::GetSizeByDataType(inputDtype);
    calcTypeSize = ge::GetSizeByDataType(inputDtype);
    switch (inputDtype) {
        case ge::DT_FLOAT16:
            bmmDtype = matmul_tiling::DataType::DT_FLOAT16;
            bmm1OutDtype = isHighPercision ? matmul_tiling::DataType::DT_FLOAT : matmul_tiling::DataType::DT_FLOAT16;
            tilingKeyDType = isHighPercision ? DtypeEnum::FLOAT16_PRECISION : DtypeEnum::FLOAT16;
            calcTypeSize = isHighPercision ? ge::GetSizeByDataType(ge::DT_FLOAT) : ge::GetSizeByDataType(inputDtype);
            break;
        case ge::DT_FLOAT:
            bmmDtype = matmul_tiling::DataType::DT_FLOAT;
            bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
            tilingKeyDType = DtypeEnum::FLOAT32;
            isHighPercision = false;
            break;
        case ge::DT_BF16:
            bmmDtype = matmul_tiling::DataType::DT_BF16;
            bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
            tilingKeyDType = DtypeEnum::BFLOAT16;
            calcTypeSize = ge::GetSizeByDataType(ge::DT_FLOAT);
            isHighPercision = false;
            break;
        default:
            VECTOR_INNER_ERR_REPORT_TILIING(opName, "not support input dtype: %s for now",
                                            ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str());
            return false;
    }

    bmm2OutDtype = bmm1OutDtype;
    OP_LOGD(opName, "get high precision flag: %d", isHighPercision);
    return true;
}

bool FlashAttentionScoreTilingBase::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    size_t idx = 0;
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx++);
    auto keepProbPtr = attrs->GetAttrPointer<float>(idx++);
    auto preTokensPtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto nextTokensPtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto n1SizePtr = attrs->GetAttrPointer<uint32_t>(idx++);
    inputLayout = attrs->GetAttrPointer<char>(idx++);

    preTokens = *preTokensPtr;
    nextTokens = *nextTokensPtr;
    keepProb = *keepProbPtr;
    scaleValue = *scaleValuePtr;
    n1Size = *n1SizePtr;
    OP_TILING_CHECK(n1Size == 0, VECTOR_INNER_ERR_REPORT_TILIING(opName, "head num is zero"), return false);

    if (attrs->GetAttrNum() > idx) {
        auto implModePtr = attrs->GetAttrPointer<uint32_t>(idx++);
        isHighPercision = (*implModePtr == 0);
    }

    if (attrs->GetAttrNum() > idx) {
        auto sparseModePtr = attrs->GetAttrPointer<int64_t>(idx++);
        sparseMode = *sparseModePtr;
        if (sparseMode == LEFT_UP_CAUSAL) {
            attenMaskCompressMode = static_cast<uint32_t>(LEFT_UP_CAUSAL_MODE);
        } else if (sparseMode == RIGHT_DOWN_CAUSAL) {
            attenMaskCompressMode = static_cast<uint32_t>(RIGHT_DOWN_CAUSAL_MODE);
        }
        OP_LOGD(opName, "The current value of attenMaskCompressMode is %u.", attenMaskCompressMode);
    }
    return true;
}

bool FlashAttentionScoreTilingBase::AnalyzeLayout()
{
    auto &queryShape = context_->GetInputShape(0)->GetStorageShape();
    auto &keyShape = context_->GetInputShape(1)->GetStorageShape();

    size_t layoutLen = strlen(inputLayout);
    OP_LOGD(opName, "get input_layout [%s]", inputLayout);
    OP_TILING_CHECK(queryShape.GetDimNum() != layoutLen || keyShape.GetDimNum() != layoutLen,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName, "invalid layout[%s]", inputLayout), return false);
    OP_TILING_CHECK(
        !Analyze3DimLayout(queryShape, keyShape, layoutLen) || !Analyze4DimLayout(queryShape, keyShape, layoutLen),
        VECTOR_INNER_ERR_REPORT_TILIING(opName, "get unsupported layout: %s", inputLayout), return false);
    OP_TILING_CHECK(gSize == 0, VECTOR_INNER_ERR_REPORT_TILIING(opName, "get unsupported layout: %s", inputLayout),
                    return false);
    return true;
}

bool FlashAttentionScoreTilingBase::Analyze3DimLayout(const gert::Shape &queryShape, const gert::Shape &keyShape,
                                                      size_t layoutLen)
{
    int64_t h1 = 0;
    int64_t h2 = 0;
    if (layoutLen == 3UL) {
        if (inputLayout[0] == 'B' && inputLayout[1] == 'S' && inputLayout[2] == 'H') {  // 2: H idx
            bSize = queryShape.GetDim(0);
            s1Size = queryShape.GetDim(1);
            s2Size = keyShape.GetDim(1);
            h1 = queryShape.GetDim(2);  // 2: H idx
            h2 = keyShape.GetDim(2);    // 2: H idx
            s1StrideSize = h1;
            s2StrideSize = h2;
            tilingData.inputParams.set_layoutType(LAYOUT_BSH);
            tilingKeyLayout = LayoutEnum::BSND;
        } else if (inputLayout[0] == 'S' && inputLayout[1] == 'B' && inputLayout[2] == 'H') {  // 2: H idx
            s1Size = queryShape.GetDim(0);
            s2Size = keyShape.GetDim(0);
            bSize = queryShape.GetDim(1);
            h1 = queryShape.GetDim(2);  // 2: H idx
            h2 = keyShape.GetDim(2);    // 2: H idx
            s1StrideSize = h1 * bSize;
            s2StrideSize = h2 * bSize;
            tilingData.inputParams.set_layoutType(LAYOUT_SBH);
            tilingKeyLayout = LayoutEnum::SBND;
        }
        OP_TILING_CHECK(h1 == 0 || h2 == 0, VECTOR_INNER_ERR_REPORT_TILIING(opName, "H is zero"), return false);
        dSize = h1 / n1Size;
        gSize = h1 / h2;
        n2Size = h2 / dSize;
    }

    return true;
}

bool FlashAttentionScoreTilingBase::Analyze4DimLayout(const gert::Shape &queryShape, const gert::Shape &keyShape,
                                                      size_t layoutLen)
{
    if (layoutLen == 4UL) {
        // 2: N idx, 3: D idx
        if (inputLayout[0] == 'B' && inputLayout[1] == 'S' && inputLayout[2] == 'N' && inputLayout[3] == 'D') {
            bSize = queryShape.GetDim(0);
            s1Size = queryShape.GetDim(1);
            s2Size = keyShape.GetDim(1);
            n2Size = keyShape.GetDim(2);  // 2: N idx
            OP_TILING_CHECK(n2Size == 0, VECTOR_INNER_ERR_REPORT_TILIING(opName, "N2 is zero"), return false);
            gSize = queryShape.GetDim(2) / n2Size;  // 2: N idx
            dSize = queryShape.GetDim(3);           // 3: D idx
            s1StrideSize = gSize * n2Size * dSize;
            s2StrideSize = n2Size * dSize;
            tilingData.inputParams.set_layoutType(LAYOUT_BSND);
            tilingKeyLayout = LayoutEnum::BSND;
        } else if (inputLayout[0] == 'B' && inputLayout[1] == 'N' &&
                   // 2: S idx, 3: N idx
                   inputLayout[2] == 'S' && inputLayout[3] == 'D') {
            bSize = queryShape.GetDim(0);
            n2Size = keyShape.GetDim(1);  // 1: N idx
            OP_TILING_CHECK(n2Size == 0, VECTOR_INNER_ERR_REPORT_TILIING(opName, "N2 is zero"), return false);
            gSize = queryShape.GetDim(1) / n2Size;
            s1Size = queryShape.GetDim(2);  // 2: S idx
            s2Size = keyShape.GetDim(2);    // 2: S idx
            dSize = queryShape.GetDim(3);   // 3: D idx
            s1StrideSize = dSize;
            s2StrideSize = dSize;
            tilingData.inputParams.set_layoutType(LAYOUT_BNSD);
            tilingKeyLayout = LayoutEnum::BNSD;
        }
    }

    return true;
}

SparseEnum FlashAttentionScoreTilingBase::GetPrefixNList() {
    auto prefixN = context_->GetInputTensor(PREFIX_INPUT_INDEX);
    if (prefixN == nullptr) {
        OP_LOGW(opName, "[%s]prefixN is null pointer while sparse mode is prefix", templateName);
        return SparseEnum::ALL;
    }

    auto &prefixShape = prefixN->GetShape().GetStorageShape();
    if (prefixShape.GetDimNum() != 1 ||
        prefixShape.GetDim(0) != bSize ||
        prefixShape.GetDim(0) > MAX_PREFIX_N_SIZE) {
        OP_LOGW(opName, "[%s]prefixN shape is invalid", templateName);
        return SparseEnum::ALL;
    }
    /* Get Data from tensor. */
    auto data = prefixN->GetData<int64_t>();
    if (data == nullptr) {
        OP_LOGW(opName, "[%s]prefixN data is null pointer", templateName);
        return SparseEnum::ALL;
    }

    int64_t *prefixNList = tilingData.inputParams.get_prefixNList();
    memcpy_s(prefixNList, MAX_PREFIX_N_SIZE * sizeof(int64_t), data, bSize * sizeof(int64_t));
    return SparseEnum::PREFIX;
}

bool FlashAttentionScoreTilingBase::AnalyzeOptionalInput()
{
    // 0: (B,N2,G,S1,S2), 1: (B,N2,G,1,S2)
    PseShapeType pseShapeType = PSE_B_N2_G_1_S2;
    auto pseShape = context_->GetOptionalInputShape(PSE_INPUT_INDEX);
    if (pseShape != nullptr) {
        pseExistFlag = 1;
        if (pseShape->GetStorageShape().GetDimNum() > 1 &&
            pseShape->GetStorageShape().GetDim(pseShape->GetStorageShape().GetDimNum() - 2) > 1) {  // 2: pre last axiss
            pseShapeType = PSE_B_N2_G_S1_S2;
        }
    }
    tilingData.inputParams.set_pseShapeType(pseShapeType);

    auto attenMaskShape = context_->GetOptionalInputShape(ATTENTION_MASK_INPUT_INDEX);
    auto attenMaskInput = context_->GetOptionalInputDesc(ATTENTION_MASK_INPUT_INDEX);
    if (attenMaskShape != nullptr && attenMaskInput != nullptr) {
        attenMaskExistFlag = 1;
        auto attenMaskType = attenMaskInput->GetDataType();
        OP_TILING_CHECK(
            attenMaskType != ge::DT_BOOL && attenMaskType != ge::DT_UINT8,
            VECTOR_INNER_ERR_REPORT_TILIING(opName, "invalid attenMask dtype[%s], only support bool or uint8",
                                            ge::TypeUtils::DataTypeToSerialString(attenMaskType).c_str()),
            return false);

        tilingData.inputParams.set_attenMaskDataType(1);
        // 0: (B,N2,G,S1,S2), 1: (B,1,1,S1,S2), 2: (1,1,1,S1,S2)
        AttenMaskShapeType attenMaskShapeType = ATTEN_B_N2_G_S1_S2;
        auto &attenMaskStorageShape = attenMaskShape->GetStorageShape();
        if (attenMaskStorageShape.GetDimNum() == ATTENTION_MASK_DIM_NUM) {
            if (attenMaskStorageShape.GetDim(0) == 1 && bSize > 1) {
                attenMaskShapeType = ATTEN_1_1_1_S1_S2;
            } else if (attenMaskStorageShape.GetDim(1) == 1 && n1Size > 1) {
                attenMaskShapeType = ATTEN_B_1_1_S1_S2;
            }
        } else if ((attenMaskStorageShape.GetDimNum() > 1 &&
                   attenMaskStorageShape.GetDim(attenMaskStorageShape.GetDimNum() - 2) == s1Size &&  // 2 dim
                   attenMaskStorageShape.GetDim(attenMaskStorageShape.GetDimNum() - 1) == s2Size) ||
                   (attenMaskCompressMode != static_cast<uint32_t>(NO_COMPRESS_MODE))) {
            attenMaskShapeType = ATTEN_1_1_1_S1_S2;  // maybe [S1, S2]
        }

        tilingData.inputParams.set_attenMaskShapeType(attenMaskShapeType);

        if (attenMaskStorageShape.GetDimNum() > 1) {
            tilingData.inputParams.set_attenMaskS2Size(
                attenMaskStorageShape.GetDim(attenMaskStorageShape.GetDimNum() - 1));
        }
    }

    auto dropMaskShape = context_->GetOptionalInputShape(DROP_MASK_INPUT_INDEX);
    if (dropMaskShape != nullptr) {
        dropMaskExistFlag = 1;
    }

    OP_LOGD(opName, "pseExistFlag: %d, attenMaskExistFlag: %d, dropMaskExistFlag: %d", pseExistFlag, attenMaskExistFlag,
            dropMaskExistFlag);
    return true;
}

ge::graphStatus FlashAttentionScoreTilingBase::DoOpTiling()
{
    OP_LOGD(opName, "[%s]try template[%s]", templateName, expectTemplate.ToString().c_str());
    if (!MatchTemplate()) {
        OP_LOGD(opName, "[%s]not match template[%s]", templateName, expectTemplate.ToString().c_str());
        return ge::GRAPH_PARAM_INVALID;
    }

    SetCoreParams();
    SetMultiCoreParams();
    SetTensorSizeParams();
    SetSparseParams();
    return ge::GRAPH_SUCCESS;
}

bool FlashAttentionScoreTilingBase::MatchTemplate()
{
    // UB Size calc logic: s1s2 * X * sizeof(T) + s1d * Y * sizeof(T) + s1 * expNum * 32 + s1 * 64 + apiTmp
    BufferNum bufferNum;
    GetBufferNum(bufferNum);

    s1BasicBlock = std::numeric_limits<int64_t>::max();
    s2BasicBlock = std::numeric_limits<int64_t>::max();
    CalcS1S2BasicBlock(bufferNum);

    if (s2BasicBlock == std::numeric_limits<int64_t>::max()) {
        OP_LOGD(opName,
                "[%s]can't find proper S1S2 basic block for shape: S1[%ld] S2[%ld], D[%ld], minS1BasicBlock[%ld], "
                "dtype[%s], high precision[%d]",
                templateName, s1Size, s2Size, dSize, GetMinS1BasicBlock(),
                ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str(), isHighPercision);
        return false;
    }

    CalcDBasicBlock(bufferNum);
    actualTemplate.splitS1 = s1BasicBlock < alignedS1 ? 1 : 0;
    actualTemplate.splitS2 = s2BasicBlock < alignedS2 ? 1 : 0;
    actualTemplate.splitD = dBasicBlock < alignedD ? 1 : 0;

    if (IsTemplateMatched()) {
        (void)CalcUBSize(bufferNum, s1BasicBlock, s2BasicBlock, dBasicBlock);
        OP_LOGD(opName, "[%s]final basic block: [%ld, %ld, %ld], match template[%s]", templateName, s1BasicBlock,
                s2BasicBlock, dBasicBlock, actualTemplate.ToString().c_str());
        return true;
    }

    return false;
}

void FlashAttentionScoreTilingBase::CalcS1S2BasicBlock(const BufferNum &bufferNum)
{
    // calc s1 s2 first, we set d basic block as s2 now
    const int64_t actualD = expectTemplate.splitD == 0 ? alignedD : FRACTAL_NUM;  // if split d we use min s2 16
    int64_t maxS1BasicBlock = CalcMaxS1BasicBlockSize(actualD, bufferNum);
    maxS1BasicBlock = std::min(maxS1BasicBlock, alignedS1);
    if (maxS1BasicBlock == 0) {
        return;
    }

    for (int64_t tmpS1BasicBlock = std::min(GetMinS1BasicBlock(), maxS1BasicBlock); tmpS1BasicBlock <= maxS1BasicBlock;
         tmpS1BasicBlock += FRACTAL_NUM) {
        int64_t tmpS2BasicBlock = CalcMaxS2BasicBlockSize(bufferNum, tmpS1BasicBlock);
        tmpS2BasicBlock = std::min(tmpS2BasicBlock, alignedS2);
        for (; tmpS2BasicBlock >= FRACTAL_NUM; tmpS2BasicBlock -= FRACTAL_NUM) {
            // drop mask bug workaround
            if (dropMaskExistFlag == 1 &&
                (tmpS2BasicBlock <= BYTE_BLOCK || CalcTailSize(alignedS2, tmpS2BasicBlock) <= BYTE_BLOCK)) {
                continue;
            }

            int64_t tmpDBasicBlock = expectTemplate.splitD == 1 ? std::min(tmpS2BasicBlock, alignedD) : alignedD;
            OP_LOGD(opName, "[%s]try basic block: [%ld, %ld]", templateName, tmpS1BasicBlock, tmpS2BasicBlock);
            if (CalcUBSize(bufferNum, tmpS1BasicBlock, tmpS2BasicBlock, tmpDBasicBlock, 1) &&
                SetMatMulTiling(tmpS1BasicBlock, tmpS2BasicBlock, tmpDBasicBlock)) {
                break;
            }
        }

        // check whether is valid, if tmpS1BasicBlock is too big, then there is no proper tmpS2BasicBlock
        if (tmpS2BasicBlock < FRACTAL_NUM) {
            break;
        }

        OP_LOGD(opName, "[%s]get candidate basic block: [%ld, %ld]", templateName, tmpS1BasicBlock, tmpS2BasicBlock);
        if (s2BasicBlock == std::numeric_limits<int64_t>::max()) {
            s1BasicBlock = tmpS1BasicBlock;
            s2BasicBlock = tmpS2BasicBlock;
        } else if (s2BasicBlock == tmpS2BasicBlock && s1BasicBlock < tmpS1BasicBlock) {
            s1BasicBlock = tmpS1BasicBlock;
        } else {
            break;
        }
    }
}

void FlashAttentionScoreTilingBase::CalcDBasicBlock(const BufferNum &bufferNum)
{
    int64_t maxDBasicBlock = alignedD;
    if (bufferNum.bufferS1DNum > 0) {
        // UB Size calc logic: s1s2 * X * sizeof(T) + s1d * Y * sizeof(T) + s1 * expNum * 32 + s1 * 64 + apiTmp
        int64_t apiMinUBSize = CalcMinApiTmpSize(s1BasicBlock, s2BasicBlock, 1);
        int64_t s1s2BlockSize = s1BasicBlock * s2BasicBlock * bufferNum.bufferS1S2Num * inputDtypeBytes;
        // here 2 means FlashSoftMax sum and max output
        int64_t s1BlockSize = s1BasicBlock * (bufferNum.bufferExpNum + 2) * BYTE_BLOCK;
        maxDBasicBlock = (static_cast<int64_t>(aicoreParams_.ubSize) - apiMinUBSize - s1s2BlockSize - s1BlockSize) /
                         s1BasicBlock / bufferNum.bufferS1DNum / inputDtypeBytes;
        maxDBasicBlock = AlignDown(maxDBasicBlock, FRACTAL_NUM);
        OP_LOGD(opName, "get max dBasicBlock [%d] by UB", maxDBasicBlock);
        maxDBasicBlock = std::max(maxDBasicBlock, s2BasicBlock);
        maxDBasicBlock = std::min(alignedD, maxDBasicBlock);
    }

    if (maxDBasicBlock == s2BasicBlock) {
        dBasicBlock = s2BasicBlock;
        return;
    }

    for (int64_t tmpDBasicBlock = s2BasicBlock + FRACTAL_NUM; tmpDBasicBlock <= maxDBasicBlock;
         tmpDBasicBlock += FRACTAL_NUM) {
        matmul_tiling::MatmulApiTiling bmm2;
        OP_LOGD(opName, "[%s]try basic d block: [%ld]", templateName, tmpDBasicBlock);
        if (!SetBmm2TilingInput(s1BasicBlock, s2BasicBlock, tmpDBasicBlock, 1, bmm2) ||
            bmm2.GetTiling(tilingData.bmm2TilingData) == -1) {
            dBasicBlock = tmpDBasicBlock - FRACTAL_NUM;
            return;
        }
    }

    dBasicBlock = maxDBasicBlock;
}

int64_t FlashAttentionScoreTilingBase::CalcMaxS1BasicBlockSize(int64_t actualD, const BufferNum &bufferNum) const
{
    // if S2 basic block is min value 16, s1 basic block can reach max value, then we get:
    // s1 * 16 * X * sizeof(T) + s1d * Y * sizeof(T) + s1 * expNum * 32 + s1 * 64 + apiTmp =>
    // s1 * (16 * X + D * Y + (expNum + 2) * (32 / sizeof(T))) * sizeof(T) + apiTmp
    // just ignore apiTmp now, consider it at last
    int64_t alignUnit = BYTE_BLOCK / inputDtypeBytes;
    int64_t maxS1BasicBlock =
        aicoreParams_.ubSize / inputDtypeBytes /
        (FRACTAL_NUM * bufferNum.bufferS1S2Num + actualD * bufferNum.bufferS1DNum +
         (bufferNum.bufferExpNum + 2) * alignUnit);  // here 2 means FlashSoftMax sum and max output
    return AlignDown(maxS1BasicBlock, FRACTAL_NUM);
}

int64_t FlashAttentionScoreTilingBase::CalcMaxS2BasicBlockSize(const BufferNum &bufferNum,
                                                               int64_t tmpS1BasicBlock) const
{
    // used UB: s1s2 * X * sizeof(T) + s1d * Y * sizeof(T) + s1 * expNum * 32 + s1 * 64 + apiTmp
    // if D full load, use alignedD in above formula
    // if D not full load, use S2 basic block var in above formula
    // just ignore apiTmp now, consider it at last
    int64_t tmpS2BasicBlock;
    if (expectTemplate.splitD == 0) {
        // here 2 means FlashSoftMax sum and max output
        tmpS2BasicBlock = (aicoreParams_.ubSize - tmpS1BasicBlock * (bufferNum.bufferExpNum + 2) * BYTE_BLOCK -
                           tmpS1BasicBlock * alignedD * bufferNum.bufferS1DNum * inputDtypeBytes) /
                          (tmpS1BasicBlock * bufferNum.bufferS1S2Num * inputDtypeBytes);
    } else {
        // here 2 means FlashSoftMax sum and max output
        tmpS2BasicBlock = (aicoreParams_.ubSize - tmpS1BasicBlock * (bufferNum.bufferExpNum + 2) * BYTE_BLOCK) /
                          (tmpS1BasicBlock * (bufferNum.bufferS1DNum + bufferNum.bufferS1S2Num) * inputDtypeBytes);
    }
    return std::min(AlignDown(tmpS2BasicBlock, FRACTAL_NUM), alignedS2);
}

bool FlashAttentionScoreTilingBase::CalcUBSize(const BufferNum &bufferNum, int64_t tmpS1BasicBlock,
                                               int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch)
{
    // used UB: s1s2 * X * sizeof(T) + s1d * Y * sizeof(T) + s1 * expNum * 32 + s1 * 64 + apiTmp
    int64_t usedUBSize = 0;
    usedUBSize += batch * tmpS1BasicBlock * tmpS2BasicBlock * bufferNum.bufferS1S2Num * inputDtypeBytes;
    usedUBSize += batch * tmpS1BasicBlock * tmpDBasicBlock * bufferNum.bufferS1DNum * inputDtypeBytes;
    // here 2 means FlashSoftMax sum and max output
    usedUBSize += batch * tmpS1BasicBlock * (bufferNum.bufferExpNum + 2) * BYTE_BLOCK;

    int64_t apiMinUBSize = CalcMinApiTmpSize(tmpS1BasicBlock, tmpS2BasicBlock, batch);
    if (apiMinUBSize + usedUBSize <= static_cast<int64_t>(aicoreParams_.ubSize)) {
        apiMaxUBSize = aicoreParams_.ubSize - usedUBSize;
        minUsedUBSize = usedUBSize + apiMinUBSize;
        return true;
    } else {
        return false;
    }
}

int64_t FlashAttentionScoreTilingBase::CalcMinApiTmpSize(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                                                         int64_t batch)
{
    auto tmpShape = Shape({batch, tmpS1BasicBlock, tmpS2BasicBlock});
    int64_t softmaxFlashTmpSize = 0;
    if (expectTemplate.splitS2 == 1) {
        softmaxFlashTmpSize = AscendC::GetSoftMaxFlashV2MinTmpSize(tmpShape, calcTypeSize, sizeof(float), true,
                                                                   IsBasicBlockInSoftMax(tmpShape));
    }
    int64_t softmaxTmpSize = AscendC::GetSoftMaxMinTmpSize(tmpShape, calcTypeSize, true);
    int64_t dropoutTmpSize = 0;
    if (dropMaskExistFlag == 1) {
        dropoutTmpSize = AscendC::GetDropOutMinTmpSize(tmpShape, calcTypeSize, true);
    }
    int64_t attenMaskTmpSize = 0;
    if (attenMaskExistFlag == 1) {
        attenMaskTmpSize = AscendC::GetSelectWithBytesMaskMinTmpSize(tmpShape, Shape({1}), calcTypeSize, tmpShape,
                                                                     sizeof(uint8_t), false);
    }
    return std::max({softmaxFlashTmpSize, softmaxTmpSize, dropoutTmpSize, attenMaskTmpSize});
}

bool FlashAttentionScoreTilingBase::IsBasicBlockInSoftMax(const ge::Shape &shape) const
{
    // 2 axes at least
    if (shape.GetDimNum() < 2) {
        return false;
    }

    int64_t lastAxis = shape.GetDim(shape.GetDimNum() - 1);
    // last axis should be less than 2048 and fullfil 64 times
    int64_t basicLastAxis = 64;
    int64_t lastAxisNum = 2048;
    if (lastAxis > lastAxisNum || lastAxis % basicLastAxis != 0) {
        return false;
    }

    int64_t preAxes = 1;
    for (size_t idx = 0; idx < shape.GetDimNum() - 1; ++idx) {
        preAxes *= shape.GetDim(idx);
    }

    // all axes except last one should be 8 times
    return preAxes % 8 == 0;
}

void FlashAttentionScoreTilingBase::SetCoreParams()
{
    auto &coreParams = tilingData.coreParams;
    coreParams.set_s1BaseSize(s1BasicBlock);
    coreParams.set_s1BaseTailSize(CalcTailSize(s1Size, s1BasicBlock));
    coreParams.set_s1OuterSize(CeilDivision(s1Size, s1BasicBlock));
    coreParams.set_s2BaseSize(s2BasicBlock);
    coreParams.set_s2BaseTailSize(CalcTailSize(s2Size, s2BasicBlock));
    coreParams.set_s2OuterSize(CeilDivision(s2Size, s2BasicBlock));
    if (expectTemplate.splitS2 == 1) {
        nRatio = std::min(GetNRatio(), coreParams.get_s2OuterSize());
        coreParams.set_s2OuterSize(CeilDivision(coreParams.get_s2OuterSize(), nRatio));
    } else if (expectTemplate.splitS1 == 1) {
        nRatio = std::min(GetNRatio(), coreParams.get_s1OuterSize());
    } else {
        nRatio = 1;
    }
    coreParams.set_nRatio(nRatio);

    coreParams.set_dBaseSize(dBasicBlock);
    coreParams.set_dBaseTailSize(CalcTailSize(dSize, dBasicBlock));
    coreParams.set_dOuterSize(CeilDivision(dSize, dBasicBlock));
    SetMultiBatchCoreParams();
}

void FlashAttentionScoreTilingBase::SetMultiBatchCoreParams()
{
    auto &coreParams = tilingData.coreParams;
    coreParams.set_bBaseSize(1);
    coreParams.set_bBaseTailSize(1);
    coreParams.set_bOuterSize(bSize);

    coreParams.set_n2BaseSize(1);
    coreParams.set_n2BaseTailSize(1);
    coreParams.set_n2OuterSize(n2Size);

    coreParams.set_gBaseSize(1);
    coreParams.set_gBaseTailSize(1);
    coreParams.set_gOuterSize(gSize);
}

void FlashAttentionScoreTilingBase::SetMultiCoreParams()
{
    auto &multiCoreParams = tilingData.multiCoreParams;
    auto &coreParams = tilingData.coreParams;
    int64_t totalSize = coreParams.get_bOuterSize() * coreParams.get_n2OuterSize() * coreParams.get_gOuterSize() *
                        coreParams.get_s1OuterSize();
    int64_t actualUsedAivNum = std::min(totalSize, static_cast<int64_t>(aivNum));
    multiCoreParams.set_coreNum(static_cast<int32_t>(actualUsedAivNum));
    multiCoreParams.set_totalSize(totalSize);
    multiCoreParams.set_splitFactorSize(CeilDivision(totalSize, actualUsedAivNum));
    multiCoreParams.set_splitFactorTailSize(CalcTailSize(totalSize, multiCoreParams.get_splitFactorSize()));
}

ge::graphStatus FlashAttentionScoreTilingBase::DoLibApiTiling()
{
    (void)SetMatMulTiling(s1BasicBlock, s2BasicBlock, dBasicBlock, batchBasic);
    SetSoftMaxTiling();
    SetDataCopyTransposeTiling();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreTilingBase::PostTiling()
{
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());  // already check capcity in CheckContext
    auto platformInfo = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto aicNum = ascendcPlatform.GetCoreNumAic();
    auto blockDim = ascendcPlatform.CalcTschBlockDim(tilingData.multiCoreParams.get_coreNum(), aicNum, aivNum);
    context_->SetBlockDim(blockDim);
    OP_LOG_FULL(DLOG_DEBUG, opName, "[%s]tiling data: %s", templateName, DebugTilingData(context_).c_str());
    OP_LOGD(opName, "[%s]tiling data size: %zu", templateName, tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

bool FlashAttentionScoreTilingBase::SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                                                       int64_t tmpDBasicBlock, int64_t batch,
                                                       matmul_tiling::MatmulApiTiling &bmm1)
{
    bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
    bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, true);
    bmm1.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, bmm1OutDtype);
    bmm1.SetShape(std::min(tmpS1BasicBlock, s1Size), std::min(tmpS2BasicBlock, s2Size), dSize);
    bmm1.SetOrgShape(s1Size, s2Size, s1StrideSize, s2StrideSize);
    bmm1.SetBias(false);
    bmm1.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize);
    bmm1.SetFixSplit(tmpS1BasicBlock, tmpS2BasicBlock);

    return true;
}

bool FlashAttentionScoreTilingBase::SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                                                    int64_t tmpDBasicBlock, int64_t batch)
{
    matmul_tiling::MatmulApiTiling bmm1;
    matmul_tiling::MatmulApiTiling bmm2;

    if (!SetBmm1TilingInput(tmpS1BasicBlock, tmpS2BasicBlock, tmpDBasicBlock, batch, bmm1) ||
        !SetBmm2TilingInput(tmpS1BasicBlock, tmpS2BasicBlock, tmpDBasicBlock, batch, bmm2)) {
        return false;
    }

    if (bmm1.GetTiling(tilingData.bmm1TilingData) == -1) {
        return false;
    }
    tilingData.bmm1TilingData.set_shareMode(0);
    tilingData.bmm1TilingData.set_shareL1Size(aicoreParams_.l1Size);
    tilingData.bmm1TilingData.set_shareL0CSize(aicoreParams_.l0cSize);

    if (bmm2.GetTiling(tilingData.bmm2TilingData) == -1) {
        return false;
    }

    tilingData.bmm2TilingData.set_shareMode(0);
    tilingData.bmm2TilingData.set_shareL1Size(aicoreParams_.l1Size);
    tilingData.bmm2TilingData.set_shareL0CSize(aicoreParams_.l0cSize);

    return true;
}

void FlashAttentionScoreTilingBase::SetSoftMaxTiling()
{
    auto softmaxShape = Shape({batchBasic, std::min(s1BasicBlock, alignedS1), std::min(s2BasicBlock, alignedS2)});

    AscendC::SoftMaxTilingFunc(softmaxShape, sizeof(float), apiMaxUBSize, tilingData.softmaxTilingData);
    AscendC::SoftMaxFlashV2TilingFunc(softmaxShape, calcTypeSize, sizeof(float), apiMaxUBSize,
                                      tilingData.softmaxFlashTilingData, true, IsBasicBlockInSoftMax(softmaxShape));
    AscendC::SoftMaxTilingFunc(softmaxShape, sizeof(float), apiMaxUBSize, tilingData.softmaxTilingDataTailCore);
    AscendC::SoftMaxFlashV2TilingFunc(softmaxShape, calcTypeSize, sizeof(float), apiMaxUBSize,
                                      tilingData.softmaxTilingDataTailCore, true, false);
}

void FlashAttentionScoreTilingBase::SetDataCopyTransposeTiling()
{
    auto &coreParams = tilingData.coreParams;
    auto transposeSrcShape = Shape({coreParams.get_bBaseSize(), 1, std::min(s1BasicBlock, alignedS1),
                                    coreParams.get_gBaseSize() * std::min(dBasicBlock, alignedD)});
    auto transposeDstShape = Shape({bSize, n1Size, s1Size, n1Size * dSize});
    GetDataCopyTransposeTiling(transposeDstShape, transposeSrcShape, inputDtypeBytes, tilingData.transposeTilingData);
}

void FlashAttentionScoreTilingBase::SetTensorSizeParams()
{
    auto &tensorSizeParams = tilingData.tensorSizeParams;
    auto &coreParams = tilingData.coreParams;
    int64_t batchInnerSize = coreParams.get_bBaseSize() * coreParams.get_n2BaseSize() * coreParams.get_gBaseSize();
    tensorSizeParams.set_bmm1ResUbSize(batchInnerSize * s1BasicBlock * s2BasicBlock);
    tensorSizeParams.set_attenMaskUbSize(attenMaskExistFlag * batchInnerSize * s1BasicBlock * s2BasicBlock);
    if (tilingData.inputParams.get_pseShapeType() == PSE_B_N2_G_S1_S2) {
        tensorSizeParams.set_pseUbSize(pseExistFlag * batchInnerSize * s1BasicBlock * s2BasicBlock);
    } else {
        tensorSizeParams.set_pseUbSize(pseExistFlag * batchInnerSize * s2BasicBlock);  // PSE_B_N2_G_1_S2
    }

    if (dropMaskExistFlag == 1) {
        tensorSizeParams.set_dropMaskUbSize(batchInnerSize * s1BasicBlock *
                                            AlignUp(s2BasicBlock, DROP_MASK_ALIGN_UNIT) / BYTE_BIT_NUM /
                                            inputDtypeBytes);
    }

    if (inputDtype == ge::DT_BF16 || isHighPercision) {
        if (expectTemplate.splitS2 == 1) {
            tensorSizeParams.set_castUbSize(batchInnerSize * s1BasicBlock * std::max(s2BasicBlock, dBasicBlock));
        } else {
            tensorSizeParams.set_castUbSize(batchInnerSize * s1BasicBlock * s2BasicBlock);
        }
    }
    tensorSizeParams.set_softmaxMaxUbSize(batchInnerSize * s1BasicBlock * (BYTE_BLOCK / sizeof(float)));
    tensorSizeParams.set_softmaxSumUbSize(batchInnerSize * s1BasicBlock * (BYTE_BLOCK / sizeof(float)));
    tensorSizeParams.set_softmaxExpUbSize(batchInnerSize * s1BasicBlock * (BYTE_BLOCK / calcTypeSize));
    tensorSizeParams.set_apiTmpBufferBytes(apiMaxUBSize);

    tensorSizeParams.set_bmm2ResUbSize(batchInnerSize * s1BasicBlock * dBasicBlock);
}

bool FlashAttentionScoreTilingBase::InitSparseValidArray(std::vector<int64_t> &sparseValidArray,
                                                         int64_t bIdx)
{
    OP_TILING_CHECK(sparseValidArray.size() == 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName, "Sparse valid array size should be larger than 0."),
                    return false);
    uint8_t sparseType = tilingData.inputParams.get_sparseType();
    if (sparseType == static_cast<uint8_t>(SparseEnum::PREFIX)) {
        for (int64_t i = 0; i < static_cast<int64_t>(sparseValidArray.size()); i++) {
            int64_t s2IgnoredEndLen = tilingData.inputParams.get_s1Size() -
                                      tilingData.coreParams.get_s1BaseSize() * (i + 1);
            int64_t s2EndLen = 0;
            s2IgnoredEndLen = std::max((int64_t)0, s2IgnoredEndLen);
            if (tilingData.inputParams.get_s2Size() > s2IgnoredEndLen) {
                s2EndLen = tilingData.inputParams.get_s2Size() - s2IgnoredEndLen;
            } else {
                s2EndLen = tilingData.inputParams.get_s2Size();
            }
            s2EndLen = std::max(s2EndLen, tilingData.inputParams.get_prefixNList()[bIdx]);
            s2EndLen = std::min(s2EndLen, tilingData.inputParams.get_s2Size());
            sparseValidArray[i] = CeilDivision(s2EndLen, s2BasicBlock);
        }
    } else {
        int64_t s2BlkNum = CeilDivision(s2Size, s2BasicBlock);
        int64_t validS1Size = CeilDivision(preTokens, s1BasicBlock);
        int64_t validS2Size = CeilDivision(nextTokens, s2BasicBlock);
        for (int64_t i = 0; i < static_cast<int64_t>(sparseValidArray.size()); i++) {
            int64_t reduceBlk = (i < validS1Size) ?
                                 0 : (CeilDivision((i + 1) * s1BasicBlock - preTokens, s2BasicBlock) - 1);
            int64_t addBlk = std::min(s2BlkNum - validS2Size,
                                      CeilDivision((i + 1) * s1BasicBlock + nextTokens, s2BasicBlock) - validS2Size);
            sparseValidArray[i] = validS2Size - reduceBlk + addBlk;
        }
    }
    return true;
}

bool FlashAttentionScoreTilingBase::PartitionSparseData(const std::vector<int64_t> &sparseRollingArray,
                                                        int64_t sparseRollingArraySum,
                                                        int64_t sparseArraySize,
                                                        int64_t loadMaxEachCore,
                                                        std::vector<int64_t> &partitionResult)
{
    OP_TILING_CHECK(partitionResult.size() == 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName, "partitionResult size should be larger than 0."),
                    return false);

    OP_TILING_CHECK(sparseRollingArraySum <= 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName, "sparseRollingArraySum should be larger than 0."),
                    return false);
    int64_t s1OuterCutEachCore = loadMaxEachCore / sparseRollingArraySum;
    int64_t s1OuterLoadEachCore = s1OuterCutEachCore * sparseRollingArraySum;
    int64_t s1OuterNumEachCore = s1OuterCutEachCore * sparseRollingArray.size();

    int64_t targetCoreNum = partitionResult.size();
    int64_t coreIdx = 0;
    int64_t rollingIdx = 0;
    int64_t loadSize = s1OuterLoadEachCore;
    partitionResult[0] = 0;
    for (int64_t i = s1OuterNumEachCore; i < sparseArraySize; i++, rollingIdx++) {
        rollingIdx = (static_cast<uint64_t>(rollingIdx) >= sparseRollingArray.size()) ? 0 : rollingIdx;
        int64_t loadNext = sparseRollingArray[rollingIdx];
        bool needOneMoreCore = (loadSize + loadNext) > loadMaxEachCore;
        if (needOneMoreCore && coreIdx >= (targetCoreNum - 1)) {
            return false;
        }

        if (needOneMoreCore) {
            partitionResult[++coreIdx] = i;
            i += s1OuterNumEachCore;
            i--;
            rollingIdx--;
            loadSize = s1OuterLoadEachCore;
            continue;
        }

        loadSize += loadNext;
    }

    std::fill(partitionResult.begin() + coreIdx + 1, partitionResult.end(), sparseArraySize);
    return true;
}

void FlashAttentionScoreTilingBase::SetPrefixSparseStartIdx(const std::vector<std::vector<int64_t>> &sparseValidArray,
                                                            MultiCoreParams &multiCoreParams) {
    int64_t validAivNum = std::min(static_cast<int64_t>(multiCoreParams.get_coreNum()), MAX_AIV_NUM);
    int64_t totalSize = multiCoreParams.get_totalSize();  // BN2GS1.o
    int64_t *sparseStartIdx = multiCoreParams.get_sparseStartIdx();
    if (totalSize <= validAivNum) {
        int64_t idx = 0;
        for (; idx < totalSize; ++idx) {
            sparseStartIdx[idx] = idx;
        }
        for (; idx < validAivNum; ++idx) {
            sparseStartIdx[idx] = totalSize;
        }
        return;
    }

    int64_t loadTotal = 0;
    /* Need to adapt when we split b. */
    for (int64_t i = 0; i < bSize; i++) {
        loadTotal += std::accumulate(sparseValidArray[i].begin(), sparseValidArray[i].end(), 0);
    }
    int64_t n2G = tilingData.coreParams.get_n2OuterSize() * tilingData.coreParams.get_gOuterSize();
    loadTotal *= n2G;

    auto loadEachCoreExpect = CeilDivision(loadTotal, validAivNum);
    int64_t s1OuterSize = tilingData.coreParams.get_s1OuterSize();
    int64_t tempBlock = 0;
    int64_t coreIdx = 0;
    int64_t loadStartIdx = 0;

    for (int64_t bNGS1Index = 0; bNGS1Index < bSize * n2G * s1OuterSize; ++bNGS1Index) {
        int64_t bIdx = bNGS1Index / (n2G * s1OuterSize);
        if (s1OuterSize == 0) {
            continue;
        }
        int64_t s1Idx = bNGS1Index % s1OuterSize;
        auto currBlockNum = sparseValidArray[bIdx][s1Idx];
        if (tempBlock >= loadEachCoreExpect) {
            if ((tempBlock + currBlockNum - loadEachCoreExpect) >= (loadEachCoreExpect - (tempBlock))) {
                /* 不累加当前block */
                sparseStartIdx[coreIdx++] = loadStartIdx;
                loadStartIdx = bNGS1Index;
                // 下一个核使用当前这个S1
                tempBlock = currBlockNum;
            } else {
                sparseStartIdx[coreIdx++] = loadStartIdx;
                loadStartIdx = (bNGS1Index+1);
                tempBlock = 0;
            }
        } else {
            tempBlock += currBlockNum;
        }
    }

    if (tempBlock != 0) {
        sparseStartIdx[coreIdx++] = loadStartIdx;
    }

    /* 将没用到的核的start index置为最大值 */
    for (; coreIdx < MAX_AIV_NUM; ++coreIdx) {
        sparseStartIdx[coreIdx] = totalSize;
    }
}

bool FlashAttentionScoreTilingBase::SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray,
                                                      MultiCoreParams &multiCoreParams)
{
    // to avoid buffer overflow, or maybe sometimes we want to only verify single core
    int64_t validAivNum = std::min(static_cast<int64_t>(multiCoreParams.get_coreNum()), MAX_AIV_NUM);
    int64_t totalSize = multiCoreParams.get_totalSize();  // BN2GS1.o
    int64_t *sparseStartIdx = multiCoreParams.get_sparseStartIdx();
    for (int64_t idx = 0; idx < MAX_AIV_NUM; ++idx) {
        sparseStartIdx[idx] = totalSize;
    }

    if (totalSize <= validAivNum) {
        for (int64_t idx = 0; idx < totalSize; ++idx) {
            sparseStartIdx[idx] = idx;
        }

        return true;
    }

    // Minimize the max load each core to find a load balance result.
    // The range of max load each core is (loadEachCoreLowerBound, loadEachCoreUpperBound].
    std::vector<int64_t> partitionResult(validAivNum, totalSize);
    std::vector<int64_t> lastValidPartitionResult(validAivNum, totalSize);
    int64_t sparseArraySum = std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0);
    int64_t loadTotal = sparseArraySum * (totalSize / sparseValidArray.size());
    OP_TILING_CHECK(validAivNum <= 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName, "validAivNum should be larger than 0."),
                    return false);
    int64_t loadEachCoreLowerBound = loadTotal / validAivNum - 1;
    int64_t loadEachCoreUpperBound = CeilDivision(loadTotal, validAivNum) +
                                     (*std::max_element(sparseValidArray.begin(), sparseValidArray.end()));
    while (loadEachCoreLowerBound + 1 < loadEachCoreUpperBound) {
        int64_t loadMax = loadEachCoreLowerBound + (loadEachCoreUpperBound - loadEachCoreLowerBound) / 2;
        if ((loadMax * validAivNum >= loadTotal) &&
            PartitionSparseData(sparseValidArray, sparseArraySum, totalSize, loadMax, partitionResult)) {
            loadEachCoreUpperBound = loadMax;
            lastValidPartitionResult.swap(partitionResult);
            continue;
        }
        loadEachCoreLowerBound = loadMax;
    }

    for (int64_t idx = 0; idx < validAivNum; ++idx) {
        sparseStartIdx[idx] = lastValidPartitionResult[idx];
    }

    if (CheckLogLevel(OP, DLOG_DEBUG) == 1) {
        PrintSparseMaxMinLoadPerCore(sparseValidArray, sparseStartIdx, validAivNum,
                                     CeilDivision(loadTotal, validAivNum));
    }
    return true;
}

void FlashAttentionScoreTilingBase::PrintSparseMaxMinLoadPerCore(const std::vector<int64_t> &sparseValidArray,
                                                                 int64_t *sparseStartIdx, int32_t validAivNum,
                                                                 int64_t avgLoadSize)
{
    int64_t maxLoadSize = 0;
    int64_t minLoadSize = std::numeric_limits<int64_t>::max();
    int64_t totalSize = tilingData.multiCoreParams.get_totalSize();
    int64_t s1OuterSize = tilingData.coreParams.get_s1OuterSize();
    if (s1OuterSize == 0) {
        return;
    }
    for (int64_t idx = 0; idx < validAivNum; ++idx) {
        int64_t startIdx = sparseStartIdx[idx];
        int64_t endIdx = totalSize;
        if (idx + 1 < validAivNum) {
            endIdx = sparseStartIdx[idx + 1];
        }

        if (startIdx >= endIdx) {
            minLoadSize = 0;
            break;
        }

        int64_t s1OuterStartIdx = startIdx % s1OuterSize;
        int64_t s1OuterEndIdx = endIdx % s1OuterSize;
        int64_t loadSize = 0;
        if (s1OuterEndIdx > s1OuterStartIdx) {
            loadSize = std::accumulate(sparseValidArray.begin() + s1OuterStartIdx,
                                       sparseValidArray.begin() + s1OuterEndIdx, 0);
        } else {
            loadSize = std::accumulate(sparseValidArray.begin() + s1OuterStartIdx, sparseValidArray.end(), 0);
            loadSize = std::accumulate(sparseValidArray.begin(), sparseValidArray.begin() + s1OuterEndIdx, loadSize);
        }

        int64_t s1OuterLoop = (endIdx / s1OuterSize) - (startIdx / s1OuterSize);
        if (s1OuterLoop > 1) {
            if (s1OuterEndIdx > s1OuterStartIdx) {
                loadSize += s1OuterLoop * std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0);
            } else {
                loadSize += (s1OuterLoop - 1) * std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0);
            }
        }

        maxLoadSize = std::max(maxLoadSize, loadSize);
        minLoadSize = std::min(minLoadSize, loadSize);
    }

    OP_LOGD(opName, "[%s]each core load: max[%ld], min[%ld], avg[%ld]", templateName, maxLoadSize, minLoadSize,
            avgLoadSize);
}

void FlashAttentionScoreTilingBase::SetSparseParams()
{
    if (tilingData.inputParams.get_sparseType() == static_cast<uint8_t>(SparseEnum::ALL)) {
        return;
    }

    if (expectTemplate.splitS2 == 0) {
        OP_LOGI(opName, "[%s]match not split S2 template, close sparse feature", templateName);
        tilingData.inputParams.set_sparseType(static_cast<uint8_t>(SparseEnum::ALL));
        return;
    }

    auto &coreParams = tilingData.coreParams;
    coreParams.set_s1SparseValidSize(preTokens);
    coreParams.set_s2SparseValidSize(nextTokens);

    auto &multiCoreParams = tilingData.multiCoreParams;
    if (tilingData.inputParams.get_sparseType() == static_cast<uint8_t>(SparseEnum::PREFIX))  {
        std::vector<std::vector<int64_t>> sparseValidArray;
        for (int64_t bIdx = 0; bIdx < bSize; bIdx++) {
            sparseValidArray.emplace_back(std::vector<int64_t>(coreParams.get_s1OuterSize(), 0));
            if (tilingData.inputParams.get_prefixNList()[bIdx] >= s2Size) {
                tilingData.inputParams.set_sparseType(static_cast<uint8_t>(SparseEnum::ALL));
                return;
            }
            InitSparseValidArray(sparseValidArray.back(), bIdx);
        }
        SetPrefixSparseStartIdx(sparseValidArray, multiCoreParams);
    } else {
        std::vector<int64_t> sparseValidArray(coreParams.get_s1OuterSize(), 0);
        InitSparseValidArray(sparseValidArray, 0);
        SetSparseStartIdx(sparseValidArray, multiCoreParams);
    }
}

ge::graphStatus FlashAttentionScoreTilingBase::GetWorkspaceSize()
{
    auto &tensorSizeParams = tilingData.tensorSizeParams;
    auto &coreParams = tilingData.coreParams;

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    int64_t bmm1Byetes = coreParams.get_nRatio() * tensorSizeParams.get_bmm1ResUbSize() * calcTypeSize;
    int64_t bmm2Byetes = tensorSizeParams.get_bmm2ResUbSize() * calcTypeSize;
    workspaces[0] = static_cast<size_t>((bmm1Byetes + bmm2Byetes) * aivNum) + WORK_SPACE_RESERVE_SIZE;
    return ge::GRAPH_SUCCESS;
}

class FlashAttentionScoreTilingSplitD : public FlashAttentionScoreTilingBase {
public:
    explicit FlashAttentionScoreTilingSplitD(gert::TilingContext *context) : FlashAttentionScoreTilingBase(context)
    {
        splitBatchAxis = AxisEnum::G;
        expectTemplate.splitD = 1;
        templateName = "SplitD";
    }
    ~FlashAttentionScoreTilingSplitD() override = default;

protected:
    bool AnalyzeDtype() override
    {
        OP_TILING_CHECK(!FlashAttentionScoreTilingBase::AnalyzeDtype(),
                        VECTOR_INNER_ERR_REPORT_TILIING(opName, "fail to anlyze base dtype"), return false);
        bmm2OutDtype = bmmDtype;
        return true;
    }

    void GetBufferNum(BufferNum &bufferNum) const override
    {
        bufferNum.bufferS1S2Num = 3;  // 3 s1*s2 block with input dtype
        bufferNum.bufferS1DNum = 0;
        bufferNum.bufferExpNum = 0;
        if (inputDtype == ge::DT_BF16 || isHighPercision) {
            bufferNum.bufferS1S2Num = 7;  // 7 s1*s2 block with input dtype, here fp32 buffer_num * 2
        }

        bool existOptionalInput = (pseExistFlag + attenMaskExistFlag + dropMaskExistFlag > 0);
        if (!existOptionalInput) {
            --bufferNum.bufferS1S2Num;
        }
    }

    bool IsTemplateMatched() const override
    {
        // no not split D template when S1 S2 full load for now
        return expectTemplate.splitS1 == actualTemplate.splitS1 && expectTemplate.splitS2 == actualTemplate.splitS2;
    }

    uint64_t GetTilingKey() const override
    {
        return GET_TILINGKEY(AxisEnum::NONE, AxisEnum::D, splitBatchAxis, tilingKeyDType, tilingKeyLayout,
                             SparseEnum::ANY);
    }

    bool SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm1) override
    {
        bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, true);
        bmm1.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, bmm1OutDtype);
        bmm1.SetShape(std::min(tmpS1BasicBlock, s1Size), std::min(tmpS2BasicBlock, s2Size), dSize);
        bmm1.SetOrgShape(s1Size, s2Size, s1StrideSize, s2StrideSize);
        bmm1.SetALayout(bSize, s1Size, n2Size, gSize, dSize);
        bmm1.SetBLayout(bSize, s2Size, n2Size, 1, dSize);
        bmm1.SetCLayout(bSize, s1Size, n2Size, gSize, s2Size);
        bmm1.SetBatchNum(batch);
        bmm1.SetBias(false);
        bmm1.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize);
        bmm1.SetFixSplit(tmpS1BasicBlock, tmpS2BasicBlock);

        return true;
    }

    bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm2) override
    {
        bmm2.SetAType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::NZ, bmmDtype, false);
        bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2OutDtype);
        bmm2.SetShape(std::min(tmpS1BasicBlock, s1Size), std::min(tmpDBasicBlock, dSize),
                      std::min(tmpS2BasicBlock, s2Size));
        bmm2.SetOrgShape(s1Size, s1StrideSize, s2Size, s2StrideSize);  // consider broadcst, N same as A tensor
        bmm2.SetALayout(bSize, s1Size, n2Size, gSize, s2Size);
        bmm2.SetBLayout(bSize, s2Size, n2Size, 1, dSize);
        bmm2.SetCLayout(bSize, s1Size, n2Size, gSize, dSize);
        bmm2.SetBatchNum(batch);
        bmm2.SetBias(false);
        bmm2.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize);
        bmm2.SetFixSplit(tmpS1BasicBlock, tmpDBasicBlock);

        return true;
    }

    void SetMultiBatchCoreParams() override
    {
        splitBatchAxis = AxisEnum::G;

        CalcBatchBasic();
        FlashAttentionScoreTilingBase::SetMultiBatchCoreParams();
        if (batchBasic == 1 && gSize == 1) {
            // if can only load single batch, go to split N2 batch template, split G template is not ready for now
            splitBatchAxis = AxisEnum::N2;
            return;
        }
        auto &coreParams = tilingData.coreParams;
        coreParams.set_gBaseSize((int32_t)std::min(batchBasic, gSize));
        coreParams.set_gBaseTailSize(CalcTailSize(gSize, batchBasic));
        coreParams.set_gOuterSize(CeilDivision(gSize, batchBasic));
        if (batchBasic <= gSize) {
            return;
        }

        splitBatchAxis = AxisEnum::N2;
        int64_t leftBatch = batchBasic / gSize;
        coreParams.set_n2BaseSize(std::min(leftBatch, n2Size));
        coreParams.set_n2BaseTailSize(CalcTailSize(n2Size, leftBatch));
        coreParams.set_n2OuterSize(CeilDivision(n2Size, leftBatch));
        // when layout is BSH (BSND), bmm does not support loading multi-line S calculations, skip split B template
        if (leftBatch <= n2Size || tilingKeyLayout == LayoutEnum::BSND) {
            batchBasic = leftBatch * gSize;
            return;
        }

        splitBatchAxis = AxisEnum::B;
        leftBatch = leftBatch / n2Size;
        coreParams.set_bBaseSize(std::min(leftBatch, bSize));
        coreParams.set_bBaseTailSize(CalcTailSize(bSize, leftBatch));
        coreParams.set_bOuterSize(CeilDivision(bSize, leftBatch));
        batchBasic = leftBatch * gSize * n2Size;
    }

    void CalcBatchBasic()
    {
        int64_t totalBatch = bSize * n2Size * gSize;
        batchBasic = std::min(static_cast<int64_t>(aicoreParams_.ubSize) / minUsedUBSize, totalBatch);
        BufferNum bufferNum;
        GetBufferNum(bufferNum);
        int64_t tmpBatch = batchBasic;
        if (CalcUBSize(bufferNum, s1BasicBlock, s2BasicBlock, dBasicBlock, tmpBatch) &&
            SetMatMulTiling(s1BasicBlock, s2BasicBlock, dBasicBlock, tmpBatch)) {
            batchBasic = tmpBatch;
            ++tmpBatch;
            while (tmpBatch <= totalBatch && CalcUBSize(bufferNum, s1BasicBlock, s2BasicBlock, dBasicBlock, tmpBatch) &&
                   SetMatMulTiling(s1BasicBlock, s2BasicBlock, dBasicBlock, tmpBatch)) {
                batchBasic = tmpBatch;
                ++tmpBatch;
            }
        } else {
            --tmpBatch;
            while (tmpBatch > 1 && (!CalcUBSize(bufferNum, s1BasicBlock, s2BasicBlock, dBasicBlock, tmpBatch) ||
                                    !SetMatMulTiling(s1BasicBlock, s2BasicBlock, dBasicBlock, tmpBatch))) {
                --tmpBatch;
            }
            batchBasic = tmpBatch;
        }
    }

private:
    AxisEnum splitBatchAxis;
};

class FlashAttentionScoreTilingSplitS1D : public FlashAttentionScoreTilingBase {
public:
    explicit FlashAttentionScoreTilingSplitS1D(gert::TilingContext *context) : FlashAttentionScoreTilingBase(context)
    {
        expectTemplate.splitS1 = 1;
        expectTemplate.splitD = 1;
        templateName = "SplitS1D";
    }
    ~FlashAttentionScoreTilingSplitS1D() override = default;

protected:
    bool AnalyzeDtype() override
    {
        OP_TILING_CHECK(!FlashAttentionScoreTilingBase::AnalyzeDtype(),
                        VECTOR_INNER_ERR_REPORT_TILIING(opName, "fail to anlyze base dtype"), return false);
        bmm2OutDtype = bmmDtype;
        return true;
    }

    void GetBufferNum(BufferNum &bufferNum) const override
    {
        bufferNum.bufferS1S2Num = 3;  // 3 s1*s2 block with input dtype
        bufferNum.bufferS1DNum = 0;
        bufferNum.bufferExpNum = 0;
        if (inputDtype == ge::DT_BF16 || isHighPercision) {
            bufferNum.bufferS1S2Num = 6;  // 6 s1*s2 block with input dtype, here fp32 buffer_num * 2
        }

        bool existOptionalInput = (pseExistFlag + attenMaskExistFlag + dropMaskExistFlag > 0);
        if (!existOptionalInput) {
            --bufferNum.bufferS1S2Num;
        }
    }

    bool SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm1) override
    {
        bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, true);
        bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1OutDtype);
        bmm1.SetShape(std::min(tmpS1BasicBlock, s1Size), std::min(tmpS2BasicBlock, s2Size), dSize);
        bmm1.SetOrgShape(s1Size, s2Size, s1StrideSize, s2StrideSize);
        bmm1.SetBias(false);
        bmm1.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize);
        bmm1.SetFixSplit(tmpS1BasicBlock, tmpS2BasicBlock);

        return true;
    }

    bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm2) override
    {
        bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2OutDtype);
        bmm2.SetShape(std::min(tmpS1BasicBlock, s1Size), std::min(tmpDBasicBlock, dSize),
            std::min(tmpS2BasicBlock, s2Size));
        bmm2.SetOrgShape(s1Size, s1StrideSize, tmpS2BasicBlock, s2StrideSize);
        bmm2.SetBias(false);
        bmm2.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize);
        bmm2.SetFixSplit(tmpS1BasicBlock, tmpDBasicBlock);
        return true;
    }

    int64_t CalcMinApiTmpSize(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t batch) override
    {
        auto tmpShape = Shape({batch, tmpS1BasicBlock, tmpS2BasicBlock});
        int64_t softmaxFlashTmpSize = 0;
        if (expectTemplate.splitS2 == 1) {
            softmaxFlashTmpSize = AscendC::GetSoftMaxFlashV2MinTmpSize(tmpShape, calcTypeSize, sizeof(float), true,
                                                                       IsBasicBlockInSoftMax(tmpShape));
        }
        int64_t softmaxTmpSize = 32768;
        int64_t dropoutTmpSize = 0;
        if (dropMaskExistFlag == 1) {
            dropoutTmpSize = AscendC::GetDropOutMinTmpSize(tmpShape, calcTypeSize, true);
        }
        int64_t attenMaskTmpSize = 0;
        if (attenMaskExistFlag == 1) {
            attenMaskTmpSize = AscendC::GetSelectWithBytesMaskMinTmpSize(tmpShape, Shape({1}), calcTypeSize, tmpShape,
                                                                        sizeof(uint8_t), false);
        }
        return std::max({softmaxFlashTmpSize, softmaxTmpSize, dropoutTmpSize, attenMaskTmpSize});
    }

    bool IsTemplateMatched() const override
    {
        // always split D template when split S1 for now
        return expectTemplate.splitS1 == actualTemplate.splitS1 && expectTemplate.splitS2 == actualTemplate.splitS2;
    }

    uint64_t GetTilingKey() const override
    {
        // not care about layout in tiling key, pass BSND(enum value is 0)
        return GET_TILINGKEY(AxisEnum::S1, AxisEnum::D, AxisEnum::NONE, tilingKeyDType, LayoutEnum::BSND,
                             SparseEnum::ANY);
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        auto &coreParams = tilingData.coreParams;
        size_t *workspaces = context_->GetWorkspaceSizes(1);
        int64_t bmm1Size = coreParams.get_nRatio() * coreParams.get_s1BaseSize() * coreParams.get_s2BaseSize();
        int64_t bmm1AlignByetes = bmm1Size * calcTypeSize;
        int64_t stage1AlignByetes = bmm1Size * inputDtypeBytes;
        workspaces[0] = static_cast<size_t>((bmm1AlignByetes + stage1AlignByetes) * aivNum) +
            WORK_SPACE_RESERVE_SIZE;
        return ge::GRAPH_SUCCESS;
    }
};

class FlashAttentionScoreTilingSplitS1DDBHighPerformace : public FlashAttentionScoreTilingBase {
public:
    explicit FlashAttentionScoreTilingSplitS1DDBHighPerformace(gert::TilingContext *context)
        : FlashAttentionScoreTilingBase(context)
    {
        expectTemplate.splitS1 = 1;
        expectTemplate.splitD = 1;
        templateName = "SplitS1DDBHighPerformace";
    }
    ~FlashAttentionScoreTilingSplitS1DDBHighPerformace() override = default;

protected:
    int64_t s1Ratio = 1;
    int64_t WORKSPACE_LIMIT = 131072; // 8*128*128
    int64_t SOFTMAX_EXTRA_SIZE = 512;
    int64_t S1D_HIGH_PERF_BUFFER_NUM = 4;
    int64_t S2SIZE_LIMIT = 1024;
    int64_t IS_BASIC_BLOCK_NUM = 64;
    int64_t MIN_SIZE_LIMIT = 16384; // 128*128

    bool AnalyzeDtype() override
    {
        OP_TILING_CHECK(!FlashAttentionScoreTilingBase::AnalyzeDtype(),
            VECTOR_INNER_ERR_REPORT_TILIING(opName, "fail to anlyze base dtype"), return false);
        bmm2OutDtype = bmmDtype;
        return true;
    }

    void SetCoreParams() override
    {
        // 稀疏场景不开启S1轴N:1配比
        FlashAttentionScoreTilingBase::SetCoreParams();
        if (tilingData.inputParams.get_sparseType() != static_cast<uint8_t>(SparseEnum::ALL)) {
            return;
        }
        SetMultiCoreParams();
        auto &coreParams = tilingData.coreParams;
        auto &multiCoreParams = tilingData.multiCoreParams;
        // 当前能分满核，考虑增大N
        while (multiCoreParams.get_totalSize() > ((GetNRatio() - 1) * aivNum / GetNRatio()) &&
            s1BasicBlock * GetNRatio() < s1Size && GetNRatio() * s1BasicBlock * alignedS2 <= WORKSPACE_LIMIT) {
            s1Ratio++;
            FlashAttentionScoreTilingBase::SetCoreParams();
            // S1轴N:1使能
            coreParams.set_s1OuterSize(CeilDivision(coreParams.get_s1OuterSize(), GetNRatio()));
            coreParams.set_s1BaseTailSize(CalcTailSize(s1Size, s1BasicBlock * GetNRatio()));
            SetMultiCoreParams();
        }
        // 分不满核，减小N
        if (multiCoreParams.get_totalSize() <= ((GetNRatio() - 1) * aivNum / GetNRatio()) ||
            s1BasicBlock * GetNRatio() > s1Size || GetNRatio() * s1BasicBlock * alignedS2 > WORKSPACE_LIMIT) {
            s1Ratio--;
            FlashAttentionScoreTilingBase::SetCoreParams();
            coreParams.set_s1OuterSize(CeilDivision(coreParams.get_s1OuterSize(), GetNRatio()));
            coreParams.set_s1BaseTailSize(CalcTailSize(s1Size, s1BasicBlock * GetNRatio()));
        }
    }

    void SetSparseParams() override
    {
        if (tilingData.inputParams.get_sparseType() == static_cast<uint8_t>(SparseEnum::ALL)) {
            return;
        }
        auto &coreParams = tilingData.coreParams;
        coreParams.set_s1SparseValidSize(preTokens);
        coreParams.set_s2SparseValidSize(nextTokens);
        auto &multiCoreParams = tilingData.multiCoreParams;
        if (tilingData.inputParams.get_sparseType() == static_cast<uint8_t>(SparseEnum::PREFIX))  {
            std::vector<std::vector<int64_t>> sparseValidArray;
            for (int64_t bIdx = 0; bIdx < bSize; bIdx++) {
                sparseValidArray.emplace_back(std::vector<int64_t>(coreParams.get_s1OuterSize(), 0));
                if (tilingData.inputParams.get_prefixNList()[bIdx] >= s2Size) {
                    tilingData.inputParams.set_sparseType(static_cast<uint8_t>(SparseEnum::ALL));
                    return;
                }
                InitSparseValidArray(sparseValidArray.back(), bIdx);
            }
            SetPrefixSparseStartIdx(sparseValidArray, multiCoreParams);
        } else {
            std::vector<int64_t> sparseValidArray(coreParams.get_s1OuterSize(), 0);
            InitSparseValidArray(sparseValidArray, 0);
            SetSparseStartIdx(sparseValidArray, multiCoreParams);
        }
    }

    int64_t GetNRatio() override
    {
        return s1Ratio;
    }

    void GetSparseInfo() override
    {
        auto &inputParams = tilingData.inputParams;

        SparseEnum sparseType = SparseEnum::ALL;
        OP_LOGD(opName,
            "check sparse feature: preTokens[%ld], nextTokens[%ld], s1[%ld], s2[%ld], attenMaskExistFlag[%d]",
            preTokens, nextTokens, s1Size, s2Size, attenMaskExistFlag);
        if (attenMaskExistFlag == 1 && s1Size == s2Size) {
            if (preTokens >= s1Size && nextTokens == 0) {
                sparseType = SparseEnum::CAUSAL;
                preTokens = s1Size; // if sparse type is causal, template always need preTokens equal to s1Size
            } else if (preTokens > 0 && preTokens < s1Size - 1 && nextTokens > 0 && nextTokens < s2Size - 1) {
                sparseType = SparseEnum::BAND;
            } else if (preTokens > 0 && preTokens < s1Size - 1 && nextTokens == 0) {
                sparseType = SparseEnum::BANDX1;
            }
        }
        if (sparseMode == static_cast<int64_t>(SparseEnum::PREFIX)) {
            sparseType = GetPrefixNList();
        }
        inputParams.set_sparseType(static_cast<uint8_t>(sparseType));
        inputParams.set_preTokens(preTokens);
        inputParams.set_nextTokens(nextTokens);
    }

    bool CalcUBSize(const BufferNum &bufferNum, int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
        int64_t tmpDBasicBlock, int64_t batch) override
    {
        apiMaxUBSize = tmpS1BasicBlock * tmpS2BasicBlock * sizeof(float) + SOFTMAX_EXTRA_SIZE;
        return true;
    }

    void GetBufferNum(BufferNum &bufferNum) const override
    {
        bufferNum.bufferS1S2Num = S1D_HIGH_PERF_BUFFER_NUM;
    }

    void CalcS1S2BasicBlock(const BufferNum &bufferNum) override
    {
        s1BasicBlock = std::min(64L, alignedS1);
        s2BasicBlock = std::min(128L, alignedS2);
    }

    void CalcDBasicBlock(const BufferNum &bufferNum) override
    {
        dBasicBlock = std::min(128L, alignedD);
    }

    bool SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm1) override
    {
        bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, true);
        bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1OutDtype);
        bmm1.SetShape(std::min(tmpS1BasicBlock, s1Size), std::min(tmpS2BasicBlock, s2Size), dSize);
        bmm1.SetOrgShape(s1Size, s2Size, s1StrideSize, s2StrideSize);
        bmm1.SetBias(false);
        bmm1.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize);
        bmm1.SetFixSplit(tmpS1BasicBlock, tmpS2BasicBlock);
        return true;
    }

    bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm2) override
    {
        bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2OutDtype);
        bmm2.SetShape(std::min(tmpS1BasicBlock, s1Size), std::min(tmpDBasicBlock, dSize), s2Size);
        bmm2.SetOrgShape(s1Size, s1StrideSize, s2Size, s2StrideSize);
        bmm2.SetBias(false);
        bmm2.SetFixSplit(tmpS1BasicBlock, tmpDBasicBlock);
        return true;
    }

    void SetSoftMaxTiling() override
    {
        auto softmaxShape = Shape({ 8 * 1024 / alignedS2, alignedS2 });
        AscendC::SoftMaxTilingFunc(softmaxShape, sizeof(float), apiMaxUBSize, tilingData.softmaxTilingData);
    }

    bool IsTemplateMatched() const override
    {
        if (s1Size != s2Size) {
            return false;
        }
        if (s2Size > S2SIZE_LIMIT || s2Size % IS_BASIC_BLOCK_NUM != 0) {
            return false;
        }
        if (s1Size * s2Size < MIN_SIZE_LIMIT) {
            return false;
        }
        return true;
    }

    uint64_t GetTilingKey() const override
    {
      return GET_TILINGKEY(AxisEnum::S1, AxisEnum::D, AxisEnum::NONE, tilingKeyDType, LayoutEnum::BSND,
          SparseEnum::ANY, PerformanceOrientedEnum::BIG_DOUBLE_BUFFER);
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        auto &coreParams = tilingData.coreParams;
        size_t *workspaces = context_->GetWorkspaceSizes(1);
        int64_t bmm1Size = CeilDivision(coreParams.get_nRatio() * coreParams.get_s1BaseSize() * s2Size, 256L) * 256L;
        int64_t bmm1AlignByetes = bmm1Size * calcTypeSize * 2;
        int64_t stage1AlignByetes = bmm1Size * inputDtypeBytes * 2;
        workspaces[0] = static_cast<size_t>((bmm1AlignByetes + stage1AlignByetes) * aivNum) + WORK_SPACE_RESERVE_SIZE;
        return ge::GRAPH_SUCCESS;
    }
};

class FlashAttentionScoreTilingSplitS1S2 : public FlashAttentionScoreTilingBase {
public:
    explicit FlashAttentionScoreTilingSplitS1S2(gert::TilingContext *context) : FlashAttentionScoreTilingBase(context)
    {
        expectTemplate.splitS1 = 1;
        expectTemplate.splitS2 = 1;
        templateName = "SplitS1S2";
    }
    ~FlashAttentionScoreTilingSplitS1S2() override = default;

protected:
    int64_t GetNRatio() override
    {
        return 8L;
    }

    void GetBufferNum(BufferNum &bufferNum) const override
    {
        bufferNum.bufferS1S2Num = HIGH_PERF_BUFFER_NUM;
    }

    void CalcS1S2BasicBlock(const BufferNum &bufferNum) override
    {
        s1BasicBlock = std::min(64L, alignedS1);
        s2BasicBlock = std::min(128L, alignedS2);
    }

    void CalcDBasicBlock(const BufferNum &bufferNum) override { dBasicBlock = std::min(128L, alignedD); }

    bool SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm1) override
    {
        bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, true);
        bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1OutDtype);
        bmm1.SetShape(std::min(tmpS1BasicBlock, s1Size), std::min(tmpS2BasicBlock, s2Size), dSize);
        bmm1.SetOrgShape(s1Size, tmpS2BasicBlock * tilingData.coreParams.get_nRatio(), s1StrideSize, s2StrideSize);
        bmm1.SetBias(false);
        bmm1.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize);
        bmm1.SetFixSplit(tmpS1BasicBlock, tmpS2BasicBlock);

        return true;
    }

    bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm2) override
    {
        bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2OutDtype);
        bmm2.SetShape(std::min(tmpS1BasicBlock, s1Size), std::min(tmpDBasicBlock, dSize),
                      std::min(tmpS2BasicBlock, s2Size));
        bmm2.SetOrgShape(s1Size, s2StrideSize, std::min(tmpS2BasicBlock, s2Size), s2StrideSize);
        bmm2.SetBias(false);
        bmm2.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize);
        bmm2.SetFixSplit(tmpS1BasicBlock, tmpDBasicBlock);

        return true;
    }

    uint64_t GetTilingKey() const override
    {
      // not care about layout in tiling key, pass BSND(enum value is 0)
      return GET_TILINGKEY(AxisEnum::S1, AxisEnum::S2, AxisEnum::NONE, tilingKeyDType, LayoutEnum::BSND,
                           SparseEnum::ANY);
    }

    bool IsCapable() override
    {
      if (dSize > HIGH_PERF_SUPPORT_D_LIMIT) {
        return false;
      }
      return true;
    }

    bool CalcUBSize(const BufferNum& bufferNum, int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                    int64_t tmpDBasicBlock, int64_t batch) override
    {
      apiMaxUBSize = HIGH_PERF_API_BUFFER_MULTIPLE * tmpS1BasicBlock * tmpS2BasicBlock * sizeof(float);
      return true;
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        auto &tensorSizeParams = tilingData.tensorSizeParams;
        auto &coreParams = tilingData.coreParams;

        size_t* workspaces = context_->GetWorkspaceSizes(1);
        int64_t bmm1Byetes = coreParams.get_nRatio() * tensorSizeParams.get_bmm1ResUbSize() * calcTypeSize;
        int64_t bmm2Byetes = tensorSizeParams.get_bmm1ResUbSize() * calcTypeSize;
        workspaces[0] = static_cast<size_t>((bmm1Byetes * HIGH_PERF_WORKSPACE_MULTIPLE + bmm2Byetes) * aivNum) +
                        WORK_SPACE_RESERVE_SIZE;
        return ge::GRAPH_SUCCESS;
    }
};

class FlashAttentionScoreTilingSplitS1S2HighPerformance : public FlashAttentionScoreTilingBase {
public:
    explicit FlashAttentionScoreTilingSplitS1S2HighPerformance(gert::TilingContext *context) : FlashAttentionScoreTilingBase(context)
    {
        expectTemplate.splitS1 = 1;
        expectTemplate.splitS2 = 1;
        templateName = "SplitS1S2HighPerformance";
    }
    ~FlashAttentionScoreTilingSplitS1S2HighPerformance() override = default;

protected:
    int64_t GetNRatio() override
    {
        return 4L;
    }

    void GetBufferNum(BufferNum &bufferNum) const override
    {
        bufferNum.bufferS1S2Num = HIGH_PERF_BUFFER_NUM;
    }

    void CalcS1S2BasicBlock(const BufferNum &bufferNum) override
    {
        s1BasicBlock = std::min(64L, alignedS1);
        s2BasicBlock = std::min(128L, alignedS2);
    }

    void CalcDBasicBlock(const BufferNum &bufferNum) override { dBasicBlock = std::min(128L, alignedD); }

    bool SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm1) override
    {
        bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, true);
        bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1OutDtype);
        bmm1.SetShape(std::min(tmpS1BasicBlock, s1Size), std::min(tmpS2BasicBlock, s2Size), dSize);
        bmm1.SetOrgShape(s1Size, tmpS2BasicBlock * tilingData.coreParams.get_nRatio(), s1StrideSize, s2StrideSize);
        bmm1.SetBias(false);
        bmm1.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize);
        bmm1.SetFixSplit(tmpS1BasicBlock, tmpS2BasicBlock);

        return true;
    }

    bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm2) override
    {
        bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2OutDtype);
        bmm2.SetShape(std::min(tmpS1BasicBlock, s1Size), std::min(tmpDBasicBlock, dSize),
                      std::min(tmpS2BasicBlock, s2Size));
        bmm2.SetOrgShape(s1Size, s2StrideSize, std::min(tmpS2BasicBlock, s2Size), s2StrideSize);
        bmm2.SetBias(false);
        bmm2.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize);
        bmm2.SetFixSplit(tmpS1BasicBlock, tmpDBasicBlock);

        return true;
    }

    uint64_t GetTilingKey() const override
    {
      // not care about layout in tiling key, pass BSND(enum value is 0)
      return GET_TILINGKEY(AxisEnum::S1, AxisEnum::S2, AxisEnum::NONE, tilingKeyDType, LayoutEnum::BSND,
                           SparseEnum::ANY, PerformanceOrientedEnum::BIG_BUFFER);
    }

    bool IsCapable() override
    {
      if (dSize > HIGH_PERF_SUPPORT_D_LIMIT) {
        return false;
      }
    
      // 16对齐
      if (dSize % 16 != 0) {
        return false;
      }

      if (s2Size % HIGH_PERF_SUPPORT_D_LIMIT != 0) {
        return false;
      }
      return true;
    }

    bool CalcUBSize(const BufferNum& bufferNum, int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                    int64_t tmpDBasicBlock, int64_t batch) override
    {
      apiMaxUBSize = HIGH_PERF_API_BUFFER_MULTIPLE * tmpS1BasicBlock * tmpS2BasicBlock * sizeof(float);
      return true;
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        auto &tensorSizeParams = tilingData.tensorSizeParams;
        auto &coreParams = tilingData.coreParams;

        size_t *workspaces = context_->GetWorkspaceSizes(1);
        int64_t bmm1Byetes = coreParams.get_nRatio() * tensorSizeParams.get_bmm1ResUbSize() * calcTypeSize;
        int64_t bmm2Byetes = tensorSizeParams.get_bmm2ResUbSize() * calcTypeSize;
        workspaces[0] = static_cast<size_t>((bmm1Byetes * HIGH_PERF_WORKSPACE_MULTIPLE + bmm2Byetes) * aivNum) + WORK_SPACE_RESERVE_SIZE;
        return ge::GRAPH_SUCCESS;
    }
};

class FlashAttentionScoreTilingSplitS1S2DBHighPerformance : public FlashAttentionScoreTilingBase {
public:
    explicit FlashAttentionScoreTilingSplitS1S2DBHighPerformance(gert::TilingContext *context) : FlashAttentionScoreTilingBase(context)
    {
        expectTemplate.splitS1 = 1;
        expectTemplate.splitS2 = 1;
        templateName = "SplitS1S2DBHighPerformance";
    }
    ~FlashAttentionScoreTilingSplitS1S2DBHighPerformance() override = default;

protected:
    const int64_t highPerfSupportS2Limit = 64;
    const int64_t minSizeLimit = 16384; // 128*128

    int64_t GetNRatio() override
    {
        return 8L;
    }

    void GetBufferNum(BufferNum &bufferNum) const override
    {
        bufferNum.bufferS1S2Num = HIGH_PERF_BUFFER_NUM;
    }

    void CalcS1S2BasicBlock(const BufferNum &bufferNum) override
    {
        s1BasicBlock = std::min(64L, alignedS1);
        // d轴为64
        if (dSize == 64) {
          s1BasicBlock = std::min(128L, alignedS1);
        }
        s2BasicBlock = std::min(128L, alignedS2);
    }

    void CalcDBasicBlock(const BufferNum &bufferNum) override { dBasicBlock = std::min(128L, alignedD); }

    bool SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm1) override
    {
        bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, true);
        bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1OutDtype);
        bmm1.SetShape(std::min(tmpS1BasicBlock, s1Size), std::min(tmpS2BasicBlock, s2Size), dSize);
        bmm1.SetOrgShape(s1Size, tmpS2BasicBlock * tilingData.coreParams.get_nRatio(), s1StrideSize, s2StrideSize);
        bmm1.SetBias(false);
        bmm1.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize);
        bmm1.SetFixSplit(tmpS1BasicBlock, tmpS2BasicBlock);

        return true;
    }

    bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm2) override
    {
        bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2OutDtype);
        bmm2.SetShape(std::min(tmpS1BasicBlock, s1Size), dSize,
                      std::min(tmpS2BasicBlock * tilingData.coreParams.get_nRatio(), s2Size));
        bmm2.SetOrgShape(s1Size, s2StrideSize, std::min(tmpS2BasicBlock * tilingData.coreParams.get_nRatio(), s2Size), s2StrideSize);
        bmm2.SetBias(false);
        bmm2.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize);
        bmm2.SetFixSplit(tmpS1BasicBlock, tmpDBasicBlock);

        return true;
    }

    uint64_t GetTilingKey() const override
    {
      // not care about layout in tiling key, pass BSND(enum value is 0)
      return GET_TILINGKEY(AxisEnum::S1, AxisEnum::S2, AxisEnum::NONE, tilingKeyDType, LayoutEnum::BSND,
                           SparseEnum::ANY, PerformanceOrientedEnum::BIG_DOUBLE_BUFFER);
    }

    bool IsCapable() override
    {
      // 16对齐
      if (dSize % 16 != 0) {
        return false;
      }

      if (s2Size % highPerfSupportS2Limit != 0) {
        return false;
      }

      if (s1Size * s2Size <= minSizeLimit) {
        return false;
      }

      return true;
    }

    bool IsTemplateMatched() const override { return true; }

    bool CalcUBSize(const BufferNum& bufferNum, int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                    int64_t tmpDBasicBlock, int64_t batch) override
    {
      apiMaxUBSize = HIGH_PERF_API_BUFFER_MULTIPLE * tmpS1BasicBlock * tmpS2BasicBlock * sizeof(float);
      return true;
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        auto &tensorSizeParams = tilingData.tensorSizeParams;
        auto &coreParams = tilingData.coreParams;

        size_t *workspaces = context_->GetWorkspaceSizes(1);
        int64_t bmm1Byetes = coreParams.get_nRatio() * tensorSizeParams.get_bmm1ResUbSize() * calcTypeSize;

        // dSize小于128的场景，无需切D， workspace占用较小
        if (dSize <= 128) {
          // stage1占用2倍的空间，stage2占用2倍的空间
          workspaces[0] = static_cast<size_t>((bmm1Byetes * 2 + 2 * coreParams.get_s1BaseSize() * dSize * calcTypeSize) * aivNum) +
                          WORK_SPACE_RESERVE_SIZE;
        } else {
          // 切D场景，stage1占用2倍的空间，stage2需要占用4倍空间
          workspaces[0] = static_cast<size_t>((bmm1Byetes * 2 + 4 * coreParams.get_s1BaseSize() * dSize * calcTypeSize) * aivNum) +
                          WORK_SPACE_RESERVE_SIZE;
        }

        return ge::GRAPH_SUCCESS;
    }

    void SetSoftMaxTiling() override
    {
        auto softmaxShape = Shape({s1BasicBlock / GetNRatio(), s2BasicBlock * GetNRatio()});

        AscendC::SoftMaxFlashV2TilingFunc(softmaxShape, calcTypeSize, sizeof(float), apiMaxUBSize,
                                          tilingData.softmaxFlashTilingData, true, IsBasicBlockInSoftMax(softmaxShape));
    }
};

class FlashAttentionScoreTilingDropMask : public FlashAttentionScoreTilingBase {
public:
    explicit FlashAttentionScoreTilingDropMask(gert::TilingContext *context) : FlashAttentionScoreTilingBase(context) {}
    ~FlashAttentionScoreTilingDropMask() override = default;

protected:
    ge::graphStatus DoOpTiling() override
    {
        auto &inputParams = tilingData.inputParams;
        auto &dropmaskParams = tilingData.dropmaskParams;
        if (inputParams.get_s2Size() % FRACTAL_NUM == 0) {
            dropmaskParams.set_shapeAlignFlag(1);
            return ge::GRAPH_PARAM_INVALID;
        }
        dropmaskParams.set_shapeAlignFlag(0);
        int64_t shapeTotalSize = inputParams.get_bSize() * inputParams.get_n2Size() * inputParams.get_gSize() *
                                 inputParams.get_s1Size() * inputParams.get_s2Size();
        int64_t ubCalFactor = 256;
        // 保证每核计算数据量256倍数
        int64_t shapeSplitCoreSize = CeilDivision(shapeTotalSize, ubCalFactor);
        int64_t shapeSingleCoreSize = CeilDivision(shapeSplitCoreSize, static_cast<int64_t>(aivNum));

        // 单次ub计算量为x个元素,空间占用：x/8 * 1 [1个 uint8]+ 2x * 2[2个fp16,select的src和res] + x * 1 [1个uint8] = 6x
        //  ub能计算的最大元素数, 向下256对齐
        int64_t baseUbCalSize = AlignDown(CeilDivision(static_cast<int64_t>(aicoreParams_.ubSize), 6L), ubCalFactor);
        baseUbCalSize = std::min(baseUbCalSize, shapeSingleCoreSize * ubCalFactor);  //
        // ub 的外层循环次数
        int64_t multiCoreFactorSize = CeilDivision(shapeSingleCoreSize * ubCalFactor, baseUbCalSize);

        dropmaskParams.set_shapeTotalSize(shapeTotalSize);
        dropmaskParams.set_multiCoreFactorSize(static_cast<int32_t>(multiCoreFactorSize));
        dropmaskParams.set_multiCoreTotalSize(CeilDivision(shapeSplitCoreSize * ubCalFactor, baseUbCalSize));
        dropmaskParams.set_baseUbCalSize(static_cast<int32_t>(baseUbCalSize));
        dropmaskParams.set_workspaceSize(shapeTotalSize);
        return ge::GRAPH_PARAM_INVALID;
    }

    void GetBufferNum(BufferNum &bufferNum) const override {}

    uint64_t GetTilingKey() const override { return 0UL; }
};

class FlashAttentionScoreTilingBenchmark : public FlashAttentionScoreTilingBase {
public:
    explicit FlashAttentionScoreTilingBenchmark(gert::TilingContext *context) : FlashAttentionScoreTilingBase(context)
    {
        expectTemplate.splitS1 = 1;
        expectTemplate.splitS2 = 1;
        templateName = "SplitS1S2DBHighPerformance";
    }
    ~FlashAttentionScoreTilingBenchmark() override = default;

protected:
    int64_t GetNRatio() override
    {
        return 8L;
    }

    void GetBufferNum(BufferNum &bufferNum) const override
    {
        bufferNum.bufferS1S2Num = HIGH_PERF_BUFFER_NUM;
    }

    void CalcS1S2BasicBlock(const BufferNum &bufferNum) override
    {
        s1BasicBlock = std::min(64L, alignedS1);
        int64_t specialDSize = 64;
        if (dSize == specialDSize) {
          s1BasicBlock = std::min(128L, alignedS1);
        }
        s2BasicBlock = std::min(128L, alignedS2);
    }

    void CalcDBasicBlock(const BufferNum &bufferNum) override { dBasicBlock = std::min(128L, alignedD); }

    bool SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm1) override
    {
        bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, true);
        bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1OutDtype);
        bmm1.SetShape(std::min(tmpS1BasicBlock, s1Size), std::min(tmpS2BasicBlock, s2Size), dSize);
        bmm1.SetOrgShape(s1Size, tmpS2BasicBlock * tilingData.coreParams.get_nRatio(), s1StrideSize, s2StrideSize);
        bmm1.SetBias(false);
        bmm1.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize);
        bmm1.SetFixSplit(tmpS1BasicBlock, tmpS2BasicBlock);

        return true;
    }

    bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm2) override
    {
        bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2OutDtype);
        bmm2.SetShape(std::min(tmpS1BasicBlock, s1Size), std::min(tmpDBasicBlock, dSize),
                      std::min(tmpS2BasicBlock * tilingData.coreParams.get_nRatio(), s2Size));
        bmm2.SetOrgShape(s1Size, s2StrideSize, std::min(tmpS2BasicBlock * tilingData.coreParams.get_nRatio(), s2Size), s2StrideSize);
        bmm2.SetBias(false);
        bmm2.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize);
        bmm2.SetFixSplit(tmpS1BasicBlock, tmpDBasicBlock);

        return true;
    }

    uint64_t GetTilingKey() const override
    {
        // not care about layout in tiling key, pass BSND(enum value is 0)
        uint64_t benchmarkKey = 1071;
        return benchmarkKey;
    }

    bool IsCapable() override
    {
        auto inputParams = tilingData.inputParams;
        int64_t b = inputParams.get_bSize();
        int64_t n = inputParams.get_n2Size();
        int64_t s1 = inputParams.get_s1Size();
        int64_t s2 = inputParams.get_s2Size();
        int64_t preTokens = inputParams.get_preTokens();
        int64_t nextTokens = inputParams.get_nextTokens();
        int64_t d = inputParams.get_dSize();

        bool isFull = preTokens >= s1 && nextTokens >= s2;
        // special case for benchmark
        int64_t specialB = 24;
        int64_t n9K = 5;
        int64_t n2K = 10;
        int64_t s9K = 9216;
        int64_t s2K = 2304;
        int64_t specialD = 64;

        size_t layoutLen = strlen(inputLayout);
        if (layoutLen == 4UL) {
          if (inputLayout[0] == 'B' && inputLayout[1] == 'N' && inputLayout[2] == 'S' && inputLayout[3] == 'D') {
            return false;
          }
        }

        if (isFull && isHighPercision && b == specialB && n == n9K && s1 == s9K && s2 == s9K && d == specialD) {
          return true;
        }

        if (isFull && isHighPercision && b == specialB && n == n2K && s1 == s2K && s2 == s2K && d == specialD) {
          return true;
        }

          return false;
    }

    bool CalcUBSize(const BufferNum& bufferNum, int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                    int64_t tmpDBasicBlock, int64_t batch) override
    {
        apiMaxUBSize = HIGH_PERF_API_BUFFER_MULTIPLE * tmpS1BasicBlock * tmpS2BasicBlock * sizeof(float);
        return true;
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        auto &tensorSizeParams = tilingData.tensorSizeParams;
        auto &coreParams = tilingData.coreParams;

        size_t *workspaces = context_->GetWorkspaceSizes(1);
        int64_t bmm1Byetes = coreParams.get_nRatio() * tensorSizeParams.get_bmm1ResUbSize() * calcTypeSize;
        int64_t depths = 2;
        workspaces[0] = static_cast<size_t>((bmm1Byetes * depths +
            depths * tensorSizeParams.get_bmm1ResUbSize() * calcTypeSize) * aivNum) + WORK_SPACE_RESERVE_SIZE;
        return ge::GRAPH_SUCCESS;
    }

    void SetSoftMaxTiling() override
    {
        auto softmaxShape = Shape({s1BasicBlock / GetNRatio(), s2BasicBlock * GetNRatio()});

        AscendC::SoftMaxFlashV2TilingFunc(softmaxShape, calcTypeSize, sizeof(float), apiMaxUBSize,
                                          tilingData.softmaxFlashTilingData, true,
                                          IsBasicBlockInSoftMax(softmaxShape));
    }
};

class FlashAttentionScoreTilingSplitS1S2D : public FlashAttentionScoreTilingBase {
public:
    explicit FlashAttentionScoreTilingSplitS1S2D(gert::TilingContext *context) : FlashAttentionScoreTilingBase(context)
    {
        templateName = "SplitS1S2D";
        expectTemplate.splitS1 = 1;
        expectTemplate.splitS2 = 1;
        expectTemplate.splitD = 1;
    }
    ~FlashAttentionScoreTilingSplitS1S2D() override = default;

protected:
    bool AnalyzeDtype() override
    {
        OP_TILING_CHECK(!FlashAttentionScoreTilingBase::AnalyzeDtype(),
                        VECTOR_INNER_ERR_REPORT_TILIING(opName, "fail to anlyze base dtype"), return false);
        bmm2OutDtype = bmmDtype;
        return true;
    }

    bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling &bmm2) override
    {
        OP_TILING_CHECK(
            bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false) != 0,
            OP_LOGD(opName, "failed to set a type of bmm1"), return false);
        OP_TILING_CHECK(
            bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false) != 0,
            OP_LOGD(opName, "failed to set b type of bmm1"), return false);
        OP_TILING_CHECK(bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2OutDtype) != 0,
                        OP_LOGD(opName, "failed to set c type of bmm1"), return false);
        // m s1Size n dSize k s2Size
        OP_TILING_CHECK(bmm2.SetShape(std::min(tmpS1BasicBlock, s1Size), std::min(tmpDBasicBlock, dSize), s2Size) != 0,
                        OP_LOGD(opName, "failed to set shape of bmm1"), return false);
        OP_TILING_CHECK(bmm2.SetOrgShape(s1Size, s1StrideSize, s2Size, s2StrideSize) != 0,
                        OP_LOGD(opName, "failed to set org shape of bmm1"), return false);
        bmm2.SetBias(false);
        OP_TILING_CHECK(bmm2.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize) != 0,
                        OP_LOGD(opName, "failed to set buffer space of bmm1"), return false);
        OP_TILING_CHECK(bmm2.SetFixSplit(tmpS1BasicBlock, tmpDBasicBlock) != 0,
                        OP_LOGD(opName, "failed to set fix split of bmm1"), return false);
        return true;
    }

    int64_t CalcMinApiTmpSize(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t batch) override
    {
        auto tmpShape = Shape({tmpS1BasicBlock, tmpS2BasicBlock});
        maxSumTmpSize = GetMaxSumMinTmpSize(tmpShape, calcTypeSize);
        maxSumUpdateTmpSize = GetMaxSumUpdateMinTmpSize(tmpShape, calcTypeSize);
        customSoftmaxTmpSize = GetCustomSoftmaxMinTmpSize(tmpShape, calcTypeSize);
        int64_t dropoutTmpSize = 0L;
        if (dropMaskExistFlag == 1) {
            dropoutTmpSize = AscendC::GetDropOutMinTmpSize(tmpShape, calcTypeSize, true);
        }
        int64_t attenMaskTmpSize = 0L;
        if (attenMaskExistFlag == 1) {
            attenMaskTmpSize = AscendC::GetSelectWithBytesMaskMinTmpSize(tmpShape, Shape({1}), calcTypeSize, tmpShape,
                                                                         sizeof(uint8_t), false);
        }
        return std::max({maxSumTmpSize, maxSumUpdateTmpSize, customSoftmaxTmpSize, dropoutTmpSize, attenMaskTmpSize});
    }

    void GetBufferNum(BufferNum &bufferNum) const override
    {
        auto bufferS1S2NumTQueInStage1 = 0;
        if (pseExistFlag != 0 || attenMaskExistFlag != 0) {
            bufferS1S2NumTQueInStage1 += 1;
        }
        auto bufferS1S2NumTQueOutStage1 = 1;
        if (inputDtype == ge::DT_BF16 || isHighPercision) {
            bufferS1S2NumTQueOutStage1 += 1;
        }
        auto bufferS1S2NumTBufStage1 = 0;
        if (inputDtype == ge::DT_BF16 || isHighPercision) {
            bufferS1S2NumTBufStage1 += 1;
        }

        auto bufferS1S2NumTQueInStage2 = 1;
        if (inputDtype == ge::DT_BF16 || isHighPercision) {
            bufferS1S2NumTQueInStage2 += 1;
        }
        if (dropMaskExistFlag) {
            bufferS1S2NumTQueInStage2 += 1;
        }
        auto bufferS1S2NumTQueOutStage2 = 1;
        auto bufferS1S2NumTBufStage2 = 0;
        if (inputDtype == ge::DT_BF16 || isHighPercision) {
            bufferS1S2NumTBufStage2 += 2;
        }

        bufferNum.bufferS1S2Num = std::max(bufferS1S2NumTQueInStage1, bufferS1S2NumTQueInStage2) +
                                  std::max(bufferS1S2NumTQueOutStage1, bufferS1S2NumTQueOutStage2) +
                                  std::max(bufferS1S2NumTBufStage1, bufferS1S2NumTBufStage2);
        bufferNum.bufferS1DNum = 0;
        bufferNum.bufferExpNum = 0;

        OP_LOGD(opName, "bufferNum %d %d %d", bufferNum.bufferS1S2Num, bufferNum.bufferS1DNum, bufferNum.bufferExpNum);
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        auto &tensorSizeParams = tilingData.tensorSizeParams;
        auto &coreParams = tilingData.coreParams;
        auto &multiCoreParams = tilingData.multiCoreParams;

        size_t *workspaces = context_->GetWorkspaceSizes(1);
        auto usedCoreNum =
            std::min(multiCoreParams.get_totalSize(), static_cast<int64_t>(multiCoreParams.get_coreNum()));
        // 空间说明：
        // section0: mm1 n比1 * usedCoreNum
        // section1: stage1 mm1 copy out / stage copy out (float16/float32) * usedCoreNum
        // section2: stage2 copy out (bfloat16/float16 高精度) * usedCoreNum
        auto s1S2NRation = coreParams.get_nRatio() * tensorSizeParams.get_bmm1ResUbSize();
        auto userWorkspaceBytes = s1S2NRation * calcTypeSize * usedCoreNum;

        tensorSizeParams.set_wkspSection1OffsetBytes(userWorkspaceBytes);
        size_t s1S2AlignCalcTBytes =
            s1BasicBlock * AlignUp(static_cast<size_t>(s2Size * calcTypeSize), static_cast<size_t>(BYTE_BLOCK));
        userWorkspaceBytes += s1S2AlignCalcTBytes * usedCoreNum;

        if (inputDtype == ge::DT_BF16 || isHighPercision) {
            tensorSizeParams.set_wkspSection2OffsetBytes(userWorkspaceBytes);
            size_t s1S2AlignInputTBytes =
                s1BasicBlock * AlignUp(static_cast<size_t>(s2Size * inputDtypeBytes), static_cast<size_t>(BYTE_BLOCK));
            userWorkspaceBytes += s1S2AlignInputTBytes * usedCoreNum;
        }

        // reserved for framework
        workspaces[0] = userWorkspaceBytes + WORK_SPACE_RESERVE_SIZE;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoLibApiTiling() override
    {
        OP_TILING_CHECK(!SetMatMulTiling(s1BasicBlock, s2BasicBlock, dBasicBlock),
                        VECTOR_INNER_ERR_REPORT_TILIING(opName, "fail to get tiling of matmul"),
                        return ge::GRAPH_FAILED);

        uint32_t inputDtypeOfMaxSumBytes = 4;
        if (inputDtype == ge::DT_FLOAT16 && !isHighPercision) {
            inputDtypeOfMaxSumBytes = 2; // 2:高性能模式下的float16输入字节数
        }

        OP_TILING_CHECK(!MaxSumTilingFunc(ge::Shape({s1BasicBlock, s2BasicBlock}), inputDtypeOfMaxSumBytes,
                                          apiMaxUBSize, tilingData.maxSumTiling),
                        VECTOR_INNER_ERR_REPORT_TILIING(opName, "fail to get tiling of max sum"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(!MaxSumUpdateTilingFunc(ge::Shape({s1BasicBlock, s2BasicBlock}), inputDtypeOfMaxSumBytes,
                                                apiMaxUBSize, tilingData.maxSumUpdateTiling),
                        VECTOR_INNER_ERR_REPORT_TILIING(opName, "fail to get tiling of max sum update"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(!CustomSoftmaxTilingFunc(ge::Shape({s1BasicBlock, s2BasicBlock}), inputDtypeOfMaxSumBytes,
                                                 apiMaxUBSize, tilingData.customSoftmaxTiling),
                        VECTOR_INNER_ERR_REPORT_TILIING(opName, "fail to get tiling of custom softmax"),
                        return ge::GRAPH_FAILED);

        return ge::GRAPH_SUCCESS;
    }

    bool IsTemplateMatched() const override
    {
        // default template, only check S2
        if (actualTemplate.splitS1 == 0 || actualTemplate.splitS2 == 0 || actualTemplate.splitD == 0) {
            OP_LOGI(opName, "[%s]match default template, actual split policy[%s]", templateName,
                    actualTemplate.ToString().c_str());
        }
        return true;
    }

    uint64_t GetTilingKey() const override
    {
        // not care about layout in tiling key, pass BSND(enum value is 0)
        return GET_TILINGKEY(AxisEnum::S2, AxisEnum::D, AxisEnum::NONE, tilingKeyDType, LayoutEnum::BSND,
                             SparseEnum::ANY);
    }

    void SetTensorSizeParams() override
    {
        FlashAttentionScoreTilingBase::SetTensorSizeParams();
        auto &tensorSizeParams = tilingData.tensorSizeParams;
        if (inputDtype == ge::DT_BF16 || isHighPercision) {
            tensorSizeParams.set_inputQueBytes(batchBasic * s1BasicBlock * s2BasicBlock * calcTypeSize);
            tensorSizeParams.set_outputQueBytes(batchBasic * s1BasicBlock * s2BasicBlock * calcTypeSize);
        } else {
            tensorSizeParams.set_inputQueBytes(batchBasic * s1BasicBlock * s2BasicBlock * inputDtypeBytes);
            tensorSizeParams.set_outputQueBytes(batchBasic * s1BasicBlock * s2BasicBlock * inputDtypeBytes);
        }

        int32_t softmaxOutQueBytes =
            (tensorSizeParams.get_softmaxMaxUbSize() + tensorSizeParams.get_softmaxSumUbSize()) * sizeof(float);

        tensorSizeParams.set_tmpBufBytes(aicoreParams_.ubSize - tensorSizeParams.get_inputQueBytes() -
                                         tensorSizeParams.get_outputQueBytes() - softmaxOutQueBytes);

        auto usedTmpBufBytes = 0;
        // stage1
        tensorSizeParams.set_maxSumApiOffsetBytes(usedTmpBufBytes);
        tensorSizeParams.set_maxSumApiSize((tensorSizeParams.get_tmpBufBytes() - usedTmpBufBytes) / sizeof(float));
        tensorSizeParams.set_pseTbufOffsetBytes(usedTmpBufBytes);
        if (attenMaskExistFlag == 1) {
            tensorSizeParams.set_attenMaskApiOffsetBytes(usedTmpBufBytes);
            tensorSizeParams.set_attenMaskApiSize((tensorSizeParams.get_tmpBufBytes() - usedTmpBufBytes) /
                                                  sizeof(uint8_t));
        }

        // stage2
        usedTmpBufBytes = 0;
        if (inputDtype == ge::DT_BF16 || isHighPercision) {
            tensorSizeParams.set_bmm1ProcessTInStage2OffsetBytes(usedTmpBufBytes);
            usedTmpBufBytes += batchBasic * s1BasicBlock * s2BasicBlock * calcTypeSize;
            tensorSizeParams.set_bmm1ProcessTInStage2Size(batchBasic * s1BasicBlock * s2BasicBlock);
        }
        tensorSizeParams.set_customSoftmaxApiOffsetBytes(usedTmpBufBytes);
        tensorSizeParams.set_customSoftmaxApiSize((tensorSizeParams.get_tmpBufBytes() - usedTmpBufBytes) /
                                                  sizeof(float));
        tensorSizeParams.set_dropoutApiOffsetBytes(usedTmpBufBytes);
        tensorSizeParams.set_dropoutApiSize((tensorSizeParams.get_tmpBufBytes() - usedTmpBufBytes) / sizeof(float));
    }

    void GetSparseInfo() override
    {
        auto &inputParams = tilingData.inputParams;

        SparseEnum sparseType = SparseEnum::ALL;
        OP_LOGD(opName,
                "check sparse feature: preTokens[%ld], nextTokens[%ld], s1[%ld], s2[%ld], attenMaskExistFlag[%d]",
                preTokens, nextTokens, s1Size, s2Size, attenMaskExistFlag);
        if (attenMaskExistFlag == 1) {
            preTokens = std::min(s1Size - 1, preTokens);
            nextTokens = std::min(s2Size - 1, nextTokens);
            if (preTokens == s1Size - 1 && nextTokens == 0) {
                sparseType = SparseEnum::CAUSAL;
            } else if (preTokens < s1Size - 1 || nextTokens < s2Size - 1) {
                sparseType = SparseEnum::BAND;
            }

            if (sparseMode == static_cast<int64_t>(SparseEnum::CAUSAL)) {
                sparseType = SparseEnum::CAUSAL;
            } else if (sparseMode == static_cast<int64_t>(SparseEnum::PREFIX)) {
                sparseType = GetPrefixNList();
            }
        }

        inputParams.set_sparseType(static_cast<uint8_t>(sparseType));
        inputParams.set_preTokens(preTokens);
        inputParams.set_nextTokens(nextTokens);
    }

private:
    int64_t maxSumTmpSize = 0L;
    int64_t maxSumUpdateTmpSize = 0L;
    int64_t customSoftmaxTmpSize = 0L;
};

REGISTER_TILING_TEMPLATE("FlashAttentionScore", FlashAttentionScoreTilingBenchmark, 97);
REGISTER_TILING_TEMPLATE("FlashAttentionScore", FlashAttentionScoreTilingSplitS1S2DBHighPerformance, 98);
REGISTER_TILING_TEMPLATE("FlashAttentionScore", FlashAttentionScoreTilingSplitS1S2HighPerformance, 99);
REGISTER_TILING_TEMPLATE("FlashAttentionScore", FlashAttentionScoreTilingSplitD, 100);
REGISTER_TILING_TEMPLATE("FlashAttentionScore", FlashAttentionScoreTilingSplitS1D, 101);
REGISTER_TILING_TEMPLATE("FlashAttentionScore", FlashAttentionScoreTilingSplitS1S2, 102);
REGISTER_TILING_TEMPLATE("FlashAttentionScore", FlashAttentionScoreTilingSplitS1S2D, 103);
}  // namespace FA
}  // namespace optiling
