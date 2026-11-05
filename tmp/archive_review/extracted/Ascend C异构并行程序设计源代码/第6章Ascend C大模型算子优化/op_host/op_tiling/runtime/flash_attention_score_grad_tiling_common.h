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

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FLASHATTENTIONSCOREGRAD_COMMON_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FLASHATTENTIONSCOREGRAD_COMMON_H_
#include <cstdint>
#include "register/tilingdata_base.h"
namespace optiling {
constexpr int64_t BYTE_PER_BLOCK = 32;  // 32 B in block
constexpr int HIGH_PRECISION = 0;
constexpr int HIGH_PERFORMANCE = 1;

constexpr const char *BSH_STR = "BSH";
constexpr const char *SBH_STR = "SBH";
constexpr const char *BNSD_STR = "BNSD";
constexpr const char *BSND_STR = "BSND";
constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;
constexpr size_t DIM_2 = 2;
constexpr size_t DIM_3 = 3;
constexpr size_t DIM_4 = 4;
constexpr size_t LAST_AXIS_IDX = 1;
constexpr size_t SEC_LAST_AXIS_IDX = 2;
constexpr uint32_t MULT_BASE = 2;
constexpr uint32_t SOFTMAX_REMAIN_SIZE = 8 * 1024;
constexpr uint32_t API_RSDV_BUFFER_SIZE = 4 * 1024;
constexpr int64_t SINGLE_VEC_INST_DATASIZE = 256;
constexpr uint32_t DEFAULT_DATA_TYPE_SIZE = 4;
constexpr uint32_t DEFAULT_MASK = 64;
constexpr uint32_t FP16_DATA_TYPE_SIZE = 2;
constexpr uint32_t BF16_DATA_TYPE_SIZE = 2;
constexpr uint32_t FP16_MASK = 128;
constexpr uint32_t BF16_MASK = 64;
constexpr uint32_t FRACTAL_NUM = 16; // 16 is 分形大小
constexpr uint32_t CUBE_ALIGN_NUM = 16;  // 16 is cube align num
constexpr uint32_t BYTE_BLOCK = 32;  // 32 B in block
constexpr uint32_t BATCH_MAX_SIZE = 64;

enum TilingDataType {
    FP16 = 1,
    BF16 = 2,
    FP32 = 3,
    INHP
};

enum class InputLayout {
    BSH = 0,
    SBH = 1,
    BNSD = 2,
    BSND
};

enum AxisIdx {
    B = 0,
    S = 1,
    H = 2
};

enum Axis4Idx {
    AXIS4_B = B,
    AXIS4_S = S,
    AXIS4_N = 2,
    AXIS4_D = 3
};

/* layout和b,s,h三根轴的位置关系映射 */
const std::vector<std::vector<size_t>> LAYOUT_TO_AXIS {
    // 3根轴对应dimid
    {0, 1, 2}, // BSH
    {1, 0, 2}, // SBH
    // 4根轴对应dimid
    {0, 2, 1, 3}, // BNSD
    {0, 1, 2, 3} // BSND
};

enum InputIndex {
    QUERY = 0,
    KEY,
    VALUE,
    DY,
    PSE_SHIFT,
    DROP_MASK,
    PADDING_MASK,
    ATTEN_MASK,
    SOFTMAX_MAX,
    SOFTMAX_SUM,
    SOFTMAX_IN,
    ATTENTION_IN,
    PREFIX_N,
};

enum AttenMaskCompressMode {
    NO_COMPRESS_MODE = 0,
    LEFT_UP_CAUSAL_MODE,
    RIGHT_DOWN_CAUSAL_MODE,
    BAND_EQUAL_S_MODE
};

enum AttrIndex {
    SCALE_VALUE = 0,
    KEEP_PROB,
    PRE_TOKENS,
    NEXT_TOKENS,
    HEAD_NUM,
    INPUT_LAYOUT,
    INNER_PRECISE,
    SPARSE_MODE
};

enum PseShapeType {
    PSE_SHAPE_TYPE_BNSS,
    PSE_SHAPE_TYPE_BN1S
};

enum AttenDataType {
    ATTEN_MASK_TYPE_SAME = 0, // 0 表示 AttenMask 数据类型与 qkv 一致
    ATTEN_MASK_TYPE_U8_BOOL = 1 // 1 表示 AttenMask 数据类型为 u8 bool
};

enum AttenShapeType {
    ATTEN_MASK_SHAPE_TYPE_SS,
    ATTEN_MASK_SHAPE_TYPE_B1SS,
    ATTEN_MASK_SHAPE_TYPE_BNSS
};

enum SparseMode {
    NO_MASK = 0, // 未传入attenmask，不做mask操作
    ALL_MASK,
    LEFT_UP_CAUSAL,  // 左上角点划分的三角部分
    RIGHT_DOWN_CAUSAL = 3, // 右下角点划分的下三角部分
    BAND = 4,
    PREFIX = 5
};

constexpr uint32_t ATTEN_MASK_SHAPE_TEMP_DIMS = 0;  // 0 是 B1SS 及 SS差异轴索引, S不可能为 1

struct TempParams { // 频繁使用的中间态临时变量
    uint32_t usedUBSize;
    uint32_t tilingKey;
    uint32_t apiClcQueueSize = 0;
};

inline uint32_t Gcd(uint32_t a, uint32_t b) // a >= b
{
    if (b > a) {
        return Gcd(b, a);
    }
    if (a % b == 0) {
        return b;
    }
    return Gcd(b, a % b);
}

inline int64_t CeilCommon(int64_t num1, int64_t num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

inline int64_t Align(const int64_t n)
{
    return (n + BYTE_PER_BLOCK - 1) & (~(BYTE_PER_BLOCK - 1));
}

inline uint32_t AlignData(const uint32_t a, const uint32_t b) {
    if (b == 0) {
      return a;
    }
    return (a + b - 1) / b * b;
}

} // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_FLASHATTENTIONSCOREGRAD_COMMON_H_