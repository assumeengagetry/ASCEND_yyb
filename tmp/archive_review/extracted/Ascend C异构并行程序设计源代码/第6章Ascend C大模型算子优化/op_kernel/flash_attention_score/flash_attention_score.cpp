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

#include "kernel_operator.h"
#include "flash_attention_score_split_b_n_nd_to_nd_atten_mask_with_transpose.h"
#include "sparse_flash_attention_score_base.h"
#include "sparse_flash_attention_score_split_b_n_nd_to_nd_atten_mask_with_transpose.h"
#include "flash_attention_score_special_split_b_n_nd_to_nd_atten_mask_with_transpose.h"
#include "sparse_flash_attention_score_special_base.h"
#include "flash_attention_score_empty_input.h"
#include "flash_attention_score_split_b_bn2gs1s2d.h"
#include "flash_attention_score_split_bn2_n2gs1s2d.h"
#include "flash_attention_score_split_bn2gs1_s1s2_high_performance.h"
#include "flash_attention_score_split_bn2gs1_s1s2.h"
#include "flash_attention_score_split_bn2gs1_s1s2d.h"
#include "flash_attention_score_split_bn2gs1_s1d.h"
#include "flash_attention_score_benchmark.h"

#include "flash_attention_score_split_bn2gs1_s1s2_db_high_performance.h"
#include "flash_attention_score_split_bn2gs1_s1d_db_high_performance.h"

#define INVOKE_FA_GENERAL_OP_IMPL(templateClass, ...)                                                         \
    do {                                                                                                      \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGeneralTilingData, tilingDataIn, tiling);              \
        const FlashAttentionScoreGeneralTilingData *__restrict tilingData = &tilingDataIn;                    \
        bmm1tiling = &(tilingData->bmm1TilingData);                                                           \
        bmm2tiling = &(tilingData->bmm2TilingData);                                                           \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                                   \
        templateClass<__VA_ARGS__> op;                                                                        \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling);            \
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut, \
            attentionOut, user, tilingData, &tPipe);                                                          \
        op.Process();                                                                                         \
    } while (0)

extern "C" __global__ __aicore__ void flash_attention_score(__gm__ uint8_t *query, __gm__ uint8_t *key,
    __gm__ uint8_t *value, __gm__ uint8_t *pse, __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *prefix, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
    __gm__ uint8_t *softmaxOut, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    TPipe tPipe;
    set_mask_norm();
    SetSysWorkspace(workspace);

    if (TILING_KEY_IS(90) || TILING_KEY_IS(92) || TILING_KEY_IS(94)) {
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreTilingData, tiling_data_in, tiling);
        const FlashAttentionScoreTilingData *__restrict tiling_data = &tiling_data_in;
        if (TILING_KEY_IS(90)) {
            FlashAttentionScoreEmptyInput<half> op;
            op.Init(softmaxMax, softmaxSum, attentionOut, tiling_data);
            op.Process();
        } else if (TILING_KEY_IS(92)) {
            FlashAttentionScoreEmptyInput<float> op;
            op.Init(softmaxMax, softmaxSum, attentionOut, tiling_data);
            op.Process();
        } else {
            FlashAttentionScoreEmptyInput<bfloat16_t> op;
            op.Init(softmaxMax, softmaxSum, attentionOut, tiling_data);
            op.Process();
        }
        return;
    }

    const TCubeTiling *__restrict bmm1tiling;
    const TCubeTiling *__restrict bmm2tiling;

    if (TILING_KEY_IS(10000000000000200059UL)) { // SplitD: FLOAT16 BSND/BSH B
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBBn2gs1s2, half, half, false, LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000000210059UL)) { // SplitD: FLOAT16 SBH B
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBBn2gs1s2, half, half, false, LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000000220059UL)) { // SplitD: FLOAT16 BNSD B
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBBn2gs1s2, half, half, false, LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000000000201059UL)) { // SplitD: FLOAT32 BSND/BSH B
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBBn2gs1s2, float, float, false, LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000000211059UL)) { // SplitD: FLOAT32 SBH B
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBBn2gs1s2, float, float, false, LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000000221059UL)) { // SplitD: FLOAT32 BNSD B
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBBn2gs1s2, float, float, false, LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000000000202059UL)) { // SplitD: BF16 BSND/BSH B
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBBn2gs1s2, bfloat16_t, float, false, LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000000212059UL)) { // SplitD: BF16 SBH B
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBBn2gs1s2, bfloat16_t, float, false, LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000000222059UL)) { // SplitD: BF16 BNSD B
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBBn2gs1s2, bfloat16_t, float, false, LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000000000203059UL)) { // SplitD: FLOAT16_PRECISION BSND/BSH B
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBBn2gs1s2, half, float, false, LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000000213059UL)) { // SplitD: FLOAT16_PRECISION SBH B
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBBn2gs1s2, half, float, false, LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000000223059UL)) { // SplitD: FLOAT16_PRECISION BNSD B
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBBn2gs1s2, half, float, false, LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000000000200159UL)) { // SplitD: FLOAT16 BSND/BSH N2
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1s2N2gs1s2, half, half, false, LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000000210159UL)) { // SplitD: FLOAT16 SBH N2
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1s2N2gs1s2, half, half, false, LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000000220159UL)) { // SplitD: FLOAT16 BNSD N2
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1s2N2gs1s2, half, half, false, LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000000000201159UL)) { // SplitD: FLOAT32 BSND/BSH N2
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1s2N2gs1s2, float, float, false, LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000000211159UL)) { // SplitD: FLOAT32 SBH N2
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1s2N2gs1s2, float, float, false, LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000000221159UL)) { // SplitD: FLOAT32 BNSD N2
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1s2N2gs1s2, float, float, false, LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000000000202159UL)) { // SplitD: BF16 BSND/BSH N2
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1s2N2gs1s2, bfloat16_t, float, false, LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000000212159UL)) { // SplitD: BF16 SBH N2
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1s2N2gs1s2, bfloat16_t, float, false, LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000000222159UL)) { // SplitD: BF16 BNSD N2
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1s2N2gs1s2, bfloat16_t, float, false,
            LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000000000203159UL)) { // SplitD: FLOAT16_PRECISION BSND/BSH N2
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1s2N2gs1s2, half, float, false, LayoutMode::BSNGD);
        return;
    } else if (TILING_KEY_IS(10000000000000213159UL)) { // SplitD: FLOAT16_PRECISION SBH N2
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1s2N2gs1s2, half, float, false, LayoutMode::SBNGD);
        return;
    } else if (TILING_KEY_IS(10000000000000223159UL)) { // SplitD: FLOAT16_PRECISION BNSD N2
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1s2N2gs1s2, half, float, false, LayoutMode::BNGS1S2);
        return;
    } else if (TILING_KEY_IS(10000000000000200953UL)) { // SplitS1D: FLOAT16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1S1d, half);
        return;
    } else if (TILING_KEY_IS(10000000000000201953UL)) { // SplitS1D: FLOAT32
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1S1d, float);
        return;
    } else if (TILING_KEY_IS(10000000000000202953UL)) { // SplitS1D: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1S1d, bfloat16_t, float);
        return;
    } else if (TILING_KEY_IS(10000000000000203953UL)) { // SplitS1D: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1S1d, half, float);
        return;
    } else if (TILING_KEY_IS(10000000000000200943UL)) { // SplitS1S2: FLOAT16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1s2S1s2, half);
        return;
    } else if (TILING_KEY_IS(10000000000000201943UL)) { // SplitS1S2: FLOAT32
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1s2S1s2, float);
        return;
    } else if (TILING_KEY_IS(10000000000000202943UL)) { // SplitS1S2: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1s2S1s2, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000000203943UL)) { // SplitS1S2: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1s2S1s2, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000000200954UL)) { // SplitS1S2D: FLOAT16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1S1s2d, half);
        return;
    } else if (TILING_KEY_IS(10000000000000201954UL)) { // SplitS1S2D: FLOAT32
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1S1s2d, float);
        return;
    } else if (TILING_KEY_IS(10000000000000202954UL)) { // SplitS1S2D: BFLOAT16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1S1s2d, bfloat16_t, float);
        return;
    } else if (TILING_KEY_IS(10000000000000203954UL)) { // SplitS1S2D: FLOAT16 high precision
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1S1s2d, half, float);
        return;
    } else if (TILING_KEY_IS(10000000000001200943UL)) { // SplitS1S2HighPerf: FLOAT16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1s2S1s2HighPerf, half);
        return;
    } else if (TILING_KEY_IS(10000000000001201943UL)) { // SplitS1S2HighPerf: FLOAT32
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1s2S1s2HighPerf, float);
        return;
    } else if (TILING_KEY_IS(10000000000001202943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1s2S1s2HighPerf, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000001203943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1s2S1s2HighPerf, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000002200943UL)) { // SplitS1S2HighPerf: FLOAT16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf, half);
        return;
    } else if (TILING_KEY_IS(10000000000002201943UL)) { // SplitS1S2HighPerf: FLOAT32
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf, float);
        return;
    } else if (TILING_KEY_IS(10000000000002202943UL)) { // SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000002203943UL)) { // SplitS1S2HighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1s2S1s2DBHighPerf, half, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000002202953UL)) { // SplitBn2gs1s2S1dDBHighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf, bfloat16_t, float, true);
        return;
    } else if (TILING_KEY_IS(10000000000002203953UL)) { // SplitBn2gs1s2S1dDBHighPerf: FLOAT16_PRECISION
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreSplitBn2gs1s2S1dDBHighPerf, half, float, true);
        return;
    }

    /*
    拆解TilingData 数据；
    * */
    uint32_t tmp_block_idx = GetBlockIdx();
    GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreTilingData, tiling_data_in, tiling);
    const FlashAttentionScoreTilingData *__restrict tiling_data = &tiling_data_in;

    if (tmp_block_idx % tiling_data->attentionScoreCoreParams.s1 >= tiling_data->attentionScoreCoreParams.formerNum) {
        bmm1tiling = &(tiling_data->bmm1TilingDataTailCore);
        bmm2tiling = &(tiling_data->bmm2TilingDataTailCore);
    } else {
        bmm1tiling = &(tiling_data->bmm1TilingData);
        bmm2tiling = &(tiling_data->bmm2TilingData);
    }

    /*
    获取Op可用WorkSpace空间
    * */
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    if (TILING_KEY_IS(2)) {
        FlashAttentionScoreAttenND2NDWithTrans<half> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(1002)) {
        FlashAttentionScoreAttenND2NDWithTrans<half, float> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(52)) {
        FlashAttentionScoreAttenND2NDWithTrans<half, half, true> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(1052)) {
        FlashAttentionScoreAttenND2NDWithTrans<half, float, true> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(6)) {
        FlashAttentionScoreAttenND2NDWithTrans<float> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(56)) {
        FlashAttentionScoreAttenND2NDWithTrans<float, float, true> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(8)) {
        FlashAttentionScoreAttenND2NDWithTrans<bfloat16_t, float> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(58)) {
        FlashAttentionScoreAttenND2NDWithTrans<bfloat16_t, float, true> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        FlashAttentionScore<half> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(1003)) {
        FlashAttentionScore<half, float> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(53)) {
        FlashAttentionScore<half, half, true> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(1053)) {
        FlashAttentionScore<half, float, true> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(7)) {
        FlashAttentionScore<float> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(57)) {
        FlashAttentionScore<float, float, true> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(9)) {
        FlashAttentionScore<bfloat16_t, float> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(59)) {
        FlashAttentionScore<bfloat16_t, float, true> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(18)) {
        SparseFlashAttentionScoreAttenND2NDWithTrans<half> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(1018)) {
        SparseFlashAttentionScoreAttenND2NDWithTrans<half, float> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(68)) {
        SparseFlashAttentionScoreAttenND2NDWithTrans<half, half, true> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(1068)) {
        SparseFlashAttentionScoreAttenND2NDWithTrans<half, float, true> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(22)) {
        SparseFlashAttentionScoreAttenND2NDWithTrans<float> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(72)) {
        SparseFlashAttentionScoreAttenND2NDWithTrans<float, float, true> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(24)) {
        SparseFlashAttentionScoreAttenND2NDWithTrans<bfloat16_t, float> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(74)) {
        SparseFlashAttentionScoreAttenND2NDWithTrans<bfloat16_t, float, true> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(19)) {
        SparseFlashAttentionScoreBase<half> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(1019)) {
        SparseFlashAttentionScoreBase<half, float> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(69)) {
        SparseFlashAttentionScoreBase<half, half, true> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(1069)) {
        SparseFlashAttentionScoreBase<half, float, true> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(23)) {
        SparseFlashAttentionScoreBase<float> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(73)) {
        SparseFlashAttentionScoreBase<float, float, true> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(25)) {
        SparseFlashAttentionScoreBase<bfloat16_t, float> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(75)) {
        SparseFlashAttentionScoreBase<bfloat16_t, float, true> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(1060)) {
        FlashAttentionScoreAttenSpecialND2NDWithTrans<half, float, true> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(1061)) {
        SparseFlashAttentionScoreSpecialBase<bfloat16_t, float, true> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);
        op.Init(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax, softmaxSum, softmaxOut,
            attentionOut, user, tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(1071)) {
        INVOKE_FA_GENERAL_OP_IMPL(FlashAttentionScoreBenchmark, half, float, true);
        return;
    }
}
