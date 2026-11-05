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

#include "flash_attention_score_grad_split_b_n_nd_nd_nd.h"
#include "flash_attention_score_grad_split_b_n_nd_nd_nd_special.h"
#include "flash_attention_score_grad_split_b_n_s_nd_nd_nd.h"
#include "flash_attention_score_grad_split_b_n_nd_nd_nd_s1_s2.h"
#include "flash_attention_score_grad_split_b_n_s_nd_nd_nd_s1_s2.h"
#include "sparse_flash_attention_score_grad_split_b_n_nd_nd_nd.h"
#include "sparse_flash_attention_score_grad_split_b_n_s_nd_nd_nd.h"
#include "sparse_flash_attention_score_grad_split_b_n_s_nd_nd_nd_special.h"
#include "sparse_flash_attention_score_grad_split_b_n_s_nd_nd_nd_f16_special.h"
#include "sparse_flash_attention_score_grad_split_b_n_s_nd_nd_nd_special_square.h"
#include "flash_attention_score_grad_empty.h"
#include "flash_attention_score_grad_cast.h"
#include "flash_attention_score_grad_us1s2_bbn2gs1s2.h"
#include "flash_attention_score_grad_us1s2_bbn2gs1s2_pre.h"
#include "flash_attention_score_grad_us1s2_bbn2gs1s2_post.h"
#include "flash_attention_score_grad_u_s1s2_b_bn2_dropout.h"
#include "flash_attention_score_grad_u_s1s2_b_bn2_cast.h"
#include "flash_attention_score_grad_u_s1s2_b_bn2.h"
#include "flash_attention_score_grad_bng_gs1s2.h"
#include "flash_attention_score_grad_Ubn2_Bb.h"
#include "sparse_flash_attention_score_grad_us1s2_bbn.h"
#include "flash_attention_score_grad_u_s1s2_b_bn_dropout.h"
#include "sparse_flash_attention_score_grad_us1s2_bbn_post.h"
#include "flash_attention_score_grad_g_bn.h"
#include "flash_attention_score_grad_ngs1s2_bn.h"
#include "flash_attention_score_grad_ngs1s2_bn_dropout.h"
#include "flash_attention_score_grad_ngs1s2_bn_muls.h"

#define INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(INPUT_TYPE)                                                           \
    do {                                                                                                            \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2, tiling_data_in, tiling_data);  \
        const FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2* __restrict tilingData = &tiling_data_in;             \
        const TCubeTiling* __restrict bmm1tiling = &(tilingData->mm1TilingData);                                    \
        const TCubeTiling* __restrict bmm2tiling = &(tilingData->mm2TilingData);                                    \
        const TCubeTiling* __restrict bmm3tiling = &(tilingData->mm3TilingData);                                    \
        FlashAttentionScoreGradS1S2BNGS1S2Pre<INPUT_TYPE, float> opPre;                                             \
        opPre.Init(dq, dk, dv, drop_mask, user, tilingData, &pipeIn);                                               \
        opPre.Process();                                                                                            \
        opPre.SyncALLCores();                                                                                       \
        pipeIn.Destroy();                                                                                           \
                                                                                                                    \
        TPipe pipeBase;                                                                                             \
        FlashAttentionScoreGradUs1s2Bbn2gs1s2<INPUT_TYPE, float> op;                                                \
        REGIST_MATMUL_OBJ(&pipeBase, GetSysWorkSpacePtr(),                                                          \
                        op.mm1, bmm1tiling,                                                                         \
                        op.mm2, bmm1tiling,                                                                         \
                        op.mm3, bmm2tiling,                                                                         \
                        op.mm4, bmm3tiling);                                                                        \
        op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum,    \
                dq, dk, dv, dpse, user, tilingData, &pipeBase);                                                     \
        op.Process();                                                                                               \
        op.SyncALLCores();                                                                                          \
        pipeBase.Destroy();                                                                                         \
        TPipe pipePost;                                                                                             \
        FlashAttentionScoreGradS1S2BNGS1S2Post<INPUT_TYPE, float> opPost;                                           \
        opPost.Init(dq, dk, dv, user, tilingData, &pipePost);                                                       \
        opPost.Process();                                                                                           \
    } while (0)

#define INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(INPUT_TYPE)                                                             \
  do {                                                                                                           \
    GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingDataUs1s2Bbn2, tiling_data_in, tiling_data);        \
    const FlashAttentionScoreGradTilingDataUs1s2Bbn2* __restrict tilingData = &tiling_data_in;                   \
    if (g_coreType == AIV && tilingData->opInfo.keepProb < 1 &&                                                  \
        tilingData->opInfo.S2 % tilingData->splitCoreParams.baseN != 0) {                                        \
      FAGUs1s2Bbn2Dropout opDropout;                                                                             \
      opDropout.Init(drop_mask, user, tilingData, &pipeIn);                                                      \
      opDropout.Process();                                                                                       \
    }                                                                                                            \
    pipeIn.Destroy();                                                                                            \
    TPipe pipeOp;                                                                                                \
    const TCubeTiling* __restrict bmm1tiling = &(tilingData->mm1TilingData);                                     \
    const TCubeTiling* __restrict bmm31tiling = &(tilingData->mm31TilingData);                                   \
    const TCubeTiling* __restrict bmm4tiling = &(tilingData->mm4TilingData);                                     \
    FlashAttentionScoreGradUs1s2Bbn<INPUT_TYPE, float> op;                                                       \
    REGIST_MATMUL_OBJ(&pipeOp, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm1tiling, op.mm4, bmm4tiling, \
                      op.mm3_1, bmm31tiling);                                                                    \
    op.Init(query, key, value, dy, pse_shift, drop_mask, padding_mask, atten_mask, softmax_max, softmax_sum,     \
            softmax_in, attention_in, dq, dk, dv, dpse, user, tilingData, &pipeOp);                              \
    op.Process();                                                                                                \
    pipeOp.Destroy();                                                                                            \
    TPipe pipeCast;                                                                                              \
    FlashAttentionScoreGradUs1s2BbnCast<INPUT_TYPE> opCast;                                                      \
    opCast.Init(dq, dk, dv, user, tilingData, &pipeCast);                                                        \
    opCast.Process();                                                                                            \
  } while (0)

#define INVOKE_FAG_GENERAL_BNG_GS1S2_IMPL(INPUT_TYPE)                                                                \
  do {                                                                                                               \
    GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingDataBNGgs1s2, tiling_data_in, tiling_data);             \
    const FlashAttentionScoreGradTilingDataBNGgs1s2* __restrict tilingData = &tiling_data_in;                        \
    const TCubeTiling* __restrict bmm1tiling = &(tilingData->mm1TilingData);                                         \
    const TCubeTiling* __restrict bmm2tiling = &(tilingData->mm2TilingData);                                         \
    const TCubeTiling* __restrict bmm3tiling = &(tilingData->mm3TilingData);                                         \
    FlashAttentionScoreGradSplitBNG<INPUT_TYPE, float, true> op;                                                     \
    REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm1tiling, op.mm3, bmm2tiling,     \
                      op.mm4, bmm3tiling);                                                                           \
    op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq, dk, \
            dv, user, tilingData, &pipeIn);                                                                          \
    op.Process();                                                                                                    \
    pipeIn.Destroy();                                                                                                \
  } while (0)

#define INVOKE_FAG_GENERAL_BNGS1S2_B_IMPL(INPUT_TYPE)                                                                \
  do {                                                                                                               \
    GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradUbngs1s2BbTilingData, tiling_data_in, tiling_data);           \
    const FlashAttentionScoreGradUbngs1s2BbTilingData* __restrict tilingData = &tiling_data_in;                      \
    const TCubeTiling* __restrict bmm1tiling = &(tilingData->mm1AndMm2TilingData);                                   \
    const TCubeTiling* __restrict bmm2tiling = &(tilingData->mm31TilingData);                                        \
    const TCubeTiling* __restrict bmm3tiling = &(tilingData->mm32AndMm4TilingData);                                  \
    FlashAttentionScoreGradUbn2Bb<INPUT_TYPE, float, LayoutMode::BSNGD> op;                                          \
    REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.m_bmm1, bmm1tiling, op.m_bmm2, bmm1tiling, op.m_bmm3,        \
                      bmm2tiling, op.m_bmm4, bmm3tiling);                                                            \
    op.Init(query, key, value, dy, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq, dk, \
            dv, dpse, user, tilingData, &pipeIn);                                                                    \
    op.Process();                                                                                                    \
    pipeIn.Destroy();                                                                                                \
  } while (0)

#define INVOKE_FAG_GENERAL_S1S2_BN_IMPL(INPUT_TYPE)                                                                 \
  do {                                                                                                              \
    GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradUs1s2BbnTilingData, tiling_data_in, tiling_data);            \
    const FlashAttentionScoreGradUs1s2BbnTilingData* __restrict tilingData = &tiling_data_in;                       \
    if (g_coreType == AIV && tilingData->baseParams.keepProb < 1) {                                                 \
      TPipe pipeInDp;                                                                                               \
      FAGUs1s2BbnDropout opDropout;                                                                                 \
      opDropout.Init(drop_mask, user, tilingData, &pipeInDp);                                                       \
      opDropout.Process();                                                                                          \
      pipeInDp.Destroy();                                                                                           \
    }                                                                                                               \
    pipeIn.Destroy();                                                                                               \
    TPipe pipeOp;                                                                                                   \
    const TCubeTiling* __restrict bmm1tiling = &(tilingData->mm1TilingData);                                        \
    const TCubeTiling* __restrict bmm3tiling = &(tilingData->mm3TilingData);                                        \
    const TCubeTiling* __restrict bmm4tiling = &(tilingData->mm4TilingData);                                        \
    SparseFlashAttentionScoreGradUS1S2BBN<INPUT_TYPE, float, true> op;                                              \
    REGIST_MATMUL_OBJ(&pipeOp, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm1tiling, op.bmm3, bmm3tiling, \
                      op.bmm4, bmm4tiling);                                                                         \
    uint64_t bmm31Offset;                                                                                           \
    uint64_t bmm32Offset;                                                                                           \
    uint64_t bmm4Offset;                                                                                            \
    op.Init(query, key, value, dy, pse_shift, drop_mask, padding_mask, atten_mask, softmax_max, softmax_sum,        \
            softmax_in, attention_in, dq, dk, dv, dpse, user, tilingData, &pipeOp, &bmm31Offset, &bmm32Offset,      \
            &bmm4Offset);                                                                                           \
    op.Process();                                                                                                   \
    pipeOp.Destroy();                                                                                               \
    TPipe pipePost;                                                                                                 \
    FlashAttentionScoreGradUs1s2BbnPost<INPUT_TYPE, float, true> opPostScalar;                                      \
    opPostScalar.Init(dq, dk, dv, user, tilingData, &pipePost, bmm31Offset, bmm32Offset, bmm4Offset);               \
    opPostScalar.Process();                                                                                         \
    pipePost.Destroy();                                                                                             \
  } while (0)

#define INVOKE_FAG_GENERAL_G_BN_IMPL(INPUT_TYPE)                                                                     \
  do {                                                                                                               \
    GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingDataUgBbn, tiling_data_in, tiling_data);                \
    const FlashAttentionScoreGradTilingDataUgBbn* __restrict tilingData = &tiling_data_in;                           \
    const TCubeTiling* __restrict bmm1tiling = &(tilingData->mm1TilingData);                                         \
    const TCubeTiling* __restrict bmm2tiling = &(tilingData->mm2TilingData);                                         \
    const TCubeTiling* __restrict bmm3tiling = &(tilingData->mm3TilingData);                                         \
    FlashAttentionScoreGradUgBbn<INPUT_TYPE, float, true> op;                                                        \
    REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm1tiling, op.mm3, bmm2tiling,     \
                      op.mm4, bmm3tiling);                                                                           \
    op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq, dk, \
            dv, user, tilingData, &pipeIn);                                                                          \
    op.Process();                                                                                                    \
    pipeIn.Destroy();                                                                                                \
  } while (0)

#define INVOKE_FAG_GENERAL_S1S2_BNGS1S2_IMPL(INPUT_TYPE)                                                             \
  do {                                                                                                               \
    GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingDataUngs1s2Bbn, tiling_data_in, tiling_data);           \
    const FlashAttentionScoreGradTilingDataUngs1s2Bbn* __restrict tilingData = &tiling_data_in;                      \
    if (g_coreType == AIV && tilingData->opInfo.keepProb < 1) {                                                      \
        TPipe pipeInDp;                                                                                              \
        FAGNgs1s2Bn2Dropout opDropout;                                                                               \
        opDropout.Init(drop_mask, user, &tiling_data_in, &pipeInDp);                                                 \
        opDropout.Process();                                                                                         \
        pipeInDp.Destroy();                                                                                          \
    }                                                                                                                \
    pipeIn.Destroy();                                                                                                \
    TPipe pipeOp;                                                                                                    \
    const TCubeTiling* __restrict bmm1tiling = &(tilingData->mm1AndMm2TilingData);                                   \
    const TCubeTiling* __restrict bmm2tiling = &(tilingData->mm1AndMm2TilingData);                                   \
    const TCubeTiling* __restrict bmm3tiling = &(tilingData->mm31TilingData);                                        \
    const TCubeTiling* __restrict bmm4tiling = &(tilingData->mm32AndMm4TilingData);                                  \
    FlashAttentionScoreGradUngs1s2Bbn<INPUT_TYPE, float, true> op;                                                   \
    REGIST_MATMUL_OBJ(&pipeOp, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm2tiling, op.mm31, bmm3tiling,    \
                      op.mm32, bmm4tiling, op.mm4, bmm4tiling);                                                      \
    op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq, dk, \
            dv, user, tilingData, &pipeOp);                                                                          \
    op.Process();                                                                                                    \
    pipeOp.Destroy();                                                                                                \
    TPipe pipeMuls;                                                                                                  \
    FlashAttentionScoreGradUngs1s2BbnMuls<INPUT_TYPE> opMuls;                                                        \
    opMuls.Init(dq, dk, user, tilingData, &pipeMuls);                                                                \
    opMuls.Process();                                                                                                \
    pipeMuls.Destroy();                                                                                              \
  } while (0)

// implementation of kernel function
extern "C" __global__ __aicore__ void flash_attention_score_grad(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* dy, __gm__ uint8_t* pse_shift,
    __gm__ uint8_t* drop_mask, __gm__ uint8_t* padding_mask, __gm__ uint8_t* atten_mask, __gm__ uint8_t* softmax_max,
    __gm__ uint8_t* softmax_sum, __gm__ uint8_t* softmax_in, __gm__ uint8_t* attention_in, __gm__ uint8_t* prefix,
    __gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv, __gm__ uint8_t* dpse, __gm__ uint8_t* workspace,
    __gm__ uint8_t* tiling_data) {
  TPipe pipeIn;
  set_mask_norm();
  SetSysWorkspace(workspace);
  __gm__ uint8_t* user = GetUserWorkspace(workspace);

  if (TILING_KEY_IS(10000000000000003434UL)) {
    INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half);
    return;
  } else if (TILING_KEY_IS(10000000000000002434UL)) {
    INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t);
    return;
  } else if (TILING_KEY_IS(10000000000000020134UL)) {
    INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(half);
    return;
  } else if (TILING_KEY_IS(10000000000000022134UL)) {
    INVOKE_FAG_GENERAL_S1S2_BN2_IMPL(bfloat16_t);
    return;
  } else if (TILING_KEY_IS(10000000000000003012UL)) {
    INVOKE_FAG_GENERAL_BNG_GS1S2_IMPL(half);
    return;
  } else if (TILING_KEY_IS(10000000000000002012UL)) {
    INVOKE_FAG_GENERAL_BNG_GS1S2_IMPL(bfloat16_t);
    return;
  } else if (TILING_KEY_IS(10000000000000103099UL)) {
    INVOKE_FAG_GENERAL_BNGS1S2_B_IMPL(half);
    return;
  } else if (TILING_KEY_IS(10000000000000102099UL)) {
    INVOKE_FAG_GENERAL_BNGS1S2_B_IMPL(bfloat16_t);
    return;
  } else if (TILING_KEY_IS(10000000000000113234UL)) {
    INVOKE_FAG_GENERAL_S1S2_BN_IMPL(half);
    return;
  } else if (TILING_KEY_IS(10000000000000112234UL)) {
    INVOKE_FAG_GENERAL_S1S2_BN_IMPL(bfloat16_t);
    return;
  } else if (TILING_KEY_IS(10000000000000000129UL)) {
    INVOKE_FAG_GENERAL_G_BN_IMPL(half);
    return;
  } else if (TILING_KEY_IS(10000000000000002129UL)) {
    INVOKE_FAG_GENERAL_G_BN_IMPL(bfloat16_t);
    return;
  } else if (TILING_KEY_IS(10000000000000003199UL)) {
    INVOKE_FAG_GENERAL_S1S2_BNGS1S2_IMPL(half);
    return;
  } else if (TILING_KEY_IS(10000000000000002199UL)) {
    INVOKE_FAG_GENERAL_S1S2_BNGS1S2_IMPL(bfloat16_t);
    return;
  }

  GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingData, tiling_data_in, tiling_data);
  const FlashAttentionScoreGradTilingData* __restrict tilingData = &tiling_data_in;

  const TCubeTiling* __restrict bmm1tiling;
  const TCubeTiling* __restrict bmm2tiling;
  const TCubeTiling* __restrict bmm3tiling;

  bmm1tiling = &(tilingData->mm1TilingData);
  bmm2tiling = &(tilingData->mm2TilingData);
  bmm3tiling = &(tilingData->mm3TilingData);

  if (TILING_KEY_IS(1)) {
    FlashAttentionScoreGradBNNdNdNd<half, half> op;
    REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm1tiling, op.mm3, bmm2tiling);
    op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq, dk, dv,
            user, tilingData, &pipeIn);
    op.Process();
    pipeIn.Destroy();

    TPipe pipeCast;
    FlashAttentionScoreGradCast<half, half> opCast;
    opCast.Init(dq, dk, dv, user, tilingData, &pipeCast);
    opCast.Process();
  } else if (TILING_KEY_IS(10001)) {
    FlashAttentionScoreGradBNNdNdNdS1S2<half, half> op;
    REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm1tiling, op.mm3, bmm2tiling, op.mm4,
                      bmm3tiling);
    op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq, dk, dv,
            user, tilingData, &pipeIn);
    op.Process();
    pipeIn.Destroy();

    TPipe pipeCast;
    FlashAttentionScoreGradCast<half, half> opCast;
    opCast.Init(dq, dk, dv, user, tilingData, &pipeCast);
    opCast.Process();
  } else if (TILING_KEY_IS(3)) {
    FlashAttentionScoreGradBNSNdNdNd<half, half> op;
    REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm1tiling, op.mm3, bmm2tiling);
    op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq, dk, dv,
            user, tilingData, &pipeIn);
    op.Process();
    pipeIn.Destroy();

    TPipe pipeCast;
    FlashAttentionScoreGradCast<half, half> opCast;
    opCast.Init(dq, dk, dv, user, tilingData, &pipeCast);
    opCast.Process();
  } else if (TILING_KEY_IS(10003)) {
    FlashAttentionScoreGradBNSNdNdNdS1S2<half, half> op;
    REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm1tiling, op.mm3, bmm2tiling, op.mm4,
                      bmm3tiling);
    op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq, dk, dv,
            user, tilingData, &pipeIn);
    op.Process();
    pipeIn.Destroy();

    TPipe pipeCast;
    FlashAttentionScoreGradCast<half, half> opCast;
    opCast.Init(dq, dk, dv, user, tilingData, &pipeCast);
    opCast.Process();
  } else if (TILING_KEY_IS(1017)) {  // fp16, BN: S等长, 高精度模式
    FlashAttentionScoreGradBNNdNdNd<half, float, true> op;
    REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm1tiling, op.mm3, bmm2tiling);
    op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq, dk, dv,
            user, tilingData, &pipeIn);
    op.Process();
    pipeIn.Destroy();

    TPipe pipeCast;
    FlashAttentionScoreGradCast<half, float> opCast;
    opCast.Init(dq, dk, dv, user, tilingData, &pipeCast);
    opCast.Process();
  } else if (TILING_KEY_IS(2017)) {  // fp16, high precise, BN: special template
    FlashAttentionScoreGradBNNdNdNdSpecial<half, float, true> op;
    REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm1tiling, op.mm3, bmm2tiling);
    op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq, dk, dv,
            user, tilingData, &pipeIn);
    op.Process();
    pipeIn.Destroy();

    TPipe pipeCast;
    FlashAttentionScoreGradCast<half, float> opCast;
    opCast.Init(dq, dk, dv, user, tilingData, &pipeCast);
    opCast.Process();
  } else if (TILING_KEY_IS(1019)) {  // fp16, BNS: S等长, 高精度模式
    FlashAttentionScoreGradBNSNdNdNd<half, float, true> op;
    REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm1tiling, op.mm3, bmm2tiling);
    op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq, dk, dv,
            user, tilingData, &pipeIn);
    op.Process();
    pipeIn.Destroy();

    TPipe pipeCast;
    FlashAttentionScoreGradCast<half, float> opCast;
    opCast.Init(dq, dk, dv, user, tilingData, &pipeCast);
    opCast.Process();
  } else if (TILING_KEY_IS(10017)) {  // fp16, BN: S不等长, 高精度模式
    FlashAttentionScoreGradBNNdNdNdS1S2<half, float, true> op;
    REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm1tiling, op.mm3, bmm2tiling, op.mm4,
                      bmm3tiling);
    op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq, dk, dv,
            user, tilingData, &pipeIn);
    op.Process();
    pipeIn.Destroy();

    TPipe pipeCast;
    FlashAttentionScoreGradCast<half, float> opCast;
    opCast.Init(dq, dk, dv, user, tilingData, &pipeCast);
    opCast.Process();
  } else if (TILING_KEY_IS(10019)) {  // fp16, BNS: S不等长, 高精度模式
    FlashAttentionScoreGradBNSNdNdNdS1S2<half, float, true> op;
    REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm1tiling, op.mm3, bmm2tiling, op.mm4,
                      bmm3tiling);
    op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq, dk, dv,
            user, tilingData, &pipeIn);
    op.Process();
    pipeIn.Destroy();

    TPipe pipeCast;
    FlashAttentionScoreGradCast<half, float> opCast;
    opCast.Init(dq, dk, dv, user, tilingData, &pipeCast);
    opCast.Process();
  } else if (TILING_KEY_IS(5)) {
    SparseFlashAttentionScoreGradBNNdNdNd<half, half> op;
    REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm1tiling, op.mm3, bmm2tiling);
    op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq, dk, dv,
            user, tilingData, &pipeIn);
    op.Process();
    pipeIn.Destroy();

    TPipe pipeCast;
    FlashAttentionScoreGradCast<half, half> opCast;
    opCast.Init(dq, dk, dv, user, tilingData, &pipeCast);
    opCast.Process();
  } else if (TILING_KEY_IS(7)) {
    SparseFlashAttentionScoreGradBNSNdNdNd<half, half> op;
    REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm1tiling, op.mm3, bmm2tiling);
    op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq, dk, dv,
            user, tilingData, &pipeIn);
    op.Process();
    pipeIn.Destroy();

    TPipe pipeCast;
    FlashAttentionScoreGradCast<half, half> opCast;
    opCast.Init(dq, dk, dv, user, tilingData, &pipeCast);
    opCast.Process();
  } else if (TILING_KEY_IS(107)) {
    SparseFlashAttentionScoreGradBNSNdNdNdF16Special<half, half> op;
    REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm1tiling, op.mm3, bmm2tiling);
    op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq, dk, dv,
            user, tilingData, &pipeIn);
    op.Process();
    pipeIn.Destroy();

    TPipe pipeCast;
    FlashAttentionScoreGradCast<half, half> opCast;
    opCast.Init(dq, dk, dv, user, tilingData, &pipeCast);
    opCast.Process();
  } else if (TILING_KEY_IS(9)) {
    FlashAttentionScoreGradBNNdNdNd<float, float> op;
    REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm1tiling, op.mm3, bmm2tiling);
    op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq, dk, dv,
            user, tilingData, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(11)) {
    FlashAttentionScoreGradBNSNdNdNd<float, float> op;
    REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm1tiling, op.mm3, bmm2tiling);
    op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq, dk, dv,
            user, tilingData, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(13)) {
    SparseFlashAttentionScoreGradBNNdNdNd<float, float> op;
    REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm1tiling, op.mm3, bmm2tiling);
    op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq, dk, dv,
            user, tilingData, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(15)) {
    SparseFlashAttentionScoreGradBNSNdNdNd<float, float> op;
    REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm1tiling, op.mm3, bmm2tiling);
    op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq, dk, dv,
            user, tilingData, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(17)) {
    FlashAttentionScoreGradBNNdNdNd<bfloat16_t, float, true> op;
    REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm1tiling, op.mm3, bmm2tiling);
    op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq, dk, dv,
            user, tilingData, &pipeIn);
    op.Process();
    pipeIn.Destroy();

    TPipe pipeCast;
    FlashAttentionScoreGradCast<bfloat16_t, float> opCast;
    opCast.Init(dq, dk, dv, user, tilingData, &pipeCast);
    opCast.Process();
  } else if (TILING_KEY_IS(19)) {
    FlashAttentionScoreGradBNSNdNdNd<bfloat16_t, float, true> op;
    REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm1tiling, op.mm3, bmm2tiling);
    op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq, dk, dv,
            user, tilingData, &pipeIn);
    op.Process();
    pipeIn.Destroy();

    TPipe pipeCast;
    FlashAttentionScoreGradCast<bfloat16_t, float> opCast;
    opCast.Init(dq, dk, dv, user, tilingData, &pipeCast);
    opCast.Process();
  } else if (TILING_KEY_IS(21)) {
    SparseFlashAttentionScoreGradBNNdNdNd<bfloat16_t, float, true> op;
    REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm1tiling, op.mm3, bmm2tiling);
    op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq, dk, dv,
            user, tilingData, &pipeIn);
    op.Process();
    pipeIn.Destroy();

    TPipe pipeCast;
    FlashAttentionScoreGradCast<bfloat16_t, float> opCast;
    opCast.Init(dq, dk, dv, user, tilingData, &pipeCast);
    opCast.Process();
  } else if (TILING_KEY_IS(23)) {
    SparseFlashAttentionScoreGradBNSNdNdNd<bfloat16_t, float, true> op;
    REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm1tiling, op.mm3, bmm2tiling);
    op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq, dk, dv,
            user, tilingData, &pipeIn);
    op.Process();
    pipeIn.Destroy();

    TPipe pipeCast;
    FlashAttentionScoreGradCast<bfloat16_t, float> opCast;
    opCast.Init(dq, dk, dv, user, tilingData, &pipeCast);
    opCast.Process();
  } else if (TILING_KEY_IS(2023)) {  // bf16, sparse, BNS, special template
    SparseFlashAttentionScoreGradBNSNdNdNdSpecial<bfloat16_t, float, true> op;
    REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm1tiling, op.mm3, bmm2tiling);
    op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq, dk, dv,
            user, tilingData, &pipeIn);
    op.Process();
    pipeIn.Destroy();

    TPipe pipeCast;
    FlashAttentionScoreGradCast<bfloat16_t, float> opCast;
    opCast.Init(dq, dk, dv, user, tilingData, &pipeCast);
    opCast.Process();
  } else if (TILING_KEY_IS(2123)) {  // bf16, sparse, BNS, special template, square basic block
    SparseFlashAttentionScoreGradBNSNdNdNdSpecialSquare<bfloat16_t, float, true> op;
    REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm1tiling, op.mm3, bmm2tiling);
    op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq, dk, dv,
            user, tilingData, &pipeIn);
    op.Process();
    pipeIn.Destroy();

    TPipe pipeCast;
    FlashAttentionScoreGradCast<bfloat16_t, float> opCast;
    opCast.Init(dq, dk, dv, user, tilingData, &pipeCast);
    opCast.Process();
  } else if (TILING_KEY_IS(1021)) {  // fp16, sparse, BN: S等长, 高精度模式
    SparseFlashAttentionScoreGradBNNdNdNd<half, float, true> op;
    REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm1tiling, op.mm3, bmm2tiling);
    op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq, dk, dv,
            user, tilingData, &pipeIn);
    op.Process();
    pipeIn.Destroy();

    TPipe pipeCast;
    FlashAttentionScoreGradCast<half, float> opCast;
    opCast.Init(dq, dk, dv, user, tilingData, &pipeCast);
    opCast.Process();
  } else if (TILING_KEY_IS(1023)) {  // fp16, sparse, BNS: S等长, 高精度模式
    SparseFlashAttentionScoreGradBNSNdNdNd<half, float, true> op;
    REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm1tiling, op.mm3, bmm2tiling);
    op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq, dk, dv,
            user, tilingData, &pipeIn);
    op.Process();
    pipeIn.Destroy();

    TPipe pipeCast;
    FlashAttentionScoreGradCast<half, float> opCast;
    opCast.Init(dq, dk, dv, user, tilingData, &pipeCast);
    opCast.Process();
  } else if (TILING_KEY_IS(2223)) {  // fp16, sparse, BNS: S等长, 高精度模式, special
    SparseFlashAttentionScoreGradBNSNdNdNdSpecial<half, float, true> op;
    REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm1tiling, op.mm3, bmm2tiling);
    op.Init(key, value, dy, query, pse_shift, drop_mask, atten_mask, attention_in, softmax_max, softmax_sum, dq, dk, dv,
            user, tilingData, &pipeIn);
    op.Process();
    pipeIn.Destroy();

    TPipe pipeCast;
    FlashAttentionScoreGradCast<half, float> opCast;
    opCast.Init(dq, dk, dv, user, tilingData, &pipeCast);
    opCast.Process();
  } else if (TILING_KEY_IS(90)) {
    FlashAttentionScoreGradEmpty<DTYPE_DQ> op;
    op.Init(dq, dk, dv, dpse, tilingData);
    op.Process();
  }
}
