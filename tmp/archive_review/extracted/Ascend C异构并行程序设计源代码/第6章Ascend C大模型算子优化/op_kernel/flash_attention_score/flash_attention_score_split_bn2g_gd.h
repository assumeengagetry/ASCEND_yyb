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

#ifndef FLASH_ATTENTION_SCORE_SPLIT_B_N2_G_D_H
#define FLASH_ATTENTION_SCORE_SPLIT_B_N2_G_D_H

#include "flash_attention_score_common.h"
#include "kernel_data_copy_transpose.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"


// INPUT_T - means data type for input
// T       - means data type when calc

using matmul::MatmulType;

template <typename INPUT_T, typename T = INPUT_T, bool isBasicBlock = false, LayoutMode layout = LayoutMode::BSNGD>
class FlashAttentionScoreSplitBn2gGd : public FlashAttentionScoreCommon<INPUT_T, T, isBasicBlock> {
  public:
    __aicore__ inline FlashAttentionScoreSplitBn2gGd();

  // 主流程函数
    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* pse,
                                __gm__ uint8_t* dropMask, __gm__ uint8_t* paddingMask, __gm__ uint8_t* attenMask,
                                __gm__ uint8_t* softmaxMax, __gm__ uint8_t* softmaxSum, __gm__ uint8_t* softmaxOut,
                                __gm__ uint8_t* attentionOut, __gm__ uint8_t* workspace,
                                const FlashAttentionScoreGeneralTilingData* __restrict tiling, TPipe* tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void initTilingData();
    __aicore__ inline void SplitNGroups(int64_t splitBaseNum, int64_t multiCoreInnerLimit);
    __aicore__ inline void ProcessBlock(int64_t countBase);
    __aicore__ inline void ProcessStage1(AscendC::LocalTensor<T>& stage1ResUb);
    __aicore__ inline void ProcessStage2(AscendC::LocalTensor<T>& stage1ResUb);
    __aicore__ inline void InitSoftmaxTensor();

    // offset计算相关函数
    __aicore__ inline void ComputeAxisUnitSize();
    __aicore__ inline void ComputeIdx();
    __aicore__ inline void AllocBmm1ResUb(AscendC::LocalTensor<T>& bmm1ResUb);
    __aicore__ inline int64_t ComputePseOffset();
    __aicore__ inline int64_t ComputeAttenMaskOffset();
    __aicore__ inline int64_t ComputeDropOffset();
    __aicore__ inline int64_t ComputeSoftmaxOutOffset();
    __aicore__ inline void Bmm1SetTensorA();
    __aicore__ inline void Bmm1SetTensorB();
    __aicore__ inline void Bmm1SetTail(int64_t countBase);
    __aicore__ inline void Bmm2SetTensor(AscendC::LocalTensor<T>& stage1ResUb);
    __aicore__ inline void Bmm2Final();
    __aicore__ inline void Bmm2GetTensorC(AscendC::LocalTensor<T>& bmm2FinalResUb);
    // 计算相关代码
    __aicore__ inline void SoftmaxCompute(AscendC::LocalTensor<T>& dstTensor, AscendC::LocalTensor<T>& srcTensor);

    using aType1 = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false, layout>;
    using bType1 = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, true, layout>;
    using cType1 = MatmulType<TPosition::VECCALC, CubeFormat::ND, T, false, LayoutMode::BNGS1S2>;
    using biasTpye1 = MatmulType<TPosition::GM, CubeFormat::ND, float, false, LayoutMode::BNGS1S2>;
    matmul::Matmul<aType1, bType1, cType1, biasTpye1> bmm1;
    
    using aType2 = MatmulType<TPosition::VECCALC, CubeFormat::ND, INPUT_T, false, LayoutMode::BNGS1S2>;
    using bType2 = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, false, layout>;
    using cType2 = MatmulType<TPosition::GM, CubeFormat::ND, T, false, layout>;
    using biasTpye2 = MatmulType<TPosition::GM, CubeFormat::ND, float, false, layout>;
    matmul::Matmul<aType1, bType2, cType2, biasTpye2> bmm2;
    
    // 用于计算不同layout下的各轴单位数据量
    // Q、bmm2Output
    int64_t qBUnitSize;
    int64_t qN2UnitSize;
    int64_t qGUnitSize;
    // K、V
    int64_t kvBoUnitSize;
    int64_t kvN2oUnitSize;
    // attenMask
    int64_t attenBUnitSize;
    int64_t attenN2UnitSize;
    int64_t attenGUnitSize;
    // dropMask
    int64_t dropBUnitSize;
    int64_t dropN2UnitSize;
    int64_t dropGUnitSize;
    // pse
    int64_t pseBUnitSizeType1;
    int64_t pseN2UnitSizeType1;
    int64_t pseGUnitSizeType1;
    int64_t pseBUnitSizeType2;
    int64_t pseN2UnitSizeType2;
    int64_t pseGUnitSizeType2;
    // softmaxOutput
    int64_t softmaxBUnitSize;
    int64_t softmaxN2UnitSize;
    int64_t softmaxGUnitSize;
    // 资源分配
    AscendC::TQue<QuePosition::VECIN, 1> bmm2ResQueue; // 存bmm2 res

    // todo 添加tilingdata变量定义
    int64_t batchSize;
    int64_t nSize;
    int64_t groupSize;
    int64_t qSeqSize;
    int64_t dSize;
    int64_t kSeqSize;

    int64_t multiCoreInnerSize;
    int64_t multiCoreTotalSize;
    int64_t n2OuterSize;
    int64_t gOuterSize;
    int64_t gBaseSize;
    int64_t gBaseTailSize;
    int64_t n2oGo;

    // 全局变量定义
    int64_t multiCoreInnerIdx;
    int64_t attenGUnitSizeType1;
    int64_t attenN2UnitSizeType1;
    int64_t attenBUnitSizeType1;
    int64_t attenBUnitSizeType2;
    int8_t layoutType;
    int64_t nRatio;
    int64_t gRealSize;
};

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline FlashAttentionScoreSplitBn2gGd<INPUT_T, T, isBasicBlock, layout>::FlashAttentionScoreSplitBn2gGd() {};

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBn2gGd<INPUT_T, T, isBasicBlock, layout>::Init(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* pse, __gm__ uint8_t* dropMask,
    __gm__ uint8_t* paddingMask, __gm__ uint8_t* attenMask, __gm__ uint8_t* softmaxMax, __gm__ uint8_t* softmaxSum,
    __gm__ uint8_t* softmaxOut, __gm__ uint8_t* attentionOut, __gm__ uint8_t* workspace,
    const FlashAttentionScoreGeneralTilingData* __restrict tiling, TPipe* tPipe) {
  this->BaseInit(query, key, value, pse, dropMask, paddingMask, attenMask, softmaxMax,
                 softmaxSum, softmaxOut, attentionOut, workspace, tiling, tPipe);
  this->initTilingData();
}

// todo：是否新定义变量？直接使用tiling的会不会更好
template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBn2gGd<INPUT_T, T, isBasicBlock, layout>::initTilingData() {
  batchSize = this->tilingData->inputParams.bSize; // B
  nSize = this->tilingData->inputParams.n2Size; // N
  groupSize = this->tilingData->inputParams.gSize; // G
  qSeqSize = this->tilingData->inputParams.s1Size; // S1
  dSize = this->tilingData->inputParams.dSize; // D
  kSeqSize = this->tilingData->inputParams.s2Size; // S2
  layoutType = this->tilingData->inputParams.layoutType;
  nRatio = this->tilingData->coreParams.nRatio;

  // todo: 补全
  multiCoreInnerSize = this->tilingData->multiCoreParams.splitFactorSize; // BNG/core_num 核内计算gBaseSize个S1*D
  multiCoreTotalSize = this->tilingData->multiCoreParams.totalSize;

  // base块 数据量大小 gBaseSize*S1*D
  n2OuterSize = this->tilingData->coreParams.n2OuterSize;
  gOuterSize = this->tilingData->coreParams.gOuterSize; // G被切分的base块数
  gBaseSize = this->tilingData->coreParams.gBaseSize; // G轴在ub切分的内轴大小
  gBaseTailSize = this->tilingData->coreParams.gBaseTailSize; // G轴尾块大小
}

// 计算各轴单位数据量(unitSize)，用于与各轴idx相乘计算offset
// todo: 整体逻辑&代码检查
template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBn2gGd<INPUT_T, T, isBasicBlock, layout>::ComputeAxisUnitSize() {
  // BSNGD:0; SBNGD:1; BNGSD:2
  // Q、KV、PSE、attenMASK、dropMask
  if (this->layoutType == 1) {  // BSNGD
    // Qshape: BSNGD
    // Q.G: G_b * D
    this->qGUnitSize = this->gBaseSize * this->dSize;
    // Q.N: G*D
    this->qN2UnitSize = this->groupSize * this->dSize;
    // Q.B: S1*N2*G*D
    this->qBUnitSize = this->qSeqSize * this->nSize * this->qN2UnitSize;

    // KVshape相同: BSN1D
    // KV.N: 1*D
    this->kvN2oUnitSize = this->dSize;
    // KV.B: S2*N2*1*D
    this->kvBoUnitSize = this->kSeqSize * this->nSize * this->dSize;

  } else if (this->layoutType == 2) {  // SBNGD
    // Q:
    // G_b*D
    this->qGUnitSize = this->gBaseSize * this->dSize;
    // G*D
    this->qN2UnitSize = this->groupSize * this->dSize;
    // N2*G*D
    this->qBUnitSize = this->nSize * this->groupSize * this->dSize;

    // KV: G=1
    // 1*D
    this->kvN2oUnitSize = this->dSize;
    // N2*1*D
    this->kvBoUnitSize = this->nSize * this->dSize;
  } else if (this->layoutType == 3) {  // BNGSD
    // Q:
    // G: G_b*S1*D
    this->qGUnitSize = this->gBaseSize * this->qSeqSize * this->dSize;
    // N: G*S1*D
    this->qN2UnitSize = this->groupSize * this->qSeqSize * this->dSize;
    // N2*G*S1*D
    this->qBUnitSize = this->nSize * this->qN2UnitSize;

    // KV: G=1
    // 1*S2*D
    this->kvN2oUnitSize = this->kSeqSize * this->dSize;
    // N2*1*S2*D
    this->kvBoUnitSize = this->nSize * this->kvN2oUnitSize;
  }
  // todo
  // attenMask[B,N2,G,S1,S2]
  // G_b*S1*S2
  this->attenGUnitSizeType1 = this->gBaseSize * this->qSeqSize * this->kSeqSize;
  // G*S1*S2
  this->attenN2UnitSizeType1 = this->groupSize * this->qSeqSize * this->kSeqSize;
  // N2*G*S1*S2
  this->attenBUnitSizeType1 = this->nSize * this->attenN2UnitSizeType1;
  // attenMask[B,1,1,S1,S2]
  // S1*S2
  this->attenBUnitSizeType2 = this->qSeqSize * this->kSeqSize;
  // attenMask[1,1,1,S1,S2]

  // dropMask
  // G_b*S1*S2
  this->dropGUnitSize = this->attenGUnitSizeType1;
  // G*S1*S2
  this->dropN2UnitSize = this->groupSize * this->qSeqSize * this->kSeqSize;
  // N2*G*S1*S2
  this->dropBUnitSize = this->nSize * this->dropN2UnitSize;

  // pse[B,N2,G,S1,S2]
  // S1*S2
  this->pseGUnitSizeType1 = this->dropGUnitSize;
  // G*S1*S2
  this->pseN2UnitSizeType1 = this->dropN2UnitSize;
  // N2*G*S1*S2
  this->pseBUnitSizeType1 = this->dropBUnitSize;

  // pse[B,N2,G,1,S2]
  // g_b*1*S2
  this->pseGUnitSizeType2 = this->gBaseSize * this->kSeqSize;
  // G*1*S2
  this->pseN2UnitSizeType2 = this->groupSize * this->kSeqSize;
  // N2*G*1*S2
  this->pseBUnitSizeType2 = this->nSize * this->pseN2UnitSizeType2;

  // softmaxOutput[B,N2,G,S1,1]
  // g_b*S1*1
  this->softmaxGUnitSize = this->gBaseSize * this->qSeqSize;
  // G*S1*1
  this->softmaxN2UnitSize = this->groupSize * this->qSeqSize;
  // N2*G*S1*1
  this->softmaxBUnitSize = this->nSize * this->softmaxN2UnitSize;
}


template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBn2gGd<INPUT_T, T, isBasicBlock, layout>::Process() {

    this->multiCoreInnerIdx = this->blockIdx * this->multiCoreInnerSize; // 当前核 在总base块中的偏移量
    int64_t multiCoreInnerLimit = this->Min(this->multiCoreInnerIdx + this->multiCoreInnerSize, this->multiCoreTotalSize); // 当前核 需要计算的base块数量

    // 计算各轴的单位数据量
    this->ComputeAxisUnitSize();
    
    // BN2Gb开多核场景，b,n2对应的OuterSize等于实际shape
    // this->gOuterSize // 单位N base块数量
    this->n2oGo = this->n2OuterSize * this->gOuterSize;  // 单位B base块数量
    
    // 起始offset可能在G块内部，此时需要计算在第一个不完整的G中的计算量
    this->boIdx = this->multiCoreInnerIdx / this->n2oGo; // batch idx
    this->n2oIdx = this->multiCoreInnerIdx % this->n2oGo / this->gOuterSize; // N idx
    this->Bmm1SetTensorB();

    int64_t firstSplitBaseNum = this->Min(this->gOuterSize - this->multiCoreInnerIdx % this->gOuterSize, this->multiCoreInnerSize);
    this->SplitNGroups(firstSplitBaseNum, multiCoreInnerLimit);

    // 起始块计算完成后，计算剩下整块，
    // 起始地址是一个新的G，但单核base块很多时，每次跨轴都进一个新的循环
    for (int64_t gOuterLoopIdx = 0;
        gOuterLoopIdx < this->CeilDiv((this->multiCoreInnerSize - firstSplitBaseNum), this->gOuterSize) &&
         this->multiCoreInnerIdx < multiCoreInnerLimit;
        gOuterLoopIdx++) {
      // 当前进行N:1分组的base块个数
      int64_t splitBaseNum = this->Min(this->gOuterSize, this->multiCoreInnerSize - firstSplitBaseNum);
      firstSplitBaseNum += splitBaseNum;

      // 为了不在for循环里重复设置K矩阵，将该操作提到循环外面
      this->boIdx = this->multiCoreInnerIdx / this->n2oGo;
      this->n2oIdx = this->multiCoreInnerIdx % this->n2oGo / this->gOuterSize;
      this->Bmm1SetTensorB();

      this->SplitNGroups(splitBaseNum, multiCoreInnerLimit);
    }
}


/*
 * 函数功能：对输入的base块进行N:1分组，默认输入的base块都在同一轴内，即属于同一组S1
 * 输入参数：
 *          splitBaseNum: 用于进行N:1分组的base块个数
 *          multiCoreInnerLimit: 最多计算base块个数
 */
template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBn2gGd<INPUT_T, T, isBasicBlock, layout>::SplitNGroups(int64_t splitBaseNum,
                                                                                 int64_t multiCoreInnerLimit
                                                                                 ) {
  // 对数据块进行N:1分组，循环的次数即为分组数，分组数=ceil(splitBaseNum, N)
  for (int64_t splitTimes = 0;
       splitTimes < this->CeilDiv(splitBaseNum, this->nRatio) && // 可以实现几个 N:1
       this->multiCoreInnerIdx < multiCoreInnerLimit; // 当前base块idx < 当前核需要计算的base块
       splitTimes++) {
    
    // 当前N:1组所计算的base块数量，小于等于N
    int64_t countBase = this->Min(this->nRatio, splitBaseNum - splitTimes * this->nRatio);
    
    // 计算各轴idx
    this->ComputeIdx();

    // 对当前N:1组进行attention计算，最多不超过N个base块
    this->ProcessBlock(countBase);

    // 更新idx，偏移至下个N:1组的首地址
    this->multiCoreInnerIdx += countBase;
  }
}

// 计算各轴index
template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBn2gGd<INPUT_T, T, isBasicBlock, layout>::ComputeIdx() {
  this->boIdx = this->multiCoreInnerIdx / this->n2oGo; // batch idx
  this->n2oIdx = this->multiCoreInnerIdx % this->n2oGo / this->gOuterSize; // n2 idx
  this->goIdx = this->multiCoreInnerIdx % this->gOuterSize; // G内base块的idx
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBn2gGd<INPUT_T, T, isBasicBlock, layout>::ProcessBlock(int64_t countBase) {
  // todo change bmm
  for (int64_t splitNIdx = 0; splitNIdx < countBase; splitNIdx++) {
    // 计算g的realsize，用于后续vector和bmm2计算
    if (this->goIdx == gOuterSize - 1) {
      this->gRealSize = this->gBaseTailSize; // G轴切分后,g的长的
    } else {
      this->gRealSize = this->gBaseSize;
    }
    this->InitSoftmaxTensor();
    
    AscendC::LocalTensor<T> stage1ResUb = this->stage1ResQueue.template AllocTensor<T>(); // from common.h

    this->ProcessStage1(stage1ResUb); // 增加stage1操作，完成get_tensorC到bmm2之间的操作
    this->ProcessStage2(stage1ResUb); // 包含bmm2操作

    // 每次循环处理一个S1 base块，因此处理下一块前索引需要+1
    this->goIdx += 1;
  }
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBn2gGd<INPUT_T, T, isBasicBlock, layout>::Bmm1SetTensorA() {
  // 计算各轴offset
  int64_t qBOffset = this->boIdx * this->qBUnitSize;
  int64_t qN2Offset = this->n2oIdx * this->qN2UnitSize;
  int64_t qGOffset = this->goIdx * this->qGUnitSize;

  // 当前base块offset
  int64_t qCoreOffset = qBOffset + qN2Offset + qGOffset;

  this->bmm1.SetTensorA(this->queryGm[qCoreOffset]);
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBn2gGd<INPUT_T, T, isBasicBlock, layout>::Bmm1SetTensorB() {
  // 计算各轴offset
  int64_t kBOffset = this->boIdx * this->kvBoUnitSize;
  int64_t kN2Offset = this->n2oIdx * this->kvN2oUnitSize;

  // 当前base块offset
  int64_t kCoreOffset = kBOffset + kN2Offset;

  this->bmm1.SetTensorB(this->keyGm[kCoreOffset], true);
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBn2gGd<INPUT_T, T, isBasicBlock, layout>::Bmm1SetTail(int64_t countBase) {
  // 整块场景
  int64_t NGroupRealSize = this->gBaseSize * countBase;

  // 有尾块的场景，可能由多个g_b和一个g_t组成一个N:1
  // todo: 是否要加unlikely?
  if ((this->multiCoreInnerIdx + countBase) % this->gOuterSize == 0) {
    NGroupRealSize = this->gBaseSize * (countBase - 1) + this->gBaseTailSize;
  }

  this->bmm1.SetTail(NGroupRealSize, -1, -1);
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBn2gGd<INPUT_T, T, isBasicBlock, layout>::InitSoftmaxTensor() {
  uint32_t maxSumShape[] = {static_cast<uint32_t>(this->gRealSize * this->qSeqSize), 8};
  this->softmaxSumUb = this->softmaxOutQueue.template AllocTensor<float>();
  this->softmaxMaxUb = this->softmaxOutQueue.template AllocTensor<float>();
  this->softmaxMaxUb.SetShapeInfo(ShapeInfo(2, maxSumShape, AscendC::DataFormat::ND));
  this->softmaxSumUb.SetShapeInfo(ShapeInfo(2, maxSumShape, AscendC::DataFormat::ND));
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBn2gGd<INPUT_T, T, isBasicBlock, layout>::ProcessStage1(AscendC::LocalTensor<T>& stage1ResUb) {
  
  int64_t pseOffset = this->ComputePseOffset(); // 计算pse的offset
  this->PseCopyIn(pseOffset, this->qSeqSize, this->kSeqSize, this->gRealSize);

  // 获取bmm1的计算结果
  AscendC::LocalTensor<T> bmm1ResUb = this->bmm1ResQueue.template AllocTensor<T>();
  this->AllocBmm1ResUb(bmm1ResUb); // 为bmm1ResUb分配空间

  this->Bmm1SetTensorA();
  this->bmm1.template IterateBatch<true>(bmm1ResUb, this->gRealSize, 1, false);

  this->PseComputeWithBatch(bmm1ResUb, this->gRealSize, this->qSeqSize, this->kSeqSize);

  int64_t attenMaskOffset = this->ComputeAttenMaskOffset(); // 计算attenMask的offset
  this->AttenMaskCopyIn(attenMaskOffset, this->qSeqSize, this->kSeqSize, this->gRealSize);
  // todo：attentionmask在特定layout下的broadcastadd需要修改，当前仅对齐shape，精度可能有问题
  this->AttenMaskCompute(bmm1ResUb, this->qSeqSize, this->kSeqSize, this->gRealSize);

  if (this->hasDropMask) {
    // softmax的计算结果需要执行dropout操作
    int64_t dropMaskOffset = this->ComputeDropOffset();
    this->DropoutCopyIn(attenMaskOffset, this->gRealSize * this->qSeqSize, this->kSeqSize);
    this->SoftmaxCompute(bmm1ResUb, bmm1ResUb);
    this->DropoutCompute(stage1ResUb, bmm1ResUb, this->qSeqSize, this->kSeqSize, this->gRealSize);
  } else {
    this->SoftmaxCompute(stage1ResUb, bmm1ResUb);
  }

  // todo: tbuf和tque如何用同一个变量名表示？后面怎么free？
  this->Stage1Cast(stage1ResUb, stage1ResUb); // from common.h

  this->bmm1ResQueue.FreeTensor(bmm1ResUb);
  return;
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline int64_t FlashAttentionScoreSplitBn2gGd<INPUT_T, T, isBasicBlock, layout>::ComputePseOffset() {
  // 默认shape为[B,N2,G,S1,S2]
  int64_t pseBOffset = this->boIdx * this->pseBUnitSizeType1;
  int64_t pseN2Offset = this->n2oIdx * this->pseN2UnitSizeType1;
  int64_t pseGOffset = this->goIdx * this->pseGUnitSizeType1;

  // 当pse的shape为[B,N,G,1,S2]
  if (this->tilingData->inputParams.pseShapeType == 1) {
    pseBOffset = this->boIdx * this->pseBUnitSizeType2;
    pseN2Offset = this->n2oIdx * this->pseN2UnitSizeType2;
    pseGOffset = this->goIdx * this->pseGUnitSizeType2;
  }

  return pseBOffset + pseN2Offset + pseGOffset;
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBn2gGd<INPUT_T, T, isBasicBlock, layout>::AllocBmm1ResUb(
    AscendC::LocalTensor<T>& bmm1ResUb) {
  // todo: 这里的kSeqSize是否要对齐后的？
  uint32_t bmm1ResShape[] = {static_cast<uint32_t>(this->gRealSize * this->qSeqSize), static_cast<uint32_t>(this->kSeqSize)};
  bmm1ResUb.SetShapeInfo(ShapeInfo(2, bmm1ResShape, AscendC::DataFormat::ND));
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline int64_t FlashAttentionScoreSplitBn2gGd<INPUT_T, T, isBasicBlock, layout>::ComputeAttenMaskOffset() {
  // 默认shape为[B,N2,G,S1,S2]
  int64_t attenBOffset = this->boIdx * this->attenBUnitSizeType1;
  int64_t attenN2Offset = this->n2oIdx * this->attenN2UnitSizeType1;
  int64_t attenGOffset = this->goIdx * this->attenGUnitSizeType1;

  // 如果shape为[1,1,1,S1,S2]
  if (this->tilingData->inputParams.attenMaskShapeType == 2) {
    attenBOffset = 0;
    attenN2Offset = 0;
    attenGOffset = 0;
  } else if (this->tilingData->inputParams.attenMaskShapeType == 1) {
    // 如果shape为[B,1,1,S1,S2]
    attenBOffset = this->boIdx * this->attenBUnitSizeType2;
    attenN2Offset = 0;
    attenGOffset = 0;
  }

  return attenBOffset + attenN2Offset + attenGOffset;
}


template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBn2gGd<INPUT_T, T, isBasicBlock, layout>::ProcessStage2(AscendC::LocalTensor<T>& stage1ResUb) {
  this->Bmm2SetTensor(stage1ResUb);
  this->stage1ResQueue.FreeTensor(stage1ResUb);
  this->Bmm2Final();
  return;
}

// todo: 非对齐场景padding 0 需要特殊处理
template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBn2gGd<INPUT_T, T, isBasicBlock, layout>::SoftmaxCompute(
    AscendC::LocalTensor<T>& dstTensor, AscendC::LocalTensor<T>& srcTensor) {
  AscendC::LocalTensor<uint8_t> apiTmpBuffer = this->apiTBuf.template Get<uint8_t>();
  AscendC::SoftMax<T, true, isBasicBlock>(dstTensor, this->softmaxSumUb, this->softmaxMaxUb, srcTensor, apiTmpBuffer,
                                 this->tilingData->softmaxTilingData);
  int64_t softmaxGmOffset = this->ComputeSoftmaxOutOffset();
  this->SoftmaxCopyOut(softmaxGmOffset, this->gRealSize * this->qSeqSize * 8);
  this->softmaxOutQueue.FreeTensor(this->softmaxSumUb);
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline int64_t FlashAttentionScoreSplitBn2gGd<INPUT_T, T, isBasicBlock, layout>::ComputeSoftmaxOutOffset() {
  // shape为[B,N2,G,S1,1]
  int64_t softmaxBOffset = this->boIdx * this->softmaxBUnitSize;
  int64_t softmaxN2Offset = this->n2oIdx * this->softmaxN2UnitSize;
  int64_t softmaxGOffset = this->goIdx * this->softmaxGUnitSize;

  return softmaxBOffset + softmaxN2Offset + softmaxGOffset;
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline int64_t FlashAttentionScoreSplitBn2gGd<INPUT_T, T, isBasicBlock, layout>::ComputeDropOffset() {
  // shape为[B,N2,G,S1,S2]
  int64_t dropBOffset = this->boIdx * this->dropBUnitSize;
  int64_t dropN2Offset = this->n2oIdx * this->dropN2UnitSize;
  int64_t dropGOffset = this->goIdx * this->dropGUnitSize;

  return dropBOffset + dropN2Offset + dropGOffset;
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBn2gGd<INPUT_T, T, isBasicBlock, layout>::Bmm2SetTensor(
    AscendC::LocalTensor<T>& stage1ResUb) {
  // todo bf16实现

  int64_t vBOffset = this->boIdx * this->kvBoUnitSize;
  int64_t vN2Offset = this->n2oIdx * this->kvN2oUnitSize;
  int64_t vCoreOffset = vBOffset + vN2Offset;

  this->bmm2.SetTensorA(stage1ResUb);
  this->bmm2.SetTensorB(this->valueGm[vCoreOffset]);
}

template <typename INPUT_T, typename T, bool isBasicBlock, LayoutMode layout>
__aicore__ inline void FlashAttentionScoreSplitBn2gGd<INPUT_T, T, isBasicBlock, layout>::Bmm2Final() {

  // s2全载情况下，只有一次循环，[B,N,S,D]场景下可以直接输出到GM上
  // Unitsize可以复用Q的
  int64_t bmm2BOffset = this->boIdx * this->qBUnitSize;
  int64_t bmm2N2Offset = this->n2oIdx * this->qN2UnitSize;
  int64_t bmm2GOffset = this->goIdx * this->qGUnitSize;

  // 当前base块offset
  int64_t bmm2CoreOffset = bmm2BOffset + bmm2N2Offset + bmm2GOffset;

  this->bmm2.template IterateBatch<true>(this->attentionOutGm[bmm2CoreOffset], this->gRealSize, 1, true);
}

#endif  // FLASH_ATTENTION_SCORE_SPLIT_B_N2_G_D_H
