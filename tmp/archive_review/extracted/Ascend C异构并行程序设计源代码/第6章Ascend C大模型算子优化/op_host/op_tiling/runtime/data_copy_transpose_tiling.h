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

#ifndef DATA_COPY_TRANSPOSE_H
#define DATA_COPY_TRANSPOSE_H

#include "graph/tensor.h"
#include "flash_attention_score_tiling.h"


inline void GetDataCopyTransposeTiling(const ge::Shape dstShape, const ge::Shape srcShape, const uint32_t typeSize,
                                       optiling::CopyTransposeTiling& tiling) {
  std::vector<int64_t> dstShapeInfo = dstShape.GetDims();
  std::vector<int64_t> srcShapeInfo = srcShape.GetDims();

  tiling.set_dstShapeB(dstShapeInfo[0]);
  tiling.set_dstShapeN(dstShapeInfo[1]);
  tiling.set_dstShapeS(dstShapeInfo[2]);
  tiling.set_dstShapeH(dstShapeInfo[3]);
  tiling.set_dstShapeHN(tiling.get_dstShapeH() / tiling.get_dstShapeN());
  
  tiling.set_srcShapeB(srcShapeInfo[0]);
  tiling.set_srcShapeN(srcShapeInfo[1]);
  tiling.set_srcShapeS(srcShapeInfo[2]);
  tiling.set_srcShapeHN(srcShapeInfo[3]);
  tiling.set_originalShapeNLen(tiling.get_srcShapeHN() * typeSize);
  tiling.set_shapeSHValue(tiling.get_dstShapeS() * tiling.get_dstShapeH());
  tiling.set_shapeNsValue(tiling.get_dstShapeN() * tiling.get_dstShapeS());
  tiling.set_shapeNsnValue(tiling.get_dstShapeN() * tiling.get_srcShapeS() * tiling.get_srcShapeN());
  tiling.set_shapeBHValue(tiling.get_dstShapeB() * tiling.get_dstShapeH());
}

#endif  // DATA_COPY_TRANSPOSE_H
