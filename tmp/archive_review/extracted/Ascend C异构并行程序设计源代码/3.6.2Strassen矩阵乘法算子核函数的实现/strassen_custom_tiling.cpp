#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include "tiling/tiling_api.h"
using namespace matmul_tiling;
using namespace std;

uint8_t *GetTilingBuf(optiling::TCubeTiling *tilingData) {
  uint32_t tilingSize = tilingData->GetDataSize();
  uint8_t *buf = (uint8_t *)malloc(tilingSize);
  tilingData->SaveToBuffer(buf, tilingSize);
  return buf;
}

uint8_t *GenerateTiling() {
  int M = 16;
  int N = 16;
  int K = 16;
  TPosition leftPos = TPosition::VECOUT;
  CubeFormat leftFormat = CubeFormat::ND;
  DataType leftDtype = DataType::DT_FLOAT16;
  int transposeA = 0;

  TPosition rightPos = TPosition::VECOUT;
  CubeFormat rightFormat = CubeFormat::ND;
  DataType rightDtype = DataType::DT_FLOAT16;
  int transposeB = 0;

  TPosition resPos = TPosition::VECIN;
  CubeFormat resFormat = CubeFormat::ND;
  DataType resDtype = DataType::DT_FLOAT;

  TPosition biasPos = TPosition::VECOUT;
  CubeFormat biasFormat = CubeFormat::ND;
  DataType biasDtype = DataType::DT_FLOAT;
  int isBias = 0;

  int usedCoreNum = 1;
  int runMode = 1;
  int32_t baseM = 16;
  int32_t baseN = 16;

  optiling::TCubeTiling tilingData;
  tilingData.set_usedCoreNum(usedCoreNum);
  MatmulApiTiling tilingApi;
  tilingApi.SetAType(leftPos, leftFormat, leftDtype, bool(transposeA));
  tilingApi.SetBType(rightPos, rightFormat, rightDtype, bool(transposeB));
  tilingApi.SetCType(resPos, resFormat, resDtype);
  tilingApi.SetBiasType(biasPos, biasFormat, biasDtype);

  tilingApi.SetOrgShape(M, N, K);
  tilingApi.SetShape(M, N, K);
  tilingApi.SetBias(bool(isBias));
  tilingApi.SetBufferSpace(-1, -1, -1);

  int64_t res = tilingApi.GetTiling(tilingData);
  if (res == -1) {
    std::cout << "gen tiling failed" << std::endl;
  }
  return GetTilingBuf(&tilingData);
}
