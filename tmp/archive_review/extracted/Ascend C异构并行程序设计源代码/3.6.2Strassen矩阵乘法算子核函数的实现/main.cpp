/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * This file constains code of cpu debug and npu code.We read data from bin file
 * and write result to file.
 */

#include "data_utils.h"
#ifndef __CCE_KT_TEST__
#include "acl/acl.h"
#include "aclrtlaunch_strassen_custom.h"
#else
#include "tikicpulib.h"
extern "C" void strassen_custom(uint8_t *a, uint8_t *b, uint8_t *c, uint8_t *workspace, uint8_t *tiling);
#endif
extern uint8_t *GenerateTiling();

int32_t main(int32_t argc, char *argv[]) {
  size_t aFileSize = 32 * 32 * sizeof(uint16_t);   // uint16_t represent half
  size_t bFileSize = 32 * 32 * sizeof(uint16_t);  // uint16_t represent half
  size_t cFileSize = 32 * 32 * sizeof(float);
  size_t workspaceSize = 100663296 *  sizeof(float);
  size_t tilingFileSize = 48 * sizeof(uint32_t);
  uint32_t blockDim = 1;

#ifdef __CCE_KT_TEST__
  uint8_t *a = (uint8_t *)AscendC::GmAlloc(aFileSize);
  uint8_t *b = (uint8_t *)AscendC::GmAlloc(bFileSize);
  uint8_t *c = (uint8_t *)AscendC::GmAlloc(cFileSize);
  uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);
  uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingFileSize);

  ReadFile("./input/x1_gm.bin", aFileSize, a, aFileSize);
  // PrintData(a, 16, printDataType::HALF);
  ReadFile("./input/x2_gm.bin", bFileSize, b, bFileSize);
  // PrintData(b, 16, printDataType::HALF);
  memcpy_s(tiling, tilingFileSize, GenerateTiling(), tilingFileSize);
  ReadFile("./input/workspace.bin", workspaceSize, workspace,workspaceSize);
  // PrintData(tiling, 16, printDataType::UINT32_T);
  AscendC::SetKernelMode(KernelMode::MIX_MODE);
  ICPU_RUN_KF(strassen_custom, blockDim, a, b, c, workspace, tiling);

  // PrintData(c, 16, printDataType::FLOAT);
  WriteFile("./output/output.bin", c, cFileSize);

  AscendC::GmFree((void *)a);
  AscendC::GmFree((void *)b);
  AscendC::GmFree((void *)c);
  AscendC::GmFree((void *)workspace);
  AscendC::GmFree((void *)tiling);
#else
  CHECK_ACL(aclInit(nullptr));
  aclrtContext context;
  int32_t deviceId = 0;
  CHECK_ACL(aclrtSetDevice(deviceId));
  CHECK_ACL(aclrtCreateContext(&context, deviceId));
  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream));

  uint8_t *aHost;
  uint8_t *aDevice;
  CHECK_ACL(aclrtMallocHost((void **)(&aHost), aFileSize));
  CHECK_ACL(aclrtMalloc((void **)&aDevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ReadFile("./input/x1_gm.bin", aFileSize, aHost, aFileSize);
  // PrintData(aHost, 16, printDataType::HALF);
  CHECK_ACL(aclrtMemcpy(aDevice, aFileSize, aHost, aFileSize,ACL_MEMCPY_HOST_TO_DEVICE));

  uint8_t *bHost;
  uint8_t *bDevice;
  CHECK_ACL(aclrtMallocHost((void **)(&bHost), bFileSize));
  CHECK_ACL(aclrtMalloc((void **)&bDevice, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ReadFile("./input/x2_gm.bin", bFileSize, bHost, bFileSize);
  // PrintData(bHost, 16, printDataType::HALF);
  CHECK_ACL(aclrtMemcpy(bDevice, bFileSize, bHost, bFileSize,ACL_MEMCPY_HOST_TO_DEVICE));

  uint8_t *tilingHost;
  uint8_t *tilingDevice;
  CHECK_ACL(aclrtMallocHost((void **)(&tilingHost), tilingFileSize));
  CHECK_ACL(aclrtMalloc((void **)&tilingDevice, tilingFileSize,ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMemcpy(tilingHost, tilingFileSize, GenerateTiling(),tilingFileSize, ACL_MEMCPY_HOST_TO_HOST));
  // PrintData(tilingHost, 16, printDataType::UINT32_T);
  CHECK_ACL(aclrtMemcpy(tilingDevice, tilingFileSize, tilingHost,tilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

  uint8_t *cHost;
  uint8_t *cDevice;
  CHECK_ACL(aclrtMallocHost((void **)(&cHost), cFileSize));
  CHECK_ACL(aclrtMalloc((void **)&cDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

  uint8_t *workspaceHost;
  uint8_t *workspaceDevice;
  CHECK_ACL(aclrtMallocHost((void **)(&workspaceHost), workspaceSize));
  CHECK_ACL(aclrtMalloc((void **)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ReadFile("./input/workspace.bin", workspaceSize, workspaceHost, workspaceSize);
  // PrintData(workspaceHost, 100663296, printDataType::FLOAT);
  CHECK_ACL(aclrtMemcpy(workspaceDevice, workspaceSize, workspaceHost,workspaceSize, ACL_MEMCPY_HOST_TO_DEVICE));

  ACLRT_LAUNCH_KERNEL(strassen_custom)
  (blockDim, stream, aDevice, bDevice, cDevice, workspaceDevice, tilingDevice);
  CHECK_ACL(aclrtSynchronizeStream(stream));

  CHECK_ACL(aclrtMemcpy(cHost, cFileSize, cDevice, cFileSize,ACL_MEMCPY_DEVICE_TO_HOST));
  // PrintData(cHost, 16, printDataType::FLOAT);
  WriteFile("./output/output.bin", cHost, cFileSize);
  
  CHECK_ACL(aclrtFree(aDevice));
  CHECK_ACL(aclrtFreeHost(aHost));
  CHECK_ACL(aclrtFree(bDevice));
  CHECK_ACL(aclrtFreeHost(bHost));
  CHECK_ACL(aclrtFree(workspaceDevice));
  CHECK_ACL(aclrtFreeHost(workspaceHost));
  CHECK_ACL(aclrtFree(tilingDevice));
  CHECK_ACL(aclrtFreeHost(tilingHost));
  CHECK_ACL(aclrtFree(cDevice));
  CHECK_ACL(aclrtFreeHost(cHost));
  
  CHECK_ACL(aclrtDestroyStream(stream));
  CHECK_ACL(aclrtDestroyContext(context));
  CHECK_ACL(aclrtResetDevice(deviceId));
  CHECK_ACL(aclFinalize());
#endif
  return 0;
}