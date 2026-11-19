/**
 * Based on Huawei's supplied 3.6.1 Sinh and 3.3 vector programming samples.
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
 * float32 Sinh(x) = (exp(x) - exp(-x)) * 0.5 for x in [1, 10].
 * Only the Ascend C basic vector APIs are used.
 */
#include "kernel_operator.h"

constexpr int32_t TOTAL_LENGTH = 8 * 2048;
constexpr int32_t USE_CORE_NUM = 8;
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;
constexpr int32_t TILE_NUM = 8;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM;
static_assert(TILE_LENGTH == 128 && TILE_LENGTH * sizeof(float) % 32 == 0,
              "Each tile must be 32-byte aligned.");

class KernelSinhTBuf {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z)
    {
        xGm.SetGlobalBuffer((__gm__ float *)x + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        zGm.SetGlobalBuffer((__gm__ float *)z + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(expBuffer, TILE_LENGTH * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < TILE_NUM * BUFFER_NUM; ++i) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[progress * TILE_LENGTH], TILE_LENGTH);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        // A named temporary keeps exp(x) separate from the original input tile.
        AscendC::LocalTensor<float> expLocal = expBuffer.Get<float>();
        AscendC::Exp(expLocal, xLocal, TILE_LENGTH);
        // Compute exp(-x) directly; xLocal remains unchanged throughout Compute.
        AscendC::Muls(zLocal, xLocal, -1.0f, TILE_LENGTH);
        AscendC::Exp(zLocal, zLocal, TILE_LENGTH);
        AscendC::Sub(zLocal, expLocal, zLocal, TILE_LENGTH);
        AscendC::Muls(zLocal, zLocal, 0.5f, TILE_LENGTH);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        AscendC::DataCopy(zGm[progress * TILE_LENGTH], zLocal, TILE_LENGTH);
        outQueueZ.FreeTensor(zLocal);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> expBuffer;
    AscendC::GlobalTensor<float> xGm, zGm;
};

extern "C" __global__ __aicore__ void sinhtbuf_custom(GM_ADDR x, GM_ADDR z)
{
    KernelSinhTBuf op;
    op.Init(x, z);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void sinhtbuf_custom_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *z)
{
    sinhtbuf_custom<<<blockDim, nullptr, stream>>>(x, z);
}
#endif
