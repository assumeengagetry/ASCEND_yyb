/** Unary float32 Sinh KernelLaunch host, adapted from the supplied Ascend C samples. */
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
extern void sinh_custom_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *z);
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void sinh_custom(GM_ADDR x, GM_ADDR z);
#endif

constexpr uint32_t BLOCK_DIM = 8;
constexpr size_t ELEMENT_COUNT = 8 * 2048;
constexpr size_t BYTE_SIZE = ELEMENT_COUNT * sizeof(float);

bool ReadExact(const char *path, void *buffer) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(BYTE_SIZE)) {
        std::fprintf(stderr, "[ERROR] %s must contain %zu bytes\n", path, BYTE_SIZE);
        return false;
    }
    file.seekg(0);
    return static_cast<bool>(file.read(static_cast<char *>(buffer), BYTE_SIZE));
}
bool WriteExact(const char *path, const void *buffer) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    return static_cast<bool>(file.write(static_cast<const char *>(buffer), BYTE_SIZE));
}
#ifndef ASCENDC_CPU_DEBUG
#define CHECK_ACL(call) do { \
    aclError code = (call); \
    if (code != ACL_SUCCESS) { \
        std::fprintf(stderr, "[ERROR] %s: %d\n", #call, code); return 1; \
    } \
} while (0)
#endif

int main() {
#ifdef ASCENDC_CPU_DEBUG
    auto x = static_cast<uint8_t *>(AscendC::GmAlloc(BYTE_SIZE));
    auto z = static_cast<uint8_t *>(AscendC::GmAlloc(BYTE_SIZE));
    if (!x || !z) return 1;
    if (!ReadExact("input/input_x.bin", x)) return 1;
    // Fill output with NaNs so omitted tiles cannot accidentally pass validation.
    std::memset(z, 0xff, BYTE_SIZE);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(sinh_custom, BLOCK_DIM, x, z);
    bool written = WriteExact("output/output_z.bin", z);
    AscendC::GmFree(x); AscendC::GmFree(z);
    if (!written) return 1;
#else
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
    uint8_t *xHost = nullptr, *zHost = nullptr;
    uint8_t *xDevice = nullptr, *zDevice = nullptr;
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&xHost), BYTE_SIZE));
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&zHost), BYTE_SIZE));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&xDevice), BYTE_SIZE, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&zDevice), BYTE_SIZE, ACL_MEM_MALLOC_HUGE_FIRST));
    if (!ReadExact("input/input_x.bin", xHost)) return 1;
    CHECK_ACL(aclrtMemcpy(xDevice, BYTE_SIZE, xHost, BYTE_SIZE, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemset(zDevice, BYTE_SIZE, 0xff, BYTE_SIZE));
    sinh_custom_do(BLOCK_DIM, stream, xDevice, zDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(zHost, BYTE_SIZE, zDevice, BYTE_SIZE, ACL_MEMCPY_DEVICE_TO_HOST));
    if (!WriteExact("output/output_z.bin", zHost)) return 1;
    CHECK_ACL(aclrtFree(xDevice)); CHECK_ACL(aclrtFree(zDevice));
    CHECK_ACL(aclrtFreeHost(xHost)); CHECK_ACL(aclrtFreeHost(zHost));
    CHECK_ACL(aclrtDestroyStream(stream)); CHECK_ACL(aclrtResetDevice(deviceId)); CHECK_ACL(aclFinalize());
#endif
    std::printf("CALL_SUCCESS variant=Sinh run_mode=%s blockDim=8 elements=16384 dtype=float32 bytes=65536\n", RUN_MODE_LABEL);
    return 0;
}
