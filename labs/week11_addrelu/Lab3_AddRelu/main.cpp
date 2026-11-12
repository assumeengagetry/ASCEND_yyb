/** Adapted from the supplied Huawei Ascend C 3.3 Add sample. */
#include <cstdint>
#include <cstdio>
#include <fstream>
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
extern void addrelu_custom_do(uint32_t blockDim, void *stream,
                             uint8_t *x, uint8_t *y, uint8_t *z);
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void addrelu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z);
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
    auto y = static_cast<uint8_t *>(AscendC::GmAlloc(BYTE_SIZE));
    auto z = static_cast<uint8_t *>(AscendC::GmAlloc(BYTE_SIZE));
    if (!x || !y || !z) return 1;
    if (!ReadExact("input/input_x.bin", x) || !ReadExact("input/input_y.bin", y)) return 1;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(addrelu_custom, BLOCK_DIM, x, y, z);
    bool written = WriteExact("output/output_z.bin", z);
    AscendC::GmFree(x); AscendC::GmFree(y); AscendC::GmFree(z);
    if (!written) return 1;
    std::printf("CPU_CALL_SUCCESS blockDim=8 elements=16384 dtype=float32 bytes=65536\n");
#else
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
    uint8_t *xHost = nullptr, *yHost = nullptr, *zHost = nullptr;
    uint8_t *xDevice = nullptr, *yDevice = nullptr, *zDevice = nullptr;
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&xHost), BYTE_SIZE));
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&yHost), BYTE_SIZE));
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&zHost), BYTE_SIZE));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&xDevice), BYTE_SIZE, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&yDevice), BYTE_SIZE, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&zDevice), BYTE_SIZE, ACL_MEM_MALLOC_HUGE_FIRST));
    if (!ReadExact("input/input_x.bin", xHost) || !ReadExact("input/input_y.bin", yHost)) return 1;
    CHECK_ACL(aclrtMemcpy(xDevice, BYTE_SIZE, xHost, BYTE_SIZE, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(yDevice, BYTE_SIZE, yHost, BYTE_SIZE, ACL_MEMCPY_HOST_TO_DEVICE));
    addrelu_custom_do(BLOCK_DIM, stream, xDevice, yDevice, zDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(zHost, BYTE_SIZE, zDevice, BYTE_SIZE, ACL_MEMCPY_DEVICE_TO_HOST));
    if (!WriteExact("output/output_z.bin", zHost)) return 1;
    CHECK_ACL(aclrtFree(xDevice)); CHECK_ACL(aclrtFree(yDevice)); CHECK_ACL(aclrtFree(zDevice));
    CHECK_ACL(aclrtFreeHost(xHost)); CHECK_ACL(aclrtFreeHost(yHost)); CHECK_ACL(aclrtFreeHost(zHost));
    CHECK_ACL(aclrtDestroyStream(stream)); CHECK_ACL(aclrtResetDevice(deviceId)); CHECK_ACL(aclFinalize());
    std::printf("NPU_CALL_SUCCESS blockDim=8 elements=16384 dtype=float32 bytes=65536\n");
#endif
    return 0;
}
