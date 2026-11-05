/**
 * @file main.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "acl/acl.h"
#include <cstdio>
extern void hello_world_do(uint32_t coreDim, void *stream);

#define ACL_CHECK(call) do { \
    const aclError error = (call); \
    if (error != ACL_SUCCESS) { \
        std::fprintf(stderr, "[ERROR] %s failed: %d\n", #call, error); \
        return 1; \
    } \
} while (0)

int32_t main(int argc, char const *argv[])
{
    ACL_CHECK(aclInit(nullptr));
    int32_t deviceId = 0;
    ACL_CHECK(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    constexpr uint32_t blockDim = 8;
    hello_world_do(blockDim, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());
    return 0;
}
