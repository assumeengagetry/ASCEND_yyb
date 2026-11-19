#include <acl/acl.h>
#include "aclnn_leaky_relu_custom.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static void Check(int64_t status, const char *operation) {
    if (status != 0) {
        const char *detail = aclGetRecentErrMsg();
        throw std::runtime_error(std::string(operation) + " failed, code=" +
                                 std::to_string(status) + (detail ? ": " + std::string(detail) : ""));
    }
}

struct Resources {
    bool initialized = false;
    bool deviceSet = false;
    int device = 0;
    aclrtStream stream = nullptr;
    void *input = nullptr;
    void *output = nullptr;
    void *workspace = nullptr;
    aclTensor *inputTensor = nullptr;
    aclTensor *outputTensor = nullptr;
    ~Resources() {
        if (stream) aclrtSynchronizeStream(stream);
        if (inputTensor) aclDestroyTensor(inputTensor);
        if (outputTensor) aclDestroyTensor(outputTensor);
        if (workspace) aclrtFree(workspace);
        if (input) aclrtFree(input);
        if (output) aclrtFree(output);
        if (stream) aclrtDestroyStream(stream);
        if (deviceSet) aclrtResetDevice(device);
        if (initialized) aclFinalize();
    }
};

static std::vector<int64_t> ParseShape(const std::string &text) {
    std::vector<int64_t> shape;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        size_t used = 0;
        int64_t dim = std::stoll(item, &used);
        if (used != item.size() || dim <= 0) throw std::runtime_error("shape dimensions must be positive integers");
        shape.push_back(dim);
    }
    if (shape.empty()) throw std::runtime_error("shape is empty");
    return shape;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <float32> <shape: 8,200,1024> <input.bin> <output.bin> <negative_slope>\n";
        return 2;
    }
    try {
        const std::string dtypeName = argv[1];
        const aclDataType dtype = dtypeName == "float16" ? ACL_FLOAT16 : ACL_FLOAT;
        if (dtypeName != "float32") throw std::runtime_error("unsupported dtype");
        const size_t itemSize = dtypeName == "float16" ? 2 : 4;
        const auto shape = ParseShape(argv[2]);
        std::vector<int64_t> strides(shape.size(), 1);
        uint64_t numel = 1;
        for (size_t i = shape.size(); i-- > 0;) {
            strides[i] = static_cast<int64_t>(numel);
            if (static_cast<uint64_t>(shape[i]) > std::numeric_limits<int64_t>::max() / numel)
                throw std::runtime_error("shape size overflow");
            numel *= static_cast<uint64_t>(shape[i]);
        }
        if (numel > std::numeric_limits<size_t>::max() / itemSize) throw std::runtime_error("tensor byte size overflow");
        const size_t bytes = static_cast<size_t>(numel) * itemSize;
        std::vector<char> hostInput(bytes), hostOutput(bytes);
        std::ifstream in(argv[3], std::ios::binary | std::ios::ate);
        if (!in || in.tellg() != static_cast<std::streamoff>(bytes)) throw std::runtime_error("input byte count does not match shape and dtype");
        in.seekg(0);
        if (!in.read(hostInput.data(), bytes)) throw std::runtime_error("input read failed");

        Resources r;
        r.device = 0;
        const double slope = std::stod(argv[5]);
        Check(aclInit(nullptr), "aclInit");
        r.initialized = true;
        Check(aclrtSetDevice(r.device), "aclrtSetDevice");
        r.deviceSet = true;
        Check(aclrtCreateStream(&r.stream), "aclrtCreateStream");
        Check(aclrtMalloc(&r.input, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc input");
        Check(aclrtMalloc(&r.output, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc output");
        Check(aclrtMemcpy(r.input, bytes, hostInput.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy H2D");
        Check(aclrtMemset(r.output, bytes, 0xff, bytes), "NaN initialize output");
        r.inputTensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                                       ACL_FORMAT_ND, shape.data(), shape.size(), r.input);
        r.outputTensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                                        ACL_FORMAT_ND, shape.data(), shape.size(), r.output);
        if (!r.inputTensor || !r.outputTensor) throw std::runtime_error("aclCreateTensor failed");
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        Check(aclnnLeakyReluCustomGetWorkspaceSize(r.inputTensor, slope, r.outputTensor, &workspaceSize, &executor),
              "aclnnLeakyReluCustomGetWorkspaceSize");
        if (workspaceSize) Check(aclrtMalloc(&r.workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc workspace");
        Check(aclnnLeakyReluCustom(r.workspace, workspaceSize, executor, r.stream), "aclnnLeakyReluCustom");
        Check(aclrtSynchronizeStream(r.stream), "aclrtSynchronizeStream");
        Check(aclrtMemcpy(hostOutput.data(), bytes, r.output, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy D2H");
        std::ofstream out(argv[4], std::ios::binary);
        if (!out || !out.write(hostOutput.data(), bytes)) throw std::runtime_error("output write failed");
        std::cout << "NPU_CALL_SUCCESS dtype=" << dtypeName << " shape=" << argv[2] << " numel=" << numel
                  << " negative_slope=" << slope << " device=" << r.device << " workspace_bytes=" << workspaceSize << "\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }
}
