// PyTorch dispatcher adapter for the installed, custom Ascend C LeakyRelu kernel.
// This is an inference-only adapter. It does not substitute a built-in activation.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/core/DeviceGuard.h>
#include <atomic>
#include <cmath>
#include <vector>
#include "acl/acl.h"
#include "aclnn_leaky_relu_custom.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace {
std::atomic<int64_t> calls{0};
void Check(int result, const char* what) {
    TORCH_CHECK(result == ACL_SUCCESS, what, " failed, ACL code=", result);
}
at::Tensor LeakyReluNpu(const at::Tensor& x, double negativeSlope) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "NPU tensor required");
    TORCH_CHECK(x.scalar_type() == at::kFloat && x.is_contiguous(), "contiguous float32 ND tensor required");
    TORCH_CHECK(x.numel() > 0 && x.numel() % 8 == 0, "element count must be a positive multiple of 8");
    TORCH_CHECK(std::isfinite(negativeSlope) && negativeSlope != 0.0, "finite, nonzero negative slope required");
    c10::DeviceGuard guard(x.device());
    auto currentStream = c10_npu::getCurrentNPUStream(x.device().index());
    // Drain torch_npu's task queue before passing the data pointers to direct ACL.
    currentStream.synchronize();
    at::Tensor y = at::empty_like(x);
    std::vector<int64_t> dims(x.sizes().begin(), x.sizes().end());
    std::vector<int64_t> strides(x.strides().begin(), x.strides().end());
    aclTensor* input = aclCreateTensor(dims.data(), dims.size(), ACL_FLOAT,
        strides.data(), 0, ACL_FORMAT_ND, dims.data(), dims.size(), x.data_ptr());
    aclTensor* output = aclCreateTensor(dims.data(), dims.size(), ACL_FLOAT,
        strides.data(), 0, ACL_FORMAT_ND, dims.data(), dims.size(), y.data_ptr());
    TORCH_CHECK(input && output, "aclCreateTensor failed");
    uint64_t workspaceBytes = 0;
    aclOpExecutor* executor = nullptr;
    auto sizeCode = aclnnLeakyReluCustomGetWorkspaceSize(input, negativeSlope, output, &workspaceBytes, &executor);
    if (sizeCode != ACL_SUCCESS) {
        aclDestroyTensor(input); aclDestroyTensor(output); Check(sizeCode, "GetWorkspaceSize");
    }
    at::Tensor workspace = at::empty({static_cast<int64_t>(workspaceBytes)}, x.options().dtype(at::kByte));
    void* workspacePtr = workspaceBytes ? workspace.data_ptr() : nullptr;
    auto runCode = aclnnLeakyReluCustom(workspacePtr, workspaceBytes, executor, currentStream.stream());
    auto syncCode = aclrtSynchronizeStream(currentStream.stream());
    aclDestroyTensor(input); aclDestroyTensor(output);
    Check(runCode, "aclnnLeakyReluCustom"); Check(syncCode, "aclrtSynchronizeStream");
    ++calls;
    return y;
}
}
TORCH_LIBRARY(week13, m) {
    m.def("leaky_relu(Tensor x, float negative_slope=0.1) -> Tensor");
    m.def("call_count() -> int", []() -> int64_t { return calls.load(); });
}
TORCH_LIBRARY_IMPL(week13, PrivateUse1, m) {
    m.impl("leaky_relu", &LeakyReluNpu);
}
