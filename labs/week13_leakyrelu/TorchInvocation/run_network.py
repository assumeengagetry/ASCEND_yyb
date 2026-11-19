"""Compare original and custom-activation MLP inference on the same real NPU."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
import torch_npu

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)
SEED = 20261119
SLOPE = 0.1
RTOL, ATOL = 1e-4, 1e-5

torch.manual_seed(SEED)
torch.npu.set_device(0)
# Direct ACL adapter accepts ND storage, so disable internal blocked layouts.
torch.npu.config.allow_internal_format = False
torch.ops.load_library(str(ROOT / 'build/week13_leakyrelu_npu.so'))

class MLP(torch.nn.Module):
    def __init__(self, custom=False):
        super().__init__()
        self.fc1 = torch.nn.Linear(1024, 1024)
        self.activation = torch.nn.LeakyReLU(SLOPE)
        self.fc2 = torch.nn.Linear(1024, 64)
        self.custom = custom
    def forward(self, x):
        hidden = self.fc1(x)
        # Replacement location: the activation between fc1 and fc2.
        # Output shape of fc1 is [1600,1024], exactly 8*200*1024 float32 values.
        activated = (torch.ops.week13.leaky_relu(hidden, SLOPE) if self.custom
                     else self.activation(hidden))
        return self.fc2(activated)

def compare(actual, reference):
    a, r = actual.astype(np.float64), reference.astype(np.float64)
    error = np.abs(a-r)
    good = np.isfinite(a) & np.isfinite(r) & (error <= ATOL + RTOL*np.abs(r))
    return {'passed': bool(good.all()), 'numel': int(a.size),
            'failed_elements': int((~good).sum()), 'max_abs_error': float(error.max()),
            'mean_abs_error': float(error.mean()),
            'max_rel_error': float((error/np.maximum(np.abs(r), 1e-12)).max())}

# Initialize once on CPU, copy exact weights to the two NPU networks.
base_cpu = MLP().eval()
custom_cpu = copy.deepcopy(base_cpu)
custom_cpu.custom = True
x_cpu = torch.randn(1600, 1024, dtype=torch.float32)
np.save(RESULTS / 'input.npy', x_cpu.numpy())
torch.save(base_cpu.state_dict(), RESULTS / 'weights.pt')
base_npu = base_cpu.npu().eval()
custom_npu = custom_cpu.npu().eval()
x_npu = x_cpu.npu()
count_before = torch.ops.week13.call_count()
with torch.inference_mode():
    hidden = base_npu.fc1(x_npu)
    built_in_activation = torch.nn.functional.leaky_relu(hidden, SLOPE)
    custom_activation = torch.ops.week13.leaky_relu(hidden, SLOPE)
    torch.npu.synchronize()
    hidden_np = hidden.cpu().numpy()
    np.save(RESULTS / 'activation_input.npy', hidden_np)
    expected_activation = np.where(hidden_np >= 0, hidden_np, np.float32(SLOPE)*hidden_np)
    actual_activation = custom_activation.cpu().numpy()
    builtin_activation_np = built_in_activation.cpu().numpy()
    np.save(RESULTS / 'activation_custom.npy', actual_activation)
    np.save(RESULTS / 'activation_builtin.npy', builtin_activation_np)
    np.save(RESULTS / 'activation_golden.npy', expected_activation)
    original_output = base_npu(x_npu)
    replaced_output = custom_npu(x_npu)
    torch.npu.synchronize()
    original_np = original_output.cpu().numpy()
    replaced_np = replaced_output.cpu().numpy()
np.save(RESULTS / 'original_output.npy', original_np)
np.save(RESULTS / 'replaced_output.npy', replaced_np)

summary = {
    'seed': SEED, 'device': str(x_npu.device), 'torch': torch.__version__,
    'torch_npu': torch_npu.__version__, 'negative_slope': SLOPE,
    'network': 'Linear(1024,1024) -> LeakyReLU(0.1) -> Linear(1024,64)',
    'input_shape': [1600, 1024], 'output_shape': [1600, 64],
    'replacement': 'activation between fc1 and fc2',
    'custom_api': 'torch.ops.week13.leaky_relu -> C++ PrivateUse1 -> aclnnLeakyReluCustom',
    'rtol': RTOL, 'atol': ATOL,
    'direct_activation_vs_math': compare(actual_activation, expected_activation),
    'direct_activation_vs_builtin_npu': compare(actual_activation, builtin_activation_np),
    'network_original_vs_replaced': compare(replaced_np, original_np),
    'custom_kernel_calls': int(torch.ops.week13.call_count() - count_before),
    'negative_activation_inputs': int((hidden_np < 0).sum()),
    'nonnegative_activation_inputs': int((hidden_np >= 0).sum()),
    'first_original_values': original_np.reshape(-1)[:8].tolist(),
    'first_replaced_values': replaced_np.reshape(-1)[:8].tolist(),
    'scope': 'inference only; autograd and performance not evaluated',
}
summary['passed'] = all(summary[k]['passed'] for k in (
    'direct_activation_vs_math', 'direct_activation_vs_builtin_npu', 'network_original_vs_replaced')) and summary['custom_kernel_calls'] >= 2
(RESULTS / 'verification.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
print('NETWORK_REPLACEMENT_PASS' if summary['passed'] else 'NETWORK_REPLACEMENT_FAIL')
raise SystemExit(0 if summary['passed'] else 1)
