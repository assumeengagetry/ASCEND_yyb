from pathlib import Path
import json
import os
import subprocess
import sysconfig
import torch
import torch_npu
from torch.utils.cpp_extension import include_paths, library_paths

root = Path(__file__).resolve().parent
cann = Path(os.environ.get('ASCEND_HOME_PATH', '/usr/local/Ascend/ascend-toolkit/latest'))
npu = Path(torch_npu.__file__).resolve().parent
vendor = Path(os.environ.get('WEEK13_OP_API', str(root.parent / 'custom_opp/vendors/lab_week13/op_api')))
assert (vendor / 'include/aclnn_leaky_relu_custom.h').exists(), vendor
build = root / 'build'
build.mkdir(exist_ok=True)
output = build / 'week13_leakyrelu_npu.so'
includes = include_paths() + [str(npu/'include'), str(cann/'include'), str(vendor/'include'), sysconfig.get_path('include')]
libraries = library_paths() + [str(npu/'lib'), str(cann/'lib64'), str(vendor/'lib')]
abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
command = [os.environ.get('CXX', 'g++'), '-O2', '-std=c++17', '-fPIC', '-shared',
           f'-D_GLIBCXX_USE_CXX11_ABI={abi}', str(root/'leaky_relu_npu.cpp'), '-o', str(output)]
command += ['-I'+p for p in includes]
command += ['-L'+p for p in libraries]
command += ['-Wl,-rpath,'+p for p in libraries]
command += ['-ltorch', '-ltorch_cpu', '-lc10', '-ltorch_npu', '-lcust_opapi', '-lnnopbase', '-lascendcl']
(build/'compile_command.json').write_text(json.dumps(command, indent=2)+'\n')
print('BUILD_COMMAND', ' '.join(command), flush=True)
subprocess.run(command, check=True)
torch.ops.load_library(str(output))
print('EXTENSION_BUILD_SUCCESS', output)
print('torch', torch.__version__, 'torch_npu', torch_npu.__version__, 'CXX11_ABI', abi)
