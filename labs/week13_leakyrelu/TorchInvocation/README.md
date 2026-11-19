# PyTorch自定义LeakyRelu整网适配

本目录针对当前环境的PyTorch 2.7.1与torch_npu 2.7.1.post2构建独立C++扩展。未重新编译或替换系统torch_npu，采用PyTorch公开的TORCH_LIBRARY注册分发机制，将NPU（PrivateUse1）设备上的`torch.ops.week13.leaky_relu`映射到本周生成、编译、安装的自定义算子两段式接口。

调用链为：

```text
torch.ops.week13.leaky_relu(hidden, 0.1)
  -> C++ PrivateUse1 adapter
  -> aclnnLeakyReluCustomGetWorkspaceSize
  -> aclnnLeakyReluCustom
  -> installed Ascend C leaky_relu_custom kernel
```

这与课件旧版torch_npu 2.1.0/op-plugin源码注册的目标一致，但使用当前环境支持的独立扩展方式。C++适配器没有调用内置LeakyReLU或逐元素PyTorch表达式计算输出。Python中的NumPy公式只用于独立数学基准。

## 复现

先完成本周LeakyReluCustom算子安装，使`../custom_opp/vendors/lab_week13/op_api/`下的头文件和`libcust_opapi.so`可用，再执行：

```bash
bash run.sh
```

`build_extension.py`通过已安装PyTorch的C++扩展工具构建`build/week13_leakyrelu_npu.so`，并动态注册算子。`run_network.py`在真实NPU上运行基准与替换后的网络。

## 网络与替换位置

网络为 `Linear(1024,1024) -> LeakyReLU(0.1) -> Linear(1024,64)`。输入为`[1600,1024]`，中间激活为`[1600,1024]`，总计1,638,400个float32元素，等于课件8×200×1024。替换位置在第一层线性层和第二层线性层之间。两网使用完全相同的权重和输入，只改变激活调用。代码关闭内部阻塞格式，以确保交给ACL接口的张量使用ND物理布局。

程序单独比较自定义激活与NumPy数学基准、与内置NPU激活，并比较原始和替换后的整网输出。验证要求全部元素满足`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`，不存在失败比例豁免。C++内部调用计数器验证实际执行了自定义aclnn接口。

`results/`保存输入、权重、激活输入、三种激活结果、两种网络输出及汇总`verification.json`。本实验只验证推理，没有注册autograd，也未用同步适配器测量性能。为保证直接ACL调用与torch_npu异步队列顺序一致，适配器在接口调用前后同步当前流。
