# 第13周 算子工程化开发及整网调用

姓名：许盛凯　学号：2024141460445

## 实验完成情况

本实验从`LeakyReluCustom.json`出发，实际调用当前CANN的`msopgen gen -c ai_core-Ascend910B4 -lan cpp`生成工程，补全host侧算子注册、属性读取、shape/dtype推导及tiling，编写Ascend C核函数，编译源码与二进制安装包，安装到本周隔离目录，再通过生成的aclnn两段式接口执行单算子与PyTorch整网调用。

核心算子：`y = max(x,0) + a*min(x,0)`。本题输入与输出均为float32、ND，shape `[8,200,1024]`，共1,638,400元素；`a`是float属性`negative_slope`，默认0.1，必须有限且非0。使用该公式可正确处理大于1的斜率，不采用仅适用于部分斜率范围的`max(x,a*x)`。补充支持任意正元素数量（含尾块）；PyTorch插件接口限定连续ND且元素数为8的倍数，课程规模满足该限制。

## 环境

Huawei Cloud ModelArts、Huawei Cloud EulerOS 2.0 aarch64、Ascend910B4、CANN8.5.2、GCC10.3.1、CMake3.22.0、Python3.12.0、NumPy1.26.4、PyTorch2.7.1+cpu及torch_npu2.7.1.post2。PyTorch版本字符串中的+cpu不代表本实验在CPU运行：Ascend支持由torch_npu提供，运行结果记录设备为npu:0。

云端工程位于第13周实验的持久工作目录。
安装位置：`custom_opp/vendors/lab_week13`，仅通过当前进程环境导入，不覆盖系统算子目录或torch_npu安装。

## 工程与复现

- `LeakyReluCustom.json`：msopgen创建描述，包含float属性。
- `generated_template/`：本次msopgen生成、补全前的原始模板。
- `LeakyReluCustom/`：完整补全后的算子工程。
- `create_project.sh [目标目录]`：重新生成模板并复制已实现的host/kernel/config，默认输出新目录`GeneratedLeakyReluCustom`；已有目标会拒绝覆盖。
- `build_install.sh`：源码与二进制编译、生成安装包并隔离部署。
- `AclNNInvocation/`：C++两段式aclnn调用、五组数据生成和逐元素验证。
- `TorchInvocation/`：C++ PrivateUse1插件、原/替换MLP、原始数据与验证结果。
- `package/custom_opp_hce_aarch64.run`：本次成功构建并部署的二进制安装包。
- `show_results.sh`：只读显示已经完成的真实验证结果，供屏显截图使用，不重新占用NPU。

```bash
# 在同类Ascend910B4 / CANN8.5.2环境，从本目录执行
bash create_project.sh GeneratedLeakyReluCustom
bash build_install.sh
bash AclNNInvocation/run.sh
bash TorchInvocation/run.sh
bash show_results.sh
```

`build_install.sh`第一阶段关闭二进制打包、生成并部署源码包，供后续编译找到注册元数据；第二阶段开启二进制，完成binary目标、重新打包并部署。首次直接开启binary后先package因二进制目录尚不存在失败（02日志）；随后修正为分阶段后，旧环境包名euleros与本次hce不一致（03日志），最终使用实际生成的`custom_opp_*_aarch64.run`后成功（04日志）。这些是构建脚本适配问题，未通过改变数值验证标准处理。

## 分块与计算

host以2048元素为一片，最多启动8个执行块。第b块负责片号b、b+blockDim等；末片使用DataCopyPad准确处理实际长度。每个输入、输出队列各有2个2048float缓冲，另有1个2048float临时缓冲。基础API为Maxs、Mins、Muls、Add；显式PIPE_V屏障表达依赖。Host从属性读取a并写入tiling，核函数使用该动态属性。

aclnn应用先创建输入输出aclTensor，再调用`aclnnLeakyReluCustomGetWorkspaceSize`获取workspace与executor，随后执行`aclnnLeakyReluCustom`，同步流、拷回真实设备输出。输出先初始化为NaN，避免遗漏写回通过检查。

## 单算子实测

`AclNNInvocation/results/verification_all.json`记录5/5通过：

| 用例 | 元素数 | a | 最大绝对误差 | 失败元素 |
|---|---:|---:|---:|---:|
| 随机与边界 | 1638400 | 0.1 | 0 | 0 |
| 同一随机输入 | 1638400 | 1.5 | 0 | 0 |
| 全负输入（含0） | 1638400 | 0.2 | 0 | 0 |
| 全正输入（含0） | 1638400 | 0.1 | 0 | 0 |
| 尾块补充 | 257 | 0.1 | 0 | 0 |

使用固定随机种子（见数据生成脚本），输入范围[-10,10]，显式加入0、正负端点、小数以及片边界值。数学基准先以float64计算`x`或`float32(a)*x`，再舍入float32。逐元素标准为`abs(actual-reference) <= 1e-6 + 1e-6*abs(reference)`，检查长度与有限值，任一元素失败即不通过。每组保存input.bin、golden.bin、output.bin、运行日志和验证JSON。

## PyTorch框架适配与整网替换

课件给出旧版torch_npu整仓重编示例。本次使用与现有PyTorch2.7.1/torch_npu2.7.1.post2匹配的独立C++扩展，不替换系统框架。插件通过`TORCH_LIBRARY`声明schema，`TORCH_LIBRARY_IMPL(..., PrivateUse1, ...)`注册NPU实现，直接调用生成的两段式aclnn接口。链路为：

`torch.ops.week13.leaky_relu → C++ PrivateUse1 dispatcher → aclnnLeakyReluCustomGetWorkspaceSize / aclnnLeakyReluCustom → Ascend C kernel`

这不是在Python中用内置激活或where代替设备核函数。插件根据当前NPU流同步，调用后同步完成再释放描述符/工作区；记录实际调用计数。为了准确连接连续ND数据指针，关闭该进程的torch_npu内部blocked格式，检查dtype和连续性。环境未装Ninja，编译脚本读取当前PyTorch头文件路径、库路径与CXX11_ABI=1，以g++直接生成共享库；未安装其它依赖或替换框架。

网络是`Linear(1024,1024) → LeakyReLU(0.1) → Linear(1024,64)`，输入`[1600,1024]`，第一层输出含1,638,400个元素。替换位置是fc1与fc2之间的activation，源码中已有注释。两个网络从同一CPU初始化复制完全相同的权重，再同在NPU上推理；相同输入、固定随机种子。

`TorchInvocation/results/verification.json`记录：

- 自定义激活与数学基准：1,638,400元素，最大绝对误差0，失败0。
- 自定义激活与内置NPU激活：1,638,400元素，最大绝对误差0，失败0。
- 原网络与替换网络：输出`[1600,64]`，102,400元素，最大绝对误差0，失败0。
- 自定义aclnn调用计数2（一次独立激活检查，一次整网替换）。

整网比较固定容差rtol=1e-4、atol=1e-5，未放宽。保存输入、权重、激活输入/输出/基准、原网络和替换网络输出。实验只覆盖推理，不包含autograd/训练与性能测量；直接同步适配器不是吞吐优化方案。

## 证据

`evidence/00_environment.log`为当日实测环境；01为msopgen；02/03保留构建适配失败；04成功构建和安装；05为aclnn Host编译；06为5组NPU单算子验证；07为整网NPU运行输出。报告截图由本地网页控制台通过SSH执行show_results.sh，明确显示归档的真实结果；不冒称再次运行内核。
