# Ascend C Sigmoid 实验

许盛凯 2024141460445，实验日期 2026-10-29。

本工程完成教材要求的 Sigmoid 算子实现、编译、隔离安装和 ACLNN NPU 调用验证。工程由预装 CANN 的 msOpGen 生成，参考上传 RAR 中的 `4.1-4.5Ascend C算子开发流程/AddCustom` 和 `AclNNInvocation`。RAR 未包含微认证原始 SigmoidCustom 试题包；此工程为按课程描述完成的实验实现，16 组结果来自自行设计的公开测试，不代表认证平台判分结果。

## 已实测环境

- Huawei ModelArts，ma-user，EulerOS 2.0 SP10，aarch64。
- Ascend 910B4，npu-smi 25.5.1，CANN 8.0.0 用户目录 Toolkit。
- Python 3.9.10，CMake 3.16.5，GCC 7.3.0。
- ND 连续张量，float16 和 float32，非空 shape；输入输出同形同类型。

## 在原云主机复现

```bash
cd /home/ma-user/work/ascend_lab_20261029
source env.sh
cd SigmoidCustom
bash run.sh
```

完整重建和部署：

```bash
source /home/ma-user/work/ascend_lab_20261029/env.sh
cd /home/ma-user/work/ascend_lab_20261029/SigmoidCustom
bash rebuild.sh
bash run.sh
```

工程复制到其他目录时，先把整个实验目录解压，再运行 `bash setup_env.sh` 和 `source env.sh`。`env.sh` 的 Toolkit 路径对应原机器，其他机器须按实际 CANN 安装位置修改。Python venv 和系统 Toolkit 未打入 ZIP。

仅安装已编译算子包：

```bash
source env.sh
bash custom_opp_euleros_aarch64.run --quiet --install-path="$ASCEND_LAB_ROOT/install"
cd SigmoidCustom
bash run.sh
```

该安装包仅在本次 EulerOS aarch64、CANN 8.0.0、Ascend 910B4 环境验证。

## 验证

`bash run.sh` 运行 16 组案例：float16/float32 × 8 种 shape。含 [8,2048]、[1]、[15]、[17]、[255]、[257]、[4096]、[65536]。输入是固定随机种子 20261029 的 [-12,12] 均匀样本和 [-1000,1000] 内的有限极值。参考值从实际量化后的输入以 NumPy float64 稳定公式计算。

逐元素条件为 `abs(actual-reference) <= atol + rtol*abs(reference)`。float16 使用 atol=1e-3、rtol=1e-3；float32 使用 atol=1e-6、rtol=1e-4。16/16 PASS，所有失败元素数为0。float16 最大绝对误差约 2.44048503e-4，float32 约 1.02409965e-7。

原始输入输出、float64 基准、每例 NPU 回显及结构化结果位于 `SigmoidCustom/AclNNInvocation/results/`。配置、依赖安装、编译、部署和调试回显位于同级目录外的 [`../evidence/`](../evidence/)。完整 CANN Toolkit 为镜像预装，历史安装日志为2025年，本次未重新下载或安装 Toolkit。

## 主要文件

- `SigmoidCustom/op_host/`：注册、shape/type 推导、64位元素数和分片计算。
- `SigmoidCustom/op_kernel/`：Ascend C 核函数，2048元素一片，最多8核循环分片，DataCopyPad尾块。
- `SigmoidCustom/AclNNInvocation/`：两阶段自定义 ACLNN 接口及 NumPy 精度验证。
- `SigmoidCustom/rebuild.sh`：配置、host打包、kernel编译、最终打包及隔离安装。
- `env.sh`：激活 CANN、专用 venv、CA证书及自定义算子路径。
- `requirements.txt`：本次实测的13项 Python 库版本。

实现使用 float32 中间结果计算 `1 / (1 + exp(-x))`，最终使用 `Div`，避免 910B 上单独 `Reciprocal` 近似误差过大。非连续张量、空张量、NaN/Inf 和性能未纳入本次验证。
