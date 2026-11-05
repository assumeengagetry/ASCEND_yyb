# SigmoidCustom ACLNN NPU 验证

先在实验根目录 source env.sh，并完成 SigmoidCustom 的编译安装，再执行：

```bash
bash run.sh
```

默认测试 float16 和 float32，shape 分别为 `(8,2048)`、`(1)`、`(15)`、`(17)`、`(255)`、`(257)`、`(4096)`、`(65536)`。输入包含固定随机数据和有限极值。NumPy float64 稳定 sigmoid 为参考值；参考值从实际量化后的输入计算。

单组测试：

```bash
bash run.sh --dtype float32 --shape 8,2048
bash run.sh --dtype float16 --numel 257
```

设备逻辑 ID 默认为 0，可用 `--device` 修改。程序调用生成的 `aclnnSigmoidCustomGetWorkspaceSize` 和 `aclnnSigmoidCustom`，输出逐组误差和 PASS/FAIL；`results/verification.json` 保存结构化结果，各子目录保存输入、输出、float64 参考值及真实 NPU 运行日志。

float32 误差条件为 `abs(actual - expected) <= 1e-6 + 1e-4 * abs(expected)`；float16 条件为 `abs(actual - expected) <= 1e-3 + 1e-3 * abs(expected)`。

可通过 `ASCEND_CANN_PACKAGE_PATH` 和 `OP_API_DIR` 指定 CANN 和自定义算子 API 安装位置。脚本仅在当前工程生成构建与验证结果，不清理用户日志。
