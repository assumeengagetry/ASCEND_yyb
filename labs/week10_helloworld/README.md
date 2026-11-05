# 第10周 HelloWorld 算子实验

姓名：许盛凯　学号：2024141460445　日期：2026年11月5日

## 目录

- `Lab2_helloworld/`：完整可编译算子工程。
- `evidence/`：环境、构建、两次NPU运行日志、结构化核验结果与原始回显排版图。
- `original_source/`：用户提供RAR中“2.3.4调用一个昇腾算子”的原始代码。

## 实测环境及复现

华为云ModelArts，Huawei Cloud EulerOS 2.0 aarch64，Ascend910B4，CANN8.5.2。
在有相同NPU及Toolkit的机器上：

```bash
cd Lab2_helloworld
bash run.sh -v Ascend910B4
```

脚本优先使用 `ASCEND_INSTALL_PATH` 或 `ASCEND_HOME_PATH`；否则检查系统安装路径，再使用用户安装路径。默认系统路径为 `/usr/local/Ascend/ascend-toolkit/latest`。CPU或无NPU的机器无法直接运行此工程。

验收：程序返回0，输出包含8条 `Hello World!!!`；原始日志中应出现AIV-0至AIV-7，`block dim=8`、`total_block_num=8`。最终脚本打印 `[PASS] Hello World lines: 8 / 8`。

## 源码来源与改动

源自所提供的《Ascend C异构并行程序设计源代码.rar》内“2.3.4调用一个昇腾算子”，对应讲义的Lab2_helloworld。原始源码已包含printf及8核常量，不存在待填空占位符。本次保留核心逻辑，修改默认SOC为Ascend910B4，显式调用AscendC::printf，增加ACL返回值检查，并使脚本在编译/运行失败时立即停止、检查管道退出状态及问候行数。

未修改系统Toolkit或Shell启动配置；工程位于云端持久目录 `/home/ma-user/work/ascend_week10_20261105/`。无需自定义OPP包安装。

运行图说明：`evidence/06_npu_output_render.png` 由真实NPU日志排版生成，保留8组核编号、版本及问候语。图中仅省略冗长的打印缓冲区元数据；完整原始字段见03_npu_run.log。该图没有伪称为原生终端截图。
