# 第11周 向量编程基础算子 AddRelu

姓名：许盛凯　学号：2024141460445　实验日期：2026年11月12日

## 实验内容

实现 z=max(x+y,0)，输入为float32、ND、shape 8×2048、取值[-10,10]。启动8个逻辑执行块，每核处理2048个元素；TILE_NUM=8、BUFFER_NUM=2、TILE_LENGTH=128，每核循环16次。核心计算使用融合双目指令 `AscendC::AddRelu`，输入/输出通过CopyIn、Compute、CopyOut及双缓冲队列传递。

源码基于所给RAR的“3.3向量编程范式”Add工程改造，原始文件保存在 `original_source/`。RAR没有独立的Lab3_AddRelu目录。讲义正文要求float，但main示例残留sizeof(uint16_t)；本实验统一为float32与sizeof(float)，每份张量65536字节。

## 环境及复现

本次实机：Huawei Cloud EulerOS 2.0 aarch64，Ascend910B4，CANN8.5.2，CMake3.22.0，GCC10.3.1。云端持久目录：`/home/ma-user/work/ascend_week11_20261112/`。

在同类环境中运行：

```bash
cd Lab3_AddRelu
bash run.sh -r cpu -v Ascend910B4
bash run.sh -r npu -v Ascend910B4
bash run_all.sh
# 可选：补充仿真随机用例
bash run.sh -r sim -v Ascend910B4
```

`run.sh`默认运行random用例；可增加 `--case boundary`，或选择all_negative、all_positive、cancellation。默认随机种子20261112。CPU使用CANN的tikicpulib执行相同核函数；NPU使用ACL申请设备内存并通过KernelLaunch调用，不用NumPy替代设备计算。

本次sim随机用例也通过：16384元素，失败0、最大绝对误差0。CANN8.5.2模型库依赖未完整传递到链接接口，已在工程CMake的sim分支补入model_top、stars、ffts_model、mcu_wrapper、mcu_loop和pem_davinci，用`--no-as-needed`保留依赖；无需修改系统库。NPU/sim模式在配置前重新生成对应build目录，避免厂商对象合并脚本复用已转换的.o；CPU保留增量构建。

模拟器以Ascend910B4构建，但厂商模型日志中可见Ascend910B1共享配置路径；该记录不替代真实910B4验证，也不作为硬件耗时或性能结论。

## 结果与证据

- `Lab3_AddRelu/results/`：CPU和NPU各5组，输入bin、NPU/CPU原始输出bin、golden、metadata及运行与验证日志。
- `verification_all.json`：10组汇总，全部通过，每组失败元素0，最大绝对误差0。
- `evidence/`：环境、编译、单组运行与回归日志及屏显。
- `Lab3_AddRelu/results_extra/sim_random/`：补充的模拟器随机测试原始输入、输出、golden与校验；不计入CPU/NPU的10组汇总。
- `evidence/08_sim_build.log`保留首次仿真链接失败，`09_sim_linkfix_build_and_run.log`记录修复后的构建运行，`10_final_npu_recheck.log`记录最终脚本的NPU复验。
- `scripts/verify_result.py`：检查元素数、文件字节数、有限值、输出非负、独立float64相加后float32舍入基准及逐元素误差（rtol=atol=1e-6），任何失败元素均不通过。

单独验证归档结果：在Lab3_AddRelu中执行

```bash
python3 scripts/verify_result.py results/npu_random/output/output_z.bin \
  results/npu_random/output/golden.bin \
  --input-x results/npu_random/input/input_x.bin \
  --input-y results/npu_random/input/input_y.bin \
  --json /tmp/addrelu_recheck.json
```

工程固定为16384元素，不是任意shape/任意dtype通用算子。本实验未测量性能。

## API资料

融合双目AddRelu接口的功能、float支持和前n个元素形式可见昇腾官方API：
https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/apiref/ascendcopapi/atlasascendc_api_07_0043.html

本次实际可用性另以已安装CANN8.5.2头文件、CPU执行及真实NPU编译运行结果确认。
