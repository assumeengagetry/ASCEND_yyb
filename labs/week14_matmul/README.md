# 第14周 矩阵运算性能探究

许盛凯 2024141460445。

## 实验规格

A:4096×4096、B:4096×5120，fp16、ND、随机值[1,10]，使用固定种子（见数据生成脚本）；C:4096×5120、float32，纯A@B无bias。输出约116107至131200，超出fp16有限范围，因此使用float32输出与累加。原始3.4教材示例保存在original_source，包含bias；本题已去掉bias。

4个对比配置均由同一KernelLaunch可执行程序和同一Matmul高阶API核函数实现，仅Host tiling参数不同：

| 配置 | blockDim | usedCoreNum | baseM,baseN,baseK |
| --- | --- | --- | --- |
| cores5_base128 | 5 | 10 | 128,128,128 |
| cores20_base64 | 20 | 40 | 64,64,64 |
| cores20_base128 | 20 | 40 | 128,128,128 |
| cores20_base128x256 | 20 | 40 | 128,256,64 |

实测Ascend910B4为20个Cube、40个Vector核；该Matmul从Vector侧发起，1个混合AI Core对应1 AIC与2 AIV，SetDim=2×blockDim。msprof的Block Dim与Mix Block Dim也分别记录这两个值。5核配置singleCore=2048×1024×4096，20核配置512×1024×4096，均无尾块。

## 复现

环境：ModelArts EulerOS2.0 aarch64，CANN8.5.2，Ascend910B4，GCC10.3.1，CMake3.22.0，Python3.12.0，NumPy1.26.4。

```bash
bash build.sh
python3 scripts/gen_data.py --output-dir data
bash run_all.sh
python3 summarize.py
bash show_results.sh
```

云端工程位于第14周实验的持久工作目录。`run_one.sh`先切回工程目录，保证msprof更改工作目录后仍读取同一输入。每个配置的20次执行为5次预热+15次采样，各次均完成流同步；msprof采集所有20条核任务，统计脚本按任务起始时间排序后丢弃前5条。真实设备运行已与第13周任务错开，采集窗口内无其它实验占用NPU。

## 正确性验证

共享`data/`只有一份input_a.bin、input_b.bin、golden.bin与metadata.json，记录SHA256。golden通过CPU NumPy float32矩阵乘获得；另有32个float64点积抽查，详见metadata。每配置保存实际设备output.bin，全20,971,520元素检查尺寸、finite、正值和预先固定的逐元素阈值：`abs(actual-golden)<=0.02+1e-4*abs(golden)`，任一个元素不满足即失败。验证器逐块读取并统计最大绝对/相对误差及RMSE，完整结果见results各配置verification.json；没有用golden替代设备输出。

## 性能证据与统计口径

原始msprof数据在`profiles/`，每配置对应`PROF_*/mindstudio_profiler_output/op_summary_*.csv`。`performance_summary.json`列出实际被使用的CSV相对路径、SHA256、15个原始测量值、中位数/最小/最大/均值/标准差；CSV简表见`performance_summary.csv`。

直接保留并分别报告厂商原始列名：`Task Duration(us)`、`aicore_time(us)`、`aic_mac_time(us)`、`aic_mte2_time(us)`。这些是不同计量口径，流水可重叠，不能将MAC/MTE2相加视为总时间；CPU生成golden的耗时、Host构建和进程启动时间不算算子性能。统计脚本只取Op Name=matmul_custom，断言20条且核数匹配。

## 本次实测结果

以下均为排除5次预热后15次采样的中位数，单位微秒。

| 配置 | Task Duration | AI Core | MAC | MTE2 |
| --- | ---: | ---: | ---: | ---: |
| 5核 128×128×128 | 4239.120 | 4030.008 | 2834.469 | 3955.756 |
| 20核 64×64×64 | 6096.400 | 5635.587 | 1062.392 | 5577.001 |
| 20核 128×128×128 | 1565.080 | 1507.625 | 712.628 | 1482.120 |
| 20核 128×256×64 | 1203.440 | 1186.418 | 711.773 | 1170.292 |

20核相对5核的AI Core时间加速为2.6731倍，Task Duration加速为2.7086倍，均低于理想4倍。三种20核基本块的实测排序与P的预测一致，其中128×256×64最快，但单次实验不能证明普遍最优。四配置分别对20,971,520输出元素验证，合计83,886,080个元素，失败数均0；最大绝对误差0.2109375、最大相对误差1.69764726e-6。

## CANN8.5.2适配说明

1. Matmul混合核使用系统workspace。初版手动设置workspace仍导致等待；最终在CMake定义`HAVE_WORKSPACE HAVE_TILING`，由KernelLaunch生成的wrapper设置系统指针并执行`matmul::clearWorkspace`，核内只读取`GetSysWorkSpacePtr()`。不得再次用偏移后的用户workspace覆盖系统指针。02/05/07/08号日志保留构建和排障过程。
2. `SetFixSplit`第三参数在当前API不保证固定baseK。64×64会自动得到baseK256，已先检测并中止，未把它误报为64³。最终先用`SetFixSplit(baseM,baseN,-1)`取得合法tiling，再将该配置返回的baseK从256降为64，并将stepKa、stepKb、depthA1、depthB1同比乘4，保持生成的L1分块跨度和缓存字节预算。这样实际L0基本块是64³；其正确性重新对全矩阵验证通过。其它两个块自动得到所需K，未覆盖。
3. 每配置`tiling.json`保存自动与实际K、相关缓存参数、完整TCubeTiling结构的int32字数组；`tiling.bin`保存送入设备的原始结构。任何请求与最终基本块不符均退出。修改其它形状时必须重新验证分片整除与内存预算，不能直接推广本固定配置。

## 结论边界

性能数值以最终summary为准。理想情况下20核相对5核加速4倍，正确的时间比为T5/T20=20/5；课件将执行时间比与核心数比的方向混写。实际受调度、搬运、分块、数据复用等影响，原因只能作为解释假设，不能仅凭总时间确定Cache或带宽瓶颈。理论P=baseM*baseK*baseN/(4096*(baseM*baseK+baseK*baseN))，以元素数计分母；3块P依次1/128、1/64、1/48。输入不变、输出全部验证通过是比较性能的前提。

## 官方适配依据

- KernelLaunch的HAVE_WORKSPACE/HAVE_TILING及Matmul系统workspace说明：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/opdevg/Ascendcopdevg/atlas_ascendc_10_0079.html
- SetFixSplit的baseK限制：https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0681.html
- 官方Matmul分块性能实践：https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0041.html
