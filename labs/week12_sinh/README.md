# 第12周 向量编程进阶算子 Sinh

姓名：许盛凯　学号：2024141460445

## 交付内容与结果

- `Lab4_Sinh/`：不使用 TBuf，复用输入 LocalTensor 保存 exp(-x)。
- `Lab4_SinhTBuf/`：使用512字节 TBuf<VECCALC> 保存 exp(x)，保留输入 LocalTensor。
- 两版均用基础 API `Exp → Muls(-1) → Exp → Sub → Muls(0.5)`，计算 sinh(x)=(exp(x)-exp(-x))/2；未使用高阶 Sinh API。
- 输入/输出 float32、ND、shape 8×2048、输入范围[1,10]；8个执行块，每块2048元素；每片128元素，每块16轮；每个张量65536字节。
- 两版分别完成CPU和真实NPU的random、boundary、ramp、ones、tens五组测试，20/20通过，总327680个输出元素。最大绝对误差0.0009765625，最大相对误差1.223703004833321e-7，每组失败元素0。
- 同一运行模式下，两版10组对应输出逐位一致；这是数值一致性结果，不代表性能一致或TBuf更快。本次未测量性能。

## 环境与复现

本次实测 Huawei Cloud ModelArts、EulerOS 2.0 aarch64、Ascend910B4、CANN8.5.2、GCC10.3.1、CMake3.22.0、Python3.12.0、NumPy1.26.4。物理NPU编号1，ACL使用容器逻辑设备0。
云端工程位于第12周实验的持久工作目录。

在有对应Ascend设备和CANN Toolkit的环境中，从本目录运行：

```bash
for project in Lab4_Sinh Lab4_SinhTBuf; do
  (cd "$project" && bash run.sh -r cpu -v Ascend910B4)
  (cd "$project" && bash run.sh -r npu -v Ascend910B4)
done
bash run_all.sh
bash show_npu.sh
```

`run.sh`默认random，可指定`--case boundary/ramp/ones/tens`。CPU通过tikicpulib调用同一核函数，NPU通过ACL与KernelLaunch执行真实设备计算。脚本保留`-r sim`构建入口，但本周未运行sim，不计入通过项。`run_all.sh`在已构建CPU/NPU程序上运行全部用例并归档，最后调用`summarize.py`校验两版数据与输出一致性。

## 原始源码与必要修改

`original_source/`原样保留所给RAR的“3.6.1双曲正弦函数算子核函数的实现”。原包没有独立Lab4_SinhTBuf，需要补写。教材原例是half、范围[-10,10]，课件main残留sizeof(uint16_t)和无关y参数；本实验按课件正文统一float32、[1,10]、sizeof(float)和一输入x/输出z接口。CMake和Host适配CANN8.5.2。

初版采用原示例Exp与Reciprocal组合，随机输入在CPU/NPU均有2015个元素超出预先固定阈值，最大相对误差3.43673783e-4。保留失败日志02/03/04；最终改为直接计算两个指数，阈值保持不变，05/06记录修正后CPU/NPU结果。无TBuf版在局部输入上就地计算，不改变全局输入。

## 验证与证据

`verification_all.json`：20组汇总及10组两版比较。每个工程`results/{cpu,npu}_{case}/`包含实际输入、golden、原始设备/CPU输出、元数据、生成/执行/校验日志。

验证参考为`float32(sinh(float64(x)))`，与设备基础API计算路径独立。所有元素必须满足：

```text
abs(actual - reference) <= 1e-6 + 1e-5 * abs(reference)
```

验证器检查每文件65536字节、16384个float32元素、全部有限、输入范围、输出正值、golden正确性和各核失败数。任一元素失败即整体失败，无失败比例豁免。输出先初始化为NaN，避免遗漏写回被误判通过。

在无NPU的电脑上重新核验归档，例如：

```bash
cd Lab4_Sinh
python3 scripts/verify_result.py \
  results/npu_random/output/output_z.bin \
  results/npu_random/output/golden.bin \
  --input-x results/npu_random/input/input_x.bin \
  --json /tmp/sinh_recheck.json
```

`evidence/01_environment.log`记录本次环境；02至04为初版失败；05、06为修正后构建运行；07为20组回归；08为网页控制台实时SSH运行的完整原始输出；09为对应真实浏览器屏显截图。

截图来源：本地网页控制台通过SSH实际执行服务器上的`show_npu.sh`，将未改写的标准输出显示于浏览器后截图。它不是JupyterLab终端截图，也不是由日志绘制的模拟终端图片。原生终端自动化受限，采用此方式完成运行屏显采集。`show_npu.sh`可直接在服务器重现同一输出。

工程固定为本题shape/dtype/范围，不宣称支持任意尺寸、负数或极值溢出场景。私钥与连接认证配置不属于提交内容。
