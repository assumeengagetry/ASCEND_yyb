# 智能边缘计算与 Ascend C 实验

## 目录导航

| 目录 | 内容 |
| --- | --- |
| [materials/](materials/) | 原始课件、教材源码压缩包和实验报告模板 |
| [labs/](labs/) | 按第8至11周分类的实验工程、操作指南和运行记录 |
| [output/](output/) | 按周分类的 Word 报告和提交压缩包 |
| [scripts/](scripts/) | 报告生成与打包脚本 |
| [tmp/](tmp/) | 报告渲染、解压查看、字体和构建过程文件 |
| [archive/](archive/) | 重复副本及本次整理的移动记录 |

## 分周查找

| 周次 | 实验内容 | 工程与说明 | 报告和提交文件 |
| --- | --- | --- | --- |
| 第8周 | Atlas 200 DK + DHT11 云边协同 | [实验目录](labs/week8_atlas_dht11/README.md) | [output/week8/](output/week8/) |
| 第9周 | Ascend C 环境搭建与 Sigmoid 自定义算子 | [实验目录](labs/week9_ascendc/README.md) | [output/week9/](output/week9/) |
| 第10周 | HelloWorld 算子 | [实验说明](labs/week10_helloworld/README.md) | [output/week10/](output/week10/) |
| 第11周 | 向量编程基础算子 AddRelu | [实验说明](labs/week11_addrelu/README.md) | [output/week11/](output/week11/) |

课程资料分为 `materials/courseware/`（PDF、PPT）、`materials/source/`（原始 RAR）和 `materials/templates/`（报告模板）。

## 脚本

从项目根目录运行，使用已安装对应依赖的 Python 环境：

```bash
python3 scripts/build_week8_report.py
python3 scripts/build_week9_report.py
python3 scripts/build_week10_report.py
python3 scripts/package_week10.py
python3 scripts/build_week11_report.py
python3 scripts/package_week11.py
```

报告脚本需要 `python-docx`；第9至11周还需要 `Pillow` 及脚本指定的本机字体。生成命令会更新相应周次的报告；第10周使用第9周已有报告、第11周使用第10周报告作为格式参考，先确认参考文件可用。

工程在有对应 Ascend NPU 与 CANN Toolkit 的环境中运行，具体命令见各实验说明。`tmp/` 中保留原有过程文件。

2026年11月5日整理时保留了全部原有文件，报告和压缩包未重新生成，脚本和说明中的相关路径已同步更新。根目录原有的两份相同副本归入 [archive/duplicates/](archive/duplicates/)，移动清单见 [整理记录](archive/organization_2026-11-05.json)。
