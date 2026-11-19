# 智能边缘计算与 Ascend C 实验

## 目录导航

| 目录 | 内容 |
| --- | --- |
| [materials/](materials/) | 原始课件、教材源码压缩包和实验报告模板 |
| [labs/](labs/) | 按第8至14周分类的实验工程、操作指南和运行记录 |
| [output/](output/) | 第8至14周独立交付目录；第12至14周为 PDF 报告和各周提交 ZIP |
| [scripts/](scripts/) | 报告生成与打包脚本 |
| [tmp/](tmp/) | 报告渲染、解压查看、字体和构建过程文件 |
| [archive/](archive/) | 编辑源文件、历史版本、合并包、解压副本和移动记录 |

## 分周查找

| 周次 | 实验内容 | 工程与说明 | 报告和提交文件 |
| --- | --- | --- | --- |
| 第8周 | Atlas 200 DK + DHT11 云边协同 | [实验目录](labs/week8_atlas_dht11/README.md) | [output/week8/](output/week8/)（含重写版） |
| 第9周 | Ascend C 环境搭建与 Sigmoid 自定义算子 | [实验目录](labs/week9_ascendc/README.md) | [output/week9/](output/week9/) |
| 第10周 | HelloWorld 算子 | [实验说明](labs/week10_helloworld/README.md) | [output/week10/](output/week10/) |
| 第11周 | 向量编程基础算子 AddRelu | [实验说明](labs/week11_addrelu/README.md) | [output/week11/](output/week11/) |
| 第12周 | 向量编程进阶算子 Sinh 与 SinhTBuf | [实验说明](labs/week12_sinh/README.md) | [output/week12/](output/week12/) |
| 第13周 | 算子工程化开发及PyTorch整网调用 | [实验说明](labs/week13_leakyrelu/README.md) | [output/week13/](output/week13/) |
| 第14周 | 矩阵运算性能探究 | [实验说明](labs/week14_matmul/README.md) | [output/week14/](output/week14/) |

课程资料分为 `materials/courseware/`（PDF、PPT）、`materials/source/`（原始 RAR）和 `materials/templates/`（报告模板）。

打开对应周次的 `output/weekN/README.md` 可直接找到报告、独立提交 ZIP 和实验工程。第12至14周每周各有一份 PDF 和一个完整 ZIP；提交一周时使用该周 ZIP 即可。

合并包已归入 `archive/combined/`，Word 编辑源文件位于 `archive/report_sources/`，历史报告位于 `archive/report_history/`。这些文件不混入各周的交付目录。

## 脚本

从项目根目录运行，使用已安装对应依赖的 Python 环境：

```bash
python3 scripts/build_week8_report.py
python3 scripts/build_week9_report.py
python3 scripts/build_week10_report.py
python3 scripts/package_week10.py
python3 scripts/build_week11_report.py
python3 scripts/package_week11.py
python3 scripts/build_week12_report.py
python3 scripts/package_week12.py
python3 scripts/build_week13_report.py
python3 scripts/package_week13.py
python3 scripts/build_week14_report.py
python3 scripts/package_week14.py
```

报告脚本需要 `python-docx`；第9至14周还需要 `Pillow` 及脚本指定的本机字体。第10周使用第9周已有报告、第11周使用第10周报告、第12周使用第11周报告作为格式参考。第13周引用 `archive/report_sources/week12/` 中的源文件，第14周使用 `archive/report_sources/week14/template_reference.docx`。

第12至14周的生成脚本把编辑源文件写入 `archive/report_sources/weekN/`，PDF 成品放在 `output/weekN/`；相应打包脚本读取该周现有 PDF。可选合并脚本 `package_week12_14.py` 只写入 `archive/combined/week12-14/`。

工程在有对应 Ascend NPU 与 CANN Toolkit 的环境中运行，具体命令见各实验说明。`tmp/` 中保留原有过程文件。

各周报告、提交包及实验数据保持原内容；移动时逐文件核对 SHA-256。查看 [按周整理记录](archive/organization_by_week.json) 可追溯原路径与新路径。

## 版本管理

Git 保存实验源码、PDF 报告、报告编辑源文件、验证 JSON、运行日志、性能 CSV/DB 和截图。临时目录、编译产物、大型输入输出数组与权重、提交 ZIP、历史及重复副本只保留在本地，具体规则见 [.gitignore](.gitignore)。

从仓库重新下载后，按各周实验说明生成数据并运行实验，再使用打包脚本生成完整提交 ZIP。本机的 `output/weekN/` 中仍保留已完成的完整提交包。
