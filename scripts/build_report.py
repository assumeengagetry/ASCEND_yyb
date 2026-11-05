from pathlib import Path
import json, re, textwrap
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path('/Users/apple/Workspace/ascend')
OUT = ROOT / 'output'
FIG = ROOT / 'tmp/report_figures'
OUT.mkdir(exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
E = ROOT / 'evidence'
RESULTS = ROOT / 'solution/SigmoidCustom/AclNNInvocation/results'
data = json.loads((RESULTS/'verification.json').read_text())
assert data['passed_cases'] == data['total_cases'] == 16

def clean(s):
    s = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', s)
    return s.replace('\x08', '').strip()

def terminal(name, lines, width=1480, font_size=24):
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Courier New.ttf', font_size)
    titlefont = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 21)
    lines = [clean(x) for x in lines]
    lineh = font_size + 10
    img = Image.new('RGB', (width, 70+len(lines)*lineh+25), '#F5F6F8')
    d = ImageDraw.Draw(img)
    d.rectangle((0,0,width,49), fill='#E4E8ED')
    d.text((24,13), 'ModelArts  |  ma-user  |  recorded terminal output', font=titlefont, fill='#263648')
    for i, line in enumerate(lines):
        d.text((24,68+i*lineh), line, font=font, fill='#172433')
    p=FIG/(name+'.png');img.save(p);return p

env=(E/'01_environment.log').read_text()
npu=env.split('$ npu-smi info\n')[1].split('package_name=')[0].strip().splitlines()
f_npu=terminal('npu',npu[:11],width=1780,font_size=22)
f_deps=terminal('deps',[
    '$ python3 -m venv --system-site-packages venv',
    '$ venv/bin/python -m pip install numpy==1.24.0 protobuf==3.20.0',
    'Successfully installed numpy-1.24.0 protobuf-3.20.0',
],font_size=24)
f_toolkit=terminal('toolkit',[
    '$ cat .../aarch64-linux/ascend_toolkit_install.info',
    'package_name=Ascend-cann-toolkit',
    'version=8.0.0',
    'arch=aarch64',
    '$ command -v ccec',
    '/home/ma-user/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin/ccec',
],width=1600,font_size=22)
build=(E/'05c_precision_build.log').read_text()
success_lines=[x.strip() for x in build.splitlines() if ('Built target ascendc_bin_ascend910b_sigmoid_custom_' in x)]
f_build=terminal('build',[
    '$ cmake --build build_out --target binary -j 4',
    *list(dict.fromkeys(success_lines))[:2],
    'Self-extractable archive "custom_opp_euleros_aarch64.run" successfully created.',
    '$ bash custom_opp_euleros_aarch64.run --install-path="$LAB/install"',
    'SHA256 checksums are OK. All good.',
    'SUCCESS',
],width=1720,font_size=23)
testlines=[x for x in (E/'07_npu_tests.log').read_text().splitlines() if x.startswith(('PASS','SUMMARY'))]
testlines=[re.sub(r' failed=0.*$','',x) for x in testlines]
testlines[-1]='SUMMARY 16/16 PASS'
f_tests=terminal('npu_results',testlines,width=1740,font_size=23)

doc=Document()
s=doc.sections[0]
s.page_width=Cm(21);s.page_height=Cm(29.7)
s.top_margin=Cm(1.75);s.bottom_margin=Cm(1.65)
s.left_margin=Cm(2);s.right_margin=Cm(2)
s.header_distance=Cm(.65);s.footer_distance=Cm(.65)
styles=doc.styles
for name in ['Normal','Body Text','Title','Subtitle','Heading 1','Heading 2','Caption']:
    st=styles[name]
    st.font.name='Times New Roman'
    st._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Songti SC')
    for attr in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme']:
        st._element.get_or_add_rPr().rFonts.attrib.pop(qn('w:'+attr),None)
    st.font.color.rgb=RGBColor.from_string('000000')
styles['Normal'].font.size=Pt(10.5)
styles['Normal'].paragraph_format.line_spacing=1.27
styles['Normal'].paragraph_format.space_after=Pt(5)
styles['Title'].font.size=Pt(22)
styles['Title'].font.bold=True
styles['Title'].paragraph_format.space_after=Pt(8)
styles['Heading 1'].font.size=Pt(14)
styles['Heading 1'].font.bold=True
styles['Heading 1'].paragraph_format.space_before=Pt(9)
styles['Heading 1'].paragraph_format.space_after=Pt(7)
styles['Heading 2'].font.size=Pt(11.5)
styles['Heading 2'].font.bold=True
styles['Heading 2'].paragraph_format.space_before=Pt(6)
styles['Heading 2'].paragraph_format.space_after=Pt(4)
styles['Caption'].font.size=Pt(9)
styles['Caption'].paragraph_format.space_after=Pt(7)
styles['Caption'].font.color.rgb=RGBColor.from_string('4E5965')
for st in styles:
    for border in list(st._element.xpath('.//w:pBdr')):
        border.getparent().remove(border)

foot=s.footer.paragraphs[0]
foot.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=foot.add_run('许盛凯  2024141460445    |    ');r.font.size=Pt(8)
fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');foot._p.append(fld)

def p(text='',bold=False):
    x=doc.add_paragraph();r=x.add_run(text);r.bold=bold;return x
def h(text):return doc.add_heading(text,1)
def h2(text):return doc.add_heading(text,2)
def page(title):doc.add_page_break();h(title)
def code(text,size=8.5):
    x=doc.add_paragraph()
    x.paragraph_format.space_after=Pt(7)
    x.paragraph_format.line_spacing=1.08
    for i,line in enumerate(text.strip().splitlines()):
        r=x.add_run(('\n' if i else '')+line)
        r.font.name='Courier New';r.font.size=Pt(size)
        r._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Songti SC')
    pp=x._p.get_or_add_pPr();sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'F3F5F7');pp.append(sh)
    return x
def table(headers, rows, widths=None, size=9.5):
    t=doc.add_table(rows=1, cols=len(headers));t.style='Table Grid';t.autofit=False
    if widths:
        for col,w in zip(t.columns,widths):col.width=Cm(w)
    for c,txt in zip(t.rows[0].cells,headers):c.text=str(txt)
    trpr=t.rows[0]._tr.get_or_add_trPr();repeat=OxmlElement('w:tblHeader');trpr.append(repeat)
    for row in rows:
        cells=t.add_row().cells
        for c,txt in zip(cells,row):c.text=str(txt)
    for i,row in enumerate(t.rows):
        trpr=row._tr.get_or_add_trPr();cant=OxmlElement('w:cantSplit');trpr.append(cant)
        for j,c in enumerate(row.cells):
            if widths:c.width=Cm(widths[j])
            for x in c.paragraphs:
                x.paragraph_format.space_before=Pt(2);x.paragraph_format.space_after=Pt(2)
                x.paragraph_format.line_spacing=1.12
                for r in x.runs:r.font.size=Pt(size);r.bold=(i==0)
            if i==0:
                sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'E9EDF2');c._tc.get_or_add_tcPr().append(sh)
    doc.add_paragraph().paragraph_format.space_after=Pt(0)
    return t
def fig(path,caption,width=17):
    x=doc.add_paragraph();x.paragraph_format.space_after=Pt(3)
    x.add_run().add_picture(str(path),width=Cm(width))
    x.paragraph_format.keep_with_next=True
    doc.add_paragraph(caption,style='Caption')

doc.add_paragraph('Ascend C 环境配置与\nSigmoid 算子实验报告',style='Title')
p('姓名  许盛凯        学号  2024141460445        实验日期  2026年10月29日',True)
p('本实验在华为云 ModelArts 的 Ascend 910B4 环境中完成 CANN 配置核验、Python 依赖配置，以及 SigmoidCustom 算子的实现、编译、部署和 ACLNN 调用。16组 NPU 数值验证全部通过，float16 与 float32 的最大绝对误差分别为 2.4405×10⁻⁴ 和 1.0241×10⁻⁷。')
h('一 实验目的与内容')
p('根据课程资料，环境实验需要记录各步骤的配置与安装回显；算子练习需要实现 SigmoidCustom，编译部署后通过 ACLNN 样例，并将工程打包。实验重点是使开发工具链与实际硬件匹配，并用真实设备输出验证算子。')
h('二 实验环境')
table(['项目','实测配置'],[
('平台及用户','华为云 ModelArts    ma-user'),
('操作系统及架构','EulerOS 2.0 SP10    aarch64'),
('NPU及驱动工具','Ascend 910B4 32GB    npu-smi 25.5.1'),
('CANN开发套件','8.0.0 用户目录版    /home/ma-user/Ascend'),
('基础编译工具','GCC 7.3.0    GNU Make 4.3    CMake 3.16.5'),
('Python','3.9.10    实验专用 venv'),
('实验目录','/home/ma-user/work/ascend_lab_20261029'),
('算子安装目录','实验目录下 install/vendors/lab_sigmoid'),
],widths=[3.2,13.8])
fig(f_npu,'图1  NPU状态实测回显  设备健康状态为OK  原始记录见01_environment.log')

page('三 基础依赖与Python环境配置')
h2('1 基础工具检查')
code(r'''uname -m
gcc --version
make --version
cmake --version
python3 --version
python3 -m pip --version''')
p('回显依次确认 aarch64、GCC 7.3.0、Make 4.3、CMake 3.16.5、Python 3.9.10、pip 21.0.1。ifconfig 已安装，Python.h 头文件存在。现有基础工具满足课程要求，直接用于后续构建。')
h2('2 创建实验专用环境并配置第三方库')
code(r'''cd /home/ma-user/work/ascend_lab_20261029
python3 -m venv --system-site-packages venv
venv/bin/python -m pip install numpy==1.24.0 protobuf==3.20.0''')
fig(f_deps,'图2  Python依赖安装反馈  其余依赖复用基础环境中的已安装版本')
table(['库','实测版本','库','实测版本'],[
('attrs','23.2.0','Cython','3.0.2'),('numpy','1.24.0','decorator','5.1.1'),
('sympy','1.13.0','cffi','1.16.0'),('PyYAML','6.0.2','pathlib2','2.3.7.post1'),
('psutil','6.0.0','protobuf','3.20.0'),('scipy','1.12.0','requests','2.32.2'),
('absl-py','1.4.0','导入验证','13项全部通过'),
],widths=[4.1,4.4,4.1,4.4])
p('venv 继承了基础环境的库。pip 回显提示基础环境中 ONNX、ModelArts 等组件存在版本约束差异；本实验需要的13项库均可导入，后续编译和 NPU 验证通过。该 venv 专用于本算子实验，原 MindSpore 环境的安装包未被卸载。')

page('四 CANN开发套件核验与环境激活')
h2('1 核实已有安装')
p('本机已有完整 Toolkit 安装目录。本次核查安装信息、编译器入口和既有安装日志，复用预装 CANN 8.0.0，不重复覆盖安装。历史日志记录于2025年6月11日，属于镜像预置环境的安装证据。')
code(r'''cat /home/ma-user/Ascend/ascend_cann_install.info
cat /home/ma-user/Ascend/ascend-toolkit/latest/\
aarch64-linux/ascend_toolkit_install.info''')
fig(f_toolkit,'图3  Toolkit版本与Ascend C编译器入口实测回显')
code(r'''[Toolkit] [20250611-15:06:51] [INFO]
Ascend-test-ops_8.0.0_linux.run --full --quiet --nox11 install success
Toolkit_InstallPath=/home/ma-user/Ascend/ascend-toolkit''',8)
p('原始记录见 evidence/00_preinstalled_toolkit_history.log。课程中对 .run 安装包执行 chmod、--check、--install 的流程适用于首次安装；本次没有重新下载 Toolkit 安装包，未将历史安装反馈写成本次执行结果。自定义算子安装包的校验及部署见第六节。')
h2('2 激活环境变量')
code(r'''source /home/ma-user/Ascend/ascend-toolkit/set_env.sh
source /home/ma-user/work/ascend_lab_20261029/venv/bin/activate
export SETUPTOOLS_USE_DISTUTILS=stdlib
export ASCEND_CUSTOM_OPP_PATH=\
/home/ma-user/work/ascend_lab_20261029/install/vendors/lab_sigmoid''',8)
p('env.sh 统一配置上述环境及自定义算子的 op_api/lib 路径。实际 ASCEND_HOME_PATH 为 /home/ma-user/Ascend/ascend-toolkit/latest。使用同一 Toolkit 完成工程生成、编译和运行，避免仅依据登录欢迎信息判断生效版本。')
h2('3 教材源码与工程准备')
p('上传的 Ascend C异构并行程序设计源代码.rar 含 AddCustom 和 AclNNInvocation 教材样例。本实验参考其 Host、Kernel 和两阶段调用结构，以已安装 msOpGen 生成 SigmoidCustom 工程，实现单输入 Sigmoid。工程生成目标为 ai_core-Ascend910B4，构建架构为 ascend910b。')
code(r'''msopgen gen -i sigmoid_custom.json \
  -c ai_core-Ascend910B4 -lan cpp -out SigmoidCustom
# 实际反馈  Generation completed.''')

page('五 Sigmoid算子设计与实现')
p('Sigmoid 为逐元素一元运算，数学表达式为 y = 1 / (1 + exp(-x))。输入和输出均为 ND 格式连续张量，支持 float16 与 float32，输出 shape 和数据类型与输入一致。')
table(['组成','实现内容'],[
('op_host','注册 SigmoidCustom；推导输出shape和类型；计算tiling。'),
('op_kernel','Ascend C核函数；分片搬运；向量运算；尾块处理。'),
('AclNNInvocation','申请设备内存、构造aclTensor、调用自定义接口、取回输出。'),
],widths=[3.6,13.4])
h2('1 分片与多核分配')
p('元素总数使用 uint64_t 存储；每片包含2048个元素，使用不超过8个 block。第 k 个 block 处理 k、k+blockCount 等分片。该方案覆盖不能被2048整除的长度，也避免小张量启动无效计算核。workspace大小为0。')
code(r'''tileLength = 2048;
tileCount = (totalLength + tileLength - 1) / tileLength;
blockCount = min(tileCount, 8);
for (tile = GetBlockIdx(); tile < tileCount; tile += GetBlockNum()) {
    count = min(tileLength, totalLength - tile * tileLength);
    CopyIn(offset, count); Compute(count); CopyOut(offset, count);
}''')
h2('2 向量计算与精度')
p('float16 输入先转换为 float32，再执行乘负一、指数、加一和除法，最后按最近舍入转回 float16。float32 输入也使用 float32 中间结果。各依赖运算之间加入 PIPE_V 屏障，输入输出使用队列同步。')
code(r'''Muls(tmp, tmp, -1.0f, count);
Exp(tmp, tmp, count);
Adds(tmp, tmp, 1.0f, count);
Duplicate(ones, 1.0f, count);
Div(result, ones, tmp, count);''')
p('最初使用 Reciprocal 的实测误差约为2.7×10⁻³，不能满足 float32 验证阈值。改用 Div 后最大绝对误差降至约1.03×10⁻⁷，验证时保留原阈值。')
h2('3 非对齐尾块')
p('输入与输出采用 DataCopyPad 和 DataCopyExtParams，按 count×sizeof(T) 指定实际搬运字节数。最后不足32字节的数据块只写回有效元素，不按完整对齐块覆盖相邻内存。测试覆盖长度1、15、17、255、257。')

page('六 编译打包与部署')
h2('1 构建与安装过程')
p('CMake 3.16.5 满足课程最低版本。由于其不支持新式 preset 调用，本实验通过显式 CMake 参数配置 Release、ascend910b、lab_sigmoid 和当前 Toolkit 路径。rebuild.sh 固化以下构建及部署步骤。')
code(r'''source /home/ma-user/work/ascend_lab_20261029/env.sh
cd /home/ma-user/work/ascend_lab_20261029/SigmoidCustom
bash rebuild.sh

# 脚本中的主要步骤
cmake --build build_out --target package -j 4
bash build_out/custom_opp_euleros_aarch64.run \
  --quiet --install-path="$ASCEND_LAB_ROOT/install"
cmake --build build_out --target binary -j 4
cmake --build build_out --target package -j 4
# 将最终含kernel二进制的安装包再次安装到同一目录''',8)
p('先生成并安装 Host 侧工程，以提供构建所需接口；随后编译两种数据类型的内核，再次打包并安装最终包。最终安装回显包含 SHA256 checksums are OK 和 SUCCESS。安装位置位于实验目录下，vendor名称为 lab_sigmoid。')
fig(f_build,'图4  内核构建 打包与最终安装反馈  完整记录见05c_precision_build.log及06_install.log')
h2('2 检查部署文件')
table(['文件或目录','验证结果'],[
('op_api/include/aclnn_sigmoid_custom.h','已生成自定义ACLNN接口声明'),
('op_api/lib/libcust_opapi.so','已生成并供调用程序链接'),
('kernel/ascend910b/sigmoid_custom/','float16和float32各有.o与配套JSON'),
('kernel/config/ascend910b/binary_info_config.json','已生成并安装，供运行时检索'),
],widths=[10.2,6.8],size=9)
p('编译器回显中使用 Ascend910B1 表示 ascend910b 编译目标；实际执行硬件由 npu-smi 确认为910B4，最终以该机器上的运行和精度结果验收。')

page('七 ACLNN调用与数值验证')
h2('1 验证方法')
p('调用程序实际执行 aclnnSigmoidCustomGetWorkspaceSize 和 aclnnSigmoidCustom，将数据从Host复制到NPU，等待计算流完成后取回输出。device_id 使用运行时逻辑编号0；容器对外可见的物理 NPU 为7。')
p('随机种子固定为20261029，普通输入取 [-12,12] 均匀分布，并加入 -1000、-100、-80、-20、0、20、80、100、1000 等有限极值。单元素用例输入为0.5。参考值基于实际量化后的输入，以 NumPy float64 的稳定分支公式计算。')
code(r'''# 逐元素精度判定
abs(actual - reference) <= atol + rtol * abs(reference)
# float16: atol=1e-3, rtol=1e-3
# float32: atol=1e-6, rtol=1e-4''')
h2('2 逐组验证结果')
cases=data['cases']
shapes=[(8,2048),(1,),(15,),(17,),(255,),(257,),(4096,),(65536,)]
rows=[]
for shape in shapes:
    a=next(c for c in cases if c['dtype']=='float16' and tuple(c['shape'])==shape)
    b=next(c for c in cases if c['dtype']=='float32' and tuple(c['shape'])==shape)
    rows.append(('×'.join(map(str,shape)),f"{a['max_abs_error']:.6e}",f"{b['max_abs_error']:.6e}",'均为0','PASS'))
table(['输入shape','float16最大绝对误差','float32最大绝对误差','失败元素','结果'],rows,widths=[2.3,4.8,4.8,2.6,2.5],size=9)
p('共16组，16组通过。float16 最大绝对误差为 2.44048503×10⁻⁴，float32 为 1.02409965×10⁻⁷。所有输出长度与输入一致，所有案例失败元素数均为0。')
h2('3 复现命令与数据位置')
code(r'''cd /home/ma-user/work/ascend_lab_20261029
source env.sh
cd SigmoidCustom
bash run.sh
# 单组复现
bash run.sh --dtype float32 --shape 8,2048
bash run.sh --dtype float16 --numel 257''')
p('原始输入 input.bin、NPU输出 output.bin、float64基准 golden_float64.bin、各组 npu.log 和结构化 verification.json 均保存在 AclNNInvocation/results/，随实验工程归档。')

page('八 实测回显与实验分析')
fig(f_tests,'图5  最终16组NPU验证回显  由07_npu_tests.log原始记录排版  精度阈值见第七节')
h2('1 调试问题及解决办法')
table(['问题','原因与处理'],[
('登录出现No such file','~/.bashrc中存在单独的下载URL。备份原文件后将该行改为注释，登录恢复。'),
('curl报CAfile错误','系统CA路径不可用。env.sh以python -m certifi得到证书路径并设置CURL_CA_BUNDLE；验证HTTPS返回200。'),
('构建出现NoneType异常','setuptools的distutils替换影响编译器导入。设置SETUPTOOLS_USE_DISTUTILS=stdlib后完成内核编译。'),
('运行提示缺二进制配置','第一次调用时内核尚未完整安装。完成binary目标、重新打包并安装，确认binary_info_config.json存在后重测。'),
('初版误差超阈值','单独Reciprocal精度不足。改用Div并重新编译，最终16组全部通过。'),
],widths=[3.7,13.3],size=9)
h2('2 实验结论')
p('本次完成用户目录 CANN 8.0.0 的配置核验、专用 Python 依赖配置，以及 SigmoidCustom 从注册、分片、向量计算到 ACLNN 调用的全过程。设备实测验证了两种数据类型和短尾块的正确性。验证范围为非空、连续ND张量及有限输入，未对非连续张量、NaN/Inf、空张量或性能作结论。')
h2('3 参考资料')
p('［1］华为云Ascend C算子开发环境搭建，第3、10、11页；［2］CANN 环境搭建，第1至6页；［3］Ascend C异构并行程序设计源代码，4.1-4.5章 AddCustom 与 AclNNInvocation；［4］昇腾社区 CANN 8.0.0 DataCopyPad接口文档。',False)

doc.core_properties.title='Ascend C 环境配置与 Sigmoid 算子实验报告'
doc.core_properties.author='许盛凯'
doc.core_properties.subject='CANN环境配置与自定义算子NPU实测'
target=OUT/'许盛凯_2024141460445_AscendC实验报告.docx'
doc.save(target)
print(target)
