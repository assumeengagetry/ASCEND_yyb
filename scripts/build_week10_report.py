from pathlib import Path
import hashlib,json,zipfile,copy
from docx import Document
from docx.shared import Cm,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from PIL import Image,ImageDraw,ImageFont

R=Path(__file__).resolve().parents[1]
E=R/'labs/week10_helloworld/evidence'
T=R/'tmp/week10'
OUT=R/'output/week10'
REF=R/'output/week9/许盛凯_2024141460445_AscendC实验报告.docx'
OUT.mkdir(parents=True,exist_ok=True)
T.mkdir(parents=True,exist_ok=True)
assert hashlib.sha256(REF.read_bytes()).hexdigest()=='3d8234ee8065639d42bde24fa181c253890d0fee728330d7de269a1fe3319706'
assert json.loads((E/'verification.json').read_text())['passed']
name='2024141460445-许盛凯-第10讲（11月5日）-HelloWorld算子-实验报告'
D=Document(REF)
body=D._element.body
for el in list(body):
 if el.tag!=qn('w:sectPr'): body.remove(el)
# Remove replaced figure relationships; retain the source styles and footer.
for rid,rel in list(D.part.rels.items()):
 if rel.reltype.endswith('/image'): D.part.drop_rel(rid)

def p(s='',bold=False):
 x=D.add_paragraph();r=x.add_run(s);r.bold=bold;return x

def h(s):return D.add_heading(s,1)
def h2(s):return D.add_heading(s,2)
def page(s):D.add_page_break();h(s)
def code(s,size=8.5):
 x=D.add_paragraph();f=x.paragraph_format
 f.space_after=Pt(7);f.line_spacing=1.08
 for i,line in enumerate(s.strip().splitlines()):
  r=x.add_run(('\n' if i else '')+line);r.font.name='Courier New';r.font.size=Pt(size)
  r._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Songti SC')
 return x

def table(headers,rows,widths):
 t=D.add_table(rows=1,cols=len(headers));t.style='Table Grid';t.autofit=False
 t.alignment=WD_TABLE_ALIGNMENT.CENTER
 for c,w in zip(t.columns,widths):c.width=Cm(w)
 for c,s in zip(t.rows[0].cells,headers):c.text=s
 for row in rows:
  for c,s in zip(t.add_row().cells,row):c.text=str(s)
 bd=OxmlElement('w:tblBorders')
 for k in ['top','left','bottom','right','insideH','insideV']:
  b=OxmlElement('w:'+k);b.set(qn('w:val'),'single');b.set(qn('w:sz'),'4');b.set(qn('w:color'),'D9D9D9');bd.append(b)
 t._tbl.tblPr.append(bd)
 for i,row in enumerate(t.rows):
  trpr=row._tr.get_or_add_trPr();trpr.append(OxmlElement('w:cantSplit'))
  if i==0:trpr.append(OxmlElement('w:tblHeader'))
  for j,c in enumerate(row.cells):
   c.width=Cm(widths[j]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   for x in c.paragraphs:
    x.paragraph_format.space_before=Pt(3);x.paragraph_format.space_after=Pt(3);x.paragraph_format.line_spacing=1.12
    for r in x.runs:r.font.size=Pt(9.5);r.bold=i==0
   if i==0:
    sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'E9EDF2');c._tc.get_or_add_tcPr().append(sh)
 p().paragraph_format.space_after=Pt(0)
 return t

def fig(path,caption,width=17):
 x=D.add_paragraph();x.paragraph_format.keep_with_next=True;x.paragraph_format.space_after=Pt(3)
 x.add_run().add_picture(str(path),width=Cm(width))
 D.add_paragraph(caption,'Caption')

D.add_paragraph('HelloWorld 算子\n第10周实验报告','Title')
p('姓名  许盛凯        学号  2024141460445        日期  2026年11月5日',True)
p('本实验在华为云 ModelArts 的 Ascend 910B4 上，通过 KernelLaunch 方式执行 HelloWorld 算子。将启动核数设为8后，设备日志记录 AIV-0 至 AIV-7 共8个执行块，每个执行块输出一次 Hello World!!!，程序退出码为0，满足讲义要求。')
h('一 实验目的与要求')
p('掌握 Ascend C 核函数的定义及设备侧打印方法，理解 Host 侧 ACL 资源管理与 <<<>>> 内核启动语法，完成工程配置、编译和真实 NPU 调用。实验提交物包括完整算子工程、运行屏显以及结果分析。')
h('二 实验环境')
table(['项目','本次实测配置'],[
 ('平台','华为云 ModelArts    用户 ma-user'),
 ('系统及架构','Huawei Cloud EulerOS 2.0    aarch64'),
 ('计算设备','Ascend 910B4    32 GB HBM    健康状态 OK'),
 ('驱动查询工具','npu-smi 25.5.1'),
 ('CANN及路径','8.5.2    /usr/local/Ascend/ascend-toolkit/latest'),
 ('工具链','GCC 10.3.1    CMake 3.22.0    Python 3.12.0'),
 ('工程目录','/home/ma-user/work/ascend_week10_20261105/'),
], [3.3,13.7])
p('Toolkit 的 latest 实际指向 /usr/local/Ascend/cann-8.5.2。npu-smi 中可见物理设备编号5，程序通过 ACL 使用当前环境的逻辑设备编号0。环境核验记录见 evidence/01_environment.log。')
h('三 实验原理')
p('Host 侧初始化 ACL，选择设备并创建 stream；hello_world_do 将 blockDim 和 stream 传给 <<<>>> 启动表达式，把核函数提交到设备。核函数调用 AscendC::printf 输出问候语。Host 随后调用 aclrtSynchronizeStream 等待任务完成，再销毁流、复位设备并结束 ACL。')


page('四 工程准备与关键代码')
p('源码取自所给 RAR 的“2.3.4调用一个昇腾算子”目录，对应讲义中的 Lab2_helloworld。压缩包中的核函数及8核常量已完整提供；本次保留其逻辑，按实机修改 SoC 配置并完善运行检查。')
code('''Lab2_helloworld/
├── CMakeLists.txt     编译配置
├── hello_world.cpp    Ascend C核函数及启动包装
├── main.cpp           ACL初始化及核函数调用
└── run.sh             配置 编译 运行与输出核验''')
h2('1 设备与Toolkit配置')
code('''set(SOC_VERSION "Ascend910B4" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH
    "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "CANN path")
# 保留教材ascendc.cmake导入及ascendc_library构建方式''',8)
p('依据 npu-smi 的 Name=910B4 设置 Ascend910B4。CMake 从当前 Toolkit 导入 ascendc.cmake，将 hello_world.cpp 编译为 kernels 静态库，并链接到 main。')
h2('2 设备侧打印与KernelLaunch')
code('''#include "kernel_operator.h"
extern "C" __global__ __aicore__ void hello_world()
{
    AscendC::printf("Hello World!!!\\n");
}
void hello_world_do(uint32_t blockDim, void *stream)
{
    hello_world<<<blockDim, nullptr, stream>>>();
}''')
p('extern "C" 指定 C 链接方式；__global__ 与 __aicore__ 标识设备侧核函数。启动表达式的第一个参数为执行块数，第三个参数指定计算流。')
h2('3 Host侧调用与核数设置')
code('''aclInit(nullptr);
int32_t deviceId = 0;
aclrtSetDevice(deviceId);
aclrtStream stream = nullptr;
aclrtCreateStream(&stream);
constexpr uint32_t blockDim = 8;
hello_world_do(blockDim, stream);
aclrtSynchronizeStream(stream);
aclrtDestroyStream(stream);
aclrtResetDevice(deviceId);
aclFinalize();''')
p('以上展示主调用流程。提交工程对各 ACL 调用使用 ACL_CHECK 检查返回值，遇到错误时打印接口名及错误码并返回非零状态，避免将失败运行判为成功。')

page('五 编译运行与屏显结果')
h2('1 执行命令')
code('''cd /home/ma-user/work/ascend_week10_20261105/Lab2_helloworld
bash run.sh -v Ascend910B4''',8)
p('run.sh 自动激活 Toolkit 环境，配置 CMake，构建 main 并运行。首次编译反馈“[100%] Built target main”，随后输出8条问候语，脚本反馈“[PASS] Hello World lines: 8 / 8”。构建与运行完整日志保存在 evidence/02_build_and_run.log。')
shot=E/'06_actual_terminal.png'
if shot.exists():
 fig(shot,'图1  ModelArts终端真实运行屏显  本次NPU执行结果',17)
else:
 # This is explicitly a log rendering, never represented as an actual screenshot.
 lines=(E/'03_npu_run.log').read_text().splitlines()
 short=[]
 for line in lines:
  if line.startswith('opType='):line=line.split(', block_remain_len=')[0]
  short.append(line)
 font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Courier New.ttf',20)
 im=Image.new('RGB',(1500,34+29*len(short)),'white');dr=ImageDraw.Draw(im)
 for i,line in enumerate(short):dr.text((14,12+29*i),line,font=font,fill='black')
 image=T/'raw_output_render.png';im.save(image)
 fig(image,'图1  本次NPU原始日志排版图  调试元数据仅展示与核数有关的字段',17)
h2('2 结果核验')
table(['检查项','实际结果','判定'],[
 ('Host程序退出码','0','通过'),
 ('问候语条数','8条 Hello World!!!','通过'),
 ('执行块编号','AIV-0 至 AIV-7 无缺失','通过'),
 ('执行块配置','block dim=8  total_block_num=8','通过'),
 ('运行时版本','CANN Version: 8.5.2','一致'),
],[4.6,9.9,2.5])
p('完整构建调用与单独执行 main 两次均得到8条输出。核验脚本还检查每个 AIV 编号恰好出现一次，结构化结果见 evidence/verification.json，原始屏显见 03_npu_run.log 和 04_output_msg.txt。')

page('六 实验分析与总结')
h2('1 为什么输出8次')
p('blockDim=8 使运行时启动8个逻辑执行块，设备函数内没有分支或提前返回，每个块均执行一次 AscendC::printf。因此预期输出条数为8，实测 AIV-0 至 AIV-7 与8条问候语相互印证。该结果验证了本次启动配置；它不表示芯片只有8个物理计算核。')
h2('2 为什么需要流同步')
p('<<<>>> 向指定 stream 提交设备任务，Host 提交后可以继续执行。aclrtSynchronizeStream 等待流上任务结束，并提供检查设备执行结果的同步点。同步成功后再释放流和设备资源，保证本次执行流程完整。')
h2('3 如何理解调试输出')
p('每组日志均包含 opType=hello_world、AIV编号、block dim 和 CANN版本，随后是 Hello World!!!。调试元数据与问候语共同说明核函数已在设备侧执行。本次观察到编号按0至7展示，但并行核的屏显顺序不能用于推断严格的串行执行顺序。')
p('日志的 TimeStamp: 202507 为调试头字段；本次实验日期以环境日志记录的2026年11月5日为准。')
h2('4 适配与运行检查')
p('本次将原工程的 Ascend310P3 改为 Ascend910B4，使用实机 Toolkit 路径，显式调用 AscendC::printf，完善 ACL 返回值及脚本检查。原始教材文件保存在 original_source/。')
p('CMake 子工程提示 DYNAMIC_MODE、BUILD_MODE 等传入变量未使用；构建及两次NPU执行均成功，未修改系统 Toolkit。')
h2('5 实验结论')
p('本实验完成了 HelloWorld 工程配置、8块并行启动和真实 NPU 输出验证，实测结果满足讲义要求，掌握了 Host 资源管理、Device 执行和流同步的基本流程。本算子无张量输入，不涉及数值精度或性能测试。')
h2('6 参考资料')

p('［1］HelloWorld算子实验讲义，第1至5页。\n［2］Ascend C异构并行程序设计源代码，2.3.4调用一个昇腾算子目录。\n［3］本次ModelArts环境与NPU运行原始日志，2026年11月5日。')

D.core_properties.title='HelloWorld算子 第10周实验报告'
D.core_properties.author='许盛凯';D.core_properties.subject='Ascend C KernelLaunch 8核NPU运行验证'
path=OUT/(name+'.docx');D.save(path)
# Restore all reference preserve-only XML parts byte for byte.
preserve=['word/styles.xml','word/stylesWithEffects.xml','word/numbering.xml','word/fontTable.xml','word/theme/theme1.xml','word/settings.xml','word/webSettings.xml','word/footer1.xml']
with zipfile.ZipFile(REF) as z: saved={n:z.read(n) for n in preserve if n in z.namelist()}
with zipfile.ZipFile(path) as z: parts={n:z.read(n) for n in z.namelist()}
parts.update(saved)
with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
 for n,data in parts.items():z.writestr(n,data)
print(path)
