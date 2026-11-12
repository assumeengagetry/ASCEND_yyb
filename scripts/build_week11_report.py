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
E=R/'labs/week11_addrelu/evidence'
T=R/'tmp/week11'
OUT=R/'output/week11'
REF=R/'output/week10/2024141460445-许盛凯-第10讲（11月5日）-HelloWorld算子-实验报告.docx'
OUT.mkdir(parents=True,exist_ok=True)
T.mkdir(parents=True,exist_ok=True)
assert hashlib.sha256(REF.read_bytes()).hexdigest()=='4bd1b9b56cb3735c07baab9fa4251a6423ed5bc800bc72eb135d8f276d65a857'
V=json.loads((R/'labs/week11_addrelu/Lab3_AddRelu/results/verification_all.json').read_text())
assert V['passed_cases']==V['total_cases']==10
assert all(c['passed'] and c['max_abs_error']==0 and c['failed_elements']==0 for c in V['cases'])
CASES={c['case']:c for c in V['cases']}
name='2024141460445-许盛凯-第11讲（11月12日）-向量编程基础算子-实验报告'
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

D.add_paragraph('向量编程基础算子\n第11周实验报告','Title')
p('姓名  许盛凯        学号  2024141460445        日期  2026年11月12日',True)
p('本实验基于 Ascend C 向量编程范式实现 AddRelu 算子，完成分块、双缓冲队列及 KernelLaunch 调用。在 ModelArts 的 Ascend 910B4 上执行随机、边界、全负、全正和完全相消五类数据，CPU 调试与真实 NPU 共10组测试全部通过，每组16384个元素，最大绝对误差均为0。')
h('一 实验目的与要求')
p('理解 Global Memory 与 Local Memory 之间的数据搬运，掌握 CopyIn、Compute、CopyOut 三阶段及队列同步；使用基础 API 的矢量双目指令完成 zᵢ=max(xᵢ+yᵢ,0)，并对真实设备输出进行逐元素校验。')
p('输入x、y及输出z均为float32、ND，shape为8×2048；输入范围[-10,10]，输出范围[0,20]。使用8个逻辑执行块，提交完整工程、运行屏显和结果分析。')
h('二 实验环境')
table(['项目','本次实测配置'],[
 ('平台与系统','华为云 ModelArts；Huawei Cloud EulerOS 2.0；aarch64'),
 ('计算设备','Ascend 910B4；32 GB HBM；健康状态 OK'),
 ('CANN与驱动工具','CANN 8.5.2；npu-smi 25.5.1'),
 ('工具链','GCC 10.3.1；CMake 3.22.0；Python 3.12.0；NumPy 1.26.4'),
 ('Toolkit路径','/usr/local/Ascend/ascend-toolkit/latest → cann-8.5.2'),
],[3.3,13.7])
p('本次 npu-smi 可见物理设备编号3，Host 通过 ACL 使用当前容器映射的逻辑设备编号0。工程存放在持久目录 /home/ma-user/work/ascend_week11_20261112/，环境原始记录见 evidence/01_environment.log。')
h('三 实验原理')
p('AddRelu 将两个输入逐元素相加，再把负和截断为0，正和保持不变。融合 AddRelu 一次表达这一双目运算，符合讲义“仅使用基础 API 中矢量双目指令”的计算要求。数据搬运及队列管理由 DataCopy、TPipe 和 TQue 完成。')

page('四 工程准备与关键代码')
h2('1 源码来源与数据类型修正')
p('所给 RAR 包含“3.3向量编程范式”中的 half 类型 Add 示例，未包含独立的 Lab3_AddRelu。本实验以该工程为基础改写，原始文件保存在 original_source/。讲义正文要求 float，但 main 示例残留 sizeof(uint16_t)；本次核函数、Host 内存、数据文件和验证器统一使用 float32 及 sizeof(float)，每份张量均为65536字节。')
h2('2 核间与核内数据划分')
table(['参数','数值','含义'],[
 ('TOTAL_LENGTH / USE_CORE_NUM','16384 / 8','每核 BLOCK_LENGTH=2048元素'),
 ('TILE_NUM / BUFFER_NUM','8 / 2','每核16次 CopyIn→Compute→CopyOut'),
 ('TILE_LENGTH','128','2048÷8÷2；每片512字节'),
 ('队列缓存','3×2×512=3072字节','两输入队列及一输出队列的张量缓存'),
],[5.3,3.6,8.1])
p('第 b 个核的起点为 b×2048，第 i 轮处理核内偏移 i×128 的128个元素，b=0…7，i=0…15。最后一片覆盖16256…16383，全部区间连续且无重叠。512字节片长满足32字节对齐。上述3072字节只计算显式队列张量，不含框架内部开销。')
code('''constexpr int32_t BLOCK_LENGTH = 8 * 2048 / 8;
constexpr int32_t TILE_NUM = 8, BUFFER_NUM = 2;
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM;
xGm.SetGlobalBuffer((__gm__ float*)x
                   + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(float));
// yGm、zGm和另外两个队列按相同规则初始化
for (int32_t i = 0; i < TILE_NUM * BUFFER_NUM; ++i) {
    CopyIn(i); Compute(i); CopyOut(i);
}''',8)
h2('3 搬运与融合计算')
p('CopyIn 分配两个 LocalTensor，以 DataCopy 搬入当前片并入队；Compute 出队取得输入、分配输出后执行以下计算；CopyOut 取出输出、搬回全局内存并释放。EnQue/DeQue 建立阶段间依赖，FreeTensor 使双缓冲槽位可复用。')
code('''auto xLocal = inQueueX.DeQue<float>();
auto yLocal = inQueueY.DeQue<float>();
auto zLocal = outQueueZ.AllocTensor<float>();
AscendC::AddRelu(zLocal, xLocal, yLocal, TILE_LENGTH);
outQueueZ.EnQue<float>(zLocal);
inQueueX.FreeTensor(xLocal);
inQueueY.FreeTensor(yLocal);''')

page('五 编译运行与屏显结果')
h2('1 Host调用与构建')
p('main.cpp 严格检查输入文件长度，申请 Host/Device 缓冲区并搬入x、y，启动8个执行块；流同步成功后拷回z、写出 output_z.bin，最后释放内存、销毁流、复位设备并结束 ACL。各 ACL 接口检查返回值，失败即返回非零退出码。')
code('''constexpr uint32_t BLOCK_DIM = 8;
constexpr size_t BYTE_SIZE = 8 * 2048 * sizeof(float);
// aclInit → aclrtSetDevice(0) → aclrtCreateStream
// 申请Host/Device内存，读取输入，H2D搬运
addrelu_custom_do(BLOCK_DIM, stream, xDevice, yDevice, zDevice);
aclrtSynchronizeStream(stream);
// D2H搬运，写出z，释放资源

// addrelu_custom_do内部提交设备核函数
addrelu_custom<<<blockDim, nullptr, stream>>>(x, y, z);''')
p('CPU 路径使用 CANN tikicpulib 的 GmAlloc、SetKernelMode(AIV_MODE) 及 ICPU_RUN_KF(addrelu_custom, 8, …)，执行同一份 Ascend C 核函数。NPU 路径通过 ascendc.cmake 编译 kernels 静态库并链接 Host；SoC 设置为 Ascend910B4。')
h2('2 执行命令')
code('''cd /home/ma-user/work/ascend_week11_20261112/Lab3_AddRelu
bash run.sh -r cpu -v Ascend910B4
bash run.sh -r npu -v Ascend910B4
bash run_all.sh''',8)
p('run.sh 激活 Toolkit、按模式构建、生成数据、执行程序并验证。run_all.sh 使用已构建的 CPU 和 NPU 程序，分别运行五类数据并逐组归档输入、输出、golden、元数据和日志。编译及回归记录见 evidence/03_cpu_build.log、05_npu_random.log、06_regression.log。')
h2('3 数据生成与验证方法')
p('固定种子为20261112。随机用例从[-10,10]采样并转为float32；边界用例在各128元素片边界及其相邻位置、首尾位置注入±10、0、相消及接近0的正负和。另设全负、全正和 y=-x 三类用例，检查截断、保留及零分支。')
p('NumPy 生成 float32 golden。验证器另以 float64 精确相加两个float32输入，再舍入为float32并取max(·,0)，独立核对 golden；随后检查文件长度、有限值、输入范围、输出非负以及全部16384个元素。误差准则为：')
code('''abs(actual - reference) <= 1e-6 + 1e-6 * abs(reference)
# 任一元素失败即不通过；同时记录每核失败数和最大绝对误差''')

page('五 编译运行与屏显结果 续')
h2('4 真实NPU终端屏显')
p('以下为 ModelArts JupyterLab 终端实际执行随机用例的屏幕截图，裁切显示命令与输出区域。完整截图保存在 evidence/07_actual_terminal.png；图中 NPU_CALL_SUCCESS 表明 Host 设备调用完成，验证器显示 failed=0、max_abs_error=0，并输出 test pass。')
shot=T/'actual_terminal_report_crop.png'
Image.open(E/'07_actual_terminal.png').crop((290,326,1390,760)).save(shot)
fig(shot,'图1  ModelArts真实终端中的AddRelu NPU运行及逐元素验证',17)
h2('5 CPU与NPU测试汇总')
rows=[]
labels=[('random','随机输入'),('boundary','片与核边界'),('all_negative','全负输入'),('all_positive','全正输入'),('cancellation','完全相消')]
for key,label in labels:
 cpu=CASES['cpu_'+key];npu=CASES['npu_'+key]
 rows.append((label,'通过 / 0','通过 / 0','0 / 0'))
table(['用例','CPU 结论 / 失败数','NPU 结论 / 失败数','最大绝对误差\nCPU / NPU'],rows,[4.0,4.4,4.4,4.2])
p('每组均为16384元素、float32，每个输入、golden和实际输出文件均为65536字节。CPU与NPU共检查163840个输出元素，各组8个核的失败计数全为0，输出全部为有限非负值。汇总文件为 Lab3_AddRelu/results/verification_all.json。')
p('随机用例的8216个负和均被截断为0，8168个正和保留；边界用例同时覆盖负、零、正三个分支。全负与相消用例输出全0，全正用例输出等于逐元素和，说明算子完成了加法及ReLU的组合语义。')

page('六 实验分析与总结')
h2('1 数值结果与分支覆盖')
p('边界用例包含8162个负和、99个零和、8123个正和，全部得到正确输出。下表摘录真实NPU结果，按8位有效数字显示；完整float32数据见 results/。')
samples=[('随机0',CASES['npu_random']['sample_elements'][0]),('随机1',CASES['npu_random']['sample_elements'][1]),('边界9',CASES['npu_boundary']['sample_elements'][9]),('边界10',CASES['npu_boundary']['sample_elements'][10])]
table(['样例索引','x','y','实际z'],[
 (label,format(s['x'],'.8g'),format(s['y'],'.8g'),format(s['actual'],'.8g')) for label,s in samples
],[3.0,4.6,4.6,4.8])
h2('2 分块与队列机制分析')
p('8×16×128=16384，数据可整片处理。各核读写独立区间，边界测试与每核0失败验证了偏移及覆盖。EnQue/DeQue 表达就绪关系，FreeTensor 复用队列槽位，双缓冲为搬运与计算流水提供条件。')
h2('3 适配过程中解决的问题')
p('验证器改为检查全部float32元素，并拒绝损坏文件、错误长度及负输出。首次NPU构建因空 CMAKE_BUILD_TYPE 导致合并脚本缺少 --build-type 参数，设置Debug后成功；NPU及sim脚本使用干净重建，避免厂商对象合并脚本的增量构建问题。')
sim_note=R/'tmp/week11/sim_report_note.txt'
if sim_note.exists():p('sim首次链接缺少模型库依赖；在工程sim分支补入model_top、stars等6个库后，干净构建并通过1组随机测试，16384元素的最大误差为0。修复后的完整运行记录见 evidence/09_sim_linkfix_build_and_run.log。')
else:p('工程保留sim入口；本次额外检查时，预装仿真库在链接阶段出现未解析符号，详见 evidence/08_sim_build.log，未将其计为通过。实验正确性结论来自已完成的CPU调试和真实NPU测试。')
h2('4 实验结论')
p('完成 AddRelu 的数据生成、核函数、CPU与NPU调用及验证，五类数据均通过，满足讲义8核和双目计算要求。实验加深了对张量分块、队列同步及融合计算的理解。工程固定shape与dtype，未测量性能。')
h2('5 参考资料')
x=p('［1］向量编程基础算子实验讲义，第1至6页。\n［2］Ascend C异构并行程序设计源代码，3.3向量编程范式目录。\n［3］AddRelu API（8.1.RC1），8.5.2接口已由头文件及实测确认。  ')
x.paragraph_format.line_spacing=1.05
for r in x.runs:r.font.size=Pt(9.5)
# A breakable URL in a small reference paragraph, with an actual hyperlink target.
u='https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/apiref/ascendcopapi/atlasascendc_api_07_0043.html'
rid=D.part.relate_to(u,'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink',is_external=True)
link=OxmlElement('w:hyperlink');link.set(qn('r:id'),rid)
r=OxmlElement('w:r');rp=OxmlElement('w:rPr');sz=OxmlElement('w:sz');sz.set(qn('w:val'),'17');rp.append(sz);r.append(rp)
txt=OxmlElement('w:t');txt.text='AddRelu官方文档';r.append(txt);link.append(r);x._p.append(link)
x.add_run('\n［4］本次ModelArts环境及原始运行记录，2026年11月12日。').font.size=Pt(9.5)

D.core_properties.title='向量编程基础算子 第11周实验报告'
D.core_properties.author='许盛凯'
D.core_properties.subject='Ascend C AddRelu float32 8核CPU与NPU验证'
path=OUT/(name+'.docx');D.save(path)
preserve=['word/styles.xml','word/stylesWithEffects.xml','word/numbering.xml','word/fontTable.xml','word/theme/theme1.xml','word/settings.xml','word/webSettings.xml','word/footer1.xml']
with zipfile.ZipFile(REF) as z:saved={n:z.read(n) for n in preserve if n in z.namelist()}
with zipfile.ZipFile(path) as z:parts={n:z.read(n) for n in z.namelist()}
parts.update(saved)
with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
 for n,data in parts.items():z.writestr(n,data)
print(path)

