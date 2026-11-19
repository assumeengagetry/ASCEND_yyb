from pathlib import Path
import hashlib,json,zipfile
from docx import Document
from docx.shared import Cm,Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from PIL import Image
R=Path(__file__).resolve().parents[1]
LAB=R/'labs/week12_sinh'; E=LAB/'evidence'; T=R/'tmp/week12'; OUT=R/'archive/report_sources/week12'
REF=R/'output/week11/2024141460445-许盛凯-第11讲（11月12日）-向量编程基础算子-实验报告.docx'
REF_HASH='370a7ff80b993184ecbe491013e6a64c0185d23f28cb6b7e444d2ee9aa4bbdb5'
assert hashlib.sha256(REF.read_bytes()).hexdigest()==REF_HASH
V=json.loads((LAB/'verification_all.json').read_text())
assert V['total_cases']==V['passed_cases']==20
assert all(c['passed'] and c['failed_elements']==0 for c in V['cases'])
assert all(c['bitwise_equal'] for c in V['version_comparisons'])
CASES={(c['project'],c['mode'],c['case']):c for c in V['cases']}
name='2024141460445-许盛凯-第12讲-向量编程进阶算子-实验报告'
OUT.mkdir(parents=True,exist_ok=True); T.mkdir(parents=True,exist_ok=True)
D=Document(REF)
for el in list(D._element.body):
 if el.tag!=qn('w:sectPr'): D._element.body.remove(el)
for rid,rel in list(D.part.rels.items()):
 if rel.reltype.endswith(('/image','/hyperlink')): D.part.drop_rel(rid)
def p(s='',bold=False):
 x=D.add_paragraph();r=x.add_run(s);r.bold=bold;return x

def h(s):return D.add_heading(s,1)
def h2(s):return D.add_heading(s,2)
def page(s):
 x=h(s);x.paragraph_format.page_break_before=True;return x
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
 x.paragraph_format.line_spacing=1.0;x.alignment=WD_ALIGN_PARAGRAPH.CENTER
 x.add_run().add_picture(str(path),width=Cm(width))
 D.add_paragraph(caption,'Caption')

D.add_paragraph('向量编程进阶算子\n第12周实验报告','Title')
p('姓名  许盛凯        学号  2024141460445',True)
p('本实验使用 Ascend C 基础 API 实现 Sinh，完成不使用 TBuf 和使用 TBuf 的两个版本。在 ModelArts 的 Ascend 910B4 上，对五类输入分别进行 CPU 调试与真实 NPU 运行，共20组测试、327680个输出元素全部通过逐元素校验；两种实现的对应输出逐位一致。')
h('一 实验目的与要求')
p('掌握基础矢量 API 的组合计算，理解 TQue 数据流转和 TBuf 临时存储。完成 Lab4_Sinh、Lab4_SinhTBuf 两个工程的数据生成、核函数、CPU/NPU 调用与验证，保存屏显并分析结果。')
p('输入为 float32、ND，shape 为8×2048，元素取值范围[1,10]，启动8个逻辑执行块。不得直接调用高阶数学库中的 Sinh；两版均采用 CopyIn、Compute、CopyOut 三阶段数据通路。')
h('二 实验环境')
table(['项目','本次实测配置'],[
 ('平台与系统','华为云 ModelArts；Huawei Cloud EulerOS 2.0；aarch64'),
 ('计算设备','Ascend 910B4；32 GB HBM；健康状态 OK'),
 ('CANN与驱动工具','CANN 8.5.2；npu-smi 25.5.1'),
 ('开发工具链','GCC 10.3.1；CMake 3.22.0；Python 3.12.0'),
 ('Toolkit路径','/usr/local/Ascend/ascend-toolkit/latest → cann-8.5.2'),
],[3.3,13.7])
p('本次 npu-smi 显示物理设备编号1，Host 使用容器内 ACL 逻辑设备编号0。工程保存于 /home/ma-user/work/ 下的本周实验目录。环境记录见 evidence/01_environment.log。')
h('三 实验原理')
p('双曲正弦满足 sinh(x)=(exp(x)−exp(−x))/2。本实验先用 Exp 计算正指数，再用 Muls 取负并通过 Exp 计算负指数，最后用 Sub 求差、Muls 乘0.5。该路径只组合基础 API；两版执行同一运算顺序，仅中间结果的存储位置不同。')

page('四 工程准备与关键代码')
h2('1 源码来源与类型修正')
p('配套 RAR 的“3.6.1双曲正弦函数算子核函数的实现”提供 half 类型 Sinh 示例，但没有完整的两个 Lab4 目录。本实验保留该源码于 original_source/，结合向量编程工程补全两个交付工程。课件 main 示例残留 sizeof(uint16_t) 及三地址参数模板；本次统一为 float32、sizeof(float) 和一输入一输出的 (x,z) 接口，每份张量为65536字节。')
h2('2 核间划分与双缓冲')
table(['参数','数值','对应关系'],[
 ('TOTAL_LENGTH / USE_CORE_NUM','16384 / 8','每核处理2048元素'),
 ('TILE_NUM / BUFFER_NUM','8 / 2','每核循环16次'),
 ('TILE_LENGTH','128','每片512字节，满足32字节对齐'),
 ('TQue显式张量缓存','2048字节','输入和输出队列各2×512字节'),
],[5.3,3.6,8.1])
p('第 b 个执行块从 b×2048 开始，第 i 轮处理 i×128 偏移处的128个元素。8×16×128=16384，最后一片覆盖16256…16383，无尾块和跨核写入。CopyIn 将当前片搬入输入队列；Compute 出队并计算；CopyOut 将结果搬回对应 GM 区间，随后释放队列张量。')
code('''xGm.SetGlobalBuffer((__gm__ float*)x
    + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(float));
pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(float));
for (int32_t i = 0; i < TILE_NUM * BUFFER_NUM; ++i) {
    CopyIn(i); Compute(); CopyOut(i);
}''')
h2('3 不使用TBuf的计算')
code('''auto xLocal = inQueueX.DeQue<float>();
auto zLocal = outQueueZ.AllocTensor<float>();
AscendC::Exp(zLocal, xLocal, TILE_LENGTH);
AscendC::Muls(xLocal, xLocal, -1.0f, TILE_LENGTH);
AscendC::Exp(xLocal, xLocal, TILE_LENGTH);
AscendC::Sub(zLocal, zLocal, xLocal, TILE_LENGTH);
AscendC::Muls(zLocal, zLocal, 0.5f, TILE_LENGTH);
outQueueZ.EnQue(zLocal);
inQueueX.FreeTensor(xLocal);''')
p('此版没有 TBuf 对象。xLocal 在搬入后只供本片使用，因此可以原位取负并改写为负指数；zLocal 从正指数逐步变为差值与最终输出。EnQue/DeQue 表达阶段间就绪关系，FreeTensor 使队列槽位可复用。')

page('四 工程准备与关键代码 续')
h2('4 使用TBuf保存中间量')
p('TBuf 只用于核内临时存储，不需要像输入输出队列那样入队和出队。Init 中为 expBuffer 分配一个128元素的 float 缓冲区，Compute 中通过 Get<float>() 取得 LocalTensor。每次计算复用同一缓冲区。')
code('''AscendC::TBuf<AscendC::QuePosition::VECCALC> expBuffer;
// Init 中申请512字节的临时区
pipe.InitBuffer(expBuffer, TILE_LENGTH * sizeof(float));

// Compute 中取得原始输入和输出张量后
auto expLocal = expBuffer.Get<float>();
AscendC::Exp(expLocal, xLocal, TILE_LENGTH);
AscendC::Muls(zLocal, xLocal, -1.0f, TILE_LENGTH);
AscendC::Exp(zLocal, zLocal, TILE_LENGTH);
AscendC::Sub(zLocal, expLocal, zLocal, TILE_LENGTH);
AscendC::Muls(zLocal, zLocal, 0.5f, TILE_LENGTH);''')
p('expLocal 的语义始终是 exp(x)，zLocal 依次存储 −x、exp(−x) 和最终结果，xLocal 保持原值。两版显式队列缓存均为2048字节，TBuf 版额外使用512字节，合计2560字节；这些数值不包含框架内部开销。')
h('五 编译运行与结果验证')
h2('1 Host调用和构建')
p('CPU 路径使用 tikicpulib 的 GmAlloc、SetKernelMode(AIV_MODE) 和 ICPU_RUN_KF，运行同一份核函数。NPU 路径由 ACL 初始化、创建流、申请内存、H2D搬运、KernelLaunch、流同步、D2H搬运及资源释放组成，各 ACL 调用检查返回值。输出区预填 NaN 字节，遗漏计算的区域不能作为有效结果通过校验。')
code('''constexpr uint32_t BLOCK_DIM = 8;
constexpr size_t BYTE_SIZE = 8 * 2048 * sizeof(float);
sinh_custom_do(BLOCK_DIM, stream, xDevice, zDevice);
// wrapper 内部使用 sinh_custom<<<blockDim, nullptr, stream>>>(x,z);
// TBuf 工程对应 sinhtbuf_custom_do / sinhtbuf_custom
aclrtSynchronizeStream(stream); // 完成后才读取设备输出''')
h2('2 复现命令')
code('''cd "$WEEK12_WORKDIR"  # 本周实验工程根目录
for project in Lab4_Sinh Lab4_SinhTBuf; do
  (cd "$project"
   bash run.sh -r cpu -v Ascend910B4
   bash run.sh -r npu -v Ascend910B4)
done
bash run_all.sh''',8)
p('run.sh 构建并验证单组数据；run_all.sh 逐组运行并归档数据和日志。两工程保留 sim 入口，本次未执行仿真，结果仅统计 CPU 调试与真实 NPU。')

page('五 编译运行与结果验证 续')
h2('3 数据生成与逐元素校验')
p('五类数据均保持 float32、16384元素和[1,10]范围。random 使用固定随机种子（见生成脚本）；boundary 在片与核边界及其邻近位置加入1、10及相邻可表示值；ramp 在区间内递增；ones、tens 分别检查上下界常量。两版和两种执行模式使用相同输入。')
p('golden 由 NumPy 以 float64 计算 sinh，再舍入为 float32；验证器根据实际输入重新计算这一数学基准，核对 golden 未被替换。校验还检查文件字节数、元素数、输入范围、有限值、输出正值和每核失败数；全部元素必须满足预先固定的误差准则：')
code('''abs(actual - reference) <= 1e-6 + 1e-5 * abs(reference)
reference = float32(sinh(float64(input_x)))
# 任一元素不满足即失败；输出或golden含NaN/Inf也失败''')
h2('4 二十组测试汇总')
rows=[]
for key,label in [('random','随机'),('boundary','分块边界'),('ramp','递增'),('ones','全1'),('tens','全10')]:
 cs=[CASES[(project,mode,key)] for project in ('Lab4_Sinh','Lab4_SinhTBuf') for mode in ('cpu','npu')]
 rows.append((label,'4 / 4','0',f"{max(c['max_abs_error'] for c in cs):.7g}",f"{max(c['max_rel_error'] for c in cs):.7g}"))
table(['用例','通过组数','失败元素','最大绝对误差','最大相对误差'],rows,[2.8,2.6,2.6,4.5,4.5])
p('每类包含 Sinh CPU、Sinh NPU、SinhTBuf CPU、SinhTBuf NPU 四组。20组共327680个输出元素均通过，每组8个执行块的失败计数均为0。两版在同一模式同一输入下的10次输出比较均逐位一致，说明 TBuf 的引入没有改变本次计算结果。汇总见 verification_all.json。')
h2('5 真实NPU输出样例')
s=CASES[('Lab4_Sinh','npu','random')]['sample_elements']
table(['索引','输入x','数学基准','NPU输出'],[(str(x['index']),f"{x['x']:.9g}",f"{x['reference']:.9g}",f"{x['actual']:.9g}") for x in [s[0],s[1],s[6],s[7]]],[1.8,4.6,5.3,5.3])
p('表中采用9位有效数字；索引0和7有舍入差异，完整二进制结果保留全部精度。')

page('五 编译运行与结果验证 续')
h2('6 实时NPU运行屏显')
p('图1、图2裁取本地网页控制台经 SSH 在 ModelArts 执行 show_npu.sh 的真实输出与校验区域。两版均为 NPU 执行，failed=0 且 test pass；完整日志和截图保存在 evidence/。')
shot=E/'09_live_npu_console.png'
assert shot.exists(), 'Waiting for actual browser screenshot'
im=Image.open(shot)
assert im.size==(725,1160),im.size
for label,box in [('Sinh',(22,356,703,679)),('SinhTBuf',(22,793,703,1127))]:
 crop=T/f'npu_console_{label}_undated.png'
 im.crop(box).save(crop)
 fig(crop,f'图{1 if label=="Sinh" else 2}  {label}真实NPU输出与校验屏显（裁切）',17)

page('六 实验分析与总结')
h2('1 倒数路径误差与修正')
p('参考实现以 Reciprocal(exp(x)) 计算负指数。首轮 CPU 与 NPU 随机测试均有2015个元素超出误差界限，最大相对误差约3.43674×10⁻⁴；失败记录保留在 evidence/02、03、04号日志。')
p('最终改用两次 Exp 直接计算正、负指数，保持 rtol=1e-5、atol=1e-6。修正后随机测试失败数为0，最大相对误差约1.22370×10⁻⁷，五类数据均通过；成功构建与运行见05、06号日志。')
h2('2 两种中间存储方式的比较')
table(['比较项','Lab4_Sinh','Lab4_SinhTBuf'],[
 ('中间量位置','复用输入与输出LocalTensor','exp(x)保存在独立TBuf'),
 ('输入局部张量','计算中被改写为exp(−x)','计算后仍保持原始x'),
 ('显式缓存总量','2048字节','2560字节'),
 ('基础计算API次数','2次Exp、2次Muls、1次Sub','2次Exp、2次Muls、1次Sub'),
 ('本次同模式输出比较','与TBuf版逐位一致','与无TBuf版逐位一致'),
],[3.4,6.8,6.8])
p('无 TBuf 版节省临时空间，但需跟踪张量的中间含义；TBuf 版额外使用512字节，使输入、中间量和输出的职责更清楚。TBuf 本身不保证更快或更精确，本实验未计时，不给出性能优劣结论。')
h2('3 数据范围与验证结论')
p('Sinh 在[1,10]内从约1.175增至11013.23。最大绝对误差为0.0009765625，最大相对误差约1.224×10⁻⁷，均满足固定逐元素准则。“通过”并不等于数学真值零误差。边界与递增数据还检验了分片和核间输出覆盖。')
p('工程固定为16384个 float32 元素，未覆盖任意长度尾块、其他类型或区间外数据；不能将本次结果推广到指数溢出或接近0时的相消误差情形。')
h2('4 实验结论与参考资料')
p('完成两种 Sinh 工程及 CPU/NPU 调用，20组测试全部通过。基础 API 可组合实现双曲正弦，TBuf 用额外空间显式组织中间量。工程包保存完整源码、数据、失败与成功日志以及截图。')
x=p('［1］《向量编程进阶算子》实验讲义，第1至8页。\n［2］《Ascend C异构并行程序设计源代码》，3.6.1双曲正弦函数算子核函数的实现。\n［3］本次 ModelArts 原始环境、构建与运行记录。')
x.paragraph_format.line_spacing=1.05
for r in x.runs:r.font.size=Pt(9.5)
D.core_properties.title='向量编程进阶算子 第12周实验报告'
D.core_properties.author='许盛凯'
D.core_properties.subject='Ascend C Sinh与SinhTBuf float32 8核CPU及NPU验证'
for el in list(D.core_properties._element):
 if el.tag.rsplit('}',1)[-1] in {'created','modified','lastPrinted'}:
  D.core_properties._element.remove(el)
path=OUT/(name+'.docx'); D.save(path)
preserve=['_rels/.rels','customXml/_rels/item1.xml.rels','customXml/item1.xml','customXml/itemProps1.xml','word/styles.xml','word/stylesWithEffects.xml','word/numbering.xml','word/fontTable.xml','word/theme/theme1.xml','word/settings.xml','word/webSettings.xml','word/footer1.xml']
with zipfile.ZipFile(REF) as z:saved={n:z.read(n) for n in preserve}
with zipfile.ZipFile(path) as z:parts={n:z.read(n) for n in z.namelist()}
parts.update(saved)
with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
 for n,data in parts.items():z.writestr(n,data)
assert hashlib.sha256(REF.read_bytes()).hexdigest()==REF_HASH
print(path)
