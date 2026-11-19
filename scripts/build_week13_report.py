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
LAB=R/'labs/week13_leakyrelu'; E=LAB/'evidence'; T=R/'tmp/week13'; OUT=R/'archive/report_sources/week13'
REF=R/'archive/report_sources/week12/2024141460445-许盛凯-第12讲-向量编程进阶算子-实验报告.docx'
REF_HASH='8510c08300c164f50df53e2390990225cacd6b2c8f57449c72300178746bce28'
assert hashlib.sha256(REF.read_bytes()).hexdigest()==REF_HASH
assert (T/'artifact.md').exists()
A=json.loads((LAB/'AclNNInvocation/results/verification_all.json').read_text())
N=json.loads((LAB/'TorchInvocation/results/verification.json').read_text())
assert A['passed_cases']==A['total_cases']==5 and all(c['passed'] for c in A['cases'])
assert N['passed'] and N['custom_kernel_calls']>=2
CASES={c['case']:c for c in A['cases']}
name='2024141460445-许盛凯-第13讲-算子工程化开发及整网调用-实验报告'
OUT.mkdir(parents=True,exist_ok=True);T.mkdir(parents=True,exist_ok=True)
D=Document(REF)
for el in list(D._element.body):
 if el.tag!=qn('w:sectPr'):D._element.body.remove(el)
for rid,rel in list(D.part.rels.items()):
 if rel.reltype.endswith(('/image','/hyperlink')):D.part.drop_rel(rid)
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

D.add_paragraph('算子工程化开发及整网调用\n第13周实验报告','Title')
p('姓名  许盛凯        学号  2024141460445',True)
p('本实验从 JSON 定义生成 LeakyReluCustom 工程，补全 Host、Tiling 与 Ascend C 核函数，编译安装自定义算子包，并通过 aclnn 两段式接口完成单算子验证。随后编写 PyTorch C++ 适配扩展，将 MLP 中的激活替换为该自定义算子，在真实 NPU 上比较替换前后结果。')
h('一 实验目的与要求')
p('掌握算子定义、工程生成、编译部署、FrameworkLaunch 调用和 AI 框架注册分发的完整流程。基础任务使用 float32、ND，输入输出均为8×200×1024，泄露系数 a 自定义且不能为0；整网任务须给出网络、算子参数与替换位置，提交完整代码、屏显和结果分析。')
h('二 实验环境与算子参数')
table(['项目','本次实验配置'],[
 ('平台','ModelArts；Huawei Cloud EulerOS 2.0；aarch64；Ascend 910B4'),
 ('工具链','CANN 8.5.2；GCC 10.3.1；CMake 3.22.0；Python 3.12.0'),
 ('框架',f"PyTorch {N['torch']}；torch_npu {N['torch_npu']}；设备 {N['device']}"),
 ('工程位置','/home/ma-user/work/ 下的本周实验目录'),
],[3.3,13.7])
table(['参数','形状或取值','数据类型与格式'],[
 ('算子类型 / 核函数','LeakyReluCustom / leaky_relu_custom','Ascend C 自定义算子'),
 ('输入x、输出y','[8,200,1024]；1638400元素','float32；ND；各6553600字节'),
 ('属性a','negative_slope；默认0.1；非零有限值','float标量属性，不是输入张量'),
],[4.0,7.5,5.5])
page('三 实验原理与工程生成')
h2('1 数学计算与实现')
p('LeakyReLU 对非负元素保持原值，对负元素乘泄露系数 a，即 y=x（x≥0），y=a·x（x<0）。实现采用 max(x,0)+a·min(x,0)，由 Maxs、Mins、Muls、Add 组合完成。该表达式在 a>1 时仍正确，因此验证包含 a=1.5，避免误用 max(x,a·x)。')

h2('2 JSON定义')

p('LeakyReluCustom.json 定义算子类型、C++语言、输入输出的float/ND支持范围，以及默认值0.1的 negative_slope 属性。模板生成后保留原始 generated_template/，在 LeakyReluCustom/ 中补全实现，便于核对生成内容与手写内容。')
code('''{
  "op": "LeakyReluCustom", "language": "cpp",
  "input_desc": [{"name": "x", "param_type": "required",
                  "format": ["ND"], "type": ["float"]}],
  "output_desc": [{"name": "y", "param_type": "required",
                   "format": ["ND"], "type": ["float"]}],
  "attr": [{"name": "negative_slope", "param_type": "optional",
            "type": "float", "default_value": 0.1}]
}''')
h2('3 使用MsOpGen生成框架')
code('''source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
msopgen gen -i LeakyReluCustom.json -c ai_core-Ascend910B4 \\
    -lan cpp -out LeakyReluCustom''')
p('生成日志 evidence/01_msopgen.log 记录了 JSON 解析、输入输出与属性识别，以及 op_host、op_kernel、framework 等文件生成成功。上述命令用于首次生成，已有手写代码时应使用新的输出目录，避免覆盖补全后的实现。')
table(['文件或目录','补全职责'],[
 ('op_host/leaky_relu_custom.cpp','注册float32/ND、属性和推导函数；计算分块参数'),
 ('op_host/leaky_relu_custom_tiling.h','序列化totalLength、tileLength与negativeSlope'),
 ('op_kernel/leaky_relu_custom.cpp','分块搬运、基础API计算及写回'),
 ('AclNNInvocation/','两段式单算子调用、输入生成与逐元素校验'),
 ('TorchInvocation/','PyTorch注册分发、NPU适配与整网替换验证'),
],[7.0,10.0])
page('四 Host分块与设备核函数')
h2('1 Tiling参数与块分配')
p('Host 把输入元素数作为 totalLength，每片处理2048元素，将总片数向上取整，再把执行块数设为 min(总片数,8)。参数写入 TilingData，由 CANN 框架传递给设备核函数；本算子不额外申请计算workspace。')
code('''constexpr uint32_t tile = 2048;
const uint64_t tiles = (n + tile - 1) / tile;
data.set_totalLength(n);
data.set_tileLength(tile);
data.set_negativeSlope(*a);
context->SetBlockDim(tiles < 8 ? tiles : 8);
context->GetWorkspaceSizes(1)[0] = 0;''')
p('课件主输入共有800片，8个执行块各处理100片。设备采用交错分配：第 b 块依次处理 b、b+8、b+16…号片。每片8192字节；输入和输出TQue各有两个槽位，另有一个临时TBuf，总显式局部张量空间为40960字节。')
h2('2 设备侧计算与尾片搬运')
code('''for (uint64_t t = GetBlockIdx(); t < tiles; t += GetBlockNum()) {
    uint64_t offset = t * tile_;
    uint32_t count = total_ - offset < tile_ ? total_ - offset : tile_;
    // DataCopyPad 按count*sizeof(float)搬入有效数据
    auto x = qx_.DeQue<float>();
    auto y = qy_.AllocTensor<float>();
    auto neg = tmp_.Get<float>();
    Maxs(y, x, 0.0f, count);
    Mins(neg, x, 0.0f, count);
    PipeBarrier<PIPE_V>();
    Muls(neg, neg, slope_, count);
    PipeBarrier<PIPE_V>();
    Add(y, y, neg, count);
    // 入输出队列，按相同有效字节数搬回GM并释放张量
}''')
p('DataCopyPad 按有效字节长度搬入和搬出，因此非2048整片长度不会越界。计算只处理count个元素；Maxs/Mins分别得到非负和负分支，向量流水屏障保证临时结果在后续计算前可用。257元素用例专门验证短尾片路径。')
h2('3 设备入口')
code('''extern "C" __global__ __aicore__ void leaky_relu_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(data, tiling);
    KernelLeakyRelu op;
    op.Init(x, y, data.totalLength, data.tileLength, data.negativeSlope);
    op.Process();
}''',8)

page('五 编译部署与单算子调用')
h2('1 编译二进制算子包并安装')
p('使用 Release、ascend910b 和独立厂商目录 lab_week13。build_install.sh 先安装基础接口包，再构建binary并重装，生成含设备二进制的 custom_opp_hce_aarch64.run；最终 ENABLE_BINARY_PACKAGE=True。')
code('''bash build_install.sh
# 安装包部署命令由脚本执行
bash LeakyReluCustom/build_out/custom_opp_hce_aarch64.run \\
    --quiet --install-path="$PWD/custom_opp"
source env.sh
cd AclNNInvocation
bash run.sh''',8)
p('env.sh 设置自定义OPP和op_api库路径，系统OPP仍指向Toolkit。日志中 BUILD_INSTALL_BINARY_SUCCESS 确认安装后的 ascend910b 二进制配置存在。')
h2('2 aclnn两段式接口')
code('''uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
aclnnLeakyReluCustomGetWorkspaceSize(
    inputTensor, slope, outputTensor, &workspaceSize, &executor);
// workspaceSize非零时分配设备workspace
aclnnLeakyReluCustom(workspace, workspaceSize, executor, stream);
aclrtSynchronizeStream(stream);
// D2H拷回输出，保存二进制并与golden比较''')
p('第一阶段生成执行器与workspace需求，第二阶段向流提交运算。Host检查shape、文件长度和返回码，以float32/ND创建张量，预填NaN后执行，流同步完成才读回输出。')
h2('3 单算子实测结果')
rows=[]
for key,label in [('random_a01','随机 a=0.1'),('random_a15','随机 a=1.5'),('negative_a02','负值与零 a=0.2'),('positive_a01','非负 a=0.1'),('tail_257_a01','257元素尾片')]:
 c=CASES[key];rows.append((label,str(c['elements']),str(c['failed_elements']),f"{c['max_abs_error']:.8g}"))
table(['用例','输出元素数','失败元素','最大绝对误差'],rows,[5.3,4.0,3.5,4.2])
p('使用固定随机种子（见生成脚本），主输入范围[-10,10]，覆盖零、端点和分块边界。golden按分段公式以float64计算后舍入float32；逐元素误差界为1e-6+1e-6·abs(reference)。五组均通过，原始数据见 AclNNInvocation/results/。')

page('六 PyTorch适配插件与整网替换')
h2('1 针对当前框架的注册方式')
p('课件展示了旧版 PyTorch 2.1.0 的 op-plugin YAML注册及整仓重编译。本实验依据当前已安装框架构建独立 C++ 扩展，通过 TORCH_LIBRARY 定义算子、TORCH_LIBRARY_IMPL 将 PrivateUse1设备分发到NPU实现，并由torch.ops.load_library 加载。没有重装系统 torch_npu。整网构建和运行命令为 bash TorchInvocation/run.sh。')
code('''TORCH_LIBRARY(week13, m) {
    m.def("leaky_relu(Tensor x, float negative_slope=0.1) -> Tensor");
}
TORCH_LIBRARY_IMPL(week13, PrivateUse1, m) {
    m.impl("leaky_relu", &LeakyReluNpu);
}
// Python
torch.ops.load_library("build/week13_leakyrelu_npu.so")
y = torch.ops.week13.leaky_relu(x, 0.1)''')
p('C++适配器检查NPU、连续float32及参数合法性，将输入和新建输出封装为ND aclTensor，直接调用本次生成的 aclnnLeakyReluCustomGetWorkspaceSize 与 aclnnLeakyReluCustom。它没有用内置LeakyReLU或Python逐元素表达式代替设备计算，也没有使用课件示例的 EXEC_NPU_CMD 宏。')
h2('2 调用链与同步')
code('''torch.ops.week13.leaky_relu
  -> PyTorch PrivateUse1 dispatcher
  -> LeakyReluNpu C++ adapter
  -> aclnnLeakyReluCustomGetWorkspaceSize / aclnnLeakyReluCustom
  -> installed Ascend C leaky_relu_custom''')
p('适配器使用输入设备的当前NPU流，在直接ACL调用前后同步，使 PyTorch 已排队计算与自定义运算的数据依赖一致。每次自定义调用成功后增加计数器，验证程序据此检查确实走过自定义接口。该同步实现以正确性验证为目标，不能据此推断算子性能。')
h2('3 网络结构和替换位置')
table(['阶段','算子与参数','张量形状'],[
 ('输入','固定种子生成的float32输入','[1600,1024]'),
 ('第一层','Linear(1024,1024)','[1600,1024]'),
 ('替换位置','LeakyReLU(0.1) → 自定义LeakyRelu','[1600,1024]；1638400元素'),
 ('第二层','Linear(1024,64)','[1600,64]；102400元素'),
],[3.0,8.0,6.0])
p('两网共享完全相同的权重与输入，只替换两层之间的激活；中间元素数与课件一致。')

page('七 整网结果与运行证据')
h2('1 网络替换代码')
code('''hidden = self.fc1(x)
# 原网络：activated = self.activation(hidden)
activated = (torch.ops.week13.leaky_relu(hidden, 0.1)
             if self.custom else self.activation(hidden))
return self.fc2(activated)''')
p('验证在 torch.inference_mode() 下执行，并关闭 torch_npu 内部阻塞格式，使传入ACL的张量保持ND布局。单独比较自定义激活与NumPy数学基准、内置NPU激活，再比较两网最终输出。所有元素满足 abs(error)≤1e-5+1e-4·abs(reference) 才算通过。')
rows=[]
for key,label in [('direct_activation_vs_math','自定义激活 / 数学基准'),('direct_activation_vs_builtin_npu','自定义激活 / 内置NPU'),('network_original_vs_replaced','替换整网 / 原始整网')]:
 c=N[key];rows.append((label,str(c['numel']),str(c['failed_elements']),f"{c['max_abs_error']:.8g}"))
table(['比较对象','输出元素数','失败元素','最大绝对误差'],rows,[6.0,3.5,3.0,4.5])
p(f"自定义接口成功调用{N['custom_kernel_calls']}次。输入、权重、中间激活及两网输出保存在 TorchInvocation/results/。")
h2('2 实验运行屏显')
p('图1裁取本地网页控制台经SSH读取已保存真实NPU结果的输出区域；完整截图及原始日志随工程保存。')
shot=E/'live_console.png';assert shot.exists(), 'Awaiting real screenshot'
im=Image.open(shot)
# Crop only surrounding page chrome and blank right space; all result text remains.
assert im.size == (1280, 720)
cropped=T/'live_console_results.png'
im.crop((24,259,1000,675)).save(cropped)
fig(cropped,'图1  自定义LeakyRelu单算子及整网验证结果汇总屏显（裁切）',17)

page('八 实验分析与总结')
h2('1 构建过程中的适配')
p('首次打包开启二进制选项时，设备binary目录尚未生成；改为先安装基础接口包，再显式构建binary并重新打包。另一次脚本沿用euleros包名，当前系统实际生成hce包名，已改为匹配当前产物。02、03号日志保留失败，04号日志记录最终安装成功。')

h2('2 结果解释与范围')
p('单算子验证覆盖负分支、非负分支、非默认泄露系数和短尾片。a=1.5用例检验了实现没有错误地把LeakyReLU简化为max(x,a·x)。整网验证在保持输入与权重不变的条件下检查激活和最终输出；通过结论只针对本次输入与推理路径。')
p(f"激活输入含{N['negative_activation_inputs']}个负值和{N['nonnegative_activation_inputs']}个非负值，覆盖两个分支。两次自定义调用分别来自独立激活验证与整网前向中的激活替换。")
p('独立扩展完成真实的PyTorch注册分发与自定义aclnn调用，与课件旧版实现的集成目标一致，但没有按旧版本命令重新构建整个torch_npu。内核已通过257元素尾片验证；当前PyTorch适配器仍限制连续float32输入的元素数为8的倍数。扩展未注册autograd，本实验不验证反向传播、训练收敛或性能提升；当前流同步也会影响耗时，不能将该实现作为性能优化结论。')
h2('3 实验结论与参考资料')
p('完成 JSON驱动生成、Host与设备实现、二进制编译部署、aclnn单算子调用和PyTorch整网替换。单算子五组、激活两项比较与整网输出比较均通过，说明自定义算子已进入实际框架调用链。')
x=p('［1］《算子工程化开发及整网调用》实验讲义，第1至6页。\n［2］本次 MsOpGen生成模板、LeakyReluCustom源码、ACL及PyTorch调用工程。\n［3］本次ModelArts实测日志与二进制结果。')
x.paragraph_format.line_spacing=1.05
for r in x.runs:r.font.size=Pt(9.5)

D.core_properties.title='算子工程化开发及整网调用 第13周实验报告'
D.core_properties.author='许盛凯';D.core_properties.subject='LeakyReluCustom工程化开发 aclnn与PyTorch MLP整网替换'
for el in list(D.core_properties._element):
 if el.tag.rsplit('}',1)[-1] in {'created','modified','lastPrinted'}:
  D.core_properties._element.remove(el)
path=OUT/(name+'.docx');D.save(path)
preserve=['_rels/.rels','customXml/_rels/item1.xml.rels','customXml/item1.xml','customXml/itemProps1.xml','word/styles.xml','word/stylesWithEffects.xml','word/numbering.xml','word/fontTable.xml','word/theme/theme1.xml','word/settings.xml','word/webSettings.xml','word/footer1.xml']
with zipfile.ZipFile(REF) as z:saved={n:z.read(n) for n in preserve}
with zipfile.ZipFile(path) as z:parts={n:z.read(n) for n in z.namelist()}
parts.update(saved)
with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
 for n,data in parts.items():z.writestr(n,data)
assert hashlib.sha256(REF.read_bytes()).hexdigest()==REF_HASH
print(path)
