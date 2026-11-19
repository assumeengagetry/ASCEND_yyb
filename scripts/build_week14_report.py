"""Build the week 14 report exclusively from recorded experiment data.

Run with the Codex bundled Python. The canonical report inputs are generated
from the experiment's raw CSV and verification JSON by prepare_report_data().
Missing data or screenshots deliberately stop creation instead of filling values.
"""
from pathlib import Path
import hashlib, json, zipfile, math, csv
from docx import Document
from docx.shared import Cm, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parents[1]
LAB = R / 'labs/week14_matmul'
TMP = R / 'tmp/week14'
OUT = R / 'archive/report_sources/week14'
REF = R / 'archive/report_sources/week14/template_reference.docx'
REF_HASH = '8f0460f66fc97ed50de7c07080d95a97f3b00a9205da32bba20d5e5885792792'
NAME = '2024141460445-许盛凯-第14讲-矩阵运算性能探究-实验报告'
PRESERVE = ['_rels/.rels','customXml/_rels/item1.xml.rels','customXml/item1.xml','customXml/itemProps1.xml',
            'word/styles.xml','word/stylesWithEffects.xml','word/numbering.xml','word/fontTable.xml',
            'word/theme/theme1.xml','word/settings.xml','word/webSettings.xml','word/footer1.xml']

def prepare_report_data():
    source = LAB / 'performance_summary.json'
    raw = json.loads(source.read_text())
    assert raw['actual_date'] == '2026-11-19'
    expected = [('cores5_base128','A 5核',5,[128,128,128]),
                ('cores20_base64','B 64³',20,[64,64,64]),
                ('cores20_base128','C 128³',20,[128,128,128]),
                ('cores20_base128x256','D 128×256',20,[128,256,64])]
    by_name={c['name']:c for c in raw['configurations']}
    cases=[]
    for name,label,cores,base_shape in expected:
        c=by_name[name]; t=c['tiling']; v=c['verification']; m=c['metrics']
        assert c['total_calls']==20 and c['warmup_excluded']==5 and c['samples']==15
        assert t['blockDim']==cores and t['usedCoreNum']==2*cores
        assert [t[k] for k in ('baseM','baseN','baseK')]==base_shape
        assert v['passed'] and v['failed_elements']==0 and v['numel']==4096*5120
        assert v['all_finite'] and v['rtol']==1e-4 and v['atol']==.02
        for metric in ('Task Duration(us)','aicore_time(us)','aic_mac_time(us)','aic_mte2_time(us)'):
            values=m[metric]['samples'];assert len(values)==15
            assert all(math.isfinite(x) and x>0 for x in values)
            assert math.isclose(float(np.median(values)),m[metric]['median'],rel_tol=1e-12)
        csv_path=LAB/c['source_csv']
        assert csv_path.exists(), f'Missing raw CSV: {csv_path}'
        assert hashlib.sha256(csv_path.read_bytes()).hexdigest()==c['source_csv_sha256']
        with csv_path.open() as stream:
            rows=[r for r in csv.DictReader(stream) if r['Op Name']=='matmul_custom']
        rows.sort(key=lambda r:float(r['Task Start Time(us)']))
        assert len(rows)==20
        assert all(int(r['Block Dim'])==cores and int(r['Mix Block Dim'])==2*cores for r in rows)
        for metric in ('Task Duration(us)','aicore_time(us)','aic_mac_time(us)','aic_mte2_time(us)'):
            assert [float(r[metric]) for r in rows[5:]]==m[metric]['samples']
        cases.append(dict(label=label,name=name,cores=cores,base=base_shape,
            used_core_num=t['usedCoreNum'],single_core=[t[k] for k in ('singleCoreM','singleCoreN','singleCoreK')],
            duration_us=m['aicore_time(us)']['samples'],mac_us=m['aic_mac_time(us)']['samples'],
            mte2_us=m['aic_mte2_time(us)']['samples'],verification=v,tiling=t))
    mac_speed=np.median(cases[0]['mac_us'])/np.median(cases[2]['mac_us'])
    mte_speed=np.median(cases[0]['mte2_us'])/np.median(cases[2]['mte2_us'])
    base_gain=(1-np.median(cases[3]['duration_us'])/np.median(cases[2]['duration_us']))*100
    data=dict(cases=cases,metadata=raw['input_metadata'],samples=15,warmup=5,
        environment={'device':'华为云ModelArts；Ascend 910B4；20个Cube、40个Vector核',
                     'software':'EulerOS 2.0 aarch64；CANN 8.5.2；GCC 10.3.1；CMake 3.22.0'},
        metric_mapping=[('aicore_time(us)','aicore_time(us)','AI Core时间，主比较指标'),
                        ('aic_mac_time(us)','aic_mac_time(us)','矩阵计算活动时间'),
                        ('aic_mte2_time(us)','aic_mte2_time(us)','矩阵输入搬运活动时间')],
        tiling_note='SetFixSplit文档规定baseK参数仅支持-1，不能据此锁定K［4］。64×64请求自动得到K=256，首次运行被检查中止；最终把baseK改为64，并将stepKa、stepKb、depthA1、depthB1各乘4，保持生成的L1跨度与预算。实际tiling及200字节原始结构随结果保存，四配置的最终采样输出均通过全量验证。',
        analysis={'core':f'MAC中位数从2834.469降为712.628 μs，缩短{mac_speed:.3f}倍，接近理想4倍；MTE2仅缩短{mte_speed:.3f}倍，与总体加速幅度接近。这支持进一步检查搬运相关等待的假设，但活动计数不能证明其为唯一瓶颈。',
                  'base':f'三种配置的P排序与本次aicore_time排序一致。128×256×64相对128³将总时间降低{base_gain:.2f}%；二者MAC分别为712.628与711.773 μs，接近，而MTE2从1482.120降为1170.292 μs。该现象与搬运相关活动改善相符，但完整tiling也有变化，尚不能仅由P认定因果。'})
    TMP.mkdir(parents=True,exist_ok=True)
    (TMP/'report_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    return data

def draw_chart(cases):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1,2,figsize=(10.0,3.25),layout='constrained')
    selections = [([cases[0], cases[2]], ['5 cores','20 cores'], '(a) Core count; base 128 x 128 x 128'),
                  ([cases[1], cases[2], cases[3]], ['64 x 64 x 64','128 x 128 x 128','128 x 256 x 64'], '(b) Base shape; 20 cores')]
    for ax,(items, labels, title) in zip(axes,selections):
        x=np.arange(len(items)); med=[float(np.median(c['duration_us'])) for c in items]
        lo=[np.min(c['duration_us']) for c in items]; hi=[np.max(c['duration_us']) for c in items]
        ax.bar(x,med,width=.52,color=['#546A7B','#317B80','#80A7A4'][:len(items)],zorder=2)
        ax.errorbar(x,med,yerr=[np.array(med)-lo,np.array(hi)-med],fmt='none',color='#222222',capsize=4,lw=1,zorder=3)
        for i,c in enumerate(items):
            jitter=np.linspace(-.10,.10,len(c['duration_us']))
            ax.scatter(i+jitter,c['duration_us'],s=10,color='#26333B',alpha=.5,zorder=4)
        for i,v in enumerate(med):ax.text(i,hi[i]+max(hi)*.04,f'{v:.2f}',ha='center',va='bottom',fontsize=9)
        ax.set_xticks(x,labels,fontsize=8.5);ax.set_ylabel('AI Core time (us)');ax.set_title(title,fontsize=10)
        ax.set_ylim(0,max(hi)*1.23);ax.grid(axis='y',color='#dddddd',lw=.6,zorder=0)
    target=TMP/'performance.png';fig.savefig(target,dpi=200);plt.close(fig);return target

def build(data):
    assert hashlib.sha256(REF.read_bytes()).hexdigest()==REF_HASH
    assert (TMP/'artifact.md').exists()
    OUT.mkdir(parents=True,exist_ok=True); TMP.mkdir(parents=True,exist_ok=True)
    cases=data['cases']; meta=data['metadata']; samples=data['samples']; warmup=data['warmup']
    med=lambda c:float(np.median(c['duration_us']))
    core_speed=med(cases[0])/med(cases[2]); efficiency=core_speed/4
    best=min(cases[1:],key=med); base_speed=med(cases[1])/med(best)
    maxerr=max(c['verification']['max_abs_error'] for c in cases)
    maxrel=max(c['verification']['max_rel_error'] for c in cases)
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
        x=D.add_paragraph();f=x.paragraph_format;f.space_after=Pt(7);f.line_spacing=1.08
        for i,line in enumerate(s.strip().splitlines()):
            r=x.add_run(('\n' if i else '')+line);r.font.name='Courier New';r.font.size=Pt(size)
            r._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Songti SC')
        return x
    def table(headers,rows,widths):
        t=D.add_table(rows=1,cols=len(headers));t.style='Table Grid';t.autofit=False;t.alignment=WD_TABLE_ALIGNMENT.CENTER
        for c,w in zip(t.columns,widths):c.width=Cm(w)
        for c,s in zip(t.rows[0].cells,headers):c.text=s
        for row in rows:
            for c,s in zip(t.add_row().cells,row):c.text=str(s)
        borders=OxmlElement('w:tblBorders')
        for k in ['top','left','bottom','right','insideH','insideV']:
            b=OxmlElement('w:'+k);b.set(qn('w:val'),'single');b.set(qn('w:sz'),'4');b.set(qn('w:color'),'D9D9D9');borders.append(b)
        t._tbl.tblPr.append(borders)
        for i,row in enumerate(t.rows):
            rp=row._tr.get_or_add_trPr();rp.append(OxmlElement('w:cantSplit'))
            if i==0:rp.append(OxmlElement('w:tblHeader'))
            for j,c in enumerate(row.cells):
                c.width=Cm(widths[j]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                for x in c.paragraphs:
                    x.paragraph_format.space_before=Pt(3);x.paragraph_format.space_after=Pt(3);x.paragraph_format.line_spacing=1.12
                    for r in x.runs:r.font.size=Pt(9.5);r.bold=i==0
                if i==0:
                    sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'E9EDF2');c._tc.get_or_add_tcPr().append(sh)
        blank=p();blank.paragraph_format.space_after=Pt(0);blank.paragraph_format.space_before=Pt(0);blank.paragraph_format.line_spacing=1.0
        for r in blank.runs:r.font.size=Pt(3)
        return t
    def fig(path,caption,width=17):
        x=D.add_paragraph();x.paragraph_format.keep_with_next=True;x.paragraph_format.space_after=Pt(3)
        x.paragraph_format.line_spacing=1.0;x.alignment=WD_ALIGN_PARAGRAPH.CENTER
        x.add_run().add_picture(str(path),width=Cm(width));D.add_paragraph(caption,'Caption')
    def base(c):return '×'.join(map(str,c['base']))
    def val(v):return f'{v:.3f}'

    D.add_paragraph('矩阵运算性能探究\n第14周实验报告','Title')
    p('姓名  许盛凯        学号  2024141460445',True)
    p(f'本实验在 Ascend 910B4 上使用 Matmul 高阶 API，比较5核与20核，以及20核下三种基本块。四种配置使用完全相同的输入，均通过20971520个输出元素的全量校验。以 msprof 稳态aicore_time中位数计，5核到20核的加速比为{core_speed:.3f}，20核中耗时最小的是{base(best)}。')
    h('一 实验目的与要求')
    p('通过 KernelLaunch 调用矩阵乘法，使用 msprof 采集真实 NPU 性能，检验“增加核数带来线性加速”和“基本块计算访存比越大性能越好”两项假设。采用控制变量方法，区分数学模型、设备实测和原因推断。')
    h('二 实验环境与数据规模')
    table(['项目','本次实验配置'],[
        ('平台与设备',data['environment']['device']),
        ('系统与软件',data['environment']['software']),
        ('核心与调用','Cube与Vector配比1:2；KernelLaunch；ACL逻辑设备0'),
        ('性能来源','CANN msprof设备采集；保留op_summary CSV原始列名'),
    ],[3.3,13.7])
    table(['张量','形状和类型','存储大小'],[
        ('输入A','4096×4096，fp16，ND','33554432字节，32 MiB'),
        ('输入B','4096×5120，fp16，ND','41943040字节，40 MiB'),
        ('输出C','4096×5120，float32，ND','83886080字节，80 MiB'),
    ],[2.4,9.6,5.0])
    p('计算 C=A×B，无bias、不转置，约需2MNK=171798691840次浮点运算。输入先取[1,10]随机数再舍入为fp16，采用固定种子，并显式放入1和10。两矩阵在所有配置中共用同一对二进制文件。')
    p(f'本次CPU基准输出范围为[{meta["golden_range"][0]:.3f}, {meta["golden_range"][1]:.3f}]，超过fp16有限最大值65504，故输出与累加采用float32。该选择保留课件规定的fp16输入，避免输出溢出影响验证。')

    page('三 理论假设与对照设计')
    h2('1 核数与时间比的方向')
    p('在计算总量固定、负载均衡且忽略其他开销的理想模型中，t(n)与1/n成正比。A、B分别使用a、b个核时，应有t_A/t_B≈b/a，或从A到B的加速比≈b/a。课件将“执行时间比”写作a/b，方向相反，本报告据定义修正。')
    code('t(5) / t(20) = 20 / 5 = 4  （理想值）\n并行效率 = [t(5) / t(20)] / 4',9)
    p('Matmul从Vector侧发起。混合执行时Host的blockDim表示Cube核组数，Tiling使用2×blockDim个逻辑子任务。本实验检查实际返回的usedCoreNum和分块值，不把40个Vector任务误写成40个独立Cube核。')
    h2('2 基本块计算访存比')
    p('按课件模型，以16×16×16为一个理想Cube计算单位：Cy=baseM×baseK×baseN/4096；D=baseM×baseK+baseK×baseN；P=Cy/D。这里D以fp16元素数计，输入字节数为2D；它不含输出写回，也不是实际测得的总访存量。')
    table(['基本块 M×N×K','Cy','D 元素','P'],[
        ('64×64×64','64','8192','0.0078125'),
        ('128×128×128','512','32768','0.015625'),
        ('128×256×64','512','24576','0.0208333'),
    ],[6.1,2.2,4.1,4.6])
    p('P模型预测128×256×64优于128×128×128，后者优于64×64×64。该排序仅考虑一次基本块的输入复用；它没有包含核间分配、尾块、流水调度和实际存储层次，须与测量结果对照。')
    h2('3 四种实测配置')
    table(['配置','Cube核','Vector任务','实际基本块 M×N×K','单任务 M×N'],[
        (c['label'],c['cores'],c['used_core_num'],base(c),'×'.join(map(str,c['single_core'][:2]))) for c in cases
    ],[2.4,2.0,2.5,5.4,4.7])
    p('5核和20核比较固定基本块为128×128×128；基本块比较固定20核。所有配置由同一构建产物执行，输入路径、类型和验证准则一致。实际singleCore及step/depth等tiling字段可能随配置变化，本实验比较完整配置，不能把全部差异只归因于P。')

    page('四 算子实现与性能采集')
    h2('1 Host配置与核函数')
    code('''tiler.SetDim(2 * blockDim);
tiler.SetAType(GM, ND, DT_FLOAT16);
tiler.SetBType(GM, ND, DT_FLOAT16);
tiler.SetCType(GM, ND, DT_FLOAT);
tiler.SetShape(4096, 5120, 4096);
tiler.SetOrgShape(4096, 5120, 4096);
tiler.SetFixSplit(baseM, baseN, -1);
tiler.GetTiling(tiling);''')
    p('以上为tiling流程简写，省略命名空间与检查，完整代码见main.cpp。请求基本块后还须核对返回值，最终对照表记录实际生效的baseM、baseN、baseK。')
    p(data['tiling_note'])
    code('''mm.SetTensorA(aGlobal); mm.SetTensorB(bGlobal);
mm.IterateAll(cGlobal); mm.End();''')
    p('核函数按singleCoreM、singleCoreN和逻辑编号计算A、B、C偏移，以REGIST_MATMUL_OBJ注册计算对象。CMake启用HAVE_WORKSPACE和HAVE_TILING，由生成的wrapper管理系统workspace。输出区预填无效值，遗漏写入会导致有限值检查失败。')
    h2('2 计时区间与统计方式')
    p(f'每配置先执行{warmup}次预热，再保留{samples}次稳态KernelLaunch样本；每次调用后同步同一流。输入生成、CPU基准、文件读写、H2D和D2H不计入aicore_time。msprof以PipeUtilization采集20条同名核任务，按开始时间排序后剔除前{warmup}次。')
    p('以稳态中位数比较性能，并报告均值、样本标准差、最小与最大值。图中散点保留全部样本。硬件流水可重叠，MAC与MTE2不能相加解释总时长。')
    h2('3 指标列名对应关系')
    table(['课件名称','本次msprof CSV列名','报告含义'],data['metric_mapping'],[4.5,6.3,6.2])
    p('本次CSV直接提供课件要求的三列，单位均为微秒。另保留Task Duration(us)任务持续时间作为补充记录；图表和加速比统一采用aicore_time(us)，不混用两种口径。')

    page('五 正确性验证与原始数据')
    h2('1 独立CPU基准')
    p('CPU端NumPy将已落盘的fp16输入转为float32再执行矩阵乘法，不调用Ascend或待测核函数。每配置最后一次NPU调用的20971520个float32输出与完整基准逐元素比较；没有将15次采样误计为15份已分别校验的输出。')
    p(f'为检查CPU基准自身，另用float64点积复核{meta["float64_spot_check_count"]}个坐标，包含四角、分块邻近位置和固定种子随机点，最大相对差为{meta["float64_spot_check_max_rel_error"]:.7g}。这是32点高精度抽查，而非全矩阵float64验证。')
    code('abs(actual - golden) <= 0.02 + 1e-4 * abs(golden)\n全部元素同时满足字节数、有限值、正值和误差准则才算通过。',9)
    p('误差阈值在设备测试前固定。由于4096项点积的浮点累加顺序可能不同，“通过”不等于与CPU逐位相同；报告同时披露实际误差，而非仅显示通过标记。')
    h2('2 四配置全量验证')
    table(['配置','检查元素数','失败数','最大绝对误差','最大相对误差'],[
        (c['label'],'20971520',c['verification']['failed_elements'],
         f"{c['verification']['max_abs_error']:.7g}",f"{c['verification']['max_rel_error']:.7g}") for c in cases
    ],[2.3,3.4,2.0,4.65,4.65])
    p(f'四配置共核对83886080个输出元素；最大绝对误差{maxerr:.7g}，最大相对误差{maxrel:.7g}。每组原始输出bin和校验JSON均保留，可脱离NPU重算校验。')
    h2('3 输出样例')
    chosen=cases[2]['verification']['sample_elements']
    table(['坐标 行 列','20核128³输出','CPU golden','绝对误差'],[
        (f"({s['row']}, {s['column']})",f"{s['actual']:.6f}",f"{s['golden']:.6f}",f"{s['abs_error']:.7g}")
        for s in [chosen[0],chosen[len(chosen)//2],chosen[-1]]
    ],[3.4,4.9,4.9,3.8])
    p('data/metadata.json记录输入形状、实际范围、种子、CPU基准方法和SHA256。性能对照均复用这组文件，验证结果保存于各配置目录。摘要与图表从原始CSV和JSON提取，未用屏显文字代替原始测量。')

    page('六 msprof性能结果')
    h2('1 稳态AI Core耗时')
    table(['配置','均值 μs','中位数 μs','标准差 μs','最小至最大 μs'],[
        (c['label'],val(np.mean(c['duration_us'])),val(med(c)),val(np.std(c['duration_us'],ddof=1)),
         f"{min(c['duration_us']):.3f}–{max(c['duration_us']):.3f}") for c in cases
    ],[2.5,3.2,3.2,3.2,4.9])
    p(f'A为5核128³，B/C/D为20核三种基本块，对应上一表。每行使用{samples}次稳态样本；标准差为样本标准差。所有数值单位均为微秒，不包含预热{warmup}次。原始任务逐行记录随工程保存，表内仅为显示而保留三位小数。')
    fig(draw_chart(cases),'图1  msprof真实AI Core耗时  柱为中位数  散点为单次样本  误差线为最小至最大')
    h2('2 核数比较')
    p(f'固定128×128×128，5核中位数为{med(cases[0]):.3f} μs，20核为{med(cases[2]):.3f} μs。实测加速比{core_speed:.3f}，相对于理想4倍的并行效率为{efficiency*100:.2f}%。这反映本矩阵规模与所测配置的总体缩放，不能推广为所有输入的固定加速比。')
    h2('3 基本块比较')
    p(f'固定20核，{base(best)}中位数最小，为{med(best):.3f} μs；相对64×64×64的加速比为{base_speed:.3f}。三个配置的完整分布见图1右图；不能仅从一次最快值判断优劣。')
    p('本实验属于同一会话内的重复执行热态测量，采样数量有限，且性能采集本身可能影响运行。结论限于所测设备、输入和配置，不代表冷启动、跨会话或未启用profiler的应用端到端延迟。')

    page('七 实验分析与假设修正')
    h2('1 矩阵计算和搬运指标')
    table(['配置','AI Core中位数 μs','MAC中位数 μs','MTE2中位数 μs'],[
        (c['label'],val(med(c)),val(np.median(c['mac_us'])),val(np.median(c['mte2_us']))) for c in cases
    ],[2.6,4.8,4.8,4.8])
    p('MAC列表示Cube矩阵计算相关活动，MTE2列表示相应搬入活动。各流水可以重叠，活动时长也可能按核统计；本表只比较同一工具、相同采集选项下的同名指标，不用分项相减推断“其他开销”。')
    h2('2 对线性加速假设的检验')
    p(f'理想t(5)/t(20)=4，实际为{core_speed:.3f}。核数变化同时改变单核负责的矩阵范围，只有实际观察到接近4倍时才能在本条件下认为线性模型近似成立；偏离理想值本身不能区分调度、访存、流水或频率因素。')
    p(data['analysis']['core'])
    h2('3 对计算访存比假设的检验')
    p(data['analysis']['base'])
    p('Cy和D是基本块的理论计数，P只刻画模型中的输入复用。若实际排序与P排序不符，应将结论修正为：P是候选分块的筛选指标，实际优劣仍须在固定输入上验证，并同时检查合法tiling、并行度和测得的执行活动。')
    h2('4 结论边界与后续实验')
    p('本次没有采集全部Cache命中率、内存带宽或核间逐核负载，因此不将“Cache命中提高”“带宽饱和”“频率变化”写成已证实原因。下一步可在固定基本块下增加10核、15核，交替重复配置顺序，再补充带宽、Cache及每核活动指标，以区分可能原因。')
    p('验证范围仅为指定M、N、K及本次fp16正值输入，未覆盖任意维度、转置、bias或其他数据类型。正确性结论来自全量float32基准比较，高精度float64检查仅覆盖32个坐标。')

    page('八 运行屏显与复现资料')
    shot=LAB/'evidence/live_console_large.png'
    assert shot.exists(),'Waiting for actual week14 browser console screenshot'
    im=Image.open(shot)
    assert im.width>=900 and im.height>=744
    crop=TMP/'console_report_crop.png'
    im.crop((20,200,880,744)).save(crop)
    fig(crop,'图2  经SSH读取真实CSV的汇总屏显  仅裁去网页标题及外部空白',17)
    p('在工程目录labs/week14_matmul/中执行下列命令可构建、生成同一组输入、采集并校验四种配置。完整源码、原始输入输出、CSV、tiling与校验JSON随提交包保留。')
    code('bash build.sh\npython3 scripts/gen_data.py --output-dir data\nbash run_all.sh\npython3 summarize.py',8.5)
    p('参考资料：\n［1］《矩阵运算性能探究》，第1至6页，华为人才在线课程文档。\n［2］《Ascend C异构并行程序设计源代码》，3.4矩阵编程范式。\n［3］本次ModelArts原始构建、运行、msprof CSV与验证JSON。')

    linkp=p('［4］昇腾官方 CANN 8.5.0 SetFixSplit接口说明。')
    rid=D.part.relate_to('https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0681.html','http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink',is_external=True)
    hl=OxmlElement('w:hyperlink');hl.set(qn('r:id'),rid);rr=OxmlElement('w:r');tt=OxmlElement('w:t');tt.text=' 官方文档';rr.append(tt);hl.append(rr);linkp._p.append(hl)
    D.core_properties.title='矩阵运算性能探究 第14周实验报告'
    D.core_properties.author='许盛凯'
    D.core_properties.subject='Ascend C Matmul 核数与基本块性能对照'
    target=OUT/(NAME+'.docx');D.save(target)
    with zipfile.ZipFile(REF) as z:saved={n:z.read(n) for n in PRESERVE}
    with zipfile.ZipFile(target) as z:parts={n:z.read(n) for n in z.namelist()}
    parts.update(saved)
    # The final report contains no calendar dates, including core metadata.
    from lxml import etree
    core = etree.fromstring(parts['docProps/core.xml'])
    for tag in ('{http://purl.org/dc/terms/}created', '{http://purl.org/dc/terms/}modified',
                '{http://schemas.openxmlformats.org/package/2006/metadata/core-properties}lastPrinted'):
        for node in list(core.findall(tag)):
            core.remove(node)
    parts['docProps/core.xml'] = etree.tostring(core, xml_declaration=True, encoding='UTF-8', standalone=True)
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for n,v in parts.items():z.writestr(n,v)
    assert hashlib.sha256(REF.read_bytes()).hexdigest()==REF_HASH
    print(target)
    return target

if __name__=='__main__':
    build(prepare_report_data())
