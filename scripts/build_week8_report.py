from pathlib import Path
import re
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT

root=Path(__file__).resolve().parents[1]
src=root/'labs/week8_atlas_dht11/云边协同实验报告.md'
doc=Document()
sec=doc.sections[0]
sec.page_width=Cm(21); sec.page_height=Cm(29.7)
sec.top_margin=Cm(1.9); sec.bottom_margin=Cm(1.9)
sec.left_margin=Cm(2); sec.right_margin=Cm(2)
for name in ['Normal','Title','Heading 1','Heading 2']:
 st=doc.styles[name]
 st.font.name='Times New Roman'
 st._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Songti SC')
 st.font.color.rgb=RGBColor(0,0,0)
st=doc.styles['Normal']; st.font.size=Pt(10.5)
st.paragraph_format.line_spacing=1.2
st.paragraph_format.space_after=Pt(5)
for name,size in [('Title',20),('Heading 1',14),('Heading 2',11.5)]:
 st=doc.styles[name]; st.font.size=Pt(size); st.font.bold=True
 st.paragraph_format.space_before=Pt(11)
 st.paragraph_format.space_after=Pt(5)
 st.paragraph_format.keep_with_next=True

def add_text(p,text):
 for piece in re.split(r'(`[^`]+`|\*\*[^*]+\*\*)',text):
  if piece.startswith('`') and piece.endswith('`'):
   r=p.add_run(piece[1:-1]); r.font.name='Courier New'; r.font.size=Pt(9)
  elif piece.startswith('**'):
   p.add_run(piece[2:-2]).bold=True
  else:p.add_run(piece)

def table(rows):
 rows=[[c.strip() for c in row.strip().strip('|').split('|')] for row in rows]
 rows=[r for r in rows if not all(re.fullmatch(r':?-+:?',c) for c in r)]
 t=doc.add_table(rows=1,cols=len(rows[0]));t.alignment=WD_TABLE_ALIGNMENT.CENTER
 t.style='Table Grid';t.autofit=False
 if len(rows[0])==6: widths=[1.2,3.2,2.5,2.8,3.2,4.1]
 elif len(rows[0])==3:widths=[3,5.6,8.4]
 else:widths=[4,13]
 for i,w in enumerate(widths):t.columns[i].width=Cm(w)
 for ri,row in enumerate(rows):
  cells=t.rows[0].cells if ri==0 else t.add_row().cells
  trPr=cells[0]._tc.getparent().get_or_add_trPr()
  trPr.append(OxmlElement('w:cantSplit'))
  if ri==0: trPr.append(OxmlElement('w:tblHeader'))
  for ci,text in enumerate(row):
   if len(rows[0])==6:text=text.replace('【待填写】','待填')
   cells[ci].width=Cm(widths[ci]); cells[ci].vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   p=cells[ci].paragraphs[0]; p.paragraph_format.space_after=Pt(3);p.paragraph_format.space_before=Pt(3);p.paragraph_format.line_spacing=1.1
   add_text(p,text)
   for r in p.runs:r.font.size=Pt(9.5); r.bold=ri==0
   tcPr=cells[ci]._tc.get_or_add_tcPr();m=OxmlElement('w:tcMar')
   for edge in ['top','left','bottom','right']:
    e=OxmlElement('w:'+edge);e.set(qn('w:w'),'75');e.set(qn('w:type'),'dxa');m.append(e)
   tcPr.append(m)
   if ri==0:
    shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'F2F2F2');tcPr.append(shade)
 doc.add_paragraph().paragraph_format.space_after=Pt(0)

lines=src.read_text().splitlines();i=0
while i<len(lines):
 line=lines[i].strip()
 if not line:i+=1;continue
 if line.startswith('|'):
  rows=[]
  while i<len(lines) and lines[i].strip().startswith('|'):
   rows.append(lines[i]);i+=1
  table(rows);continue
 if line.startswith('# '):
  p=doc.add_paragraph(style='Title');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
  add_text(p,'基于 Atlas 200 DK 与 DHT11 的\n云边协同温湿度采集实验')
 elif line.startswith('## '):
  p=doc.add_paragraph(style='Heading 1');add_text(p,line[3:])
 elif line.startswith('### '):
  p=doc.add_paragraph(style='Heading 2');add_text(p,line[4:])
 else:
  p=doc.add_paragraph();add_text(p,line)
  if line.startswith('【插入图'):
   p.paragraph_format.space_before=Pt(8);p.paragraph_format.space_after=Pt(10)
   for r in p.runs:r.italic=True;r.font.color.rgb=RGBColor(80,80,80)
 i+=1
for style in doc.styles:
 for border in list(style._element.xpath('.//w:pBdr')):
  border.getparent().remove(border)
for p in doc.paragraphs:
 for border in list(p._p.xpath('.//w:pBdr')):
  border.getparent().remove(border)
# A compact page number helps match screenshots and printed report pages.
f=sec.footer.paragraphs[0];f.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=f.add_run();r.font.size=Pt(9)
fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');r._r.addnext(fld)
doc.core_properties.title='基于Atlas 200 DK与DHT11的云边协同温湿度采集实验'
doc.core_properties.author=''
out=root/'output/week8/云边协同实验报告.docx'
out.parent.mkdir(parents=True,exist_ok=True)
doc.save(out)
print(out)
