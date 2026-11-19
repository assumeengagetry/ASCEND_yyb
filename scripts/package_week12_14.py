from pathlib import Path
from zipfile import ZipFile,ZIP_STORED,ZIP_DEFLATED
import hashlib
R=Path(__file__).resolve().parents[1]
OUT=R/'archive/combined/week12-14'
OUT.mkdir(parents=True,exist_ok=True)
name='2024141460445-许盛凯-第12至14周-实验报告与完整工程'
files={}
for week in (12,13,14):
 for suffix in ('.pdf','.zip'):
  found=list((R/f'output/week{week}').glob('*'+suffix))
  assert len(found)==1,(week,suffix,found)
  p=found[0];files[f'第{week}周/{p.name}']=p
readme='''# 第12至14周实验交付

许盛凯　2024141460445

每周目录均含PDF报告与完整提交ZIP。每个周次ZIP内含该周报告、源码、复现说明、真实输入输出、日志、验证JSON、屏显及SHA256清单；第14周还包含msprof原始CSV/DB和全部对比配置。

- 第12周：向量编程进阶算子，Sinh及SinhTBuf，20组CPU/NPU验证通过。
- 第13周：算子工程化开发及整网调用，LeakyReluCustom封装、ACLNN验证和PyTorch网络替换均通过。
- 第14周：矩阵运算性能探究，四种配置全矩阵验证通过，5次预热后统计15次msprof测量。

第14周保留共享输入、golden及四份真实矩阵输出，因而完整材料体积较大。单独阅读时直接打开各目录的PDF报告即可。
'''
manifest='\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+n for n,p in sorted(files.items()))+'\n'
out=OUT/(name+'.zip')
with ZipFile(out,'w',ZIP_STORED) as z:
 for n,p in sorted(files.items()):z.write(p,n)
 z.writestr('使用说明.md',readme,compress_type=ZIP_DEFLATED)
 z.writestr('SHA256SUMS.txt',manifest,compress_type=ZIP_DEFLATED)
with ZipFile(out) as z:
 assert z.testzip() is None
 for n,p in files.items():assert hashlib.sha256(z.read(n)).hexdigest()==hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'使用说明.md').write_text(readme)
print(f'{out}\n{out.stat().st_size} bytes; all 6 deliverable hashes verified')
