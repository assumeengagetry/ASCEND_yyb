from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import hashlib,json
R=Path(__file__).resolve().parents[1]
name='2024141460445-许盛凯-第14讲-矩阵运算性能探究-实验报告'
report=R/'output/week14'/(name+'.pdf')
assert report.is_file()
lab=R/'labs/week14_matmul'
checks=list((lab/'results').glob('*/verification.json'))
assert len(checks)==4 and all(json.loads(p.read_text())['passed'] for p in checks)
files={report.name:report}
for p in lab.rglob('*'):
 if not p.is_file():continue
 rel=p.relative_to(lab)
 if any(x.startswith('build_') or x in ('build','__pycache__','CMakeFiles') for x in rel.parts):continue
 if p.name=='.DS_Store' or p.suffix in ('.pyc','.o','.so','.a'):continue
 assert p.suffix not in ('.pem','.key','.docx','.doc') and p.name not in ('ssh_config','known_hosts')
 files['Matmul实验工程/'+rel.as_posix()]=p
manifest='\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+n for n,p in sorted(files.items()))+'\n'
out=report.with_suffix('.zip')
with ZipFile(out,'w',ZIP_DEFLATED,compresslevel=6) as z:
 for n,p in sorted(files.items()):z.write(p,n)
 z.writestr('SHA256SUMS.txt',manifest)
with ZipFile(out) as z:
 assert z.testzip() is None
 for n,p in files.items():
  assert hashlib.sha256(z.read(n)).hexdigest()==hashlib.sha256(p.read_bytes()).hexdigest(),n
print(f'{out}\n{len(files)+1} files; {out.stat().st_size} bytes; verified')
