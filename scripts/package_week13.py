from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import hashlib,json
R=Path(__file__).resolve().parents[1]
name='2024141460445-许盛凯-第13讲-算子工程化开发及整网调用-实验报告'
report=R/'output/week13'/(name+'.pdf')
assert report.is_file()
lab=R/'labs/week13_leakyrelu'
a=json.loads((lab/'AclNNInvocation/results/verification_all.json').read_text())
n=json.loads((lab/'TorchInvocation/results/verification.json').read_text())
assert a['passed_cases']==a['total_cases']==5 and n['passed']
files={report.name:report}
for p in lab.rglob('*'):
 if not p.is_file():continue
 rel=p.relative_to(lab)
 if any(x.startswith('build_') or x in ('build','__pycache__','install','CMakeFiles') for x in rel.parts):continue
 if p.name=='.DS_Store' or p.suffix in ('.pyc','.o','.so','.a'):continue
 assert p.suffix not in ('.pem','.key','.docx','.doc') and p.name not in ('ssh_config','known_hosts')
 files['LeakyRelu实验工程/'+rel.as_posix()]=p
manifest='\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+n for n,p in sorted(files.items()))+'\n'
out=report.with_suffix('.zip')
with ZipFile(out,'w',ZIP_DEFLATED) as z:
 for n,p in sorted(files.items()):z.write(p,n)
 z.writestr('SHA256SUMS.txt',manifest)
with ZipFile(out) as z:
 assert z.testzip() is None
 for n,p in files.items():assert z.read(n)==p.read_bytes(),n
print(f'{out}\n{len(files)+1} files; {out.stat().st_size} bytes; verified')
