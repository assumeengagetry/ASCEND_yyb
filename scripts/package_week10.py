from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import hashlib,json
root=Path(__file__).resolve().parents[1]
out=root/'output/week10'
lab=root/'labs/week10_helloworld'
out.mkdir(parents=True,exist_ok=True)
name='2024141460445-许盛凯-第10讲（11月5日）-HelloWorld算子-实验报告'
report=out/(name+'.docx')
files={report.name:report}
for p in lab.rglob('*'):
 if p.is_file() and '__pycache__' not in p.parts:
  files['HelloWorld实验工程/'+str(p.relative_to(lab))]=p
assert not any(n.endswith(('.pem','.key')) for n in files)
manifest='\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+n for n,p in sorted(files.items()))+'\n'
archive=out/(name+'.zip')
with ZipFile(archive,'w',ZIP_DEFLATED) as z:
 for n,p in sorted(files.items()):z.write(p,n)
 z.writestr('SHA256SUMS.txt',manifest)
with ZipFile(archive) as z:
 assert z.testzip() is None
 for n,p in files.items():assert z.read(n)==p.read_bytes(),n
 assert report.name in z.namelist()
 assert 'HelloWorld实验工程/Lab2_helloworld/hello_world.cpp' in z.namelist()
print('Archive:',archive)
print('Files:',len(files)+1,'Bytes:',archive.stat().st_size)
print('Archive and all file hashes verified: PASS')
