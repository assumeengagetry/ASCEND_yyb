from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import hashlib,json
R=Path(__file__).resolve().parents[1]
name='2024141460445-许盛凯-第12讲-向量编程进阶算子-实验报告'
report=R/'output/week12'/(name+'.pdf')
assert report.is_file()
summary=json.loads((R/'labs/week12_sinh/verification_all.json').read_text())
assert summary['total_cases']==summary['passed_cases']==20
assert all(x['bitwise_equal'] for x in summary['version_comparisons'])
files={report.name:report}
for p in (R/'labs/week12_sinh').rglob('*'):
 if not p.is_file():continue
 rel=p.relative_to(R/'labs/week12_sinh')
 if any(x.startswith('build_') or x in ('__pycache__','npuchk','log','sim_log') for x in rel.parts):continue
 if p.name in ('stub_reg.log','.DS_Store') or p.suffix in ('.pyc','.o','.so','.a'):continue
 assert p.suffix not in ('.pem','.key','.docx','.doc') and p.name not in ('ssh_config','known_hosts')
 files['Sinh实验工程/'+rel.as_posix()]=p
assert 'Sinh实验工程/Lab4_Sinh/sinh_custom.cpp' in files
assert 'Sinh实验工程/Lab4_SinhTBuf/sinh_custom.cpp' in files
manifest='\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+n for n,p in sorted(files.items()))+'\n'
out=report.with_suffix('.zip')
with ZipFile(out,'w',ZIP_DEFLATED) as z:
 for n,p in sorted(files.items()):z.write(p,n)
 z.writestr('SHA256SUMS.txt',manifest)
with ZipFile(out) as z:
 assert z.testzip() is None
 for n,p in files.items(): assert z.read(n)==p.read_bytes(),n
print(f'{out}\n{len(files)+1} files; {out.stat().st_size} bytes; all hashes verified')
