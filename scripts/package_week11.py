from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json

root = Path(__file__).resolve().parents[1]
out = root / 'output/week11'
lab = root / 'labs/week11_addrelu'
name = '2024141460445-许盛凯-第11讲（11月12日）-向量编程基础算子-实验报告'
report = out / (name + '.docx')
verification = json.loads((lab/'Lab3_AddRelu/results/verification_all.json').read_text())
assert verification['passed_cases'] == verification['total_cases'] == 10
assert (lab/'evidence/07_actual_terminal.png').is_file()
files = {report.name: report}
for path in lab.rglob('*'):
    if not path.is_file():
        continue
    rel = path.relative_to(lab)
    if any(p == '__pycache__' or p.startswith('build_') for p in rel.parts):
        continue
    if path.suffix in {'.pyc', '.jpg'}:
        continue
    files['AddRelu实验工程/' + str(rel)] = path
assert not any(n.endswith(('.pem', '.key')) for n in files)
manifest = '\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+n
                     for n, p in sorted(files.items())) + '\n'
archive = out / (name + '.zip')
with ZipFile(archive, 'w', ZIP_DEFLATED) as z:
    for n, p in sorted(files.items()):
        z.write(p, n)
    z.writestr('SHA256SUMS.txt', manifest)
with ZipFile(archive) as z:
    assert z.testzip() is None
    for n, p in files.items():
        assert z.read(n) == p.read_bytes(), n
    assert report.name in z.namelist()
    assert 'AddRelu实验工程/Lab3_AddRelu/addrelu_custom.cpp' in z.namelist()
print('Archive:', archive)
print('Files:', len(files)+1, 'Bytes:', archive.stat().st_size)
print('Archive and all file hashes verified: PASS')
