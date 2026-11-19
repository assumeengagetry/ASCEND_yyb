from pathlib import Path
import numpy as np
import json
root=Path(__file__).resolve().parent
cases=[]
comparisons=[]
for project in ('Lab4_Sinh','Lab4_SinhTBuf'):
 for mode in ('cpu','npu'):
  for case in ('random','boundary','ramp','ones','tens'):
   d=root/project/'results'/f'{mode}_{case}'
   r=json.loads((d/'output/verification.json').read_text())
   assert r['passed'],d
   r.update(project=project,mode=mode,case=case)
   cases.append(r)
for mode in ('cpu','npu'):
 for case in ('random','boundary','ramp','ones','tens'):
  a=root/'Lab4_Sinh'/'results'/f'{mode}_{case}'
  b=root/'Lab4_SinhTBuf'/'results'/f'{mode}_{case}'
  assert (a/'input/input_x.bin').read_bytes()==(b/'input/input_x.bin').read_bytes()
  x=np.fromfile(a/'output/output_z.bin',dtype='<f4')
  y=np.fromfile(b/'output/output_z.bin',dtype='<f4')
  same=np.array_equal(x,y)
  comparisons.append(dict(mode=mode,case=case,bitwise_equal=bool(same),max_abs_difference=float(np.max(np.abs(x.astype(float)-y)))))
  assert same,(mode,case)
summary=dict(total_cases=len(cases),passed_cases=sum(c['passed'] for c in cases),total_elements=sum(c['numel'] for c in cases),max_abs_error=max(c['max_abs_error'] for c in cases),max_rel_error=max(c['max_rel_error'] for c in cases),cases=cases,version_comparisons=comparisons)
(root/'verification_all.json').write_text(json.dumps(summary,indent=2)+'\n')
print(f"SUMMARY {summary['passed_cases']}/{summary['total_cases']} PASS elements={summary['total_elements']}")
print(f"max_abs_error={summary['max_abs_error']:.9g} max_rel_error={summary['max_rel_error']:.9g}")
print('Sinh vs SinhTBuf: 10/10 comparisons bitwise equal')
