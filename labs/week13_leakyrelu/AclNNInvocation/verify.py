from pathlib import Path
import numpy as np
import subprocess,json,hashlib
R=Path(__file__).resolve().parent
N=8*200*1024
rng=np.random.default_rng(20261119)
base=rng.uniform(-10,10,N).astype(np.float32)
# Endpoints, signs and short-range values at tile and core-relevant boundaries.
base[:8]=[-10,-1,-1e-5,-0.0,0.0,1e-5,1,10]
base[2047:2050]=[-10,0,10]
base[-3:]=[-10,0,10]
cases=[('random_a01',base,0.1),('random_a15',base,1.5),('negative_a02',-np.abs(base),0.2),('positive_a01',np.abs(base),0.1),('tail_257_a01',base[:257].copy(),0.1)]
summary=[]
for name,x,a in cases:
 d=R/'results'/name;d.mkdir(parents=True,exist_ok=True)
 x.tofile(d/'input.bin')
 shape='8,200,1024' if len(x)==N else str(len(x))
 golden=np.where(x.astype(np.float64)>=0,x.astype(np.float64),x.astype(np.float64)*float(np.float32(a))).astype(np.float32)
 golden.tofile(d/'golden.bin')
 cmd=[str(R/'main'),'float32',shape,str(d/'input.bin'),str(d/'output.bin'),str(a)]
 p=subprocess.run(cmd,capture_output=True,text=True)
 (d/'run.log').write_text('$ '+' '.join(cmd)+'\n'+p.stdout+p.stderr)
 if p.returncode: raise RuntimeError(p.stdout+p.stderr)
 actual=np.fromfile(d/'output.bin',dtype=np.float32)
 assert actual.size==x.size and np.isfinite(actual).all()
 err=np.abs(actual.astype(float)-golden.astype(float))
 ok=err<=1e-6+1e-6*np.abs(golden.astype(float))
 s=dict(case=name,shape=[8,200,1024] if len(x)==N else [len(x)],elements=int(x.size),negative_slope=a,passed=bool(ok.all()),failed_elements=int((~ok).sum()),max_abs_error=float(err.max()),max_rel_error_nonzero=float(np.max(err/np.maximum(np.abs(golden),1e-30))),negative_count=int((x<0).sum()),zero_count=int((x==0).sum()),positive_count=int((x>0).sum()),samples=[dict(index=i,x=float(x[i]),reference=float(golden[i]),actual=float(actual[i])) for i in range(8)],input_sha256=hashlib.sha256((d/'input.bin').read_bytes()).hexdigest())
 (d/'verification.json').write_text(json.dumps(s,indent=2)+'\n')
 summary.append(s)
 print(f"PASS {name} elements={x.size} slope={a} failed={s['failed_elements']} max_abs_error={s['max_abs_error']}",flush=True)
 assert s['passed']
result=dict(total_cases=len(summary),passed_cases=sum(c['passed'] for c in summary),cases=summary)
(R/'results/verification_all.json').write_text(json.dumps(result,indent=2)+'\n')
print('ACLNN SUMMARY 5/5 PASS')
