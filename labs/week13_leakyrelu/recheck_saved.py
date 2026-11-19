"""Recheck saved real outputs without an Ascend device (NumPy only)."""
from pathlib import Path
import numpy as np
import json,hashlib
R=Path(__file__).resolve().parent
s=json.loads((R/'AclNNInvocation/results/verification_all.json').read_text())
for c in s['cases']:
 d=R/'AclNNInvocation/results'/c['case']
 x=np.fromfile(d/'input.bin',np.float32);y=np.fromfile(d/'output.bin',np.float32);g=np.fromfile(d/'golden.bin',np.float32)
 assert x.size==y.size==g.size==c['elements']
 assert all(np.isfinite(z).all() for z in (x,y,g))
 assert hashlib.sha256((d/'input.bin').read_bytes()).hexdigest()==c['input_sha256']
 expected=np.where(x.astype(float)>=0,x.astype(float),x.astype(float)*float(np.float32(c['negative_slope']))).astype(np.float32)
 assert np.array_equal(g,expected)
 assert np.all(np.abs(y.astype(float)-expected)<=1e-6+1e-6*np.abs(expected))
 print('RECHECK_PASS',c['case'],x.size)
d=R/'TorchInvocation/results'
s=json.loads((d/'verification.json').read_text())
x=np.load(d/'activation_input.npy');a=np.load(d/'activation_custom.npy')
g=np.where(x.astype(float)>=0,x.astype(float),x.astype(float)*float(np.float32(s['negative_slope']))).astype(np.float32)
assert np.array_equal(a,g)
assert np.array_equal(a,np.load(d/'activation_builtin.npy'))
assert np.array_equal(np.load(d/'original_output.npy'),np.load(d/'replaced_output.npy'))
assert s['custom_kernel_calls']==2 and s['passed']
print('RECHECK_PASS activation and complete network outputs')
