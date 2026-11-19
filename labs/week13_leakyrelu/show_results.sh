#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python3 - <<'PY'
from pathlib import Path
import json
print('WEEK 13 | LeakyReluCustom | Ascend 910B4 | CANN 8.5.2')
print('Course: 2026-11-26 | Executed: 2026-11-19')
print('Archived real NPU results (this screen reads saved verification files)')
print('API: aclnnLeakyReluCustomGetWorkspaceSize -> aclnnLeakyReluCustom')
a=json.loads(Path('AclNNInvocation/results/verification_all.json').read_text())
for c in a['cases']:
 print(f"{c['case']:<20} elements={c['elements']:<7} slope={c['negative_slope']:<4} failed={c['failed_elements']} max_abs={c['max_abs_error']}")
print(f"ACLNN SUMMARY: {a['passed_cases']}/{a['total_cases']} PASS")
p=Path('TorchInvocation/results/verification.json')
n=json.loads(p.read_text())
print('\nNETWORK: '+n['network'])
print('REPLACE: '+n['replacement'])
print('PLUGIN: '+n['custom_api'])
print('DEVICE: '+n['device']+' | torch='+n['torch']+' | torch_npu='+n['torch_npu'])
print('INPUT: '+str(n['input_shape'])+' | OUTPUT: '+str(n['output_shape']))
print('Custom kernel calls: '+str(n['custom_kernel_calls']))
for k in ('direct_activation_vs_math','direct_activation_vs_builtin_npu','network_original_vs_replaced'):
 c=n[k];print(f"{k}: elements={c['numel']} failed={c['failed_elements']} max_abs={c['max_abs_error']}")
print('NETWORK_REPLACEMENT_PASS' if n['passed'] else 'NETWORK_REPLACEMENT_FAIL')
assert a['passed_cases']==a['total_cases']==5 and n['passed']
PY
