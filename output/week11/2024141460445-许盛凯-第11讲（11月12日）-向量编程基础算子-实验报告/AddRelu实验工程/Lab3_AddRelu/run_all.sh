#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
CANN_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
source "$CANN_PATH/bin/setenv.bash"
mkdir -p results
for mode in cpu npu; do
    for case_name in random boundary all_negative all_positive cancellation; do
        result_dir="results/${mode}_${case_name}"
        mkdir -p "$result_dir"
        python3 scripts/gen_data.py --case "$case_name" > "$result_dir/generate.log"
        "./build_$mode/main" > "$result_dir/run.log" 2>&1
        python3 scripts/verify_result.py output/output_z.bin output/golden.bin > "$result_dir/verify.log" 2>&1
        cp -R input output "$result_dir/"
        cp case_metadata.json "$result_dir/"
        echo "PASS mode=$mode case=$case_name elements=16384"
    done
done
python3 - <<'PY'
from pathlib import Path
import json
cases=[]
for p in sorted(Path('results').glob('*/output/verification.json')):
    r=json.loads(p.read_text());r['case']=p.parents[1].name;cases.append(r)
assert len(cases)==10 and all(c['passed'] for c in cases)
summary={'total_cases':len(cases),'passed_cases':sum(c['passed'] for c in cases),'cases':cases}
Path('results/verification_all.json').write_text(json.dumps(summary,indent=2)+'\n')
print('SUMMARY 10/10 PASS max_abs_error='+str(max(c['max_abs_error'] for c in cases)))
PY
