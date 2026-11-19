#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
printf 'WEEK 12 | Sinh / SinhTBuf | Ascend 910B4 | CANN 8.5.2\n'
date -Is
for project in Lab4_Sinh Lab4_SinhTBuf; do
 (
 cd "$project"
 printf '\n$ %s: generate random -> build_npu/main -> verify\n' "$project"
 python3 scripts/gen_data.py --case random
 ./build_npu/main
 python3 scripts/verify_result.py output/output_z.bin output/golden.bin
 )
done
