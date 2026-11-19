#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
for project in Lab4_Sinh Lab4_SinhTBuf; do
 (
 cd "$project"
 for mode in cpu npu; do
  for case_name in random boundary ramp ones tens; do
   result_dir="results/${mode}_${case_name}"
   mkdir -p "$result_dir"
   python3 scripts/gen_data.py --case "$case_name" > "$result_dir/generate.log"
   rm -f output/output_z.bin output/verification.json
   "./build_$mode/main" > "$result_dir/run.log" 2>&1
   python3 scripts/verify_result.py output/output_z.bin output/golden.bin > "$result_dir/verify.log" 2>&1
   cp -R input output "$result_dir/"
   cp case_metadata.json "$result_dir/"
   printf 'PASS project=%s mode=%s case=%s elements=16384\n' "$project" "$mode" "$case_name"
  done
 done
 )
done
python3 summarize.py
