#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
mkdir -p results profiles evidence
npu-smi info > evidence/01_environment.log
for cfg in '5 128 128 128 cores5_base128' '20 64 64 64 cores20_base64' '20 128 128 128 cores20_base128' '20 128 256 64 cores20_base128x256'; do
 read -r cores bm bn bk name <<< "$cfg"
 mkdir -p "results/$name" "profiles/$name"
 printf '\n[RUN] %s\n' "$name"
 msprof --application="$PWD/run_one.sh $cores $bm $bn $bk 15 $PWD/results/$name/output.bin" --output="$PWD/profiles/$name" --aic-metrics=PipeUtilization --ai-core=on --task-time=on --ascendcl=on 2>&1 | tee "results/$name/profile.log"
 python3 scripts/verify_result.py "results/$name/output.bin" data/golden.bin --json "results/$name/verification.json" | tee "results/$name/verify.log"
done
python3 summarize.py
