#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
WEEK13_ROOT="$(cd .. && pwd)"
export ASCEND_CUSTOM_OPP_PATH="$WEEK13_ROOT/custom_opp/vendors/lab_week13${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export WEEK13_OP_API="$WEEK13_ROOT/custom_opp/vendors/lab_week13/op_api"
export LD_LIBRARY_PATH="$WEEK13_OP_API/lib:$LD_LIBRARY_PATH"
python3 build_extension.py
python3 run_network.py
