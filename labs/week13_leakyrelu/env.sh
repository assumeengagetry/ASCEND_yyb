#!/usr/bin/env bash
LAB13_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export ASCEND_CUSTOM_OPP_PATH="$LAB13_ROOT/custom_opp/vendors/lab_week13${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$LAB13_ROOT/custom_opp/vendors/lab_week13/op_api/lib:$LD_LIBRARY_PATH"
export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp
