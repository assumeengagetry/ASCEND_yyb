#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ../env.sh
OP_LIB="$LAB13_ROOT/custom_opp/vendors/lab_week13/op_api"
g++ -std=c++17 -O2 main.cpp -I "$ASCEND_HOME_PATH/include" -I "$OP_LIB/include" -L "$ASCEND_HOME_PATH/lib64" -L "$OP_LIB/lib" -Wl,-rpath,"$OP_LIB/lib" -lcust_opapi -lascendcl -lnnopbase -o main
echo ACLNN_HOST_BUILD_SUCCESS
