#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
LAB_ROOT="${LAB_ROOT:-$(cd ../.. && pwd)}"
if [ -f "$LAB_ROOT/env.sh" ]; then
    source "$LAB_ROOT/env.sh"
fi
ASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-$HOME/Ascend/ascend-toolkit/latest}}"
OP_API_DIR="${OP_API_DIR:-$LAB_ROOT/install/vendors/lab_sigmoid/op_api}"
export LD_LIBRARY_PATH="$OP_API_DIR/lib:$ASCEND_CANN_PACKAGE_PATH/lib64:${LD_LIBRARY_PATH:-}"
cmake -S . -B build -DASCEND_CANN_PACKAGE_PATH="$ASCEND_CANN_PACKAGE_PATH" -DOP_API_DIR="$OP_API_DIR"
cmake --build build -j 4
python3 verify.py --executable ./build/execute_sigmoid "$@"
