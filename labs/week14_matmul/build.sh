#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
cmake -E rm -rf build_npu
cmake -S . -B build_npu -DSOC_VERSION=Ascend910B4 -DRUN_MODE=npu
cmake --build build_npu -j4
