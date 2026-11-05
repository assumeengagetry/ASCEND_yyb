#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export ASCEND_LAB_ROOT="${ASCEND_LAB_ROOT:-$(cd .. && pwd)}"
if [ -f "$ASCEND_LAB_ROOT/env.sh" ]; then source "$ASCEND_LAB_ROOT/env.sh"; fi
export SETUPTOOLS_USE_DISTUTILS=stdlib
TOOLKIT="${ASCEND_HOME_PATH:-$HOME/Ascend/ascend-toolkit/latest}"
if [ ! -f "$TOOLKIT/set_env.sh" ]; then
    echo "CANN toolkit not found: set ASCEND_HOME_PATH" >&2
    exit 1
fi
source "$TOOLKIT/set_env.sh"
cmake -S . -B build_out \
  -DCMAKE_BUILD_TYPE=Release -DENABLE_SOURCE_PACKAGE=True \
  -DENABLE_BINARY_PACKAGE=True -DASCEND_COMPUTE_UNIT=ascend910b \
  -DENABLE_TEST=False -Dvendor_name=lab_sigmoid \
  -DASCEND_CANN_PACKAGE_PATH="$TOOLKIT" \
  -DASCEND_PYTHON_EXECUTABLE="$(command -v python3)" \
  -DCMAKE_INSTALL_PREFIX="$PWD/build_out" -DENABLE_CROSS_COMPILE=False
cmake --build build_out --target package -j 4
bash build_out/custom_opp_euleros_aarch64.run --quiet \
  --install-path="$ASCEND_LAB_ROOT/install"
cmake --build build_out --target binary -j 4
cmake --build build_out --target package -j 4
bash build_out/custom_opp_euleros_aarch64.run --quiet \
  --install-path="$ASCEND_LAB_ROOT/install"
test -s "$ASCEND_LAB_ROOT/install/vendors/lab_sigmoid/op_impl/ai_core/tbe/kernel/config/ascend910b/binary_info_config.json"
echo "BUILD_AND_INSTALL_SUCCESS"
