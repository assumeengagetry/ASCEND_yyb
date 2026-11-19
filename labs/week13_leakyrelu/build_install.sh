#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export SETUPTOOLS_USE_DISTUTILS=stdlib
root="$PWD"
cd LeakyReluCustom
cmake -S . -B build_out -DCMAKE_BUILD_TYPE=Release -DENABLE_SOURCE_PACKAGE=True -DENABLE_BINARY_PACKAGE=False -DASCEND_COMPUTE_UNIT=ascend910b -DENABLE_TEST=False -Dvendor_name=lab_week13 -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest -DASCEND_PYTHON_EXECUTABLE="$(command -v python3)" -DCMAKE_INSTALL_PREFIX="$PWD/build_out" -DENABLE_CROSS_COMPILE=False
cmake --build build_out --target package -j 4
bash build_out/custom_opp_*_aarch64.run --quiet --install-path="$root/custom_opp"
cmake -S . -B build_out -DENABLE_BINARY_PACKAGE=True
cmake --build build_out --target binary -j 4
cmake --build build_out --target package -j 4
bash build_out/custom_opp_*_aarch64.run --quiet --install-path="$root/custom_opp"
test -s "$root/custom_opp/vendors/lab_week13/op_impl/ai_core/tbe/kernel/config/ascend910b/binary_info_config.json"
echo BUILD_INSTALL_BINARY_SUCCESS
