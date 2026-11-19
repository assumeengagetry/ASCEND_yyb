#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
TARGET="${1:-GeneratedLeakyReluCustom}"
if [[ -e "$TARGET" ]]; then echo "Target already exists: $TARGET" >&2; exit 2; fi
msopgen gen -i LeakyReluCustom.json -c ai_core-Ascend910B4 -lan cpp -out "$TARGET"
cp LeakyReluCustom/op_host/leaky_relu_custom.cpp "$TARGET/op_host/"
cp LeakyReluCustom/op_host/leaky_relu_custom_tiling.h "$TARGET/op_host/"
cp LeakyReluCustom/op_kernel/leaky_relu_custom.cpp "$TARGET/op_kernel/"
cp LeakyReluCustom/CMakePresets.json "$TARGET/"
echo "GENERATED_AND_COMPLETED $TARGET"
