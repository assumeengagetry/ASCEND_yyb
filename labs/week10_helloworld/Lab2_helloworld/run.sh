#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

SOC_VERSION=Ascend910B4
while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--soc-version) SOC_VERSION="${2:?Missing SoC version}"; shift 2 ;;
        *) echo "Usage: bash run.sh [-v Ascend910B4]" >&2; exit 2 ;;
    esac
done

if [[ -n "${ASCEND_INSTALL_PATH:-}" ]]; then
    CANN_PATH="$ASCEND_INSTALL_PATH"
elif [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    CANN_PATH="$ASCEND_HOME_PATH"
elif [[ -d /usr/local/Ascend/ascend-toolkit/latest ]]; then
    CANN_PATH=/usr/local/Ascend/ascend-toolkit/latest
else
    CANN_PATH="$HOME/Ascend/ascend-toolkit/latest"
fi
source "$CANN_PATH/bin/setenv.bash"
printf '[CONFIG] SOC_VERSION=%s\n[CONFIG] CANN_PATH=%s\n' "$SOC_VERSION" "$CANN_PATH"
cmake -S . -B build -DSOC_VERSION="$SOC_VERSION" \
    -DASCEND_CANN_PACKAGE_PATH="$CANN_PATH"
cmake --build build -j 4
cmake --install build

./build/main 2>&1 | tee output_msg.txt
count=$(grep -cF 'Hello World!!!' output_msg.txt || true)
if [[ "$count" -ne 8 ]]; then
    echo "[FAIL] Expected 8 Hello World lines, found $count" >&2
    exit 1
fi
echo "[PASS] Hello World lines: $count / 8"
