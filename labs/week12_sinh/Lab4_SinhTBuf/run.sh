#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
MODE=npu
SOC=Ascend910B4
CASE=random
while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--run-mode) MODE="${2:?mode required}"; shift 2 ;;
        -v|--soc-version) SOC="${2:?SoC required}"; shift 2 ;;
        --case) CASE="${2:?case required}"; shift 2 ;;
        *) echo 'Usage: bash run.sh -r cpu|sim|npu -v Ascend910B4 [--case random]'; exit 2 ;;
    esac
done
case "$MODE" in cpu|sim|npu) ;; *) exit 2 ;; esac
CANN_PATH="${ASCEND_INSTALL_PATH:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}"
source "$CANN_PATH/bin/setenv.bash"
printf '[CONFIG] variant=SinhTBuf mode=%s soc=%s dtype=float32 shape=8x2048 case=%s\n' "$MODE" "$SOC" "$CASE"
# CANN nested build steps transform objects in place; start device modes from a fresh build.
if [[ "$MODE" != cpu ]]; then
    cmake -E rm -rf "build_$MODE"
fi
cmake -S . -B "build_$MODE" -DRUN_MODE="$MODE" -DSOC_VERSION="$SOC" -DASCEND_CANN_PACKAGE_PATH="$CANN_PATH"
if [[ "$MODE" == cpu ]]; then
    cmake --build "build_$MODE" --target main -j 4
else
    # CANN merge steps transform intermediate objects in place; rebuild device modes cleanly.
    cmake --build "build_$MODE" --target main --clean-first -j 4
fi
python3 scripts/gen_data.py --case "$CASE"
if [[ "$MODE" == sim ]]; then
    export LD_LIBRARY_PATH="$CANN_PATH/tools/simulator/$SOC/lib:$CANN_PATH/runtime/lib64/stub:$LD_LIBRARY_PATH"
    export CAMODEL_LOG_PATH="$PWD/sim_log"
    mkdir -p "$CAMODEL_LOG_PATH"
fi
rm -f output/output_z.bin
"./build_$MODE/main"
python3 scripts/verify_result.py output/output_z.bin output/golden.bin
