#!/bin/bash
export ASCEND_LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /home/ma-user/Ascend/ascend-toolkit/set_env.sh
source "$ASCEND_LAB_ROOT/venv/bin/activate"
export SETUPTOOLS_USE_DISTUTILS=stdlib
export CURL_CA_BUNDLE="$(python -m certifi)"
export ASCEND_CUSTOM_OPP_PATH="$ASCEND_LAB_ROOT/install/vendors/lab_sigmoid"
export LD_LIBRARY_PATH="$ASCEND_LAB_ROOT/install/vendors/lab_sigmoid/op_api/lib:${LD_LIBRARY_PATH}"
