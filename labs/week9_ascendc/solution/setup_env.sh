#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python3 -m venv --system-site-packages venv
venv/bin/python -m pip install -r requirements.txt
echo "Run: source env.sh"
