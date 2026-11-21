#!/usr/bin/env bash
# CPU package work only. Original SIF and authored task sources are never edited.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
chmod +x a40/training_python.sh
bash a40/hpc_environment.sh python-cpu -m pip install --no-deps --upgrade \
  --target /workspace/go2/.extra_overlay 'lxml==5.4.0'
if [[ ! -x .usd_converter/bin/python ]]; then
  bash a40/hpc_environment.sh python-cpu -m venv .usd_converter
fi
TASK_REQUIREMENTS=config/usd-converter.in
if [[ -s config/usd-converter.lock.txt ]]; then TASK_REQUIREMENTS=config/usd-converter.lock.txt; fi
bash a40/hpc_environment.sh python-cpu a40/usd_converter.py -m pip install -r "$TASK_REQUIREMENTS"
bash a40/hpc_environment.sh python-cpu a40/usd_converter.py a40/check_usd_converter.py
bash a40/hpc_environment.sh python-cpu a40/usd_converter.py -m pip freeze > config/usd-converter.lock.txt
bash a40/hpc_environment.sh python-cpu a40/activate_usd_isolation.py
