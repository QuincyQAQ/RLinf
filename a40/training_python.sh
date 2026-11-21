#!/usr/bin/env bash
# Called by official python.sh after SDK environment setup. Keep the optional
# standalone URDF exporter out of locomotion's process; it needs a different lxml.
set -euo pipefail
TASK_FILTERED_PATH=()
IFS=: read -ra TASK_PYTHON_PATHS <<< "${PYTHONPATH:-}"
for TASK_PATH in "${TASK_PYTHON_PATHS[@]}"; do
  [[ "$TASK_PATH" == /isaac-sim/exts/isaacsim.asset.exporter.urdf/pip_prebundle ]] && continue
  [[ -n "$TASK_PATH" ]] && TASK_FILTERED_PATH+=("$TASK_PATH")
done
export PYTHONPATH="$(IFS=:; echo "${TASK_FILTERED_PATH[*]}")"
exec /isaac-sim/kit/python/bin/python3 "$@"
