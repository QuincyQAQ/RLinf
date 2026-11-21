#!/usr/bin/env bash
# Python entry inside the official Isaac Lab container; package installs stay in this workspace.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x /isaac-sim/python.sh ]]; then
  echo 'This entrypoint must run inside the prepared Isaac Lab 2.3.2 container.' >&2
  exit 2
fi
export PYTHONNOUSERSITE=1
export PYTHONPATH="$TASK_ROOT/.python_overlay:$TASK_ROOT/benchmark:$TASK_ROOT/IsaacLab/source/isaaclab:$TASK_ROOT/IsaacLab/source/isaaclab_assets:$TASK_ROOT/IsaacLab/source/isaaclab_tasks:$TASK_ROOT/IsaacLab/source/isaaclab_rl:${PYTHONPATH:-}"
if [[ -s "$TASK_ROOT/validation/extra_overlay_ready.json" ]]; then
  export PYTHONPATH="$TASK_ROOT/.extra_overlay:$PYTHONPATH"
fi
if [[ "${GO2_RLINF:-0}" == 1 ]]; then
  export PYTHONPATH="$TASK_ROOT/.rlinf_overlay:$TASK_ROOT/RLinf:$TASK_ROOT/a40:$PYTHONPATH"
  export RLINF_EXT_MODULE=go2_isaaclab
  export RLINF_LOG_LEVEL=INFO
fi
if [[ -s "$TASK_ROOT/validation/usd_isolation_ready.json" ]]; then
  export PYTHONEXE="$TASK_ROOT/a40/training_python.sh"
fi
exec /isaac-sim/python.sh "$@"
