#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
if [[ -n ${CANN_ENV_FILE:-} ]]; then source "$CANN_ENV_FILE"; fi
python_bin=${GO2_PYTHON:-$repo_root/.venv/bin/python}
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export EMBODIED_PATH="$repo_root/examples/embodiment"
unset RLINF_EXT_MODULE
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0
export HCCL_HOST_SOCKET_PORT_RANGE=auto HCCL_NPU_SOCKET_PORT_RANGE=auto
export RAY_enable_worker_prestart=0 RAY_prestart_worker_first_driver=0
cd "$repo_root"
config=${1:-go2_mujoco_ppo_ascend}
if [[ $# -gt 0 ]]; then shift; fi
exec "$python_bin" "$repo_root/examples/embodiment/train_embodied_agent.py" \
    --config-name "$config" "$@"
