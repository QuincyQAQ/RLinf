#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
upstream=${GO2_MJLAB_PATH:-$repo_root/../unitree_rl_mjlab}
python_bin=${GO2_PYTHON:-$repo_root/.venv-mjlab/bin/python}
run_dir=${GO2_MJLAB_RUN_DIR:-$repo_root/outputs/go2_mjlab/upstream}
revision=1425b15f73bd4095f0df53709d7c389c3eb9e790

actual=$(git -C "$upstream" rev-parse HEAD)
if [[ $actual != "$revision" ]]; then
    echo "Expected unitree_rl_mjlab revision $revision, found $actual." >&2
    exit 1
fi
git -C "$upstream" diff --exit-code HEAD -- src scripts setup.py
"$python_bin" - <<'PY'
from importlib.metadata import version

for package, expected in {
    "mjlab": "1.2.0",
    "mujoco-warp": "3.5.0",
    "mujoco": "3.5.0",
    "warp-lang": "1.12.0",
    "rsl-rl-lib": "5.0.1",
}.items():
    actual = version(package)
    if actual != expected:
        raise SystemExit(f"Expected {package}=={expected}, found {actual}.")
PY

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
mkdir -p "$run_dir"
cd "$run_dir"
exec "$python_bin" "$upstream/scripts/train.py" Unitree-Go2-Flat \
    --env.scene.num-envs=4096 --agent.logger=tensorboard "$@"
