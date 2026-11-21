#!/usr/bin/env bash
# Execute only on the future A40 host. Never modifies the existing isaaclab environment.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
if [[ "${1:-}" != "--install" ]]; then
  echo "On A40: bash a40/prepare_environment.sh --install"
  echo "Requires Python 3.11, NVIDIA graphics/Vulkan drivers, network access and disk space."
  echo "Installs only into $TASK_ROOT/.venv; does not use sudo."
  exit 0
fi
python3.11 -c 'import sys; assert sys.version_info[:2] == (3, 11)'
python3.11 -m venv .venv
TASK_PYTHON="$TASK_ROOT/.venv/bin/python"
export PIP_CONSTRAINT="$TASK_ROOT/config/constraints.txt"
"$TASK_PYTHON" -m pip install 'pip>=24,<26' setuptools wheel toml
"$TASK_PYTHON" -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
"$TASK_PYTHON" -m pip install 'isaacsim[all,extscache]==5.1.0.0' --extra-index-url https://pypi.nvidia.com
"$TASK_PYTHON" -m pip install tensordict==0.8.3 'rsl-rl-lib==3.1.2'
# Install only the four required Lab packages. Avoid installing unrelated RL frameworks.
"$TASK_PYTHON" -m pip install -e ./IsaacLab/source/isaaclab -e ./IsaacLab/source/isaaclab_assets \
  -e ./IsaacLab/source/isaaclab_tasks -e './IsaacLab/source/isaaclab_rl[rsl-rl]' -e ./benchmark
"$TASK_PYTHON" -m pip check
mkdir -p validation
"$TASK_PYTHON" -m pip freeze > validation/a40_pip_freeze.txt
"$TASK_PYTHON" -m pip inspect > validation/a40_pip_inspect.json
nvidia-smi > validation/a40_nvidia_smi.txt
if command -v vulkaninfo >/dev/null 2>&1; then
  vulkaninfo --summary > validation/a40_vulkan_summary.txt
fi
"$TASK_PYTHON" a40/reference.py check
echo "Environment installed. A40 smoke/runtime checks still required before formal training."
