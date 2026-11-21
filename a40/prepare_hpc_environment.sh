#!/usr/bin/env bash
# CPU-only environment installation on the login node. Never calls srun/sbatch or --nv.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
if [[ "${1:-}" != --install ]]; then
  echo 'Usage: bash a40/prepare_hpc_environment.sh --install'
  echo 'Requires the official container already downloaded; CPU preparation only.'
  exit 0
fi
bash a40/hpc_environment.sh inspect --report /workspace/go2/validation/container_base_inventory.json
python3 - <<'PY'
import json
from pathlib import Path
root=Path.cwd()
expected=json.loads((root/'config/reference.json').read_text())['selected_package_versions']
actual=json.loads((root/'validation/container_base_inventory.json').read_text())
repairs=[]
for name,version in expected.items():
    if name=='isaacsim':
        if actual['packages'].get(name) not in (None,version):
            raise SystemExit('Unexpected Isaac Sim image metadata; do not replace the selected image silently')
        continue
    if actual['packages'].get(name)!=version: repairs.append(name+'=='+version)
(root/'validation/package_repairs.json').write_text(json.dumps(repairs,indent=2))
(root/'validation/torch_repairs.txt').write_text('\n'.join(x for x in repairs if x.startswith(('torch==','torchvision=='))))
(root/'validation/python_repairs.txt').write_text('\n'.join(x for x in repairs if not x.startswith(('torch==','torchvision=='))))
PY
mkdir -p .python_overlay
mapfile -t TASK_TORCH_REPAIRS < validation/torch_repairs.txt
if [[ ${#TASK_TORCH_REPAIRS[@]} -gt 0 ]]; then
  bash a40/hpc_environment.sh python-cpu -m pip install --no-deps --upgrade \
    --target /workspace/go2/.python_overlay --constraint /workspace/go2/config/constraints.txt \
    --index-url https://download.pytorch.org/whl/cu128 "${TASK_TORCH_REPAIRS[@]}"
fi
mapfile -t TASK_PYTHON_REPAIRS < validation/python_repairs.txt
if [[ ${#TASK_PYTHON_REPAIRS[@]} -gt 0 ]]; then
  bash a40/hpc_environment.sh python-cpu -m pip install --no-deps --upgrade \
    --target /workspace/go2/.python_overlay --constraint /workspace/go2/config/constraints.txt \
    "${TASK_PYTHON_REPAIRS[@]}"
fi
bash a40/hpc_environment.sh python-cpu a40/finalize_cpu_environment.py
echo 'CPU environment prepared. No GPU allocation, simulator or training was started.'
