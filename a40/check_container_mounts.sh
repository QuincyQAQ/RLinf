#!/usr/bin/env bash
# Exercise container filesystem/cache behavior on CPU only. No Kit or GPU work.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
if ! command -v apptainer >/dev/null 2>&1; then module load apptainer-1.4.5; fi
python3 - <<'PY'
import shutil
if shutil.disk_usage('/tmp').free < 30*1024**3:
    raise SystemExit('Need at least 30 GiB node-local temporary space for the container preflight')
PY
TASK_SIF_TMP="$(mktemp -d /tmp/go2_sif_preflight.XXXXXX)"
export APPTAINER_TMPDIR="$TASK_SIF_TMP"
trap 'rmdir "$TASK_SIF_TMP" 2>/dev/null || true' EXIT
mkdir -p "$TASK_ROOT/.isaacsim_cache/kit/"{cache,data,logs}
apptainer exec --cleanenv --writable-tmpfs --home "$TASK_ROOT/.container_home" \
  --bind "$TASK_ROOT/.isaacsim_cache/kit/cache:/isaac-sim/kit/cache" \
  --bind "$TASK_ROOT/.isaacsim_cache/kit/data:/isaac-sim/kit/data" \
  --bind "$TASK_ROOT/.isaacsim_cache/kit/logs:/isaac-sim/kit/logs" \
  containers/isaac-lab_2.3.2.sif /bin/sh -c \
  'test -x /isaac-sim/python.sh && printf "CPU container cache preflight\n" > /isaac-sim/kit/cache/.go2_mount_preflight'
python3 - <<'PY'
from pathlib import Path
import json,time
assert Path('.isaacsim_cache/kit/cache/.go2_mount_preflight').read_text().startswith('CPU container')
Path('validation/container_mount_preflight.json').write_text(json.dumps({
    'status':'passed','finished_unix':time.time(),'image':'official SIF',
    'temporary_extraction':'node-local /tmp','cache_write_verified':True,
    'kit_started':False,'simulation_started':False,'gpu_used':False,
    'scope':'login-node container filesystem check; compute-node GPU checks still required'
},indent=2))
PY
