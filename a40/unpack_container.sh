#!/usr/bin/env bash
# Reuse the official SIF contents without extracting 20+ GB on every invocation.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
if ! command -v apptainer >/dev/null 2>&1; then module load apptainer-1.4.5; fi
TASK_DEST="$TASK_ROOT/containers/isaac-lab_2.3.2_unpacked"
[[ ! -e "$TASK_DEST" ]] || { echo "Unpack destination already exists; inspect before retrying" >&2; exit 2; }
apptainer build --sandbox "$TASK_DEST" containers/isaac-lab_2.3.2.sif
python3 - <<'PY'
from pathlib import Path
import json,time
p=Path('containers/isaac-lab_2.3.2_unpacked')
if not (p/'.singularity.d').is_dir() or not (p/'isaac-sim/python.sh').is_file():
    raise SystemExit('Official image extraction is incomplete')
Path('validation/unpacked_container.json').write_text(json.dumps({'status':'passed','path':str(p),
    'source':'containers/isaac-lab_2.3.2.sif','finished_unix':time.time(),
    'source_contents_modified':False,'gpu_used':False,'simulation_started':False},indent=2))
PY
