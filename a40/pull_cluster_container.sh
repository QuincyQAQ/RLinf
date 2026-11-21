#!/usr/bin/env bash
set -uo pipefail
cd /hpc2hdd/home/qli736/Unitree_GO2_A40_reference_20261118
module load apptainer-1.4.5
export APPTAINER_CACHEDIR="$PWD/.cache/apptainer"
export APPTAINER_TMPDIR="$PWD/.cache/apptainer-tmp"
export APPTAINER_MKSQUASHFS_PROCS=2
export APPTAINER_MKSQUASHFS_MEM=2G
export GOMAXPROCS=2
TASK_CPU_LIST="$(python3 -c 'import os; print(",".join(map(str,sorted(os.sched_getaffinity(0))[:2])))')"
nice -n 15 taskset -c "$TASK_CPU_LIST" apptainer pull containers/isaac-lab_2.3.2.sif docker://nvcr.io/nvidia/isaac-lab:2.3.2
TASK_PULL_STATUS=$?
python3 - "$TASK_PULL_STATUS" <<'R'
import json,sys,time
from pathlib import Path
Path('validation/container_pull_status.json').write_text(json.dumps({'exit_code':int(sys.argv[1]),'finished_unix':time.time(),'image':'nvcr.io/nvidia/isaac-lab:2.3.2','gpu_used':False,'slurm_job_submitted':False},indent=2))
R
exit "$TASK_PULL_STATUS"
