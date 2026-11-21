#!/usr/bin/env bash
#SBATCH --job-name=go2_fullppo_huber
#SBATCH --partition=emergency_gpua40
#SBATCH --account=root
#SBATCH --qos=emergency_gpua40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --nice=0
#SBATCH --exclude=gpu3-13
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
TASK_EXTRA=()
case "${1:?A2 or B1}" in
  A2) ;;
  B1) TASK_EXTRA+=(--b1-recovered-job 14032252) ;;
  *) echo 'This folder supports A2/B1 here; B2 uses b2_continuation.sbatch.' >&2; exit 2 ;;
esac
bash a40/hpc_environment.sh run-rlinf-fullppo --task "$1" "${TASK_EXTRA[@]}"
