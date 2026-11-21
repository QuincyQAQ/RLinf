#!/usr/bin/env bash
#SBATCH --job-name=go2_video
#SBATCH --partition=emergency_gpua40
#SBATCH --account=root
#SBATCH --qos=emergency_gpua40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
if [[ $# -lt 3 ]]; then
  echo 'Usage: sbatch a40/record_video.sbatch TASK /workspace/go2/CHECKPOINT /workspace/go2/OUTPUT.mp4 [extra options]' >&2
  exit 2
fi
VIDEO_TASK=$1
VIDEO_CHECKPOINT=$2
VIDEO_OUTPUT=$3
shift 3
bash a40/hpc_environment.sh run-video --task "$VIDEO_TASK" --checkpoint "$VIDEO_CHECKPOINT" --output "$VIDEO_OUTPUT" "$@" --execute
