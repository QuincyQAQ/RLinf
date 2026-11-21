#!/usr/bin/env bash
#SBATCH --job-name=go2_ABC_C_reproduce_s0
#SBATCH --partition=emergency_gpua40
#SBATCH --qos=emergency_gpua40
#SBATCH --account=root
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=gpu3-13
set -euo pipefail
cd "${GO2_ABC_PROJECT:?Set the independent branch checkout path}"
git diff --exit-code HEAD
git rev-parse HEAD
mkdir -p data/go2_abc/scheduler
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv > "data/go2_abc/scheduler/gpu_${SLURM_JOB_ID}.csv"
bash examples/embodiment/go2_abc/container.sh --python -m examples.embodiment.go2_abc.reproduce
