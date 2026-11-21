#!/usr/bin/env bash
#SBATCH --job-name=go2_B2_postbaseline
#SBATCH --partition=emergency_gpua40
#SBATCH --account=root
#SBATCH --qos=emergency_gpua40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=gpu3-13
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
bash a40/hpc_environment.sh run-rlinf-continuation --gate-receipt /workspace/go2/validation/b2_postbaseline_gpu_gate.json
