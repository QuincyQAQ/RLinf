#!/usr/bin/env bash
#SBATCH --job-name=go2_native_ppo_smoke
#SBATCH --partition=debug
#SBATCH --qos=debug
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
bash a40/hpc_environment.sh run-rlinf-probe
