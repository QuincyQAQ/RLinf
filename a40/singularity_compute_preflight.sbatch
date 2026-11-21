#!/usr/bin/env bash
#SBATCH --job-name=go2_container_preflight
#SBATCH --partition=debug
#SBATCH --account=root
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
TASK_RUNTIME=/hpc2ssd/softwares/singularity-ce/4.1.3/bin/singularity
export SINGULARITY_TMPDIR="${SLURM_TMPDIR:-/tmp}/go2_singularity_${SLURM_JOB_ID}"
mkdir -p "$SINGULARITY_TMPDIR"
echo "Container preflight: $(hostname) $(date -Is) job=$SLURM_JOB_ID"
cat /proc/sys/kernel/unprivileged_userns_clone
TASK_ROOT="$PWD"
TASK_CACHE="$TASK_ROOT/.isaacsim_caches/preflight_$SLURM_JOB_ID"
mkdir -p "$TASK_CACHE/"{cache,data,logs} "$TASK_ROOT/.container_homes/preflight_$SLURM_JOB_ID"
"$TASK_RUNTIME" exec --cleanenv --nv --writable-tmpfs \
  --home "$TASK_ROOT/.container_homes/preflight_$SLURM_JOB_ID" \
  --bind "$TASK_ROOT:/workspace/go2" --bind "$TASK_ROOT/IsaacLab:/workspace/isaaclab" \
  --bind "$TASK_CACHE/cache:/isaac-sim/kit/cache" \
  --bind "$TASK_CACHE/data:/isaac-sim/kit/data" \
  --bind "$TASK_CACHE/logs:/isaac-sim/kit/logs" \
  --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?}" \
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --pwd /workspace/go2 \
  containers/isaac-lab_2.3.2_unpacked bash /workspace/go2/a40/container_python.sh -c \
  'import torch,json; assert torch.cuda.is_available(); name=torch.cuda.get_device_name(0); assert "A40" in name; print(json.dumps(dict(status="passed",gpu=name,visible_gpus=torch.cuda.device_count(),torch=torch.__version__))); print("CONTAINER_COMPUTE_PREFLIGHT_PASSED")'
