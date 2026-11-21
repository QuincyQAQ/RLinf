#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
ABC_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
ABC_RUNTIME="$(realpath "${GO2_ABC_RUNTIME:?Set GO2_ABC_RUNTIME to the prepared A40 runtime}")"
ABC_ENGINE="${GO2_ABC_CONTAINER:-singularity}"
ABC_IMAGE="${GO2_ABC_IMAGE:-$ABC_RUNTIME/containers/isaac-lab_2.3.2.sif}"
ABC_SESSION="${SLURM_JOB_ID:-cpu}_${SLURM_ARRAY_TASK_ID:-0}_$$"
ABC_PRIVATE="$ABC_PROJECT/data/go2_abc/container/$ABC_SESSION"
mkdir -p "$ABC_PRIVATE"/{home,tmp,cache,data,logs}
ABC_GPU=(--env CUDA_VISIBLE_DEVICES=)
for ABC_ARGUMENT in "$@"; do
  if [[ "$ABC_ARGUMENT" == --execute ]]; then
    : "${SLURM_JOB_ID:?An allocated A40 Slurm job is required}"
    ABC_VISIBLE="${CUDA_VISIBLE_DEVICES:?The allocation must expose its GPU mask}"
    mapfile -t ABC_NVML_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
    if [[ ${#ABC_NVML_IDS[@]} -eq 1 ]]; then ABC_VISIBLE="${ABC_NVML_IDS[0]}"; fi
    ABC_GPU=(--nv --env "CUDA_VISIBLE_DEVICES=$ABC_VISIBLE" --env "SLURM_JOB_ID=$SLURM_JOB_ID")
  fi
done
exec "$ABC_ENGINE" exec --cleanenv --writable-tmpfs "${ABC_GPU[@]}" \
  --home "$ABC_PRIVATE/home" --pwd /workspace/project \
  --bind "$ABC_PROJECT:/workspace/project" --bind "$ABC_RUNTIME:/workspace/go2" \
  --bind "$ABC_RUNTIME/IsaacLab:/workspace/isaaclab" --bind "$ABC_PRIVATE/tmp:/tmp" \
  --bind "$ABC_PRIVATE/cache:/isaac-sim/kit/cache" \
  --bind "$ABC_PRIVATE/data:/isaac-sim/kit/data" --bind "$ABC_PRIVATE/logs:/isaac-sim/kit/logs" \
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env PYTHONUNBUFFERED=1 --env GO2_RLINF=1 \
  --env GO2_ABC_RUNTIME=/workspace/go2 --env GO2_ABC_DATA=/workspace/project/data/go2_abc \
  "$ABC_IMAGE" bash /workspace/go2/a40/container_python.sh \
  -m examples.embodiment.go2_abc.run "$@"
