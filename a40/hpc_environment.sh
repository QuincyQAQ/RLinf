#!/usr/bin/env bash
# No scheduler requests are made here. CPU environment work is separate from allocated GPU runs.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_MODE="${1:-inspect}"
if [[ $# -gt 0 ]]; then shift; fi
TASK_RUNTIME=apptainer
# Enabled only after a real allocated A40 check, not a login-node test.
if [[ -s "$TASK_ROOT/validation/compute_singularity_ready.json" ]]; then
  TASK_RUNTIME=/hpc2ssd/softwares/singularity-ce/4.1.3/bin/singularity
fi
if [[ "$TASK_RUNTIME" == apptainer ]] && ! command -v apptainer >/dev/null 2>&1; then
  if type module >/dev/null 2>&1; then
    module load apptainer-1.4.5
  elif [[ -x /hpc2ssd/softwares/apptainer-1.4.5/bin/apptainer ]]; then
    export PATH="/hpc2ssd/softwares/apptainer-1.4.5/bin:$PATH"
  else
    echo 'Apptainer 1.4.5 is unavailable; load the cluster module first.' >&2
    exit 2
  fi
fi
TASK_IMAGE="$TASK_ROOT/containers/isaac-lab_2.3.2.sif"
[[ -s "$TASK_IMAGE" ]] || { echo "Container not downloaded: $TASK_IMAGE" >&2; exit 2; }
if [[ -s "$TASK_ROOT/validation/unpacked_container.json" && -d "$TASK_ROOT/containers/isaac-lab_2.3.2_unpacked/.singularity.d" ]]; then
  TASK_IMAGE="$TASK_ROOT/containers/isaac-lab_2.3.2_unpacked"
fi
TASK_HOME="$TASK_ROOT/.container_home"
TASK_CACHE_ROOT="$TASK_ROOT/.isaacsim_cache"
TASK_INSTANCE_SUFFIX=""
if [[ "$TASK_MODE" == run-rlinf-probe ]]; then TASK_INSTANCE_SUFFIX=_native_probe; fi
if [[ "$TASK_MODE" == run* && -n "${SLURM_JOB_ID:-}" ]]; then
  TASK_JOB_KEY="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}${TASK_INSTANCE_SUFFIX}"
  TASK_HOME="$TASK_ROOT/.container_homes/$TASK_JOB_KEY"
  TASK_CACHE_ROOT="$TASK_ROOT/.isaacsim_caches/$TASK_JOB_KEY"
fi
mkdir -p "$TASK_HOME" "$TASK_ROOT/validation"
mkdir -p "$TASK_CACHE_ROOT/kit/"{cache,data,logs}
TASK_CONTAINER=("$TASK_RUNTIME" exec --cleanenv --home "$TASK_HOME" \
  --bind "$TASK_ROOT:/workspace/go2" --bind "$TASK_ROOT/IsaacLab:/workspace/isaaclab" \
  --bind "$TASK_CACHE_ROOT/kit/cache:/isaac-sim/kit/cache" \
  --bind "$TASK_CACHE_ROOT/kit/data:/isaac-sim/kit/data" \
  --bind "$TASK_CACHE_ROOT/kit/logs:/isaac-sim/kit/logs" \
  --pwd /workspace/go2 --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env PYTHONUNBUFFERED=1)
TASK_PYTHON=/workspace/go2/a40/container_python.sh
if [[ "$TASK_MODE" == *rlinf* || "$TASK_MODE" == run-b1-diagnose || "$TASK_MODE" == run-confirmatory-eval ]]; then
  TASK_CONTAINER+=(--env GO2_RLINF=1)
fi
case "$TASK_MODE" in
  inspect|verify)
    TASK_VERIFY=()
    if [[ "$TASK_MODE" == verify ]]; then TASK_VERIFY=(--verify); fi
    exec "${TASK_CONTAINER[@]}" --env CUDA_VISIBLE_DEVICES= "$TASK_IMAGE" bash "$TASK_PYTHON" \
      /workspace/go2/a40/check_container_environment.py "${TASK_VERIFY[@]}" \
      --report "/workspace/go2/validation/container_${TASK_MODE}.json" "$@"
    ;;
  python-cpu|python-rlinf-cpu)
    exec "${TASK_CONTAINER[@]}" --env CUDA_VISIBLE_DEVICES= "$TASK_IMAGE" bash "$TASK_PYTHON" "$@"
    ;;
  plan)
    exec "${TASK_CONTAINER[@]}" --env CUDA_VISIBLE_DEVICES= "$TASK_IMAGE" bash "$TASK_PYTHON" \
      /workspace/go2/a40/reference.py plan --python "$TASK_PYTHON" "$@"
    ;;
  run|run-rlinf|run-probe|run-rlinf-probe|run-video|run-rlinf-continuation|run-rlinf-fullppo|run-b1-diagnose|run-confirmatory-eval)
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
      echo 'GPU execution requires an existing authorized Slurm allocation. This script never requests one.' >&2
      exit 2
    fi
    if [[ "$TASK_RUNTIME" == apptainer ]]; then
      TASK_IMAGE="$TASK_ROOT/containers/isaac-lab_2.3.2.sif"
    else
      # SIF's root-owned /isaac-sim is mode 750. The existing official-image
      # extraction is user-owned; Singularity underlay can use it on GPFS.
      TASK_IMAGE="$TASK_ROOT/containers/isaac-lab_2.3.2_unpacked"
      [[ -x "$TASK_IMAGE/isaac-sim/python.sh" ]] || exit 2
    fi
    TASK_TMP_BASE="${SLURM_TMPDIR:-/tmp}"
    if [[ ! -d "$TASK_TMP_BASE" ]]; then TASK_TMP_BASE=/tmp; fi
    case "$(stat -f -c %T "$TASK_TMP_BASE")" in
      gpfs|nfs|nfs4|lustre) TASK_TMP_BASE=/tmp ;;
    esac
    TASK_LOCAL_TMP="$TASK_TMP_BASE/go2_isaac_${USER}_${SLURM_JOB_ID}${TASK_INSTANCE_SUFFIX}"
    mkdir -p "$TASK_LOCAL_TMP"
    python3 - "$TASK_LOCAL_TMP" <<'PY'
import shutil,sys
if shutil.disk_usage(sys.argv[1]).free < 30*1024**3:
    raise SystemExit('Need at least 30 GiB node-local temporary space for the original SIF runtime')
PY
    export APPTAINER_TMPDIR="$TASK_LOCAL_TMP"
    export SINGULARITY_TMPDIR="$TASK_LOCAL_TMP"
    # Host /tmp/isaaclab may belong to another user. Isolate all simulator temp files.
    TASK_CONTAINER+=(--bind "$TASK_LOCAL_TMP:/tmp")
    TASK_GPU_ENV=(--env "SLURM_JOB_ID=$SLURM_JOB_ID" --env TMPDIR=/tmp)
    if [[ "$TASK_MODE" == *rlinf* || "$TASK_MODE" == run-b1-diagnose || "$TASK_MODE" == run-confirmatory-eval ]]; then
      mkdir -p "$TASK_LOCAL_TMP/ray"
      TASK_GPU_ENV+=(--env RAY_TMPDIR=/tmp/ray)
    fi
    if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
      TASK_GPU_ENV+=(--env "SLURM_ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID" --env "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID")
    fi
    if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
      TASK_GPU_VISIBLE="$CUDA_VISIBLE_DEVICES"
      # Some compute nodes expose a single permitted NVML device while the
      # batch mask still uses its host ordinal. Preserve the same allocated
      # card using its UUID rather than an ordinal that can change in a container.
      mapfile -t TASK_VISIBLE_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
      if [[ ${#TASK_VISIBLE_UUIDS[@]} -eq 1 && "${TASK_VISIBLE_UUIDS[0]}" == GPU-* ]]; then
        TASK_GPU_VISIBLE="${TASK_VISIBLE_UUIDS[0]}"
        if [[ "$TASK_MODE" == *rlinf* || "$TASK_MODE" == run-b1-diagnose || "$TASK_MODE" == run-confirmatory-eval ]]; then
          # This RLinf release accepts integer visible IDs only. NVML's sole
          # permitted device index is the allocation's container-visible ordinal.
          TASK_GPU_VISIBLE="$(nvidia-smi --query-gpu=index --format=csv,noheader)"
          [[ "$TASK_GPU_VISIBLE" =~ ^[0-9]+$ ]] || exit 2
        fi
      fi
      echo "Allocated GPU: host mask=$CUDA_VISIBLE_DEVICES container identifier=$TASK_GPU_VISIBLE"
      TASK_GPU_ENV+=(--env "CUDA_VISIBLE_DEVICES=$TASK_GPU_VISIBLE")
    fi
    if [[ "$TASK_MODE" == run-video ]]; then
      exec "${TASK_CONTAINER[@]}" --nv --writable-tmpfs "${TASK_GPU_ENV[@]}" "$TASK_IMAGE" bash "$TASK_PYTHON" \
        /workspace/go2/a40/record_video.py "$@"
    fi
    if [[ "$TASK_MODE" == run-confirmatory-eval ]]; then
      exec "${TASK_CONTAINER[@]}" --nv --writable-tmpfs "${TASK_GPU_ENV[@]}" "$TASK_IMAGE" bash "$TASK_PYTHON" \
        /workspace/go2/a40/a1_confirmatory_eval.py "$@" --python "$TASK_PYTHON"
    fi
    if [[ "$TASK_MODE" == run-b1-diagnose ]]; then
      exec "${TASK_CONTAINER[@]}" --nv --writable-tmpfs "${TASK_GPU_ENV[@]}" "$TASK_IMAGE" bash "$TASK_PYTHON" \
        /workspace/go2/a40/${GO2_B1_ENTRY:-b1_diagnostic_run.py} "$@" --python "$TASK_PYTHON"
    fi
    if [[ "$TASK_MODE" == run-rlinf-fullppo ]]; then
      exec "${TASK_CONTAINER[@]}" --nv --writable-tmpfs "${TASK_GPU_ENV[@]}" "$TASK_IMAGE" bash "$TASK_PYTHON" \
        /workspace/go2/a40/fullppo_huber_suite.py "$@" --python "$TASK_PYTHON"
    fi
    if [[ "$TASK_MODE" == run-rlinf-continuation ]]; then
      exec "${TASK_CONTAINER[@]}" --nv --writable-tmpfs "${TASK_GPU_ENV[@]}" "$TASK_IMAGE" bash "$TASK_PYTHON" \
        /workspace/go2/a40/b2_continuation.py "$@" --python "$TASK_PYTHON"
    fi
    TASK_ENTRY=/workspace/go2/a40/reference.py
    if [[ "$TASK_MODE" == run-rlinf ]]; then TASK_ENTRY=/workspace/go2/a40/rlinf_reference.py; fi
    if [[ "$TASK_MODE" == run-probe ]]; then
      exec "${TASK_CONTAINER[@]}" --nv --writable-tmpfs "${TASK_GPU_ENV[@]}" "$TASK_IMAGE" bash "$TASK_PYTHON" \
        /workspace/go2/a40/runtime_probe.py "$@"
    fi
    if [[ "$TASK_MODE" == run-rlinf-probe ]]; then
      exec "${TASK_CONTAINER[@]}" --nv --writable-tmpfs "${TASK_GPU_ENV[@]}" "$TASK_IMAGE" bash "$TASK_PYTHON" \
        /workspace/go2/a40/compute_rlinf_probe.py --python "$TASK_PYTHON" "$@"
    fi
    exec "${TASK_CONTAINER[@]}" --nv --writable-tmpfs "${TASK_GPU_ENV[@]}" "$TASK_IMAGE" bash "$TASK_PYTHON" \
      "$TASK_ENTRY" "$@" --python "$TASK_PYTHON"
    ;;
  *) echo "Usage: bash a40/hpc_environment.sh {inspect|verify|python-cpu|plan|run} [args]" >&2; exit 2 ;;
esac
