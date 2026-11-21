#!/usr/bin/env bash
#SBATCH --job-name=go2_compute_ppo_smoke
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
echo "Real PPO preflight job=$SLURM_JOB_ID host=$(hostname) $(date -Is)"
for TASK_PROBE_MODE in smoke capacity; do
  bash a40/hpc_environment.sh run-probe --headless --task-key A2 --seed 0 --mode "$TASK_PROBE_MODE" \
    --report "/workspace/go2/validation/compute_ppo_${SLURM_JOB_ID}/${TASK_PROBE_MODE}.json" \
    --kit_args=--/persistent/isaac/asset_root/cloud=/workspace/go2/assets/isaac_sim_5.1
  python3 - "$SLURM_JOB_ID" "$TASK_PROBE_MODE" <<'PY'
import json,sys
from pathlib import Path
data=json.loads(Path('validation/compute_ppo_'+sys.argv[1]+'/'+sys.argv[2]+'.json').read_text())
assert data['status']=='passed',data.get('error',data['status'])
PY
done
python3 - "$SLURM_JOB_ID" <<'PY'
import json,sys
from pathlib import Path
for mode in ['smoke','capacity']:
    data=json.loads(Path('validation/compute_ppo_'+sys.argv[1]+'/'+mode+'.json').read_text())
    assert data['status']=='passed',data.get('error',data['status'])
    assert data['physics_or_training_executed'] and data[mode]['ppo_updates']==2
    assert data[mode]['changed_weights'] and data[mode]['finite_weights']
    print(mode,data[mode],flush=True)
print('REAL_PPO_COMPUTE_SMOKE_PASSED',flush=True)
PY
