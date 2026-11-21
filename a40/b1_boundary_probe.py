"""Real-GPU reset-path probe; one intentionally invalid simulated state, never hardware."""
import argparse,os
from reference import ROOT,source_recipe
from b2_continuation import run,write,validate
p=argparse.ArgumentParser();p.add_argument('--python',required=True);a=p.parse_args()
root=ROOT/'outputs'/('b1_boundary_probe_'+os.environ['SLURM_JOB_ID']);root.mkdir()
source=ROOT/'benchmark/logs/rsl_rl/b1_source_repair_14029762_1000_training/2026-11-20_11-11-51_B1_rough_2048_s0/model_2650.pt'
experiment='b1_boundary_probe_'+os.environ['SLURM_JOB_ID']
args=source_recipe('B1',0)
for flag,value in [('--max_iterations',2),('--num_envs',64),('--experiment_name',experiment)]:args[args.index(flag)+1]=str(value)
args+=['--pretrained_checkpoint',str(source),'--kit_args=--/app/extensions/excluded/0=omni.kit.telemetry --/persistent/isaac/asset_root/cloud='+str(ROOT/'assets/isaac_sim_5.1')]
run([a.python,ROOT/'a40/train_b1_resume_finite.py',*args],root/'training',900,
    GO2_B1_NUMERIC_GUARD='1',GO2_B1_INJECT_BOUNDARY_TEST='1')
log=(root/'training/console.log').read_text(errors='replace')
assert '[B1_ACTION_BOUND_TEST] raw=10000 recorded=100.0 history=100.0' in log
assert 'trained_iterations=2/2' in log
models=list((ROOT/'benchmark/logs/rsl_rl'/experiment).glob('*/model_*.pt'))
final=max(models,key=lambda f:f.stat().st_mtime_ns)
proof=validate(final,source)
write(root/'result.json',dict(status='passed',action_history_bounded=True,raw_action=10000,recorded_action=100,optimizer_updates_finite=True,proof=proof))
