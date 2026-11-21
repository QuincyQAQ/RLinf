"""Allocated, bounded, original-config B1 numerical diagnosis from last good weights."""
import argparse
import json
import os
from pathlib import Path
from reference import ROOT,source_recipe
from b2_continuation import run,validate,write
p=argparse.ArgumentParser();p.add_argument('--python',required=True);p.add_argument('--guard',action='store_true');a=p.parse_args()
assert os.environ.get('SLURM_JOB_ID')
root=ROOT/'outputs'/('b1_diagnostic_'+os.environ['SLURM_JOB_ID']);root.mkdir()
source=ROOT/'benchmark/logs/rsl_rl/b1_diagnostic_14031449/2026-11-20_13-15-36_B1_rough_2048_s0/model_2850.pt'
validate(source)
args=source_recipe('B1',0)
for flag,value in [('--max_iterations',149),('--experiment_name','b1_diagnostic_'+os.environ['SLURM_JOB_ID'])]:
    args[args.index(flag)+1]=str(value)
args+=['--pretrained_checkpoint',str(source),'--kit_args=--/app/extensions/excluded/0=omni.kit.telemetry --/persistent/isaac/asset_root/cloud='+str(ROOT/'assets/isaac_sim_5.1')]
write(root/'protocol.json',dict(source=str(source),optimizer_restored=True,diagnostic_only=True,updates=149,task='B1',seed=0,num_envs=2048,recipe_changes=[]))
run([a.python,ROOT/'a40/train_b1_resume_finite.py',*args],root/'training',3600,
    GO2_B1_DIAGNOSTICS=str(root/'failure_capture'),GO2_B1_NUMERIC_GUARD='1' if a.guard else '0')
write(root/'completion.json',dict(status='completed_without_reproducing_failure'))
