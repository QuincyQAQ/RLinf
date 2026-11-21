"""Allocated-compute native PPO checks using the original Isaac environment."""
import argparse
import json
import os
import subprocess
from pathlib import Path
import torch
from reference import ROOT, read_manifest, run_evaluation
from rlinf_reference import run_phase


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--python',type=Path,required=True)
    p.add_argument('--task',choices=['A2','B1','B2'],default='A2')
    p.add_argument('--evaluate',action='store_true')
    p.add_argument('--reuse-root',type=Path)
    a=p.parse_args();root=ROOT/'validation'/('compute_rlinf_'+os.environ['SLURM_JOB_ID'])
    root.mkdir(parents=True,exist_ok=True)
    result={}
    with (root/'config_check.log').open('w') as log:
        subprocess.run([str(a.python),str(ROOT/'a40/runtime_probe.py'),'--task-key',a.task,
                        '--seed','0','--mode','config','--headless','--report',str(root/'effective_config.json')],
                       stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    assert json.loads((root/'effective_config.json').read_text())['status']=='passed'
    for stage,count in [('smoke',64),('capacity',read_manifest()['tasks'][a.task]['num_envs'])]:
        if a.reuse_root:
            assert json.loads((a.reuse_root/'effective_config.json').read_text())['task']==a.task
            previous_result=json.loads((a.reuse_root/'result.json').read_text())[stage]
            assert previous_result['status']=='passed' and previous_result['num_envs']==count
            model=Path(previous_result['checkpoint'])
        else:
            model=run_phase(a.task,root/stage,a.python,count,2)
        state=torch.load(model,map_location='cpu',weights_only=False)
        assert all(torch.isfinite(v).all() for v in state.values())
        assert list(state['actor.0.weight'].shape)==[512,235]
        assert (state['std']>0).all()
        previous=Path(str(model).replace('/global_step_2/','/global_step_1/'))
        old=torch.load(previous,map_location='cpu',weights_only=False)
        changed={key:any(not torch.equal(v,old[k]) for k,v in state.items() if k.startswith(key))
                 for key in ['actor.','critic.']}
        assert all(changed.values()),'Actor/critic failed to update'
        result[stage]=dict(status='passed',num_envs=count,updates=2,checkpoint=str(model),finite_weights=True,
                           positive_std=True,changed=changed)
        (root/'result.json').write_text(json.dumps(result,indent=2))
    if a.evaluate:
        from rlinf.models.embodiment.go2_policy.go2_policy import Go2Policy
        policy=Go2Policy(obs_dim=235,critic_obs_dim=235)
        policy.load_state_dict(state,strict=True)
        exported=root/'policy_for_eval.pt'
        torch.save(dict(model_state_dict=state,optimizer_state_dict=torch.optim.Adam(policy.parameters()).state_dict(),
                        iter=2,infos={'inference_export_only':True,'native_checkpoint':str(model)}),exported)
        run_evaluation(a.task,0,exported,a.python,root/'evaluation',dict(os.environ,GO2_ASSET_SOURCE='mirror'),[a.task])
        result['evaluation']=json.loads((root/'evaluation'/a.task/f'verified_{a.task}.json').read_text())
        assert result['evaluation']['required_obstacle_metrics_complete']
    result['status']='passed'
    result['task']=a.task
    (root/'result.json').write_text(json.dumps(result,indent=2))
    print('NATIVE_PPO_COMPUTE_SMOKE_PASSED',json.dumps(result),flush=True)
