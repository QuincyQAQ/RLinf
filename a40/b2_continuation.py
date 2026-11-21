"""Full-baseline B2 continuation: fixed budget, two RLinf recipes, PPO control."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from reference import ROOT, SHORT, evaluation_argv, source_recipe, verify_source, read_manifest
from rlinf_reference import make_config

SOURCE = ROOT/'benchmark/logs/rsl_rl/unitree_go2_b2_posture_apr15_refined_1024_seeds/2026-11-19_20-31-52_B2_stairs_posture_apr15_refined_1024_s0'
RECIPES = {
    'rlinf_conservative_mse': dict(lr=3e-4, entropy_bonus=.003, value_loss_type='mse', desired_kl=.005),
    'rlinf_conservative_huber': dict(lr=3e-4, entropy_bonus=.003, value_loss_type='huber', desired_kl=.005),
}

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj, indent=2))
    temporary.replace(path)

def run(command, directory, timeout, **env):
    directory.mkdir(parents=True, exist_ok=True)
    write(directory/'command.json', dict(argv=list(map(str,command)), overrides=env))
    with (directory/'console.log').open('w') as log:
        subprocess.run(list(map(str,command)), cwd=ROOT, env=dict(os.environ, **env),
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)

def state(path):
    import torch
    d=torch.load(path,map_location='cpu',weights_only=False)
    return d.get('model_state_dict', d)

def validate(path, initial=None):
    import torch
    from rlinf.models.embodiment.go2_policy.go2_policy import Go2Policy
    d=state(path)
    Go2Policy(obs_dim=235,critic_obs_dim=235).load_state_dict(d,strict=True)
    assert all(torch.isfinite(v).all() for v in d.values()), 'Nonfinite weights'
    assert (d['std']>0).all(), 'Nonpositive standard deviation'
    changes={}
    if initial is not None:
        old=state(initial)
        for prefix in ['actor.', 'critic.']:
            changes[prefix]=any(not torch.equal(v,old[k]) for k,v in d.items() if k.startswith(prefix))
            assert changes[prefix], f'{prefix} parameters did not update'
    return dict(status='passed', strict_load=True, finite=True, positive_std=True,
                changed=changes, checkpoint=str(path))

def native(python, initial, directory, recipe, count, updates, task='B2', start_iteration=None):
    import yaml
    cfg=make_config(task,directory,num_envs=count,updates=updates)
    cfg['actor']['model']['initial_checkpoint']=str(initial)
    cfg['env']['train']['continuation_start_iteration']=(read_manifest()['tasks'][task]['max_iterations']
        if start_iteration is None else int(start_iteration))
    if task!='B2':
        cfg['env']['train']['disable_kit_telemetry']=True
    if task=='B1':
        cfg['actor']['model']['distribution_safety_patch']=True
        for split in ['train','eval']:
            if cfg['env'].get(split) is not None:
                cfg['env'][split]['b1_numeric_guard']=os.environ.get('GO2_B1_NUMERIC_GUARD')=='1'
    cfg['actor']['optim'].update(lr=recipe['lr'],value_lr=recipe['lr'])
    cfg['algorithm'].update({k:v for k,v in recipe.items() if k!='lr'})
    directory.mkdir(parents=True)
    config=directory/'config';config.mkdir()
    (config/'experiment.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    # Ray socket paths must fit Unix-domain path limits.
    ray=Path(os.environ.get('RAY_TMPDIR','/tmp/ray'))/(recipe['value_loss_type']+'_'+directory.name[:2])
    ray.mkdir(parents=True,exist_ok=True)
    run([python, ROOT/'RLinf/examples/embodiment/train_embodied_agent.py',
         '--config-path',config,'--config-name','experiment'],directory,
         1800 if updates==2 else 14400,RAY_ADDRESS='local',RAY_TMPDIR=str(ray))
    checkpoint=directory/cfg['runner']['logger']['experiment_name']/f'checkpoints/global_step_{updates}/actor/model_state_dict/full_weights.pt'
    receipt=validate(checkpoint,initial)
    write(directory/'validation.json',receipt)
    exported=directory/'policy.pt'
    import torch
    # Reference evaluator expects the original RSL checkpoint envelope.
    # Optimizer here is an inference placeholder, never used for continuation.
    from rlinf.models.embodiment.go2_policy.go2_policy import Go2Policy
    policy=Go2Policy(obs_dim=235,critic_obs_dim=235)
    torch.save(dict(model_state_dict=state(checkpoint),iter=updates,
                    optimizer_state_dict=__import__('torch').optim.Adam(policy.parameters(),lr=.001).state_dict(),
                    infos={'inference_export_only':True,'native_checkpoint':str(checkpoint)}),exported)
    return exported

def ordinary(python, initial, directory, count, updates, task='B2'):
    args=source_recipe(task,0)
    experiment=task.lower()+'_postbaseline_'+os.environ.get('SLURM_JOB_ID','cpu')+'_'+directory.parent.name+'_'+directory.name
    replacements={'--num_envs':count,'--max_iterations':updates,'--experiment_name':experiment}
    if task=='B2':
        replacements.update({'--posture_height_weight_early':-3.5,'--posture_height_weight_late':-3.5,
                             '--posture_orientation_weight_early':-2.,'--posture_orientation_weight_late':-2.})
    if task!='B2':
        args.append('--kit_args=--/app/extensions/excluded/0=omni.kit.telemetry --/persistent/isaac/asset_root/cloud='+str(ROOT/'assets/isaac_sim_5.1'))
    for flag,value in replacements.items():
        if flag in args:args[args.index(flag)+1]=str(value)
        else:args += [flag,str(value)]
    args+=['--pretrained_checkpoint',str(initial)]
    run([python,ROOT/'a40/train_continuation_rsl.py',*args],directory,
        1800 if updates==2 else 10800)
    files=list((ROOT/'benchmark/logs/rsl_rl'/experiment).glob('*/model_*.pt'))
    assert files,'Ordinary PPO did not save a checkpoint'
    checkpoint=max(files,key=lambda p:p.stat().st_mtime_ns)
    write(directory/'validation.json',validate(checkpoint,initial))
    return checkpoint

def evaluate(python, checkpoint, directory, seed, task='B2'):
    command=evaluation_argv(task,0,checkpoint,python,directory)
    command[1]=str(ROOT/'a40/evaluate_continuation.py')
    if task!='B2':
        command.append('--kit_args=--/app/extensions/excluded/0=omni.kit.telemetry --/persistent/isaac/asset_root/cloud='+str(ROOT/'assets/isaac_sim_5.1'))
    run(command,directory,1200,GO2_EVAL_SEED=str(seed))
    f=directory/(SHORT[task]+'_s0')/(SHORT[task]+'.json')
    data=json.loads(f.read_text());episodes=data['episodes'];summary=data['summary']
    assert len(episodes)>=1024 and len(episodes)==summary['num_episodes']
    metrics=[('episode_success','success_rate')]
    if task in ['B2','C2']:
        metrics.append(('physical_obstacle_passage','physical_obstacle_passage_rate'))
    for field,key in metrics:
        assert all(e.get(field) is not None for e in episodes)
        assert abs(sum(bool(e[field]) for e in episodes)/len(episodes)-summary[key])<1e-10
    if task in ['B2','C2']:
        assert f.with_suffix('.geometry.json').is_file(),'Missing terrain geometry'
    write(directory/'verified.json',dict(status='passed',seed=seed,artifact=str(f),summary=summary))
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--python',type=Path,required=True)
    p.add_argument('--cpu-only',action='store_true')
    p.add_argument('--gate-only',action='store_true')
    p.add_argument('--gate-receipt',type=Path,default=ROOT/'validation/b2_postbaseline_gpu_gate.json')
    p.add_argument('--reuse-gate-root',type=Path)
    a=p.parse_args()
    verify_source()
    files=sorted(SOURCE.glob('model_*.pt'),key=lambda p:p.stat().st_mtime_ns)
    assert files,'Missing completed baseline checkpoints'
    initial=files[-1]
    baseline=ROOT/'outputs/train_s0_20261119_203027_599630/B2/verified_B2.json'
    assert baseline.is_file(),'Completed baseline evaluation is required'
    proof=validate(initial)
    protocol=dict(task='B2',training_seed=0,source=str(initial),source_selection='latest checkpoint by write time from completed 2200-update baseline; no score cherry-picking',
                  additional_updates=1000,num_envs=1024,rollout_steps=24,additional_transitions_per_lane=24576000,
                  recipes=RECIPES,control='original RSL PPO continuing identical weights',
                  optimizer='fresh for all three lanes',environment='fresh for all three lanes; terrain RNG state not restored',
                  posture_schedule='late weights from the first continuation step for all lanes',
                  curve_seed=12345,holdout_seeds=[23456,34567],
                  selection='two predeclared recipes, both fully reported; holdout never used to select or stop training',
                  primary_metric='physical_obstacle_passage_rate',secondary=['success_rate','joint_tracking_and_passage_success_rate'],
                  caveat='one training seed; tuned configuration comparison, not isolated framework causal effect',
                  sources=['https://rlinf.readthedocs.io/en/latest/rst_source/reference/algorithms/ppo.html',
                           'https://github.com/RLinf/RLinf/blob/main/examples/embodiment/config/maniskill_ppo_mlp.yaml'])
    if a.cpu_only:
        import torch
        assert not torch.cuda.is_initialized()
        write(ROOT/'validation/b2_postbaseline_cpu.json',dict(**proof,protocol=protocol,cuda_initialized=False))
        print('B2_CONTINUATION_CPU_PASSED',flush=True);return
    assert os.environ.get('SLURM_JOB_ID'),'GPU allocation required'
    root=ROOT/'outputs'/('b2_postbaseline_'+os.environ['SLURM_JOB_ID']);root.mkdir()
    write(root/'protocol.json',protocol)
    def progress(stage,**extra):write(root/'progress.json',dict(stage=stage,time=time.time(),**extra))
    try:
        lanes={'rsl_control':None,**RECIPES}
        if a.gate_receipt and not a.gate_only:
            receipt=json.loads(a.gate_receipt.read_text())
            assert receipt['status']=='passed' and receipt['protocol']==protocol
            for lane in lanes:
                for stage in ['smoke','capacity']:
                    assert receipt['lanes'][lane][stage]['status']=='passed'
            write(root/'gpu_gate.json',receipt)
            progress('gpu_gate_reused',receipt=str(a.gate_receipt))
        else:
            progress('config_gate')
            run([a.python,ROOT/'a40/runtime_probe.py','--headless','--mode','config',
                 '--task-key','B2','--seed','0','--report',root/'effective_config.json'],root/'config_gate',600)
            assert json.loads((root/'effective_config.json').read_text())['status']=='passed'
            gate={}
            for lane,recipe in lanes.items():
                gate[lane]={}
                for stage,count in [('smoke',64),('capacity',1024)]:
                    progress(stage,lane=lane)
                    if a.reuse_gate_root:
                        old=a.reuse_gate_root/lane/stage
                        receipt_file=old/'validation.json'
                        assert json.loads((a.reuse_gate_root/'protocol.json').read_text())==protocol
                        if receipt_file.is_file():
                            previous=json.loads(receipt_file.read_text())
                            assert previous['status']=='passed'
                            if recipe is not None:
                                import yaml
                                previous_cfg=yaml.safe_load((old/'config/experiment.yaml').read_text())
                                assert previous_cfg['env']['train']['total_num_envs']==count
                                assert previous_cfg['runner']['max_steps']==2
                                assert previous_cfg['actor']['model']['initial_checkpoint']==str(initial)
                                assert previous_cfg['actor']['optim']['lr']==recipe['lr']
                                for key,value in recipe.items():
                                    if key!='lr':assert previous_cfg['algorithm'][key]==value
                            else:
                                command=json.loads((old/'command.json').read_text())['argv']
                                assert command[command.index('--num_envs')+1]==str(count)
                                assert command[command.index('--max_iterations')+1]=='2'
                            gate[lane][stage]=dict(validate(Path(previous['checkpoint']),initial),reused_from=str(old))
                            continue
                    path=root/lane/stage
                    cp=ordinary(a.python,initial,path,count,2) if recipe is None else native(a.python,initial,path,recipe,count,2)
                    gate[lane][stage]=validate(cp,initial)
            receipt=dict(status='passed',protocol=protocol,lanes=gate,job_id=os.environ['SLURM_JOB_ID'])
            write(root/'gpu_gate.json',receipt)
            if a.gate_only:
                assert a.gate_receipt,'Gate-only mode requires a receipt path'
                write(a.gate_receipt,receipt)
                progress('gate_completed');return
        progress('checkpoint_curve')
        curve=[]
        # File-order fractions are checkpoint history positions, not asserted update numbers.
        for i,index in enumerate(sorted(set([int((len(files)-1)*.6),int((len(files)-1)*.8),len(files)-1]))):
            curve.append(dict(checkpoint=str(files[index]),history_index=index,
                              summary=evaluate(a.python,files[index],root/'curve'/str(i),12345)))
        rates=[x['summary']['physical_obstacle_passage_rate'] for x in curve]
        write(root/'baseline_curve.json',dict(points=curve,late_change=rates[-1]-rates[-2],
              plateau_candidate=abs(rates[-1]-rates[-2])<=.02,
              caveat='two late snapshots on one eval seed only; not statistical proof of plateau'))
        results={'baseline':{}}
        for seed in protocol['holdout_seeds']:
            results['baseline'][str(seed)]=evaluate(a.python,initial,root/'holdout_baseline'/str(seed),seed)
        for lane,recipe in lanes.items():
            progress('training',lane=lane,updates=1000)
            path=root/lane/'training'
            cp=ordinary(a.python,initial,path,1024,1000) if recipe is None else native(a.python,initial,path,recipe,1024,1000)
            results[lane]={}
            for seed in protocol['holdout_seeds']:
                progress('evaluation',lane=lane,seed=seed)
                results[lane][str(seed)]=evaluate(a.python,cp,root/lane/'evaluation'/str(seed),seed)
            write(root/'partial_results.json',results)
        write(root/'result.json',dict(status='verified',protocol=protocol,results=results))
        progress('completed')
    except BaseException as exc:
        write(root/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        progress('failed',error=repr(exc));raise

if __name__=='__main__':main()
