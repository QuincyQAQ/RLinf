"""Frozen B2-Huber recipe transferred to four other complete-PPO specialists."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
from reference import ROOT, read_manifest, verify_source, source_recipe
from b2_continuation import RECIPES, write, run, native, ordinary, evaluate, validate

SOURCES = {'A2': ('unitree_go2_a2_baseline_2048_seeds/2026-11-19_20-27-02_A2_omni_2048_s0/model_1499.pt', 'train_s0_20261119_202408_239260/A2/training/console.log'), 'B1': ('unitree_go2_b1_baseline_2048_seeds/2026-11-19_22-18-47_B1_rough_2048_s0/model_2999.pt', 'train_s0_20261119_221656_974275/B1/training/console.log')}

def protocol(task):
    spec=read_manifest()['tasks'][task]
    return dict(task=task,training_seed=0,source=str(ROOT/'benchmark/logs/rsl_rl'/SOURCES[task][0]),
                training_log=str(ROOT/'outputs'/SOURCES[task][1]),completed_ppo_updates=spec['max_iterations'],
                additional_updates=1000,total_training_updates=spec['max_iterations']+1000,
                num_envs=spec['num_envs'],rollout_steps=24,
                additional_transitions_per_lane=1000*24*spec['num_envs'],
                recipe=dict(RECIPES['rlinf_conservative_huber']),
                recipe_selection='frozen B2 Huber recipe; no per-task hyperparameter search',
                source_selection='final saved checkpoint of verified complete PPO training, not best evaluation checkpoint',
                lanes=['rlinf_huber','rsl_control'],optimizer='fresh in both lanes',
                environment='fresh in both lanes; terrain RNG/curriculum state not restored',
                holdout_seeds=[23456,34567],episodes_per_evaluation_minimum=1024,
                metrics=['tracking','velocity errors']+(['physical passage','tracking/passage intersection'] if task=='C2' else []),
                b1_safety='requested original finite/std compatibility patch in both learners' if task=='B1' else 'not enabled',
                caveat='one training seed; parameterized framework comparison, not a framework-only causal effect')

def check_source(p):
    expected=p['completed_ppo_updates']
    assert f'trained_iterations={expected}/{expected}' in Path(p['training_log']).read_text(errors='replace')
    return validate(Path(p['source']))

def repair_b1(python,root):
    import torch
    source=ROOT/'benchmark/logs/rsl_rl'/SOURCES['B1'][0]
    history=[]
    # Recovery order is by recency, never by measured task score.
    for number in [1100,1050,1000]:
        initial=source.parent/f'model_{number}.pt'
        validate(initial)
        checkpoint=torch.load(initial,map_location='cpu',weights_only=False)
        prefix=int(checkpoint['iter'])+1
        remaining=3000-prefix
        folder=root/('b1_source_repair_'+str(number))
        attempt=dict(source=str(initial),prefix_updates=prefix,remaining_updates=remaining,
                     optimizer_restored=True,environment_reset=True,seed=0)
        try:
            for stage,count,updates in [('smoke',64,2),('capacity',2048,2),('training',2048,remaining)]:
                experiment='b1_source_repair_'+os.environ['SLURM_JOB_ID']+'_'+str(number)+'_'+stage
                args=source_recipe('B1',0)
                for flag,value in [('--max_iterations',updates),('--num_envs',count),('--experiment_name',experiment)]:
                    args[args.index(flag)+1]=str(value)
                args+=['--pretrained_checkpoint',str(initial),
                       '--kit_args=--/app/extensions/excluded/0=omni.kit.telemetry --/persistent/isaac/asset_root/cloud='+str(ROOT/'assets/isaac_sim_5.1')]
                run([python,ROOT/'a40/train_b1_resume_finite.py',*args],folder/stage,1800 if updates==2 else 14400)
                log=(folder/stage/'console.log').read_text(errors='replace')
                assert f'trained_iterations={updates}/{updates}' in log
                files=list((ROOT/'benchmark/logs/rsl_rl'/experiment).glob('*/model_*.pt'))
                final=max(files,key=lambda f:f.stat().st_mtime_ns)
                proof=validate(final,initial)
                assert int(torch.load(final,map_location='cpu',weights_only=False)['iter'])==prefix+updates-1
                write(folder/stage/'validation.json',proof)
            attempt.update(status='passed',final_checkpoint=str(final),completed_total_updates=3000)
            history.append(attempt)
            write(root/'source_repair.json',dict(status='passed',attempts=history,
                  caveat='resumed PPO and optimizer with fresh environment; discarded failed attempts are extra compute'))
            return final
        except Exception as exc:
            attempt.update(status='failed',error=repr(exc));history.append(attempt)
            write(root/'source_repair.json',dict(status='retrying',attempts=history))
    raise RuntimeError('B1 source recovery failed all three predeclared restart points')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',choices=list(SOURCES),required=True)
    parser.add_argument('--python',type=Path,required=True)
    parser.add_argument('--cpu-only',action='store_true')
    parser.add_argument('--repair-b1',action='store_true')
    parser.add_argument('--b1-recovered-job',type=int,help='Completed unchanged-recipe diagnostic recovery job')
    args=parser.parse_args();verify_source();p=protocol(args.task)
    initial=Path(p['source'])
    if args.b1_recovered_job:
        assert args.task=='B1' and not args.repair_b1
        os.environ['GO2_B1_NUMERIC_GUARD']='1'
        p['numerical_validity_repair']='B1 raw-action history feedback bounded to +/-100 before ActionManager; same in both learners/evaluation; no reward or termination changes'
        recovered=ROOT/'outputs'/('b1_diagnostic_'+str(args.b1_recovered_job))
        assert json.loads((recovered/'completion.json').read_text())['status']=='completed_without_reproducing_failure'
        recovery_log=(recovered/'training/console.log').read_text(errors='replace')
        recovery_protocol=json.loads((recovered/'protocol.json').read_text())
        updates=int(recovery_protocol['updates'])
        assert f'trained_iterations={updates}/{updates}' in recovery_log
        assert '[B1_ACTION_BOUND]' in recovery_log
        files=list((ROOT/'benchmark/logs/rsl_rl'/('b1_diagnostic_'+str(args.b1_recovered_job))).glob('*/model_2999.pt'))
        assert len(files)==1
        initial=files[0]
        import torch
        assert int(torch.load(initial,map_location='cpu',weights_only=False)['iter'])==2999
        source_proof=validate(initial)
        p.update(source=str(initial),source_selection=f'action-history bounded PPO source, restored prefix{3000-updates} plus{updates} updates',
                 source_recovery_job=args.b1_recovered_job,original_invalid_checkpoint=p['source'],
                 original_training_log=p['training_log'],training_log=str(recovered/'training/console.log'),
                 restored_completed_prefix=3000-updates,recovery_updates=updates,source_optimizer_restored=True,
                 recovery_caveat='finite source recovery is not proof that the intermittent numerical failure root cause is fixed')
    elif args.repair_b1:
        assert args.task=='B1'
        initial=initial.parent/'model_1100.pt'
        source_proof=validate(initial)
        p['source_recovery_required']=True
        p['last_finite_source']=str(initial)
    else:
        source_proof=check_source(p)
    if args.cpu_only:
        import torch
        assert not torch.cuda.is_initialized()
        write(ROOT/'validation'/('fullppo_huber_'+args.task+'_cpu.json'),
              dict(status='passed',cuda_initialized=False,protocol=p,source=source_proof))
        print('FULLPPO_HUBER_CPU_PASSED',args.task,flush=True);return
    assert os.environ.get('SLURM_JOB_ID'),'Requires allocated GPU'
    root=ROOT/'outputs'/('fullppo_huber_'+args.task+'_'+os.environ['SLURM_JOB_ID']);root.mkdir()
    write(root/'protocol.json',p)
    def progress(stage,**extra):
        write(root/'progress.json',dict(stage=stage,time=time.time(),task=args.task,**extra))
    try:
        progress('config_gate')
        run([args.python,ROOT/'a40/runtime_probe.py','--headless','--mode','config','--task-key',args.task,
             '--seed','0','--disable-kit-telemetry','--report',root/'effective_config.json'],root/'config_gate',600)
        assert json.loads((root/'effective_config.json').read_text())['status']=='passed'
        if args.repair_b1:
            progress('recovering_complete_ppo_source')
            initial=repair_b1(args.python,root)
            p.update(original_invalid_checkpoint=p['source'],source=str(initial),
                     source_selection='finite complete3000-update PPO source restored from prior finite checkpoint and optimizer')
            write(root/'protocol.json',p)
        lanes={'rlinf_huber':p['recipe'],'rsl_control':None}
        gate={}
        for lane,recipe in lanes.items():
            gate[lane]={}
            for stage,count in [('smoke',64),('capacity',p['num_envs'])]:
                progress(stage,lane=lane)
                folder=root/lane/stage
                model=(ordinary(args.python,initial,folder,count,2,args.task) if recipe is None else
                       native(args.python,initial,folder,recipe,count,2,args.task))
                gate[lane][stage]=validate(model,initial)
        write(root/'gpu_gate.json',dict(status='passed',lanes=gate))
        result={'baseline':{}}
        for seed in p['holdout_seeds']:
            progress('baseline_evaluation',seed=seed)
            result['baseline'][str(seed)]=evaluate(args.python,initial,root/'baseline'/str(seed),seed,args.task)
        write(root/'partial_results.json',result)
        for lane,recipe in lanes.items():
            progress('training',lane=lane,updates=1000)
            folder=root/lane/'training'
            model=(ordinary(args.python,initial,folder,p['num_envs'],1000,args.task) if recipe is None else
                   native(args.python,initial,folder,recipe,p['num_envs'],1000,args.task))
            result[lane]={}
            for seed in p['holdout_seeds']:
                progress('evaluation',lane=lane,seed=seed)
                result[lane][str(seed)]=evaluate(args.python,model,root/lane/'evaluation'/str(seed),seed,args.task)
            write(root/'partial_results.json',result)
        write(root/'result.json',dict(status='verified',protocol=p,results=result))
        progress('completed')
    except BaseException as exc:
        write(root/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        progress('failed',error=repr(exc));raise

if __name__=='__main__':main()
