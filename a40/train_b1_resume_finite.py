"""Resume finite PPO model and optimizer; stop rather than save invalid updates."""
import importlib.util
import os
from pathlib import Path
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmark/scripts'))
spec=importlib.util.spec_from_file_location('finite_b1_original_train',ROOT/'benchmark/scripts/train.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
import torch
from train_legacy_distribution import install
install()
original_load=m.OnPolicyRunner.load
def load(self,path,*args,**kwargs):
    result=original_load(self,path,load_optimizer=True)
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    self.current_learning_iteration=int(checkpoint['iter'])+1
    self.alg.learning_rate=self.alg.optimizer.param_groups[0]['lr']
    if os.environ.get('GO2_B1_DIAGNOSTICS'):
        from b1_diagnostics import install as diagnostic_install
        diagnostic_install(self.alg.policy,self.env.unwrapped)
    if os.environ.get('GO2_B1_INJECT_BOUNDARY_TEST')=='1':
        task=self.env.unwrapped
        task.step(torch.full((task.num_envs,12),10000.,device=task.device))
        recorded=float(task.action_manager.action.abs().max())
        observed=task.observation_manager.compute()['policy']
        history=float(observed[:,36:48].abs().max())
        assert recorded<=100 and history<=100,(recorded,history)
        print('[B1_ACTION_BOUND_TEST] raw=10000 recorded='+str(recorded)+' history='+str(history),flush=True)
        task.reset()
    print('[FINITE_RESUME] Completed prefix updates',self.current_learning_iteration,flush=True)
    return result
m.OnPolicyRunner.load=load
original_step=torch.optim.Adam.step
def step(self,*args,**kwargs):
    for group in self.param_groups:
        for param in group['params']:
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise FloatingPointError('Nonfinite PPO gradient; rejecting update')
    result=original_step(self,*args,**kwargs)
    for group in self.param_groups:
        if any(not torch.isfinite(param).all() for param in group['params']):
            raise FloatingPointError('Nonfinite PPO parameter; rejecting checkpoint')
    return result
torch.optim.Adam.step=step
if os.environ.get('GO2_B1_NUMERIC_GUARD')=='1':
    from b1_numerical_boundary import install_gym_hook
    install_gym_hook()
code=0
try:m.main()
except BaseException:
    traceback.print_exc();code=1
sys.stdout.flush();sys.stderr.flush();os._exit(code)
