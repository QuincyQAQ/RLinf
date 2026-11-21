"""Opt-in B1 action/history bound; original reward and termination definitions preserved."""
import json
import os
import torch


def bounded_action(action,bound=100.):
    if not torch.isfinite(action).all():
        raise FloatingPointError('Nonfinite B1 action before simulator step')
    return action.clamp(-bound,bound)


def configure(cfg):
    """Bound actions before ActionManager stores last_action; no reward changes."""
    from isaaclab.envs import ManagerBasedRLEnv
    cfg.b1_action_bound=100.
    if not getattr(ManagerBasedRLEnv.step,'_b1_action_bound',False):
        original=ManagerBasedRLEnv.step
        def step(env,action):
            bound=getattr(env.cfg,'b1_action_bound',None)
            if bound is not None:
                action=bounded_action(action,bound)
            return original(env,action)
        step._b1_action_bound=True
        ManagerBasedRLEnv.step=step
    print('[B1_ACTION_BOUND]',json.dumps(dict(bound=100.,reward_changes=False,
          termination_changes=False,placement='before ActionManager last-action storage')),flush=True)


def install_gym_hook():
    import gymnasium as gym
    original=gym.make
    def make(task,*args,**kwargs):
        if 'B1' in str(task) and os.environ.get('GO2_B1_NUMERIC_GUARD')=='1':
            configure(kwargs['cfg'])
        return original(task,*args,**kwargs)
    gym.make=make
