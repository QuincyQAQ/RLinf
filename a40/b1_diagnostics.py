"""Capture the first numerical failure without replacing tensors or skipping updates."""
import json
import linecache
import os
from pathlib import Path
import sys
import torch


def install(policy,env=None):
    destination=Path(os.environ['GO2_B1_DIAGNOSTICS']);destination.mkdir(parents=True,exist_ok=True)
    latest={}
    def summary(value):
        if isinstance(value,torch.Tensor):
            x=value.detach()
            finite=torch.isfinite(x)
            return dict(shape=list(x.shape),finite=bool(finite.all()),
                        max_abs=float(x[finite].abs().max()) if finite.any() else None,
                        min=float(x[finite].min()) if finite.any() else None)
        return str(type(value))
    def capture(reason,extra=None):
        if (destination/'failure.pt').exists():return
        values={k:v.detach().cpu() for k,v in latest.items() if isinstance(v,torch.Tensor)}
        if 'obs_batch' in latest and hasattr(latest['obs_batch'],'to_dict'):
            values['obs_batch']=latest['obs_batch'].detach().cpu().to_dict()
        if env is not None:
            data=env.scene['robot'].data
            values['physics']={name:getattr(data,name).detach().cpu() for name in
                ['root_pos_w','root_quat_w','root_lin_vel_w','root_ang_vel_w','joint_pos','joint_vel']}
            values['ray_hits']=env.scene.sensors['height_scanner'].data.ray_hits_w.detach().cpu()
        values.update(model={k:v.detach().cpu() for k,v in policy.state_dict().items()},reason=reason)
        torch.save(values,destination/'failure.pt')
        report=dict(reason=reason,tensors={k:summary(v) for k,v in latest.items()
                                          if isinstance(v,torch.Tensor)},extra=extra)
        (destination/'failure.json').write_text(json.dumps(report,indent=2))
        print('[B1_DIAGNOSTIC]',json.dumps(report),flush=True)
    def trace(frame,event,arg):
        if frame.f_code.co_name=='update' and frame.f_code.co_filename.endswith('/algorithms/ppo.py'):
            if event=='line' and 'loss.backward()' in linecache.getline(frame.f_code.co_filename,frame.f_lineno):
                latest.clear();latest.update(frame.f_locals)
                for name in ['surrogate_loss','value_loss','loss']:
                    value=latest.get(name)
                    if isinstance(value,torch.Tensor) and not torch.isfinite(value).all():
                        capture('nonfinite '+name)
                        raise FloatingPointError('B1 nonfinite '+name+'; diagnostic saved')
            return trace
        return None
    sys.settrace(trace)
    original=torch.nn.utils.clip_grad_norm_
    def clip(parameters,*args,**kwargs):
        parameters=list(parameters)
        grads=[p.grad for p in parameters if p.grad is not None]
        if any(not torch.isfinite(g).all() for g in grads):
            capture('nonfinite gradient before clipping')
            raise FloatingPointError('B1 nonfinite preclip gradient; diagnostic saved')
        norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.detach().double()) for g in grads]))
        result=original(parameters,*args,**kwargs)
        if not torch.isfinite(result):
            capture('gradient norm reduction overflow',dict(float64_norm=float(norm)))
            raise FloatingPointError('B1 gradient norm overflow; diagnostic saved')
        return result
    torch.nn.utils.clip_grad_norm_=clip
    # A nonfinite simulator observation must be recorded before safety code can mask it.
    def before(module,args):
        obs=args[0]
        if isinstance(obs,torch.Tensor) and (not torch.isfinite(obs).all() or obs.abs().max()>1e4):
            latest['network_input']=obs;capture('invalid network input range')
            raise FloatingPointError('B1 nonfinite observation; diagnostic saved')
    policy.actor.register_forward_pre_hook(before)
    policy.critic.register_forward_pre_hook(before)
