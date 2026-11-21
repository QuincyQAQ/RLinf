#!/usr/bin/env python3
"""CPU checks of the real RLinf policy/loss/config and external env registration."""
import json
import faulthandler
faulthandler.dump_traceback_later(90, exit=True)
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf

from reference import ROOT, read_manifest
from rlinf_reference import make_config


def main():
    print("Importing RLinf native modules", flush=True)
    from rlinf.algorithms.losses import compute_ppo_actor_critic_loss
    from rlinf.algorithms.utils import masked_mean
    from rlinf.envs import get_env_cls
    from rlinf.models.embodiment.go2_policy.go2_policy import Go2Policy
    from rlinf.runners.embodied_runner import EmbodiedRunner
    from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
    from go2_isaaclab import register, Go2IsaacLabEnv

    print("Registering Isaac tasks", flush=True)
    register()
    for task in read_manifest()["tasks"]:
        cfg = OmegaConf.create(make_config(task, ROOT / "outputs/RLINF_CPU_CHECK"))
        assert get_env_cls("isaaclab", cfg.env.train) is Go2IsaacLabEnv
        assert cfg.env.train.seed == cfg.actor.seed == 0
        assert cfg.actor.global_batch_size * 4 == cfg.env.train.total_num_envs * 24
        assert cfg.algorithm.update_epoch == 5
        assert cfg.actor.model.initial_checkpoint is None
        assert cfg.env.train.init_params.id == read_manifest()["tasks"][task]["gym_id"]
    torch.set_num_threads(1)
    torch.manual_seed(0)
    policy = Go2Policy(obs_dim=235, critic_obs_dim=235, obs_normalization=False)
    batch = {"states": torch.randn(32, 235), "critic_states": torch.randn(32, 235)}
    actions, rollout = policy.predict_action_batch(batch)
    assert actions.shape == (32, 1, 12)
    predictions = policy.default_forward(rollout["forward_inputs"])
    assert torch.allclose(predictions["logprobs"], rollout["prev_logprobs"])
    loss, _ = compute_ppo_actor_critic_loss(
        logprobs=predictions["logprobs"].sum(-1, keepdim=True),
        old_logprobs=rollout["prev_logprobs"].sum(-1, keepdim=True),
        advantages=torch.randn(32, 1), values=predictions["values"],
        returns=torch.randn(32, 1), prev_values=rollout["prev_values"],
        clip_ratio_high=.2, clip_ratio_low=.2, clip_ratio_c=None,
        value_clip=.2, huber_delta=10., value_loss_type="mse",
        loss_mask=None, loss_mask_sum=None, loss_agg_func=masked_mean,
        task_type="embodied", reward_type="action_level", logprob_type="action_level",
        max_episode_steps=1000, critic_warmup=False)
    assert torch.isfinite(loss)
    loss.backward()
    assert policy.actor[0].weight.grad is not None
    assert policy.critic[0].weight.grad is not None
    assert torch.isfinite(policy.actor[0].weight.grad).all()
    report = {"status": "passed", "loss": loss.item(),
              "real_policy_forward_backward": True, "upstream_ppo_loss_backward": True,
              "registered_tasks": list(read_manifest()["tasks"]),
              "workers": [cls.__name__ for cls in (EmbodiedRunner, EmbodiedFSDPActor, EnvWorker, MultiStepRolloutWorker)],
              "rsl_rl_training_used": False, "simulation_started": False,
              "gpu_runtime_checked": False}
    (ROOT / "validation/rlinf_cpu_checks.json").write_text(json.dumps(report, indent=2))
    faulthandler.cancel_dump_traceback_later()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
