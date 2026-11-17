# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RLinf policy and PPO worker retaining the previous Go2 RSL-RL settings."""

import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from rlinf.algorithms.registry import calculate_adv_and_returns, register_policy_loss
from rlinf.models import register_model
from rlinf.models.embodiment.go2_policy.go2_policy import Go2Policy
from rlinf.utils.metric_utils import append_to_dict, compute_rollout_metrics
from rlinf.utils.nested_dict_process import put_tensor_device
from rlinf.utils.utils import masked_mean
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def gaussian_kl(old_mean, old_std, mean, std):
    """Use the same old-to-new diagonal Gaussian KL as RSL-RL PPO."""
    return (
        torch.log(std / old_std + 1e-5)
        + (old_std.square() + (old_mean - mean).square()) / (2 * std.square())
        - 0.5
    ).sum(-1)


def adaptive_learning_rate(lr, kl, target):
    """Apply the previous training run's KL thresholds and learning-rate bounds."""
    if kl > target * 2:
        return max(1e-5, lr / 1.5)
    if 0 < kl < target / 2:
        return min(1e-2, lr * 1.5)
    return lr


@register_policy_loss("go2_rsl_ppo")
def rsl_ppo_loss(**kwargs):
    """Use clipped PPO policy loss and clipped squared value error."""
    ratio = (kwargs["logprobs"] - kwargs["old_logprobs"]).exp()
    advantage = kwargs["advantages"]
    clipped = ratio.clamp(1 - kwargs["clip_ratio_low"], 1 + kwargs["clip_ratio_high"])
    mask = kwargs.get("loss_mask")
    actor = masked_mean(torch.maximum(-advantage * ratio, -advantage * clipped), mask)
    values, old, returns = kwargs["values"], kwargs["prev_values"], kwargs["returns"]
    clipped_value = old + (values - old).clamp(
        -kwargs["value_clip"], kwargs["value_clip"]
    )
    critic = masked_mean(
        torch.maximum((returns - values).square(), (returns - clipped_value).square()),
        mask,
    )
    return actor + critic, {
        "actor/policy_loss": actor.detach(),
        "actor/approx_kl": -masked_mean(
            kwargs["logprobs"] - kwargs["old_logprobs"], mask
        ).detach(),
        "actor/ratio": masked_mean(ratio, mask).detach(),
        "critic/value_loss": critic.detach(),
    }


class RslGo2Policy(Go2Policy):
    """Carry old Gaussian parameters alongside RLinf's PPO rollout data."""

    @torch.inference_mode()
    def predict_action_batch(self, env_obs, **kwargs):
        actions, result = super().predict_action_batch(env_obs, **kwargs)
        inputs = result["forward_inputs"]
        mean = self.actor(inputs["states"])
        inputs["old_mean"] = mean
        inputs["old_std"] = self.std.expand_as(mean).clone()
        return actions, result

    def default_forward(self, forward_inputs, **kwargs):
        output = super().default_forward(forward_inputs, **kwargs)
        mean = self.actor(forward_inputs["states"])
        output["action_mean"] = mean
        output["action_std"] = self.std.expand_as(mean)
        return output


def build_policy(cfg, torch_dtype):
    if torch_dtype != torch.float32:
        raise ValueError("The previous Go2 run uses float32")
    return RslGo2Policy(cfg.get("initial_checkpoint"))


def register():
    register_model("go2_rsl_policy", build_policy, force=True)


class RslGo2Actor(EmbodiedFSDPActor):
    """Use RLinf FSDP updates with global advantage normalization and adaptive KL."""

    def init_worker(self):
        seed = int(self.cfg.actor.seed) + self._rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.torch_platform.manual_seed_all(seed)
        super().init_worker()

    def save_checkpoint(self, save_path, step=0):
        super().save_checkpoint(save_path, step)
        if self._rank == 0:
            source = Path(self.cfg.runner.logger.log_path) / "training_env_state.json"
            shutil.copyfile(
                source, Path(save_path).parent / "environment_curriculum.json"
            )

    def run_training(self):
        result = super().run_training()
        with torch.no_grad():
            parameters = torch.cat(
                [p.detach().flatten() for p in self.model.parameters()]
            )
            reference = parameters.clone()
            dist.broadcast(reference, src=0)
            difference = (parameters - reference).abs().max()
            dist.all_reduce(difference, op=dist.ReduceOp.MAX)
        result["actor/policy_sync_error"] = float(difference)
        metadata = {
            "pid": os.getpid(),
            "rank": self._rank,
            "world_size": self._world_size,
            "device": str(parameters.device),
            "physical_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "optimizer_steps_since_worker_start": self.optimizer_steps,
            "policy_sync_error": float(difference),
        }
        directory = Path(self.cfg.runner.logger.log_path)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"worker_rank{self._rank}.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        return result

    def compute_advantages_and_returns(self):
        batch = self.rollout_batch
        result = calculate_adv_and_returns(
            task_type="embodied",
            adv_type="gae",
            rewards=batch["rewards"],
            dones=batch["dones"],
            values=batch["prev_values"],
            gamma=self.cfg.algorithm.gamma,
            gae_lambda=self.cfg.algorithm.gae_lambda,
            num_action_chunks=1,
            group_size=1,
            reward_type="action_level",
            loss_mask=batch.get("loss_mask"),
            loss_mask_sum=batch.get("loss_mask_sum"),
            normalize_advantages=False,
        )
        advantage = result["advantages"]
        mask = batch.get("loss_mask")
        valid = advantage if mask is None else advantage[mask]
        stats = torch.stack((valid.sum(), valid.new_tensor(valid.numel()))).to(
            self.device
        )
        dist.all_reduce(stats)
        mean = stats[0] / stats[1]
        m2 = (valid - mean.to(valid.device)).square().sum().to(self.device)
        dist.all_reduce(m2)
        std = (m2 / (stats[1] - 1)).sqrt()
        result["advantages"] = (advantage - mean.to(advantage.device)) / (
            std.to(advantage.device) + 1e-8
        )
        batch.update(result)
        metrics = compute_rollout_metrics(batch)
        metrics.update(self._group_success_metrics)
        metrics["global_advantage_mean"] = float(mean)
        metrics["global_advantage_std"] = float(std)
        return metrics

    def train_micro_batch(self, micro_batch, metrics, *, is_last):
        # One micro batch per minibatch preserves the original KL adjustment timing.
        if self.gradient_accumulation != 1:
            raise ValueError(
                "RSL-compatible Go2 updates require one micro batch per minibatch"
            )
        batch = put_tensor_device(micro_batch, self.device)
        inputs = batch["forward_inputs"]
        with torch.no_grad():
            output = self.model(
                forward_inputs=inputs,
                compute_logprobs=False,
                compute_values=False,
                compute_entropy=False,
            )
            kl = gaussian_kl(
                inputs["old_mean"],
                inputs["old_std"],
                output["action_mean"],
                output["action_std"],
            )
            mask = batch.get("loss_mask")
            valid = kl if mask is None else kl[mask.reshape(kl.shape)]
            stats = torch.stack((valid.sum(), valid.new_tensor(valid.numel())))
            dist.all_reduce(stats)
            global_kl = float(stats[0] / stats[1])
            lr = adaptive_learning_rate(
                self.optimizer.param_groups[0]["lr"],
                global_kl,
                self.cfg.algorithm.desired_kl,
            )
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            self.lr_scheduler.base_lrs = [lr] * len(self.optimizer.param_groups)
        append_to_dict(metrics, {"actor/analytic_kl": global_kl})
        super().train_micro_batch(batch, metrics, is_last=is_last)


class SeededGo2Rollout(MultiStepRolloutWorker):
    """Seed each sampling rank using the previous Go2 run's seed convention."""

    def init_worker(self):
        seed = int(self.cfg.actor.seed) + self._rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.torch_platform.manual_seed_all(seed)
        super().init_worker()


class Go2EnvWorker(EnvWorker):
    """Keep the single-owner velocity curriculum with RLinf checkpoints."""

    def _setup_env_and_wrappers(self, env_cls, env_cfg, num_envs_per_stage):
        envs = super()._setup_env_and_wrappers(env_cls, env_cfg, num_envs_per_stage)
        resume = self.cfg.runner.get("resume_dir")
        if not env_cfg.is_eval and resume:
            source = Path(resume) / "environment_curriculum.json"
            if source.exists():
                state = json.loads(source.read_text())
                for env in envs:
                    env.task.command_x = state["command_x"]
                    env.task.command_y = state["command_y"]
                    env.task.common_step = state["common_step"]
        return envs

    async def interact(self, input_channel, rollout_channel, reward_channel):
        result = await super().interact(input_channel, rollout_channel, reward_channel)
        task = self.env_list[0].task
        state = {
            "pid": os.getpid(),
            "num_envs": task.n,
            "command_x": task.command_x,
            "command_y": task.command_y,
            "common_step": task.common_step,
            "simulation_threads": int(self.cfg.env.train.threads),
            "note": (
                "Curriculum is restored; physical simulator and "
                "per-environment RNG states restart."
            ),
        }
        path = Path(self.cfg.runner.logger.log_path) / "training_env_state.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(path)
        result["lin_vel_cmd_level"] = torch.tensor([task.command_x])
        return result
