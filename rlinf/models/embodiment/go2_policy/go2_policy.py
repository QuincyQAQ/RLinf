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

"""Asymmetric 45/60-observation Go2 policy retaining RSL-RL parameter names."""

from typing import Any

import torch
from torch import nn
from torch.distributions import Normal

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType


class Go2Policy(nn.Module, BasePolicy):
    """Use the validated ELU actor and privileged critic in RLinf PPO."""

    def __init__(self, initial_checkpoint: str | None = None) -> None:
        super().__init__()
        self.actor = self._network(45, 12)
        self.critic = self._network(60, 1)
        self.std = nn.Parameter(torch.ones(12))
        if initial_checkpoint:
            state = torch.load(
                initial_checkpoint, map_location="cpu", weights_only=False
            )
            self.load_state_dict(state["model_state_dict"], strict=True)

    @staticmethod
    def _network(inputs: int, outputs: int) -> nn.Sequential:
        sizes = [inputs, 512, 256, 128, outputs]
        layers = []
        for i in range(4):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < 3:
                layers.append(nn.ELU())
        return nn.Sequential(*layers)

    def preprocess_env_obs(self, env_obs: dict) -> dict:
        """Move both actor and privileged critic observations to the policy."""
        device = next(self.parameters()).device
        return {
            key: env_obs[key].to(device=device, dtype=torch.float32)
            for key in ("states", "critic_states")
        }

    def forward(
        self, forward_type: ForwardType = ForwardType.DEFAULT, **kwargs: Any
    ) -> dict:
        """Dispatch the trainable forward through RLinf's policy interface."""
        return BasePolicy.forward(self, forward_type=forward_type, **kwargs)

    def default_forward(
        self,
        forward_inputs: dict,
        compute_logprobs: bool = True,
        compute_entropy: bool = True,
        compute_values: bool = True,
        **kwargs: Any,
    ) -> dict:
        """Recompute the per-joint log probabilities and privileged value."""
        distribution = Normal(self.actor(forward_inputs["states"]), self.std)
        result = {}
        if compute_logprobs:
            result["logprobs"] = distribution.log_prob(forward_inputs["action"])
        if compute_entropy:
            result["entropy"] = distribution.entropy()
        if compute_values:
            result["values"] = self.critic(forward_inputs["critic_states"])
        return result

    @torch.inference_mode()
    def predict_action_batch(
        self,
        env_obs: dict,
        mode: str = "train",
        calculate_logprobs: bool = True,
        calculate_values: bool = True,
        return_obs: bool = True,
        **kwargs: Any,
    ) -> tuple:
        """Return [N, 1, 12] actions and PPO rollout bookkeeping."""
        obs = self.preprocess_env_obs(env_obs)
        mean = self.actor(obs["states"])
        distribution = Normal(mean, self.std)
        if mode not in ("train", "eval"):
            raise ValueError(f"Expected train or eval mode, got {mode!r}.")
        action = distribution.sample() if mode == "train" else mean
        forward_inputs = {**obs, "action": action, "model_action": action}
        result = {
            "prev_logprobs": distribution.log_prob(action),
            "prev_values": self.critic(obs["critic_states"])
            if calculate_values
            else torch.zeros_like(action[:, :1]),
            "forward_inputs": forward_inputs,
        }
        return action[:, None, :], result
