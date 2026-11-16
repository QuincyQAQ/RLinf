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

import torch
from omegaconf import DictConfig


def get_model(cfg: DictConfig, torch_dtype: torch.dtype) -> torch.nn.Module:
    """Build the Go2 actor/critic, optionally initialized from an RSL-RL model."""
    from .go2_policy import Go2Policy

    if cfg.num_action_chunks != 1 or cfg.action_dim != 12:
        raise ValueError("Go2 requires action_dim=12 and num_action_chunks=1.")
    if torch_dtype != torch.float32:
        raise ValueError("Go2's validated PPO configuration requires float32.")
    return Go2Policy(
        cfg.get("initial_checkpoint"),
        obs_dim=int(cfg.obs_dim),
        critic_obs_dim=int(cfg.get("critic_obs_dim", 60)),
        obs_normalization=bool(cfg.get("obs_normalization", False)),
    )
