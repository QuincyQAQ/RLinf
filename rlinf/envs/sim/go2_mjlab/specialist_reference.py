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

"""Task-specific rewards from the pinned five-specialist benchmark."""

from typing import Any

import torch

from . import b2_reference_rewards as rewards
from .b2_reference import apply_b2_reference


def stand_joint_deviation(env: Any) -> torch.Tensor:
    """Penalize joint deviation when the XY command is below 0.1 m/s."""
    robot = env.scene["robot"].data
    standing = env.command_manager.get_command("twist")[:, :2].norm(dim=1) < 0.1
    return (robot.joint_pos - robot.default_joint_pos).abs().sum(dim=1) * standing


def stand_base_height(env: Any, target_height: float) -> torch.Tensor:
    """Penalize absolute base-height error only while commanded to stand."""
    standing = env.command_manager.get_command("twist")[:, :2].norm(dim=1) < 0.1
    height = env.scene["robot"].data.root_link_pos_w[:, 2]
    return (height - target_height).square() * standing


def safe_base_height(env: Any, target_height: float, sensor_name: str) -> torch.Tensor:
    """Match the gap specialist's finite-ray mean and all-missing fallback."""
    rays = env.scene[sensor_name].data.hit_pos_w[..., 2]
    valid = torch.isfinite(rays)
    ground = torch.where(valid, rays, 0.0).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
    height = env.scene["robot"].data.root_link_pos_w[:, 2]
    penalty = (height - target_height - ground).square()
    return torch.where(valid.any(dim=1), penalty, 0.0)


def apply_specialist_reference(cfg: Any, task: str) -> Any:
    """Keep the B2 recipe and apply each other specialist's reward changes."""
    from mjlab.managers import RewardTermCfg

    cfg = apply_b2_reference(cfg)
    if task == "B2_stairs":
        return cfg
    generator = cfg.scene.terrain.terrain_generator
    if generator is not None:
        generator.size = (8.0, 8.0)
        if task == "B1_rough":
            terrain = generator.sub_terrains[task]
            terrain.noise_range = (0.01, 0.06)
            terrain.noise_step = 0.01
            cfg.scene.terrain.max_init_terrain_level = 5
    rw = cfg.rewards
    rw["track_lin_vel_xy_exp"].weight = 2.0 if task in ("B1_rough", "C2_gap") else 1.5
    rw["track_ang_vel_z_exp"].weight = 1.0 if task in ("B1_rough", "C2_gap") else 0.75
    rw["dof_torques_l2"].weight = -1.0e-5
    rw["action_rate_l2"].weight = -0.01
    rw["feet_air_time"].weight = 0.08 if task == "C2_gap" else 0.25
    rw["feet_air_time"].params["threshold"] = 0.20 if task == "C2_gap" else 0.5
    rw["undesired_contacts"].weight = -1.0
    rw["dof_pos_limits"].weight = -0.5 if task == "C2_gap" else 0.0
    rw["flat_orientation_l2"].weight = -2.5 if task.startswith("A") else -5.0
    if task.startswith("A"):
        del rw["undesired_contacts"]
    if task == "A1_forward":
        del rw["base_height"]
        rw["stand_still"] = RewardTermCfg(func=stand_joint_deviation, weight=-0.15)
        rw["stand_base_height"] = RewardTermCfg(
            func=stand_base_height, weight=-2.0, params={"target_height": 0.38}
        )
    else:
        rw["base_height"].weight = -4.0 if task == "A2_omni" else -10.0
        rw["base_height"].params["target_height"] = {
            "A2_omni": 0.40,
            "B1_rough": 0.38,
            "C2_gap": 0.37,
        }[task]
        rw["base_height"].func = (
            safe_base_height if task == "C2_gap" else rewards.base_height_l2
        )
    return cfg
