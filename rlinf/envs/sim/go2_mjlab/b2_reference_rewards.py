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

"""B2 reward kernels matching the pinned benchmark and Isaac Lab 2.3.2."""

from typing import Any

import torch


def base_lin_vel(env: Any) -> torch.Tensor:
    """Return body-frame COM velocity, as Isaac Lab root_lin_vel_b does."""
    return env.scene["robot"].data.root_com_lin_vel_b


def base_ang_vel(env: Any) -> torch.Tensor:
    """Return body-frame base angular velocity."""
    return env.scene["robot"].data.root_com_ang_vel_b


def track_lin_vel_xy_exp(env: Any, std: float, command_name: str) -> torch.Tensor:
    """Track XY velocity without including vertical velocity in the exponent."""
    error = (
        env.command_manager.get_command(command_name)[:, :2] - base_lin_vel(env)[:, :2]
    )
    return torch.exp(-error.square().sum(dim=1) / std**2)


def track_ang_vel_z_exp(env: Any, std: float, command_name: str) -> torch.Tensor:
    """Track yaw rate without including roll/pitch rates in the exponent."""
    error = (
        env.command_manager.get_command(command_name)[:, 2] - base_ang_vel(env)[:, 2]
    )
    return torch.exp(-error.square() / std**2)


def lin_vel_z_l2(env: Any) -> torch.Tensor:
    """Penalize squared vertical COM velocity."""
    return base_lin_vel(env)[:, 2].square()


def ang_vel_xy_l2(env: Any) -> torch.Tensor:
    """Penalize squared roll/pitch rates in the body frame."""
    return base_ang_vel(env)[:, :2].square().sum(dim=1)


def base_height_l2(env: Any, target_height: float, sensor_name: str) -> torch.Tensor:
    """Penalize base height relative to the mean terrain ray-hit height."""
    ground = env.scene[sensor_name].data.hit_pos_w[..., 2].mean(dim=1)
    adjusted_target = target_height + ground
    return (env.scene["robot"].data.root_link_pos_w[:, 2] - adjusted_target).square()


def feet_air_time(
    env: Any, command_name: str, sensor_name: str, threshold: float
) -> torch.Tensor:
    """Reward completed flight duration only on a foot's first contact."""
    sensor = env.scene[sensor_name]
    first = sensor.compute_first_contact(env.step_dt)
    reward = ((sensor.data.last_air_time - threshold) * first).sum(dim=1)
    moving = env.command_manager.get_command(command_name)[:, :2].norm(dim=1) > 0.1
    return reward * moving


def undesired_contacts(env: Any, sensor_name: str, threshold: float) -> torch.Tensor:
    """Count thigh/calf bodies exceeding the net-force history threshold."""
    history = env.scene[sensor_name].data.force_history
    # Each leg has one thigh geom and two calf geoms; foot geoms are excluded.
    per_leg = history.reshape(history.shape[0], 4, 3, history.shape[2], 3)
    bodies = torch.stack((per_leg[:, :, 0], per_leg[:, :, 1] + per_leg[:, :, 2]), dim=2)
    return (bodies.norm(dim=-1).amax(dim=-1) > threshold).sum(dim=(1, 2))
